"""2^4 factorial of the IDS-26 GA knobs (+ DEDUP-nofill) end-to-end.

NOXO   assortative_mating_ratio=0, pool_shuffle_ratio=0 (else 0.85 / 0.8)
DEDUP  ga_dedupe_neurons            (+ the ga_dedupe_fill=False variants)
IMM10  immigrant_fraction=0.1
JOINT  ga_joint_mutation

Every combination runs ArchitectureGAStrategy._generate_offspring against the
REAL IDSEvaluator + Rust search_offspring on a tiny single-cluster dataset, in
the GA-Neurons stage shape (optimize_neurons only, bits band 4-34, neurons
5-500). Parents carry duplicate neurons (every 3rd repeats the previous one), all
at 6 bits. crossover_rate=1.0 so every bred child goes through crossover — a
CLONE copies its parent verbatim, duplicates included, by design.
Skips on a worker wheel older than ABI 16.
"""

import itertools

import numpy as np
import pytest

ram_accelerator = pytest.importorskip("ram_accelerator")
if getattr(ram_accelerator, "ABI_VERSION", 0) < 16:
	pytest.skip("worker wheel predates ga_dedupe_fill (ABI < 16)", allow_module_level=True)

from tests.test_ids_ga_knobs_e2e import _dataset, _duplicated_population, _neuron_keys  # noqa: E402
from wnn.ram.architecture.ids_evaluator import IDSEvaluator  # noqa: E402
from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig  # noqa: E402
from wnn.ram.strategies.connectivity.architecture_ga import ArchitectureGAStrategy  # noqa: E402
from wnn.ram.strategies.connectivity.framework import GAConfig  # noqa: E402
from wnn.ram.strategies.optimization_dimension import PhaseType  # noqa: E402

N_NEEDED = 10
PARENT_BITS = 6
PARENT_NEURONS = 12

# (noxo, dedupe, fill, imm10, joint): the 16 factorial cells + the 8 nofill variants.
COMBOS = [(x, d, True, i, j) for x, d, i, j in itertools.product([False, True], repeat=4)]
COMBOS += [(x, True, False, i, j) for x, i, j in itertools.product([False, True], repeat=3)]


def _combo_id(combo):
	noxo, dedupe, fill, imm, joint = combo
	parts = [n for n, on in (("NOXO", noxo), ("DEDUP" if fill else "DEDUP-nofill", dedupe),
	                         ("IMM10", imm), ("JOINT", joint)) if on]
	return "+".join(parts) or "BASE"


@pytest.fixture(autouse=True)
def _order_independent_training(monkeypatch):
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")


@pytest.fixture(scope="module")
def evaluator():
	return IDSEvaluator(_dataset(), classification="binary", num_parts=5, k_folds=5, seed=7, single_cluster=True)


def _strategy(evaluator, combo):
	noxo, dedupe, fill, imm, joint = combo
	arch = ArchitectureConfig(
		num_clusters=1, min_bits=4, max_bits=34, min_neurons=5, max_neurons=500,
		default_bits=PARENT_BITS, default_neurons=PARENT_NEURONS, total_input_bits=evaluator.total_input_bits,
		optimize_neurons=True, optimize_bits=False,
		assortative_mating_ratio=0.0 if noxo else 0.85, pool_shuffle_ratio=0.0 if noxo else 0.8,
		dedupe_neurons=dedupe, dedupe_fill=fill, joint_mutation=joint,
	)
	cfg = GAConfig(population_size=6, generations=3, crossover_rate=1.0, mutation_rate=0.1,
	               immigrant_fraction=0.1 if imm else 0.0)
	s = ArchitectureGAStrategy(arch_config=arch, ga_config=cfg, seed=5, cached_evaluator=evaluator)
	s._ensure_rng()
	s._phase_train_idx = 1
	s._offspring_seed = 2026
	return s


def _scored_population(evaluator):
	genomes = [g for g, _ in _duplicated_population(evaluator, neurons=PARENT_NEURONS, bits=PARENT_BITS)]
	return list(zip(genomes, evaluator.evaluate_offspring_batch(genomes, train_subset_idx=1)))


def _run(evaluator, combo, monkeypatch):
	"""(offspring, immigrants) — immigrants recorded at evaluate_offspring_batch."""
	immigrants = []
	original = evaluator.evaluate_offspring_batch

	def recording(genomes, train_subset_idx):
		immigrants.extend(genomes)
		return original(genomes, train_subset_idx)

	population = _scored_population(evaluator)
	s = _strategy(evaluator, combo)
	monkeypatch.setattr(evaluator, "evaluate_offspring_batch", recording)
	offspring = s._generate_offspring(population, n_needed=N_NEEDED, threshold=0.0, generation=0)
	return s, offspring, immigrants


def _assert_valid(g):
	assert len(g.connections) == sum(g.bits_per_neuron), "connections length = sum(bits)"
	assert len(g.bits_per_neuron) == sum(g.neurons_per_cluster)
	assert all(4 <= b <= 34 for b in g.bits_per_neuron), "bits band"
	assert all(5 <= n <= 500 for n in g.neurons_per_cluster), "neurons band"
	assert g.metrics is not None and np.isfinite(g.metrics.ce)


@pytest.mark.parametrize("combo", COMBOS, ids=[_combo_id(c) for c in COMBOS])
def test_combination(evaluator, combo, monkeypatch):
	noxo, dedupe, fill, imm, joint = combo
	s, offspring, immigrants = _run(evaluator, combo, monkeypatch)
	assert s._phase_type == (PhaseType.JOINT if joint else PhaseType.NEURONS)
	assert len(offspring) == N_NEEDED
	for g, _ in offspring:
		_assert_valid(g)
	assert len(immigrants) == (round(0.1 * N_NEEDED) if imm else 0)
	assert all(any(i is g for g, _ in offspring) for i in immigrants), "immigrants reach the offspring pool"
	bred = [g for g, _ in offspring if not any(g is i for i in immigrants)]
	_assert_dedupe(bred + immigrants, dedupe)
	_assert_bits_axis(bred, immigrants, joint)
	if dedupe and not fill and noxo:
		# Positional crossover deduped without refill: children lose their repeats.
		assert any(g.neurons_per_cluster[0] < PARENT_NEURONS - 1 for g in bred), "nofill child shrank"


def _assert_dedupe(genomes, dedupe):
	distinct = [len(set(_neuron_keys(g))) == len(_neuron_keys(g)) for g in genomes]
	if dedupe:
		assert all(distinct), "DEDUP: every child's neurons are distinct"
	else:
		assert not all(distinct), "without DEDUP the duplicate-heavy parents pass duplicates on"


def _assert_bits_axis(bred, immigrants, joint):
	"""NEURONS keeps every bit width at the parents' 6 (immigrants inherit the
	mode); JOINT mutates bits and its immigrants sample the band."""
	if joint:
		assert any(b != PARENT_BITS for g in bred for b in g.bits_per_neuron), "JOINT moves bits"
		for g in immigrants:
			assert len(set(g.bits_per_neuron)) > 1, "JOINT immigrant samples bits"
	else:
		assert all(b == PARENT_BITS for g in bred + immigrants for b in g.bits_per_neuron)
