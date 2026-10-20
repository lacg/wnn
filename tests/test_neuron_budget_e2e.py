"""max_neurons_total end-to-end through the REAL IDSEvaluator + Rust offspring /
neighbour search: every returned genome sums to <= the budget (2m shape)."""

import numpy as np
import pytest

ram_accelerator = pytest.importorskip("ram_accelerator")
if getattr(ram_accelerator, "ABI_VERSION", 0) < 21:
	pytest.skip("worker wheel predates max_neurons_total (ABI < 21)", allow_module_level=True)

from test_ids_ga_knobs_e2e import _dataset  # noqa: E402
from wnn.ram.architecture.ids_evaluator import IDSEvaluator  # noqa: E402
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome  # noqa: E402

TOTAL = 30


@pytest.fixture(autouse=True)
def _order_independent_training(monkeypatch):
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")


@pytest.fixture(scope="module")
def evaluator():
	return IDSEvaluator(_dataset(), classification="binary", num_parts=5, k_folds=5, seed=7, single_cluster=False)


def _genome(ev, rng, n0, n1, b=6):
	conns = rng.integers(0, ev.total_input_bits, size=(n0 + n1) * b).tolist()
	return ClusterGenome(bits_per_neuron=[b] * (n0 + n1), neurons_per_cluster=[n0, n1], connections=conns)


def _check(genomes):
	assert genomes
	for g in genomes:
		assert sum(g.neurons_per_cluster) <= TOTAL, g.neurons_per_cluster
		assert len(g.connections) == sum(g.bits_per_neuron)


@pytest.mark.parametrize("phase_type", [0, 5])
def test_offspring_fit_the_budget(evaluator, phase_type):
	rng = np.random.default_rng(3)
	# Parents straddle the budget split differently, so crossover alone can exceed it.
	pop = [(_genome(evaluator, rng, 25 - i, 5 + i), 1.0 + i / 10) for i in range(6)]
	res = evaluator.search_offspring(
		population=pop, target_count=10, max_attempts=40, accuracy_threshold=0.0,
		min_bits=4, max_bits=12, min_neurons=2, max_neurons=40,
		bits_mutation_rate=0.1 if phase_type == 5 else 0.0, neurons_mutation_rate=0.5,
		crossover_rate=1.0, tournament_size=3, train_subset_idx=1, seed=11,
		phase_type=phase_type, pool_shuffle_ratio=1.0, assortative_mating_ratio=0.0,
		max_neurons_total=TOTAL,
	)
	_check(res.genomes)


def test_neighbours_fit_the_budget(evaluator):
	rng = np.random.default_rng(4)
	out = evaluator.search_neighbors(
		genome=_genome(evaluator, rng, 15, 15), target_count=8, max_attempts=30, accuracy_threshold=0.0,
		min_bits=4, max_bits=12, min_neurons=2, max_neurons=40,
		bits_mutation_rate=0.0, neurons_mutation_rate=1.0, train_subset_idx=1, seed=5,
		max_neurons_total=TOTAL,
	)
	_check(out)
