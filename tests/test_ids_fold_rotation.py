"""ids_fold_rotation (IDS-76, worker ABI 30): re-draw the K-fold partition every GA generation.

- the param resolves strictly (bool only) and needs the strict K-fold CV search;
- off: begin_generation_folds never reaches Rust, the GA never re-scores;
- on: generation 0 is the fixed partition, later generations re-draw it, and
  generation 0 again restores the fixed-fold fitness bit for bit;
- on, through ArchitectureGAStrategy.optimize: every generation >= 1 re-draws,
  the final pool is scored on ONE partition (the last), the stage ends back on
  the fixed folds, and a pool-relative fitness is refused.

The flag-off byte-identity vs ABI 29 is scripts/ids76_flag_off_pin.py (two wheels).
"""
import pytest

from wnn.ram.experiments.ids_param_resolution import resolve_fold_rotation
from wnn.ram.experiments.params import KNOWN_PARAMS


def _abi() -> int:
	try:
		import ram_accelerator
		return int(getattr(ram_accelerator, "ABI_VERSION", 0))
	except ImportError:
		return 0


needs_abi30 = pytest.mark.skipif(_abi() < 30, reason="needs ram_accelerator ABI >= 30")
SEED = 20421
GENS = 5


# ---------------------------------------------------------------- param


def test_param_is_registered():
	assert "ids_fold_rotation" in KNOWN_PARAMS


@pytest.mark.parametrize("params, want", [({}, False), ({"ids_fold_rotation": None}, False),
	({"ids_fold_rotation": False}, False), ({"ids_fold_rotation": True}, True)])
def test_resolves(params, want):
	assert resolve_fold_rotation(params) is want


@pytest.mark.parametrize("value", ["true", "false", 1, 0])
def test_only_a_real_bool(value):
	with pytest.raises(ValueError, match="expected true or false"):
		resolve_fold_rotation({"ids_fold_rotation": value})


@pytest.mark.parametrize("params", [{"ids_offspring_folds": "phase"}, {"ids_k_folds": 1}])
def test_needs_the_strict_kfold_search(params):
	with pytest.raises(ValueError, match="ids_k_folds > 1 and ids_offspring_folds='all'"):
		resolve_fold_rotation({"ids_fold_rotation": True, **params})
	assert resolve_fold_rotation({"ids_fold_rotation": False, **params}) is False


# ---------------------------------------------------------------- evaluator


@pytest.fixture(autouse=True)
def _oi(monkeypatch):
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")


@pytest.fixture(scope="module")
def ds():
	from tests.test_ids_ga_knobs_e2e import _dataset
	return _dataset(n_train=600, n_test=100)


def _evaluator(ds, fold_rotation, offspring_folds="all"):
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	return IDSEvaluator(ds, classification="binary", num_parts=5, k_folds=5, seed=SEED, single_cluster=False,
		memory_mode=3, write_rule="vote", neuron_sample_rate=1.0, offspring_folds=offspring_folds,
		fold_rotation=fold_rotation)


def _genomes(ev, n=4, seed=1):
	import numpy as np
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	rng = np.random.default_rng(seed)
	return [ClusterGenome(bits_per_neuron=[6] * 16, neurons_per_cluster=[8, 8],
		connections=rng.integers(0, ev.total_input_bits, size=96).tolist()) for _ in range(n)]


def _rows(metrics):
	return [(m.ce, m.acc, m.f1, m.fpr) for m in metrics]


def _close(a, b):
	"""(acc, f1, fpr) exact; CE within the GPU CE tolerance the offspring-folds tests use."""
	assert (a.acc, a.f1, a.fpr) == (b.acc, b.f1, b.fpr)
	assert abs(a.ce - b.ce) <= 1e-6 * max(1.0, abs(b.ce))


@needs_abi30
def test_off_never_reaches_rust(ds):
	ev = _evaluator(ds, False)
	cache, ev._cache = ev._cache, object()   # no set_fold_generation: any call would raise
	try:
		assert ev.fold_rotation is False
		assert ev.begin_generation_folds(3) is False
	finally:
		ev._cache = cache


@needs_abi30
def test_on_redraws_and_generation_zero_restores_bit_for_bit(ds):
	off = _evaluator(ds, False)
	fixed = _rows(off.evaluate_batch(_genomes(off)))
	ev = _evaluator(ds, True)
	gs = _genomes(ev)
	assert ev.begin_generation_folds(0) is False, "a fresh cache already holds generation 0"
	assert _rows(ev.evaluate_batch(gs)) == fixed, "generation 0 = the fixed folds"
	assert ev.begin_generation_folds(1) is True
	rotated = _rows(ev.evaluate_batch(gs))
	assert rotated != fixed, "a re-drawn partition changes the CV scores"
	assert ev.begin_generation_folds(1) is False, "the held generation again is a no-op"
	assert ev.begin_generation_folds(0) is True
	assert _rows(ev.evaluate_batch(gs)) == fixed, "back on the fixed folds, bit for bit"
	ev2 = _evaluator(ds, True)
	ev2.begin_generation_folds(1)
	assert _rows(ev2.evaluate_batch(gs)) == rotated, "generation 1 is a function of (seed, 1)"


@needs_abi30
def test_on_refuses_the_legacy_single_fold_search(ds):
	with pytest.raises(ValueError, match="offspring_folds='all'"):
		_evaluator(ds, True, offspring_folds="phase")


# ---------------------------------------------------------------- GA


def _strategy(ev, aggregation="desirability"):
	from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig
	from wnn.ram.strategies.connectivity.architecture_ga import ArchitectureGAStrategy
	from wnn.ram.strategies.connectivity.framework import GAConfig
	arch = ArchitectureConfig(num_clusters=2, min_bits=6, max_bits=6, min_neurons=4, max_neurons=16, default_bits=6,
		default_neurons=8, total_input_bits=ev.total_input_bits, optimize_neurons=True, optimize_bits=False)
	cfg = GAConfig(population_size=6, generations=GENS, patience=50, check_interval=1, fitness_percentile=0.75,
		fitness_aggregation=aggregation, fitness_weight_ce=0.1, fitness_weight_f1=0.35, fitness_weight_fpr=0.35,
		fitness_weight_acc=0.2, magnitude_aware_patience=True)
	return ArchitectureGAStrategy(arch_config=arch, ga_config=cfg, seed=SEED, cached_evaluator=ev)


def _recorded(ev, monkeypatch):
	calls = []
	original = ev.begin_generation_folds

	def recording(generation):
		changed = original(generation)
		calls.append((generation, changed))
		return changed

	monkeypatch.setattr(ev, "begin_generation_folds", recording)
	return calls


@needs_abi30
def test_ga_rotates_every_generation_and_keeps_one_partition_per_pool(ds, monkeypatch):
	ev = _evaluator(ds, True)
	calls = _recorded(ev, monkeypatch)
	res = _strategy(ev).optimize(initial_population=_genomes(ev, 6, 4), train_subset_idx=1)
	gens = [g for g, _ in calls[:-1]]
	assert gens == list(range(len(gens))) and len(gens) == res.iterations_run, calls
	assert calls[0] == (0, False), "generation 0 runs on the fixed folds (no re-score)"
	assert all(changed for _, changed in calls[1:]), "every later generation re-draws (and the stage end restores)"
	assert calls[-1][0] == 0, "the stage ends back on the fixed folds"
	# The final pool was scored on ONE partition: the last generation's.
	last = gens[-1]
	ev.begin_generation_folds(last)
	again = ev.evaluate_offspring_batch([g.clone() for g in res.final_population], 1)
	for m, want in zip(res.population_metrics, again):
		_close(m, want)


@needs_abi30
def test_ga_refuses_a_pool_relative_fitness(ds):
	ev = _evaluator(ds, True)
	with pytest.raises(ValueError, match="fitness_aggregation='desirability'"):
		_strategy(ev, aggregation="harmonic").optimize(initial_population=_genomes(ev, 6, 4), train_subset_idx=1)


@needs_abi30
def test_ga_off_never_rotates(ds, monkeypatch):
	ev = _evaluator(ds, False)
	calls = _recorded(ev, monkeypatch)
	_strategy(ev).optimize(initial_population=_genomes(ev, 6, 4), train_subset_idx=1)
	assert calls == [], "the off path never asks for a partition"
