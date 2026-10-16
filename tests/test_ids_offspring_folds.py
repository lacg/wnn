"""ids_offspring_folds (worker ABI 20, 16/10/2026): the GA-fitness fold rule.

Since 28/05 (6ab34164) Rust-bred offspring, IDS-26 immigrants and TS neighbours
were scored on ONE fold fixed for the whole phase, while grid / init / re-eval
genomes carried the K-fold CV mean from evaluate_batch — tournament, mu+lambda
truncation, patience and the final pick ranked a mixed pool. The paper's
protocol (sp2027 main.tex) is the CV mean for EVERY genome.

  "all"   (default): every evaluated genome's fitness == evaluate_batch's
          K-fold mean for the same genome (k_folds = kfold_per_gen = 5).
  "phase" (legacy): bred / immigrant genomes on the single phase-fixed fold,
          exactly evaluate_genomes_kfold_hybrid(train_subset_idx).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "wnn"))
sys.path.insert(0, str(ROOT / "tests"))


def _abi() -> int:
	try:
		import ram_accelerator
		return int(getattr(ram_accelerator, "ABI_VERSION", 0))
	except ImportError:
		return 0


pytestmark = pytest.mark.skipif(_abi() < 20, reason="needs ram_accelerator ABI >= 20")
PHASE_FOLD = 2


def _evaluator(offspring_folds: str, kfold_per_gen=None, classification="binary", single_cluster=True):
	from test_ids_evaluator_streaming import _make_multiclass_dataset
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	inmem, _ = _make_multiclass_dataset()
	return IDSEvaluator(
		inmem, classification=classification, single_cluster=single_cluster, num_parts=5, k_folds=5,
		kfold_per_gen=kfold_per_gen, seed=42, empty_value=0.0, memory_mode=2, neuron_sample_rate=1.0,
		offspring_folds=offspring_folds,
	)


def _genome(seed: int, clusters: int = 1):
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	import random
	rng = random.Random(seed)
	n = 4
	return ClusterGenome(
		bits_per_neuron=[8] * (n * clusters), neurons_per_cluster=[n] * clusters,
		connections=[rng.randrange(24) for _ in range(n * clusters * 8)],
	)


def _clone(g):
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	return ClusterGenome(bits_per_neuron=list(g.bits_per_neuron), neurons_per_cluster=list(g.neurons_per_cluster),
		connections=list(g.connections))


def _offspring(ev, clusters=1):
	population = [(_genome(s, clusters), 0.5) for s in range(6)]
	return ev.search_offspring(
		population=population, target_count=6, max_attempts=30, accuracy_threshold=0.0,
		min_bits=4, max_bits=10, min_neurons=2, max_neurons=6, bits_mutation_rate=0.2,
		neurons_mutation_rate=0.2, train_subset_idx=PHASE_FOLD, seed=7, phase_type=0,
	).genomes


def _same(a, b):
	"""(acc, f1, fpr) exact; CE within the existing GPU CE tolerance."""
	assert (a.acc, a.f1, a.fpr) == (b.acc, b.f1, b.fpr)
	assert abs(a.ce - b.ce) <= 1e-6 * max(1.0, abs(b.ce))


def _legacy_fold(ev, genomes):
	bits, neurons, conns = ev._flatten_genomes(genomes)
	return ev._cache.evaluate_genomes_kfold_hybrid(bits, neurons, conns, len(genomes), PHASE_FOLD,
		ev._empty_value, ev._neuron_sample_rate, 0)


@pytest.fixture(autouse=True)
def _oi(monkeypatch):
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")


@pytest.mark.parametrize("classification, single_cluster, clusters", [("binary", True, 1), ("binary", False, 2)])
def test_all_offspring_fitness_is_the_evaluate_batch_cv_mean(classification, single_cluster, clusters):
	ev = _evaluator("all", classification=classification, single_cluster=single_cluster)
	assert ev._kfold_per_gen == 5 and ev.ga_fitness_folds() == [0, 1, 2, 3, 4]
	kids = _offspring(ev, clusters)
	assert kids
	for kid in kids:
		_same(kid.metrics, ev.evaluate_batch([_clone(kid)])[0])


def test_all_immigrants_and_ts_neighbours_are_the_cv_mean():
	ev = _evaluator("all")
	immigrants = [_genome(s) for s in (11, 12, 13)]
	got = ev.evaluate_offspring_batch(immigrants, PHASE_FOLD)
	for g, m in zip(immigrants, got):
		_same(m, ev.evaluate_batch([_clone(g)])[0])
	neighbours = ev.search_neighbors(_genome(3), target_count=3, max_attempts=15, accuracy_threshold=0.0,
		min_bits=4, max_bits=10, min_neurons=2, max_neurons=6, train_subset_idx=PHASE_FOLD, seed=5)
	assert neighbours
	for g in neighbours:
		_same(g.metrics, ev.evaluate_batch([_clone(g)])[0])


def test_phase_is_the_legacy_single_fold():
	ev = _evaluator("phase", kfold_per_gen=5)
	assert ev.ga_fitness_folds() is None
	kids = _offspring(ev)
	for kid, (ce, acc, f1, fpr, _t, _ms) in zip(kids, _legacy_fold(ev, [_clone(k) for k in kids])):
		assert (kid.metrics.ce, kid.metrics.acc, kid.metrics.f1, kid.metrics.fpr) == (ce, acc, f1, fpr)
	immigrants = [_genome(s) for s in (11, 12)]
	got = ev.evaluate_offspring_batch(immigrants, PHASE_FOLD)
	for m, (ce, acc, f1, fpr, _t, _ms) in zip(got, _legacy_fold(ev, [_clone(g) for g in immigrants])):
		assert (m.ce, m.acc, m.f1, m.fpr) == (ce, acc, f1, fpr)


def test_strict_needs_every_fold_per_generation():
	with pytest.raises(ValueError, match="kfold_per_gen=1"):
		_evaluator("all", kfold_per_gen=1)
	assert _evaluator("phase")._kfold_per_gen == 1      # legacy IDSEvaluator default
	assert _evaluator("all")._kfold_per_gen == 5        # strict default: every fold
	with pytest.raises(ValueError, match="expected 'all' or 'phase'"):
		_evaluator("some")


def test_fitness_fold_log_says_what_happens():
	assert "5-fold CV mean over all 5 folds" in _evaluator("all").describe_fitness_folds(PHASE_FOLD)
	assert "offspring/immigrants/neighbours: same mean" in _evaluator("all").describe_fitness_folds(PHASE_FOLD)
	assert f"held-out fold {PHASE_FOLD} only" in _evaluator("phase", kfold_per_gen=5).describe_fitness_folds(PHASE_FOLD)
