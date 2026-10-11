"""IDS-26 GA knobs end-to-end through the REAL IDSEvaluator + Rust search_offspring.

Skips on a worker wheel older than ABI 15 (the one that adds dedupe_neurons /
phase_type 5). Tiny synthetic binary dataset; checks behaviour, not accuracy.
"""

import numpy as np
import pandas as pd
import pytest

ram_accelerator = pytest.importorskip("ram_accelerator")
if getattr(ram_accelerator, "ABI_VERSION", 0) < 15:
	pytest.skip("worker wheel predates the IDS-26 GA knobs (ABI < 15)", allow_module_level=True)

from wnn.ram.architecture.ids_evaluator import IDSEvaluator  # noqa: E402
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome  # noqa: E402


def _dataset(n_train=200, n_test=40, n_bits=8, seed=42):
	from wnn.ids.dataset import IDSDataset
	from wnn.ids.encoded_array import InMemoryEncoded
	from wnn.representations.thermometer import ThermometerEncoder

	rng = np.random.default_rng(seed)
	frame = lambda n: pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(size=n), "c": rng.normal(size=n)})
	df_train, df_test = frame(n_train), frame(n_test)
	enc = ThermometerEncoder(n_bits=n_bits)
	enc.fit(df_train)
	x_train, total_bits = enc.transform(df_train)
	x_test, _ = enc.transform(df_test)
	y_train = (df_train["a"].to_numpy() > 0).astype(np.int64)
	y_test = (df_test["a"].to_numpy() > 0).astype(np.int64)
	return IDSDataset(
		X_train=InMemoryEncoded(x_train, total_bits=total_bits), y_train_binary=y_train, y_train_multi=y_train.copy(),
		X_test=InMemoryEncoded(x_test, total_bits=total_bits), y_test_binary=y_test, y_test_multi=y_test.copy(),
		encoder=enc, category_names=["Normal", "Attack"], feature_names=["a", "b", "c"],
	)


@pytest.fixture(autouse=True)
def _order_independent_training(monkeypatch):
	"""The legacy (non-OI) trainer writes in rayon order, so the SAME genome can
	score a slightly different CE run to run; OI training is deterministic."""
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")


@pytest.fixture(scope="module")
def evaluator():
	return IDSEvaluator(_dataset(), classification="binary", num_parts=5, k_folds=5, seed=7, single_cluster=True)


def _duplicated_population(ev, n=6, neurons=12, bits=6, seed=3):
	"""Single-cluster genomes where every 3rd neuron repeats the previous one."""
	rng = np.random.default_rng(seed)
	tib = ev.total_input_bits
	pop = []
	for i in range(n):
		conns = []
		for k in range(neurons):
			neuron = conns[-bits:] if k % 3 == 2 else rng.integers(0, tib, size=bits).tolist()
			conns.extend(neuron)
		g = ClusterGenome(bits_per_neuron=[bits] * neurons, neurons_per_cluster=[neurons], connections=conns)
		pop.append((g, 1.0 + i / 10))
	return pop


def _search(ev, pop, phase_type=0, dedupe=None, pool_shuffle_ratio=1.0, seed=11):
	common = dict(
		population=pop, target_count=8, max_attempts=40, accuracy_threshold=0.0,
		min_bits=4, max_bits=34, min_neurons=5, max_neurons=500,
		bits_mutation_rate=0.1 if phase_type == 5 else 0.0, neurons_mutation_rate=0.1,
		crossover_rate=1.0, tournament_size=3, train_subset_idx=1, seed=seed,
		phase_type=phase_type, pool_shuffle_ratio=pool_shuffle_ratio, assortative_mating_ratio=0.85,
	)
	if dedupe is None:
		return ev.search_offspring(**common)
	return ev.search_offspring(**common, dedupe_neurons=dedupe)


def _neuron_keys(g):
	off = np.cumsum([0] + list(g.bits_per_neuron))
	return [(g.bits_per_neuron[i], tuple(g.connections[off[i]:off[i + 1]])) for i in range(len(g.bits_per_neuron))]


def _signature(result):
	return [(g.bits_per_neuron, g.neurons_per_cluster, g.connections, g.metrics.ce) for g in result.genomes]


def test_default_equals_explicit_off(evaluator):
	"""Omitting dedupe_neurons and passing False breed the same offspring and score them the same."""
	pop = _duplicated_population(evaluator)
	assert _signature(_search(evaluator, pop)) == _signature(_search(evaluator, pop, dedupe=False))


def test_dedupe_children_have_distinct_neurons(evaluator):
	pop = _duplicated_population(evaluator)
	for ratio in (0.0, 1.0):
		res = _search(evaluator, pop, dedupe=True, pool_shuffle_ratio=ratio)
		assert res.genomes
		for g in res.genomes:
			keys = _neuron_keys(g)
			assert len(set(keys)) == len(keys) == sum(g.neurons_per_cluster)


def test_dedupe_off_inherits_duplicates(evaluator):
	pop = _duplicated_population(evaluator)
	res = _search(evaluator, pop, dedupe=False)
	assert any(len(set(_neuron_keys(g))) < len(_neuron_keys(g)) for g in res.genomes)


def test_joint_phase_runs_in_rust_and_respects_bands(evaluator):
	pop = _duplicated_population(evaluator)
	res = _search(evaluator, pop, phase_type=5)
	assert res.genomes
	for g in res.genomes:
		assert all(4 <= b <= 34 for b in g.bits_per_neuron)
		assert 5 <= g.neurons_per_cluster[0] <= 500
		assert len(g.connections) == sum(g.bits_per_neuron)


def test_immigrant_scoring_matches_rust_offspring_scoring(evaluator):
	"""evaluate_offspring_batch re-scores a Rust offspring to the same metrics
	Rust reported for it — immigrants and bred offspring share one evaluator."""
	res = _search(evaluator, _duplicated_population(evaluator))
	g = res.genomes[0]
	rust_metrics = (g.metrics.ce, g.metrics.acc, g.metrics.f1, g.metrics.fpr)
	clone = ClusterGenome(bits_per_neuron=list(g.bits_per_neuron), neurons_per_cluster=list(g.neurons_per_cluster),
	                      connections=list(g.connections))
	m = evaluator.evaluate_offspring_batch([clone], train_subset_idx=1)[0]
	assert (m.ce, m.acc, m.f1, m.fpr) == pytest.approx(rust_metrics)
