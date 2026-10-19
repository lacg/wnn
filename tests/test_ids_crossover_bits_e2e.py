"""ga_crossover_bits end-to-end through the REAL IDSEvaluator + Rust search_offspring.

Skips on a worker wheel older than ABI 21. Parents carry ONE width each (30 or
34) so the three rules are distinguishable: inherit mixes the two widths inside
a cluster, blend_neuron also reaches in-between widths, blend_cluster keeps one
width per cluster.
"""

import numpy as np
import pytest

ram_accelerator = pytest.importorskip("ram_accelerator")
if getattr(ram_accelerator, "ABI_VERSION", 0) < 21:
	pytest.skip("worker wheel predates ga_crossover_bits (ABI < 21)", allow_module_level=True)

from test_ids_ga_knobs_e2e import _dataset  # noqa: E402
from wnn.ram.architecture.ids_evaluator import IDSEvaluator  # noqa: E402
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome  # noqa: E402

WIDTHS = (30, 34)


@pytest.fixture(autouse=True)
def _order_independent_training(monkeypatch):
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")


@pytest.fixture(scope="module")
def evaluator():
	# Two clusters (one per class), the 2m shape.
	return IDSEvaluator(_dataset(), classification="binary", num_parts=5, k_folds=5, seed=7, single_cluster=False)


def _population(ev, n=8, neurons=10, seed=5):
	rng = np.random.default_rng(seed)
	tib = ev.total_input_bits
	pop = []
	for i in range(n):
		b = WIDTHS[i % 2]
		conns = rng.integers(0, tib, size=2 * neurons * b).tolist()
		g = ClusterGenome(bits_per_neuron=[b] * (2 * neurons), neurons_per_cluster=[neurons, neurons], connections=conns)
		pop.append((g, 1.0 + i / 10))
	return pop


def _offspring(ev, rule, pool_shuffle_ratio=0.0):
	res = ev.search_offspring(
		population=_population(ev), target_count=12, max_attempts=40, accuracy_threshold=0.0,
		min_bits=4, max_bits=34, min_neurons=5, max_neurons=500,
		bits_mutation_rate=0.0, neurons_mutation_rate=0.0,
		crossover_rate=1.0, tournament_size=3, train_subset_idx=1, seed=13,
		phase_type=0, pool_shuffle_ratio=pool_shuffle_ratio, assortative_mating_ratio=0.0,
		crossover_bits=rule,
	)
	assert res.genomes
	for g in res.genomes:
		assert len(g.connections) == sum(g.bits_per_neuron)
	return res.genomes


def _clusters(g):
	out, start = [], 0
	for n in g.neurons_per_cluster:
		out.append(g.bits_per_neuron[start:start + n])
		start += n
	return out


def test_inherit_mixes_only_parent_widths(evaluator):
	kids = _offspring(evaluator, "inherit")
	assert any(len(set(c)) > 1 for g in kids for c in _clusters(g))
	assert all(set(g.bits_per_neuron) <= set(WIDTHS) for g in kids)


@pytest.mark.parametrize("ps", [0.0, 1.0])
def test_blend_cluster_one_width_per_cluster(evaluator, ps):
	for g in _offspring(evaluator, "blend_cluster", ps):
		for c in _clusters(g):
			assert len(set(c)) == 1 and 30 <= c[0] <= 34


def test_blend_neuron_reaches_in_between_widths(evaluator):
	kids = _offspring(evaluator, "blend_neuron")
	assert all(30 <= b <= 34 for g in kids for b in g.bits_per_neuron)
	assert any(b not in WIDTHS for g in kids for b in g.bits_per_neuron)


def test_unknown_rule_is_rejected(evaluator):
	with pytest.raises(ValueError):
		_offspring(evaluator, "blend")
