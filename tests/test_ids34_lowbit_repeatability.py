"""IDS-34 (worker ABI 22, 01/11/2026): the same genome scored twice in ONE
process must give bit-identical metrics at every bit width.

Before ABI 22 the GPU marker trainer dropped writes whenever z-thread
duplicate-key slots filled a neuron's 2 x 2^bits slot region: at b = 8 (and
some b = 12) every call trained a different memory (IDS-32 §4: n=100 b=8 CE
1.0315 / 1.0264 / 1.0207 on three back-to-back calls). The Rust guard is
marker_repeatability_tests.rs; this is the same check through the production
Python route (IDSEvaluator.evaluate_batch -> K-fold CV mean -> marker trainer).
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


pytestmark = pytest.mark.skipif(_abi() < 22, reason="needs ram_accelerator ABI >= 22 (IDS-34 fix)")

N_TRAIN = 60_000
TOTAL_BITS = 64
REPEATS = 4


@pytest.fixture(scope="module")
def dataset():
	from test_ids_evaluator_streaming import _make_multiclass_dataset
	inmem, _ = _make_multiclass_dataset(n_train=N_TRAIN, n_test=2_000, n_val=1_000, total_bits=TOTAL_BITS)
	return inmem


@pytest.fixture(autouse=True)
def _oi(monkeypatch):
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")


def _evaluator(dataset, memory_mode: int, write_rule):
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	return IDSEvaluator(
		dataset, classification="binary", single_cluster=True, num_parts=5, k_folds=5, kfold_per_gen=5,
		seed=20419, empty_value=0.0, memory_mode=memory_mode, neuron_sample_rate=0.25, write_rule=write_rule,
	)


def _genome(neurons: int, bits: int, seed: int):
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	import random
	rng = random.Random(seed)
	return ClusterGenome(
		bits_per_neuron=[bits] * neurons, neurons_per_cluster=[neurons],
		connections=[rng.randrange(TOTAL_BITS) for _ in range(neurons * bits)],
	)


def _metrics(ev, genome):
	m = ev.evaluate_batch([genome])[0]
	return (m.ce, m.acc, m.f1, m.fpr)


@pytest.mark.parametrize("memory_mode, write_rule", [(2, "lattice"), (3, "lattice")], ids=["QUAD-OI", "BINARY-lattice"])
@pytest.mark.parametrize("neurons, bits", [(100, 8), (300, 8), (100, 12), (300, 12), (100, 16)])
def test_same_genome_scores_bit_identically(dataset, memory_mode, write_rule, neurons, bits):
	ev = _evaluator(dataset, memory_mode, write_rule)
	genome = _genome(neurons, bits, seed=neurons * 1000 + bits)
	first = _metrics(ev, genome)
	for rep in range(1, REPEATS):
		assert _metrics(ev, genome) == first, f"n={neurons} b={bits} mode={memory_mode}: rep {rep} differs"
