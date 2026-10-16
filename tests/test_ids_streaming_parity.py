"""IDSGenomeStreamer == in-memory IDSCacheWrapper through IDSEvaluator, for
every memory_mode x write rule x single_cluster (worker ABI 20, 16/10/2026).

Until ABI 20 the streaming path trained and scored QUAD_WEIGHTED whatever the
flow's memory_mode; 7040bcf1 then refused every other mode. The streamer now
takes the cache's settings, so on the same data both evaluators must report the
SAME numbers:
  - GA-search metrics (evaluate_batch_full): (ce, acc, f1, fpr, threshold);
  - validation: single-cluster evaluate_at_thresholds (eval/train/val scores +
    metrics at a fixed and the oracle threshold), K-cluster
    evaluate_multiclass_at_thresholds (every decode mode, incl. the binary
    two-cluster margin_train_f1 / margin_val_f1).
The pure-Rust twin (more layouts, sampling, coverage, balance) is
streamer_parity_tests.rs; this pins the Python plumbing (IDSEvaluator ->
IDSGenomeStreamerWrapper kwargs, train-calibrated threshold, rng_seed).

WNN_ORDER_INDEPENDENT_TRAIN=1 is set (production IDS): without it the 4-state
modes' legacy clamped nudge is order-dependent on both paths.
"""
import math
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

MODES = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2, "BINARY": 3, "QSR": 4, "PLN": 5}
VOTE_MODES = ("TERNARY", "BINARY", "PLN")
# (classification, single_cluster): single-cluster binary, binary two-cluster, multiclass K=4
LAYOUTS = [("binary", True), ("binary", False), ("multi", False)]
CASES = [
	(mode, rule, cls, sc, bits)
	for mode in MODES
	for rule in (("lattice", "vote") if mode in VOTE_MODES else ("lattice",))
	for cls, sc in LAYOUTS
	for bits in (8, 14)
]


def _genome(num_clusters: int, bits: int):
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	g = ClusterGenome.__new__(ClusterGenome)
	npc = 4
	g.neurons_per_cluster = [npc] * num_clusters
	g.bits_per_neuron = [bits] * (npc * num_clusters)
	g.connections = [
		(c * 3 + n * 5 + k * 7) % 24
		for c in range(num_clusters) for n in (0, 1, 0, 2) for k in range(bits)
	]
	g.threshold = None
	g.metrics = None
	return g


def _evaluators(mode, rule, cls, sc):
	from test_ids_evaluator_streaming import _make_multiclass_dataset
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	inmem, stream = _make_multiclass_dataset()
	kw = dict(
		classification=cls, num_parts=5, single_cluster=sc, seed=42, empty_value=0.5,
		memory_mode=MODES[mode], write_rule=rule, run_seed=13, neuron_sample_rate=1.0,
	)
	return IDSEvaluator(inmem, **kw), IDSEvaluator(stream, **kw)


def _close(a: float, b: float) -> bool:
	# CE may reduce on the GPU in the in-memory K-cluster fast path.
	return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-6 * max(1.0, abs(b))


@pytest.mark.parametrize("mode,rule,cls,sc,bits", CASES)
def test_streaming_matches_in_memory(monkeypatch, mode, rule, cls, sc, bits):
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")
	ev_mem, ev_str = _evaluators(mode, rule, cls, sc)
	assert ev_str._streaming_mode and not ev_mem._streaming_mode
	k = 1 if sc else (2 if cls == "binary" else 4)

	m_mem = ev_mem.evaluate_batch_full([_genome(k, bits)])[0]
	m_str = ev_str.evaluate_batch_full([_genome(k, bits)])[0]
	assert (m_str.acc, m_str.f1, m_str.fpr, m_str.threshold) == (m_mem.acc, m_mem.f1, m_mem.fpr, m_mem.threshold)
	assert _close(m_str.ce, m_mem.ce)

	if sc:
		e1, t1, v1, mets1, fp1 = ev_mem.evaluate_at_thresholds(_genome(k, bits), [0.5, -1.0])
		e2, t2, v2, mets2, fp2 = ev_str.evaluate_at_thresholds(_genome(k, bits), [0.5, -1.0])
		assert list(e2) == list(e1) and list(t2) == list(t1) and list(v2) == list(v1)
		assert [(m.ce, m.acc, m.f1, m.fpr, m.threshold) for m in mets2] == \
			[(m.ce, m.acc, m.f1, m.fpr, m.threshold) for m in mets1]
		assert fp2 == fp1
		return
	r1 = ev_mem.evaluate_multiclass_at_thresholds(_genome(k, bits))
	r2 = ev_str.evaluate_multiclass_at_thresholds(_genome(k, bits))
	assert r2["num_classes"] == r1["num_classes"] == k
	assert r2["footprint"] == r1["footprint"]
	assert r2["modes"] == r1["modes"]
	if cls == "binary":
		assert {"margin_train_f1", "margin_val_f1"} <= set(r2["modes"])
		# Binary two-cluster routes through the K-cluster decode on BOTH paths.
		assert ev_str.evaluate_at_thresholds(_genome(k, bits), [0.5])["modes"] == r1["modes"]
