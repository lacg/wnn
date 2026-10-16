"""ids_two_cluster_fitness_f1 (worker ABI 20, 16/10/2026): which F1 a binary
TWO-cluster search reports to the GA fitness calculators and the patience
tracker. "macro" (default) = the 2-class macro F1, unchanged; "attack" = the
attack class's F1. Pinned on the in-memory AND the streaming path:
  - "attack" search F1 == the attack-class F1 of the validation argmax decode
    (evaluate_multiclass_at_thresholds 'argmax' -> 'attack_f1');
  - "macro" search F1 == the default evaluator's F1 == the argmax macro F1;
  - only F1 moves (CE / acc / FPR identical);
  - non-default values are refused on single-cluster / multiclass flows.
The Rust twin is streamer_parity_tests::two_cluster_fitness_f1_attack_and_macro.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "wnn"))
sys.path.insert(0, str(ROOT / "tests"))

from wnn.ram.experiments.ids_param_resolution import resolve_two_cluster_fitness_f1
from wnn.ram.experiments.params import KNOWN_PARAMS
from wnn.ram.experiments.validation_cache_key import build_validation_cache_key


def _abi() -> int:
	try:
		import ram_accelerator
		return int(getattr(ram_accelerator, "ABI_VERSION", 0))
	except ImportError:
		return 0


needs_abi20 = pytest.mark.skipif(_abi() < 20, reason="needs ram_accelerator ABI >= 20")
TWO_CLUSTER = {"ids_classification": "binary", "ids_single_cluster": False}


# ---- params / resolution (pure Python) ---------------------------------------

def test_two_cluster_fitness_f1_is_a_known_param():
	assert "ids_two_cluster_fitness_f1" in KNOWN_PARAMS


def test_resolution_defaults_and_validation():
	assert resolve_two_cluster_fitness_f1({}) == "macro"
	assert resolve_two_cluster_fitness_f1({"ids_two_cluster_fitness_f1": None}) == "macro"
	assert resolve_two_cluster_fitness_f1({**TWO_CLUSTER, "ids_two_cluster_fitness_f1": "attack"}) == "attack"
	# The default is accepted everywhere (every existing flow).
	for cls, sc in (("binary", True), ("multi", False), ("hierarchical", False)):
		assert resolve_two_cluster_fitness_f1(
			{"ids_classification": cls, "ids_single_cluster": sc, "ids_two_cluster_fitness_f1": "macro"}) == "macro"
	with pytest.raises(ValueError, match="expected one of"):
		resolve_two_cluster_fitness_f1({**TWO_CLUSTER, "ids_two_cluster_fitness_f1": "weighted"})
	for cls, sc in (("binary", True), ("multi", False), ("hierarchical", False)):
		with pytest.raises(ValueError, match="binary two-cluster flows only"):
			resolve_two_cluster_fitness_f1(
				{"ids_classification": cls, "ids_single_cluster": sc, "ids_two_cluster_fitness_f1": "attack"})


def test_selector_does_not_touch_the_validation_cache_key():
	"""Search-only: a fixed genome's validation is identical under both values."""
	base = build_validation_cache_key(dict(TWO_CLUSTER), oi_enabled=True)
	assert build_validation_cache_key({**TWO_CLUSTER, "ids_two_cluster_fitness_f1": "attack"}, oi_enabled=True) == base


# ---- accelerator-backed ------------------------------------------------------

def _genome(bits: int):
	from test_ids_streaming_parity import _genome as parity_genome
	return parity_genome(2, bits)


def _evaluator(streaming: bool, fitness_f1: str, mode: int):
	from test_ids_evaluator_streaming import _make_multiclass_dataset
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	inmem, stream = _make_multiclass_dataset()
	return IDSEvaluator(
		stream if streaming else inmem, classification="binary", single_cluster=False, num_parts=5, seed=42,
		empty_value=0.0, memory_mode=mode, neuron_sample_rate=1.0, two_cluster_fitness_f1=fitness_f1,
	)


@needs_abi20
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("bits", [8, 14])
def test_attack_fitness_f1_is_the_attack_f1_metric(monkeypatch, streaming, bits):
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")
	default = _evaluator(streaming, "macro", 2).evaluate_batch_full([_genome(bits)])[0]
	macro_ev = _evaluator(streaming, "macro", 2)
	attack_ev = _evaluator(streaming, "attack", 2)
	macro = macro_ev.evaluate_batch_full([_genome(bits)])[0]
	attack = attack_ev.evaluate_batch_full([_genome(bits)])[0]
	argmax = macro_ev.evaluate_multiclass_at_thresholds(_genome(bits))["modes"]["argmax"]

	assert macro.f1 == default.f1 == argmax["macro_f1"]
	assert attack.f1 == argmax["attack_f1"]
	assert attack.f1 != macro.f1
	assert (attack.ce, attack.acc, attack.fpr) == (macro.ce, macro.acc, macro.fpr)


@needs_abi20
def test_attack_fitness_f1_same_on_both_paths(monkeypatch):
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")
	for mode in (0, 2, 3, 5):
		mem = _evaluator(False, "attack", mode).evaluate_batch_full([_genome(8)])[0]
		strm = _evaluator(True, "attack", mode).evaluate_batch_full([_genome(8)])[0]
		assert (strm.acc, strm.f1, strm.fpr) == (mem.acc, mem.f1, mem.fpr), f"mode {mode}"


@needs_abi20
def test_attack_refused_off_two_cluster_binary():
	from test_ids_evaluator_streaming import _make_multiclass_dataset
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	inmem, stream = _make_multiclass_dataset()
	for ds in (inmem, stream):
		for cls, sc in (("binary", True), ("multi", False)):
			with pytest.raises(ValueError, match="binary two-cluster"):
				IDSEvaluator(ds, classification=cls, single_cluster=sc, num_parts=5, seed=42,
					two_cluster_fitness_f1="attack")
		with pytest.raises(ValueError, match="expected 'macro' or 'attack'"):
			IDSEvaluator(ds, classification="binary", single_cluster=False, num_parts=5, seed=42,
				two_cluster_fitness_f1="micro")
	import ram_accelerator
	# The Rust surfaces refuse the same (a caller that bypasses IDSEvaluator).
	with pytest.raises(ValueError, match="binary two-cluster"):
		ram_accelerator.IDSGenomeStreamerWrapper(
			bits_flat=[4] * 4, neurons_flat=[4], connections=list(range(16)), num_classes=2, num_negatives=1,
			single_cluster=True, total_features=24, two_cluster_fitness_f1="attack",
		)
