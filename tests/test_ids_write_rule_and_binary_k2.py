"""ids_write_rule (MAJORITY VOTE, worker ABI 19) + binary TWO-cluster validation.

Two things land together (16/10/2026):
1. `ids_write_rule="vote"`: BINARY/TERNARY/PLN train through a signed
   class-weighted tally per cell instead of BINARY's one-shot Set / the
   TRUE-wins lattice. Default "lattice" must be bit-identical (cache keys,
   scores). The Rust parity suites live in cargo (vote_writer_tests.rs).
2. Binary two-cluster (ids_classification='binary', ids_single_cluster=false)
   validation used to fall through BOTH the single-cluster threshold block and
   the multiclass decode block (no threshold modes, no val-calibrated decode).
   It now takes the K-cluster decode block: argmax headline (what the GA
   fitness used) + margin_* modes, with K=2-only margin_train_f1 /
   margin_val_f1 (attack-F1 tau; margin_val_f1 is single-cluster val_cal's twin)
   and an 'f1' alias = attack-class F1.

Accelerator-backed tests skip unless the installed ram_accelerator is ABI >= 19.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "wnn"))

from wnn.ram.experiments.ids_param_resolution import resolve_write_rule
from wnn.ram.experiments.params import KNOWN_PARAMS
from wnn.ram.experiments.validation_cache_key import build_validation_cache_key


def _abi() -> int:
	try:
		import ram_accelerator
		return int(getattr(ram_accelerator, "ABI_VERSION", 0))
	except ImportError:
		return 0


needs_abi19 = pytest.mark.skipif(_abi() < 19, reason="needs ram_accelerator ABI >= 19")


# ---- params / resolution / cache key (pure Python) ---------------------------

def test_write_rule_is_a_known_param():
	assert "ids_write_rule" in KNOWN_PARAMS


def test_resolve_write_rule_defaults_and_validation():
	assert resolve_write_rule({}) == "lattice"
	assert resolve_write_rule({"ids_write_rule": None}) == "lattice"
	assert resolve_write_rule({"ids_write_rule": "lattice", "memory_mode": "QUAD_WEIGHTED"}) == "lattice"
	for mode in ("BINARY", "TERNARY", "PLN"):
		assert resolve_write_rule({"ids_write_rule": "vote", "memory_mode": mode}) == "vote"
	with pytest.raises(ValueError):
		resolve_write_rule({"ids_write_rule": "majority"})
	for mode in ("QUAD_WEIGHTED", "QUAD_BINARY", "QSR"):
		with pytest.raises(ValueError):
			resolve_write_rule({"ids_write_rule": "vote", "memory_mode": mode})
	with pytest.raises(ValueError):  # absent memory_mode resolves to QUAD_WEIGHTED
		resolve_write_rule({"ids_write_rule": "vote"})


def test_cache_key_unchanged_for_lattice_and_split_for_vote():
	base = {"memory_mode": "TERNARY", "ids_dataset": "unsw-nb15"}
	absent = build_validation_cache_key(base, True)
	assert build_validation_cache_key(dict(base, ids_write_rule="lattice"), True) == absent
	vote = build_validation_cache_key(dict(base, ids_write_rule="vote"), True)
	assert vote != absent and "wr=vote" in vote
	assert "wr=" not in absent


# ---- accelerator-backed ------------------------------------------------------

def _dataset(n_train=600, n_val=200, n_test=200, total_bits=24, seed=11):
	"""Binary IDS-like data with a val partition (Protocol v2). label = bit 0 OR
	bit 1, with 10% label noise so calibrated modes have something to do."""
	from wnn.ids.dataset import IDSDataset
	from wnn.ids.encoded_array import InMemoryEncoded
	from wnn.representations.thermometer import ThermometerEncoder

	rng = np.random.default_rng(seed)
	n = n_train + n_val + n_test
	df = pd.DataFrame({f"bin_{i}": rng.integers(0, 2, size=n) for i in range(total_bits)})
	y = ((df["bin_0"].values | df["bin_1"].values) ^ (rng.random(n) < 0.1)).astype(np.int64)
	parts = [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, n)]
	enc = ThermometerEncoder(n_bits=1)
	enc.fit(df.iloc[:n_train])
	X = []
	for a, b in parts:
		packed, bits = enc.transform(df.iloc[a:b])
		X.append(InMemoryEncoded(packed, total_bits=bits))
	ys = [y[a:b] for a, b in parts]
	return IDSDataset(
		X_train=X[0], y_train_binary=ys[0], y_train_multi=ys[0].copy(),
		X_test=X[2], y_test_binary=ys[2], y_test_multi=ys[2].copy(),
		X_val=X[1], y_val_binary=ys[1], y_val_multi=ys[1].copy(),
		encoder=enc, category_names=["Normal", "Attack"], feature_names=list(df.columns),
	)


def _evaluator(ds, single_cluster: bool, memory_mode: int, write_rule: str = "lattice"):
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	return IDSEvaluator(
		dataset=ds, classification="binary", num_parts=1, empty_value=0.5,
		memory_mode=memory_mode, seed=3, run_seed=3, neuron_sample_rate=1.0,
		single_cluster=single_cluster, write_rule=write_rule,
	)


def _genome(num_clusters: int, neurons=6, bits=6, total_bits=24, seed=5):
	from wnn.ram.genome import ClusterGenome
	rng = np.random.default_rng(seed)
	n = num_clusters * neurons
	conns = [int(c) for c in rng.integers(0, total_bits, size=n * bits)]
	# Every neuron sees the label bits so the task is learnable.
	for k in range(n):
		conns[k * bits] = 0
		conns[k * bits + 1] = 1
	return ClusterGenome([bits] * n, [neurons] * num_clusters, conns)


TERNARY, BINARY, QUAD_WEIGHTED = 0, 3, 2


@needs_abi19
def test_binary_two_cluster_decode_modes_have_val_calibrated_attack_f1():
	ds = _dataset()
	ev = _evaluator(ds, single_cluster=False, memory_mode=QUAD_WEIGHTED)
	out = ev.evaluate_multiclass_at_thresholds(_genome(2))
	modes = out["modes"]
	for key in ("argmax", "margin_fixed0", "margin_train_cal", "margin_val_cal", "margin_train_f1", "margin_val_f1"):
		assert key in modes, key
	attack = "Attack"
	for key, m in modes.items():
		assert m["f1"] == pytest.approx(m["per_class"][attack]["f1"]), key  # attack-class F1, not macro
		assert m["attack_f1"] == m["f1"]
		assert 0.0 <= m["fpr"] <= 1.0
	assert "tau" in modes["margin_val_f1"] and "tau" not in modes["argmax"]
	# argmax decode == the legacy evaluate_batch_full decision (same acc/FPR);
	# the legacy 'f1' there is the 2-class MACRO F1, kept here as macro_f1.
	full = ev.evaluate_batch_full([_genome(2)])[0]
	assert modes["argmax"]["acc"] == pytest.approx(full.acc)
	assert modes["argmax"]["benign_fpr"] == pytest.approx(full.fpr)
	assert modes["argmax"]["macro_f1"] == pytest.approx(full.f1)


@needs_abi19
def test_experiment_validates_binary_two_cluster_through_decode_modes():
	"""The experiment-level validation writes a decode-mode threshold_metadata
	(argmax headline + margin_val_f1) for a binary two-cluster evaluator."""
	from wnn.ram.experiments.experiment import Experiment, ExperimentConfig, ExperimentType
	from wnn.ram.metrics import GenomeType

	captured = {}

	class _Client:
		def create_validation_summary(self, **kw):
			captured.update(kw)

	ds = _dataset()
	ev = _evaluator(ds, single_cluster=False, memory_mode=TERNARY, write_rule="vote")
	exp = Experiment(
		ExperimentConfig(name="k2", experiment_type=ExperimentType.GA), ev, logger=lambda s: None,
		dashboard_client=_Client(), experiment_id=1, full_evaluator=ev,
	)
	results = {}
	exp._validate_one_genome(_genome(2), GenomeType.BEST_F1, "final", None, results,
		use_cache=False, submit_leaderboard=False)
	import json
	tm = json.loads(captured["threshold_metadata"])
	assert "argmax" in tm and "margin_val_f1" in tm and "margin_val_cal" in tm
	r = results[GenomeType.BEST_F1.value]
	assert r["f1"] == pytest.approx(tm["argmax"]["f1"])        # headline = argmax, attack F1
	assert r["fpr"] == pytest.approx(tm["argmax"]["benign_fpr"])


@needs_abi19
def test_set_write_rule_validates_and_vote_changes_training():
	ds = _dataset()
	g = _genome(1)
	lat = _evaluator(ds, True, TERNARY)
	vote = _evaluator(ds, True, TERNARY, "vote")
	s_lat = lat.evaluate_at_thresholds(g, [-1.0, 0.5])[0]
	s_vote = vote.evaluate_at_thresholds(g, [-1.0, 0.5])[0]
	assert list(s_lat) != list(s_vote)
	# default explicitly set == never set (bit-identical)
	again = _evaluator(ds, True, TERNARY)
	assert list(again.evaluate_at_thresholds(g, [-1.0, 0.5])[0]) == list(s_lat)
	with pytest.raises(ValueError):
		_evaluator(ds, True, QUAD_WEIGHTED, "vote")
	with pytest.raises(ValueError):
		lat._cache.set_write_rule("majority")


@needs_abi19
def test_vote_lowers_fpr_floor_on_attack_contaminated_addresses():
	"""Single-cluster BINARY: under Set any address touched by one attack fires;
	vote makes it fire only when attacks outnumber normals. On data where most
	normals share addresses with a few attacks, vote's FPR at the fixed 0.5
	threshold is lower."""
	ds = _dataset()
	g = _genome(1)
	_, _, _, (_, fixed_set), _ = _evaluator(ds, True, BINARY).evaluate_at_thresholds(g, [-1.0, 0.5])
	_, _, _, (_, fixed_vote), _ = _evaluator(ds, True, BINARY, "vote").evaluate_at_thresholds(g, [-1.0, 0.5])
	assert fixed_vote.fpr <= fixed_set.fpr
