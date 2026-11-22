"""IDS-70 (20/11/2026): ids_cut_method — supervised thermometer cut points fitted on TRAIN only.

quantile (default) must stay byte-identical; info_gain_mc / xgb_splits must reproduce the IDS-63
pre-check (logs/ids/encoding_B/cuts_E.json, cuts_B.json) and never read val/test rows.
The real-data tests (UNSW random_3way, flow 7585's params) run with WNN_IDS70_REAL=1; their expected
sha256 values were produced by the main-line encoder (origin 7fd1a425) and by the pre-check's
/tmp/encB/{T64,E,B}.npz, which matched each other byte-for-byte.
"""
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.ids.dataset import encode_features  # noqa: E402
from wnn.representations.category_rank_map import CategoryRankMap  # noqa: E402
from wnn.representations.thermometer import ThermometerEncoder  # noqa: E402


def _frame(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
	"""Synthetic train-like frame: 2 numeric, 1 categorical, 1 binary; 3-class label."""
	rng = np.random.default_rng(seed)
	y_multi = rng.integers(0, 3, n)
	df = pd.DataFrame({
		"a": rng.gamma(2.0, 10.0, n) + 15.0 * y_multi,
		"b": rng.normal(size=n) + (y_multi == 2),
		"proto": rng.choice(["tcp", "udp", "icmp", "arp"], n),
		"flag": rng.integers(0, 2, n),
	})
	return df, (y_multi > 0).astype(np.int64), y_multi


def _bits(enc: ThermometerEncoder, df: pd.DataFrame) -> np.ndarray:
	packed, total = enc.transform(df)
	return np.unpackbits(packed, axis=1, count=total, bitorder="little")


# ---- quantile: the default is the old encoder ---------------------------------------------------

def test_quantile_default_is_bit_identical():
	df, yb, ym = _frame(4000, 0)
	old = ThermometerEncoder(n_bits=16).fit(df)
	new = ThermometerEncoder(n_bits=16, cut_method="quantile", cut_seed=7).fit(df, y_binary=yb, y_multi=ym)
	assert np.array_equal(old.transform(df)[0], new.transform(df)[0])
	assert old.feature_bit_ranges() == new.feature_bit_ranges()


# ---- info_gain_mc --------------------------------------------------------------------------------

def test_info_gain_keeps_the_quantile_budget_and_ranks_categoricals():
	df, yb, ym = _frame(6000, 1)
	quantile = ThermometerEncoder(n_bits=16).fit(df)
	enc = ThermometerEncoder(n_bits=16, cut_method="info_gain_mc").fit(df, y_binary=yb, y_multi=ym)
	assert enc.feature_types_["proto"] == "ranked" and "proto" in enc.rank_maps_
	assert enc.feature_types_["flag"] == "binary"
	assert enc.total_bits <= quantile.total_bits
	assert _bits(enc, df).shape[1] == enc.total_bits
	ranges = enc.feature_bit_ranges()
	assert sum(e - s for s, e in ranges.values()) == enc.total_bits


def test_rank_map_unseen_and_nan_go_below_the_global_rate_rank():
	s = pd.Series(["x", "x", "y", "y", "z", "z"])
	rank_map = CategoryRankMap.fit(s, np.array([0, 0, 0, 1, 1, 1]))
	assert rank_map.ranks == {"x": 0.0, "y": 1.0, "z": 2.0}
	assert rank_map.unseen == 0.5  # global rate 0.5 sits at index 1 of [0, 0.5, 1] -> 1 - 0.5
	out = rank_map.transform(pd.Series(["z", "never", None]))
	assert out.tolist() == [2.0, 0.5, 0.5]


def test_supervised_fit_requires_aligned_train_labels():
	df, yb, ym = _frame(500, 2)
	with pytest.raises(ValueError, match="train labels"):
		ThermometerEncoder(n_bits=8, cut_method="info_gain_mc").fit(df)
	with pytest.raises(ValueError, match="train labels"):
		ThermometerEncoder(n_bits=8, cut_method="info_gain_mc").fit(df, y_binary=yb[:-1], y_multi=ym)


def test_xgb_splits_cut_numerics_only():
	pytest.importorskip("xgboost")
	df, yb, ym = _frame(3000, 3)
	enc = ThermometerEncoder(n_bits=16, cut_method="xgb_splits", cut_seed=20421).fit(df, y_binary=yb, y_multi=ym)
	assert enc.feature_types_["proto"] == "categorical" and not enc.rank_maps_
	assert len(enc.thresholds_["a"]) > 0 and np.all(np.diff(enc.thresholds_["a"]) > 0)
	again = ThermometerEncoder(n_bits=16, cut_method="xgb_splits", cut_seed=20422).fit(df, y_binary=yb, y_multi=ym)
	assert np.array_equal(enc.transform(df)[0], again.transform(df)[0]), "no subsampling: seed-invariant"


# ---- leakage: the fit sees the TRAIN rows and labels only --------------------------------------

@pytest.mark.parametrize("method", ["info_gain_mc", "xgb_splits"])
def test_fit_never_touches_val_or_test(method, monkeypatch):
	if method == "xgb_splits":
		pytest.importorskip("xgboost")
	from wnn.representations import supervised_cut_fitter as F
	train, yb, ym = _frame(3000, 4)
	seen = []
	real_fit = F.SupervisedCutFitter.fit

	def spy(self, df, feature_types, quantile_bits, y_binary, y_multi):
		seen.append((df.copy(), np.asarray(y_binary).copy(), np.asarray(y_multi).copy()))
		return real_fit(self, df, feature_types, quantile_bits, y_binary, y_multi)

	monkeypatch.setattr(F.SupervisedCutFitter, "fit", spy)
	cols = ["a", "b", "proto", "flag"]

	def encode(val, test):
		return encode_features(train, test, cols, cols, n_bits=8, feature_selection="all", df_val=val,
			cut_method=method, cut_seed=1, y_train_binary=yb, y_train_multi=ym)

	val_a, _, _ = _frame(1000, 5)
	val_b = val_a.assign(a=val_a["a"] * 100.0, proto="unseen")  # wildly different val/test rows
	_, _, enc_a, _, _ = encode(val_a, val_a)
	_, _, enc_b, _, _ = encode(val_b, val_b)
	assert len(seen) == 2
	for df, y_b, y_m in seen:
		pd.testing.assert_frame_equal(df.reset_index(drop=True), train[cols].reset_index(drop=True))
		assert np.array_equal(y_b, yb) and np.array_equal(y_m, ym)
	for col in enc_a.thresholds_:
		assert np.array_equal(enc_a.thresholds_[col], enc_b.thresholds_[col]), f"{col}: val/test moved a cut"


# ---- dedupe after supervised cuts (IDS-73, 21/11/2026) ------------------------------------------

def _dedupe_frame(n: int, seed: int):
	"""_frame + a cross-feature copy, a constant column and a NaN-bearing numeric."""
	df, yb, ym = _frame(n, seed)
	rng = np.random.default_rng(seed + 100)
	df["a_copy"] = df["a"]
	df["const"] = 3.0
	df["c"] = np.where(rng.random(n) < 0.05, np.nan, rng.normal(size=n) + (ym == 1))
	return df, yb, ym


@pytest.mark.parametrize("method", ["info_gain_mc", "xgb_splits"])
def test_dedupe_runs_on_the_full_width_supervised_cuts(method):
	df, yb, ym = _dedupe_frame(6000, 8)
	kw = dict(n_bits=16, cut_method=method, invalid_encoding="single_bit")
	off = ThermometerEncoder(**kw).fit(df, y_binary=yb, y_multi=ym)
	on = ThermometerEncoder(dedupe=True, **kw).fit(df, y_binary=yb, y_multi=ym)
	for col in on.feature_names_:
		if on.feature_types_[col] in ("numeric", "ranked"):
			assert set(on.thresholds_[col].tolist()) <= set(off.thresholds_[col].tolist()), col
	assert "a_copy" in on.dropped_features_ and "const" in on.dropped_features_
	assert on.total_bits < off.total_bits
	bits = _bits(on, df)
	assert bits.shape[1] == on.total_bits
	assert (bits.any(axis=0) & ~bits.all(axis=0)).all(), "a kept train column is constant"
	assert len({bits[:, j].tobytes() for j in range(bits.shape[1])}) == bits.shape[1], "duplicate column"
	assert on._has_flag("c") and not on._has_flag("a")


def test_dedupe_after_cuts_keeps_the_cuts_that_were_already_informative():
	"""Supervised cuts are fitted on the same budget with and without dedupe."""
	df, yb, ym = _frame(6000, 9)
	off = ThermometerEncoder(n_bits=16, cut_method="info_gain_mc").fit(df, y_binary=yb, y_multi=ym)
	on = ThermometerEncoder(n_bits=16, cut_method="info_gain_mc", dedupe=True).fit(df, y_binary=yb, y_multi=ym)
	for col in ("a", "b"):
		assert np.array_equal(on.thresholds_[col], off.thresholds_[col]), col


# ---- refusals / registry / cache key -----------------------------------------------------------

def test_refusals():
	with pytest.raises(NotImplementedError, match="streaming"):
		ThermometerEncoder(n_bits=8, cut_method="xgb_splits").begin_streaming_fit({"x": "numeric"})
	with pytest.raises(ValueError):
		ThermometerEncoder(n_bits=8, cut_method="mdlp")
	df, yb, ym = _frame(200, 6)
	with pytest.raises(NotImplementedError, match="top20_split"):
		encode_features(df, df, ["a"], ["a"], feature_selection="top20_split", cut_method="info_gain_mc",
			y_train_binary=yb, y_train_multi=ym)


def test_streaming_factory_refuses_supervised_cuts():
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
	s = IdsEvaluatorSettings.from_params({"ids_dataset": "ciciot2023_neto_full", "ids_streaming": True,
		"ids_cut_method": "info_gain_mc"})
	with pytest.raises(NotImplementedError, match="ids_cut_method"):
		IdsEvaluatorFactory._check_streaming(s)


def test_param_registry_resolution_and_cache_key():
	from wnn.ram.experiments.ids_param_resolution import resolve_cut_method
	from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
	from wnn.ram.experiments.params import KNOWN_PARAMS
	from wnn.ram.experiments.test_cache_key import build_test_cache_key
	assert "ids_cut_method" in KNOWN_PARAMS
	assert resolve_cut_method({}) == "quantile"
	assert IdsEvaluatorSettings.from_params({"ids_cut_method": "xgb_splits"}).cut_method == "xgb_splits"
	with pytest.raises(ValueError):
		resolve_cut_method({"ids_cut_method": "E"})
	base = {"ids_dataset": "unsw-nb15", "seed": 20421}
	assert build_test_cache_key(base, True) == build_test_cache_key(dict(base, ids_cut_method="quantile"), True)
	e = build_test_cache_key(dict(base, ids_cut_method="info_gain_mc"), True)
	b = build_test_cache_key(dict(base, ids_cut_method="xgb_splits"), True)
	assert "|cut=info_gain_mc:v1|" in e and "|cut=xgb_splits:v1|" in b and e != b


# ---- real data: UNSW random_3way through the worker path (WNN_IDS70_REAL=1) ---------------------

REAL = os.environ.get("WNN_IDS70_REAL") == "1"
DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
PRECHECK = Path("/Users/lacg/wnn/logs/ids/encoding_B")
EXPECTED = {
	"quantile": (1104, {"train": "efd1d965d297bb769fd3ae02d02e71fd6b3572f114dba5873fa669e069c70df1",
		"val": "6f914618089fcd2ea5943a484ccc4e4b5392c30d7b6510592f2d369a3806cf08",
		"test": "da3ed6593ffe48802a77eefe1cf55cb038fd640ce78df9dc38e4999f960756df"}),
	"info_gain_mc": (1104, {"train": "8bccee68f83e4996101b35f07bed466f04a872e9af46dd6ae2781ccdd22928ec",
		"val": "85455b98ddb0891e5d0887d7f64a305fb84bfd4163a7c2f0baae223912d77f9f",
		"test": "bf067b3e5201f186feebd5de069168392aae47772f8bdf2fb76c4f35d3e5b710"}),
	"xgb_splits": (457, {"train": "454a5e5ef54b761566f7bded6320beb1fe30a66b7f63d92a8f15ff6a27948691",
		"val": "142deb0dcffb51fd3d271b81147e80ccd0c6d88b659623401ec8ecba8ae1738d",
		"test": "366daadbeb41954d82574b88b0dd6520503638b25433ede3839d73a5f03ae26e"}),
}
PRECHECK_CUTS = {"info_gain_mc": "cuts_E.json", "xgb_splits": "cuts_B.json"}


def _flow_7585_dataset(method: str):
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
	params = json.loads(sqlite3.connect(DB, uri=True).execute(
		"select config_json from flows where id=7585").fetchone()[0])["params"]
	return IdsEvaluatorFactory(log=print).load(IdsEvaluatorSettings.from_params(dict(params, ids_cut_method=method)))


@pytest.mark.skipif(not REAL, reason="set WNN_IDS70_REAL=1 (loads UNSW-NB15 random_3way, ~10 s, ~3 GB)")
@pytest.mark.parametrize("method", list(EXPECTED))
def test_real_unsw_random_3way_matches_mainline_and_precheck(method):
	ds = _flow_7585_dataset(method)
	total, hashes = EXPECTED[method]
	assert ds.X_train.total_bits == total
	for part, digest in hashes.items():
		packed = np.ascontiguousarray(getattr(ds, f"X_{part}").as_packed_uint8())
		assert hashlib.sha256(packed.tobytes()).hexdigest() == digest, f"{method} {part}"
	if method in PRECHECK_CUTS:
		ref = json.loads((PRECHECK / PRECHECK_CUTS[method]).read_text())
		for feature, cuts in ref.items():
			assert np.array_equal(ds.encoder.thresholds_[feature], np.array(cuts)), feature
