"""IDS-52 (08/11/2026): ids_encoding_dedupe — train-informative thermometer bits.

The distributive encoder "deduped" equal quantiles by +1e-10, so integer /
heavy-tailed features produced zero-width bins: constant and duplicate bit
columns (audit: ~half of every IDS input layer). With dedupe ON only bits that
are non-constant and distinct ON THE TRAIN PARTITION survive, uninformative
features and exact-duplicate columns are dropped; OFF stays bit-identical.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.ids.duplicate_columns import distinct_columns
from wnn.representations.thermometer import ThermometerEncoder


def _frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	heavy = np.where(rng.random(n) < 0.7, 0, rng.integers(1, 6, n)).astype(np.float64)
	return pd.DataFrame({
		"heavy": heavy,                        # 6 distinct values, 70% zeros
		"smooth": rng.normal(size=n),          # continuous
		"const": np.full(n, 3.0),              # constant
		"flag": rng.integers(0, 2, n),         # binary
		"twin": heavy.copy(),                  # exact duplicate of heavy
	})


def _bits(enc: ThermometerEncoder, df: pd.DataFrame) -> np.ndarray:
	packed, total = enc.transform(df)
	return np.unpackbits(packed, axis=1, count=total, bitorder="little").astype(bool)


def _constant_or_duplicate(bits: np.ndarray) -> int:
	cols = [bits[:, j].tobytes() for j in range(bits.shape[1])]
	constant = sum(1 for j in range(bits.shape[1]) if bits[:, j].all() or not bits[:, j].any())
	return constant + (len(cols) - len(set(cols)))


def test_dedupe_off_keeps_the_original_widths_and_dead_bits():
	df = _frame()
	enc = ThermometerEncoder(n_bits=16).fit(df)
	assert enc.feature_names_ == list(df.columns)
	assert enc.total_bits == 16 * 4 + 1  # heavy/smooth/const/twin at 16b + the binary column
	assert _constant_or_duplicate(_bits(enc, df)) > 0, "fixture must exhibit the dead-bit problem"


def test_dedupe_on_every_train_bit_is_informative():
	df = _frame()
	cols, dropped = distinct_columns(df, list(df.columns))
	assert dropped == {"twin": "heavy"}
	enc = ThermometerEncoder(n_bits=16, dedupe=True).fit(df[cols])
	assert "const" in enc.dropped_features_ and "const" not in enc.feature_names_
	assert enc.per_feature_bits_["heavy"] == 5, "6 distinct values support exactly 5 thresholds"
	assert enc.per_feature_bits_["smooth"] == 16
	assert _constant_or_duplicate(_bits(enc, df[cols])) == 0


def test_dedupe_is_fitted_on_train_only():
	"""A value that appears only in test must not create a bit."""
	train = pd.DataFrame({"x": np.repeat([0.0, 2.0], 500)})
	test = pd.DataFrame({"x": [0.0, 2.0, 7.0]})
	enc = ThermometerEncoder(n_bits=8, dedupe=True).fit(train)
	assert enc.per_feature_bits_["x"] == 1
	assert _bits(enc, test).shape[1] == 1


def test_dedupe_flag_kept_only_when_train_has_invalid_rows():
	n = 1000
	df = pd.DataFrame({
		"clean": np.arange(n, dtype=np.float64),
		"dirty": np.where(np.arange(n) % 10 == 0, np.nan, np.arange(n, dtype=np.float64)),
	})
	enc = ThermometerEncoder(n_bits=8, invalid_encoding="single_bit", dedupe=True).fit(df)
	ranges = enc.feature_bit_ranges()
	assert ranges["clean"][1] - ranges["clean"][0] == 8, "no flag: train has no invalid rows"
	assert ranges["dirty"][1] - ranges["dirty"][0] == 9, "flag kept"
	assert _constant_or_duplicate(_bits(enc, df)) == 0


def test_streaming_fit_refuses_dedupe():
	with pytest.raises(NotImplementedError, match="streaming"):
		ThermometerEncoder(n_bits=8, dedupe=True).begin_streaming_fit({"x": "numeric"})


def test_param_resolution_and_cache_key():
	from wnn.ram.experiments.ids_param_resolution import resolve_encoding_dedupe
	from wnn.ram.experiments.params import KNOWN_PARAMS
	from wnn.ram.experiments.test_cache_key import build_test_cache_key
	assert "ids_encoding_dedupe" in KNOWN_PARAMS
	assert resolve_encoding_dedupe({}) is False
	assert resolve_encoding_dedupe({"ids_encoding_dedupe": "on"}) is True
	with pytest.raises(ValueError):
		resolve_encoding_dedupe({"ids_encoding_dedupe": True})
	base = {"ids_dataset": "unsw-nb15"}
	assert build_test_cache_key(base, True) == build_test_cache_key(dict(base, ids_encoding_dedupe="off"), True)
	assert "|enc=dedupe1|" in build_test_cache_key(dict(base, ids_encoding_dedupe="on"), True)


def test_dedupe_drops_cross_feature_copies_of_near_identical_columns():
	"""cicids `Avg Bwd Segment Size` ~ `Bwd Packet Length Mean`: 1e-6 rounding in a
	few rows, not an exact duplicate column — but every thermometer bit is a copy."""
	rng = np.random.default_rng(3)
	a = np.round(rng.gamma(2.0, 100.0, 5000), 6)
	b = a.copy()
	b[:7] += 1e-6
	df = pd.DataFrame({"a": a, "b": b, "c": rng.normal(size=5000)})
	cols, dropped = distinct_columns(df, list(df.columns))
	assert dropped == {}, "not an exact duplicate"
	enc = ThermometerEncoder(n_bits=32, dedupe=True).fit(df[cols])
	assert "b" in enc.dropped_features_
	assert _constant_or_duplicate(_bits(enc, df[cols])) == 0
