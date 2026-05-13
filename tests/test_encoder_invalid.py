"""Unit tests for ThermometerEncoder's invalid_encoding behavior.

Verifies:
- Back-compat: default invalid_encoding=NONE preserves old silent-fillna(0) behavior
- SINGLE_BIT: +1 flag bit per feature; flag=1 on NaN/+Inf/-Inf, value bits cleared
- Fit-time stability: NaN/±Inf in training data don't break threshold computation
- Bit-range accounting: total_bits and feature_bit_ranges reflect +1 per feature

Run: python -m pytest tests/test_encoder_invalid.py -v
Or:  python tests/test_encoder_invalid.py
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Allow running directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

from wnn.ids.encoder import ThermometerEncoder, ThermometerType, InvalidEncoding


def _train_df():
	"""Simple DataFrame with numeric, binary, and categorical features."""
	return pd.DataFrame({
		"num_a": np.array([0.0, 0.25, 0.5, 0.75, 1.0] * 10),  # 50 rows
		"num_b": np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 10),
		"bin_flag": np.array([0, 1] * 25, dtype=int),
		"cat_proto": ["tcp", "udp", "icmp", "tcp", "udp"] * 10,
	})


def _transform_bool(enc, df):
	"""Helper: call transform() and return the unpacked bool view of the
	packed output. After Phase 2 the encoder emits np.packbits(bitorder='little')
	bytes plus a total_bits scalar; tests want the bool form for bit-level checks.
	"""
	packed, total_bits = enc.transform(df)
	unpacked = np.unpackbits(packed, axis=1, count=total_bits, bitorder='little')
	return unpacked.astype(bool)


def test_backcompat_none_default():
	"""Default invalid_encoding=NONE: existing behavior unchanged."""
	df = _train_df()
	enc = ThermometerEncoder(n_bits=4)
	assert enc.invalid_encoding == InvalidEncoding.NONE
	enc.fit(df)
	out = _transform_bool(enc, df)
	# 4 bits (num_a) + 4 bits (num_b) + 1 (bin_flag) + 2 (cat: tcp/udp/icmp → 2 bits) = 11
	assert out.shape == (50, 11), f"expected (50, 11), got {out.shape}"
	assert out.dtype == bool


def test_single_bit_adds_one_per_feature():
	"""SINGLE_BIT: +1 flag per feature → 4 features × +1 = +4 output bits."""
	df = _train_df()
	enc = ThermometerEncoder(n_bits=4, invalid_encoding=InvalidEncoding.SINGLE_BIT)
	enc.fit(df)
	out = _transform_bool(enc, df)
	# 11 value bits + 4 flag bits (one per feature) = 15
	assert out.shape == (50, 15), f"expected (50, 15), got {out.shape}"
	assert enc.total_bits == 15


def test_single_bit_valid_rows_have_flag_zero():
	"""For all-valid input rows, every is_invalid flag bit should be 0."""
	df = _train_df()
	enc = ThermometerEncoder(n_bits=4, invalid_encoding=InvalidEncoding.SINGLE_BIT)
	enc.fit(df)
	out = _transform_bool(enc, df)
	# Flag bits are at positions 0, 5, 10, 12 (start of each feature's range)
	ranges = enc.feature_bit_ranges()
	for col, (start, _end) in ranges.items():
		flag_col = out[:, start]
		assert not flag_col.any(), f"feature {col} has flag=1 on all-valid data"


def test_single_bit_nan_sets_flag_and_clears_value():
	"""NaN input: flag=1, value bits cleared to 0."""
	df = _train_df()
	enc = ThermometerEncoder(n_bits=4, invalid_encoding=InvalidEncoding.SINGLE_BIT)
	enc.fit(df)

	# Corrupt row 0 with NaN in num_a
	test_df = df.copy()
	test_df.loc[0, "num_a"] = np.nan
	out = _transform_bool(enc, test_df)

	ranges = enc.feature_bit_ranges()
	start_a, end_a = ranges["num_a"]
	# Row 0: flag=1 for num_a, value bits zero
	assert out[0, start_a] == 1, "num_a flag should be 1 for NaN row"
	assert not out[0, start_a + 1:end_a].any(), "num_a value bits should be cleared for NaN"
	# Row 1 (untouched): flag=0
	assert out[1, start_a] == 0


def test_single_bit_inf_and_neg_inf():
	"""+Inf and -Inf should both set the invalid flag."""
	df = _train_df()
	enc = ThermometerEncoder(n_bits=4, invalid_encoding=InvalidEncoding.SINGLE_BIT)
	enc.fit(df)

	test_df = df.copy()
	test_df.loc[0, "num_a"] = np.inf
	test_df.loc[1, "num_a"] = -np.inf
	out = _transform_bool(enc, test_df)
	start_a, end_a = enc.feature_bit_ranges()["num_a"]

	assert out[0, start_a] == 1, "+Inf should set flag"
	assert not out[0, start_a + 1:end_a].any(), "+Inf value bits cleared"
	assert out[1, start_a] == 1, "-Inf should set flag"
	assert not out[1, start_a + 1:end_a].any(), "-Inf value bits cleared"


def test_single_bit_nan_in_binary_feature():
	"""NaN in a binary feature also sets its flag."""
	df = _train_df()
	enc = ThermometerEncoder(n_bits=4, invalid_encoding=InvalidEncoding.SINGLE_BIT)
	enc.fit(df)

	test_df = df.copy()
	test_df["bin_flag"] = test_df["bin_flag"].astype(float)
	test_df.loc[0, "bin_flag"] = np.nan
	out = _transform_bool(enc, test_df)

	start_b, end_b = enc.feature_bit_ranges()["bin_flag"]
	assert out[0, start_b] == 1, "binary NaN should set flag"
	assert not out[0, start_b + 1:end_b].any(), "binary value bit cleared for NaN"


def test_fit_robust_to_invalid_in_training_data():
	"""Training data with NaN/±Inf should not crash fit(); thresholds computed from valid rows."""
	df = _train_df()
	df.loc[0, "num_a"] = np.nan
	df.loc[1, "num_a"] = np.inf
	df.loc[2, "num_a"] = -np.inf

	enc = ThermometerEncoder(n_bits=4, invalid_encoding=InvalidEncoding.SINGLE_BIT)
	enc.fit(df)
	# Should have valid thresholds computed from non-invalid rows
	assert "num_a" in enc.thresholds_
	assert np.isfinite(enc.thresholds_["num_a"]).all(), "thresholds contain NaN/Inf"
	# And transform should work on invalid inputs
	out = _transform_bool(enc, df)
	start_a, _ = enc.feature_bit_ranges()["num_a"]
	assert out[0, start_a] == 1
	assert out[1, start_a] == 1
	assert out[2, start_a] == 1


def test_feature_bit_ranges_layout():
	"""Verify ranges are contiguous and in order, accounting for +1 per feature."""
	df = _train_df()
	enc = ThermometerEncoder(n_bits=4, invalid_encoding=InvalidEncoding.SINGLE_BIT)
	enc.fit(df)
	ranges = enc.feature_bit_ranges()
	# Ranges should be contiguous
	previous_end = 0
	for col in enc.feature_names_:
		start, end = ranges[col]
		assert start == previous_end, f"Gap before {col}: {start} != {previous_end}"
		previous_end = end
	assert previous_end == enc.total_bits


def test_string_interface_for_invalid_encoding():
	"""Passing invalid_encoding as a string should work too."""
	df = _train_df()
	enc = ThermometerEncoder(n_bits=4, invalid_encoding="single_bit")
	assert enc.invalid_encoding == InvalidEncoding.SINGLE_BIT
	enc.fit(df)
	out = _transform_bool(enc, df)
	assert out.shape[1] == 15  # same as enum version


if __name__ == "__main__":
	# Run all tests and report
	import traceback
	tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
	passed, failed = 0, 0
	for t in tests:
		try:
			t()
			print(f"  ✓ {t.__name__}")
			passed += 1
		except Exception as e:
			print(f"  ✗ {t.__name__}: {e}")
			traceback.print_exc()
			failed += 1
	print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
	sys.exit(0 if failed == 0 else 1)
