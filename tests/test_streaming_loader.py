"""Unit tests for encode_and_build_dataset_streaming (Option F item 5).

Uses a synthetic in-memory factory (no HF dependency) to verify:
- Streaming-fit encoder + StreamingEncoded build round-trips correctly.
- Re-iterable property: iter_chunks() can be called multiple times.
- Label materialization is correct (small RAM cost vs streaming features).
- Total bit width matches the in-memory path.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))

from wnn.ids.dataset import (
	encode_and_build_dataset,
	encode_and_build_dataset_streaming,
)
from wnn.ids.encoded_array import StreamingEncoded
from wnn.ids.encoder import ThermometerType


def _make_dfs(n_train=1500, n_test=300, seed=42):
	rng = np.random.default_rng(seed)
	cols = {
		"feat_a": rng.normal(0, 1, size=n_train + n_test),
		"feat_b": rng.exponential(2.0, size=n_train + n_test),
		"feat_c": rng.uniform(-5, 5, size=n_train + n_test),
		"bin_flag": rng.integers(0, 2, size=n_train + n_test),
		"label": rng.integers(0, 2, size=n_train + n_test),
		"attack_class": np.array(["Benign", "Attack1"])[rng.integers(0, 2, size=n_train + n_test)],
	}
	df = pd.DataFrame(cols)
	return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def _chunked_factory(df: pd.DataFrame, chunk_size: int):
	"""Build a re-iterable factory that yields chunks of `df`."""
	def factory():
		for start in range(0, len(df), chunk_size):
			yield df.iloc[start:start + chunk_size].copy()
	return factory


def test_streaming_builds_dataset_with_streaming_encoded_X():
	"""encode_and_build_dataset_streaming returns IDSDataset with
	StreamingEncoded X_train/X_test and materialized labels."""
	df_train, df_test = _make_dfs(n_train=1000, n_test=200)
	common = ["feat_a", "feat_b", "feat_c", "bin_flag"]

	ds_stream = encode_and_build_dataset_streaming(
		_chunked_factory(df_train, chunk_size=250),
		_chunked_factory(df_test, chunk_size=100),
		None, 1000, 200, 0,
		common_features=common,
		top_features=[],
		category_names=["Benign", "Attack1"],
		n_bits=4,
		feature_selection="all",
	)

	assert isinstance(ds_stream.X_train, StreamingEncoded)
	assert isinstance(ds_stream.X_test, StreamingEncoded)
	assert ds_stream.X_train.n_rows == 1000
	assert ds_stream.X_test.n_rows == 200
	# Labels materialized
	assert ds_stream.y_train_binary.shape == (1000,)
	assert ds_stream.y_test_binary.shape == (200,)


def test_streaming_total_bits_matches_in_memory():
	"""Same dfs through both paths must produce identical total_bits."""
	df_train, df_test = _make_dfs(n_train=800, n_test=100)
	common = ["feat_a", "feat_b", "feat_c", "bin_flag"]

	ds_mem = encode_and_build_dataset(
		df_train.copy(), df_test.copy(), None,
		common_features=common, top_features=[],
		category_names=["Benign", "Attack1"],
		n_bits=4, feature_selection="all",
	)

	ds_stream = encode_and_build_dataset_streaming(
		_chunked_factory(df_train, chunk_size=200),
		_chunked_factory(df_test, chunk_size=50),
		None, 800, 100, 0,
		common_features=common, top_features=[],
		category_names=["Benign", "Attack1"],
		n_bits=4, feature_selection="all",
	)

	assert ds_mem.X_train.total_bits == ds_stream.X_train.total_bits, \
		f"total_bits mismatch: mem={ds_mem.X_train.total_bits} stream={ds_stream.X_train.total_bits}"


def test_streaming_iter_chunks_reassembles_full_data():
	"""Iterating the StreamingEncoded train + concatenating chunks recovers
	rows equal to n_train, with matching label values from the materialized
	y_train_binary."""
	df_train, df_test = _make_dfs(n_train=600, n_test=100)
	common = ["feat_a", "feat_b", "feat_c", "bin_flag"]

	ds = encode_and_build_dataset_streaming(
		_chunked_factory(df_train, chunk_size=150),
		_chunked_factory(df_test, chunk_size=50),
		None, 600, 100, 0,
		common_features=common, top_features=[],
		category_names=["Benign", "Attack1"],
		n_bits=4, feature_selection="all",
	)

	# Iterate train stream; verify chunks reassemble correctly
	all_labels = []
	row_count = 0
	for packed_chunk, labels_chunk in ds.X_train.iter_chunks():
		all_labels.append(labels_chunk)
		row_count += packed_chunk.shape[0]

	assert row_count == 600
	streamed_labels = np.concatenate(all_labels)
	assert np.array_equal(streamed_labels, ds.y_train_binary), \
		"streamed labels diverged from materialized y_train_binary"


def test_streaming_re_iterable():
	"""StreamingEncoded.iter_chunks can be called twice; both passes yield
	identical data."""
	df_train, df_test = _make_dfs(n_train=400, n_test=80)
	common = ["feat_a", "feat_b", "feat_c", "bin_flag"]

	ds = encode_and_build_dataset_streaming(
		_chunked_factory(df_train, chunk_size=100),
		_chunked_factory(df_test, chunk_size=40),
		None, 400, 80, 0,
		common_features=common, top_features=[],
		category_names=["Benign", "Attack1"],
		n_bits=4, feature_selection="all",
	)

	pass_1 = [p.copy() for p, _ in ds.X_train.iter_chunks()]
	pass_2 = [p.copy() for p, _ in ds.X_train.iter_chunks()]
	assert len(pass_1) == len(pass_2)
	for p1, p2 in zip(pass_1, pass_2):
		assert np.array_equal(p1, p2), "two passes yielded different data — factory not re-iterable"


def test_streaming_top20_split_raises():
	"""feature_selection='top20_split' not yet supported in streaming."""
	df_train, df_test = _make_dfs(n_train=100, n_test=20)
	common = ["feat_a", "feat_b", "feat_c", "bin_flag"]

	try:
		encode_and_build_dataset_streaming(
			_chunked_factory(df_train, 25), _chunked_factory(df_test, 10),
			None, 100, 20, 0,
			common_features=common, top_features=["feat_a"],
			category_names=["Benign", "Attack1"],
			n_bits=4, feature_selection="top20_split",
		)
	except NotImplementedError as e:
		assert "top20_split" in str(e)
		return
	raise AssertionError("streaming top20_split should raise NotImplementedError")


if __name__ == "__main__":
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
	print(f"\n{passed} passed, {failed} failed of {len(tests)}")
	sys.exit(0 if failed == 0 else 1)
