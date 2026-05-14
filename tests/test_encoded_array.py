"""Tests for LazyEncodedArray implementations (InMemoryEncoded, MemmapEncoded).

Phase 4: MemmapEncoded round-trip, row_subset materialization, tmp file
cleanup, and reuse-after-restart for non-.tmp paths.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))

from wnn.ids.encoded_array import (
	InMemoryEncoded,
	LazyEncodedArray,
	MemmapEncoded,
	StreamingEncoded,
	write_packed_to_memmap,
)


def _make_bool_matrix(n_rows: int, total_bits: int, seed: int = 42) -> np.ndarray:
	"""Reproducible bool matrix for round-trip tests."""
	rng = np.random.default_rng(seed)
	return rng.integers(0, 2, size=(n_rows, total_bits), dtype=np.uint8).astype(bool)


def test_inmemory_packed_roundtrip():
	"""Sanity check InMemoryEncoded still works with packed input."""
	bools = _make_bool_matrix(100, 13)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder='little')
	enc = InMemoryEncoded(packed, total_bits=13)
	assert enc.n_rows == 100
	assert enc.total_bits == 13
	# Round-trip back to bool form
	round_tripped = enc.to_numpy_bool()
	assert np.array_equal(round_tripped, bools), "InMemoryEncoded packed→bool roundtrip"


def test_memmap_packed_roundtrip(tmp_path):
	"""MemmapEncoded preserves packed bytes after write+reopen."""
	bools = _make_bool_matrix(500, 96)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder='little')
	tmp_file = tmp_path / "rt.bin"
	packed.tofile(str(tmp_file))

	enc = MemmapEncoded(tmp_file, n_rows=500, total_bits=96, mode="r")
	assert enc.n_rows == 500
	assert enc.total_bits == 96
	# Read back individual rows and check bit-exactness
	for i in [0, 17, 250, 499]:
		row_packed = enc[i]
		row_bool = np.unpackbits(row_packed, bitorder='little')[:96].astype(bool)
		assert np.array_equal(row_bool, bools[i]), f"row {i} mismatch"


def test_memmap_iter_chunks(tmp_path):
	"""iter_chunks yields contiguous row slabs that reassemble to the full matrix."""
	bools = _make_bool_matrix(200, 32)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder='little')
	tmp_file = tmp_path / "chunks.bin"
	packed.tofile(str(tmp_file))

	enc = MemmapEncoded(tmp_file, n_rows=200, total_bits=32, mode="r")
	chunks = list(enc.iter_chunks(chunk_size=64))
	# 200 rows / 64 = 4 chunks (last is partial 8 rows)
	assert len(chunks) == 4
	assert chunks[0].shape == (64, 4)
	assert chunks[-1].shape == (8, 4)
	# Reassemble
	reassembled = np.vstack(chunks)
	assert reassembled.shape == (200, 4)
	# Bit-exact vs original packed bytes
	assert np.array_equal(reassembled, packed)


def test_memmap_row_subset_materializes(tmp_path):
	"""row_subset materializes selected rows into an InMemoryEncoded (off-memmap)."""
	bools = _make_bool_matrix(300, 24)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder='little')
	tmp_file = tmp_path / "subset.bin"
	packed.tofile(str(tmp_file))

	enc = MemmapEncoded(tmp_file, n_rows=300, total_bits=24, mode="r")
	indices = np.array([0, 5, 12, 50, 299])
	sub = enc.row_subset(indices)

	assert isinstance(sub, InMemoryEncoded)
	assert sub.n_rows == len(indices)
	assert sub.total_bits == 24

	# Check the subset values match the original
	sub_bool = sub.to_numpy_bool()
	for new_idx, orig_idx in enumerate(indices):
		assert np.array_equal(sub_bool[new_idx], bools[orig_idx]), f"subset idx {new_idx} (orig {orig_idx}) mismatch"


def test_memmap_as_packed_uint8_zero_copy(tmp_path):
	"""as_packed_uint8 returns the memmap view, not a copy."""
	bools = _make_bool_matrix(50, 16)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder='little')
	tmp_file = tmp_path / "view.bin"
	packed.tofile(str(tmp_file))

	enc = MemmapEncoded(tmp_file, n_rows=50, total_bits=16, mode="r")
	view = enc.as_packed_uint8()
	assert isinstance(view, np.memmap), "expected memmap view, got copy"
	assert view.shape == (50, 2)


def test_write_packed_to_memmap_helper(tmp_path):
	"""write_packed_to_memmap writes + opens the file, returns a MemmapEncoded."""
	bools = _make_bool_matrix(150, 40)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder='little')

	enc = write_packed_to_memmap(packed, total_bits=40, storage_dir=tmp_path, suffix=".tmp")
	assert isinstance(enc, MemmapEncoded)
	assert enc.n_rows == 150
	assert enc.total_bits == 40
	assert enc.path.suffix == ".tmp"
	assert enc.path.exists()
	# Bit-exact contents
	enc_bool = enc.to_numpy_bool()
	assert np.array_equal(enc_bool, bools)


def test_memmap_tmp_cleanup_on_del(tmp_path):
	"""__del__ unlinks .tmp files but preserves non-.tmp paths."""
	bools = _make_bool_matrix(20, 8)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder='little')

	# .tmp → should be cleaned
	tmp_enc = write_packed_to_memmap(packed, total_bits=8, storage_dir=tmp_path, suffix=".tmp")
	tmp_path_str = str(tmp_enc.path)
	del tmp_enc
	import gc
	gc.collect()  # force __del__ if reference cycle delayed it
	assert not Path(tmp_path_str).exists(), f".tmp file should be cleaned: {tmp_path_str}"

	# .keep → should be preserved
	keep_enc = write_packed_to_memmap(packed, total_bits=8, storage_dir=tmp_path, suffix=".keep")
	keep_path_str = str(keep_enc.path)
	del keep_enc
	gc.collect()
	assert Path(keep_path_str).exists(), f".keep file should be preserved: {keep_path_str}"
	# Clean up the .keep file ourselves for hygiene
	Path(keep_path_str).unlink()


def test_encode_features_memmap_e2e(tmp_path):
	"""encode_features(encoded_storage='memmap') produces a MemmapEncoded with
	bit-exact identical contents to the in-memory path."""
	import pandas as pd
	from wnn.ids.dataset import encode_features
	from wnn.ids.encoder import ThermometerType

	# Tiny tabular dataset: 100 rows, mix of numeric + binary + categorical
	rng = np.random.default_rng(42)
	df = pd.DataFrame({
		"num_a": rng.normal(size=100),
		"num_b": rng.uniform(size=100),
		"bin_flag": rng.integers(0, 2, size=100),
		"cat_proto": ["tcp", "udp", "icmp"] * 33 + ["tcp"],
	})
	df_test = df.iloc[:20].copy()
	df_train = df.iloc[20:].copy()

	# In-memory path
	X_train_mem, X_test_mem, _, _, _ = encode_features(
		df_train, df_test, ["num_a", "num_b", "bin_flag", "cat_proto"], [],
		n_bits=4, encoded_storage="memory",
	)
	# Memmap path
	X_train_mmap, X_test_mmap, _, _, _ = encode_features(
		df_train, df_test, ["num_a", "num_b", "bin_flag", "cat_proto"], [],
		n_bits=4, encoded_storage="memmap", storage_dir=tmp_path,
	)

	# Bit-exact equivalence
	assert X_train_mem.total_bits == X_train_mmap.total_bits
	assert X_train_mem.n_rows == X_train_mmap.n_rows
	assert np.array_equal(X_train_mem.as_packed_uint8(), X_train_mmap.as_packed_uint8()), \
		"memmap path produced different bytes than in-memory path"
	assert np.array_equal(X_test_mem.as_packed_uint8(), X_test_mmap.as_packed_uint8()), \
		"memmap path produced different bytes for X_test"

	# MemmapEncoded files should be on disk under tmp_path
	assert X_train_mmap.path.exists()
	assert X_test_mmap.path.exists()
	# Cleanup happens on __del__ since suffix is .tmp
	assert X_train_mmap.path.suffix == ".tmp"


def _make_streaming_factory(total_rows: int, total_bits: int, chunk_size: int, seed: int = 42):
	"""Build a re-iterable factory that yields (packed_chunk, labels_chunk) tuples.

	Used in StreamingEncoded tests to simulate an HF streaming source
	without depending on the `datasets` library or network.
	"""
	rng = np.random.default_rng(seed)
	# Generate the full dataset once, then yield slabs deterministically.
	# (In real streaming this would be a re-fetch from the source.)
	bools_full = rng.integers(0, 2, size=(total_rows, total_bits), dtype=np.uint8).astype(bool)
	packed_full = np.packbits(bools_full.astype(np.uint8), axis=1, bitorder="little")
	labels_full = rng.integers(0, 2, size=total_rows, dtype=np.int64)

	def factory():
		for start in range(0, total_rows, chunk_size):
			end = min(start + chunk_size, total_rows)
			yield (packed_full[start:end], labels_full[start:end])

	return factory, packed_full, labels_full


def test_streaming_iter_chunks_yields_tuples():
	"""iter_chunks yields (packed_chunk, labels_chunk) tuples with correct shapes."""
	factory, packed_full, labels_full = _make_streaming_factory(total_rows=100, total_bits=24, chunk_size=30)
	se = StreamingEncoded(factory, n_rows=100, total_bits=24)

	chunks = list(se.iter_chunks())
	# 100 rows / 30 = 4 chunks (30, 30, 30, 10)
	assert len(chunks) == 4
	for packed_chunk, labels_chunk in chunks:
		assert isinstance(packed_chunk, np.ndarray)
		assert packed_chunk.dtype == np.uint8
		assert isinstance(labels_chunk, np.ndarray)
		assert labels_chunk.dtype == np.int64
		# bytes_per_row should be 3 (24 bits / 8)
		assert packed_chunk.shape[1] == 3

	# Last chunk is partial (10 rows)
	assert chunks[-1][0].shape[0] == 10
	assert chunks[-1][1].shape[0] == 10


def test_streaming_re_iterable():
	"""Each call to iter_chunks returns a fresh iterator that yields the same data."""
	factory, packed_full, _ = _make_streaming_factory(total_rows=50, total_bits=16, chunk_size=20)
	se = StreamingEncoded(factory, n_rows=50, total_bits=16)

	# Two passes — second pass must yield identical data (re-iterable contract)
	chunks_1 = [(p.copy(), l.copy()) for p, l in se.iter_chunks()]
	chunks_2 = [(p.copy(), l.copy()) for p, l in se.iter_chunks()]

	assert len(chunks_1) == len(chunks_2)
	for (p1, l1), (p2, l2) in zip(chunks_1, chunks_2):
		assert np.array_equal(p1, p2)
		assert np.array_equal(l1, l2)


def test_streaming_reassembles_to_source():
	"""Concatenating chunks reconstructs the underlying full matrix."""
	factory, packed_full, labels_full = _make_streaming_factory(total_rows=200, total_bits=40, chunk_size=64)
	se = StreamingEncoded(factory, n_rows=200, total_bits=40)

	all_packed = []
	all_labels = []
	for p, l in se.iter_chunks():
		all_packed.append(p)
		all_labels.append(l)

	reassembled_p = np.vstack(all_packed)
	reassembled_l = np.concatenate(all_labels)
	assert np.array_equal(reassembled_p, packed_full)
	assert np.array_equal(reassembled_l, labels_full)


def test_streaming_metadata_properties():
	"""n_rows, total_bits, shape, bytes_per_row are reported correctly."""
	factory, _, _ = _make_streaming_factory(total_rows=1000, total_bits=96, chunk_size=128)
	se = StreamingEncoded(factory, n_rows=1000, total_bits=96)
	assert se.n_rows == 1000
	assert se.total_bits == 96
	assert se.shape == (1000, 96)
	assert se.bytes_per_row == 12  # ceil(96/8)
	assert len(se) == 1000


def test_streaming_random_access_methods_raise():
	"""__getitem__, as_packed_uint8, to_numpy_bool, row_subset raise NotImplementedError."""
	factory, _, _ = _make_streaming_factory(total_rows=10, total_bits=8, chunk_size=4)
	se = StreamingEncoded(factory, n_rows=10, total_bits=8)

	for op in (
		lambda: se[0],
		lambda: se[5:8],
		lambda: se[np.array([1, 3, 5])],
		lambda: se.as_packed_uint8(),
		lambda: se.to_numpy_bool(),
		lambda: se.row_subset(np.array([0, 1, 2])),
	):
		try:
			op()
		except NotImplementedError:
			continue
		raise AssertionError(f"streaming op should raise NotImplementedError: {op}")


def test_memmap_prefetch_touch(tmp_path):
	"""F10: prefetch('touch') reads every page via numpy.sum side-effect.

	Doesn't crash; doesn't change observable data; doesn't take excessive
	wall-clock time on a small fixture.
	"""
	import time
	bools = _make_bool_matrix(5000, 32)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder="little")
	tmp_file = tmp_path / "prefetch_touch.bin"
	packed.tofile(str(tmp_file))

	enc = MemmapEncoded(tmp_file, n_rows=5000, total_bits=32, mode="r")
	t0 = time.time()
	enc.prefetch(mode="touch")
	elapsed = time.time() - t0
	# Sanity bound: 5000 × 4 bytes = 20 KB; should be milliseconds.
	assert elapsed < 5.0, f"prefetch took too long: {elapsed:.2f}s"

	# Data still correct after prefetch (no mutation)
	for i in [0, 17, 4999]:
		row_bool = np.unpackbits(enc[i], bitorder="little")[:32].astype(bool)
		assert np.array_equal(row_bool, bools[i])


def test_memmap_prefetch_willneed_does_not_crash(tmp_path):
	"""F10: prefetch('willneed') is a best-effort hint; should never crash."""
	bools = _make_bool_matrix(100, 16)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder="little")
	tmp_file = tmp_path / "prefetch_willneed.bin"
	packed.tofile(str(tmp_file))

	enc = MemmapEncoded(tmp_file, n_rows=100, total_bits=16, mode="r")
	# Should be a no-op or quiet success on macOS
	enc.prefetch(mode="willneed")


def test_memmap_prefetch_none_is_noop(tmp_path):
	"""F10: prefetch('none') returns immediately without touching anything."""
	import time
	bools = _make_bool_matrix(1000, 24)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder="little")
	tmp_file = tmp_path / "prefetch_none.bin"
	packed.tofile(str(tmp_file))

	enc = MemmapEncoded(tmp_file, n_rows=1000, total_bits=24, mode="r")
	t0 = time.time()
	enc.prefetch(mode="none")
	elapsed = time.time() - t0
	# Must be ~0 sec (no read whatsoever)
	assert elapsed < 0.05, f"prefetch('none') should be instant, took {elapsed:.4f}s"


def test_memmap_prefetch_invalid_mode_raises(tmp_path):
	"""F10: invalid mode raises ValueError."""
	bools = _make_bool_matrix(10, 8)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder="little")
	tmp_file = tmp_path / "prefetch_bad.bin"
	packed.tofile(str(tmp_file))

	enc = MemmapEncoded(tmp_file, n_rows=10, total_bits=8, mode="r")
	try:
		enc.prefetch(mode="aggressive")
	except ValueError as e:
		assert "must be" in str(e)
		return
	raise AssertionError("invalid prefetch mode should raise ValueError")


def test_write_packed_to_memmap_with_prefetch(tmp_path):
	"""F10: write_packed_to_memmap accepts prefetch param and applies it."""
	bools = _make_bool_matrix(500, 32)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder="little")

	# With explicit prefetch — should not crash, contents preserved
	enc = write_packed_to_memmap(
		packed, total_bits=32, storage_dir=tmp_path,
		suffix=".tmp", prefetch="touch",
	)
	assert isinstance(enc, MemmapEncoded)
	enc_bool = enc.to_numpy_bool()
	assert np.array_equal(enc_bool, bools)


def test_memmap_reuse_after_close(tmp_path):
	"""A .keep file written by one MemmapEncoded can be reopened by another."""
	bools = _make_bool_matrix(80, 12)
	packed = np.packbits(bools.astype(np.uint8), axis=1, bitorder='little')

	enc1 = write_packed_to_memmap(packed, total_bits=12, storage_dir=tmp_path, suffix=".keep")
	saved_path = enc1.path
	del enc1
	import gc
	gc.collect()

	# Reopen — bit-exact same contents
	enc2 = MemmapEncoded(saved_path, n_rows=80, total_bits=12, mode="r")
	enc2_bool = enc2.to_numpy_bool()
	assert np.array_equal(enc2_bool, bools)
	# Cleanup
	del enc2
	gc.collect()
	if saved_path.exists():
		saved_path.unlink()


if __name__ == "__main__":
	import tempfile
	import traceback

	tests = [
		test_inmemory_packed_roundtrip,
	]
	# StreamingEncoded tests are fixture-free
	tests.extend([
		test_streaming_iter_chunks_yields_tuples,
		test_streaming_re_iterable,
		test_streaming_reassembles_to_source,
		test_streaming_metadata_properties,
		test_streaming_random_access_methods_raise,
	])
	# Tests needing tmp_path fixture
	tmp_tests = [
		test_memmap_packed_roundtrip,
		test_memmap_iter_chunks,
		test_memmap_row_subset_materializes,
		test_memmap_as_packed_uint8_zero_copy,
		test_write_packed_to_memmap_helper,
		test_memmap_tmp_cleanup_on_del,
		test_memmap_prefetch_touch,
		test_memmap_prefetch_willneed_does_not_crash,
		test_memmap_prefetch_none_is_noop,
		test_memmap_prefetch_invalid_mode_raises,
		test_write_packed_to_memmap_with_prefetch,
		test_memmap_reuse_after_close,
		test_encode_features_memmap_e2e,
	]
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
	for t in tmp_tests:
		with tempfile.TemporaryDirectory() as td:
			try:
				t(Path(td))
				print(f"  ✓ {t.__name__}")
				passed += 1
			except Exception as e:
				print(f"  ✗ {t.__name__}: {e}")
				traceback.print_exc()
				failed += 1
	print(f"\n{passed} passed, {failed} failed out of {len(tests) + len(tmp_tests)}")
	sys.exit(0 if failed == 0 else 1)
