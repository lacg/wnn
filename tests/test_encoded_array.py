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
	# Tests needing tmp_path fixture
	tmp_tests = [
		test_memmap_packed_roundtrip,
		test_memmap_iter_chunks,
		test_memmap_row_subset_materializes,
		test_memmap_as_packed_uint8_zero_copy,
		test_write_packed_to_memmap_helper,
		test_memmap_tmp_cleanup_on_del,
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
