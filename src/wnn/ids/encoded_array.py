"""Abstract interface for the encoded feature matrix (X_train / X_test / X_val).

The encoded matrix is the output of ThermometerEncoder — one row per dataset
sample, total_bits columns. Today it's stored as numpy bool (1 byte per bit).
This module introduces a thin abstraction layer so we can swap representations
without touching downstream consumers (IDSEvaluator, Rust accelerator boundary,
etc.).

Implementations:
- InMemoryEncoded: numpy in-memory representation. Phase 1: holds the
  existing bool array as-is (1 byte per bit). Phase 2 will switch to
  bit-packed uint8 (1 byte per 8 bits → 8x memory reduction).
- MemmapEncoded:   np.memmap-backed disk storage (Phase 4 / Option D).
- StreamingEncoded: chunk iterator with no full materialization
                    (Phase 6 / Option F, post-paper).

All implementations expose the same API so consumers don't need to know how
the data is materialized.

Design choices documented inline below — Phase 1 is intentionally zero-behavior-
change: same numpy bool array, just wrapped in a class. Phase 2 will activate
the bit-packed path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class LazyEncodedArray(ABC):
	"""Abstract base for an encoded feature matrix.

	Logical shape is (n_rows, total_bits). The physical layout may be
	numpy bool, bit-packed uint8, memmap-backed, or a chunk iterator —
	consumers should not depend on the physical layout.

	Subclasses MUST implement:
	- __getitem__(idx)   : row-wise access (int / slice / array of indices)
	- __len__()          : equivalent to n_rows
	- iter_chunks(size)  : chunked iteration (Phase 5 / F-prep)
	- as_packed_uint8()  : contiguous bit-packed bytes for the Rust accelerator
	- to_numpy_bool()    : decoded bool view (legacy fallback for debugging)
	"""

	# Subclasses set these in __init__
	_n_rows: int
	_total_bits: int

	@abstractmethod
	def __getitem__(self, idx) -> np.ndarray:
		"""Return rows at idx.

		idx may be:
		- int (single row)              → 1D array of shape (total_bits,) or (bytes_per_row,)
		- slice                         → 2D array of shape (n_selected, ...)
		- 1D array of int indices       → 2D array of shape (n_selected, ...)   ★ K-fold uses this

		Return dtype is implementation-defined (bool today, uint8 after Phase 2).
		"""
		...

	def __len__(self) -> int:
		return self._n_rows

	@abstractmethod
	def iter_chunks(self, chunk_size: int) -> Iterator[np.ndarray]:
		"""Yield successive chunks of `chunk_size` rows.

		Used by streaming evaluation in Phase 6 (Option F). For Phases 1-5,
		this is equivalent to iterating over `[X[s:s+chunk_size] for s in ...]`.
		"""
		...

	@abstractmethod
	def as_packed_uint8(self) -> np.ndarray:
		"""Return the entire matrix as a contiguous bit-packed uint8 array.

		Shape: (n_rows, ceil(total_bits / 8))
		Used by the Rust accelerator boundary (single zero-copy handoff).

		Phase 1: this re-packs the bool array on every call (slow path,
		acceptable since accelerator handoff is once-per-flow).
		Phase 2: this becomes a no-op (data is already packed).
		"""
		...

	def to_numpy_bool(self) -> np.ndarray:
		"""Return a 2D bool view of the entire matrix.

		Default implementation: unpacks from `as_packed_uint8()`. Subclasses
		holding a bool array can override for zero-copy.

		Used only for debugging / legacy paths that haven't migrated to
		packed-uint8 yet.
		"""
		packed = self.as_packed_uint8()
		# np.unpackbits returns 1 byte per bit (bool-equivalent), little-endian
		# to match the encoder's bit ordering.
		unpacked = np.unpackbits(packed, axis=1, bitorder='little')
		# Trim trailing padding bits if total_bits isn't a multiple of 8
		return unpacked[:, : self._total_bits].astype(bool)

	@property
	def n_rows(self) -> int:
		return self._n_rows

	@property
	def total_bits(self) -> int:
		return self._total_bits

	@property
	def shape(self) -> tuple[int, int]:
		"""Logical shape (n_rows, total_bits). Note: physical layout may differ."""
		return (self._n_rows, self._total_bits)

	@property
	def bytes_per_row(self) -> int:
		"""Bit-packed bytes per row (ceil(total_bits / 8))."""
		return (self._total_bits + 7) // 8

	@abstractmethod
	def row_subset(self, indices) -> "LazyEncodedArray":
		"""Return a new LazyEncodedArray containing rows at `indices`.

		Used by K-fold (`X.row_subset(fold_train_idx)`) and dataset splits
		(`split_train_validation`). The returned wrapper preserves the
		LazyEncodedArray contract — callers don't need to know whether the
		subset is materialized as numpy, memmap, or a chunk iterator.

		Default implementation in subclasses: materialize the subset into
		the same physical layout. MemmapEncoded can override to return a
		lazy view if appropriate.
		"""
		...

	def __repr__(self) -> str:
		return f"{type(self).__name__}(n_rows={self._n_rows}, total_bits={self._total_bits})"


class InMemoryEncoded(LazyEncodedArray):
	"""In-memory implementation.

	Phase 1: holds a numpy bool array (legacy layout, 1 byte per bit).
	         Wraps the existing encoder output without changing memory footprint.
	Phase 2: switches to bit-packed uint8 storage (8x less RAM).

	Detection of which layout the underlying buffer uses:
	- dtype == bool          → 1 byte per bit (legacy / Phase 1)
	- dtype == uint8 + shape matches bytes_per_row → bit-packed (Phase 2+)
	"""

	def __init__(self, data: np.ndarray, total_bits: int):
		"""Wrap a numpy array as a LazyEncodedArray.

		Args:
			data: either a bool array (n_rows, total_bits) [Phase 1 — legacy bool]
			      or a uint8 packed array (n_rows, bytes_per_row) [Phase 2+].
			total_bits: number of valid bits per row (may be less than
			      data.shape[1] * 8 if packed has trailing padding).
		"""
		self._data = data
		self._total_bits = total_bits
		self._n_rows = data.shape[0]
		# Auto-detect layout
		if data.dtype == bool or data.dtype == np.bool_:
			self._packed = False
			expected_cols = total_bits
		elif data.dtype == np.uint8:
			self._packed = True
			expected_cols = (total_bits + 7) // 8
		else:
			raise TypeError(
				f"InMemoryEncoded expects bool or uint8, got {data.dtype}"
			)
		if data.shape[1] != expected_cols:
			raise ValueError(
				f"data.shape[1]={data.shape[1]} doesn't match expected "
				f"{expected_cols} for {'packed' if self._packed else 'bool'} "
				f"layout with total_bits={total_bits}"
			)

	def __getitem__(self, idx) -> np.ndarray:
		# Pass-through. Numpy's indexing handles int/slice/array uniformly.
		return self._data[idx]

	def iter_chunks(self, chunk_size: int) -> Iterator[np.ndarray]:
		for start in range(0, self._n_rows, chunk_size):
			yield self._data[start : start + chunk_size]

	def as_packed_uint8(self) -> np.ndarray:
		if self._packed:
			# Already packed, zero copy
			return self._data
		# Phase 1: pack on demand (one-time cost at accelerator handoff)
		# bitorder='little' = bit 0 in LSB of byte 0. Matches encoder's
		# natural ordering (first threshold is bit 0).
		return np.packbits(self._data, axis=1, bitorder='little')

	def to_numpy_bool(self) -> np.ndarray:
		if not self._packed:
			# Phase 1: already bool, zero copy
			return self._data
		# Phase 2+: unpack
		return super().to_numpy_bool()

	def row_subset(self, indices) -> "InMemoryEncoded":
		"""Materialize a row subset as a new InMemoryEncoded.

		Numpy fancy indexing copies the rows; we just re-wrap. Same physical
		layout (bool stays bool, packed stays packed).
		"""
		return InMemoryEncoded(self._data[indices], total_bits=self._total_bits)


__all__ = ["LazyEncodedArray", "InMemoryEncoded"]
