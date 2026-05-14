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

import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
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


class MemmapEncoded(LazyEncodedArray):
	"""Disk-backed packed-bytes encoded matrix (np.memmap).

	Use when the in-RAM packed buffer is too tight against working memory.
	For 96b × 46-feature × 46M CIC-IoT-2023, the packed bytes are ~21 GB —
	fits in 64 GB but tight when worker + dashboard + frontend share RAM.
	Memmap lets the OS page hot rows into RAM and evict cold pages under
	pressure; sequential row iteration (typical for WNN training/eval) is
	cache-friendly.

	Storage layout matches InMemoryEncoded's packed form: a single
	(n_rows × bytes_per_row) uint8 file, LSB-first within each byte.
	Files written by `np.packbits(bool_matrix, axis=1, bitorder='little')`
	round-trip exactly.

	Path lifecycle:
	- `path.suffix == '.tmp'`: deleted on __del__ (single-flow lifetime).
	- Any other suffix (e.g. `.keep`, `.bin`): kept across runs for reuse.

	Read access:
	- `__getitem__` and `iter_chunks` page rows in via OS memmap.
	- `as_packed_uint8()` returns the memmap view (no copy).
	- `row_subset(indices)` materializes an InMemoryEncoded (small enough
	  to live in RAM; falls out of the memmap working set on next sweep).
	- `to_numpy_bool()` materializes a full bool ndarray — avoid for large
	  matrices (defeats the memmap memory bound).
	"""

	def __init__(self, path: Path | str, n_rows: int, total_bits: int, mode: str = "r"):
		"""Open or create a memmap-backed encoded matrix.

		Args:
		    path: file path (existing in mode='r'/'r+', created in 'w+').
		    n_rows: number of logical rows.
		    total_bits: bits per row (used for stride math and bool decoding).
		    mode: numpy memmap mode — 'r' read-only, 'r+' read-write existing,
		        'w+' read-write new (truncates / creates).
		"""
		self._path = Path(path)
		self._total_bits = total_bits
		self._n_rows = n_rows
		bytes_per_row = (total_bits + 7) // 8
		shape = (n_rows, bytes_per_row) if n_rows > 0 else (0, bytes_per_row)
		self._data = np.memmap(str(self._path), dtype=np.uint8, mode=mode, shape=shape)
		self._mode = mode

	def __getitem__(self, idx) -> np.ndarray:
		# Memmap supports int/slice/fancy indexing the same as ndarray.
		# Fancy indexing forces materialization of the selected rows.
		return self._data[idx]

	def iter_chunks(self, chunk_size: int) -> Iterator[np.ndarray]:
		for start in range(0, self._n_rows, chunk_size):
			yield self._data[start : start + chunk_size]

	def as_packed_uint8(self) -> np.ndarray:
		# Memmap view; consumer sees a real np.ndarray that pages on access.
		# Returning as-is is the whole point — no in-RAM copy of the full matrix.
		return self._data

	def row_subset(self, indices) -> "InMemoryEncoded":
		"""Materialize a row subset as a new in-memory packed array.

		The subset (typically a fold or split, ≪ full size) is read out
		of memmap into RAM. Returning InMemoryEncoded (not MemmapEncoded)
		keeps the working subset hot.
		"""
		subset = np.ascontiguousarray(self._data[indices])
		return InMemoryEncoded(subset, total_bits=self._total_bits)

	@property
	def path(self) -> Path:
		return self._path

	def prefetch(self, mode: str = "touch") -> None:
		"""Warm the OS page cache for this memmap (Phase F10).

		Backed-by-disk memmaps incur page faults on first read of every
		page. For K-fold flows where the same data is re-read many times
		per genome × per generation, the cold-cache cost of the first
		full pass dominates wall-clock time. Pre-faulting all pages once
		amortizes this across all subsequent reads — typically 30-100×
		speedup on the first K-fold partition transition.

		Modes:
		- "touch" (default): synchronous read of every page. ~500ms-2s
		  for 0.5-2 GB packed matrices on SSD. Best for K-fold flows.
		- "willneed": madvise(MADV_WILLNEED) — async hint to the kernel.
		  Cheap (microseconds) but best-effort; OS decides when/whether
		  to actually prefetch. Use when you're not sure the warmup
		  cost is worth paying upfront.
		- "none": no-op. Useful for tests or when memmap was just
		  written and pages are already hot.
		"""
		import mmap as mm_mod

		if mode == "none":
			return
		underlying = getattr(self._data, "_mmap", None)
		if underlying is None:
			return  # not actually backed by mmap; nothing to prefetch

		if mode == "willneed":
			try:
				underlying.madvise(mm_mod.MADV_WILLNEED)
			except (AttributeError, OSError):
				pass  # best-effort; don't fail if madvise isn't available
			return

		if mode == "touch":
			# Numpy's sum() walks the entire buffer in a fused C loop, which
			# triggers a page fault on each uncached page. The actual integer
			# return value is discarded — we want the side effect of every
			# page being faulted into the OS unified buffer cache.
			# Cost: SSD read bandwidth (~600 MB/s on M4 Max unified memory)
			# → ~1s for 600 MB, ~3.5s for 2 GB. Worth it for K-fold flows.
			try:
				_ = int(self._data.sum())
			except Exception:
				pass  # best-effort; correctness doesn't depend on this
			return

		raise ValueError(f"prefetch mode must be 'touch' | 'willneed' | 'none', got {mode!r}")

	def __del__(self):
		# Auto-clean .tmp files; preserve anything else for reuse.
		try:
			# Drop the memmap before unlinking so the file isn't held open.
			data = getattr(self, "_data", None)
			if data is not None:
				mm = getattr(data, "_mmap", None)
				if mm is not None:
					mm.close()
			path = getattr(self, "_path", None)
			if path is not None and path.suffix == ".tmp" and path.exists():
				path.unlink()
		except Exception:
			# Best-effort cleanup; never raise from __del__.
			pass


def write_packed_to_memmap(
	packed: np.ndarray,
	total_bits: int,
	storage_dir: Path | str | None = None,
	suffix: str = ".tmp",
	prefetch: str = "none",
) -> MemmapEncoded:
	"""Write a packed numpy array to a memmap file and return a MemmapEncoded.

	The packed array is consumed (NOT retained by reference) — caller should
	`del packed` afterwards to free the in-RAM copy. Encoder use case:
	build the packed matrix in RAM, hand off to this writer, drop the RAM
	copy, return the MemmapEncoded.

	Args:
	    packed: uint8 array of shape (n_rows, ceil(total_bits/8)).
	    total_bits: logical bit width (must match the packed layout).
	    storage_dir: where to write the file (default ~/.cache/wnn/encoded).
	    suffix: file suffix (".tmp" → auto-deleted on __del__,
	        ".keep" or others → preserved for reuse across runs).
	"""
	if packed.dtype != np.uint8:
		raise TypeError(f"packed must be uint8, got {packed.dtype}")
	expected_cols = (total_bits + 7) // 8
	if packed.ndim != 2 or packed.shape[1] != expected_cols:
		raise ValueError(
			f"packed shape {packed.shape} doesn't match expected (n_rows, {expected_cols}) "
			f"for total_bits={total_bits}"
		)

	if storage_dir is None:
		storage_dir = Path.home() / ".cache" / "wnn" / "encoded"
	storage_dir = Path(storage_dir)
	storage_dir.mkdir(parents=True, exist_ok=True)
	tmp_path = storage_dir / f"x_{uuid.uuid4().hex[:8]}{suffix}"

	# Write packed bytes to disk; tofile is much faster than np.save for raw uint8.
	packed.tofile(str(tmp_path))

	# Open the file read-only as a memmap. We deliberately don't keep the
	# in-RAM packed array alive — the memmap is the single source of truth.
	return MemmapEncoded(tmp_path, n_rows=packed.shape[0], total_bits=total_bits, mode="r")


class StreamingEncoded(LazyEncodedArray):
	"""Re-iterable streaming source for encoded IDS data (Phase F).

	UNLIKE InMemoryEncoded and MemmapEncoded, this class intentionally
	violates parts of the LazyEncodedArray contract — streaming sources
	cannot serve random row access without materializing the entire matrix,
	which is the whole point of streaming. Bounded memory regardless of N.

	Use case: datasets too large to fit on disk (10B+ rows), or where
	the data lives on a remote object store and we want to evaluate
	WNN genomes without ever copying it locally.

	## Contract

	Supported:
	- `iter_chunks()` — yields `(packed_chunk, labels_chunk)` tuples from
	  the underlying factory. Each call to `iter_chunks` returns a FRESH
	  iterator (the factory is invoked again), so streaming consumers can
	  make multiple passes (e.g. train then score). The factory is
	  responsible for whatever caching/re-fetching it wants to do.
	- `n_rows`, `total_bits`, `shape`, `bytes_per_row` — known up-front
	  from source metadata.

	Raises `NotImplementedError`:
	- `__getitem__`, `as_packed_uint8`, `to_numpy_bool`, `row_subset` —
	  these would require full materialization. Consumers must instead
	  use `iter_chunks` and adopt a streaming evaluation model.

	## Tuple yield shape

	The factory yields `(packed_chunk, labels_chunk)` tuples — a
	deliberate departure from the LazyEncodedArray.iter_chunks signature
	(which yields packed only). The bundling is load-bearing for streaming:
	the consumer (Rust IDSCacheStreaming) needs feature rows AND labels
	together to do per-chunk training/scoring. Splitting them would
	force the caller to coordinate two independent stream iterators —
	error-prone and synchronization-fragile.

	## n_rows must be known

	HuggingFace's `streaming=True` datasets typically expose `num_rows`
	via the dataset info. Callers must pass this through so that
	Rust-side metric accumulators can size their buffers correctly.
	"""

	def __init__(self, iter_factory, n_rows: int, total_bits: int):
		"""
		Args:
		    iter_factory: callable () → Iterator yielding
		        (packed_chunk_uint8, labels_chunk_int64) tuples.
		        Each invocation returns a fresh iterator (re-iterable
		        contract). The factory is the source of truth for chunk
		        size; this class doesn't dictate it.
		    n_rows: total rows the factory will yield across one full
		        iteration. Used for buffer sizing and progress reporting.
		    total_bits: logical bit width per row (constant across chunks).
		"""
		self._iter_factory = iter_factory
		self._n_rows = n_rows
		self._total_bits = total_bits

	def iter_chunks(self, chunk_size=None):
		"""Yield (packed_chunk, labels_chunk) tuples from a fresh factory iterator.

		`chunk_size` is accepted for API compatibility with LazyEncodedArray
		but ignored — the underlying factory dictates chunk sizes. To
		change chunk size, construct a new StreamingEncoded with a different
		factory.
		"""
		return self._iter_factory()

	def __getitem__(self, idx):
		raise NotImplementedError(
			"StreamingEncoded does not support random row access — use iter_chunks()"
		)

	def as_packed_uint8(self):
		raise NotImplementedError(
			"StreamingEncoded cannot materialize the full packed matrix — "
			"streaming consumers must use iter_chunks(). If you need a packed "
			"snapshot, materialize via write_packed_to_memmap() in a one-time "
			"pre-pass."
		)

	def to_numpy_bool(self):
		raise NotImplementedError(
			"StreamingEncoded cannot materialize a bool matrix — use iter_chunks()"
		)

	def row_subset(self, indices):
		raise NotImplementedError(
			"StreamingEncoded cannot serve row_subset — selecting arbitrary "
			"rows would require materializing the source. Use the streaming "
			"K-fold path (IDSEvaluator.get_fold_indices on a pre-streamed "
			"permutation) instead."
		)


def write_stream_to_memmap(
	streaming: "StreamingEncoded",
	storage_dir: "Path | str | None" = None,
	suffix: str = ".tmp",
	prefetch: str = "none",
) -> "tuple[MemmapEncoded, np.ndarray]":
	"""Materialize a StreamingEncoded into a MemmapEncoded by streaming chunks
	through to disk + collecting labels in RAM.

	Used by the F7 auto-detect path: when a streaming dataset is small enough
	to fit on disk + K-fold benefits more from random row access than from
	bounded memory, we drain the stream once into a memmap file and switch
	to the in-memory evaluation path for the rest of the flow.

	Args:
	    streaming: source. Must have known `n_rows` and `total_bits`.
	    storage_dir: where to write the memmap (default ~/.cache/wnn/encoded).
	    suffix: ".tmp" (auto-clean on __del__) or ".keep" (preserve).

	Returns:
	    (memmap_encoded, labels) — the memmap-backed packed matrix and the
	    np.int64 labels array of length `n_rows`. Labels are bundled with
	    chunks in StreamingEncoded; we materialize them here so the caller
	    has a complete IDSDataset-shaped output.
	"""
	if storage_dir is None:
		storage_dir = Path.home() / ".cache" / "wnn" / "encoded"
	storage_dir = Path(storage_dir)
	storage_dir.mkdir(parents=True, exist_ok=True)
	tmp_path = storage_dir / f"x_{uuid.uuid4().hex[:8]}{suffix}"

	n_rows = streaming.n_rows
	total_bits = streaming.total_bits
	bytes_per_row = (total_bits + 7) // 8

	# Pre-allocate the memmap file at full size, then fill row-by-row from
	# the stream. Writing chunk-by-chunk via np.memmap mode='w+' is a single
	# os.write per chunk — much faster than per-row tofile() calls.
	memmap = np.memmap(
		str(tmp_path), dtype=np.uint8, mode="w+",
		shape=(n_rows, bytes_per_row),
	)
	labels = np.empty(n_rows, dtype=np.int64)
	pos = 0
	for packed_chunk, labels_chunk in streaming.iter_chunks():
		n = packed_chunk.shape[0]
		memmap[pos:pos + n] = packed_chunk
		labels[pos:pos + n] = labels_chunk
		pos += n
	memmap.flush()
	assert pos == n_rows, f"stream yielded {pos} rows, expected {n_rows}"

	# Close the w+ memmap and reopen read-only to match MemmapEncoded's
	# normal lifecycle (write-once at materialization, read-only thereafter).
	del memmap
	result = MemmapEncoded(tmp_path, n_rows=n_rows, total_bits=total_bits, mode="r")
	if prefetch != "none":
		result.prefetch(prefetch)
	return result, labels


__all__ = [
	"LazyEncodedArray",
	"InMemoryEncoded",
	"MemmapEncoded",
	"StreamingEncoded",
	"write_packed_to_memmap",
	"write_stream_to_memmap",
]
