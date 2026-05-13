# Encoded array contract

This document defines the canonical contract that all consumers of encoded
IDS feature data must respect. Established through the Phase 1–5 memory
refactor (May 2026). Locked across `main`.

## Layers

| Layer | Class | Storage | Lifetime |
|---|---|---|---|
| Abstraction | `wnn.ids.encoded_array.LazyEncodedArray` (ABC) | — | always |
| In-memory | `InMemoryEncoded(packed_uint8, total_bits)` | Python heap | flow |
| Disk-backed | `MemmapEncoded(path, n_rows, total_bits)` | `np.memmap` file | flow or persistent (`.keep` suffix) |
| Streaming | `StreamingEncoded(...)` | chunk iterator | *(post-paper; Option F)* |

All implementations expose the same surface. Consumers MUST NOT branch on
the concrete subclass.

## Required API surface

```python
class LazyEncodedArray(ABC):
    n_rows: int        # property
    total_bits: int    # property — logical bit width per row
    shape: tuple       # property — (n_rows, total_bits)
    bytes_per_row: int # property — ceil(total_bits / 8)

    def __getitem__(idx) -> np.ndarray
    def __len__() -> int
    def iter_chunks(chunk_size: int) -> Iterator[np.ndarray]
    def as_packed_uint8() -> np.ndarray
    def to_numpy_bool() -> np.ndarray
    def row_subset(indices) -> LazyEncodedArray
```

### Bit ordering (locked)

- Logical bit `j` of row `i` lives at:
  - byte `(j >> 3)` within the row's `bytes_per_row` slab.
  - bit position `(j & 7)` within that byte (LSB-first).
- This matches `np.packbits(bool_matrix, axis=1, bitorder='little')`
  and the Rust `PackedBits` struct (`packed_bits.rs`).
- **Do not change this.** The full pipeline (Python encoder → numpy →
  PyO3 → Rust `PackedBits` → `compute_address_packed_bytes`) depends on
  this ordering being preserved at every layer.

### `as_packed_uint8()` semantics

- Returns a 2D `np.ndarray` of shape `(n_rows, bytes_per_row)`, dtype
  `uint8`, contents = `np.packbits` LSB-first form.
- For `InMemoryEncoded`: zero-copy view of the underlying buffer when
  the buffer is already packed; one-time `np.packbits` call when wrapping
  a legacy bool buffer.
- For `MemmapEncoded`: returns the `np.memmap` directly (zero-copy). The
  OS pages rows in on access.
- The Rust boundary (`IDSCacheWrapper.new_from_numpy`,
  `IDSCacheBuilderWrapper.add_train_chunk`) consumes this directly via
  `PackedBits::from_packed_bytes`.

### `iter_chunks(chunk_size)` semantics

- Yields successive row slabs of size `chunk_size` (last may be smaller).
- For Phase 2-onward, callers don't typically use this — the single-chunk
  path through `as_packed_uint8()` is faster when memory permits.
- Option F's streaming path will iterate via `iter_chunks` and feed each
  chunk into `IDSCacheBuilderWrapper.add_*_chunk`, so the full packed
  matrix is never materialized.

### `row_subset(indices) -> LazyEncodedArray` semantics

- Materializes the selected rows. Returns a new `LazyEncodedArray`
  (typically `InMemoryEncoded` even when called on a `MemmapEncoded` —
  fold subsets are small enough to live in RAM hot).
- Used by K-fold and 80/20 split paths.

## Encoder output

`ThermometerEncoder.transform(df) → (packed_uint8, total_bits)`. The
returned tuple is the canonical packed boundary between the Python
encoder and the rest of the pipeline.

`ThermometerEncoder.iter_chunks(df, chunk_size)` yields the same tuples
slab-by-slab — used by Option F.

## Rust boundary

Three PyO3 entry points consume packed IDS data:

| Entry | Use case | Input form |
|---|---|---|
| `IDSCacheWrapper.new_from_numpy` | One-shot, full matrix in RAM | flat uint8 array (n_rows × bytes_per_row) |
| `IDSCacheBuilderWrapper.add_*_chunk` | Chunked / streaming | flat uint8 chunk per call |
| `IDSCacheWrapper.new` (legacy) | Python-list ctor for tests | Python `list[bool]` (packed internally) |

All three converge on the same `IDSCache::new` finalization path. The
builder's `finalize()` is byte-exact equivalent to a one-shot
`new_from_numpy` call when fed the same data in one chunk.

## K-fold permutation (Phase 5)

`IDSEvaluator._kfold_perm` is a deterministic permutation of training row
indices, seeded by `seed + 7777`. `get_fold_indices(fold_idx)` returns
contiguous slabs of this permutation as `val_idx` and the rest as
`train_idx`. The contiguous-slab property is load-bearing for Option F:
when training rows come from a memmap or stream, each fold maps to a
sequential range of disk pages / stream offsets — no scattered I/O.

## What NOT to do

- Do not pass `Vec<bool>` or `np.ndarray(dtype=bool)` across the
  Python↔Rust boundary on the IDS hot path. Always go through
  `as_packed_uint8()` or `iter_chunks()`.
- Do not change bit ordering. `bitorder='little'` is locked.
- Do not branch on `isinstance(x, MemmapEncoded)` in consumers. If you
  need disk-vs-RAM behavior, ask `x.iter_chunks(...)` and let the
  implementation decide.
- Do not write to a `MemmapEncoded` whose path is in a `.tmp` suffix —
  the file is owned by the encoder's single-flow lifetime and will be
  unlinked on `__del__`.

## Phase status

| Phase | Status |
|---|---|
| 1 — `LazyEncodedArray` abstraction | merged (`7acf8bd0`) |
| 2 — Full PackedBits migration | merged (`b082a834`) |
| 3 — Inline DataFrame release | merged (`22356043`) |
| 4 — `MemmapEncoded` + plumbing | merged (`0e92da0f`, `68069e3a`) |
| 5 — F-prep (iter_chunks, K-fold perm, builder) | merged (`fc1cf525`, `388c28d8`, `0af5c1d3`) |
| 6 — 96b × 46 × 46M integration smoke | pending |
| F — `StreamingEncoded` | post-paper |
