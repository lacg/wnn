"""Compact packing of large integer-column tables for the unified checkpoint store.

The bulk of a controller MEMORY genome is millions of ``(neuron, addr, value)``
cell triples; a 500n×34b IDS genome carries comparably large connection arrays.
Serialized as nested YAML lists these blow checkpoints up to GBs / tens of
millions of lines and take *minutes* to (de)serialize (measured 13/06/2026: a
57-genome controller population = 356 MB gz / 1.6 GB / 19.5 M YAML lines, ~3 min
to load even with libyaml).

``pack_int_columns`` turns a list of equal-width int rows into ONE base64 scalar
(stdlib ``array('q')`` = C int64 + base64) that YAML stores/parses as a single
node — O(rows) bytes instead of O(rows×cols) YAML nodes. No pickle
(refactor-proof, matches the store's "plain data" contract), no numpy dependency.

If any value falls outside signed int64, ``array('q')`` raises ``OverflowError``;
the CALLER catches it and keeps the verbose list form, so correctness never
depends on the range assumption (controller cell addresses are < 2^52; this is
pure safety for exotic bit-widths).

This is the SINGLE shared primitive: each genome's ``serialize``/``deserialize``
decides *which* of its fields are bulk integer columns and delegates the
encoding here — no per-strand copy of the packing logic.

Byte order is machine-native (this is a single-host research project); a
checkpoint is not portable across architectures of differing endianness, same
as before.
"""

import array
import base64
from typing import Any

# Tag stamped into the packed payload so ``is_packed`` / decoders can detect the
# new format and fall back to the legacy nested-list shape for old checkpoints.
_PACK_TAG = "i64cols"
_TYPECODE = "q"  # signed C int64


def _to_b64(flat: "array.array") -> str:
	"""C int64 buffer → base64 ASCII scalar (one YAML node)."""
	return base64.b64encode(flat.tobytes()).decode("ascii")


def _from_b64(b64: str) -> "array.array":
	"""base64 ASCII scalar → C int64 buffer."""
	flat = array.array(_TYPECODE)
	flat.frombytes(base64.b64decode(b64))
	return flat


def pack_int_columns(rows, ncols: int) -> dict:
	"""Pack an iterable of equal-width int rows into a compact JSON-able payload.

	Raises ``OverflowError`` if any value exceeds signed int64 — callers catch
	this and keep the verbose list form (so the range assumption is never load-
	bearing for correctness).
	"""
	flat = array.array(_TYPECODE)
	n = 0
	for row in rows:
		if len(row) != ncols:
			raise ValueError(f"row width {len(row)} != ncols {ncols}")
		flat.extend(int(v) for v in row)  # OverflowError here ⇒ caller falls back
		n += 1
	return {"_packed": _PACK_TAG, "n": n, "cols": int(ncols), "b64": _to_b64(flat)}


def is_packed(x: Any) -> bool:
	"""True iff ``x`` is a packed payload produced by ``pack_int_columns`` /
	``pack_int_array`` (both share the same tag and decode dispatch)."""
	return isinstance(x, dict) and x.get("_packed") == _PACK_TAG


def unpack_int_columns(payload: dict) -> list:
	"""Inverse of ``pack_int_columns`` → list of int tuples, in original order."""
	cols = int(payload["cols"])
	n = int(payload["n"])
	flat = _from_b64(payload["b64"])
	if len(flat) != n * cols:
		raise ValueError(f"packed length {len(flat)} != n*cols {n * cols}")
	return [tuple(flat[i:i + cols]) for i in range(0, len(flat), cols)]


def pack_int_array(values) -> dict:
	"""Pack a FLAT 1-D int sequence (e.g. a genome's flat connection array) — the
	``cols=1`` form, but ``unpack_int_array`` returns a flat list, not 1-tuples.

	Shares the same byte core / tag as ``pack_int_columns`` (so ``is_packed``
	covers both); the ``cols=1`` marker is how decode knows to flatten.
	Raises ``OverflowError`` outside int64 — callers fall back to the plain list.
	"""
	flat = array.array(_TYPECODE)
	flat.extend(int(v) for v in values)  # OverflowError ⇒ caller falls back
	return {"_packed": _PACK_TAG, "n": len(flat), "cols": 1, "b64": _to_b64(flat)}


def unpack_int_array(payload: dict) -> list:
	"""Inverse of ``pack_int_array`` → flat list of ints (no tuple churn)."""
	flat = _from_b64(payload["b64"])
	if len(flat) != int(payload["n"]):
		raise ValueError(f"packed length {len(flat)} != n {payload['n']}")
	return flat.tolist()
