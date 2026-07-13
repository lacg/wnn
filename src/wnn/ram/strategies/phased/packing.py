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

Values outside signed int64 (a memory-stage GA can grow state_bits past 63 —
the 12/07/2026 pid@31337004 run died packing sb=65 addresses) are handled by a
second fixed-width format: 16 bytes/value little-endian signed ("i128"), still
ONE base64 scalar per column table. The old caller-side OverflowError fallback
to verbose YAML lists is gone — it silently exploded a ~100 MB base64 write
into a tens-of-millions-node PyYAML tree (the OOM that killed that run).

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
# Wide format: fixed 16 bytes/value little-endian signed. Payloads carry
# ``"fmt": "i128"``; its absence means the legacy int64 array (old checkpoints).
_FMT_I128 = "i128"
_I128_BYTES = 16


def _to_b64(flat: "array.array") -> str:
	"""C int64 buffer → base64 ASCII scalar (one YAML node)."""
	return base64.b64encode(flat.tobytes()).decode("ascii")


def _from_b64(b64: str) -> "array.array":
	"""base64 ASCII scalar → C int64 buffer."""
	flat = array.array(_TYPECODE)
	flat.frombytes(base64.b64decode(b64))
	return flat


def _to_b64_i128(values: list) -> str:
	"""Arbitrary Python ints (|v| < 2^127) → 16-byte-per-value base64 scalar."""
	return base64.b64encode(
		b"".join(v.to_bytes(_I128_BYTES, "little", signed=True) for v in values)
	).decode("ascii")


def _from_b64_i128(b64: str) -> list:
	"""Inverse of ``_to_b64_i128`` → list of Python ints."""
	raw = base64.b64decode(b64)
	if len(raw) % _I128_BYTES:
		raise ValueError(f"i128 payload length {len(raw)} not a multiple of {_I128_BYTES}")
	return [int.from_bytes(raw[i:i + _I128_BYTES], "little", signed=True)
	        for i in range(0, len(raw), _I128_BYTES)]


def pack_int_columns(rows, ncols: int) -> dict:
	"""Pack an iterable of equal-width int rows into a compact JSON-able payload.

	int64-range tables use the fast ``array('q')`` path; any wider value
	(e.g. cell addresses of a >63-bit genome) transparently re-packs the whole
	table in the 16-byte "i128" format — NEVER the verbose YAML-list form.
	"""
	flat = array.array(_TYPECODE)
	wide: list = []
	n = 0
	for row in rows:
		if len(row) != ncols:
			raise ValueError(f"row width {len(row)} != ncols {ncols}")
		vals = [int(v) for v in row]
		if wide:
			wide.extend(vals)
		else:
			try:
				flat.extend(vals)
			except OverflowError:
				# extend may have appended part of this row before raising —
				# keep only the n complete rows, then re-add this row whole.
				wide = flat.tolist()[: n * ncols]
				wide.extend(vals)
		n += 1
	if wide:
		return {"_packed": _PACK_TAG, "fmt": _FMT_I128, "n": n, "cols": int(ncols),
		        "b64": _to_b64_i128(wide)}
	return {"_packed": _PACK_TAG, "n": n, "cols": int(ncols), "b64": _to_b64(flat)}


def is_packed(x: Any) -> bool:
	"""True iff ``x`` is a packed payload produced by ``pack_int_columns`` /
	``pack_int_array`` (both share the same tag and decode dispatch)."""
	return isinstance(x, dict) and x.get("_packed") == _PACK_TAG


def _unpack_flat(payload: dict) -> "array.array | list":
	"""Decode either format's byte payload to a flat int sequence."""
	if payload.get("fmt") == _FMT_I128:
		return _from_b64_i128(payload["b64"])
	return _from_b64(payload["b64"])


def unpack_int_columns(payload: dict) -> list:
	"""Inverse of ``pack_int_columns`` → list of int tuples, in original order."""
	cols = int(payload["cols"])
	n = int(payload["n"])
	flat = _unpack_flat(payload)
	if len(flat) != n * cols:
		raise ValueError(f"packed length {len(flat)} != n*cols {n * cols}")
	return [tuple(flat[i:i + cols]) for i in range(0, len(flat), cols)]


def pack_int_array(values) -> dict:
	"""Pack a FLAT 1-D int sequence (e.g. a genome's flat connection array) — the
	``cols=1`` form, but ``unpack_int_array`` returns a flat list, not 1-tuples.

	Shares the same byte core / tag as ``pack_int_columns`` (so ``is_packed``
	covers both); the ``cols=1`` marker is how decode knows to flatten.
	Values outside int64 transparently use the "i128" format (never raises
	OverflowError; the callers' verbose-list fallbacks are now dead code).
	"""
	vals = [int(v) for v in values]  # materialize: generators can't be re-read
	flat = array.array(_TYPECODE)
	try:
		flat.extend(vals)
	except OverflowError:
		return {"_packed": _PACK_TAG, "fmt": _FMT_I128, "n": len(vals), "cols": 1,
		        "b64": _to_b64_i128(vals)}
	return {"_packed": _PACK_TAG, "n": len(flat), "cols": 1, "b64": _to_b64(flat)}


def unpack_int_array(payload: dict) -> list:
	"""Inverse of ``pack_int_array`` → flat list of ints (no tuple churn)."""
	flat = _unpack_flat(payload)
	if len(flat) != int(payload["n"]):
		raise ValueError(f"packed length {len(flat)} != n {payload['n']}")
	return flat.tolist() if isinstance(flat, array.array) else flat
