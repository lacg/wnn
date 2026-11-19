"""Per-generation determinism digest (Luiz 17/11/2026, after the s2 control divergence).

One line per GA generation: a sha256 over every candidate of the population entering the
generation (= everything the previous generation produced), in population order —
its structural fingerprint (genome.fingerprint(): shape, connectivity and the cells' Rust
content digest) and every metric at FULL precision (float.hex, so 1 ulp shows). Two runs
whose digest lines differ at generation g first diverged in generation g-1's evaluation or
selection; identical digests ⇒ identical populations and scores.

Process-invariant by construction: no id(), no Python hash() (PYTHONHASHSEED), no
timestamps — only repr of int tuples and float.hex.
"""
from __future__ import annotations

import dataclasses
import hashlib

DIGEST_TAG = "[gen-digest]"


def _metric_items(m) -> list:
	"""(name, canonical value) for every metric that HAS a value. None-valued fields are
	skipped (19/11/2026): a new optional field that a flag-off run never fills (e.g. ABI 59's
	episode_err_p99_deg) must not change the digest, or the cross-ABI determinism gate fails on
	identical populations. Still exact: a field going None -> value adds a token and changes it."""
	if m is None:
		return [("metrics", "None")]
	if dataclasses.is_dataclass(m):
		items = [(f.name, getattr(m, f.name)) for f in dataclasses.fields(m)]
	else:
		items = sorted(vars(m).items())
	return [(k, v.hex() if isinstance(v, float) else repr(v)) for k, v in items if v is not None]


def candidate_token(genome, metrics) -> str:
	"""One candidate as a canonical string: fingerprint + full-precision metrics."""
	fp = genome.fingerprint() if hasattr(genome, "fingerprint") else ("no-fingerprint",)
	return repr(fp) + "|" + ";".join(f"{k}={v}" for k, v in _metric_items(metrics))


def population_digest(population) -> str:
	"""sha256 (first 16 hex) over the (genome, metrics) population, in order."""
	h = hashlib.sha256()
	for genome, metrics in population:
		h.update(candidate_token(genome, metrics).encode())
		h.update(b"\n")
	return h.hexdigest()[:16]


def digest_line(name: str, generation: int, population) -> str:
	"""The log line. `generation` is the 0-based index of the generation about to run."""
	return (f"{DIGEST_TAG} {name} entering gen {generation + 1} (= after gen {generation}) "
	        f"n={len(population)} sha={population_digest(population)}")
