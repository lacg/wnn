"""
Single import point for the Rust accelerator (ram_accelerator).

Why this exists (docs/ARCHITECTURE_REVIEW_2026-06.md §2.2): 36 files used to
import the raw extension ad hoc, each with its own availability policy —
silent `except ImportError: <PyTorch fallback>` paths violated the
No-Python-Shortcuts rule and masked broken/stale builds (different memory-mode
semantics, missing cancel checks). This module centralizes:

- import + fail-loud policy (set WNN_ALLOW_PY_FALLBACK=1 to permit fallbacks)
- ABI version check (a stale build fails at import, not mid-experiment)
- the canonical genome flattener (was duplicated in 6 evaluators)

Usage:
	from wnn.accel import require_accel, accel_or_none, flatten_genomes

	ra = require_accel()          # raises ImportError/RuntimeError if unusable
	ra = accel_or_none()          # None if unavailable AND fallback allowed
	bits, neurons, conns = flatten_genomes(genomes)
"""

import os

# ABI contract with the installed extension. Must equal lib.rs ABI_VERSION.
EXPECTED_ABI = 4

BUILD_HINT = (
	"Rebuild the accelerator: cd src/wnn/ram/strategies/accelerator && "
	"unset CONDA_PREFIX && maturin develop --release (use the wnn/ venv)"
)


def _fallback_allowed() -> bool:
	return os.environ.get("WNN_ALLOW_PY_FALLBACK", "0") == "1"


_accel = None
_import_error: Exception | None = None

try:
	import ram_accelerator as _accel_mod
	_accel = _accel_mod
except ImportError as e:  # pragma: no cover - exercised only without the ext
	_import_error = e

if _accel is not None:
	_abi = getattr(_accel, "ABI_VERSION", 0)
	if _abi != EXPECTED_ABI:
		_import_error = RuntimeError(
			f"ram_accelerator ABI mismatch: installed={_abi}, expected={EXPECTED_ABI}. "
			f"The installed extension is stale. {BUILD_HINT}"
		)
		_accel = None

AVAILABLE = _accel is not None


def require_accel():
	"""Return the accelerator module or raise loudly (no silent fallback)."""
	if _accel is None:
		raise ImportError(
			f"ram_accelerator unavailable: {_import_error}. {BUILD_HINT}"
		) from _import_error
	return _accel


_warned_fallback = False


def accel_or_none():
	"""Return the accelerator, or None ONLY if WNN_ALLOW_PY_FALLBACK=1.

	Without the escape hatch this raises — a PyTorch fallback has different
	memory-mode semantics (TERNARY vs QUAD_WEIGHTED) and must never engage
	silently (No-Python-Shortcuts rule).
	"""
	global _warned_fallback
	if _accel is not None:
		return _accel
	if _fallback_allowed():
		if not _warned_fallback:
			_warned_fallback = True
			print(
				"[WNN.ACCEL] WARNING: ram_accelerator unavailable and "
				"WNN_ALLOW_PY_FALLBACK=1 — Python fallbacks engaged. Results "
				"use different memory-mode semantics; do NOT report them. "
				f"({_import_error})"
			)
		return None
	return require_accel()


def metal_available() -> bool:
	"""True if the accelerator is importable and Metal is usable."""
	if _accel is None:
		return False
	try:
		return bool(_accel.metal_available())
	except Exception:
		return False


def flatten_genomes(
	genomes,
	generate_missing_connections: bool = False,
	total_input_bits: int = 0,
	rng=None,
) -> tuple[list[int], list[int], list[int]]:
	"""Canonical flat-genome marshaller for the Rust boundary.

	Layout contract (validated Rust-side by validate_flat_genomes):
	- bits_flat:        per-NEURON bit counts, Σ neurons entries per genome
	- neurons_flat:     per-cluster neuron counts, num_clusters entries/genome
	- connections_flat: Σ bits entries per genome, or EMPTY for all genomes
	  (random-connection fallback in Rust)

	A batch where only SOME genomes carry connections silently shifts every
	subsequent genome's offsets — so this either includes connections for
	every genome or raises (or generates random ones when
	generate_missing_connections=True, the multistage behavior).
	"""
	import random as _random
	genomes = list(genomes)
	bits_flat: list[int] = []
	neurons_flat: list[int] = []
	connections_flat: list[int] = []
	n_with_connections = 0

	for g in genomes:
		bits_flat.extend(g.bits_per_neuron)
		neurons_flat.extend(g.neurons_per_cluster)
		if g.connections is not None:
			connections_flat.extend(g.connections)
			n_with_connections += 1
		elif generate_missing_connections:
			if total_input_bits <= 0:
				raise ValueError("flatten_genomes: generate_missing_connections needs total_input_bits > 0")
			r = rng if rng is not None else _random
			connections_flat.extend(
				r.randint(0, total_input_bits - 1)
				for b in g.bits_per_neuron
				for _ in range(b)
			)
			n_with_connections += 1

	if 0 < n_with_connections < len(genomes):
		raise ValueError(
			f"flatten_genomes: {n_with_connections}/{len(genomes)} genomes have "
			f"connections — a mixed batch would silently misalign every genome "
			f"after the first one without. Provide connections for all or none."
		)

	return bits_flat, neurons_flat, connections_flat
