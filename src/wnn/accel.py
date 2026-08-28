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
# ⚠️ STAGED: the ABI-7 wheel (fitness_combine, 19/08/2026) is built and waiting;
# scripts/worker_swap flips this to 7 AS PART OF the swap. It must not move
# earlier: the CONTROLLER evaluator imports wnn.accel too (evaluator.py
# accel_or_none), so bumping ahead of the install takes down every controller
# run — measured live on the 19/08 zscore smoke, rc=1 at this very assert.
# 8 (26/08/2026): desirability_fitness_combine (docs/DESIRABILITY_FITNESS_SHAPES.md). Additive.
# 9 (26/08/2026): IDSCacheWrapper.desirability_ce_anchor/.base_rate_entropy — the
#   CE half-anchor derived from the cache's OWN train labels, so binary and
#   multiclass arms each get their own scale. Additive.
# 10 (27/08/2026): IDSCacheWrapper.desirability_ce_anchor takes NO argument —
#   the calibration is now a ram_core constant and the scale comes from the
#   cache's own labels. SIGNATURE CHANGE, not additive: an ABI-9 wheel has the
#   2-arg form and would TypeError, so the bump makes the mismatch loud.
# 11 (28/08/2026): IDSCacheWrapper.set_coverage_aware — coverage-aware scoring
#   (docs/COVERAGE_AWARE_SCORER_SPEC.md). ADDITIVE, and bumped anyway ON PURPOSE.
#   The 10-wheel lacks the method, so new Python against it raised AttributeError
#   at evaluator construction and killed flow 6010 three seconds in. The ABI
#   assert is the DESIGNED place to catch that, and it did not, because an
#   additive surface change had left the version alone. Bump on any surface
#   change, additive or not — the assert is only worth what it is kept current.
EXPECTED_ABI = 11

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


def fitness_combine(
	flat:         list[float],
	n:            int,
	weights:      list[float],
	higher:       list[bool],
	aggregation:  str,
	zrank_clamp:  float,
) -> list[float]:
	"""Combine per-metric columns into one score per genome, in the WHEEL.

	Thin passthrough to `ram_core::fitness` (worker ABI 7). The controller
	reaches the SAME ram_core code through wnn.control._accel; this is the
	worker-side door to it, so IDS and controller rank by one implementation
	rather than two that drift.

	`flat` is column-major: all n values of column 0, then column 1, ... Columns
	are NEVER pre-negated — orientation travels in `higher` (True = larger is
	better), because a negation applied at one call site and not another is the
	drift this shared combine exists to prevent.

	Raises rather than falling back: a Python re-implementation would rank
	genomes differently, and a silently different ranking is unreportable
	(No-Python-Shortcuts rule).
	"""
	accel = require_accel()
	combine = getattr(accel, "fitness_combine", None)
	if combine is None:
		raise RuntimeError(
			f"ram_accelerator exposes no fitness_combine (installed ABI "
			f"{getattr(accel, 'ABI_VERSION', 0)}, needs >= 7). {BUILD_HINT}"
		)
	return list(combine(flat, n, weights, higher, aggregation, zrank_clamp))


def desirability_fitness_combine(
	flat:         list[float],
	n:            int,
	weights:      list[float],
	shapes:       list[str],
	half_anchors: list[float],
) -> list[float]:
	"""Desirability combine (worker ABI 8; docs/DESIRABILITY_FITNESS_SHAPES.md).

	Thin passthrough to `ram_core::fitness::desirability_combine_flat` — the
	same function the controller wheel exports (ABI 25 there). score =
	Σ w·h = weighted half-lives of desirability lost, LOWER = better, ABSOLUTE
	(not pool-relative). `shapes[c]` ∈ {"power", "exp"}; `half_anchors[c]` is
	where u = 0.5. Raises rather than falling back (No-Python-Shortcuts rule).
	"""
	accel = require_accel()
	combine = getattr(accel, "desirability_fitness_combine", None)
	if combine is None:
		raise RuntimeError(
			f"ram_accelerator exposes no desirability_fitness_combine (installed "
			f"ABI {getattr(accel, 'ABI_VERSION', 0)}, needs >= 8). {BUILD_HINT}"
		)
	return list(combine(flat, n, weights, shapes, half_anchors))


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
