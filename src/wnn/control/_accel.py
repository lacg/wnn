"""
Single import point for the controller Rust extension (``ram_controller``).

Mirror of :mod:`wnn.accel` (the IDS/LM worker facade), for the controller wheel.

Why this exists: the drone-controller hot-path used to live in the shared
``ram_accelerator`` wheel. On 2026-06-19 it was split into its own
``ram_controller`` wheel (crate split) so a controller-only change rebuilds ONLY
that wheel and never forces a worker swap. All of ``wnn.control`` imports the
controller surface from here instead of reaching into the extension directly, so
there is one place that:

- imports ``ram_controller`` (with a transition fallback to ``ram_accelerator``
  for the combined wheel, in case only the old build is installed),
- asserts the ABI version (a stale build fails at import, not mid-run),
- re-exports the controller surface (classes, decoders, dagger training, the
  per-process cancel flag).

IMPORTANT — the cancel flag here is the CONTROLLER process's flag
(``ram_controller``'s copy of the ``ram_core::cancel`` static), independent of
the worker's. ``wnn.accel`` must NOT be used for controller cancellation and
vice-versa.

Usage::

    from wnn.control._accel import AttitudeSim, WnnController
    from wnn.control import _accel as ra        # ra.dagger_train_inplace(...)
"""

import os

# ABI contract with the installed extension. Must equal the controller crate's
# ABI_VERSION (controller/lib.rs).
# 3 = W2 disturbances (AttitudeSim.set_disturbance / disturbance_episode_seed /
#     score_controllers_metal + eval_ensemble_closed_loop dist args).
# 4 = overactuated Phase 1 (AttitudeSim.set_geometry/step_n/perturb_geometry/
#     set_rotor_asym + geometry=/rotor_asym= on score_controllers_metal AND
#     score_controllers_cpu; None = legacy quad, bit-identical).
# 5 = overactuated Phase 2 step 1 (AllocLqrRs allocator-aware LQR teacher).
# 6 = mono/jerk semantics unified in score_controllers_cpu to the GPU kernel's
#     (mono = last decision step per episode; jerk = per-episode mean).
# 7 = overactuated Phase 2 step 2 (allocator-LQR residual baseline: alloc_*
#     kwargs on both scorers; AllocBaseline precomputed pinv).
# 8 = overactuated Phase 2 step 3 (AttitudeSim.geometry_rows exporter).
# 9 = allocation-effort metric (13-metric scorer rows; Σu² fitness input).
# 10 = effort on alloc runs = EXCESS vs the pinv optimum at equal realized
#      wrench (collective-shedding-proof); raw Σu² on non-alloc runs.
# 11 = residual anchor = NEUTRAL_DECODE (cell-semantics-derived; QUAD 0.75):
#      untrained residual exactly 0. Wheel exports NEUTRAL_DECODE.
# 12 = memory-mode-aware controller (granularity ablation): WnnController /
#      dagger_train_batch_inplace take memory_mode= (TERNARY=0 empty=0.5,
#      QUAD=2 default, BINARY=3 antagonist-pair E/I halves). Wheel exports
#      neutral_decode_for_mode(mode). QUAD bit-identical to 11.
# 13 = Phase-4 state-pressure: stateful teachers lqi (id 3) + mpcof (id 4,
#      observer fed by the rollout loop); D5 dropout / D6 latency / D7 torque
#      jitter disturbance levers + 4 RewardGatedConfigPacked dist fields.
#      Zero-default disturbances bit-identical to 12.
# 23 (19/08/2026): fitness_combine — generic ram_core fitness combines
# (harmonic|arithmetic|zscore); the fitness calculators delegate their math here.
# 24 (21/08/2026): gated_fitness_combine — viability gate (stable/err, Deb's
# rules) before the base combine. Additive; fitness_combine untouched.
# 25 (26/08/2026): desirability_fitness_combine — one continuous multiplicative
# utility (Cobb-Douglas; docs/DESIRABILITY_FITNESS_SHAPES.md); score = weighted
# half-lives of desirability lost. Additive; both prior combines untouched.
EXPECTED_ABI = 25

BUILD_HINT = (
	"Build the controller wheel: cd src/wnn/ram/strategies/accelerator && "
	"unset CONDA_PREFIX && maturin develop --release -m controller/Cargo.toml "
	"(use the wnn/ venv)"
)


def _fallback_allowed() -> bool:
	return os.environ.get("WNN_ALLOW_PY_FALLBACK", "0") == "1"


_ctl = None
_import_error: Exception | None = None
_via_fallback = False

try:
	import ram_controller as _ctl_mod
	_ctl = _ctl_mod
except ImportError as e:
	_import_error = e
	try:
		# Transition fallback: the controller symbols still live in the combined
		# ram_accelerator wheel if the split build hasn't been installed yet.
		import ram_accelerator as _ctl_mod
		_ctl = _ctl_mod
		_via_fallback = True
	except ImportError:
		pass

if _ctl is not None:
	_abi = getattr(_ctl, "ABI_VERSION", 0)
	if _abi != EXPECTED_ABI:
		_import_error = RuntimeError(
			f"ram_controller ABI mismatch: installed={_abi}, expected={EXPECTED_ABI}. "
			f"The installed extension is stale. {BUILD_HINT}"
		)
		_ctl = None

AVAILABLE = _ctl is not None


def require_accel():
	"""Return the controller extension or raise loudly (no silent fallback)."""
	if _ctl is None:
		raise ImportError(
			f"ram_controller unavailable: {_import_error}. {BUILD_HINT}"
		) from _import_error
	return _ctl


# Eager re-export of the full controller surface so callers can write
# `from wnn.control._accel import AttitudeSim` or `import ... as ra; ra.X`.
# Raises loudly at import if the extension is missing (controller code cannot
# run without it — there is no Python fallback for the Rust hot-path).
_m = require_accel()
for _name in dir(_m):
	if not _name.startswith("_"):
		globals()[_name] = getattr(_m, _name)

if _via_fallback:
	print(
		"[WNN.CONTROL._ACCEL] note: ram_controller not found; using controller "
		"symbols from the combined ram_accelerator wheel (transition fallback).",
		flush=True,
	)
