"""Guards the curriculum-on-INITIAL-CONDITIONS design (01/06/2026).

The curriculum-on-steps was empirically refuted: at dt=0.001 a 10-30 ms episode
gives ~0.003-0.22° of control authority vs a 5° threshold, so a do-nothing
hover scored identically to a perfect PID and the GA had no gradient. The fix:
FIX the horizon at a signal-bearing length and make the easy→hard axis the
initial-condition severity (tilt + body-rate). These tests fail loudly if the
schedule regresses to short, signal-free horizons.
"""

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
	"run_curriculum_ga", str(Path(__file__).parent / "run_curriculum_ga.py"))
cur = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cur)

# Empirically, control authority dominates the 5° band only at ≥~100 ms; below
# that the GA selects on initial-condition luck, not control skill.
MIN_SIGNAL_BEARING_STEPS = 100


def test_horizon_is_fixed_and_signal_bearing():
	steps = {s.steps for s in cur.DEFAULT_CURRICULUM}
	assert len(steps) == 1, f"horizon must be FIXED across stages, got {steps}"
	(only,) = steps
	assert only >= MIN_SIGNAL_BEARING_STEPS, (
		f"horizon {only} steps ({only}ms) is below the {MIN_SIGNAL_BEARING_STEPS}ms "
		f"signal floor — a do-nothing hover would be indistinguishable from a real "
		f"controller (the refuted curriculum-on-steps regime).")
	assert only == cur.FIXED_HORIZON_STEPS


def test_difficulty_axis_grows_monotonically():
	tilts = [s.tilt_deg for s in cur.DEFAULT_CURRICULUM]
	rates = [s.body_rate for s in cur.DEFAULT_CURRICULUM]
	assert tilts == sorted(tilts) and len(set(tilts)) == len(tilts), \
		f"tilt must strictly increase (easy→hard IC), got {tilts}"
	assert rates == sorted(rates), f"body-rate must be non-decreasing, got {rates}"
	# The hardest stage should be a genuinely violent disturbance.
	assert tilts[-1] >= 45.0


def test_yaw_capped_so_it_does_not_dominate():
	# Yaw tilt is capped at 45° so yaw error never swamps the roll/pitch
	# attitude objective even when the stage tilt exceeds 45°.
	for s in cur.DEFAULT_CURRICULUM:
		ec = cur._build_ec(s)
		assert math.degrees(ec.max_initial_yaw_rad) <= 45.0 + 1e-6
		assert math.degrees(ec.max_initial_tilt_rad) == pytest.approx(s.tilt_deg)
		assert ec.steps_per_episode == s.steps
