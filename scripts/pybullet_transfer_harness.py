#!/usr/bin/env python3
"""INDEPENDENT-SIMULATOR TRANSFER HARNESS — score any policy in gym-pybullet-drones.

WHY THIS EXISTS (task #4, and #5 folded into it by Luiz's 12/08 call). Every
controller number this project has ever published was measured in OUR simulator,
which we also wrote. That is the single easiest thing for a reviewer to discount,
and no amount of internal rigour answers it. This harness re-scores a trained
winner in a simulator we did NOT write, on a plant we did not tune, and — the
part that makes it a comparison rather than a demo — scores the RIVALS in the
SAME simulator through the SAME interface. A learned baseline quoted from its own
paper's simulator is not a comparison; a learned baseline run here is.

THE CONTRACT. A policy is anything with `reset()` and
`act(gyro, accel, quat, target) -> [pwm; 4] in [0,1]`. Three land here:
  * WnnPolicy      — our LUT winner, via wnn.control.evaluator.build_controller
  * PidPolicy      — the classical rival already used as our in-sim baseline
  * LearnedPolicy  — the PPO/SAC MLP baseline (task #5), trained IN THIS SIM so
                     its numbers are commensurable rather than cited across sims
Everything downstream (episode protocol, metrics, report seeds) is shared, so a
row of the output table differs ONLY by the policy.

METRICS ARE THE PROJECT'S TRIPLE, computed identically to the in-sim scorer:
stable% / mean |attitude error| (deg) / steady-state error (deg over the settle
window). Same report seeds, same tilt, same episode count — so an in-sim row and
a pybullet row for the same winner are directly comparable, and the DIFFERENCE
between them is the transfer gap this harness exists to measure.

⚠️ NOT YET RUNNABLE — gym-pybullet-drones and pybullet are NOT installed in the
venv (verified 12/08). Deferred deliberately: an L1 chain was armed, and adding
packages to the shared venv while a chain is flying is exactly the class of move
that killed three cohorts in a day. Install at a chain boundary:

    /Volumes/.../venv/bin/pip install gym-pybullet-drones

⚠️ THE TWO UNIT BRIDGES ARE THE WHOLE RISK. Both are stated explicitly below
rather than buried, because a silent error in either would masquerade as "the
controller does not transfer" — the same failure shape as reading Molchanov's
settling time as a time constant (docs/disturbance_param_sources.md S8).

  1. ACTION: our policy emits normalized PWM in [0,1]; pybullet's drone takes
     RPM. The bridge is rpm = sqrt(pwm) * MAX_RPM only if our pwm is normalized
     THRUST; it is rpm = pwm * MAX_RPM if our pwm is normalized RPM. Molchanov
     eq. (7) treats û as normalized rotor ANGULAR VELOCITY and computes force as
     f = f_max * û² — i.e. thrust ∝ û². Our sim's convention MUST be read out of
     controller.rs before this is trusted; the ACTION_MODE switch below makes the
     assumption visible and swappable instead of implicit.
  2. OBSERVATION: our features want body-frame gyro (rad/s) and accelerometer
     (m/s², gravity INCLUDED — our tilt features derive from the gravity vector).
     pybullet reports linear acceleration in the WORLD frame without gravity, so
     the accel must be rotated into the body frame and gravity added back, or the
     tilt features read zero and the controller flies blind.

USAGE (once installed):
    python scripts/pybullet_transfer_harness.py \
        --winner logs/controller/.../XXX_winner.yaml.gz \
        --episodes 100 --steps 2000 --tilt 5.0 \
        --report-seeds 99990101 99990102 99990103 99990104 99990105
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

# Keep the GPU free for whatever else is on the box; this harness is CPU-cheap.
os.environ.setdefault("WNN_CONTROLLER_GPU_EVAL", "0")


# ---------------------------------------------------------------------------
# Policy protocol — the seam that makes WNN / PID / learned commensurable
# ---------------------------------------------------------------------------

class Policy(Protocol):
	"""Anything scoreable here. `act` returns 4 normalized motor commands."""

	name: str

	def reset(self, init_yaw: float) -> None: ...

	def act(self, gyro: Sequence[float], accel: Sequence[float],
	        quat: Sequence[float], target: Sequence[float]) -> Sequence[float]: ...


class WnnPolicy:
	"""Our LUT winner. Loads the winner's OWN spec (sn, bits, mode, obs flags)
	from the checkpoint, so an sn=0 reflex and an sn=8 DFA winner both work
	without harness changes — the same property rescore_winners_steady.py relies
	on."""

	def __init__(self, winner_path: str):
		from wnn.control.phased_ga import _ctl_load
		from wnn.control.evaluator import build_controller
		payload = _ctl_load(winner_path)
		genome = payload["best_genome"]
		self._c = build_controller(genome)
		self.spec = genome.spec
		self.name = f"WNN({os.path.basename(winner_path)})"

	def reset(self, init_yaw: float) -> None:
		self._c.reset(init_yaw)

	def act(self, gyro, accel, quat, target):
		# The Rust controller's step() takes exactly this triple; quat is unused
		# by the student (it flies on IMU alone, which is the deployment story).
		return self._c.step(list(gyro), list(accel), list(target))


class PidPolicy:
	"""The classical rival, driven through the same seam. Uses the project's own
	PID so the comparison is against the SAME controller our in-sim tables use,
	not a fresh re-tuning that would confound the transfer question."""

	def __init__(self, airframe: str = "cf21_brushless"):
		from wnn.control._accel import PidFirmware  # noqa: F401  (name per ABI)
		self._pid = PidFirmware(airframe) if False else None  # wired at install
		self.name = f"PID({airframe})"
		raise NotImplementedError(
			"Wire to the project's PID once pybullet is installed: the exact "
			"entry point is whatever score_classical_baselines.py uses, so the "
			"pybullet row and the in-sim row share one implementation.")

	def reset(self, init_yaw: float) -> None: ...

	def act(self, gyro, accel, quat, target): ...


# ---------------------------------------------------------------------------
# Unit bridges — stated, switchable, and asserted rather than assumed
# ---------------------------------------------------------------------------

ACTION_THRUST_SQUARED = "thrust_sq"   # rpm = sqrt(pwm) * MAX_RPM  (thrust ∝ rpm²)
ACTION_LINEAR_RPM = "linear_rpm"      # rpm = pwm * MAX_RPM

def pwm_to_rpm(pwm: np.ndarray, max_rpm: float, mode: str) -> np.ndarray:
	"""Bridge #1. Which branch is correct depends on what our pwm NORMALIZES.

	Read controller.rs's force computation before trusting either: Molchanov
	eq. (7)+(f = f_max·û²) treats û as normalized angular velocity with thrust
	quadratic in it, which is ACTION_THRUST_SQUARED. If our sim instead maps pwm
	linearly to thrust, the correct bridge is sqrt on the THRUST side, not here.
	Getting this wrong scales every command and looks exactly like a controller
	that cannot transfer.
	"""
	p = np.clip(np.asarray(pwm, dtype=float), 0.0, 1.0)
	if mode == ACTION_THRUST_SQUARED:
		return np.sqrt(p) * max_rpm
	if mode == ACTION_LINEAR_RPM:
		return p * max_rpm
	raise ValueError(f"unknown action bridge {mode!r}")


def body_imu_from_pybullet(quat_xyzw, ang_vel_world, lin_acc_world, g: float = 9.81):
	"""Bridge #2: pybullet state → the IMU our features expect.

	Returns (gyro_body [rad/s], accel_body [m/s², gravity INCLUDED]).

	Our tilt/peraxis features are derived from the gravity vector in the
	accelerometer reading (atan2 of its components). pybullet reports world-frame
	linear acceleration WITHOUT gravity, so a naive pass-through makes every tilt
	feature read ~0 and the controller flies blind while looking healthy.
	"""
	from scipy.spatial.transform import Rotation
	R = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float))  # body→world
	Rt = R.inv()
	gyro_body = Rt.apply(np.asarray(ang_vel_world, dtype=float))
	# Specific force = a - g (an accelerometer at rest reads +g upward).
	acc_world = np.asarray(lin_acc_world, dtype=float) - np.array([0.0, 0.0, -g])
	accel_body = Rt.apply(acc_world)
	return gyro_body, accel_body


# ---------------------------------------------------------------------------
# Metrics — identical definitions to the in-sim scorer, or the rows don't compare
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
	mean_err_deg: float
	steady_err_deg: float
	stable: bool
	diverged: bool
	steps: int


@dataclass
class ScoreTable:
	policy: str
	rows: list = field(default_factory=list)

	def triple(self) -> tuple[float, float, float]:
		"""(stable%, mean err°, steady°) — the project's reporting triple, in the
		order fixed on 08/08 (memory: feedback_controller_report_triple)."""
		if not self.rows:
			return (0.0, float("nan"), float("nan"))
		stable = 100.0 * sum(r.stable for r in self.rows) / len(self.rows)
		err = float(np.mean([r.mean_err_deg for r in self.rows]))
		steady = float(np.mean([r.steady_err_deg for r in self.rows]))
		return (stable, err, steady)


def attitude_error_deg(quat_xyzw) -> float:
	"""Angle between body-z and world-z — the same 'attitude error' our sim
	reports, so the numbers mean the same thing in both simulators."""
	from scipy.spatial.transform import Rotation
	z_body = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).apply([0.0, 0.0, 1.0])
	return math.degrees(math.acos(max(-1.0, min(1.0, float(z_body[2])))))


def score_episode(env, policy: Policy, steps: int, settle_frac: float = 0.5,
                  stable_thresh_deg: float = 5.0,
                  action_mode: str = ACTION_THRUST_SQUARED) -> EpisodeResult:
	"""One episode. STEADY is the mean error over the last `settle_frac` of the
	episode — the definition our tables use; changing it here silently would make
	every comparison meaningless."""
	obs, _ = env.reset()
	policy.reset(0.0)
	errs: list[float] = []
	diverged = False
	max_rpm = float(getattr(env, "MAX_RPM", 21713.0))
	for t in range(steps):
		state = env._getDroneStateVector(0)
		quat = state[3:7]
		ang_v = state[13:16]
		lin_a = np.zeros(3)   # replaced by a finite-difference or env accessor
		gyro, accel = body_imu_from_pybullet(quat, ang_v, lin_a)
		pwm = policy.act(gyro, accel, quat, [0.0, 0.0, 0.0])
		rpm = pwm_to_rpm(np.asarray(pwm), max_rpm, action_mode)
		obs, _, term, trunc, _ = env.step(rpm.reshape(1, 4))
		e = attitude_error_deg(env._getDroneStateVector(0)[3:7])
		errs.append(e)
		if e > 90.0 or term or trunc:
			diverged = True
			break
	if not errs:
		return EpisodeResult(float("nan"), float("nan"), False, True, 0)
	k = max(1, int(len(errs) * settle_frac))
	mean_err = float(np.mean(errs))
	steady = float(np.mean(errs[-k:]))
	return EpisodeResult(mean_err, steady, (not diverged) and mean_err <= stable_thresh_deg,
	                     diverged, len(errs))


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__,
	                             formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--winner", required=True, help="path to a *_winner.yaml.gz")
	ap.add_argument("--episodes", type=int, default=100)
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--tilt", type=float, default=5.0, help="initial tilt (deg)")
	ap.add_argument("--report-seeds", type=int, nargs="+",
	                default=[99990101, 99990102, 99990103, 99990104, 99990105])
	ap.add_argument("--action-mode", default=ACTION_THRUST_SQUARED,
	                choices=[ACTION_THRUST_SQUARED, ACTION_LINEAR_RPM])
	args = ap.parse_args()

	try:
		import gym_pybullet_drones  # noqa: F401
	except ModuleNotFoundError:
		print(__doc__.split("⚠️ NOT YET RUNNABLE")[1].split("⚠️")[0].strip())
		print("\nABORT: gym-pybullet-drones is not installed. See above.")
		return 2

	raise SystemExit(
		"The env wiring is deliberately unfinished: DO NOT run this until both "
		"unit bridges are verified against controller.rs (action) and the env's "
		"IMU accessor (observation). A harness that silently mis-scales commands "
		"reports 'no transfer' and looks like a scientific result.")


if __name__ == "__main__":
	raise SystemExit(main())
