"""Episode runner + GA orchestration for WNN attitude controllers.

Stays in Python (per `project_drone_controller_paper1.md`): outer-loop
orchestration is cheap and benefits from Python's ecosystem (numpy,
matplotlib for diagnostics, pytest for unit tests). The per-step hot
path (sim physics, controller forward, decoder, reward) is in Rust —
see `wnn.control.sim`, `.controller`, `.decoders`, and `compute_reward`
re-exported below.

The GA itself reuses the existing pipeline in
`src/wnn/ram/experiments/` (same one IDS flows use). Controller-specific
glue is the episode runner + fitness function that calls into Rust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from wnn.control._accel import (
	AttitudeSim,
	WnnController,
	compute_reward,
	monotonicity_violations,
	strategy_5_qsr_weighted,
)


# Reward defaults: attitude-error² weight is implicit (compute_reward returns
# -attitude_error²). The other lambdas are zero by default and turned on by
# the training script that wants smoothness or thermometer regularization.
_DEFAULT_LAMBDA_SMOOTH = 0.0
_DEFAULT_LAMBDA_MONO = 0.0


def _euler_to_quat(roll: float, pitch: float, pitch_unused: float = 0.0) -> tuple[float, float, float, float]:
	"""Roll-pitch-yaw (XYZ extrinsic) → unit quaternion (w, x, y, z).

	Matches the convention in controller.rs / pid.py.
	"""
	cr = math.cos(roll * 0.5)
	sr = math.sin(roll * 0.5)
	cp = math.cos(pitch_unused * 0.5)  # actually pitch
	sp = math.sin(pitch_unused * 0.5)
	# Yaw passed via pitch_unused for clarity issue — fixed below
	# Use the explicit version with 3 args
	raise RuntimeError("use _euler_to_quat_xyz instead")


def _euler_to_quat_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
	"""(roll, pitch, yaw) radians → unit quaternion (w, x, y, z).

	Z-Y-X (intrinsic / Tait-Bryan) convention matching pid._quat_to_euler.
	"""
	cr = math.cos(roll * 0.5)
	sr = math.sin(roll * 0.5)
	cp = math.cos(pitch * 0.5)
	sp = math.sin(pitch * 0.5)
	cy = math.cos(yaw * 0.5)
	sy = math.sin(yaw * 0.5)
	w = cr * cp * cy + sr * sp * sy
	x = sr * cp * cy - cr * sp * sy
	y = cr * sp * cy + sr * cp * sy
	z = cr * cp * sy - sr * sp * cy
	# normalize defensively
	n = math.sqrt(w * w + x * x + y * y + z * z)
	if n > 0:
		return (w / n, x / n, y / n, z / n)
	return (1.0, 0.0, 0.0, 0.0)


@dataclass
class EpisodeConfig:
	"""Configuration for a single training/eval episode."""

	# Sim
	dt: float = 0.001                       # 1 ms = 1 kHz update
	steps_per_episode: int = 2000           # 2 s of simulated flight
	max_initial_tilt_rad: float = math.radians(60.0)
	max_initial_yaw_rad: float = math.radians(45.0)  # bound on initial yaw (not full -π to π)
	max_initial_yaw_rate: float = 0.5       # rad/s; bound on initial omega_z
	max_initial_body_rate: float = 1.0      # rad/s; bound on initial omega_x/y

	# H4 axis curriculum: which attitude axes are perturbed in the IC
	# (roll, pitch, yaw). Inactive ⇒ that axis' tilt + body rate zeroed.
	active_axes: tuple = (True, True, True)

	# Reward weights
	lambda_smooth: float = _DEFAULT_LAMBDA_SMOOTH
	lambda_mono: float = _DEFAULT_LAMBDA_MONO

	# Early termination: stop episode if attitude error exceeds this (rad).
	# Default disabled. Useful during training to skip episodes that have
	# already diverged so we don't waste compute on hopeless rollouts.
	abort_attitude_error_rad: Optional[float] = None


@dataclass
class EpisodeResult:
	"""Per-episode telemetry returned by run_episode for analysis."""
	cumulative_reward: float
	mean_attitude_error_rad: float
	max_attitude_error_rad: float
	max_omega_norm: float          # peak |omega| during the episode (rad/s)
	steps_completed: int           # < steps_per_episode if early aborted
	diverged: bool                 # True iff sim.is_unstable() fired
	mean_pwm_jerk: float           # mean |pwm[t] - pwm[t-1]| over the episode


def sample_ics_flat(seed, num_eval: int, ec, active_axes=None) -> tuple[list[float], list[float]]:
	"""Sample num_eval initial conditions as FLAT (q0, omega0) lists for the
	GPU rollout kernel. SINGLE source of truth for the RNG draw order — the
	CPU path (_sample_initial_state per episode) and the Metal path
	(score_controllers_metal) are only interchangeable if every caller draws
	ICs in exactly this order. Was duplicated in control/evaluator.py and
	control/ga_memory.py (parity by convention only).

	active_axes (H4): override the per-episode axis mask (else ec.active_axes,
	else full 3-axis). The in-search curriculum eval passes the per-gen mask here."""
	import numpy as _np
	aa = active_axes if active_axes is not None else getattr(ec, "active_axes", (True, True, True))
	rng = _np.random.default_rng(seed)
	q0: list[float] = []
	omega0: list[float] = []
	for _ in range(num_eval):
		ep_rng = _np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q, om = _sample_initial_state(
			ep_rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
			ec.max_initial_body_rate, ec.max_initial_yaw_rate, aa,
		)
		q0 += [float(x) for x in q]
		omega0 += [float(x) for x in om]
	return q0, omega0


def _sample_initial_state(
	rng: np.random.Generator,
	max_tilt: float,
	max_yaw: float,
	max_body_rate: float,
	max_yaw_rate: float,
	active_axes: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
	"""Sample (initial_q, initial_omega) within the per-config bounds.

	H4 axis curriculum: an inactive axis has its initial tilt AND body rate zeroed.
	We DRAW always (then zero) so all-axes-active stays RNG-identical to pre-H4."""
	# Tilt: sample uniform in roll/pitch up to max_tilt, and yaw in a
	# bounded range (NOT -π to π) because target attitude is also a small
	# range and we don't want yaw error to dominate the rollout.
	r = float(rng.uniform(-max_tilt, max_tilt))
	p = float(rng.uniform(-max_tilt, max_tilt))
	y = float(rng.uniform(-max_yaw, max_yaw))
	roll = r if active_axes[0] else 0.0
	pitch = p if active_axes[1] else 0.0
	yaw = y if active_axes[2] else 0.0
	q = _euler_to_quat_xyz(roll, pitch, yaw)
	ox = float(rng.uniform(-max_body_rate, max_body_rate))
	oy = float(rng.uniform(-max_body_rate, max_body_rate))
	oz = float(rng.uniform(-max_yaw_rate, max_yaw_rate))
	omega = (
		ox if active_axes[0] else 0.0,
		oy if active_axes[1] else 0.0,
		oz if active_axes[2] else 0.0,
	)
	return q, omega


# Type alias: action_fn takes (gyro, accel, target_rpy, q) → 4 motor PWMs.
# WnnController.step takes (gyro, accel, target_attitude) but doesn't see q.
# PID needs q for attitude error. We pass both to the action_fn so the runner
# is agnostic to which controller is being evaluated.
ActionFn = Callable[
	[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]],
	tuple[float, float, float, float],
]


def run_episode(
	action_fn: ActionFn,
	sim: AttitudeSim,
	config: EpisodeConfig,
	target_attitude_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
	rng: np.random.Generator | None = None,
) -> EpisodeResult:
	"""Run one episode and return per-step + cumulative metrics.

	Args:
		action_fn:        callable that computes per-step motor PWMs. Signature:
		                  (gyro, accel, target_rpy, q) → (pwm0, pwm1, pwm2, pwm3).
		                  PID + WNN controllers wrap differently; the runner
		                  stays agnostic.
		sim:              AttitudeSim instance (will be reset).
		config:           EpisodeConfig.
		target_attitude_rpy: target (roll, pitch, yaw) in radians. Default is level.
		rng:              optional numpy RNG for reproducible initial conditions.

	Returns:
		EpisodeResult with cumulative_reward + per-episode telemetry.
	"""
	if rng is None:
		rng = np.random.default_rng()

	# Target quaternion for attitude_error comparisons.
	target_q = _euler_to_quat_xyz(*target_attitude_rpy)

	# Sample + apply initial conditions.
	init_q, init_omega = _sample_initial_state(
		rng,
		config.max_initial_tilt_rad,
		config.max_initial_yaw_rad,
		config.max_initial_body_rate,
		config.max_initial_yaw_rate,
		getattr(config, "active_axes", (True, True, True)),
	)
	sim.reset(q=list(init_q), omega=list(init_omega))

	cumulative = 0.0
	sum_err = 0.0
	max_err = 0.0
	max_omega = 0.0
	prev_pwm: Optional[tuple[float, float, float, float]] = None
	sum_jerk = 0.0
	jerk_count = 0
	diverged = False
	steps_done = 0

	for step_idx in range(config.steps_per_episode):
		if sim.is_unstable():
			diverged = True
			break

		gyro, accel = sim.read_imu()
		q = sim.quaternion

		pwm = action_fn(gyro, accel, target_attitude_rpy, q)

		# Apply action to sim.
		sim.step(list(pwm))

		# Compute reward components.
		attitude_err = sim.attitude_error(target_q)
		jerk = 0.0
		if prev_pwm is not None:
			dx = pwm[0] - prev_pwm[0]
			dy = pwm[1] - prev_pwm[1]
			dz = pwm[2] - prev_pwm[2]
			dw = pwm[3] - prev_pwm[3]
			jerk = dx * dx + dy * dy + dz * dz + dw * dw
			sum_jerk += math.sqrt(jerk)
			jerk_count += 1
		prev_pwm = pwm

		reward = compute_reward(
			attitude_err,
			motor_command_jerk=jerk,
			mono_violations=0,
			lambda_smooth=config.lambda_smooth,
			lambda_mono=config.lambda_mono,
		)
		cumulative += reward
		sum_err += attitude_err
		if attitude_err > max_err:
			max_err = attitude_err
		omega = sim.angular_velocity
		om_norm = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
		if om_norm > max_omega:
			max_omega = om_norm
		steps_done = step_idx + 1

		if config.abort_attitude_error_rad is not None and attitude_err > config.abort_attitude_error_rad:
			break

	mean_err = sum_err / max(steps_done, 1)
	mean_jerk = sum_jerk / max(jerk_count, 1) if jerk_count > 0 else 0.0
	return EpisodeResult(
		cumulative_reward=cumulative,
		mean_attitude_error_rad=mean_err,
		max_attitude_error_rad=max_err,
		max_omega_norm=max_omega,
		steps_completed=steps_done,
		diverged=diverged,
		mean_pwm_jerk=mean_jerk,
	)


def make_pid_action_fn(pid) -> ActionFn:
	"""Adapter from AttitudePID to ActionFn signature.

	AttitudePID.step takes (q, gyro, target_rpy). The runner provides q in the
	last positional arg.
	"""
	def fn(gyro, accel, target_rpy, q):
		return pid.step(q, gyro, target_rpy)
	return fn


def make_wnn_action_fn(controller: WnnController) -> ActionFn:
	"""Adapter from WnnController to ActionFn signature."""
	def fn(gyro, accel, target_rpy, q):
		return tuple(controller.step(list(gyro), list(accel), list(target_rpy)))
	return fn


def fitness_function(
	action_fn: ActionFn,
	sim: AttitudeSim,
	config: EpisodeConfig,
	num_episodes: int = 30,
	seed: int = 0,
) -> tuple[float, dict]:
	"""Average cumulative reward across `num_episodes` random initial conditions.

	Used as the GA's fitness signal. Reuses the existing GA pipeline in
	`src/wnn/ram/experiments/` — this function plugs into wherever the
	IDS pipeline currently computes per-genome fitness from a dataset.

	Returns:
		(mean_reward, metrics_dict): metrics_dict has aggregate stats matching
		the controller-specific JSON schema we'll write into
		`validation_summaries.threshold_metadata`.
	"""
	rng = np.random.default_rng(seed)
	results: list[EpisodeResult] = []
	for ep_idx in range(num_episodes):
		# Sub-RNG per episode so each is reproducible from (seed, ep_idx).
		ep_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
		res = run_episode(action_fn, sim, config, rng=ep_rng)
		results.append(res)

	rewards = np.array([r.cumulative_reward for r in results])
	mean_errs = np.array([r.mean_attitude_error_rad for r in results])
	max_errs = np.array([r.max_attitude_error_rad for r in results])
	steps_done = np.array([r.steps_completed for r in results])
	diverged_count = sum(1 for r in results if r.diverged)
	steady_threshold_rad = math.radians(5.0)
	stable_count = sum(
		1 for r in results
		if (not r.diverged) and r.mean_attitude_error_rad <= steady_threshold_rad
	)

	metrics = {
		"num_episodes": num_episodes,
		"mean_reward": float(rewards.mean()),
		"std_reward": float(rewards.std()),
		"min_reward": float(rewards.min()),
		"max_reward": float(rewards.max()),
		"mean_attitude_error_rad": float(mean_errs.mean()),
		"mean_attitude_error_deg": float(math.degrees(mean_errs.mean())),
		"mean_max_attitude_error_rad": float(max_errs.mean()),
		"mean_steps_completed": float(steps_done.mean()),
		"diverged_rate": diverged_count / num_episodes,
		"stable_rate": stable_count / num_episodes,
	}
	return float(rewards.mean()), metrics


__all__ = [
	"EpisodeConfig",
	"EpisodeResult",
	"ActionFn",
	"run_episode",
	"make_pid_action_fn",
	"make_wnn_action_fn",
	"fitness_function",
]
