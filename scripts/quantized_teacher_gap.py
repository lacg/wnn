#!/usr/bin/env python3
"""Decompose the WNN student-teacher gap: how much is REPRESENTATION, not learning?

The dfa1l study's best student reaches 77% stable against its LQR teacher's 100%.
Before blaming the learner (memory capacity, GA, DAgger), measure the ceiling the
WNN's own I/O imposes: the student cannot outperform a controller that is forced to
see and act through the same quantization it is. So: run LQR itself through the
WNN's input/output resolution and score it on the same rollout path.

Variants (all through the SAME Python rollout path, so deltas are internally
paired — cross-path absolute numbers are NOT comparable to the Rust study table):

  plain    LQR as-is (anchor; should sit near its known score)
  in-q     LQR whose 6 state inputs (3 attitude errors + 3 rates) are each squashed
           to BITS_PER_FEATURE quantile thresholds -> bin midpoints, calibrated from
           PID rollouts exactly like fit_thresholds_from_pid_rollouts (quantile
           method) — the WNN's input resolution applied to the teacher
  out-q    LQR whose per-motor PWM is snapped to LEVELS uniform levels in [0,1] —
           the WNN's 16-level actuation resolution
  both     in-q + out-q — the full representational ceiling

Reading:  plain - both  = representational cost (the part no learner can recover)
          both  - WNN   = the true learning gap
If `both` collapses under L3D the way the winners did, the robustness failure is
the SUBSTRATE's resolution, not the learner — and the lever is levels/bits, not
more search.

Usage: quantized_teacher_gap.py --disturbance L2D --out /path/gap_L2D.json
"""
import argparse
import json
import math
import sys

import numpy as np

from wnn.control.dagger import eval_closed_loop_reset
from wnn.control.evaluator import fold_pool_seed
from wnn.control.optimal import LQRController
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.training import (DisturbanceConfig, EpisodeConfig,
                                  make_pid_action_fn, run_episode)


def _episode_config(a):
	return EpisodeConfig(
		dt=0.001, steps_per_episode=a.steps,
		max_initial_tilt_rad=math.radians(a.tilt),
		max_initial_yaw_rad=math.radians(a.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=DisturbanceConfig.preset(a.disturbance, seed=a.sim_seed))


def _calibrate_input_thresholds(ec, seed, bits, episodes=10):
	"""Per-component quantile thresholds for LQR's 6 inputs, from PID rollouts —
	the same calibration recipe fit_thresholds_from_pid_rollouts uses for the WNN's
	features, applied to the teacher's own input space."""
	from wnn.control._accel import AttitudeSim
	from wnn.control.optimal import _quat_to_euler
	pid = AttitudePID(AttitudePIDConfig())
	sim = AttitudeSim()
	rng = np.random.default_rng(seed)
	samples = [[] for _ in range(6)]
	orig_step = pid.step

	def tap(q, gyro, target_rpy):
		r, p, y = _quat_to_euler(q)
		for i, v in enumerate((r - target_rpy[0], p - target_rpy[1], y - target_rpy[2],
		                       gyro[0], gyro[1], gyro[2])):
			samples[i].append(float(v))
		return orig_step(q, gyro, target_rpy)

	pid.step = tap
	for _ in range(episodes):
		pid.reset()
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		run_episode(make_pid_action_fn(pid), sim, ec, rng=ep_rng)
	pid.step = orig_step
	qs = np.linspace(0.0, 1.0, bits + 2)[1:-1]          # `bits` interior quantiles
	return [np.quantile(np.array(s), qs).tolist() for s in samples]


def _quantize_to_bins(v, thresholds):
	"""Value -> its thermometer bin's midpoint (the reconstruction the WNN's
	address implies: everything in a bin is the SAME input to the memory)."""
	t = thresholds
	k = 0
	while k < len(t) and v >= t[k]:
		k += 1
	if k == 0:
		return t[0]                       # below the lowest threshold
	if k == len(t):
		return t[-1]                      # above the highest
	return 0.5 * (t[k - 1] + t[k])


class QuantizedLQR:
	"""LQR forced through the WNN's I/O resolution. AttitudePID interface."""

	def __init__(self, in_thresholds=None, out_levels=None):
		self.lqr = LQRController()
		self.in_t = in_thresholds         # list of 6 threshold lists, or None
		self.levels = out_levels          # int, or None

	def reset(self):
		self.lqr.reset()

	def step(self, q, gyro, target_rpy):
		if self.in_t is not None:
			from wnn.control.optimal import _quat_to_euler, mix_to_motors, _clip
			r, p, y = _quat_to_euler(q)
			x = [r - target_rpy[0], p - target_rpy[1], y - target_rpy[2],
			     gyro[0], gyro[1], gyro[2]]
			xq = np.array([_quantize_to_bins(v, self.in_t[i]) for i, v in enumerate(x)])
			u = -self.lqr.K @ xq
			a = self.lqr.authority
			pwm = mix_to_motors(self.lqr.hover,
				_clip(float(u[0]), -a, a), _clip(float(u[1]), -a, a), _clip(float(u[2]), -a, a))
		else:
			pwm = self.lqr.step(q, gyro, target_rpy)
		if self.levels is not None:
			n = self.levels - 1
			pwm = [round(v * n) / n for v in pwm]
		return pwm


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--disturbance", default="L2D")
	ap.add_argument("--tilt", type=float, default=5.0)
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--episodes", type=int, default=100)
	ap.add_argument("--report-seed", type=int, default=99990101)
	ap.add_argument("--sim-seed", type=int, default=911)
	# The study's spec: bits_per_feature=8, levels_per_motor=16.
	ap.add_argument("--bits", type=int, default=8)
	ap.add_argument("--levels", type=int, default=16)
	ap.add_argument("--out", required=True)
	a = ap.parse_args()

	ec = _episode_config(a)
	pool = fold_pool_seed(a.report_seed, 0)     # the fold-0 episodes, like everything else
	thr = _calibrate_input_thresholds(ec, pool, a.bits)

	variants = {
		"plain": QuantizedLQR(),
		"in-q":  QuantizedLQR(in_thresholds=thr),
		"out-q": QuantizedLQR(out_levels=a.levels),
		"both":  QuantizedLQR(in_thresholds=thr, out_levels=a.levels),
	}
	print(f"# quantized-teacher gap: {a.disturbance}, tilt {a.tilt}°, "
	      f"{a.episodes} ep x {a.steps} steps, pool seed {pool} "
	      f"(bits={a.bits}, levels={a.levels})")
	print(f"{'variant':8} {'stable%':>8} {'err°':>7} {'steady°':>8}")
	results = {}
	for name, ctl in variants.items():
		_, m = eval_closed_loop_reset(make_pid_action_fn(ctl), ctl.reset, ec,
		                              a.episodes, pool)
		st = m["stable_rate"] * 100.0
		er = m["mean_attitude_error_deg"]
		sy = m.get("mean_steady_error_deg")
		results[name] = {"stable": st, "err_deg": er, "steady_deg": sy}
		print(f"{name:8} {st:8.1f} {er:7.2f} {sy if sy is None else round(sy,2)!s:>8}")

	rep_cost = results["plain"]["stable"] - results["both"]["stable"]
	print(f"# representational cost (plain - both): {rep_cost:+.1f} pp stable")
	with open(a.out, "w") as f:
		json.dump({"meta": vars(a) | {"pool_seed": pool,
		           "path_note": "Python rollout path (eval_closed_loop_reset) — "
		                        "internally paired; NOT comparable to Rust-scored "
		                        "study-table numbers across paths."},
		           "results": results}, f, indent=1)
	print(f"# wrote {a.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
