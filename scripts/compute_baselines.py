#!/usr/bin/env python3
"""Score the 5 classical controllers (PID/LQR/MPC/LQI/MPCOF) on the study's
FIXED held-out set, through the same Rust physics engine as the WNN.

The baselines depend only on (disturbance, tilt, report-seed, episodes, steps) —
NOT on substrate / feature / memory-mode / training-seed — so they are computed
ONCE and apply to every cell of the 2x2x2x5 matrix. Writes a JSON table the
results assembler folds into the comparison.

Validated 21/07/2026: with tilt=5°, yaw=5°, L2 the Python _pid_baseline
reproduces the completed yawab run's "vs PID 85.0%/3.96°" to the decimal, and
the Rust stability definition equals score_controllers_cpu:324 — so these
baseline stable% values mean the SAME thing as the WNN stable%.

Usage:
  compute_baselines.py --disturbance L2D --tilt 5.0 --report-seed 99990101 \
      --report-episodes 100 --steps 2000 --out /path/baselines.json
"""
import argparse
import json
import math

import ram_controller as ra
from wnn.control.training import EpisodeConfig, DisturbanceConfig, sample_ics_flat

_NAMES = {0: "PID", 1: "LQR", 2: "MPC", 3: "LQI", 4: "MPCOF"}


def _dist_fields(d, seed):
	"""The 12 disturbance kwargs score_classical_baseline takes, from a
	DisturbanceConfig — mirrors evaluator._dist_packed_fields."""
	return dict(
		dist_enabled=True,
		dist_tau_bias=[float(x) for x in d.tau_bias],
		dist_gust_sigma=float(d.gust_sigma), dist_gust_tau_c=float(d.gust_tau_c),
		dist_motor_asym=[float(x) for x in d.motor_asym],
		dist_gyro_sigma=float(d.gyro_sigma), dist_gyro_bias_walk=float(d.gyro_bias_walk),
		dist_accel_sigma=float(d.accel_sigma), dist_seed=int(seed),
		dist_dropout_prob=float(d.dropout_prob),
		dist_dropout_len_steps=int(d.dropout_len_steps),
		dist_obs_delay_steps=int(d.obs_delay_steps),
		dist_torque_scale_jitter=float(d.torque_scale_jitter))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--disturbance", default="L2D")
	ap.add_argument("--tilt", type=float, default=5.0)
	ap.add_argument("--report-seed", type=int, default=99990101)
	ap.add_argument("--report-episodes", type=int, default=100)
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--stable-deg", type=float, default=5.0)
	ap.add_argument("--out", required=True)
	a = ap.parse_args()

	# The EXACT held-out EpisodeConfig the run builds (phased_ga.py:1875): tilt
	# AND yaw bounded to --tilt, the run's body/yaw-rate defaults.
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=a.steps,
		max_initial_tilt_rad=math.radians(a.tilt),
		max_initial_yaw_rad=math.radians(a.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=DisturbanceConfig.preset(a.disturbance, seed=a.report_seed))
	d = ec.disturbance
	q0, w0 = sample_ics_flat(a.report_seed, a.report_episodes, ec)
	fields = _dist_fields(d, a.report_seed)

	table = {}
	print(f"# classical baselines: {a.disturbance}, tilt={a.tilt}°, "
	      f"seed {a.report_seed}, {a.report_episodes} ep × {a.steps} steps")
	print(f"{'ctrl':7} {'stable%':>8} {'err°':>7} {'steady°':>8}")
	for tid in (0, 1, 2, 3, 4):
		st, err, steady = ra.score_classical_baseline(
			tid, list(q0), list(w0), a.steps, a.stable_deg, **fields)
		table[_NAMES[tid]] = {"stable": st * 100.0, "err_deg": err, "steady_deg": steady}
		print(f"{_NAMES[tid]:7} {st*100:8.1f} {err:7.2f} {steady:8.2f}")

	meta = {"disturbance": a.disturbance, "tilt_deg": a.tilt,
	        "report_seed": a.report_seed, "report_episodes": a.report_episodes,
	        "steps": a.steps, "stable_deg": a.stable_deg}
	import os
	os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
	json.dump({"meta": meta, "baselines": table}, open(a.out, "w"), indent=2)
	print(f"# wrote {a.out}")


if __name__ == "__main__":
	main()
