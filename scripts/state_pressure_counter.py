#!/usr/bin/env python3
"""Teacher-centric split-pressure counter (Phase-4 state-pressure instrument).

Answers "does a persistent disturbance make the drone task NEED recurrent
state?" WITHOUT the GA-search confound (which needs a flying controller to
detect conflicts — see project_controller_state_pressure_measurement).

Method: fly the PID teacher on ITSELF (so the state distribution is realistic,
near-hover) under a disturbance level, record at every step the pair
(student-observable bucket, teacher PWM action). Then bucket frames by the
INSTANTANEOUS observation a memoryless policy would see (quantized attitude +
body rates) and measure the within-bucket spread of the teacher's action:

  * within-bucket spread ≈ 0  → the same observation maps to one action →
    a memoryless (stateless) lookup CAN represent the teacher → no state needed.
  * within-bucket spread > tau → the SAME observation needs DIFFERENT actions
    (the difference is the integral/history) → the task is non-Markovian →
    recurrent STATE is required.

Self-validation: OFF (clean sim) should reproduce Phase-3's finding (~0 spread,
no state needed). L2/L3/L2D should show the spread the WNN's recurrent layer
would have to capture. Compares levels side by side.

The rollout (AttitudeSim + AttitudePidRs step) is all Rust; Python only tallies.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src/wnn")
from wnn.control import _accel as _ctl  # noqa: E402  (facade asserts ABI)
from wnn.control.training import (  # noqa: E402
	DisturbanceConfig, EpisodeConfig, apply_disturbance, sample_ics_flat,
)
from wnn.control.pid import _quat_to_euler  # noqa: E402

ram = _ctl  # the _accel facade re-exports AttitudeSim/AttitudePidRs/ABI_VERSION (ABI-checked at import)


def _pid_teacher():
	"""The canonical DAGGER PID teacher (mirrors optimal.rs pid_default_teacher)."""
	return ram.AttitudePidRs(1.2, 0.05, 0.30, 0.5, 0.6, 0.02, 0.20, 0.5, 0.5, 0.4, 0.001)


def _bucket(q, gyro, bin_att: float, bin_rate: float) -> tuple:
	"""Quantize the INSTANTANEOUS observation a memoryless policy sees:
	attitude (roll/pitch/yaw, rad) + body rates (rad/s). The teacher's integral
	is deliberately EXCLUDED — it is the hidden state we are testing for."""
	roll, pitch, yaw = _quat_to_euler(q)
	return (
		round(roll / bin_att), round(pitch / bin_att), round(yaw / bin_att),
		round(gyro[0] / bin_rate), round(gyro[1] / bin_rate), round(gyro[2] / bin_rate),
	)


def _rollout_level(level: str, args) -> dict:
	"""Fly PID-on-itself for `episodes` episodes under `level`; collect
	(bucket -> list of teacher PWM vectors)."""
	ec = EpisodeConfig(
		steps_per_episode=args.steps,
		max_initial_tilt_rad=np.deg2rad(args.init_tilt_deg),
		max_initial_yaw_rad=np.deg2rad(args.init_tilt_deg),
		max_initial_body_rate=args.init_rate,
	)
	dist = None if level.upper() in ("OFF", "NONE", "") else DisturbanceConfig.preset(level, seed=911)
	q0, omega0 = sample_ics_flat(args.seed, args.episodes, ec)
	sim = ram.AttitudeSim()
	pid = _pid_teacher()
	target = (0.0, 0.0, 0.0)
	rng = np.random.default_rng(args.seed + 7)

	buckets: dict[tuple, list] = defaultdict(list)
	flew_steps = total_steps = 0
	for ep in range(args.episodes):
		iq = q0[4 * ep:4 * ep + 4]
		iw = omega0[3 * ep:3 * ep + 3]
		sim.reset(q=list(iq), omega=list(iw))
		if dist is None:
			sim.clear_disturbance()
		else:
			apply_disturbance(sim, dist, rng)
		pid.reset()
		for _ in range(args.steps):
			if sim.is_unstable():
				break
			gyro, accel = sim.read_imu()
			q = sim.quaternion
			pwm = pid.step(q, gyro, target)         # teacher action at this frame
			# Only count frames near hover (post-transient) if requested — that is
			# where a deployed controller lives and where the integral matters.
			roll, pitch, yaw = _quat_to_euler(q)
			att_deg = np.rad2deg(max(abs(roll), abs(pitch), abs(yaw)))
			if att_deg <= args.hover_band_deg:
				buckets[_bucket(q, gyro, args.bin_att, args.bin_rate)].append(tuple(pwm))
				flew_steps += 1
			sim.step(pwm)
			total_steps += 1
	return {"buckets": buckets, "flew_steps": flew_steps, "total_steps": total_steps}


def _analyze(res: dict, tau: float) -> dict:
	"""For each multi-frame bucket, spread = max per-motor (max-min) across its
	frames. Conflict = spread > tau (same obs, materially different action)."""
	buckets = res["buckets"]
	spreads = []
	multi = conflict = 0
	for _, pwms in buckets.items():
		if len(pwms) < 2:
			continue
		multi += 1
		arr = np.asarray(pwms)                       # (n, 4)
		spread = float((arr.max(axis=0) - arr.min(axis=0)).max())
		spreads.append(spread)
		if spread > tau:
			conflict += 1
	spreads = np.asarray(spreads) if spreads else np.zeros(1)
	return {
		"frames": res["flew_steps"], "total": res["total_steps"],
		"buckets": len(buckets), "multi": multi, "conflict": conflict,
		"conflict_rate": (conflict / multi) if multi else 0.0,
		"mean_spread": float(spreads.mean()), "p95_spread": float(np.percentile(spreads, 95)),
		"max_spread": float(spreads.max()),
	}


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--levels", type=str, default="OFF,L2",
	                help="comma list of disturbance levels to compare (OFF,L1,L2,L3,L2D,L3D)")
	ap.add_argument("--episodes", type=int, default=80)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--init-tilt-deg", type=float, default=10.0)
	ap.add_argument("--init-rate", type=float, default=0.3)
	ap.add_argument("--hover-band-deg", type=float, default=8.0,
	                help="only count frames whose attitude error is within this band (post-transient hover)")
	ap.add_argument("--bin-att", type=float, default=0.02, help="attitude bucket size (rad, ~1.15°)")
	ap.add_argument("--bin-rate", type=float, default=0.05, help="body-rate bucket size (rad/s)")
	ap.add_argument("--tau", type=float, default=0.1, help="PWM-spread conflict threshold (matches split_tau)")
	ap.add_argument("--seed", type=int, default=31337002)
	args = ap.parse_args()

	print(f"[state-pressure counter] ABI={ram.ABI_VERSION} teacher=pid "
	      f"episodes={args.episodes} steps={args.steps} hover_band={args.hover_band_deg}° "
	      f"bins(att={args.bin_att}rad,rate={args.bin_rate}) tau={args.tau}")
	print(f"{'level':<6} {'frames':>8} {'buckets':>8} {'multi':>7} {'conflict':>8} "
	      f"{'conf_rate':>9} {'mean_spr':>9} {'p95_spr':>8} {'max_spr':>8}")
	rows = {}
	for level in [s.strip() for s in args.levels.split(",") if s.strip()]:
		a = _analyze(_rollout_level(level, args), args.tau)
		rows[level] = a
		print(f"{level:<6} {a['frames']:>8} {a['buckets']:>8} {a['multi']:>7} {a['conflict']:>8} "
		      f"{a['conflict_rate']*100:>8.1f}% {a['mean_spread']:>9.4f} {a['p95_spread']:>8.4f} {a['max_spread']:>8.4f}")

	# Verdict: compare each disturbed level to OFF.
	if "OFF" in rows:
		base = rows["OFF"]
		print("\n[verdict] within-bucket teacher-action spread vs clean (OFF):")
		for level, a in rows.items():
			if level == "OFF":
				continue
			dmean = a["mean_spread"] - base["mean_spread"]
			drate = (a["conflict_rate"] - base["conflict_rate"]) * 100
			needs = "STATE NEEDED" if (a["conflict_rate"] > 0.02 and drate > 1.0) else "marginal/none"
			print(f"  {level}: conflict_rate {base['conflict_rate']*100:.1f}% → {a['conflict_rate']*100:.1f}% "
			      f"(+{drate:.1f}pp), mean_spread {base['mean_spread']:.4f} → {a['mean_spread']:.4f} "
			      f"(+{dmean:.4f})  ⇒ {needs}")


if __name__ == "__main__":
	main()
