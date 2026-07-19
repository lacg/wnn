#!/usr/bin/env python3
"""Student-centric sensor-degradation counter (Phase-4 state-pressure, part 2).

The teacher-centric counter (state_pressure_counter.py) bucketed frames by an
observation derived from the CLEAN quaternion — so it was structurally BLIND to
D5 dropout / D6 latency, which corrupt only the STUDENT's sensor channel (the
teacher label always sees privileged clean q). Its finding: dynamics
disturbance alone is only weakly non-Markovian (~1.3% PWM spread).

This counter flips the bucketing to the STUDENT's side:

  * bucket = the quantized CORRUPTED observation the student actually receives
    — (gyro, accel) from sim.read_imu(), i.e. after the D4-noise → D6-latency →
    D5-freeze pipeline (controller.rs imu_observed). This is exactly the
    9-feature WnnController input layout minus the constant target.
  * label  = the teacher PWM at that frame — teacher.step(clean q, observed
    gyro, target), the verbatim DAGGER training target (dagger_train.rs:660).

Within-bucket label spread = the irreducible ambiguity a MEMORYLESS student
faces: at a fixed corrupted obs, the true optimal action varies because the
obs no longer pins down the true state. That variance is precisely what
recurrent state (filtering / dead-reckoning) must recover:

  * spread ≈ 0   → corrupted obs still determines the action → stateless OK.
  * spread > tau → same corrupted obs needs materially different actions →
    the student NEEDS state. Also reported: within-bucket spread of the TRUE
    attitude (deg) — how much state the obs fails to pin down.

Pressure sources this separates (the teacher-centric counter could not):
  D4 noise-only  → filter-pressure (can't average noise without memory)
  D6 latency     → prediction-pressure (obs is stale; must extrapolate)
  D5 freeze      → dead-reckoning-pressure (obs is FROZEN; must integrate)

Self-validation: OFF should reproduce ~0 spread (clean obs ⇒ Markovian, the
Phase-3 finding). The rollout (AttitudeSim + AttitudePidRs) is all Rust;
Python only tallies.
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

ram = _ctl  # facade re-exports AttitudeSim/AttitudePidRs/ABI_VERSION (ABI-checked)


def _pid_teacher():
	"""The canonical DAGGER PID teacher (mirrors optimal.rs pid_default_teacher)."""
	return ram.AttitudePidRs(1.2, 0.05, 0.30, 0.5, 0.6, 0.02, 0.20, 0.5, 0.5, 0.4, 0.001)


def _level_config(level: str) -> DisturbanceConfig | None:
	"""Named preset (OFF/L1/L2/L3/L2D/L3D) OR a synthetic isolation level that
	turns on ONE sensor-degradation channel at L2D's calibrated magnitude:
	  D4  = sensor noise only (L2's gyro/accel sigmas, no torque terms)
	  D5  = dropout/freeze only    D6 = latency only    D56 = both, no noise
	Isolation levels have CLEAN dynamics — any spread they show is pure
	observation-channel pressure."""
	lv = level.strip().upper()
	noise = dict(gyro_sigma=0.030, gyro_bias_walk=0.003, accel_sigma=0.30)  # L2's D4
	iso = {
		"D4": DisturbanceConfig(seed=911, **noise),
		"D5": DisturbanceConfig(seed=911, dropout_prob=0.002, dropout_len_steps=20),
		"D6": DisturbanceConfig(seed=911, obs_delay_steps=2),
		"D56": DisturbanceConfig(seed=911, dropout_prob=0.002, dropout_len_steps=20,
		                         obs_delay_steps=2),
	}
	if lv in iso:
		return iso[lv]
	return DisturbanceConfig.preset(lv, seed=911)  # None for OFF


def _bucket(gyro, accel, yaw, bin_rate: float, bin_accel: float, bin_att: float) -> tuple:
	"""Quantize the STUDENT's observation — the corrupted (gyro, accel) pair
	read_imu() returned, which is ALL the attitude information the memoryless
	controller gets (feature layout controller.rs:80). The clean quaternion is
	deliberately EXCLUDED — the student never sees it.

	yaw is None for the faithful 9-feature student (gravity is yaw-invariant ⇒
	the base student is structurally YAW-BLIND; its within-bucket spread then
	includes yaw-unobservability pressure as a constant background). Pass clean
	yaw (--obs-yaw) to emulate an obs_yaw_err-equipped student and isolate pure
	sensor-DEGRADATION pressure."""
	key = (
		round(gyro[0] / bin_rate), round(gyro[1] / bin_rate), round(gyro[2] / bin_rate),
		round(accel[0] / bin_accel), round(accel[1] / bin_accel), round(accel[2] / bin_accel),
	)
	return key if yaw is None else key + (round(yaw / bin_att),)


def _rollout_level(level: str, args) -> dict:
	"""Fly PID-on-itself under `level`; per hover-band frame collect
	bucket(corrupted obs) -> list of (teacher PWM, true euler)."""
	ec = EpisodeConfig(
		steps_per_episode=args.steps,
		max_initial_tilt_rad=np.deg2rad(args.init_tilt_deg),
		max_initial_yaw_rad=np.deg2rad(args.init_tilt_deg),
		max_initial_body_rate=args.init_rate,
	)
	dist = _level_config(level)
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
			gyro, accel = sim.read_imu()            # CORRUPTED — what the student sees
			q = sim.quaternion                       # clean — teacher privilege + gating
			pwm = pid.step(q, gyro, target)          # the DAGGER label at this frame
			roll, pitch, yaw = _quat_to_euler(q)
			att_deg = np.rad2deg(max(abs(roll), abs(pitch), abs(yaw)))
			# Hover-band gate on TRUE attitude (deployment regime; a selection
			# criterion, not an observation — the student can't compute it).
			if att_deg <= args.hover_band_deg:
				yaw_key = yaw if args.obs_yaw else None
				buckets[_bucket(gyro, accel, yaw_key, args.bin_rate, args.bin_accel,
				                args.bin_att)].append((tuple(pwm), (roll, pitch, yaw)))
				flew_steps += 1
			sim.step(pwm)
			total_steps += 1
	return {"buckets": buckets, "flew_steps": flew_steps, "total_steps": total_steps}


def _analyze(res: dict, tau: float) -> dict:
	"""Per multi-frame bucket: action spread = max per-motor (max-min) of the
	teacher labels (conflict if > tau, same rule as split_tau); attitude
	ambiguity = max per-euler-axis (max-min) of the TRUE attitude (deg) — the
	state uncertainty the corrupted obs leaves behind."""
	spreads, att_ambigs = [], []
	multi = conflict = 0
	for _, frames in res["buckets"].items():
		if len(frames) < 2:
			continue
		multi += 1
		pwms = np.asarray([f[0] for f in frames])         # (n, 4)
		spread = float((pwms.max(axis=0) - pwms.min(axis=0)).max())
		spreads.append(spread)
		if spread > tau:
			conflict += 1
		eulers = np.asarray([f[1] for f in frames])       # (n, 3) rad
		att_ambigs.append(float(np.rad2deg((eulers.max(axis=0) - eulers.min(axis=0)).max())))
	spreads = np.asarray(spreads) if spreads else np.zeros(1)
	att_ambigs = np.asarray(att_ambigs) if att_ambigs else np.zeros(1)
	return {
		"frames": res["flew_steps"], "total": res["total_steps"],
		"buckets": len(res["buckets"]), "multi": multi, "conflict": conflict,
		"conflict_rate": (conflict / multi) if multi else 0.0,
		"mean_spread": float(spreads.mean()), "p95_spread": float(np.percentile(spreads, 95)),
		"max_spread": float(spreads.max()),
		"mean_att_ambig": float(att_ambigs.mean()),
		"p95_att_ambig": float(np.percentile(att_ambigs, 95)),
	}


def _print_row(level: str, a: dict) -> None:
	print(f"{level:<6} {a['frames']:>8} {a['buckets']:>8} {a['multi']:>7} {a['conflict']:>8} "
	      f"{a['conflict_rate']*100:>8.1f}% {a['mean_spread']:>9.4f} {a['p95_spread']:>8.4f} "
	      f"{a['max_spread']:>8.4f} {a['mean_att_ambig']:>8.3f}° {a['p95_att_ambig']:>7.3f}°")


def _print_verdict(rows: dict) -> None:
	"""Each level vs OFF: does the corrupted-obs channel create enough label
	ambiguity that a stateless student materially loses fidelity?"""
	base = rows.get("OFF")
	if base is None:
		return
	print("\n[verdict] within-bucket TEACHER-LABEL spread at fixed CORRUPTED student obs, vs clean (OFF):")
	for level, a in rows.items():
		if level == "OFF":
			continue
		dmean = a["mean_spread"] - base["mean_spread"]
		drate = (a["conflict_rate"] - base["conflict_rate"]) * 100
		needs = "STATE NEEDED" if (a["conflict_rate"] > 0.02 and drate > 1.0) else "marginal/none"
		print(f"  {level}: conflict_rate {base['conflict_rate']*100:.1f}% → {a['conflict_rate']*100:.1f}% "
		      f"(+{drate:.1f}pp), mean_spread {base['mean_spread']:.4f} → {a['mean_spread']:.4f} "
		      f"(+{dmean:.4f}), att_ambig p95 {a['p95_att_ambig']:.2f}°  ⇒ {needs}")


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--levels", type=str, default="OFF,D4,D5,D6,D56,L2,L2D",
	                help="comma list: presets OFF/L1/L2/L3/L2D/L3D + isolation D4/D5/D6/D56")
	ap.add_argument("--episodes", type=int, default=80)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--init-tilt-deg", type=float, default=10.0)
	ap.add_argument("--init-rate", type=float, default=0.3)
	ap.add_argument("--hover-band-deg", type=float, default=8.0,
	                help="only count frames whose TRUE attitude error is within this band")
	ap.add_argument("--bin-rate", type=float, default=0.05, help="gyro bucket size (rad/s)")
	ap.add_argument("--bin-accel", type=float, default=0.25,
	                help="accel bucket size (m/s²; 0.25 ≈ g·0.025rad ≈ 1.5° of tilt signal)")
	ap.add_argument("--bin-att", type=float, default=0.02,
	                help="yaw bucket size (rad) when --obs-yaw is set")
	ap.add_argument("--obs-yaw", action="store_true",
	                help="add clean yaw to the bucket (emulates obs_yaw_err student; isolates "
	                     "degradation pressure from the base student's yaw-blindness)")
	ap.add_argument("--tau", type=float, default=0.1, help="PWM-spread conflict threshold (matches split_tau)")
	ap.add_argument("--seed", type=int, default=31337002)
	args = ap.parse_args()

	print(f"[sensor-degradation counter] ABI={ram.ABI_VERSION} teacher=pid "
	      f"episodes={args.episodes} steps={args.steps} hover_band={args.hover_band_deg}° "
	      f"bins(rate={args.bin_rate},accel={args.bin_accel}) tau={args.tau} "
	      f"obs_yaw={'ON' if args.obs_yaw else 'off (faithful 9-feature student, yaw-blind)'}")
	print(f"{'level':<6} {'frames':>8} {'buckets':>8} {'multi':>7} {'conflict':>8} "
	      f"{'conf_rate':>9} {'mean_spr':>9} {'p95_spr':>8} {'max_spr':>8} {'att_amb':>9} {'att_p95':>8}")
	rows = {}
	for level in [s.strip() for s in args.levels.split(",") if s.strip()]:
		a = _analyze(_rollout_level(level, args), args.tau)
		rows[level] = a
		_print_row(level, a)
	_print_verdict(rows)


if __name__ == "__main__":
	main()
