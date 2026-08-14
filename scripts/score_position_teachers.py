#!/usr/bin/env python3
"""Measure the SCOPE C STAGE 2 full-state teacher against chunk B's bar.

THE QUESTION: can the classical teacher hover to a POINT? DAgger cannot teach
position control from an expert that has none, so this must pass before the WNN
is given position features at all (docs/scope_c_stage2_chunk_b_teacher.md).

THE BAR (pre-registered, from the scope C spec): position error comparable to
Molchanov et al. 2019's classical baseline — 0.11 / 0.19 / 0.21 / 0.24 m across
their configurations. This is the TEACHER's bar, not the WNN's.

WHAT IS AND IS NOT MEASURED HERE, stated plainly because it is easy to overclaim:
  * Episodes START displaced from the target and the teacher must fly back, so
    this is position HOLD, not hover-in-place.
  * Attitude comes from a Mahony estimate of the noisy IMU under --estimator
    (the 13/08/2026 rule). POSITION and VELOCITY are handed to the teacher
    directly — there is no GPS/vision model in this simulator, so a position
    estimator is a separate question. That assumption is DISCLOSED, not hidden.
  * The teacher is a CASCADE (position → tilt reference → attitude teacher →
    motors) and must be reported as one. The WNN is the monolithic controller.

Usage:
  PYTHONPATH=src/wnn python scripts/score_position_teachers.py \
      --airframe cf21_brushless --episodes 50 --steps 6000 --estimator
"""
import argparse
import math
import statistics

from wnn.control import _accel as rc
from wnn.control.airframe import Airframe
from wnn.control.training import DisturbanceConfig

TEACHERS = {"pid": 0, "lqr": 1, "mpc": 2, "lqi": 3, "mpcof": 4}

# Molchanov et al. 2019 (arXiv:1903.04628), Euclidean position error, metres.
MOLCHANOV_M = (0.11, 0.19, 0.21, 0.24)


def parse_args():
	ap = argparse.ArgumentParser(description="Full-state (position-cascade) teacher scores")
	ap.add_argument("--airframe", default="cf21_brushless")
	ap.add_argument("--teachers", default="pid,lqr,mpc,lqi,mpcof")
	ap.add_argument("--episodes", type=int, default=50)
	ap.add_argument("--steps", type=int, default=6000, help="1 kHz steps; 6000 = 6 s")
	ap.add_argument("--seed", type=int, default=99990101)
	ap.add_argument("--offset", type=float, default=1.0,
	                help="max initial |offset| per axis, metres")
	ap.add_argument("--tilt", type=float, default=5.0, help="max initial tilt, degrees")
	ap.add_argument("--disturbance", default="L4C")
	ap.add_argument("--estimator", action=argparse.BooleanOptionalAction, default=True,
	                help="attitude from a Mahony filter on the noisy IMU (default ON)")
	ap.add_argument("--pos-omega-n", type=float, default=1.0)
	ap.add_argument("--pos-zeta", type=float, default=1.0)
	ap.add_argument("--max-tilt-deg", type=float, default=30.0)
	return ap.parse_args()


def draw_episodes(a):
	"""(init_qs, init_omegas, init_p) flat lists — small random attitude AND a
	position offset, so every episode is a genuine fly-back."""
	import numpy as np
	rng = np.random.default_rng(a.seed)
	qs, oms, ps = [], [], []
	tilt = math.radians(a.tilt)
	for _ in range(a.episodes):
		# Small random attitude via axis-angle, the same shape sample_ics uses.
		ax = rng.normal(size=3)
		ax /= max(float(np.linalg.norm(ax)), 1e-9)
		ang = float(rng.uniform(-tilt, tilt))
		h = ang / 2.0
		qs += [math.cos(h), *(float(x) * math.sin(h) for x in ax)]
		oms += [float(v) for v in rng.uniform(-0.2, 0.2, 3)]
		ps += [float(v) for v in rng.uniform(-a.offset, a.offset, 3)]
	return qs, oms, ps


def dist_kwargs(a):
	d = DisturbanceConfig.preset(a.disturbance, seed=911)
	return dict(
		dist_enabled=True,
		dist_tau_bias=[float(x) for x in d.tau_bias],
		dist_gust_sigma=float(d.gust_sigma), dist_gust_tau_c=float(d.gust_tau_c),
		dist_motor_asym=[float(x) for x in d.motor_asym],
		dist_gyro_sigma=float(d.gyro_sigma), dist_gyro_bias_walk=float(d.gyro_bias_walk),
		dist_accel_sigma=float(d.accel_sigma), dist_seed=int(a.seed),
		dist_dropout_prob=float(d.dropout_prob),
		dist_dropout_len_steps=int(d.dropout_len_steps),
		dist_obs_delay_steps=int(d.obs_delay_steps),
		dist_torque_scale_jitter=float(d.torque_scale_jitter))


def airframe_kwargs(af):
	"""The af_* plant fields score_position_teacher takes (no PID cascade: the
	cascade gains ride along via the EpisodeConfig path elsewhere; here the
	teacher is built from the plant)."""
	return dict(
		af_arm_length=float(af.arm_length), af_k_thrust=float(af.k_thrust),
		af_k_drag=float(af.k_drag), af_inertia=[float(x) for x in af.inertia],
		af_gravity=float(af.gravity), af_dt=0.001)


def main():
	a = parse_args()
	af = Airframe.preset(a.airframe)
	qs, oms, ps = draw_episodes(a)
	kw = {**dist_kwargs(a), **airframe_kwargs(af),
	      "use_estimator": bool(a.estimator)}

	print(f"# full-state teacher (position cascade) — {a.airframe}, {a.disturbance}, "
	      f"{a.episodes} eps × {a.steps} steps ({a.steps/1000:.0f} s), "
	      f"start offset ≤ {a.offset} m/axis, attitude "
	      f"{'ESTIMATOR-fed' if a.estimator else 'ORACLE-fed'}")
	print(f"# position loop: omega_n={a.pos_omega_n} zeta={a.pos_zeta} "
	      f"max_tilt={a.max_tilt_deg}°   |   BAR: Molchanov "
	      f"{'/'.join(f'{m:.2f}' for m in MOLCHANOV_M)} m")
	print(f"{'teacher':9} {'pos_err m':>10} {'final m':>9} {'stable%':>8} "
	      f"{'att err°':>9} {'steady°':>8}  verdict")

	start = statistics.mean([
		math.dist((0, 0, 0), ps[i * 3:i * 3 + 3]) for i in range(a.episodes)])
	print(f"{'(start)':9} {start:10.3f} {'—':>9} {'—':>8} {'—':>9} {'—':>8}"
	      f"  mean initial displacement")

	for name in [t.strip() for t in a.teachers.split(",") if t.strip()]:
		pos_err, final_err, stable, att_err, steady = rc.score_position_teacher(
			TEACHERS[name], qs, oms, ps, float(af.mass), a.steps,
			pos_omega_n=a.pos_omega_n, pos_zeta=a.pos_zeta,
			pos_max_tilt_deg=a.max_tilt_deg, **kw)
		# The bar is on the SETTLED error, so final position error is the one to
		# read against Molchanov; mean includes the fly-back transient.
		verdict = ("PASS" if final_err <= max(MOLCHANOV_M)
		           else "over bar" if final_err < 1.0 else "FAIL")
		print(f"{name:9} {pos_err:10.3f} {final_err:9.3f} {stable*100:8.1f} "
		      f"{att_err:9.2f} {steady:8.2f}  {verdict}")

	print("\n# DISCLOSURE: cascade (position → tilt ref → attitude teacher → motors),")
	print("#   position/velocity handed over directly (no GPS/vision model in this sim).")


if __name__ == "__main__":
	main()
