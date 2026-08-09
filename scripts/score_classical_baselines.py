#!/usr/bin/env python3
"""Score the five CLASSICAL teachers (PID/LQR/MPC/LQI/MPCOF) closed-loop on the
committee protocol, so the WNN table has its classical column.

COMPARABILITY IS THE WHOLE POINT. Same physics engine (score_classical_baseline is
the Rust sim the WNN rows fly), same episode ICs (draw_ics on the report seed — the
exact pools the committee rows used, so classical-vs-committee is common random
numbers), same L4C disturbance resolved the same way (disturbance_stream keyed on the
report seed), same airframe INCLUDING the firmware PID cascade (af_pid_* from
airframe_kwargs — omitting it silently scores the legacy synthetic-plant PID, the
L2 wrong-aircraft trap).

Reuses ensemble_teachers' episode_config/draw_ics via import — one implementation,
per the no-duplicates rule.

Usage:
  PYTHONPATH=src/wnn python scripts/score_classical_baselines.py \
      --seeds 99990101,...  --disturbance L4C --airframe cf21_brushless \
      [--episodes 100] [--steps 2000] [--teachers pid,lqr,mpc,lqi,mpcof]
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ensemble_teachers as et   # episode_config, draw_ics — shared, not copied

from wnn.control import _accel as ra
from wnn.control.evaluator import disturbance_stream

# Teacher::from_id (dagger_train.rs): 0=PID 1=LQR 2=MPC 3=LQI 4=MPCOF.
TEACHER_IDS = {"pid": 0, "lqr": 1, "mpc": 2, "lqi": 3, "mpcof": 4}


def parse_args():
	ap = argparse.ArgumentParser(description="Classical-teacher closed-loop baselines")
	ap.add_argument("--seeds", required=True, help="comma list of report seeds")
	ap.add_argument("--teachers", default="pid,lqr,mpc,lqi,mpcof")
	ap.add_argument("--episodes", type=int, default=100)
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--tilt", type=float, default=5.0)
	ap.add_argument("--body-rate", type=float, default=0.5)
	ap.add_argument("--yaw-rate", type=float, default=0.3)
	ap.add_argument("--disturbance", default="OFF")
	ap.add_argument("--airframe", default="")
	return ap.parse_args()


def main():
	args = parse_args()
	ec = et.episode_config(args)
	dist = getattr(ec, "disturbance", None)
	af_kw = ec.airframe_kwargs()   # af_* sim fields AND af_pid_* cascade — the
	                               # classical entry point takes both.
	seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
	print(f"Classical baselines: {args.teachers} | {len(seeds)} seeds × "
	      f"{args.episodes} eps × {args.steps} steps | disturbance "
	      f"{args.disturbance} | airframe {args.airframe or 'SYNTHETIC'}")
	for name in [t.strip() for t in args.teachers.split(",") if t.strip()]:
		tid = TEACHER_IDS[name]
		rows = []
		for rs in seeds:
			qs, oms = et.draw_ics(rs, args.episodes, ec)
			if dist is None:
				dkw = dict(dist_enabled=False)
			else:
				dseed, asym = disturbance_stream(dist, rs)
				dkw = dict(dist_enabled=True,
				           dist_tau_bias=[float(x) for x in dist.tau_bias],
				           dist_gust_sigma=float(dist.gust_sigma),
				           dist_gust_tau_c=float(dist.gust_tau_c),
				           dist_motor_asym=[float(x) for x in asym],
				           dist_gyro_sigma=float(dist.gyro_sigma),
				           dist_gyro_bias_walk=float(dist.gyro_bias_walk),
				           dist_accel_sigma=float(dist.accel_sigma),
				           dist_seed=dseed,
				           dist_dropout_prob=float(dist.dropout_prob),
				           dist_dropout_len_steps=int(dist.dropout_len_steps),
				           dist_obs_delay_steps=int(dist.obs_delay_steps),
				           dist_torque_scale_jitter=float(dist.torque_scale_jitter))
			stable, err_deg, steady_deg = ra.score_classical_baseline(
				tid, qs, oms, args.steps, 5.0, **dkw, **af_kw)
			rows.append((stable * 100.0, err_deg, steady_deg))
		def ms(i):
			v = [r[i] for r in rows]
			return statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0
		(sm, ss), (em, es), (tm, ts) = ms(0), ms(1), ms(2)
		per = "  ".join(f"{r[0]:.0f}" for r in rows)
		print(f"  [{name:6s}] stable {sm:5.1f}±{ss:4.1f}%  err {em:.2f}±{es:.2f}°  "
		      f"steady {tm:.2f}±{ts:.2f}°   (per-seed: {per})", flush=True)


if __name__ == "__main__":
	main()
