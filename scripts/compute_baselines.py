#!/usr/bin/env python3
"""Score the 5 classical controllers (PID/LQR/MPC/LQI/MPCOF) on the study's
FIXED held-out set, through the same Rust physics engine as the WNN.

The baselines depend only on (disturbance, tilt, report-seed, episodes, steps) —
NOT on substrate / feature / memory-mode / training-seed — so they are computed
ONCE and apply to every cell of the 2x2x2x5 matrix. Writes a JSON table the
results assembler folds into the comparison.

The 21/07/2026 validation note here used to say these values were confirmed against
the Python _pid_baseline's "vs PID 85.0%/3.96°". That check was against the WRONG
reference: _pid_baseline is eval_closed_loop_reset, which redraws motor asymmetry
per EPISODE and ignores the disturbance-stream seed, so it is not the twin of the
WNN scorer. Agreeing with it was not evidence of matched conditions. Three separate
mismatches were found on 29/07/2026 and are documented at their fix sites below:
symmetric motor_asym, the un-XOR'd stream seed, and sampling from the report seed
instead of the fold-0 pool. Each moved PID by roughly 10pp.

Usage:
  compute_baselines.py --disturbance L2D --tilt 5.0 --report-seed 99990101 \
      --report-episodes 100 --steps 2000 --out /path/baselines.json
"""
import argparse
import json
import math
import statistics

import ram_controller as ra
from wnn.control.evaluator import apply_motor_fault, disturbance_stream, fold_pool_seed
from wnn.control.training import EpisodeConfig, DisturbanceConfig, sample_ics_flat

_NAMES = {0: "PID", 1: "LQR", 2: "MPC", 3: "LQI", 4: "MPCOF"}


def _score_seed(seed, a):
	"""Score all 5 classical controllers on ONE held-out draw (report-seed).
	The seed drives BOTH the initial-condition draw (sample_ics_flat) and the
	per-episode disturbance stream, so each seed is an independent held-out set.
	Returns {name: (stable%, err°, steady°)}."""
	# The disturbance must be built the way phased_ga builds it (preset seeded with
	# --sim-seed, 911 by default), NOT with the report seed: the WNN cells fly that
	# configuration, and a baseline flown on a different one is not a comparator.
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=a.steps,
		max_initial_tilt_rad=math.radians(a.tilt),
		max_initial_yaw_rad=math.radians(a.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=DisturbanceConfig.preset(a.disturbance, seed=a.sim_seed))
	# Motor-fault experiment: SAME injection as phased_ga (shared helper), or the
	# baseline flies a healthy aircraft against a WNN trained on a broken one.
	if a.motor_fault:
		apply_motor_fault(ec.disturbance, a.motor_fault)
	# The scorer does NOT sample from the report seed when K>1: score_genomes calls
	# _advance_fold first, which swaps in _fold_seeds[fold]. A held-out report builds a
	# fresh evaluator and scores once, so it always lands on fold 0. Sampling from the
	# raw report seed instead — as this did until 29/07/2026 — flies the baselines on
	# episodes no WNN cell ever saw (PID 100.0% vs 89.0% on the canonical seed).
	pool = (seed if a.eval_folds <= 1
	        else fold_pool_seed(seed, a.fold_index))   # K=1 keeps the raw seed
	q0, w0 = sample_ics_flat(pool, a.report_episodes, ec)
	# Mirror evaluator._score_batch (evaluator.py:1236-1237): the stream seed is the
	# preset seed XOR the score seed, and the per-airframe motor-asymmetry draw is
	# resolved FROM that stream seed. Passing the raw d.motor_asym instead — as this
	# script did until 29/07/2026 — flies the baselines on a PERFECTLY SYMMETRIC
	# quadrotor while every WNN cell carries an ~8% weak motor, which is the defect
	# L2D exists to model. Measured cost of that mismatch: PID 97.0% -> 89.0% stable.
	dseed, asym = disturbance_stream(ec.disturbance, pool)
	fields = _dist_fields(ec.disturbance, dseed, asym)
	out = {}
	for tid in (0, 1, 2, 3, 4):
		st, err, steady = ra.score_classical_baseline(
			tid, list(q0), list(w0), a.steps, a.stable_deg, **fields)
		out[_NAMES[tid]] = (st * 100.0, err, steady)
	return out


def _dist_fields(d, seed, motor_asym=None):
	"""The 12 disturbance kwargs score_classical_baseline takes, from a
	DisturbanceConfig — mirrors evaluator._dist_packed_fields.

	motor_asym MUST be the RESOLVED per-airframe draw, not d.motor_asym: the raw
	field is the fixed multiplier (1,1,1,1) and carries none of the asymmetry.
	"""
	return dict(
		dist_enabled=True,
		dist_tau_bias=[float(x) for x in d.tau_bias],
		dist_gust_sigma=float(d.gust_sigma), dist_gust_tau_c=float(d.gust_tau_c),
		dist_motor_asym=[float(x) for x in (motor_asym
		                                    if motor_asym is not None else d.motor_asym)],
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
	# Multi-seed held-out: each seed is an independent held-out draw. The
	# baseline ±SD is therefore TEST-SET variance (same controller, different
	# episode set) — NOT the WNN cells' training-seed variance. Default keeps
	# the single fixed held-out (99990101) for backward compatibility.
	ap.add_argument("--report-seeds", type=int, nargs="+", default=None,
	                help="held-out report-seeds; overrides --report-seed for multi-seed ±SD")
	ap.add_argument("--report-episodes", type=int, default=100)
	ap.add_argument("--steps", type=int, default=2000)
	# MUST match phased_ga's hardcoded DisturbanceConfig.preset(level, seed=911).
	# Change it only if phased_ga changes, or the baselines stop being comparators.
	ap.add_argument("--sim-seed", type=int, default=911,
	                help="disturbance-preset seed; mirrors phased_ga (911)")
	# MUST match the scoring run's --num-eval-folds and the fold the held-out lands on
	# (fold 0: a held-out builds a fresh evaluator and scores exactly once).
	ap.add_argument("--eval-folds", type=int, default=5,
	                help="K used by the scoring run; 1 = legacy raw-seed pool")
	ap.add_argument("--fold-index", type=int, default=0,
	                help="which fold the held-out scored on (held-out is always 0)")
	ap.add_argument("--motor-fault", type=str, default=None,
	                help="'idx:factor' fixed motor fault — MUST match the WNN run's")
	ap.add_argument("--stable-deg", type=float, default=5.0)
	ap.add_argument("--out", required=True)
	a = ap.parse_args()

	seeds = a.report_seeds if a.report_seeds else [a.report_seed]

	# Score every controller on every held-out seed → per-controller triples.
	per_seed = {}  # name -> list of (stable%, err°, steady°), one per seed
	for s in seeds:
		res = _score_seed(s, a)
		for name, tri in res.items():
			per_seed.setdefault(name, []).append(tri)

	def _agg(xs):
		return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)

	table = {}
	print(f"# classical baselines: {a.disturbance}, tilt={a.tilt}°, "
	      f"{len(seeds)} seed(s) {seeds}, {a.report_episodes} ep × {a.steps} steps")
	print(f"{'ctrl':7} {'stable%':>13} {'err°':>13} {'steady°':>13}")
	for tid in (0, 1, 2, 3, 4):
		name = _NAMES[tid]
		tris = per_seed[name]
		st_m, st_s = _agg([t[0] for t in tris])
		er_m, er_s = _agg([t[1] for t in tris])
		sy_m, sy_s = _agg([t[2] for t in tris])
		table[name] = {
			"stable": st_m, "err_deg": er_m, "steady_deg": sy_m,
			"stable_std": st_s, "err_std": er_s, "steady_std": sy_s,
			"n_seeds": len(tris),
			"per_seed": {str(seeds[i]): list(tris[i]) for i in range(len(tris))},
		}
		print(f"{name:7} {st_m:6.1f}±{st_s:4.1f}  {er_m:5.2f}±{er_s:4.2f}  "
		      f"{sy_m:5.2f}±{sy_s:4.2f}")

	meta = {"disturbance": a.disturbance, "tilt_deg": a.tilt,
	        "report_seed": seeds[0], "report_seeds": seeds,
	        "report_episodes": a.report_episodes,
	        "steps": a.steps, "stable_deg": a.stable_deg,
	        "sim_seed": a.sim_seed,
	        "eval_folds": a.eval_folds, "fold_index": a.fold_index,
	        "motor_fault": a.motor_fault,
	        "pool_seeds": {str(s): (s if a.eval_folds <= 1
	                                else fold_pool_seed(s, a.fold_index)) for s in seeds},
	        "conditions_note": "ICs sampled from the fold-index POOL seed (see pool_seeds), "
	                           "NOT the report seed; disturbance stream = sim_seed XOR "
	                           "pool_seed; motor asymmetry is the RESOLVED per-airframe "
	                           "draw. All three match the WNN scorer "
	                           "(evaluator.py fold_pool_seed + :1236-1237), so the "
	                           "baselines fly the same aircraft on the same episodes as "
	                           "the WNN cells. Files written before 29/07/2026 have all "
	                           "three wrong and are NOT comparable to WNN rows.",
	        "variance_note": "baseline ±SD = test-set variance across held-out "
	                         "seeds; WNN cell ±SD = training-seed variance on the "
	                         "fixed held-out (99990101)."}
	import os
	os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
	json.dump({"meta": meta, "baselines": table}, open(a.out, "w"), indent=2)
	print(f"# wrote {a.out}")


if __name__ == "__main__":
	main()
