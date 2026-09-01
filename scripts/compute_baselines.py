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

from wnn.control.classical_baseline import HoldoutDraw, TeacherFeed, score_all
from wnn.control.evaluator import apply_motor_fault
from wnn.control.training import EpisodeConfig, DisturbanceConfig

_NAMES = {0: "PID", 1: "LQR", 2: "MPC", 3: "LQI", 4: "MPCOF"}

# Rivals first, then the oracle-fed upper bounds (13/08/2026 rule). Both feeds
# are emitted so the table can never be read as a single convention, and the
# per-controller gap between them IS the cost of state estimation.
_FEEDS = (TeacherFeed(use_estimator=True), TeacherFeed(use_estimator=False))


def _score_seed(seed, a):
	"""Score all 5 classical controllers on ONE held-out draw (report-seed).
	The seed drives BOTH the initial-condition draw (sample_ics_flat) and the
	per-episode disturbance stream, so each seed is an independent held-out set.
	Returns {name: (stable%, err°, steady°)}.

	The pool-seeding / stream-XOR / resolved-asymmetry logic that makes this a
	comparator now lives in wnn.control.classical_baseline, so phased_ga's banner
	and this table cannot drift apart again (they did, for 10 days)."""
	# The disturbance must be built the way phased_ga builds it (preset seeded with
	# --sim-seed, 911 by default), NOT with the report seed: the WNN cells fly that
	# configuration, and a baseline flown on a different one is not a comparator.
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=a.steps,
		max_initial_tilt_rad=math.radians(a.tilt),
		max_initial_yaw_rad=math.radians(a.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=DisturbanceConfig.preset(a.disturbance, seed=a.sim_seed),
		# THE VERTICAL PLANT. Off by default, so every table written before
		# 01/09/2026 reproduces bit-identically — translation=False leaves the
		# integrator inert and each attitude-only episode unchanged. With it on,
		# the aircraft can drift and fall while holding attitude, which is the
		# plant every WNN run since 17/08 has flown. A baseline computed without
		# it is NOT a comparator for those runs.
		translation=bool(a.translation),
		max_initial_xy_offset_m=float(a.xy_offset),
		airframe=(None if not a.airframe else
		          __import__('wnn.control.airframe', fromlist=['Airframe'])
		          .Airframe.preset(a.airframe)))
	# Motor-fault experiment: SAME injection as phased_ga (shared helper), or the
	# baseline flies a healthy aircraft against a WNN trained on a broken one.
	if a.motor_fault:
		apply_motor_fault(ec.disturbance, a.motor_fault)
	draw = HoldoutDraw(seed=seed, episodes=a.report_episodes, steps=a.steps,
	                   stable_deg=a.stable_deg, eval_folds=a.eval_folds,
	                   fold_index=a.fold_index)
	out = {}
	for feed in _FEEDS:
		out.update(score_all(ec, draw, feed))
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--disturbance", default="L2D")
	# Airframe: None = the pre-airframe synthetic plant (back-compat). Any
	# name from wnn.control.airframe, which carries the citation.
	ap.add_argument("--airframe", default=None,
	                help="airframe preset (e.g. cf21_brushless); omit for legacy plant")
	ap.add_argument("--translation", action=argparse.BooleanOptionalAction, default=False,
	                help="Integrate vertical translation in the plant, matching phased_ga's "
	                     "--translation. REQUIRED to compare against any WNN run from the "
	                     "altitude regimen; without it the baselines fly a different "
	                     "aircraft. Enables the alt/pos columns. Default OFF, which "
	                     "reproduces every pre-01/09/2026 table bit-identically.")
	ap.add_argument("--xy-offset", type=float, default=0.0,
	                help="Max initial horizontal displacement (m). Mirrors phased_ga's "
	                     "--xy-offset; x/y never leave the origin at 0.0, so pos3d is then "
	                     "the vertical error alone. Requires --translation.")
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

	# Mirrors phased_ga.py:3064. Vertical dynamics need mass, and the synthetic
	# default plant has none — a silent fallback here would produce a table that
	# looks like an altitude comparator and is not one.
	if a.translation and not a.airframe:
		raise SystemExit("--translation requires --airframe: mass is a PLANT parameter "
		                 "and the synthetic default has none.")
	if a.xy_offset > 0.0 and not a.translation:
		raise SystemExit("--xy-offset requires --translation: x/y cannot move without it.")

	seeds = a.report_seeds if a.report_seeds else [a.report_seed]

	# Score every controller on every held-out seed → per-controller triples.
	# score_all returns FIVE fields per controller — (stable%, err°, steady°,
	# alt_m, pos3d_m). The first three were the only ones aggregated until
	# 01/09/2026, so every earlier table silently dropped altitude: the regimen
	# was named for a column the comparison never carried.
	per_seed = {}  # name -> list of 5-tuples, one per seed
	for s in seeds:
		res = _score_seed(s, a)
		for name, tri in res.items():
			per_seed.setdefault(name, []).append(tri)

	def _agg(xs):
		return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)

	def _row(name):
		"""Aggregate one controller's per-seed rows, print it, return the entry."""
		tris = per_seed[name]
		st_m, st_s = _agg([t[0] for t in tris])
		er_m, er_s = _agg([t[1] for t in tris])
		sy_m, sy_s = _agg([t[2] for t in tris])
		entry = {"stable": st_m, "err_deg": er_m, "steady_deg": sy_m,
		         "stable_std": st_s, "err_std": er_s, "steady_std": sy_s,
		         "n_seeds": len(tris),
		         "per_seed": {str(seeds[i]): list(tris[i]) for i in range(len(tris))}}
		line = (f"{name:14} {st_m:6.1f}±{st_s:4.1f}  {er_m:5.2f}±{er_s:4.2f}  "
		        f"{sy_m:5.2f}±{sy_s:4.2f}")
		# alt/pos are meaningless without the vertical plant — z never moves, so
		# emitting 0.000 would read as PERFECT altitude hold. Omit instead.
		if a.translation:
			al_m, al_s = _agg([t[3] for t in tris])
			po_m, po_s = _agg([t[4] for t in tris])
			entry.update(alt_m=al_m, alt_std=al_s, pos3d_m=po_m, pos3d_std=po_s)
			line += f"  {al_m:6.3f}±{al_s:5.3f}  {po_m:6.3f}±{po_s:5.3f}"
		print(line)
		return entry

	table = {}
	print(f"# classical baselines: {a.disturbance}, tilt={a.tilt}°, "
	      f"{len(seeds)} seed(s) {seeds}, {a.report_episodes} ep × {a.steps} steps")
	hdr = f"{'ctrl':14} {'stable%':>13} {'err°':>13} {'steady°':>13}"
	if a.translation:
		hdr += f" {'alt m':>14} {'pos3d m':>14}"
	print(hdr)
	for feed in _FEEDS:
		print("# " + ("RIVALS — estimator-fed: THE comparison" if feed.use_estimator
		               else "oracle-fed — informational upper bound, NOT the comparison"))
		for tid in (0, 1, 2, 3, 4):
			name = feed.label_for(_NAMES[tid])
			table[name] = _row(name)

	meta = {"disturbance": a.disturbance, "tilt_deg": a.tilt,
	        # THE REGIMEN. A file without this key predates 01/09/2026 and is
	        # attitude-only: do NOT quote it beside a WNN run that flew
	        # --translation, and note that such files carry no alt column at all.
	        "translation": bool(a.translation),
	        "xy_offset_m": float(a.xy_offset),
	        "airframe": a.airframe,
	        "report_seed": seeds[0], "report_seeds": seeds,
	        "report_episodes": a.report_episodes,
	        "steps": a.steps, "stable_deg": a.stable_deg,
	        "sim_seed": a.sim_seed,
	        "eval_folds": a.eval_folds, "fold_index": a.fold_index,
	        "motor_fault": a.motor_fault,
	        # Same helper the scorer used, so the recorded pool CANNOT disagree
	        # with the pool the episodes were actually drawn from.
	        "pool_seeds": {str(s): HoldoutDraw(seed=s, episodes=a.report_episodes,
	                                           steps=a.steps, stable_deg=a.stable_deg,
	                                           eval_folds=a.eval_folds,
	                                           fold_index=a.fold_index).pool_seed()
	                       for s in seeds},
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
