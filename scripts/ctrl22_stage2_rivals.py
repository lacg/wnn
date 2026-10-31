#!/usr/bin/env python3
"""CTRL-22 — the stage-2 RIVAL BAR: the classical controllers scored on the HORIZONTAL regime.

Nobody had scored PID/LQR/MPC/LQI/MPCOF with a horizontal start offset, so stage 2
(CTRL-8) had no rival bar. This scores them through the SAME comparator every WNN
rival row uses (classical_baseline.score_all -> score_classical_baseline, the
s2_init_x/y path), on the SAME episodes the WNN scorer flies: the regime is
phased_ga's OWN parser + episode_config_from_args on the anchor's exact argv
(--run-args-file), with only --xy-offset and --steps overridden. Estimator-fed
(the 13/08 compare-estimator rule). Score-only — no training, nothing written back.

Grid: xy offset {0 (stage-1 control row), --xy} x steps {2000, 6000} x the run's
report seeds; per teacher mean +/- SD over seeds of stable % / err deg / steady deg
/ alt m / pos (3-D) m. Also the mean initial horizontal displacement (the start bar
a position-holding controller must beat).

	PYTHONPATH=src/wnn python scripts/ctrl22_stage2_rivals.py \\
	    --run-args-file experiments/crn_noise_probe/ARGS_<anchor>.txt --out experiments/ctrl22/rivals.json
"""
import argparse
import json
import math
import shlex
import statistics
import sys
import time


def _parse():
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--run-args-file", required=True, help="the anchor's phased_ga argv ('#' lines ignored)")
	ap.add_argument("--xy", type=float, default=0.5, help="stage-2 --xy-offset (m, per axis)")
	ap.add_argument("--steps", type=int, nargs="+", default=[2000, 6000])
	ap.add_argument("--out", required=True)
	return ap.parse_args()


def _argv(path):
	with open(path) as f:
		return shlex.split(" ".join(l for l in f.read().splitlines() if not l.lstrip().startswith("#")))


def _regime(base_argv, xy, steps):
	"""(args, ec) exactly as phased_ga would build them, with xy/steps overridden (argparse last-wins)."""
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args
	extra = ["--steps", str(steps), "--xy-offset", str(xy)]
	if xy <= 0.0:
		# ABI 32: the tilt reference needs the horizontal channel; the xy-0 control
		# row is scored against level (with no horizontal start the reference IS level).
		extra += ["--att-err-ref", "level"]
	args = build_arg_parser().parse_args(base_argv + extra)
	return args, episode_config_from_args(args)


def _start_xy(ec, seed, episodes, folds):
	"""Mean initial horizontal displacement of the draw (same sampler as the scorer)."""
	from wnn.control.classical_baseline import HoldoutDraw
	from wnn.control.training import sample_horizontal_ics_flat
	if ec.max_initial_xy_offset_m <= 0.0:
		return 0.0
	pool = HoldoutDraw(seed=seed, episodes=episodes, steps=1, eval_folds=folds).pool_seed()
	x0, y0 = sample_horizontal_ics_flat(pool, episodes, ec)
	return statistics.mean(math.hypot(x, y) for x, y in zip(x0, y0))


def _cell(base_argv, xy, steps):
	"""{teacher: [row per report seed]} + start displacement per seed, for one (xy, steps)."""
	from wnn.control.classical_baseline import HoldoutDraw, TeacherFeed, score_all
	args, ec = _regime(base_argv, xy, steps)
	folds = int(args.num_eval_folds)
	rows, starts = {}, []
	for seed in args.report_seeds:
		draw = HoldoutDraw(seed=int(seed), episodes=int(args.report_episodes),
		                   steps=ec.steps_per_episode, eval_folds=folds)
		for name, r in score_all(ec, draw, TeacherFeed(use_estimator=True)).items():
			rows.setdefault(name, []).append(r)
		starts.append(_start_xy(ec, int(seed), int(args.report_episodes), folds))
	return rows, starts, args


def _ms(v):
	return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


def main():
	a = _parse()
	base = _argv(a.run_args_file)
	out = {"run_args_file": a.run_args_file, "feed": "estimator", "cells": []}
	cols = ("stable_pct", "err_deg", "steady_deg", "alt_m", "pos3d_m")
	for xy in (0.0, a.xy):
		for steps in a.steps:
			t0 = time.time()
			rows, starts, args = _cell(base, xy, steps)
			cell = {"xy_offset_m": xy, "steps": steps, "seeds": [int(s) for s in args.report_seeds],
			        "episodes": int(args.report_episodes), "start_xy_m": _ms(starts), "teachers": {}}
			print(f"\n== xy {xy} m · {steps} steps ({steps / 1000:.0f} s) · {len(starts)} seeds x "
			      f"{args.report_episodes} ep · start |xy| {cell['start_xy_m'][0]:.3f} m · {time.time() - t0:.0f}s")
			print(f"  {'teacher':12s} " + "  ".join(f"{c:>15s}" for c in cols))
			for name, rs in rows.items():
				stats = {c: _ms([r[i] for r in rs]) for i, c in enumerate(cols)}
				cell["teachers"][name] = {"per_seed": rs, **stats}
				print(f"  {name:12s} " + "  ".join(f"{m:8.3f}±{s:<6.3f}" for m, s in stats.values()))
			out["cells"].append(cell)
			sys.stdout.flush()
	with open(a.out, "w") as f:
		json.dump(out, f, indent=1)
	print(f"\n# wrote {a.out}")


if __name__ == "__main__":
	main()
