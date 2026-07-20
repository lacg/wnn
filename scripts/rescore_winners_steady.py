#!/usr/bin/env python
"""Re-score saved controller winners to recover the transient/display metrics.

WHY (20/07/2026): the CPU scorer used to hardcode the 7 transient metrics
(steady, rise, settle_abs, settle_rel, itae, iae, ise) to 0.0, so every run with
WNN_CONTROLLER_GPU_EVAL=0 reported `steady=0.00°` and the held-out triple was
incomplete. cpu_score.rs now computes all 13 (kernel-mirrored definitions,
parity-tested against controller_rollout.metal). This re-scores winners saved
BEFORE the fix so their steady° can be reported honestly.

SCORE-ONLY: loads each winner's already-trained cells and scores them on the
FRESH report seed — no retraining, so the numbers are directly comparable to the
run's own held-out line (which is also score-only for a MEMORY-stage winner).
Each winner's OWN spec (sn, bits, mode, obs flags) rides along in the file, so
sn=0 reflex and sn=8/16 DFA winners are both handled.

Usage:
  python scripts/rescore_winners_steady.py [--steps N] [--episodes N] path.yaml.gz ...
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# The CPU path is the one that was broken; score through it explicitly so this
# tool verifies the fix (and does not contend with the IDS worker for the GPU).
os.environ.setdefault("WNN_CONTROLLER_GPU_EVAL", "0")

from wnn.control.phased_ga import build_arg_parser, _rg_config, _pid_baseline, _ctl_load
from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
from wnn.control.training import EpisodeConfig, DisturbanceConfig


def _args_for(disturbance: str, steps: int, episodes: int):
	"""The non-grid knobs the winners were searched under (C10 weights, folds 5)."""
	return build_arg_parser().parse_args([
		"--levels", "16", "--num-eval-folds", "5",
		"--eval-episodes", str(episodes), "--report-episodes", str(episodes),
		"--steps", str(steps), "--tilt", "5.0",
		"--fit-weight-err-sq", "0.4", "--fit-weight-stable", "0.3",
		"--fit-weight-jerk", "0.2", "--fit-weight-mono", "0.1",
		"--report-seed", "99990101", "--base-seed", "31337002",
		"--teacher", "lqr", "--disturbance", disturbance,
	])


def rescore(path: str, disturbance: str, steps: int, episodes: int) -> dict | None:
	blob = _ctl_load(path)
	spec, winner = blob["spec"], blob["best_genome"]
	if winner.cells is None:
		print(f"  !! {os.path.basename(path)}: winner carries NO cells — cannot score-only")
		return None
	args = _args_for(disturbance, steps, episodes)
	dist = DisturbanceConfig.preset(args.disturbance, seed=911)
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate,
		max_initial_yaw_rate=args.yaw_rate, disturbance=dist)
	rep_thr = fit_thresholds_from_pid_rollouts(
		spec, num_episodes=10, seed=args.report_seed,
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))
	ev = ControllerEvaluator(
		spec, num_eval_episodes=args.report_episodes, seed=args.report_seed,
		episode_config=ec, thresholds=rep_thr,
		rg_config=_rg_config(args, ec, args.report_seed),
		max_train_workers=args.train_workers)
	m = ev.score_genomes([winner])[0]
	return {
		"name": os.path.basename(path).replace("_winner.yaml.gz", ""),
		"sn": getattr(spec, "state_neurons", "?"),
		"stable": m.acc * 100.0,
		"err": m.mean_attitude_error_deg,
		"steady": getattr(m, "mean_steady_error_deg", None),
	}


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("winners", nargs="+")
	ap.add_argument("--disturbance", default="L2")
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--episodes", type=int, default=100)
	a = ap.parse_args()

	rows = []
	for p in a.winners:
		if not os.path.exists(p):
			print(f"  !! missing: {p}")
			continue
		print(f"[rescore] {os.path.basename(p)} ...", flush=True)
		r = rescore(p, a.disturbance, a.steps, a.episodes)
		if r:
			rows.append(r)
	if not rows:
		return
	args = _args_for(a.disturbance, a.steps, a.episodes)
	dist = DisturbanceConfig.preset(args.disturbance, seed=911)
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate,
		max_initial_yaw_rate=args.yaw_rate, disturbance=dist)
	pid = _pid_baseline(ec, args.report_episodes, args.report_seed)

	bar = "=" * 78
	print(f"\n{bar}\n  HELD-OUT RE-SCORE (CPU scorer, transient metrics restored) — "
	      f"seed {args.report_seed}, {a.disturbance}, {a.steps} steps\n{bar}")
	print(f"  {'winner':<34} {'sn':>3} {'stable':>8} {'err':>8} {'steady':>8}")
	for r in rows:
		sty = f"{r['steady']:.2f}°" if r["steady"] is not None else "n/a"
		print(f"  {r['name']:<34} {r['sn']:>3} {r['stable']:>7.1f}% {r['err']:>7.2f}° {sty:>8}")
	print(f"  {'PID (same conditions)':<34} {'—':>3} "
	      f"{pid['stable_rate']*100:>7.1f}% {pid['mean_attitude_error_deg']:>7.2f}° {'—':>8}")
	print(bar)


if __name__ == "__main__":
	main()
