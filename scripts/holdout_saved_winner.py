#!/usr/bin/env python
"""Held-out ONE saved controller winner — NO grid, NO population, NO watchdog.

Loads a winner genome saved by ternary_grid_winner_holdout.py (ternary_grid_winner.pkl.gz)
and does EXACTLY: train that single genome on the TRAIN seed (K=5 accumulate, the search
regime) → score its cells on the FRESH report seed (score_genomes, no retrain). That's it.

This is the repeatable "1 freaking genome, train + held-out test" tool. It touches one
genome; peak memory is one TERNARY controller's cells (tiny vs a pop=30 grid). Nothing to
kill from our side — if macOS jetsam takes it, rerun. K=1 is never used.

Usage:  python scripts/holdout_saved_winner.py [path.pkl.gz]
"""
import gzip
import math
import pickle
import sys
import time

from wnn.control.phased_ga import build_arg_parser, _rg_config, _pid_baseline
from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
from wnn.control.training import EpisodeConfig, DisturbanceConfig

DEFAULT_PKL = "/Users/lacg/wnn/logs/controller/ternary_grid_winner.pkl.gz"
# Same non-grid CLI knobs the winner was searched under (steps/tilt/folds/report).
ARGV = [
	"--levels", "16", "--num-eval-folds", "5", "--eval-episodes", "100",
	"--report-episodes", "100", "--steps", "1000", "--tilt", "5.0",
	"--fit-weight-err-sq", "0.4", "--fit-weight-stable", "0.3", "--fit-weight-jerk", "0.2", "--fit-weight-mono", "0.1",
	"--report-seed", "99990101", "--base-seed", "31337002", "--teacher", "lqr",
	"--memory-mode", "TERNARY",
]


def main():
	pkl = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PKL
	with gzip.open(pkl, "rb") as f:
		blob = pickle.load(f)
	spec   = blob["spec"]
	winner = blob["winner"]
	train_seed  = blob["seeds"]["train"]
	report_seed = blob["seeds"]["report"]
	grid_thr = blob.get("grid_thresholds")

	args = build_arg_parser().parse_args(ARGV)
	dist = DisturbanceConfig.preset(args.disturbance, seed=911)
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt), max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate, disturbance=dist)

	print(f"\n{'#'*72}\n# 1-GENOME held-out (loaded {pkl})\n"
	      f"# winner sn={getattr(spec,'state_neurons','?')} b={getattr(spec,'state_bits','?')} "
	      f"mode={getattr(spec,'memory_mode','?')}  train={train_seed} report={report_seed}\n{'#'*72}")
	t0 = time.time()

	# 1. TRAIN the single genome on the train seed, K=5 accumulate (search regime).
	if grid_thr is None:
		grid_thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=train_seed,
			geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))
	train_ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes, seed=train_seed,
		episode_config=ec, thresholds=grid_thr, rg_config=_rg_config(args, ec, train_seed),
		max_train_workers=args.train_workers, num_eval_folds=args.num_eval_folds)
	winner.cells = None
	tr = train_ev._evaluate_core([winner], write_back=True)
	tr_m = tr[0][0] if isinstance(tr[0], tuple) else tr[0]
	print(f"  [train@{train_seed} K={args.num_eval_folds}] stable={tr_m.acc*100:.1f}% "
	      f"err={tr_m.mean_attitude_error_deg:.2f}° cells={'yes' if winner.cells is not None else 'NONE!'}")

	# 2. SCORE those cells on the fresh report seed — no retrain.
	rep_thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=report_seed,
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))
	rep_ev = ControllerEvaluator(spec, num_eval_episodes=args.report_episodes, seed=report_seed,
		episode_config=ec, thresholds=rep_thr, rg_config=_rg_config(args, ec, report_seed),
		max_train_workers=args.train_workers)
	held = rep_ev.score_genomes([winner])[0]
	pid_m = _pid_baseline(ec, args.report_episodes, report_seed)

	bar = "=" * 72
	sty = getattr(held, "mean_steady_error_deg", None)
	print(f"\n{bar}\n  HELD-OUT [1 genome] — train {train_seed} (K={args.num_eval_folds}) → "
	      f"score FRESH {report_seed}\n{bar}")
	print(f"  RESULT: stable={held.acc*100:.1f}%  err={held.mean_attitude_error_deg:.2f}°"
	      + (f"  steady={sty:.2f}°" if sty is not None else "") + f"  reward={held.fitness:.2f}")
	print(f"  vs PID: stable={pid_m['stable_rate']*100:.1f}%  err={pid_m['mean_attitude_error_deg']:.2f}°")
	print(f"{bar}\n[done {time.time()-t0:.0f}s]")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
