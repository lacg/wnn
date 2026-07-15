#!/usr/bin/env python
"""Reproduce the TERNARY grid winner (sn=12 b=30, ~90-91% during-search stable) and run
the HONEST held-out on it — NOTHING ELSE. No GA-Neurons, no GA-Memory.

The grid is deterministic given seeds/args, so re-running ONLY stage-0 grid with the
identical CLI of c10_gran_ternary_20260715_p30s1000 regenerates the winner (it appears
as an `expand … sn=12 … stable≈90%` line → GRID WINNER).

HELD-OUT METHOD (matches how production holds out a MEMORY-stage winner):
  1. TRAIN the winner arch on the TRAIN seed, EXACTLY as the grid did — same evaluator
     config, num_eval_folds=5 K-fold ACCUMULATE (write_back stamps the trained cells).
  2. SCORE those cells on a FRESH report seed (score_genomes → NO retrain).
This is train-on-A → test-on-B: the true generalization number.

WHY NOT phased_ga._holdout_report directly: for an ARCH-only genome (grid winners carry
no cells) it RE-trains on the report seed with the evaluator's DEFAULT num_eval_folds=1
(K=1, undertrained) — that diverges (measured 8% stable / 24.7°, an artifact, not a real
result). Production never hits that path because real held-out winners (MEMORY stage)
carry cells → score-only. We replicate the score-only semantics here.
"""
import math
import time

from wnn.control.phased_ga import (build_arg_parser, stage0_grid, _rg_config,
                                   _pid_baseline, _save_winner)
from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
from wnn.control.training import EpisodeConfig, DisturbanceConfig
from wnn.seeds import resolve_seed_set, log_seed_set

# EXACT CLI of the p30s1000 TERNARY arm (run_gran_5arm_capped.sh, STEPS=1000 POP=30).
ARGV = [
	"--grid-state-neurons", "8", "12", "16", "--grid-bits", "24", "30", "--levels", "16",
	"--skip-stages", "bits,connections", "--lamarckian", "--saturation-grow-gain", "1.0",
	"--neurons-gens", "60", "--neurons-patience", "3", "--memory-gens", "120", "--memory-patience", "2",
	"--pop", "30", "--num-eval-folds", "5", "--check-interval", "2", "--magnitude-aware-patience",
	"--eval-episodes", "100", "--memory-eval-episodes", "200", "--steps", "1000",
	"--max-state-neurons", "24", "--max-output-neurons", "128", "--tilt", "5.0",
	"--fit-weight-err-sq", "0.4", "--fit-weight-stable", "0.3", "--fit-weight-jerk", "0.2", "--fit-weight-mono", "0.1",
	"--report-seed", "99990101", "--report-episodes", "100", "--holdout-pop-sample", "8",
	"--base-seed", "31337002", "--runs", "1", "--teacher", "lqr",
	"--memory-mode", "TERNARY",
]
SAVE = "/Users/lacg/wnn/logs/controller/ternary_grid_winner.yaml.gz"  # canonical (_ctl_load / --seed-winner)


def _mk_ec(args):
	dist = DisturbanceConfig.preset(args.disturbance, seed=911)
	return EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate,
		disturbance=dist,
	)


def _held_out(args, ec, winner_spec, winner, grid_thr, train_seed, report_seed):
	"""Train winner on train_seed (grid-identical, K=5 accumulate) → score on report_seed."""
	# 1. TRAIN on train seed — evaluator built identically to the grid's shared evaluator.
	train_ev = ControllerEvaluator(
		winner_spec, num_eval_episodes=args.eval_episodes, seed=train_seed,
		episode_config=ec, thresholds=grid_thr, rg_config=_rg_config(args, ec, train_seed),
		max_train_workers=args.train_workers, num_eval_folds=args.num_eval_folds)
	winner.cells = None                       # grid discards cells; start clean
	tr = train_ev._evaluate_core([winner], write_back=True)   # stamps trained cells
	tr_m = tr[0][0] if isinstance(tr[0], tuple) else tr[0]
	print(f"  [train@{train_seed} K={args.num_eval_folds}] winner trained: "
	      f"stable={tr_m.acc*100:.1f}% err={tr_m.mean_attitude_error_deg:.2f}° "
	      f"cells={'yes' if getattr(winner, 'cells', None) is not None else 'NONE!'}")

	# 2. SCORE trained cells on the FRESH report seed (no retrain).
	rep_thr = fit_thresholds_from_pid_rollouts(
		winner_spec, num_episodes=10, seed=report_seed,
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))
	rep_ev = ControllerEvaluator(
		winner_spec, num_eval_episodes=args.report_episodes, seed=report_seed,
		episode_config=ec, thresholds=rep_thr, rg_config=_rg_config(args, ec, report_seed),
		max_train_workers=args.train_workers)
	held = rep_ev.score_genomes([winner])[0]
	pid_m = _pid_baseline(ec, args.report_episodes, report_seed)
	return held, pid_m


def main():
	args = build_arg_parser().parse_args(ARGV)
	ec = _mk_ec(args)
	base = args.base_seed if args.base_seed is not None else args.seed
	s = resolve_seed_set(base=base, run_index=0,
	                     train=args.train_seed, test=args.test_seed, val=args.val_seed)
	log_seed_set(s)

	print(f"\n{'#'*72}\n# TERNARY grid-winner HONEST held-out (grid-only; NO GA)\n{'#'*72}")
	t0 = time.time()

	# STAGE 0 — deterministic grid → regenerates the sn=12 b=30 winner.
	winner_spec, seed_population, m0, dt0, grid_thr = stage0_grid(args, ec, s.train)
	winner = seed_population[0]
	# SAVE immediately (canonical schema-2 yaml.gz, cells packed) so the winner is never
	# lost again and is loadable by _ctl_load / --seed-winner. Seeds/thresholds are NOT
	# stored (deterministic from --base-seed; thresholds are PID-fit + arch-independent).
	_save_winner(SAVE, args, winner_spec, winner, seed_population, m0)
	print(f"\n[grid done {dt0:.0f}s] winner sn={getattr(winner_spec,'state_neurons','?')} "
	      f"b={getattr(winner_spec,'state_bits','?')} during-search: "
	      f"stable={m0.acc*100:.1f}% err={m0.mean_attitude_error_deg:.2f}°  (saved → {SAVE})")

	# HONEST held-out.
	held, pid_m = _held_out(args, ec, winner_spec, winner, grid_thr, s.train, args.report_seed)
	bar = "=" * 72
	sty = getattr(held, "mean_steady_error_deg", None)
	print(f"\n{bar}\n  HONEST HELD-OUT [GRID-WINNER sn=12 b=30 TERNARY] — train seed {s.train} "
	      f"(K={args.num_eval_folds}) → score FRESH seed {args.report_seed}\n{bar}")
	print(f"  RESULT (held-out):  stable={held.acc*100:.1f}%  err={held.mean_attitude_error_deg:.2f}°"
	      + (f"  steady={sty:.2f}°" if sty is not None else "") + f"  reward={held.fitness:.2f}")
	_bl = pid_m.get("label", "PID") if isinstance(pid_m, dict) else "PID"
	print(f"  vs {_bl} (held-out): stable={pid_m['stable_rate']*100:.1f}%  err={pid_m['mean_attitude_error_deg']:.2f}°")
	print(bar)
	print(f"\n[ALL DONE {time.time()-t0:.0f}s]  held-out: stable={held.acc*100:.1f}% err={held.mean_attitude_error_deg:.2f}°")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
