#!/usr/bin/env python
"""Parametrized grid-winner HONEST held-out for the controller granularity ablation —
ONE script for ALL modes (supersedes the binary/qsr one-offs; no duplicates).

Runs stage-0 grid ONLY (NO GA-Neurons, NO GA-Memory): the deterministic grid regenerates
the winner, we TRAIN it on the train seed (num_eval_folds=5 K-fold ACCUMULATE, write_back
stamps cells) and SCORE those cells on a FRESH report seed (score_genomes → NO retrain).
That is train-on-A → test-on-B: the true generalization number. We replicate the
score-only semantics production uses for a MEMORY-stage winner (avoids the _holdout_report
K=1 undertraining artifact for arch-only genomes).

Usage:
  gran_grid_winner_holdout.py --memory-mode TERNARY --pop 50 --steps 2000 \
      --save logs/controller/ternary_grid_winner_s2000p50.yaml.gz
"""
import argparse
import math
import time

from wnn.control.phased_ga import (build_arg_parser, stage0_grid, _rg_config,
                                   _pid_baseline, _save_winner)
from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
from wnn.control.training import EpisodeConfig, DisturbanceConfig
from wnn.seeds import resolve_seed_set, log_seed_set


def _wrapper_args():
	p = argparse.ArgumentParser(description="Parametrized granularity grid-winner held-out")
	p.add_argument("--memory-mode", required=True,
	               help="TERNARY | QUAD_WEIGHTED | BINARY | QSR | PLN")
	p.add_argument("--pop", type=int, default=50)
	p.add_argument("--steps", type=int, default=2000)
	p.add_argument("--base-seed", type=int, default=31337002,
	               help="grid base seed (multi-seed replications use distinct base seeds)")
	p.add_argument("--save", default=None, help="canonical yaml.gz path for the saved winner")
	return p.parse_args()


def _phased_argv(w):
	"""Build the phased_ga CLI — identical recipe across modes; only mode/pop/steps vary.
	Matches the binqsr_v3 chain (STEPS=2000 POP=50) so winners are reproducible."""
	_bsfx = "" if w.base_seed == 31337002 else f"_b{w.base_seed}"
	save = w.save or f"/Users/lacg/wnn/logs/controller/{w.memory_mode.lower()}_grid_winner_s{w.steps}p{w.pop}{_bsfx}.yaml.gz"
	return [
		"--grid-state-neurons", "8", "12", "16", "--grid-bits", "24", "30", "--levels", "16",
		"--skip-stages", "bits,connections", "--lamarckian", "--saturation-grow-gain", "1.0",
		"--neurons-gens", "60", "--neurons-patience", "3", "--memory-gens", "120", "--memory-patience", "2",
		"--pop", str(w.pop), "--num-eval-folds", "5", "--check-interval", "2", "--magnitude-aware-patience",
		"--eval-episodes", "100", "--memory-eval-episodes", "200", "--steps", str(w.steps),
		"--max-state-neurons", "24", "--max-output-neurons", "128", "--tilt", "5.0",
		"--fit-weight-err-sq", "0.4", "--fit-weight-stable", "0.3", "--fit-weight-jerk", "0.2", "--fit-weight-mono", "0.1",
		"--report-seed", "99990101", "--report-episodes", "100", "--holdout-pop-sample", "8",
		"--base-seed", str(w.base_seed), "--runs", "1", "--teacher", "lqr",
		"--memory-mode", w.memory_mode,
	], save


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


def _spec_shape(spec):
	"""(sn, sb) with the REAL attr names (state_bits_per_neuron, not state_bits)."""
	sn = getattr(spec, "state_neurons", "?")
	sb = getattr(spec, "state_bits_per_neuron", getattr(spec, "state_bits", "?"))
	return sn, sb


def main():
	w = _wrapper_args()
	# Friendly alias: "QUAD" → the canonical mode string the evaluator/phased_ga expect.
	if w.memory_mode.upper() == "QUAD":
		w.memory_mode = "QUAD_WEIGHTED"
	phased_argv, save = _phased_argv(w)
	args = build_arg_parser().parse_args(phased_argv)
	ec = _mk_ec(args)
	base = args.base_seed if args.base_seed is not None else args.seed
	s = resolve_seed_set(base=base, run_index=0,
	                     train=args.train_seed, test=args.test_seed, val=args.val_seed)
	log_seed_set(s)

	print(f"\n{'#'*72}\n# {w.memory_mode} grid-winner HONEST held-out "
	      f"(grid-only; NO GA)  pop={w.pop} steps={w.steps}\n{'#'*72}")
	t0 = time.time()

	# STAGE 0 — deterministic grid → regenerates the winner.
	winner_spec, seed_population, m0, dt0, grid_thr = stage0_grid(args, ec, s.train)
	winner = seed_population[0]
	_save_winner(save, args, winner_spec, winner, seed_population, m0)
	sn, sb = _spec_shape(winner_spec)
	print(f"\n[grid done {dt0:.0f}s] winner sn={sn} b={sb} during-search: "
	      f"stable={m0.acc*100:.1f}% err={m0.mean_attitude_error_deg:.2f}°  (saved → {save})")

	held, pid_m = _held_out(args, ec, winner_spec, winner, grid_thr, s.train, args.report_seed)
	bar = "=" * 72
	sty = getattr(held, "mean_steady_error_deg", None)
	print(f"\n{bar}\n  HONEST HELD-OUT [GRID-WINNER sn={sn} b={sb} {w.memory_mode}] — "
	      f"train seed {s.train} (K={args.num_eval_folds}) → score FRESH seed {args.report_seed}\n{bar}")
	print(f"  RESULT (held-out):  stable={held.acc*100:.1f}%  err={held.mean_attitude_error_deg:.2f}°"
	      + (f"  steady={sty:.2f}°" if sty is not None else "") + f"  reward={held.fitness:.2f}")
	_bl = pid_m.get("label", "PID") if isinstance(pid_m, dict) else "PID"
	print(f"  vs {_bl} (held-out): stable={pid_m['stable_rate']*100:.1f}%  err={pid_m['mean_attitude_error_deg']:.2f}°")
	print(bar)
	print(f"\n[ALL DONE {time.time()-t0:.0f}s]  {w.memory_mode} held-out: "
	      f"stable={held.acc*100:.1f}% err={held.mean_attitude_error_deg:.2f}°  grid={dt0:.0f}s")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
