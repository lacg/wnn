#!/usr/bin/env python
"""Leak-free TOP-K held-out for the controller granularity ablation.

Loads a saved grid population (schema-2 yaml.gz), trains the TOP-K genomes on the TRAIN
seed (K=5 accumulate), SELECTS the best on a VAL seed, and REPORTS that pick's held-out on
a separate TEST/report seed. Pick-on-VAL, report-on-TEST = train/val/test separation, so it
is NOT a test-set leak (unlike "pick the best generalizer on the held-out", which selects on
the test draw — the very leak _holdout_report guards against).

Improves winner selection over grid-winner-#1 (which is crowned by the noisy during-search
fitness — see project_gran_ablation_winner_variance) WITHOUT leaking. Also prints the #1
during-search winner's held-out for comparison. Reuses the saved grid (no re-run) → cheap.

Usage: gran_topk_holdout.py <winner.yaml.gz> [--top-k 8] [--base-seed N] [--report-seed N] [--steps N]
"""
import argparse
import math
import sys
import time

from wnn.control.phased_ga import build_arg_parser, _rg_config, _pid_baseline, _ctl_load
from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
from wnn.control.training import EpisodeConfig, DisturbanceConfig
from wnn.seeds import resolve_seed_set


def _wrap():
	p = argparse.ArgumentParser(description="Leak-free top-K train/val/test held-out")
	p.add_argument("path")
	p.add_argument("--top-k", type=int, default=8)
	p.add_argument("--base-seed", type=int, default=31337002)
	p.add_argument("--report-seed", type=int, default=99990101)
	p.add_argument("--steps", type=int, default=2000)
	return p.parse_args()


def _mk_ec(args):
	dist = DisturbanceConfig.preset(args.disturbance, seed=911)
	return EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt), max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate, disturbance=dist)


def _ev(spec, seed, ec, args, folds=None):
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed,
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))
	kw = dict(num_eval_episodes=args.report_episodes, seed=seed, episode_config=ec, thresholds=thr,
		rg_config=_rg_config(args, ec, seed), max_train_workers=args.train_workers)
	if folds is not None:
		kw["num_eval_folds"] = folds
	return ControllerEvaluator(spec, **kw)


def _triple(m):
	sty = getattr(m, "mean_steady_error_deg", None)
	return (m.acc * 100, m.mean_attitude_error_deg, sty)


def main():
	w = _wrap()
	blob = _ctl_load(w.path)
	spec = blob["spec"]
	pop = blob.get("population") or [blob["best_genome"]]
	mode = getattr(spec, "memory_mode", "?")
	topk = pop[: w.top_k]

	argv = [
		"--levels", "16", "--num-eval-folds", "5", "--eval-episodes", "100", "--report-episodes", "100",
		"--steps", str(w.steps), "--tilt", "5.0",
		"--fit-weight-err-sq", "0.4", "--fit-weight-stable", "0.3", "--fit-weight-jerk", "0.2", "--fit-weight-mono", "0.1",
		"--report-seed", str(w.report_seed), "--base-seed", str(w.base_seed), "--teacher", "lqr",
		"--memory-mode", str(mode),
	]
	args = build_arg_parser().parse_args(argv)
	ec = _mk_ec(args)
	s = resolve_seed_set(base=w.base_seed, run_index=0,
	                     train=args.train_seed, test=args.test_seed, val=args.val_seed)
	train_seed, val_seed, test_seed = s.train, s.val, w.report_seed

	print(f"\n{'#'*72}\n# {mode} TOP-{w.top_k} leak-free held-out (train→pick-on-VAL→report-on-TEST)\n"
	      f"# {w.path}\n# train={train_seed} val={val_seed} test={test_seed}\n{'#'*72}")
	t0 = time.time()

	# 1. TRAIN the top-K on the train seed (K=5 accumulate) — one mixed-shape batch.
	for g in topk:
		g.cells = None
	train_ev = _ev(spec, train_seed, ec, args, folds=args.num_eval_folds)
	train_ev._evaluate_core(topk, write_back=True)
	print(f"  [train@{train_seed} K={args.num_eval_folds}] trained {len(topk)} genomes")

	# 2. SELECT the best on the VAL seed (score-only; cells trained on train seed).
	val_ev = _ev(spec, val_seed, ec, args)
	val_m = val_ev.score_genomes(topk)
	# rank by val stable% desc, tie-break lower err
	order = sorted(range(len(topk)), key=lambda i: (-val_m[i].acc, val_m[i].mean_attitude_error_deg))
	best_i = order[0]
	val_line = " ".join(f"g{i}:{val_m[i].acc*100:.0f}%" for i in range(len(topk)))
	print(f"  [val@{val_seed}] stables: {val_line}")
	print(f"  [select] best-on-val = genome #{best_i} (val stable={val_m[best_i].acc*100:.1f}%); "
	      f"grid-winner #0 val stable={val_m[0].acc*100:.1f}%")

	# 3. REPORT the picked genome AND the grid-winner #0 on the TEST seed (no retrain).
	test_ev = _ev(spec, test_seed, ec, args)
	held = test_ev.score_genomes([topk[best_i], topk[0]])
	sel_s, sel_e, sel_sty = _triple(held[0])
	w0_s, w0_e, w0_sty = _triple(held[1])
	pid_m = _pid_baseline(ec, args.report_episodes, test_seed)

	bar = "=" * 72
	print(f"\n{bar}\n  {mode} — TOP-{w.top_k} leak-free RESULT (test seed {test_seed})\n{bar}")
	print(f"  SELECTED (pick-on-val #{best_i}):  stable={sel_s:.1f}%  err={sel_e:.2f}°"
	      + (f"  steady={sel_sty:.2f}°" if sel_sty is not None else ""))
	print(f"  grid-winner #0 (during-search):    stable={w0_s:.1f}%  err={w0_e:.2f}°"
	      + (f"  steady={w0_sty:.2f}°" if w0_sty is not None else ""))
	print(f"  vs PID:                            stable={pid_m['stable_rate']*100:.1f}%  err={pid_m['mean_attitude_error_deg']:.2f}°")
	print(bar)
	print(f"[TOPK DONE {time.time()-t0:.0f}s]  {mode}: selected stable={sel_s:.1f}% err={sel_e:.2f}° "
	      f"(vs #0 {w0_s:.1f}%/{w0_e:.2f}°)  seed={w.base_seed}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
