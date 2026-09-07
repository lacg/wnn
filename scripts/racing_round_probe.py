#!/usr/bin/env python3
"""RACING PREDICTIVITY PROBE, ROUND RUNG — does the score after r of R DAgger rounds
predict the score after all R? (07/09/2026, Luiz: "the DAgger round inside a fold".)

WHY THIS RUNG AT ALL. The FOLD rung was probed 07/09 and closed: cutting at a fold
boundary is EXACT (identical=True, max|dreward|=0.000000) but the rank correlation
between an early fold and the last one is ~0.2 at CONNECTIONS and ~0.1 at GRID —
noise. The diagnosis matters more than the number: the K folds are five DRAWS OF THE
SAME DISTRIBUTION, so ranking after fold 1 is re-measuring luck, not watching a
policy improve. Rounds are different in kind — round 3 of 8 is a genuinely
less-trained policy on a real learning curve — which is why this rung is worth its
own probe rather than an assumption either way.

⚠️ THE ORDER OF THE TWO CHECKS IS DELIBERATELY REVERSED FROM THE FOLD PROBE.
That probe established exactness FIRST and then found the signal was unusable — the
exactness work was wasted. Worse, exactness here is already known to be FALSE without
a Rust change: `RewardGatedConfigPacked::round_tilt_rad` (dagger_train.rs:551) ramps
the curriculum tilt as `frac = it / (num_rounds - 1)`, so training 3 rounds and then
5 more ramps easy->full TWICE and is NOT the same schedule as one 8-round call. A
resumable rung would need `num_rounds` split into (round_offset, total_rounds).
So this probe measures PREDICTIVITY ONLY, and pays for the Rust exactness work ONLY
if the signal survives. If predictivity fails again, exactness never mattered.

METHOD. One banked stage population -> 60 real offspring (the strategy's own
tournament/crossover/mutation, via racing_fold_probe's shared helpers). For each r in
--rounds-grid, every candidate is trained INDEPENDENTLY from the SAME parent cells on
the SAME single fold seed with rg_config.num_rounds = r, then CRN-scored on all pools.
That is exactly "what does a policy trained for r rounds look like" — no resume, so no
exactness precondition. Then for each r < R: Spearman(rank at r, rank at R), how much
of the true top third the r-round top third keeps, and the REGRET (best true fitness
among r-round survivors vs best overall).

COST. Training is linear in rounds, so the grid 1..8 costs sum(r)/R = 4.5 full
trainings plus one scoring pass per r — about 40-60 min for 60 candidates at one fold,
against ~65 min for the fold probe. Cheap enough to answer the question outright.

Usage (the chain wraps this):
  PYTHONPATH=src/wnn python scripts/racing_round_probe.py \
      --ckpt <stageN.yaml.gz> --recipe-args "<phased_ga argv>" \
      --out experiments/racing_markers/PROBE_rounds_<tag>.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse, never re-implement: these are the SAME helpers the fold probe used, so the
# two rungs are measured by identical offspring generation, metrics and fitness.
from racing_fold_probe import (  # noqa: E402
	build_episode_config, build_evaluator_and_strategy, fitness_of, make_offspring,
	metrics_from_scored, spearman,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
	p = argparse.ArgumentParser(description=__doc__,
	                            formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--ckpt", required=True, help="stage checkpoint (yaml.gz) seeding the offspring")
	p.add_argument("--recipe-args", required=True, help="the phased_ga argv of the recipe, one string")
	p.add_argument("--out", required=True, help="JSON results path")
	p.add_argument("--candidates", type=int, default=60)
	p.add_argument("--keep-fraction", type=float, default=1.0 / 3.0)
	p.add_argument("--rounds-grid", default="1,2,3,4,5,6,8",
	               help="round counts to train at; the LAST is the reference (full) training")
	p.add_argument("--dry-run", action="store_true")
	return p.parse_args()


def train_at_rounds(ev, genomes, inits, shape_keys, fold_seed_of, rounds, log):
	"""Train every candidate for exactly `rounds` DAgger rounds from its parent cells.

	INDEPENDENT of every other r: same starting cells, same fold seed, only the round
	budget differs. No resume, so `round_tilt_rad`'s ramp is a correct single ramp for
	this r — which is the whole reason this probe needs no exactness precondition."""
	import copy
	rg = copy.copy(ev.rg_config)
	rg.num_rounds = int(rounds)
	saved, ev.rg_config = ev.rg_config, rg
	try:
		t0 = time.time()
		tasks = [(gi, [fold_seed_of[gi]]) for gi in range(len(genomes))]
		trained = ev._train_genomes_rust_batched(genomes, tasks, init_override=inits)
		t1 = time.time()
		scored = ev._score_fitness([c for (c, _s) in trained], shape_keys)
		t2 = time.time()
	finally:
		ev.rg_config = saved
	log(f"rounds {rounds}: train {t1 - t0:.0f}s  score {t2 - t1:.0f}s")
	return scored, dict(rounds=int(rounds), train_s=round(t1 - t0, 1), score_s=round(t2 - t1, 1))


def analyse(fit_by_r, grid, keep_fraction):
	"""Per cut r: rank correlation against the full-budget ranking, how much of the true
	top third survives, and the regret of racing at r. Lower fitness = better."""
	full = fit_by_r[grid[-1]]
	n = len(full)
	k = max(1, int(round(n * keep_fraction)))
	true_order = sorted(range(n), key=lambda i: full[i])
	true_top = set(true_order[:k])
	best_true = full[true_order[0]]
	cuts = []
	for r in grid[:-1]:
		f = fit_by_r[r]
		kept = sorted(range(n), key=lambda i: f[i])[:k]
		surv_best = min(full[i] for i in kept)
		cuts.append(dict(
			cut_after_round=int(r),
			spearman=round(spearman(f, full), 3),
			top_kept=len(true_top & set(kept)),
			top_size=k,
			# Regret in the fitness units selection actually ranks on.
			regret=round(surv_best - best_true, 4),
			true_best_survives=bool(true_order[0] in kept),
			# Fraction of the full training budget spent to reach this cut.
			train_units=round(r / grid[-1], 3)))
	return cuts, k


def main():
	pa = parse_args()
	from wnn.control import phased_ga as pg
	from wnn.control.checkpoint_io import load_controller_checkpoint
	args = pg.build_arg_parser().parse_args(pa.recipe_args.split())
	seed = int(args.base_seed)
	log = lambda s: print(f"[racing-round-probe] {time.strftime('%H:%M:%S')} {s}", flush=True)
	grid = [int(x) for x in pa.rounds_grid.split(",") if x.strip()]
	if len(grid) < 2:
		raise SystemExit("--rounds-grid needs at least two values (the cuts + the full reference)")

	payload = load_controller_checkpoint(pa.ckpt)
	if payload is None:
		raise SystemExit(f"checkpoint not loadable: {pa.ckpt}")
	population = list(payload.get("population") or [])
	spec = payload.get("spec")
	if not population or spec is None:
		raise SystemExit("checkpoint carries no population/spec")
	log(f"loaded {len(population)} genomes from {os.path.basename(pa.ckpt)} (stage {payload.get('stage_name')})")

	ec = build_episode_config(args)
	ev, strat, gacfg = build_evaluator_and_strategy(args, ec, spec, seed)
	log(f"evaluator: folds={ev.num_eval_folds} eval_episodes={ev.num_eval} crn={ev.score_crn} "
	    f"recipe rounds/fold={ev.rg_config.num_rounds} grid={grid}")
	if grid[-1] != ev.rg_config.num_rounds:
		log(f"NOTE: grid reference {grid[-1]} != the recipe's {ev.rg_config.num_rounds} rounds/fold — "
		    f"the 'full' column is the grid's last value, not the recipe default.")

	if pa.dry_run:
		parent_fit = [0.0] * len(population)
		log("dry-run: parents NOT scored (uniform tournament)")
	else:
		t0 = time.time()
		parent_fit = fitness_of(gacfg, ev.score_genomes(population))
		log(f"parents scored ({time.time() - t0:.0f}s); best fitness {min(parent_fit):.4f}")
	kids = make_offspring(strat, population, parent_fit, pa.candidates)
	inherited = sum(1 for g in kids if getattr(g, "cells", None) is not None)
	log(f"{len(kids)} offspring generated ({inherited} carry inherited cells)")
	if pa.dry_run:
		log("dry-run: stopping before training")
		return 0

	from wnn.control import _accel as ra
	N = len(kids)
	ev._ensure_ga_ready()
	ev._cur_axes = ev._active_axes(0)
	ev._advance_fold()
	base = ev._train_base_seeds(N, 0)
	shape_keys = [ev._shape_key(g) for g in kids]
	# ONE fold seed, reused at every r: the round budget is the only thing that varies.
	fold_seed_of = [base[gi] for gi in range(N)]
	inits = [(getattr(g, "cells", None) or ra.GenomeCells()) for g in kids]

	fit_by_r, timing, per_r = {}, [], {}
	for r in grid:
		scored, t = train_at_rounds(ev, kids, inits, shape_keys, fold_seed_of, r, log)
		fit_by_r[r] = fitness_of(gacfg, metrics_from_scored(scored))
		per_r[str(r)] = [dict(reward=float(rw), stable=float(m.get("stable_rate", 0.0)),
		                      err=float(m.get("mean_attitude_error_deg", 0.0)))
		                 for rw, m in scored]
		timing.append(t)

	cuts, k = analyse(fit_by_r, grid, pa.keep_fraction)
	result = dict(
		ckpt=pa.ckpt, rung="dagger_round", stage=payload.get("stage_name"),
		recipe_args=pa.recipe_args, seed=seed, candidates=N, inherited_cells=inherited,
		rounds_grid=grid, keep_fraction=pa.keep_fraction, top_size=k,
		# Recorded, NOT measured: resume-at-a-round is not exact today. See the module
		# docstring — round_tilt_rad ramps on num_rounds, so a split call ramps twice.
		exactness=dict(measured=False,
		               reason="round_tilt_rad ramps frac=it/(num_rounds-1); a split call ramps "
		                      "the curriculum twice. Would need (round_offset, total_rounds) in "
		                      "dagger_train.rs. Deliberately NOT paid for before the signal."),
		timing=timing, cuts=cuts, per_rounds=per_r)
	os.makedirs(os.path.dirname(pa.out) or ".", exist_ok=True)
	with open(pa.out, "w") as f:
		json.dump(result, f, indent=1)
	log(f"wrote {pa.out}")
	for c in cuts:
		log(f"  cut after round {c['cut_after_round']}: spearman {c['spearman']:+.3f}  "
		    f"top-third kept {c['top_kept']}/{c['top_size']}  regret {c['regret']:.4f}  "
		    f"true-best survives {c['true_best_survives']}  train-units {c['train_units']:.2f}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
