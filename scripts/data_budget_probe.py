#!/usr/bin/env python3
"""Phase B of the ceiling pipeline: is the student-teacher gap DATA STARVATION?

The gap decomposition (30/07) put the representational ceiling at 96% stable under
L2D while the best student sits at 92 (its cell) / 77±15 (cell mean). Each genome in
the GA learns from 8 rounds x 24 episodes of DAgger — crumbs, because the GA
multiplies training cost by population size. Sajus-style RL learners see orders of
magnitude more environment steps. Before redesigning the training scheme, measure
what data alone buys: retrain the WINNING architecture once (no GA) at 1x and Nx the
study's DAgger budget, through the SAME evaluator machinery the study used, and
score all three states on the same held-out episodes.

  saved      the winner's cells exactly as the study left them (anchor)
  retrain-1x same arch, cells wiped, retrained at the study budget — the internally
             paired baseline for the scale comparison (fold-accumulation means this
             does NOT have to equal the study table number; see rescore_winners)
  retrain-Nx same arch, cells wiped, budget x N

Reading: (Nx - 1x) is the pure data effect. If it recovers a large share of
(ceiling - student), the learning gap is mostly starvation and the cheap fix is
budget, not a new algorithm. If Nx ~= 1x, the algorithm is the wall and the long
closed-loop memory-GA (phase A) is the right bet.

Training fidelity: mirrors phased_ga._holdout_report's train path verbatim —
ControllerEvaluator on the winner's TRAIN seed (resolve_seed_set of its base seed),
K=5, one _evaluate_core(write_back=True) pass — then scores on the report seed the
way everything since 29/07 scores (fresh evaluator, fold-0 pool).

Usage: data_budget_probe.py --winner logs/controller/dfa1l/dfa_9feat_BINARY_s31337002_winner.yaml.gz \
           --scale 16 --out experiments/dfa1l_markers/data_budget_probe.json
"""
import argparse
import json
import math
import sys
import time

from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.evaluator import (ControllerEvaluator, EpisodeConfig,
                                   fit_thresholds_from_pid_rollouts)
from wnn.control.reward_gated import RewardGatedConfig
from wnn.control.training import DisturbanceConfig
from wnn.seeds import resolve_seed_set

# The study's per-genome DAgger budget (RewardGatedConfig defaults): 8 x 24.
BASE_ROUNDS, BASE_EPISODES = 8, 24


def _ec(a):
	return EpisodeConfig(
		dt=0.001, steps_per_episode=a.steps,
		max_initial_tilt_rad=math.radians(a.tilt),
		max_initial_yaw_rad=math.radians(a.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=DisturbanceConfig.preset(a.disturbance, seed=911))


def _rg(seed, ec, a, rounds, episodes):
	rg = RewardGatedConfig(seed=seed, episode_config=ec)
	rg.num_rounds, rg.episodes_per_round = rounds, episodes
	rg.steps_per_episode, rg.progress = a.steps, False
	rg.teacher = a.teacher
	return rg


def _train(genome, spec, ec, train_seed, a, rounds, episodes):
	"""Wipe cells IN PLACE + retrain — phased_ga._holdout_report's train path
	verbatim (`_g.cells = None` then _evaluate_core(write_back=True)). GenomeCells
	is a Rust object and cannot be deepcopied, so the arms run sequentially on the
	one genome: score the saved cells FIRST, then each retrain wipes and rebuilds."""
	genome.cells = None
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=train_seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=a.episodes, seed=train_seed,
	                         episode_config=ec, thresholds=thr,
	                         rg_config=_rg(train_seed, ec, a, rounds, episodes),
	                         num_eval_folds=5)
	t0 = time.time()
	ev._evaluate_core([genome], write_back=True)
	return genome, time.time() - t0


def _score(genome, spec, ec, report_seed, a):
	"""Held-out score, fold-0 pool — same shape as everything since 29/07."""
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=report_seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=a.episodes, seed=report_seed,
	                         episode_config=ec, thresholds=thr,
	                         rg_config=_rg(report_seed, ec, a, BASE_ROUNDS, BASE_EPISODES),
	                         num_eval_folds=5)
	m = ev.score_genomes([genome])[0]
	return {"stable": m.acc * 100.0, "err_deg": m.mean_attitude_error_deg,
	        "steady_deg": getattr(m, "mean_steady_error_deg", None)}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--winner", required=True)
	ap.add_argument("--scale", type=int, default=16,
	                help="budget multiplier for the Nx arm (applied to episodes/round)")
	ap.add_argument("--base-seed", type=int, default=None,
	                help="winner's --base-seed; default parses _sNNNN from the filename")
	ap.add_argument("--report-seed", type=int, default=99990101)
	ap.add_argument("--episodes", type=int, default=100)
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--tilt", type=float, default=5.0)
	ap.add_argument("--disturbance", default="L2D")
	ap.add_argument("--teacher", default="lqr")
	ap.add_argument("--rounds", type=int, default=BASE_ROUNDS)
	ap.add_argument("--base-episodes", type=int, default=BASE_EPISODES)
	ap.add_argument("--out", required=True)
	a = ap.parse_args()

	base = a.base_seed
	if base is None:
		import re
		m = re.search(r"_s(\d+)_winner", a.winner)
		if not m:
			print("cannot parse base seed from filename; pass --base-seed", file=sys.stderr)
			return 1
		base = int(m.group(1))
	train_seed = resolve_seed_set(base=base, run_index=0).train

	payload = load_controller_checkpoint(a.winner)
	pop = payload.get("population") or []
	genome = pop[0] if pop else payload.get("best_genome")   # population[0] = THE RESULT
	spec = payload["spec"]
	ec = _ec(a)

	arms = {}
	print(f"# data-budget probe: {a.winner}")
	print(f"# base_seed={base} train_seed={train_seed} teacher={a.teacher} "
	      f"{a.disturbance} | budget 1x = {a.rounds}x{a.base_episodes} eps, "
	      f"Nx = {a.rounds}x{a.base_episodes * a.scale}")
	arms["saved"] = {"metrics": _score(genome, spec, ec, a.report_seed, a),
	                 "train_s": None}
	print(f"saved      : {arms['saved']['metrics']}")
	for name, mult in (("retrain-1x", 1), (f"retrain-{a.scale}x", a.scale)):
		g, dt = _train(genome, spec, ec, train_seed, a, a.rounds,
		               a.base_episodes * mult)
		arms[name] = {"metrics": _score(g, spec, ec, a.report_seed, a),
		              "train_s": round(dt, 1)}
		print(f"{name:11}: {arms[name]['metrics']}  (train {dt:.0f}s)")

	with open(a.out, "w") as f:
		json.dump({"meta": {k: v for k, v in vars(a).items()} |
		           {"base_seed": base, "train_seed": train_seed,
		            "budget_note": "Nx scales episodes_per_round; rounds fixed so the "
		                           "DAgger mixing schedule is unchanged."},
		           "arms": arms}, f, indent=1)
	print(f"# wrote {a.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
