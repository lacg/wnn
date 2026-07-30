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

  saved       the winner's cells as the study left them — a REFERENCE, not the 1x
              baseline: they accumulated over 5 folds x every GA generation, so
              saved-vs-retrain conflates data volume with accumulation passes
  retrain-Nx  same arch, cells wiped, ONE training pass at N x the study budget,
              for each N in --scales (default 1 4 16)

Reading: the SHAPE across retrain arms, not any single difference. Two points
cannot separate "data helps, needs more" from "data has saturated" — and that is
exactly the decision. Roughly constant pp-per-doubling => keep buying episodes.
Flattening toward zero => the algorithm is the wall, and no affordable budget
closes the gap; go spend the time on phase A (closed-loop memory-GA) or a new
training signal instead.

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
import statistics
import sys
import time

from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.evaluator import (ControllerEvaluator, EpisodeConfig,
                                   fit_thresholds_from_pid_rollouts, fold_pool_seed)
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


def _score_one(genome, spec, ec, report_seed, a):
	"""Held-out score on ONE report seed, fold-0 pool — the 29/07 shape."""
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=report_seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=a.episodes, seed=report_seed,
	                         episode_config=ec, thresholds=thr,
	                         rg_config=_rg(report_seed, ec, a, BASE_ROUNDS, BASE_EPISODES),
	                         num_eval_folds=5)
	m = ev.score_genomes([genome])[0]
	return (m.acc * 100.0, m.mean_attitude_error_deg,
	        getattr(m, "mean_steady_error_deg", None))


def _score(genome, spec, ec, a):
	"""Mean±SD across report seeds — WITHOUT error bars a budget slope cannot be
	distinguished from scatter. The toy-budget smoke went 60 -> 70 -> 50 on a single
	seed at 10 eval episodes: pure noise read as a trend. Same test-set axis as the
	classical baselines, so an arm's ±SD is directly comparable to theirs."""
	tris = [_score_one(genome, spec, ec, rs, a) for rs in a.report_seeds]
	def ms(i):
		xs = [t[i] for t in tris if t[i] is not None]
		if not xs:
			return None
		return [statistics.mean(xs),
		        statistics.pstdev(xs) if len(xs) > 1 else 0.0]
	return {"stable": ms(0), "err_deg": ms(1), "steady_deg": ms(2),
	        "per_seed": {str(rs): list(t) for rs, t in zip(a.report_seeds, tris)}}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--winner", required=True)
	# A CURVE, not a single comparison. Two points cannot tell "data helps but needs
	# more" from "data has saturated" — and that distinction IS the decision (keep
	# buying episodes vs redesign the training signal). Three+ multipliers give the
	# shape: near-linear in log(budget) => keep scaling; flattening => algorithm wall.
	ap.add_argument("--scales", type=int, nargs="+", default=[1, 4, 16],
	                help="budget multipliers, each an arm (applied to episodes/round)")
	ap.add_argument("--base-seed", type=int, default=None,
	                help="winner's --base-seed; default parses _sNNNN from the filename")
	ap.add_argument("--report-seeds", type=int, nargs="+",
	                default=[99990101, 99990102, 99990103, 99990104, 99990105],
	                help="score every arm on all of these; gives each point a ±SD")
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
	      f"{a.disturbance} | scoring {a.episodes} eps x {a.steps} steps")
	print(f"# arms (single _evaluate_core pass each): " +
	      ", ".join(f"{m}x={a.rounds}x{a.base_episodes*m}={a.rounds*a.base_episodes*m}eps"
	                for m in a.scales))
	# `saved` FIRST — the retrain arms wipe cells in place, so this is the only
	# chance to score them. NOTE it is NOT the 1x baseline: those cells accumulated
	# across 5 folds AND every GA generation, so saved-vs-retrain mixes data volume
	# with accumulation passes. The data question is retrain-Nx vs retrain-1x.
	def fmt(m):
		st, er = m["stable"], m["err_deg"]
		return f"stable={st[0]:5.1f}±{st[1]:4.1f}  err={er[0]:5.2f}±{er[1]:4.2f}"

	arms["saved"] = {"metrics": _score(genome, spec, ec, a),
	                 "episodes": None, "train_s": None,
	                 "note": "study pipeline: multi-fold, multi-generation accumulation"}
	print(f"saved       : {fmt(arms['saved']['metrics'])}  (accumulated reference)")
	for mult in a.scales:
		eps = a.base_episodes * mult
		g, dt = _train(genome, spec, ec, train_seed, a, a.rounds, eps)
		name = f"retrain-{mult}x"
		arms[name] = {"metrics": _score(g, spec, ec, a),
		              "episodes": a.rounds * eps, "train_s": round(dt, 1)}
		print(f"{name:11} : {fmt(arms[name]['metrics'])}  "
		      f"({a.rounds * eps} eps, train {dt:.0f}s)")
	# Shape read: per-doubling slope between consecutive arms. Flattening => the
	# algorithm is the wall, not the data.
	pts = [(arms[f"retrain-{m}x"]["episodes"],
	        arms[f"retrain-{m}x"]["metrics"]["stable"][0],
	        arms[f"retrain-{m}x"]["metrics"]["stable"][1]) for m in a.scales]
	for (e0, s0, d0), (e1, s1, d1) in zip(pts, pts[1:]):
		doublings = math.log2(e1 / e0) if e0 else 0.0
		if not doublings:
			continue
		slope = (s1 - s0) / doublings
		# A slope is only worth reading if the endpoints separate beyond their spread.
		noise = math.hypot(d0, d1)
		verdict = "REAL" if abs(s1 - s0) > noise else "within noise"
		print(f"# slope {e0}->{e1} eps: {slope:+.1f} pp/doubling  "
		      f"(delta {s1-s0:+.1f} vs noise ±{noise:.1f} -> {verdict})")

	with open(a.out, "w") as f:
		json.dump({"meta": {k: v for k, v in vars(a).items()} |
		           {"base_seed": base, "train_seed": train_seed,
		            "budget_note": "Nx scales episodes_per_round; rounds fixed so the "
		                           "DAgger mixing schedule is unchanged. Each retrain "
		                           "arm is ONE _evaluate_core pass; `saved` accumulated "
		                           "over 5 folds x every GA generation, so compare "
		                           "retrain-arms to EACH OTHER for the data slope."},
		           "arms": arms}, f, indent=1)
	print(f"# wrote {a.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
