#!/usr/bin/env python3
"""Cross-teacher winner ensemble harness (task #6, teacher-hybrids-roadmap).

Scores 2+ saved phased-GA winners (winner.yaml.gz) SOLO and as a PWM committee
(mean / median vote per motor, Rust eval_ensemble_closed_loop hot loop) on
held-out seeds. Protocol mirrors scripts/e4_best_of_k.py — per-seed threshold
re-fit from PID rollouts, IC pre-draw with dagger.eval_closed_loop_reset's
numpy chain — so numbers are comparable to the E4/C2K lines. Default
steps=1000 matches the teacher-run protocol (LQR/PID/MPC screening + full).

Usage:
  PYTHONPATH=src/wnn python scripts/ensemble_teachers.py \
    --winners lqr=logs/controller/c10_lqr_teacher_20260708/seed0_base31337002_SCREENING_p32/winner.yaml.gz \
              mpc=logs/controller/c10_mpc_teacher_20260708/seed0_base31337002_SCREENING_p32/winner.yaml.gz \
              pid=logs/controller/c10_pid_teacher_20260710/seed0_base31337002_SCREENING_p32/winner.yaml.gz \
    [--pairs] [--agg both|mean|median] [--steps 1000] [--episodes 100] \
    [--seeds 99990001,99990101,12345,67890]
"""

import argparse
import gc
import math
import statistics
from itertools import combinations

from wnn.control import _accel as ra
from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.evaluator import (
	ControllerEvaluator, build_controller, controller_genome_from_arch,
	fit_thresholds_from_pid_rollouts,
)
from wnn.control.training import EpisodeConfig


def parse_args():
	ap = argparse.ArgumentParser(description="Solo + PWM-committee scoring of saved controller winners")
	ap.add_argument("--winners", nargs="+", required=True,
	                help="label=path pairs (2+); path = a schema-2 winner.yaml.gz")
	ap.add_argument("--seeds", type=str, default="99990001,99990101,12345,67890",
	                help="comma list of held-out report seeds")
	ap.add_argument("--episodes", type=int, default=100, help="episodes per seed")
	ap.add_argument("--steps", type=int, default=1000,
	                help="episode length; 1000 = the teacher-run protocol")
	ap.add_argument("--tilt", type=float, default=5.0)
	ap.add_argument("--body-rate", type=float, default=0.5)
	ap.add_argument("--yaw-rate", type=float, default=0.3)
	ap.add_argument("--agg", choices=["both", "mean", "median"], default="both",
	                help="committee vote per motor")
	ap.add_argument("--pairs", action="store_true",
	                help="also score every 2-member committee (default: full committee only)")
	return ap.parse_args()


def episode_config(args) -> EpisodeConfig:
	return EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate,
	)


def pool(vals):
	m = statistics.mean(vals)
	sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
	return m, sd


def load_winner(spec_str: str) -> dict:
	"""Load label=path WITHOUT materializing the embedded population (full-pop
	winner checkpoints are 100s of MB → tens of GB in RAM → jetsam next to the
	IDS worker; 10/07 incident). Fast path: extract_checkpoint_head streams the
	gz text only up to final_population (seconds) and caches the reduced doc as
	winner_only.yaml.gz beside the original; falls back to the event-filtered
	skip_population loader for non-standard files."""
	from pathlib import Path
	from wnn.ram.strategies.phased.checkpoint import extract_checkpoint_head
	label, _, path = spec_str.partition("=")
	if not path:
		raise SystemExit(f"--winners entry '{spec_str}' is not label=path")
	cache = Path(path).parent / "winner_only.yaml.gz"
	if not cache.exists():
		print(f"  [{label}] extracting winner-only head from {path}", flush=True)
		try:
			extract_checkpoint_head(path, cache)
		except ValueError as e:
			print(f"  [{label}] head-extract failed ({e}); event-filtered slow path", flush=True)
			from wnn.control.checkpoint_io import save_controller_checkpoint
			payload = load_controller_checkpoint(path, skip_population=True)
			if payload is None:
				raise SystemExit(f"[{label}] load failed: {path}")
			payload["population"] = []
			save_controller_checkpoint(str(cache), payload)
	payload = load_controller_checkpoint(str(cache), skip_population=True)
	if payload is None:
		raise SystemExit(f"[{label}] load failed: {cache}")
	genome = payload["best_genome"]
	if getattr(genome, "cells", None) is None:
		raise SystemExit(f"[{label}] winner carries NO cells (arch-only checkpoint): {cache}")
	meta = payload.get("meta", {})
	return {"label": label, "path": path, "spec": payload["spec"], "genome": genome,
	        "trained_steps": meta.get("steps")}


def draw_ics(seed: int, episodes: int, ec: EpisodeConfig):
	"""Pre-draw episode ICs with EXACTLY dagger.eval_closed_loop_reset's numpy
	chain so the Rust hot loop reproduces the Python-path numbers."""
	import numpy as np
	from wnn.control.training import _sample_initial_state
	rng = np.random.default_rng(seed)
	qs, oms = [], []
	for _ in range(episodes):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q, om = _sample_initial_state(
			ep_rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
			ec.max_initial_body_rate, ec.max_initial_yaw_rate)
		qs.extend(float(v) for v in q)
		oms.extend(float(v) for v in om)
	return qs, oms


def solo_score(member: dict, seeds, episodes: int, ec: EpisodeConfig) -> dict:
	rows = []
	for rs in seeds:
		thr = fit_thresholds_from_pid_rollouts(member["spec"], num_episodes=10, seed=rs)
		ev = ControllerEvaluator(member["spec"], num_eval_episodes=episodes, seed=rs,
		                         episode_config=ec, thresholds=thr)
		m = ev.score_genomes([member["genome"]])[0]
		rows.append({"stable": m.acc * 100.0, "err": m.mean_attitude_error_deg,
		             "steady": getattr(m, "mean_steady_error_deg", float("nan"))})
		del ev
		gc.collect()
	return summarize(member["label"], rows)


def committee_score(members, agg: str, seeds, episodes: int, ec: EpisodeConfig) -> dict:
	label = "+".join(m["label"] for m in members) + f" ({agg})"
	rows = []
	for rs in seeds:
		controllers = []
		for m in members:
			thr = fit_thresholds_from_pid_rollouts(m["spec"], num_episodes=10, seed=rs)
			controllers.append(build_controller(controller_genome_from_arch(m["genome"], m["spec"], thr)))
		qs, oms = draw_ics(rs, episodes, ec)
		stable, err_deg, steady_deg = ra.eval_ensemble_closed_loop(
			controllers, qs, oms, ec.steps_per_episode, agg == "median", 5.0,
			dist_enabled=False, dist_tau_bias=[0.0, 0.0, 0.0],
			dist_gust_sigma=0.0, dist_gust_tau_c=0.1,
			dist_motor_asym=[1.0, 1.0, 1.0, 1.0], dist_gyro_sigma=0.0,
			dist_gyro_bias_walk=0.0, dist_accel_sigma=0.0, dist_seed=0,
		)
		rows.append({"stable": stable * 100.0, "err": err_deg, "steady": steady_deg})
		del controllers
		gc.collect()
	return summarize(label, rows)


def summarize(label: str, rows) -> dict:
	sm, ssd = pool([r["stable"] for r in rows])
	em, esd = pool([r["err"] for r in rows])
	tm, tsd = pool([r["steady"] for r in rows])
	per_seed = "  ".join(f"{r['stable']:.0f}" for r in rows)
	print(f"  [{label:<28}] stable {sm:5.1f}±{ssd:4.1f}%  err {em:.2f}±{esd:.2f}°  "
	      f"steady {tm:.2f}±{tsd:.2f}°   (per-seed: {per_seed})", flush=True)
	return {"label": label, "stable": sm, "sd": ssd, "err": em, "steady": tm}


def committee_sets(members, want_pairs: bool):
	sets = []
	if want_pairs and len(members) > 2:
		sets.extend(combinations(members, 2))
	if len(members) >= 2:
		sets.append(tuple(members))
	return sets


def main():
	args = parse_args()
	if not hasattr(ra, "eval_ensemble_closed_loop"):
		raise SystemExit("ram_controller wheel lacks eval_ensemble_closed_loop — rebuild the controller wheel")
	seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
	ec = episode_config(args)
	members = [load_winner(w) for w in args.winners]
	for m in members:
		if m["trained_steps"] not in (None, args.steps):
			print(f"  [{m['label']}] note: trained at steps={m['trained_steps']}, scoring at {args.steps}")

	print(f"Ensemble harness: {len(members)} members, {len(seeds)} seeds × {args.episodes} eps × {args.steps} steps, tilt {args.tilt}°")
	print("\n--- SOLO (per-member held-out) ---")
	results = [solo_score(m, seeds, args.episodes, ec) for m in members]

	aggs = ["mean", "median"] if args.agg == "both" else [args.agg]
	print("\n--- COMMITTEES (PWM vote, Rust hot loop) ---")
	for combo in committee_sets(members, args.pairs):
		for agg in aggs:
			results.append(committee_score(list(combo), agg, seeds, args.episodes, ec))

	print("\n--- RANKING (by pooled stable) ---")
	for r in sorted(results, key=lambda r: -r["stable"]):
		print(f"  {r['stable']:5.1f}±{r['sd']:4.1f}%  err {r['err']:.2f}°  steady {r['steady']:.2f}°  {r['label']}")


if __name__ == "__main__":
	main()
