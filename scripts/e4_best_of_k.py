#!/usr/bin/env python3
"""E4 — best-of-K + small ensemble over EXISTING saved controller winners.

Deployment-honest protocol (plan: .claude/plans/controller_break_90_v2.md):
- SELECTION already happened on the standard report seeds (the recorded ho-mem
  numbers each run printed). We take the top-K recorded winners as candidates.
- FINAL number = re-score each candidate on FRESH seeds never used anywhere
  (train seeds 202606xx; report seeds 99990001/99990101/12345/67890) so the
  best-of-K claim carries no selection leak.
- ENSEMBLE arm: average the PWM outputs of the top-3 (by fresh pooled stable)
  each step — the FPGA-cheap "committee of 1-KB controllers".

Read-only w.r.t. the running drivers (no training, no GA). Uses each winner's
OWN spec (obs features, levels, bpf) from its checkpoint payload.

Usage: PYTHONPATH=src/wnn /Users/lacg/wnn-venv/bin/python scripts/e4_best_of_k.py
"""

import gc
import math
import statistics

from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.dagger import eval_closed_loop_reset
from wnn.control.evaluator import (
	ControllerEvaluator, build_controller, controller_genome_from_arch,
	fit_thresholds_from_pid_rollouts,
)
from wnn.control.training import EpisodeConfig

# Fresh seeds — never used as train/fold/report seeds anywhere in the line.
FRESH_SEEDS = [77770001, 77770101, 424242, 313131]
EPISODES_PER_SEED = 100
ENSEMBLE_TOP = int(__import__("os").environ.get("E4_ENSEMBLE_TOP", "3"))

# (label, recorded ho-mem stable%, path) — selection ALREADY made on the
# standard report seeds; recorded numbers shown for the fresh-vs-recorded gap.
CANDIDATES = [
	("pidmix_s10_R1",      90.0, "logs/controller/FrameFixVal_20260627/pidmix_seed20260610/winner.yaml.gz"),
	("pidmix_pwm_s09_R1",  87.2, "logs/controller/FrameFixVal_20260627/pidmix_pwm_seed20260609/winner.yaml.gz"),
	("s16_s10_R1",         86.2, "logs/controller/FrameFixVal_20260627/s16_seed20260610/winner.yaml.gz"),
	("pidmix_pwm_s09_R2",  85.2, "logs/controller/FrameFixBits_20260627/pidmix_pwm_seed20260609/winner.yaml.gz"),
	("pwm_s09_R1",         85.0, "logs/controller/FrameFixVal_20260627/pwm_seed20260609/winner.yaml.gz"),
	("pwm_s10_R2",         85.0, "logs/controller/FrameFixBits_20260627/pwm_seed20260610/winner.yaml.gz"),
	("lowedge_s16_in4_s09", 85.0, "logs/controller/LowEdge_20260701/s16_in4_seed20260609/winner.yaml.gz"),
	("stateint_A_ctrl_s09", 84.8, "logs/controller/StateIntegral_20260701/A_ctrl_seed20260609/winner.yaml.gz"),
	("s16_s09_R2",         84.0, "logs/controller/FrameFixBits_20260627/s16_seed20260609/winner.yaml.gz"),
	("bitsweep_b24_s10",   83.2, "logs/controller/BitSweep_pidmix_pwm_20260630/pidmix_pwm_b24_seed20260610/winner.yaml.gz"),
]


def episode_config() -> EpisodeConfig:
	# The standard sweep protocol: --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 --steps 500
	return EpisodeConfig(
		dt=0.001, steps_per_episode=500,
		max_initial_tilt_rad=math.radians(5.0),
		max_initial_yaw_rad=math.radians(5.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)


def pool(vals):
	m = statistics.mean(vals)
	sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
	return m, sd


def score_candidate(label: str, path: str, ec: EpisodeConfig):
	payload = load_controller_checkpoint(path)
	if payload is None:
		print(f"  [{label}] LOAD FAILED — skipping")
		return None
	spec, genome = payload["spec"], payload["best_genome"]
	meta = payload.get("meta", {})
	if meta.get("tilt_deg") not in (None, 5.0) or meta.get("steps") not in (None, 500):
		print(f"  [{label}] protocol mismatch (tilt={meta.get('tilt_deg')} steps={meta.get('steps')}) — skipping")
		return None
	if getattr(genome, "cells", None) is None:
		print(f"  [{label}] winner carries NO cells — skipping (arch-only checkpoint)")
		return None
	rows = []
	for rs in FRESH_SEEDS:
		thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=rs)
		ev = ControllerEvaluator(spec, num_eval_episodes=EPISODES_PER_SEED, seed=rs,
		                         episode_config=ec, thresholds=thr)
		m = ev.score_genomes([genome])[0]
		rows.append({
			"seed": rs, "stable": m.acc * 100.0,
			"err": m.mean_attitude_error_deg,
			"steady": getattr(m, "mean_steady_error_deg", float("nan")),
		})
		del ev
	sm, ssd = pool([r["stable"] for r in rows])
	em, esd = pool([r["err"] for r in rows])
	tm, tsd = pool([r["steady"] for r in rows])
	per_seed = "  ".join(f"{r['stable']:.0f}" for r in rows)
	print(f"  [{label}] fresh POOL: stable {sm:5.1f}±{ssd:4.1f}%  err {em:.2f}±{esd:.2f}°  "
	      f"steady {tm:.2f}±{tsd:.2f}°   (per-seed: {per_seed})", flush=True)
	return {"label": label, "path": path, "stable": sm, "sd": ssd, "err": em,
	        "steady": tm, "spec": spec, "genome": genome}


def ensemble_action_fn(controllers):
	def fn(gyro, accel, target_rpy, q):
		acc = [0.0, 0.0, 0.0, 0.0]
		for c in controllers:
			pwm = c.step(list(gyro), list(accel), list(target_rpy))
			for i in range(4):
				acc[i] += pwm[i]
		n = float(len(controllers))
		return tuple(v / n for v in acc)
	return fn


def score_ensemble(top, ec: EpisodeConfig):
	print(f"\n--- ensemble of top-{len(top)}: {[t['label'] for t in top]} (mean PWM) ---")
	rows = []
	for rs in FRESH_SEEDS:
		controllers = []
		for t in top:
			thr = fit_thresholds_from_pid_rollouts(t["spec"], num_episodes=10, seed=rs)
			controllers.append(build_controller(controller_genome_from_arch(t["genome"], t["spec"], thr)))
		def reset_all():
			for c in controllers:
				c.reset()
		_, m = eval_closed_loop_reset(ensemble_action_fn(controllers), reset_all,
		                              ec, EPISODES_PER_SEED, rs)
		# eval_closed_loop_reset returns a plain dict: stable_rate +
		# mean_attitude_error_deg (no steady on this path).
		rows.append({"seed": rs, "stable": m["stable_rate"] * 100.0,
		             "err": m["mean_attitude_error_deg"],
		             "steady": float("nan")})
		print(f"  seed {rs:>9}: stable {rows[-1]['stable']:5.1f}%  err {rows[-1]['err']:.2f}°  "
		      f"steady {rows[-1]['steady']:.2f}°", flush=True)
		del controllers
		gc.collect()
	sm, ssd = pool([r["stable"] for r in rows])
	em, esd = pool([r["err"] for r in rows])
	print(f"  ENSEMBLE POOL: stable {sm:5.1f}±{ssd:4.1f}%  err {em:.2f}±{esd:.2f}°")
	return sm, ssd


def main() -> None:
	import os
	ec = episode_config()
	only = os.environ.get("E4_ONLY")  # comma-separated labels → restrict (e.g. ensemble re-run)
	global CANDIDATES
	if only:
		keep = {s.strip() for s in only.split(",")}
		CANDIDATES = [c for c in CANDIDATES if c[0] in keep]
	print("========== E4 best-of-K over saved winners — FRESH-seed protocol ==========")
	print(f"fresh seeds {FRESH_SEEDS} x {EPISODES_PER_SEED} eps | tilt<=5° body<=0.5 yaw-rate<=0.3 steps=500")
	print(f"candidates: {len(CANDIDATES)} (selected on the RECORDED report-seed ho-mem)\n")
	results = []
	for label, recorded, path in CANDIDATES:
		print(f"[{label}] recorded ho-mem {recorded:.1f}% — loading {path}", flush=True)
		r = score_candidate(label, path, ec)
		if r is not None:
			r["recorded"] = recorded
			results.append(r)
		gc.collect()
	if not results:
		print("no candidates scored — nothing to report")
		return
	results.sort(key=lambda r: r["stable"], reverse=True)
	print("\n================= RANKING (fresh seeds) =================")
	print(f"{'label':<22} {'recorded':>8} {'fresh':>12} {'err':>7} {'steady':>7}")
	for r in results:
		print(f"{r['label']:<22} {r['recorded']:7.1f}% {r['stable']:6.1f}±{r['sd']:4.1f}% "
		      f"{r['err']:6.2f}° {r['steady']:6.2f}°")
	best = results[0]
	print(f"\nBEST-OF-K (fresh): {best['label']}  stable {best['stable']:.1f}±{best['sd']:.1f}%  "
	      f"err {best['err']:.2f}°  steady {best['steady']:.2f}°")
	if len(results) >= 2:
		score_ensemble(results[:min(ENSEMBLE_TOP, len(results))], ec)
	print("\nPID anchor on this protocol: 100.0% / 2.28° / 0.89° (scripts/pid_ki_ablation.py)")


if __name__ == "__main__":
	main()
