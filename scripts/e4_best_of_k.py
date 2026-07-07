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
	("lowedge_s16_in12_s09", 85.8, "logs/controller/LowEdge_20260701/s16_in12_seed20260609/winner.yaml.gz"),
	("stateint_B_int_s09", 84.2, "logs/controller/StateIntegral_20260701/B_integral_seed20260609/winner.yaml.gz"),
	# E2 LONG arm: trained + recorded at steps=2000 (88.8 on the 2000-step ho metric);
	# scoring HERE runs the standard 500-step protocol = the cross-arm comparable number.
	("e2_long_s09", 88.8, "logs/controller/E2Reliability_20260702/LONG_seed20260609/winner.yaml.gz"),
	# E2 ANCH arm: yaw-anchor + immigrants + 30 gens — 91.0±4.6 report-seed ho-mem,
	# the first single >90 on the standard protocol (02/07).
	("e2_anch_s09", 91.0, "logs/controller/E2Reliability_20260702/ANCH_seed20260609/winner.yaml.gz"),
	# E2 LONG seed10 (trained @2000; 88.0±4.1 on the 2000-step ho — recipe reproduces
	# across seeds, pooled 88.4±3.7). Second free C2K member.
	("e2_long_s10", 88.0, "logs/controller/E2Reliability_20260702/LONG_seed20260610/winner.yaml.gz"),
	# C2K pool (trained @2000; recorded = each cell's MEMORY 4-seed ho on its
	# OWN-horizon ruler — Finding 6, 04/07). pwm2k = the pooled-90.5 winner.
	("pwm2k_s09",  91.8, "logs/controller/C2K_20260702/PWM2K_seed20260609/winner.yaml.gz"),
	("pwm2k_s10",  89.2, "logs/controller/C2K_20260702/PWM2K_seed20260610/winner.yaml.gz"),
	("lean2k_s09", 93.5, "logs/controller/C2K_20260702/LEAN2K_seed20260609/winner.yaml.gz"),
	("lean2k_s10", 82.5, "logs/controller/C2K_20260702/LEAN2K_seed20260610/winner.yaml.gz"),
	("tilt2k_s09", 89.2, "logs/controller/C2K_20260702/TILT2K_seed20260609/winner.yaml.gz"),
	("tilt2k_s10", 74.0, "logs/controller/C2K_20260702/TILT2K_seed20260610/winner.yaml.gz"),
	("anch2k_s09", 60.0, "logs/controller/C2K_20260702/ANCH2K_seed20260609/winner.yaml.gz"),
	("anch2k_s10", 87.0, "logs/controller/C2K_20260702/ANCH2K_seed20260610/winner.yaml.gz"),
	# W1 horizon-surface cells (trained @1000/@4000; recorded = MEMORY 4-seed ho
	# on own-horizon ruler — surface finalized 05/07 23:10Z, band peaks @2000).
	("w1_h1000_s09", 84.0, "logs/controller/W1Surface_20260702/H1000_seed20260609/winner.yaml.gz"),
	("w1_h1000_s10", 85.0, "logs/controller/W1Surface_20260702/H1000_seed20260610/winner.yaml.gz"),
	("w1_h4000_s09", 80.5, "logs/controller/W1Surface_20260702/H4000_seed20260609/winner.yaml.gz"),
	("w1_h4000_s10", 87.5, "logs/controller/W1Surface_20260702/H4000_seed20260610/winner.yaml.gz"),
	# W2.3 train-under-weather winners (PWM2K recipe @2000 + L1 armed in ALL
	# training rollouts; recorded = MEMORY 4-seed ho UNDER L1 — gate 80.2
	# beaten 07/07: 93.5/89.2, pooled ~91.4).
	("w23_pwm2k_L1_s09", 93.5, "logs/controller/W23Weather_20260706/PWM2K_L1_seed20260609/winner.yaml.gz"),
	("w23_pwm2k_L1_s10", 89.2, "logs/controller/W23Weather_20260706/PWM2K_L1_seed20260610/winner.yaml.gz"),
]


def episode_config() -> EpisodeConfig:
	# The standard sweep protocol: --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 --steps 500.
	# E4_STEPS overrides episode length (e.g. 2000 for the train×eval matrix cell that
	# separates "better controller" from "more forgiving ruler" — 02/07 LONG question).
	# E4_DIST=L1|L2|L3 arms W2 weather (W2.2 brittleness audit); default OFF.
	import os
	steps = int(os.environ.get("E4_STEPS", "500"))
	dist = None
	lv = os.environ.get("E4_DIST", "OFF")
	if lv and lv.strip().upper() not in ("OFF", "", "NONE"):
		from wnn.control.training import DisturbanceConfig
		dist = DisturbanceConfig.preset(lv, seed=911)
	return EpisodeConfig(
		dt=0.001, steps_per_episode=steps,
		max_initial_tilt_rad=math.radians(5.0),
		max_initial_yaw_rad=math.radians(5.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=dist,
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
	if meta.get("tilt_deg") not in (None, 5.0):
		print(f"  [{label}] protocol mismatch (tilt={meta.get('tilt_deg')}) — skipping")
		return None
	if meta.get("steps") not in (None, ec.steps_per_episode):
		# Trained at a different episode length than this run's scoring ruler
		# (E4_STEPS) — the recorded number is not comparable.
		print(f"  [{label}] note: trained at steps={meta.get('steps')}; scoring at {ec.steps_per_episode} (recorded ho not comparable)")
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


def ensemble_action_fn(controllers, agg: str = "mean"):
	"""Combine member PWM outputs per motor. agg='mean' (committee average) or
	'median' (middle vote — robust to a single member going wild)."""
	def fn(gyro, accel, target_rpy, q):
		outs = [c.step(list(gyro), list(accel), list(target_rpy)) for c in controllers]
		if agg == "median":
			return tuple(statistics.median(o[i] for o in outs) for i in range(4))
		n = float(len(controllers))
		return tuple(sum(o[i] for o in outs) / n for i in range(4))
	return fn


def _ensemble_ics(rs: int, ec: EpisodeConfig):
	"""Pre-draw episode ICs with EXACTLY dagger.eval_closed_loop_reset's numpy
	chain (outer rng -> per-episode integer -> ep_rng -> _sample_initial_state),
	so the Rust hot loop reproduces the Python-path fresh-seed numbers."""
	import numpy as np
	from wnn.control.training import _sample_initial_state
	rng = np.random.default_rng(rs)
	qs, oms = [], []
	for _ in range(EPISODES_PER_SEED):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q, om = _sample_initial_state(
			ep_rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
			ec.max_initial_body_rate, ec.max_initial_yaw_rate)
		qs.extend(float(v) for v in q)
		oms.extend(float(v) for v in om)
	return qs, oms


def score_ensemble(top, ec: EpisodeConfig):
	agg = __import__("os").environ.get("E4_ENSEMBLE_AGG", "mean")  # mean | median
	import ram_controller as _rc
	use_rust = hasattr(_rc, "eval_ensemble_closed_loop")
	print(f"\n--- ensemble of top-{len(top)}: {[t['label'] for t in top]} ({agg} PWM, "
	      f"{'RUST' if use_rust else 'python-fallback'} loop) ---")
	rows = []
	for rs in FRESH_SEEDS:
		controllers = []
		for t in top:
			thr = fit_thresholds_from_pid_rollouts(t["spec"], num_episodes=10, seed=rs)
			controllers.append(build_controller(controller_genome_from_arch(t["genome"], t["spec"], thr)))
		if use_rust:
			qs, oms = _ensemble_ics(rs, ec)
			# W2: thread ec.disturbance into the Rust hot loop. Every parameter
			# passed explicitly by name (the pyo3 side is fully typed; disabled
			# values = pre-W2 behavior). Note the ensemble path uses the FIXED
			# motor_asym only (no per-episode mag draw).
			d = getattr(ec, "disturbance", None)
			stable, err_deg, steady_deg = _rc.eval_ensemble_closed_loop(
				controllers, qs, oms, ec.steps_per_episode, agg == "median", 5.0,
				dist_enabled=d is not None,
				dist_tau_bias=list(d.tau_bias) if d else [0.0, 0.0, 0.0],
				dist_gust_sigma=d.gust_sigma if d else 0.0,
				dist_gust_tau_c=d.gust_tau_c if d else 0.1,
				dist_motor_asym=list(d.motor_asym) if d else [1.0, 1.0, 1.0, 1.0],
				dist_gyro_sigma=d.gyro_sigma if d else 0.0,
				dist_gyro_bias_walk=d.gyro_bias_walk if d else 0.0,
				dist_accel_sigma=d.accel_sigma if d else 0.0,
				dist_seed=d.seed if d else 0,
			)
			rows.append({"seed": rs, "stable": stable * 100.0, "err": err_deg,
			             "steady": steady_deg})
		else:
			def reset_all():
				for c in controllers:
					c.reset()
			_, m = eval_closed_loop_reset(ensemble_action_fn(controllers, agg), reset_all,
			                              ec, EPISODES_PER_SEED, rs)
			# dict path: stable_rate + mean_attitude_error_deg (no steady).
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
	dist_lv = os.environ.get("E4_DIST", "OFF")
	print(f"fresh seeds {FRESH_SEEDS} x {EPISODES_PER_SEED} eps | tilt<=5° body<=0.5 yaw-rate<=0.3 steps={ec.steps_per_episode} dist={dist_lv}")
	print(f"candidates: {len(CANDIDATES)} (selected on the RECORDED report-seed ho-mem)\n")
	skip_solo = os.environ.get("E4_SKIP_SOLO") == "1"  # ensemble-only: load winners, no member rescore
	results = []
	for label, recorded, path in CANDIDATES:
		print(f"[{label}] recorded ho-mem {recorded:.1f}% — loading {path}", flush=True)
		if skip_solo:
			payload = load_controller_checkpoint(path)
			if payload is None or getattr(payload.get("best_genome"), "cells", None) is None:
				print(f"  [{label}] load failed / no cells — skipping")
				continue
			results.append({"label": label, "path": path, "stable": recorded, "sd": 0.0,
			                "err": float("nan"), "steady": float("nan"), "recorded": recorded,
			                "spec": payload["spec"], "genome": payload["best_genome"]})
		else:
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
