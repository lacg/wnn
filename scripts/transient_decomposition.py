"""D1/D2 transient decomposition — WHERE does each L4 student lose to its teacher?

For every winner checkpoint (5 teachers x 2 train seeds), on the EXACT held-out
episode pool the markers were scored on (HoldoutDraw fold-0 pool: sample_ics_flat
+ disturbance_stream, report seeds 99990101-05), trace per-step attitude error for
BOTH the student (ram_controller.trace_controller_cpu — shares rollout_one with the
production CPU scorer) and its own classical teacher (trace_classical_baseline —
the trace twin of score_classical_baseline, aggregates asserted equal).

Decomposes err into RECOVERY (0-20%), CRUISE (20-80%), STEADY (80-100%) phases.
Question (D2): does the cruise-phase deficit reproduce across ALL teachers —
i.e. is it the mechanism behind the universal ~1.29 deg student floor?

Usage:
	python scripts/transient_decomposition.py [--episodes 100] [--markers-dir logs/controller/l4teach]
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPORT_SEEDS = [99990101, 99990102, 99990103, 99990104, 99990105]
TEACHER_IDS = {"pid": 0, "lqr": 1, "mpc": 2, "lqi": 3, "mpcof": 4}
PHASES = {"RECOVERY": (0.0, 0.2), "CRUISE": (0.2, 0.8), "STEADY": (0.8, 1.0)}
STABLE_DEG = 5.0


def build_student(winner_path: str, base_seed: int):
	from wnn.control.checkpoint_io import load_controller_checkpoint
	from wnn.control.evaluator import (build_controller, controller_genome_from_arch,
	                                   fit_thresholds_from_pid_rollouts)
	from wnn.seeds import resolve_seed_set
	payload = load_controller_checkpoint(winner_path, skip_population=True)
	spec = payload["spec"]
	# The filename carries the BASE seed; the run's TRAIN seed is DERIVED from it
	# (resolve_seed_set / derive_seeds — e.g. base 31337002 -> train 3072558954).
	# Fitting thresholds on the base instead of the derived train seed misaligns
	# the address function and wrecks the student (s31337003 went 1.58° -> 18°).
	train_seed = resolve_seed_set(base=base_seed, run_index=0).train
	# Winner carries cells -> score-only -> thresholds on the TRAIN seed
	# (unconditional since 03/08/2026 — the threshold-misalignment rule).
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=train_seed)
	return build_controller(controller_genome_from_arch(payload["best_genome"], spec, thr))


def episode_config():
	from wnn.control.training import EpisodeConfig, DisturbanceConfig
	from wnn.control.airframe import Airframe
	# Mirrors the L4 screen recipe: --steps 2000 --tilt 5.0, body/yaw-rate defaults.
	return EpisodeConfig(
		dt=0.001, steps_per_episode=2000,
		max_initial_tilt_rad=math.radians(5.0),
		max_initial_yaw_rad=math.radians(5.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=DisturbanceConfig.preset("L4C", seed=911),
		airframe=Airframe.preset("cf21_brushless"),
	)


def pool_fields(ec, report_seed: int, episodes: int):
	"""(q0, w0, dist_kwargs) for the fold-0 held-out pool — the classical_baseline
	convention that reproduced the published baselines."""
	from wnn.control.classical_baseline import HoldoutDraw, _episode_fields
	draw = HoldoutDraw(seed=report_seed, episodes=episodes, steps=ec.steps_per_episode)
	return _episode_fields(ec, draw)


def af_to_ctrl_kwargs(af_kw: dict) -> dict:
	"""af_* names -> trace_controller_cpu's plain names (subset it takes)."""
	return dict(dt=af_kw["af_dt"], arm_length=af_kw["af_arm_length"],
	            k_thrust=af_kw["af_k_thrust"], k_drag=af_kw["af_k_drag"],
	            inertia=af_kw["af_inertia"], gravity=af_kw["af_gravity"])


def phase_stats(traces: list, steps: int) -> dict:
	"""Per-phase mean err (deg) + full triple (err/stable/steady) over episodes."""
	out = {p: [] for p in PHASES}
	full, steady_t, stable = [], [], 0
	tail_start = math.ceil(steps * 0.80)
	for tr in traces:
		e = np.degrees(np.array(tr))
		if e.size == 0:
			continue
		diverged = e.size < steps
		full.append(e.mean())
		if not diverged and e.mean() <= STABLE_DEG:
			stable += 1
		if e.size > tail_start:
			steady_t.append(e[tail_start:].mean())
		for p, (a, b) in PHASES.items():
			lo, hi = int(a * steps), int(b * steps)
			seg = e[lo:min(hi, e.size)]
			if seg.size:
				out[p].append(seg.mean())
	n = max(len(full), 1)
	return {
		"phases": {p: float(np.mean(v)) if v else float("nan") for p, v in out.items()},
		"err": float(np.mean(full)) if full else float("nan"),
		"stable": 100.0 * stable / n,
		"steady": float(np.mean(steady_t)) if steady_t else float("nan"),
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--episodes", type=int, default=100,
	                help="episodes per report seed (held-out used 100)")
	ap.add_argument("--markers-dir", type=str, default="logs/controller/l4teach")
	ap.add_argument("--only", type=str, default=None,
	                help="comma list of teacher names to restrict to (e.g. mpcof,lqr)")
	args = ap.parse_args()

	import ram_controller as rc
	ec = episode_config()
	# airframe_kwargs() already carries the firmware PID cascade (af_pid_*) fields.
	af_kw = ec.airframe_kwargs()
	af_full = dict(af_kw)

	pat = re.compile(r"L4T_(\w+?)_cf21_brushless_L4C_s(\d+)_winner\.yaml\.gz$")
	winners = sorted(glob.glob(os.path.join(args.markers_dir, "L4T_*_winner.yaml.gz")))
	only = set(args.only.split(",")) if args.only else None

	rows = []
	for w in winners:
		m = pat.search(w)
		if not m:
			continue
		teacher, train_seed = m.group(1), int(m.group(2))
		if only and teacher not in only:
			continue
		student = build_student(w, train_seed)
		s_traces, t_traces = [], []
		for rs in REPORT_SEEDS:
			q0, w0, dist_kw = pool_fields(ec, rs, args.episodes)
			s_traces += rc.trace_controller_cpu(
				student, q0, w0, args.episodes, ec.steps_per_episode,
				target=[0.0, 0.0, 0.0], **af_to_ctrl_kwargs(af_kw), **dist_kw)
			st, err, steady, traces = rc.trace_classical_baseline(
				TEACHER_IDS[teacher], q0, w0, ec.steps_per_episode, STABLE_DEG,
				**dist_kw, **af_full)
			# Self-check: the trace twin must agree with the production scorer.
			st0, err0, steady0 = rc.score_classical_baseline(
				TEACHER_IDS[teacher], q0, w0, ec.steps_per_episode, STABLE_DEG,
				**dist_kw, **af_full)
			assert abs(err - err0) < 1e-9 and abs(st - st0) < 1e-12, \
				f"trace twin drift: {err} vs {err0}"
			t_traces += traces
		s = phase_stats(s_traces, ec.steps_per_episode)
		t = phase_stats(t_traces, ec.steps_per_episode)
		rows.append((teacher, train_seed, s, t))
		print(f"\n=== {teacher} s{train_seed}  ({len(s_traces)} episodes x "
		      f"{len(REPORT_SEEDS)} report seeds) ===")
		print(f"  student: err {s['err']:.2f}° / stable {s['stable']:.1f}% / steady {s['steady']:.2f}°")
		print(f"  teacher: err {t['err']:.2f}° / stable {t['stable']:.1f}% / steady {t['steady']:.2f}°")
		for p in PHASES:
			sv, tv = s["phases"][p], t["phases"][p]
			ratio = sv / tv if tv > 1e-9 else float("inf")
			print(f"  {p:<9} student {sv:6.2f}°  teacher {tv:6.2f}°  ratio {ratio:5.2f}x")

	print("\n\n==== SUMMARY (student/teacher err ratio per phase; triple per side) ====")
	print(f"{'run':<16}{'RECOV':>7}{'CRUISE':>8}{'STEADY':>8}   "
	      f"{'student e/s/st':>22}   {'teacher e/s/st':>22}")
	for teacher, seed, s, t in rows:
		r = [s['phases'][p] / t['phases'][p] if t['phases'][p] > 1e-9 else float('inf')
		     for p in PHASES]
		print(f"{teacher + ' s' + str(seed):<16}{r[0]:>6.2f}x{r[1]:>7.2f}x{r[2]:>7.2f}x   "
		      f"{s['err']:5.2f}°/{s['stable']:5.1f}%/{s['steady']:4.2f}°   "
		      f"{t['err']:5.2f}°/{t['stable']:5.1f}%/{t['steady']:4.2f}°")


if __name__ == "__main__":
	main()
