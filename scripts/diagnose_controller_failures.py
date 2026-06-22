#!/usr/bin/env python3
"""Diagnostic (22/06/2026): classify a controller winner's held-out FAILURES.

Question: is the ~36% non-stable fraction SOFT (the controller settles to a flat
steady-state attitude OFFSET above the 5° stable threshold, sim never unstable —
the missing-integrator signature) or HARD (the sim diverges, is_unstable fires)?

If failures are overwhelmingly SOFT, the fix is an integral-memory channel (grow
the recurrent state universe), not more encoding/curriculum. If a meaningful slice
is HARD, the controller is losing control authority and the story is different.

Uses ONLY canonical pieces — build_controller + make_wnn_action_fn + run_episode —
and mirrors ControllerEvaluator.evaluate's build-once-then-loop + per-episode
sub-RNG convention (fitness_function), so behavior matches the held-out scorer.
Read-only: no DB, no flow, no training (MEMORY winner is score-only).
"""
from __future__ import annotations

import argparse
import math
import statistics

import numpy as np

from wnn.control._accel import AttitudeSim
from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.evaluator import (
	build_controller, controller_genome_from_arch, fit_thresholds_from_pid_rollouts,
)
from wnn.control.training import EpisodeConfig, make_wnn_action_fn, run_episode

STABLE_DEG = 5.0  # == fitness_function's steady_threshold (radians(5.0))


def classify(res) -> str:
	if res.diverged:
		return "HARD"
	if math.degrees(res.mean_attitude_error_rad) > STABLE_DEG:
		return "SOFT"
	return "STABLE"


def run_seed(action_fn, sim, ec, n, seed):
	"""Mirror fitness_function: master rng -> per-episode sub-rng -> run_episode."""
	rng = np.random.default_rng(seed)
	return [run_episode(action_fn, sim, ec, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))
	        for _ in range(n)]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--winner", default="logs/controller/W2_adaptive_bpf24_20260621/seed0_base20260609/winner.yaml.gz")
	ap.add_argument("--episodes", type=int, default=100)
	ap.add_argument("--steps", type=int, default=500)
	ap.add_argument("--tilt", type=float, default=5.0)
	ap.add_argument("--body-rate", type=float, default=0.5)
	ap.add_argument("--yaw-rate", type=float, default=0.3)
	ap.add_argument("--seeds", type=int, nargs="+", default=[99990001, 99990101, 12345, 67890])
	args = ap.parse_args()

	payload = load_controller_checkpoint(args.winner)
	spec, bg = payload["spec"], payload["best_genome"]
	ec = EpisodeConfig(dt=0.001, steps_per_episode=args.steps,
	                   max_initial_tilt_rad=math.radians(args.tilt),
	                   max_initial_yaw_rad=math.radians(args.tilt),
	                   max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate)
	sim = AttitudeSim()

	def build_action_fn(seed):
		# Mirror _holdout_report: fit thresholds fresh on this seed, materialize the
		# arch genome into a buildable ControllerGenome (carries the evolved MEMORY
		# cells via genome.cells), build the controller. Fresh build per seed resets
		# the recurrent state at the seed boundary (matches the held-out scorer).
		thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
		cg = controller_genome_from_arch(bg, spec, thr)
		return make_wnn_action_fn(build_controller(cg))

	print(f"winner : {args.winner}")
	print(f"spec   : sn={spec.state_neurons} sb={spec.state_bits_per_neuron} "
	      f"ob={spec.output_bits_per_neuron} bpf={spec.bits_per_feature} delta={spec.delta_control}")
	print(f"task   : tilt≤{args.tilt}° body-rate≤{args.body_rate} yaw-rate≤{args.yaw_rate} "
	      f"steps={args.steps}  | STABLE iff mean_err≤{STABLE_DEG}°, SOFT iff >{STABLE_DEG}° & not diverged, "
	      f"HARD iff diverged\n")

	all_rows = []
	hdr = (f"{'seed':>10} | {'stable%':>7} {'SOFT%':>6} {'HARD%':>6} | {'mean_err°':>9} | "
	       f"soft_err°(mean/max)  hard_peakω")
	print(hdr); print("-" * len(hdr))
	agg = {"STABLE": 0, "SOFT": 0, "HARD": 0}
	for s in args.seeds:
		rows = run_seed(build_action_fn(s), sim, ec, args.episodes, s)
		all_rows += rows
		cls = [classify(r) for r in rows]
		n = len(rows)
		c = {k: cls.count(k) for k in agg}
		for k in agg:
			agg[k] += c[k]
		soft = [math.degrees(r.mean_attitude_error_rad) for r, k in zip(rows, cls) if k == "SOFT"]
		hardw = [r.max_omega_norm for r, k in zip(rows, cls) if k == "HARD"]
		me = float(np.mean([math.degrees(r.mean_attitude_error_rad) for r in rows]))
		softstr = f"{np.mean(soft):.2f}/{np.max(soft):.2f}" if soft else "—"
		hardstr = f"{np.mean(hardw):.2f}" if hardw else "—"
		print(f"{s:>10} | {100*c['STABLE']/n:7.1f} {100*c['SOFT']/n:6.1f} {100*c['HARD']/n:6.1f} | "
		      f"{me:9.2f} | {softstr:>17}  {hardstr}")
	N = len(all_rows)
	print("-" * len(hdr))
	print(f"{'TOTAL':>10} | {100*agg['STABLE']/N:7.1f} {100*agg['SOFT']/N:6.1f} {100*agg['HARD']/N:6.1f} |")

	soft_all = sorted(math.degrees(r.mean_attitude_error_rad) for r in all_rows
	                  if (not r.diverged) and math.degrees(r.mean_attitude_error_rad) > STABLE_DEG)
	if soft_all:
		q = lambda p: soft_all[min(len(soft_all) - 1, int(p * len(soft_all)))]
		print(f"\nSOFT-failure mean_err° spread (n={len(soft_all)}): "
		      f"min={soft_all[0]:.2f}  p25={q(.25):.2f}  median={statistics.median(soft_all):.2f}  "
		      f"p75={q(.75):.2f}  max={soft_all[-1]:.2f}")
	fail = agg["SOFT"] + agg["HARD"]
	print(f"\nFAILURES = {100*fail/N:.1f}%  ->  SOFT {100*agg['SOFT']/N:.1f}% (steady-state offset, "
	      f"integral-recoverable)  |  HARD {100*agg['HARD']/N:.1f}% (divergent)")
	if fail:
		print(f"of the failures: {100*agg['SOFT']/fail:.0f}% are SOFT, {100*agg['HARD']/fail:.0f}% are HARD")


if __name__ == "__main__":
	main()
