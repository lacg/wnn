"""True held-out validation of a saved Plan A genome (stage checkpoint or final winner).

The phased-GA's per-gen "best" and "stage done" numbers are measured on folds
the GA itself trained on (K-fold rotation prevents single-pool overfit but
doesn't equal held-out). This script measures the genome's performance on a
COMPLETELY FRESH seed pool that the GA never saw — the honest generalization
number.

How it works:
  1. Load a pickle written by `tests/run_phased_ga.py --save-winner` or
     `--save-stage-checkpoints DIR`. The pickle has: spec, best_genome,
     population, metrics (training-side), fitness_weights, meta.
  2. Build a new ControllerEvaluator with a DIFFERENT base seed (default:
     pickle's saved_at hash + a magic offset, or user-specified --base-seed).
     With --num-eval-folds 1, this evaluator uses a single fresh pool.
  3. Re-evaluate the winning genome:
       - Arch-stage genomes (Stage 1/2/3) have cells=None → use
         `evaluate_batch` (trains cells via reward-gated adaptation, then scores).
       - Memory-stage genomes (Stage 4) have cells populated → use
         `score_genomes` (no training; just write cells + score).
  4. Print: pickle metrics (training-side) | held-out metrics | gap.

Usage:
  # Stage 1 winner held-out:
  python3 scripts/evaluate_winner_holdout.py \\
    --winner logs/controller/planAB/v3_stages_20260530_220000/stage1_neurons.pkl

  # Stage 4 / final winner held-out with explicit seed:
  python3 scripts/evaluate_winner_holdout.py \\
    --winner logs/controller/planAB/winner_planAv3_<TS>.pkl \\
    --base-seed 20260601 --eval-episodes 200

A 5-15 min run for typical stage genomes (depends on arch size + eval-episodes).
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

# Reach run_phased_ga.py's helpers + the evaluator/eval_closed_loop_reset path.
_THIS_DIR = Path(__file__).parent
_TESTS_DIR = _THIS_DIR.parent / "tests"
if str(_TESTS_DIR) not in sys.path:
	sys.path.insert(0, str(_TESTS_DIR))

# _rg_config / _pid_baseline moved from tests/run_phased_ga.py into the
# control package (commit splitting phased_ga helpers); import from there.
from wnn.control.phased_ga import _rg_config, _pid_baseline   # noqa: E402
from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts  # noqa: E402
from wnn.control.training import EpisodeConfig  # noqa: E402

from wnn.control.checkpoint_io import load_controller_checkpoint as _load_ctl_checkpoint




def _derive_holdout_seed(payload: dict) -> int:
	"""Pick a fresh-seed default — bump the pickle's saved_at_unix hash so it's
	deterministic per-pickle but distinct from anything used during training."""
	import hashlib
	saved = payload.get("meta", {}).get("saved_at_unix", 0)
	# Magic offset: 0xDEADBEEF as decimal — keeps the seed legible in logs
	# and small enough to feed numpy/Python RNGs without truncation drama.
	digest = hashlib.sha1(f"holdout-{saved}".encode()).hexdigest()[:8]
	return int(digest, 16) & 0xFFFFFFFF


def main():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--winner", type=str, required=True,
	                help="Path to pickled winner / stage checkpoint.")
	ap.add_argument("--base-seed", type=int, default=None,
	                help="Held-out base seed (default: derived from pickle meta).")
	ap.add_argument("--eval-episodes", type=int, default=100,
	                help="Episodes for held-out scoring (default 100 = 1%% granularity).")
	ap.add_argument("--steps", type=int, default=None,
	                help="Steps per episode (default: from pickle meta, fallback 500).")
	ap.add_argument("--tilt", type=float, default=None,
	                help="Initial tilt bound in degrees (default: from pickle, fallback 15).")
	# Inner-train knobs — only used for arch-stage (cells=None) genomes. For
	# memory-stage genomes (cells present), training is skipped.
	ap.add_argument("--rg-rounds", type=int, default=3)
	ap.add_argument("--rg-episodes-per-round", type=int, default=6)
	ap.add_argument("--rg-eval-episodes", type=int, default=5)
	ap.add_argument("--train-workers", type=int, default=3)
	ap.add_argument("--universe-episodes", type=int, default=3)
	args = ap.parse_args()

	# Load the pickle.
	payload_path = Path(args.winner)
	if not payload_path.exists():
		print(f"ERROR: --winner path does not exist: {payload_path}", file=sys.stderr)
		return 1
	with open(payload_path, "rb") as f:
		payload = _load_ctl_checkpoint(f.name)
	spec        = payload["spec"]
	best_genome = payload.get("best_genome")
	prev_metrics = payload.get("metrics")
	prev_weights = payload.get("fitness_weights", {})
	meta        = payload.get("meta", {})

	# Schema-2 yaml.gz checkpoints store metrics as a DICT; old .pkl stored a
	# Metrics object. Normalize a dict to attribute access so the printouts below
	# work for both formats (field names match the Metrics dataclass).
	if isinstance(prev_metrics, dict):
		from types import SimpleNamespace
		prev_metrics = SimpleNamespace(
			mean_attitude_error_deg=prev_metrics.get("mean_attitude_error_deg", float("nan")),
			acc=prev_metrics.get("acc", float("nan")),
			fitness=prev_metrics.get("fitness", float("nan")),
			ce=prev_metrics.get("ce", float("nan")),
		)

	if best_genome is None:
		print(f"ERROR: pickle has no best_genome", file=sys.stderr)
		return 1

	# Resolve held-out seed.
	holdout_seed = args.base_seed if args.base_seed is not None else _derive_holdout_seed(payload)

	# Resolve episode-config knobs (prefer pickle's recorded values; fall back to args).
	steps = args.steps if args.steps is not None else meta.get("steps", 500)
	tilt  = args.tilt  if args.tilt  is not None else meta.get("tilt_deg", 15.0)
	levels = spec.levels_per_motor

	# Genome paradigm: arch-stage (cells=None → train) vs memory-stage (cells present → score-only).
	has_cells = getattr(best_genome, "cells", None) is not None
	paradigm = "MEMORY (paradigm B, score-only)" if has_cells else "ARCH (paradigm A, reward-gated train)"

	print(f"\n{'='*72}\n  HELD-OUT VALIDATION\n{'='*72}")
	print(f"  Pickle:        {payload_path}")
	print(f"  Spec:          sn={spec.state_neurons} sb={spec.state_bits_per_neuron} "
	      f"ob={spec.output_bits_per_neuron} levels={levels}")
	print(f"  Paradigm:      {paradigm}")
	print(f"  Held-out seed: {holdout_seed}  (different from training base seed)")
	print(f"  Eval episodes: {args.eval_episodes}  (1% stability granularity)")
	print(f"  K-folds:       1  (single fresh pool — no rotation, this IS the held-out test)")
	print(f"  Steps/ep:      {steps}  Tilt: {tilt}°")
	if prev_metrics is not None:
		print(f"\n  TRAINING-SIDE (from pickle): "
		      f"err={prev_metrics.mean_attitude_error_deg:.2f}°  "
		      f"stable={prev_metrics.acc*100:.1f}%  "
		      f"reward={prev_metrics.fitness:.2f}")
	print(f"\n  Re-evaluating on held-out seed...")

	# Build episode config + threshold fit (uses the held-out seed throughout).
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=steps,
		max_initial_tilt_rad=math.radians(tilt),
		max_initial_yaw_rad=math.radians(tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)
	# PID thresholds: re-fit on held-out seed so the scorer's thresholds are
	# matched to held-out distribution rather than carrying training thresholds.
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=holdout_seed)

	# Build the held-out evaluator. num_eval_folds=1: pure single fresh pool.
	# rg_config needs the same args struct shape as run_phased_ga; we forge it.
	class _ArgsForRG: pass
	a = _ArgsForRG()
	for k, v in vars(args).items(): setattr(a, k, v)
	rg = _rg_config(a, ec, holdout_seed)

	ev = ControllerEvaluator(
		spec,
		num_eval_episodes=args.eval_episodes,
		seed=holdout_seed,
		episode_config=ec,
		thresholds=thresholds,
		rg_config=rg,
		max_train_workers=args.train_workers,
		num_eval_folds=1,           # ← key: no K-fold rotation; pure held-out
	)

	t0 = time.time()
	if has_cells:
		metrics = ev.score_genomes([best_genome])[0]
	else:
		metrics = ev.evaluate_batch([best_genome])[0]
	dt = time.time() - t0

	# PID held-out baseline for direct comparison on the same fresh seed.
	pid_m = _pid_baseline(ec, args.eval_episodes, holdout_seed + 7777)

	print(f"\n{'='*72}\n  RESULTS\n{'='*72}")
	print(f"  HELD-OUT (fresh seed {holdout_seed}):")
	print(f"     err    = {metrics.mean_attitude_error_deg:.2f}°")
	print(f"     stable = {metrics.acc*100:.1f}%")
	print(f"     reward = {metrics.fitness:.2f}")
	print(f"     CE     = {metrics.ce:.2f}")
	if prev_metrics is not None:
		dErr = metrics.mean_attitude_error_deg - prev_metrics.mean_attitude_error_deg
		dSt  = (metrics.acc - prev_metrics.acc) * 100
		print(f"\n  Δ TRAINING → HELD-OUT (>0 means held-out is WORSE):")
		print(f"     Δ err    = {dErr:+.2f}°   ({'overfit' if dErr > 1 else 'generalizes' if abs(dErr) < 1 else 'underfit?'})")
		print(f"     Δ stable = {dSt:+.1f}pp  ({'overfit' if dSt < -3 else 'generalizes' if abs(dSt) < 3 else 'gained on held-out'})")
	if pid_m is not None:
		print(f"\n  vs PID baseline (same held-out distribution, seed offset):")
		print(f"     err    = {pid_m['mean_attitude_error_deg']:.2f}°  "
		      f"(WNN delta {metrics.mean_attitude_error_deg - pid_m['mean_attitude_error_deg']:+.2f}°)")
		print(f"     stable = {pid_m['stable_rate']*100:.1f}%       "
		      f"(WNN delta {(metrics.acc - pid_m['stable_rate'])*100:+.1f}pp)")
	print(f"\n  Wall time: {dt/60:.1f} min")

	# Honest reading per the gap interpretation:
	if prev_metrics is not None:
		gap = metrics.mean_attitude_error_deg - prev_metrics.mean_attitude_error_deg
		if abs(gap) < 0.5:
			print(f"\n  Verdict: GENERALIZES — held-out matches training within 0.5°.")
		elif gap < -1:
			print(f"\n  Verdict: SURPRISING WIN — held-out is BETTER than training.")
		elif gap < 1:
			print(f"\n  Verdict: SLIGHT GAP ({gap:+.2f}°) — mild overfit but acceptable.")
		else:
			print(f"\n  Verdict: REAL OVERFIT — held-out is {gap:.2f}° worse than training.")

	return 0


if __name__ == "__main__":
	sys.exit(main())
