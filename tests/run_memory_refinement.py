"""Plan B: memory-only refinement of a saved phased-GA winner.

Loads a winner pickled by `run_phased_ga.py --save-winner X` and runs ONLY
the memory phase (Stage 4) on it under a new fitness weight schema. Use to
push the saved controller toward 100% stability AFTER Plan A has bounded
the err axis: arch is frozen; only QSR cell values evolve.

Standard recipe:
  Plan A:  --fit-weight-err-sq 0.30 --fit-weight-stable 0.50
           --fit-weight-jerk 0.10 --fit-weight-mono 0.10
           (balanced weights — finds a low-err arch)
  Plan B:  --fit-weight-err-sq 0.40 --fit-weight-stable 0.60
           --fit-weight-jerk 0    --fit-weight-mono 0
           (stability-dominant, err as a floor — squeezes stable_rate
            toward 100% without disturbing the arch)

Plan B is cheap: the memory stage in the 14-combo sweep ran in 3-48s per
combo because arch is frozen and only cells change. Even a big budget
(--memory-gens 1000 --memory-patience 40) will land in minutes, not hours.

Example:
  python tests/run_memory_refinement.py \\
    --load-winner logs/controller/planAB/winner_planA.pkl \\
    --fit-weight-err-sq 0.4 --fit-weight-stable 0.6 \\
    --memory-gens 500 --memory-patience 25 \\
    --pop 200 --eval-episodes 20 --steps 1500 \\
    --universe-episodes 8 --tilt 15 --base-seed 20260530 \\
    --save-winner logs/controller/planAB/winner_planB.pkl
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

# Import the heavy lifters from run_phased_ga.py — same dir, same env.
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
	sys.path.insert(0, str(_THIS_DIR))
from run_phased_ga import (
	_run_memory_phase, _save_winner, _stage_header,
	_print_stage_result, _pid_baseline,
)
from wnn.control.training import EpisodeConfig
from wnn.seeds import resolve_seed_set, log_seed_set, record_seed_set


def main():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--load-winner", type=str, required=True,
	                help="Pickle written by run_phased_ga.py --save-winner.")
	# Memory-phase GA budget (mirrors run_phased_ga.py defaults but heftier
	# because this is the focused refinement run).
	ap.add_argument("--memory-gens", type=int, default=500)
	ap.add_argument("--memory-patience", type=int, default=25)
	ap.add_argument("--pop", type=int, default=200)
	ap.add_argument("--elitism", type=float, default=0.2)
	ap.add_argument("--crossover-rate", type=float, default=0.5)
	# Evaluation: SHOULD match Plan A's eval-episodes (20) for an apples-to-apples
	# comparison. Smaller (e.g. 3) hides the granularity Plan B is meant to exploit.
	ap.add_argument("--eval-episodes", type=int, default=20)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	ap.add_argument("--universe-episodes", type=int, default=8)
	# Reward-gated training is irrelevant for memory-only (paradigm B = no inner
	# training), but the evaluator path still touches these knobs — keep them
	# matched to Plan A.
	ap.add_argument("--rg-rounds", type=int, default=None)
	ap.add_argument("--rg-episodes-per-round", type=int, default=None)
	ap.add_argument("--rg-eval-episodes", type=int, default=None)
	# Plan B's NEW fitness weights — stability-dominant by default.
	ap.add_argument("--fit-weight-err-sq", type=float, default=0.4)
	ap.add_argument("--fit-weight-stable", type=float, default=0.6)
	ap.add_argument("--fit-weight-jerk",   type=float, default=0.0)
	ap.add_argument("--fit-weight-mono",   type=float, default=0.0)
	ap.add_argument("--train-workers", type=int, default=4)
	# Seed plumbing — distinct from Plan A's seed by default so test/val
	# episodes aren't memorized. Override with --base-seed for reproducibility.
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--base-seed", type=int, default=None)
	ap.add_argument("--train-seed", type=int, default=None)
	ap.add_argument("--test-seed", type=int, default=None)
	ap.add_argument("--val-seed", type=int, default=None)
	# Optional: chain forward — save Plan B's refined winner too.
	ap.add_argument("--save-winner", type=str, default=None,
	                help="Pickle the post-refinement genome for further chaining.")
	# Grid args are unused but `_run_memory_phase` reads `args.grid_state_neurons`
	# via `arch_cfg.max_state_neurons = max(..., 4 * max(args.grid_state_neurons))`.
	# Reuse loaded spec's state_neurons as the only grid point so the bound is
	# consistent with the saved arch.
	args = ap.parse_args()

	# Load the saved Plan A winner.
	payload_path = Path(args.load_winner)
	if not payload_path.exists():
		print(f"ERROR: --load-winner path does not exist: {payload_path}", file=sys.stderr)
		return 1
	with open(payload_path, "rb") as f:
		payload = pickle.load(f)
	spec_loaded   = payload["spec"]
	genome_loaded = payload["genome"]
	prev_metrics  = payload.get("metrics")
	prev_weights  = payload.get("fitness_weights", {})
	prev_meta     = payload.get("meta", {})

	print(f"\n{'='*72}\n  PLAN B: MEMORY-ONLY REFINEMENT\n{'='*72}")
	print(f"  loaded:  {payload_path}")
	print(f"  spec:    sn={spec_loaded.state_neurons} "
	      f"sb={spec_loaded.state_bits_per_neuron} "
	      f"ob={spec_loaded.output_bits_per_neuron} "
	      f"levels={spec_loaded.levels_per_motor}")
	if prev_metrics is not None:
		print(f"  Plan A:  err={prev_metrics.mean_attitude_error_deg:.2f}°  "
		      f"stable={prev_metrics.acc*100:.1f}%  reward={prev_metrics.fitness:.2f}")
	print(f"  Plan A weights: {prev_weights}")
	print(f"  Plan B weights: err_sq={args.fit_weight_err_sq} "
	      f"stable={args.fit_weight_stable} jerk={args.fit_weight_jerk} "
	      f"mono={args.fit_weight_mono}")

	# Pin grid_state_neurons to the loaded sn so _run_memory_phase's max-bound
	# arithmetic matches the saved arch (it caps max_state_neurons at
	# max(default, 4 * max(grid_state_neurons))).
	args.grid_state_neurons = [spec_loaded.state_neurons]
	args.levels = prev_meta.get("levels", spec_loaded.levels_per_motor)

	# Episode config — reuse Plan A's tilt/steps if not overridden via flags.
	# argparse defaults are already applied; the user can override.
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)

	# Seed plumbing — same protocol as run_phased_ga.py.
	base = args.base_seed if args.base_seed is not None else args.seed
	s = resolve_seed_set(base=base, run_index=0,
	                     train=args.train_seed, test=args.test_seed, val=args.val_seed)
	log_seed_set(s)
	record_seed_set(s, script="run_memory_refinement", extra={
		"loaded_from": str(payload_path), "memory_gens": args.memory_gens,
		"pop": args.pop, "eval_episodes": args.eval_episodes,
		"fit_weights": {
			"err_sq": args.fit_weight_err_sq, "stable": args.fit_weight_stable,
			"jerk": args.fit_weight_jerk, "mono": args.fit_weight_mono,
		},
	})

	# Run memory refinement with loaded genome as warm-start.
	t_start = time.time()
	_stage_header(4, "MEMORY (PLAN B)", args.memory_gens, args.memory_patience, spec_loaded)
	res, ev, dt = _run_memory_phase(args, ec, spec_loaded,
	                                args.memory_gens, args.memory_patience, s.train,
	                                warm_start_genome=genome_loaded)
	m = _print_stage_result(4, "MEMORY (PLAN B)", res, args.memory_gens, dt, ev)

	# PID baseline for vs-PID line.
	pid_m = _pid_baseline(ec, args.eval_episodes, s.val)

	# Compact final summary.
	print(f"\n{'='*72}\n  PLAN B FINAL\n{'='*72}")
	if m is not None:
		print(f"  Plan B:  err={m.mean_attitude_error_deg:.2f}°  "
		      f"stable={m.acc*100:.1f}%  reward={m.fitness:.2f}")
	if prev_metrics is not None:
		dErr = m.mean_attitude_error_deg - prev_metrics.mean_attitude_error_deg
		dSt  = (m.acc - prev_metrics.acc) * 100
		print(f"  Δ Plan A → Plan B:  err={dErr:+.2f}°  stable={dSt:+.1f}pp")
	if pid_m is not None:
		print(f"  vs PID:  err={pid_m['mean_attitude_error_deg']:.2f}°  "
		      f"stable={pid_m['stable_rate']*100:.1f}%  "
		      f"reward={pid_m['mean_reward']:.2f}")
	print(f"  Total wall time: {(time.time() - t_start)/60:.1f} min")

	if args.save_winner is not None and res.best_genome is not None:
		_save_winner(args.save_winner, args, spec_loaded, res.best_genome, m)

	return 0


if __name__ == "__main__":
	sys.exit(main())
