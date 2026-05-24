"""Validation: does DAGGER fix the covariate-shift failure?

The diagnostics verdict (project_controller_state_2026_05_24) was: the WNN
learns PID open-loop (~0.02 PWM error) but loses to random closed-loop
(EDRA per-motor 39°, reservoir 23° vs random 17°) — classic behavioral-
cloning covariate shift. DAGGER trains on the STUDENT's own state
distribution, which should drop closed-loop error toward PID's.

This script trains ONE controller (fixed random connectivity + PID-fit
thresholds) three ways and compares closed-loop mean attitude error:
  - PID teacher            (the target to approach: ~4.17°)
  - Untrained (random)     (the floor: empty QSR cells → default PWM)
  - Behavioral cloning     (expert always drives — beta=1 every round)
  - DAGGER                 (student drives, beta decays)

VERDICT: DAGGER's closed-loop error should be < BC's and < untrained.
If not, DAGGER is NOT wired into the GA (Behavioral Rule 1 — validate first).

Run:  python tests/test_dagger_closed_loop.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from wnn.control.evaluator import (
	ControllerSpec, ControllerEvaluator,
	fit_thresholds_from_pid_rollouts, random_connectivity,
)
from wnn.control.dagger import DaggerConfig, train_dagger
from wnn.control.training import make_wnn_action_fn, fitness_function, EpisodeConfig
from ram_accelerator import AttitudeSim


def main():
	# Small spec so the per-step beam-search EDRA + eval run in a couple
	# minutes. state_neurons modest; 256 levels stays the paper default.
	spec = ControllerSpec(
		num_motors=4, levels_per_motor=256, bits_per_feature=8,
		input_window_k=4, state_neurons=64, state_bits_per_neuron=16,
		output_bits_per_neuron=16,
	)
	seed = 0
	print(f"Fitting thermometer thresholds from PID rollouts...")
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
	state_conn, output_conn = random_connectivity(spec, seed=seed)

	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=1500,
		max_initial_tilt_rad=math.radians(30.0),
		max_initial_yaw_rad=math.radians(30.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)
	eval_kwargs = dict(num_eval_episodes=20, num_validate_episodes=20, seed=seed, episode_config=ec)
	evaluator = ControllerEvaluator(spec, **eval_kwargs)

	# --- PID baseline (the target) ---
	pid_fit, pid_m = evaluator.evaluate_pid_baseline()
	print(f"\n[PID teacher]   fitness={pid_fit:.2f}  mean_err={pid_m['mean_attitude_error_deg']:.2f}°  stable={pid_m['stable_rate']*100:.0f}%")

	# --- Untrained controller (random connectivity, empty cells) ---
	from wnn.control.evaluator import ControllerGenome
	untrained = ControllerGenome(spec=spec, thresholds=thresholds,
		state_connections=state_conn, output_connections=output_conn)
	un_fit, un_m = evaluator.evaluate(untrained)
	print(f"[Untrained]     fitness={un_fit:.2f}  mean_err={un_m['mean_attitude_error_deg']:.2f}°  stable={un_m['stable_rate']*100:.0f}%")

	# --- Behavioral cloning (beta=1 every round → expert always drives) ---
	print(f"\nTraining BEHAVIORAL CLONING (beta=1.0 fixed)...")
	bc_cfg = DaggerConfig(num_iterations=3, episodes_per_iter=10, steps_per_episode=1500,
		beta_decay=1.0, beta_floor=1.0, eval_episodes=20, seed=seed, episode_config=ec)
	_, bc_stats = train_dagger(spec, thresholds, state_conn, output_conn, bc_cfg)

	# --- DAGGER (beta decays → student drives) ---
	print(f"\nTraining DAGGER (beta_decay=0.5)...")
	dg_cfg = DaggerConfig(num_iterations=3, episodes_per_iter=10, steps_per_episode=1500,
		beta_decay=0.5, beta_floor=0.0, eval_episodes=20, seed=seed, episode_config=ec)
	_, dg_stats = train_dagger(spec, thresholds, state_conn, output_conn, dg_cfg)

	# --- Verdict ---
	bc_err = bc_stats["iter_mean_err_deg"][bc_stats["best_iter"]]
	dg_err = dg_stats["iter_mean_err_deg"][dg_stats["best_iter"]]
	print("\n" + "=" * 64)
	print("CLOSED-LOOP MEAN ATTITUDE ERROR (lower = better)")
	print("=" * 64)
	print(f"  PID teacher       : {pid_m['mean_attitude_error_deg']:6.2f}°")
	print(f"  Untrained (random): {un_m['mean_attitude_error_deg']:6.2f}°")
	print(f"  Behavioral cloning: {bc_err:6.2f}°  (best of {len(bc_stats['iter_mean_err_deg'])} rounds)")
	print(f"  DAGGER            : {dg_err:6.2f}°  (best of {len(dg_stats['iter_mean_err_deg'])} rounds)")
	print(f"\n  DAGGER per-round err curve: {[f'{e:.2f}' for e in dg_stats['iter_mean_err_deg']]}")
	print(f"  BC     per-round err curve: {[f'{e:.2f}' for e in bc_stats['iter_mean_err_deg']]}")

	beats_untrained = dg_err < un_m["mean_attitude_error_deg"]
	beats_bc = dg_err <= bc_err
	print("\nVERDICT:")
	print(f"  DAGGER beats untrained?  {'YES' if beats_untrained else 'NO'}")
	print(f"  DAGGER <= BC?            {'YES' if beats_bc else 'NO'}")
	if beats_untrained and beats_bc:
		print("  ==> DAGGER fixes covariate shift. OK to wire into GA.")
		return 0
	print("  ==> DAGGER did NOT clearly help. Do NOT wire into GA yet; investigate.")
	return 1


if __name__ == "__main__":
	sys.exit(main())
