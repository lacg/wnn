"""Validation: does DAGGER fix the covariate-shift failure?

V2 (delta-control + small-state). The absolute-PWM / 200-neuron design lost to
an untrained controller (project_controller_state). This tests the redesign:
  - SMALL state: 16 neurons (4^16 reachable FSM states, not 4^200) → dense,
    learnable recurrent dynamics + a tiny dense state for the motor banks.
  - DELTA control: each motor WNN outputs a per-step PWM delta (accumulated),
    not an absolute throttle. Untrained → hold (stable bootstrap); the
    accumulator is the integrator; deltas are slew-bounded (±delta_max) so the
    student can't tumble — exactly what DAGGER needs to bootstrap.

All four policies are scored with the SAME per-episode-reset eval
(eval_closed_loop_reset) so the comparison is fair (recurrent state / pwm
accumulator / PID integral all reset each episode).

VERDICT gate: DAGGER should beat untrained AND behavioral cloning closed-loop.

Run:  python tests/test_dagger_closed_loop.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from ram_accelerator import AttitudeSim
from wnn.control.evaluator import (
	ControllerSpec, ControllerGenome, build_controller,
	fit_thresholds_from_pid_rollouts, random_connectivity,
)
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.dagger import DaggerConfig, train_dagger, eval_closed_loop_reset
from wnn.control.training import make_wnn_action_fn, make_pid_action_fn, EpisodeConfig


def main():
	# Small state + delta-control output.
	spec = ControllerSpec(
		num_motors=4, levels_per_motor=16, bits_per_feature=8,
		input_window_k=4, state_neurons=16, state_bits_per_neuron=20,
		output_bits_per_neuron=16,
		delta_control=True, delta_max=0.1,
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
	N_EVAL = 20
	EVAL_SEED = seed + 7_000_000   # match train_dagger's internal eval seed

	# --- PID teacher (per-episode reset) ---
	pid = AttitudePID(AttitudePIDConfig())
	pid_fit, pid_m = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, ec, N_EVAL, EVAL_SEED)
	print(f"\n[PID teacher]   mean_err={pid_m['mean_attitude_error_deg']:.2f}°  stable={pid_m['stable_rate']*100:.0f}%")

	# --- Untrained controller (delta-control → holds hover) ---
	untrained = ControllerGenome(spec=spec, thresholds=thresholds,
		state_connections=state_conn, output_connections=output_conn)
	un_ctrl = build_controller(untrained)
	un_fit, un_m = eval_closed_loop_reset(make_wnn_action_fn(un_ctrl), un_ctrl.reset, ec, N_EVAL, EVAL_SEED)
	print(f"[Untrained]     mean_err={un_m['mean_attitude_error_deg']:.2f}°  stable={un_m['stable_rate']*100:.0f}%")

	# --- Behavioral cloning (β=1 fixed → expert always drives) ---
	print(f"\nTraining BEHAVIORAL CLONING (β=1.0 fixed)...")
	bc_cfg = DaggerConfig(num_iterations=4, episodes_per_iter=10, steps_per_episode=1500,
		beta_decay=1.0, beta_floor=1.0, eval_episodes=N_EVAL, seed=seed, episode_config=ec)
	_, bc_stats = train_dagger(spec, thresholds, state_conn, output_conn, bc_cfg)

	# --- DAGGER (β decays → student drives) ---
	print(f"\nTraining DAGGER (β_decay=0.5)...")
	dg_cfg = DaggerConfig(num_iterations=4, episodes_per_iter=10, steps_per_episode=1500,
		beta_decay=0.5, beta_floor=0.0, eval_episodes=N_EVAL, seed=seed, episode_config=ec)
	_, dg_stats = train_dagger(spec, thresholds, state_conn, output_conn, dg_cfg)

	bc_err = bc_stats["iter_mean_err_deg"][bc_stats["best_iter"]]
	dg_err = dg_stats["iter_mean_err_deg"][dg_stats["best_iter"]]
	print("\n" + "=" * 64)
	print("CLOSED-LOOP MEAN ATTITUDE ERROR (lower = better)")
	print("=" * 64)
	print(f"  PID teacher       : {pid_m['mean_attitude_error_deg']:6.2f}°")
	print(f"  Untrained (hold)  : {un_m['mean_attitude_error_deg']:6.2f}°")
	print(f"  Behavioral cloning: {bc_err:6.2f}°  (best of {len(bc_stats['iter_mean_err_deg'])})")
	print(f"  DAGGER            : {dg_err:6.2f}°  (best of {len(dg_stats['iter_mean_err_deg'])})")
	print(f"\n  DAGGER err curve: {[f'{e:.2f}' for e in dg_stats['iter_mean_err_deg']]}")
	print(f"  BC     err curve: {[f'{e:.2f}' for e in bc_stats['iter_mean_err_deg']]}")
	print(f"  DAGGER stable curve: {[f'{s*100:.0f}%' for s in dg_stats['iter_stable_rate']]}")

	beats_untrained = dg_err < un_m["mean_attitude_error_deg"]
	beats_bc = dg_err <= bc_err
	print("\nVERDICT:")
	print(f"  DAGGER beats untrained?  {'YES' if beats_untrained else 'NO'}")
	print(f"  DAGGER <= BC?            {'YES' if beats_bc else 'NO'}")
	if beats_untrained and beats_bc:
		print("  ==> delta+small-state DAGGER fixes covariate shift. OK to wire into GA.")
		return 0
	print("  ==> still not clearly winning; investigate (open-loop error, Δmax, curriculum).")
	return 1


if __name__ == "__main__":
	sys.exit(main())
