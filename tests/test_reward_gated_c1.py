"""Validation: does REWARD-GATED imitation (C1) beat the imitation paradigms
that tumbled on the SAME substrate?

Context (project_controller_state_2026_05_24): on absolute-PWM + full-state
recurrence, every imitation variant (per-step EDRA, DAGGER, full/truncated
BPTT) loses to the untrained hold-hover controller. C1's hypothesis: gating
the imitation signal on whole-episode reward (throw out tumbling rollouts)
turns the per-step loss into something filtered by closed-loop stability.

Substrate (LOCKED): absolute-PWM (delta_control=False), full-state structured
connectivity, 4 state neurons.

All policies scored with the SAME per-episode-reset eval (eval_closed_loop_reset)
so the comparison is fair.

VERDICT gate: reward-gated should beat untrained AND DAGGER closed-loop.

Run:  python tests/test_reward_gated_c1.py
"""

from __future__ import annotations

import math
import sys

from wnn.control.evaluator import (
	ControllerSpec, ControllerGenome, build_controller,
	fit_thresholds_from_pid_rollouts, random_connectivity,
)
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.dagger import DaggerConfig, train_dagger, eval_closed_loop_reset
from wnn.control.reward_gated import RewardGatedConfig, reward_gated_train
from wnn.control.training import make_wnn_action_fn, make_pid_action_fn, EpisodeConfig


def main():
	# LOCKED substrate: absolute-PWM, full-state connectivity, 4 state neurons.
	spec = ControllerSpec(
		num_motors=4, levels_per_motor=16, bits_per_feature=8,
		input_window_k=4, state_neurons=4, state_bits_per_neuron=24,
		output_bits_per_neuron=24,
		delta_control=False,          # absolute PWM (the clean substrate)
	)
	seed = 0
	STEPS = 1500
	N_EVAL = 20
	EVAL_SEED = seed + 5_000_000      # match reward_gated_train's internal eval seed

	print("Fitting thermometer thresholds from PID rollouts...")
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
	state_conn, output_conn = random_connectivity(spec, seed=seed)

	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=STEPS,
		max_initial_tilt_rad=math.radians(30.0),
		max_initial_yaw_rad=math.radians(30.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)

	# --- PID teacher ---
	pid = AttitudePID(AttitudePIDConfig())
	_, pid_m = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, ec, N_EVAL, EVAL_SEED)
	print(f"\n[PID teacher]   mean_err={pid_m['mean_attitude_error_deg']:.2f}°  stable={pid_m['stable_rate']*100:.0f}%")

	# --- Untrained (absolute → holds default PWM=0.75) ---
	untrained = ControllerGenome(spec=spec, thresholds=thresholds,
		state_connections=state_conn, output_connections=output_conn)
	un_ctrl = build_controller(untrained)
	_, un_m = eval_closed_loop_reset(make_wnn_action_fn(un_ctrl), un_ctrl.reset, ec, N_EVAL, EVAL_SEED)
	print(f"[Untrained]     mean_err={un_m['mean_attitude_error_deg']:.2f}°  stable={un_m['stable_rate']*100:.0f}%")

	# --- DAGGER baseline (the paradigm C1 is meant to beat) ---
	print("\nTraining DAGGER (β_decay=0.5)...")
	dg_cfg = DaggerConfig(num_iterations=5, episodes_per_iter=12, steps_per_episode=STEPS,
		beta_decay=0.5, beta_floor=0.0, eval_episodes=N_EVAL, seed=seed, episode_config=ec)
	_, dg_stats = train_dagger(spec, thresholds, state_conn, output_conn, dg_cfg)

	# --- Reward-gated imitation (C1) ---
	print("\nTraining REWARD-GATED imitation (C1)...")
	rg_cfg = RewardGatedConfig(
		num_rounds=5, episodes_per_round=24, steps_per_episode=STEPS,
		bptt_window=32, gate_quantile=0.5, gate_running=True,
		curriculum=True, easy_tilt_deg=8.0, full_tilt_deg=30.0,
		eval_episodes=N_EVAL, seed=seed, episode_config=ec,
	)
	_, rg_stats = reward_gated_train(spec, thresholds, state_conn, output_conn, rg_cfg)

	dg_err = dg_stats["iter_mean_err_deg"][dg_stats["best_iter"]]
	rg_err = rg_stats["iter_mean_err_deg"][rg_stats["best_iter"]]
	rg_stable = rg_stats["iter_stable_rate"][rg_stats["best_iter"]]

	print("\n" + "=" * 64)
	print("CLOSED-LOOP MEAN ATTITUDE ERROR (lower = better)")
	print("=" * 64)
	print(f"  PID teacher       : {pid_m['mean_attitude_error_deg']:6.2f}°  stable={pid_m['stable_rate']*100:3.0f}%")
	print(f"  Untrained (hold)  : {un_m['mean_attitude_error_deg']:6.2f}°  stable={un_m['stable_rate']*100:3.0f}%")
	print(f"  DAGGER            : {dg_err:6.2f}°  (best of {len(dg_stats['iter_mean_err_deg'])})")
	print(f"  Reward-gated (C1) : {rg_err:6.2f}°  stable={rg_stable*100:3.0f}%  (best of {len(rg_stats['iter_mean_err_deg'])})")
	print(f"\n  RG err curve   : {[f'{e:.2f}' for e in rg_stats['iter_mean_err_deg']]}")
	print(f"  RG stable curve: {[f'{s*100:.0f}%' for s in rg_stats['iter_stable_rate']]}")
	print(f"  RG trained/ep  : {rg_stats['iter_n_trained']}  (of {rg_cfg.episodes_per_round})")
	print(f"  RG tilt curve  : {[f'{t:.0f}' for t in rg_stats['iter_tilt_deg']]}°")

	beats_untrained = rg_err < un_m["mean_attitude_error_deg"]
	beats_dagger = rg_err < dg_err
	print("\nVERDICT:")
	print(f"  Reward-gated beats untrained?  {'YES' if beats_untrained else 'NO'}")
	print(f"  Reward-gated beats DAGGER?     {'YES' if beats_dagger else 'NO'}")
	if beats_untrained and beats_dagger:
		print("  ==> Gating fixes the imitation signal. OK to wrap with the connectivity GA.")
		return 0
	print("  ==> Gating not yet winning; investigate gate policy / curriculum / window.")
	return 1


if __name__ == "__main__":
	sys.exit(main())
