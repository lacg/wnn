"""Regression tests for the jerk/mono fitness-metric plumbing (01/06/2026).

Before this fix, weight_jerk/weight_mono were silently ignored on the default
(Python) training path: reward_gated_train's stats omitted iter_motor_jerk_mean
/ iter_mono_violations, so Metrics.motor_jerk_mean / mono_violations_total
arrived None and FitnessCalculatorControllerHarmonic dropped the weights. Two
parts of the fix are guarded here:
  1. The Rust DAGGER path is now the DEFAULT (opt-out), not opt-in.
  2. The Python fallback path now surfaces jerk (via eval_closed_loop_reset)
     and mono (via monotonicity_violations on the trained output cells).
"""

import math
import os

from wnn.control.evaluator import (
	ControllerSpec, fit_thresholds_from_pid_rollouts, random_connectivity,
	_rust_dagger_enabled,
)
from wnn.control.reward_gated import RewardGatedConfig, reward_gated_train
from wnn.control.training import EpisodeConfig, make_pid_action_fn
from wnn.control.dagger import eval_closed_loop_reset
from wnn.control.pid import AttitudePID, AttitudePIDConfig


def test_rust_dagger_is_default_opt_out(monkeypatch):
	monkeypatch.delenv("WNN_RUST_DAGGER", raising=False)
	assert _rust_dagger_enabled() is True            # default ON
	for off in ("0", "false", "off", "no", "OFF"):
		monkeypatch.setenv("WNN_RUST_DAGGER", off)
		assert _rust_dagger_enabled() is False
	for on in ("1", "true", "yes", ""):
		monkeypatch.setenv("WNN_RUST_DAGGER", on)
		assert _rust_dagger_enabled() is True


def test_eval_closed_loop_surfaces_jerk():
	spec = ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
	                      input_window_k=4, state_neurons=4, state_bits_per_neuron=16,
	                      output_bits_per_neuron=16)
	ec = EpisodeConfig(dt=0.001, steps_per_episode=40, max_initial_tilt_rad=math.radians(5.0))
	pid = AttitudePID(AttitudePIDConfig())
	_, m = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, ec, 4, 999)
	assert "mean_pwm_jerk" in m
	assert m["mean_pwm_jerk"] >= 0.0


def test_python_path_plumbs_jerk_and_mono():
	# Force the Python reference path and confirm it no longer drops the metrics.
	spec = ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
	                      input_window_k=4, state_neurons=4, state_bits_per_neuron=16,
	                      output_bits_per_neuron=16)
	seed = 7
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=5, seed=seed)
	sc, oc = random_connectivity(spec, seed=seed)
	ec = EpisodeConfig(dt=0.001, steps_per_episode=30, max_initial_tilt_rad=math.radians(5.0))
	rg = RewardGatedConfig(num_rounds=2, episodes_per_round=4, steps_per_episode=30,
	                       eval_episodes=4, seed=seed, episode_config=ec)
	rg.progress = False
	_, stats = reward_gated_train(spec, thresholds, sc, oc, rg)
	# Both keys present + one entry per round, none None, all non-negative.
	assert len(stats["iter_motor_jerk_mean"]) == len(stats["iter_fitness"]) >= 1
	assert len(stats["iter_mono_violations"]) == len(stats["iter_fitness"])
	assert all(j is not None and j >= 0.0 for j in stats["iter_motor_jerk_mean"])
	assert all(v is not None and v >= 0.0 for v in stats["iter_mono_violations"])
