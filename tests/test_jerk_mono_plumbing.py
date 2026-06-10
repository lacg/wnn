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


def _mini_evaluator(seed=7):
	import numpy as np
	from wnn.control.evaluator import ControllerEvaluator, arch_shape_from_spec
	from wnn.control.recurrent_genome import RecurrentArchGenome
	spec = ControllerSpec(num_motors=4, levels_per_motor=12, bits_per_feature=8,
	                      input_window_k=4, state_neurons=4, state_bits_per_neuron=20,
	                      output_bits_per_neuron=20)
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=4, seed=seed)
	ec = EpisodeConfig(dt=0.001, steps_per_episode=40, max_initial_tilt_rad=math.radians(5.0))
	rg = RewardGatedConfig(num_rounds=2, episodes_per_round=4, steps_per_episode=40,
	                       eval_episodes=4, seed=seed, episode_config=ec)
	ev = ControllerEvaluator(spec, num_eval_episodes=6, seed=seed, episode_config=ec,
	                         thresholds=thr, rg_config=rg, num_eval_folds=1)
	shape = arch_shape_from_spec(spec)
	genomes = [RecurrentArchGenome.random(shape, state_neurons=4, output_neurons=48,
	                                       state_suffix=12, output_suffix=12,
	                                       rng=np.random.default_rng(seed + i))
	           for i in range(3)]
	return ev, genomes


def test_arch_AND_memory_stage_metrics_carry_jerk_and_mono():
	"""The orthogonality fix: jerk + mono come from the Rust scorer (single
	source), so BOTH the arch path (evaluate_batch) AND the memory path
	(score_genomes) produce Metrics with them populated — not just one stage."""
	ev, genomes = _mini_evaluator()
	arch = ev.evaluate_batch(genomes)
	assert all(m.motor_jerk_mean is not None for m in arch), "arch stage dropped jerk"
	assert all(m.mono_violations_total is not None for m in arch), "arch stage dropped mono"
	# Memory path: need cells → train with write-back, then pure-score via score_genomes.
	ev.evaluate_for_adaptation(genomes, write_back=True)
	mem = ev.score_genomes(genomes)
	assert all(m.motor_jerk_mean is not None for m in mem), "MEMORY stage dropped jerk (the bug)"
	assert all(m.mono_violations_total is not None for m in mem), "MEMORY stage dropped mono (the bug)"


def test_calculator_actually_ranks_jerk_mono_no_warn_skip():
	"""With jerk/mono now populated, the harmonic calculator must RANK on them
	(not warn-and-skip), so a jerk/mono weight genuinely participates."""
	import warnings
	from wnn.ram.fitness.FitnessCalculatorControllerHarmonic import (
		FitnessCalculatorControllerHarmonic as Calc)
	ev, genomes = _mini_evaluator(seed=11)
	ms = ev.evaluate_batch(genomes)
	with warnings.catch_warnings():
		warnings.simplefilter("error")  # any "weight ignored / None" RuntimeWarning → failure
		f_erronly = Calc(weight_err_sq=1.0).fitness(ms)
		f_multi = Calc(weight_err_sq=0.4, weight_stable=0.3,
		               weight_jerk=0.2, weight_mono=0.1).fitness(ms)
	assert len(f_erronly) == len(f_multi) == len(ms)


if __name__ == "__main__":
	test_eval_closed_loop_surfaces_jerk(); print("  eval surfaces jerk OK")
	test_python_path_plumbs_jerk_and_mono(); print("  python path plumbs jerk+mono OK")
	test_arch_AND_memory_stage_metrics_carry_jerk_and_mono(); print("  arch AND memory Metrics carry jerk+mono OK")
	test_calculator_actually_ranks_jerk_mono_no_warn_skip(); print("  calculator ranks jerk+mono (no warn-skip) OK")
	print("ALL jerk/mono plumbing tests PASS")
