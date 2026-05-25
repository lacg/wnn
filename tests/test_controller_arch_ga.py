"""End-to-end smoke test for the phase-aware controller architecture GA.

Drives ControllerArchGAStrategy (GA-Neurons / GA-Bits / GA-Connectivity) for a
couple of tiny generations through the real GenericGAStrategy loop + the now
variable-shape-aware ControllerEvaluator, asserting:
  - the GA runs end-to-end and returns a valid RecurrentArchGenome,
  - phase isolation (GA-Bits / GA-Connectivity never change neuron counts),
  - shape-grouped GPU scoring survives a population of mixed shapes (GA-Neurons),
  - the winning genome materializes into a controller that takes a step.

Tiny RewardGatedConfig + dummy thresholds keep it fast (it tests plumbing, not
training quality). Run: PYTHONPATH=src/wnn python tests/test_controller_arch_ga.py
"""

from __future__ import annotations

import math

import numpy as np

from wnn.control.evaluator import (
	ControllerSpec, ControllerEvaluator, controller_genome_from_arch, build_controller, NUM_FEATURES,
)
from wnn.control.arch_strategy import (
	controller_ga_neurons, controller_ga_bits, controller_ga_connections,
	default_controller_arch_config,
)
from wnn.control.ga_strategy import default_controller_ga_config
from wnn.control.recurrent_genome import RecurrentArchGenome
from wnn.control.training import EpisodeConfig
from wnn.control.reward_gated import RewardGatedConfig


def _make_eval(spec):
	ec = EpisodeConfig(dt=0.001, steps_per_episode=40,
	                   max_initial_tilt_rad=math.radians(10.0),
	                   max_initial_yaw_rad=math.radians(10.0),
	                   max_initial_body_rate=0.3, max_initial_yaw_rate=0.2)
	rg = RewardGatedConfig(num_rounds=1, episodes_per_round=2, steps_per_episode=40,
	                       eval_episodes=2, curriculum=False, progress=False, episode_config=ec)
	thresholds = list(np.linspace(-1.0, 1.0, NUM_FEATURES * spec.bits_per_feature))
	return ControllerEvaluator(spec, num_eval_episodes=2, seed=0, episode_config=ec,
	                           thresholds=thresholds, rg_config=rg, fitness_seeds=1,
	                           max_train_workers=1)


def _run(strategy_factory, spec, pop=4, gens=2):
	ev = _make_eval(spec)
	strat = strategy_factory(spec, ga_config=default_controller_ga_config(
		population_size=pop, generations=gens), seed=0, batch_evaluator=ev)
	init_pop = [strat.create_random_genome() for _ in range(pop)]
	for g in init_pop:
		g.assert_valid()
	res = strat.optimize(evaluate_fn=ev.evaluate_single,
	                     batch_evaluate_fn=ev.evaluate_batch, initial_population=init_pop)
	return strat, res


# A small, fast substrate: 3 state neurons × 24b, 4 levels (16 output neurons).
def _spec():
	bits = 2 * 3 + 16  # 2*state_neurons + 16 sampled
	return ControllerSpec(num_motors=4, levels_per_motor=4, bits_per_feature=8,
	                      input_window_k=4, state_neurons=3,
	                      state_bits_per_neuron=bits, output_bits_per_neuron=bits)


def test_ga_connections_runs_and_isolates():
	spec = _spec()
	strat, res = _run(controller_ga_connections, spec)
	best = res.best_genome
	assert isinstance(best, RecurrentArchGenome)
	best.assert_valid()
	# CONNECTIONS phase never changes the shape.
	assert best.state_neurons == spec.state_neurons
	assert best.output_neurons == spec.num_motors * spec.levels_per_motor
	assert best.state_suffix_width == spec.state_bits_per_neuron - 2 * spec.state_neurons
	assert isinstance(float(res.final_fitness), float)
	print(f"✓ ga_connections_runs_and_isolates (fitness={res.final_fitness:.3f})")


def test_ga_bits_runs_and_isolates():
	spec = _spec()
	strat, res = _run(controller_ga_bits, spec)
	best = res.best_genome
	best.assert_valid()
	# BITS phase never changes neuron counts.
	assert best.state_neurons == spec.state_neurons
	assert best.output_neurons == spec.num_motors * spec.levels_per_motor
	print(f"✓ ga_bits_runs_and_isolates (best state_bits={best.state_bits_per_neuron})")


def test_ga_neurons_runs_with_mixed_shapes():
	spec = _spec()
	# A NEURONS population spans shapes → exercises shape-grouped GPU scoring.
	cfg = default_controller_arch_config(spec)
	strat, res = _run(controller_ga_neurons, spec)
	best = res.best_genome
	best.assert_valid()
	assert cfg.min_state_neurons <= best.state_neurons <= cfg.max_state_neurons
	assert best.output_neurons % spec.num_motors == 0
	# The winner must materialize into a working controller.
	thresholds = list(np.linspace(-1.0, 1.0, NUM_FEATURES * spec.bits_per_feature))
	c = build_controller(controller_genome_from_arch(best, spec, thresholds))
	c.reset()
	pwm = c.step([0.01, -0.02, 0.0], [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
	assert len(pwm) == spec.num_motors
	print(f"✓ ga_neurons_runs_with_mixed_shapes (best state_neurons={best.state_neurons}, "
	      f"levels={best.output_neurons // spec.num_motors})")


if __name__ == "__main__":
	test_ga_connections_runs_and_isolates()
	test_ga_bits_runs_and_isolates()
	test_ga_neurons_runs_with_mixed_shapes()
	print("\nAll controller architecture-GA smoke tests passed.")
