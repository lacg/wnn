"""Tests for controller Lamarckian / stats-guided genesis (Phase B step 4b-1..3).

Covers: the adaptive-eval stats + Lamarckian write-back (4b-1/2), the
ControllerAdaptationStrategy genesis loop (4b-3, synaptogenesis + neurogenesis),
and the WnnType-factory LAMARCKIAN wiring. Tiny rollouts keep it fast.

Run: PYTHONPATH=src/wnn python tests/test_controller_lamarckian.py
"""

from __future__ import annotations

import math

import numpy as np

import wnn.control.arch_strategy  # noqa: F401 — registers CONTROLLER in the factory
from wnn.control.evaluator import ControllerSpec, ControllerEvaluator, AdaptationStats, NUM_FEATURES
from wnn.control.arch_strategy import controller_ga_neurons, default_controller_arch_config
from wnn.control.arch_adaptation import ControllerAdaptationStrategy, AdaptationConfig
from wnn.control.recurrent_genome import RecurrentArchGenome
from wnn.control.training import EpisodeConfig
from wnn.control.reward_gated import RewardGatedConfig
from wnn.ram.strategies.wnn_factory import WnnType, StrategyKind, create_strategy
from wnn.ram.strategies.optimization_dimension import OptimizationDimension as Dim


def _spec():
	bits = 2 * 3 + 16
	return ControllerSpec(num_motors=4, levels_per_motor=4, bits_per_feature=8,
	                      input_window_k=4, state_neurons=3,
	                      state_bits_per_neuron=bits, output_bits_per_neuron=bits)


def _make_eval(spec):
	ec = EpisodeConfig(dt=0.001, steps_per_episode=40,
	                   max_initial_tilt_rad=math.radians(10.0), max_initial_yaw_rad=math.radians(10.0),
	                   max_initial_body_rate=0.3, max_initial_yaw_rate=0.2)
	rg = RewardGatedConfig(num_rounds=1, episodes_per_round=2, steps_per_episode=40,
	                       eval_episodes=2, curriculum=False, progress=False, episode_config=ec)
	thresholds = list(np.linspace(-1.0, 1.0, NUM_FEATURES * spec.bits_per_feature))
	return ControllerEvaluator(spec, num_eval_episodes=2, seed=0, episode_config=ec,
	                           thresholds=thresholds, rg_config=rg, fitness_seeds=1, max_train_workers=1)


def _genomes(spec, n=3):
	strat = controller_ga_neurons(spec, seed=0)
	return [strat.create_random_genome() for _ in range(n)]


def test_adaptive_eval_stats():
	spec = _spec()
	ev = _make_eval(spec)
	genomes = _genomes(spec)
	results = ev.evaluate_for_adaptation(genomes, write_back=False)
	assert len(results) == len(genomes)
	for g, (metrics, stats) in zip(genomes, results):
		assert isinstance(stats, AdaptationStats)
		assert len(stats.state_cell_counts) == g.state_neurons
		assert len(stats.output_cell_counts) == g.output_neurons
		assert isinstance(float(stats.reward), float)
		# cell counts are non-negative distinct-address tallies
		assert all(c >= 0 for c in stats.state_cell_counts)
	print("✓ adaptive_eval_stats")


def test_lamarckian_write_back():
	spec = _spec()
	ev = _make_eval(spec)
	g = _genomes(spec, 1)[0]
	assert g.cells is None
	ev.evaluate_for_adaptation([g], write_back=True)
	assert g.cells is not None, "write_back must stamp trained cells into genome.cells"
	g.assert_valid()  # cells aligned + in-range for the genome's shape
	# universe size == number of trained cells (state + output)
	assert len(g.cells.state_values) == len(g.cells.state_universe)
	print(f"✓ lamarckian_write_back (state cells={len(g.cells.state_universe)}, "
	      f"output cells={len(g.cells.output_universe)})")


def test_suffix_target_log2():
	# Deterministic synaptogenesis target: want_bits = ceil(log2(mean_fill)) + margin.
	spec = _spec()
	strat = ControllerAdaptationStrategy(spec, genesis_mode="synaptogenesis", seed=0)
	prefix = 2 * spec.state_neurons  # 6
	# mean_fill = 256 → log2=8, +margin(2)=10 want_bits → want_suffix=10-6=4; from cur=16 step≤2 → 14
	tgt = strat._suffix_target([256, 256, 256], prefix=prefix, cur=16, lo=1, hi=40)
	assert tgt == 14, f"expected one step down toward 4, got {tgt}"
	# mean_fill huge → grows toward cap, one step up
	tgt_up = strat._suffix_target([2**20], prefix=prefix, cur=8, lo=1, hi=40)
	assert tgt_up == 10, f"expected one step up, got {tgt_up}"
	print("✓ suffix_target_log2")


def _run_adapt(mode, spec, iters=3, pop=4):
	ev = _make_eval(spec)
	cfg = AdaptationConfig(genesis_mode=mode, iterations=iters, population_size=pop,
	                       patience=3, warmup_iterations=1, write_back=True)
	strat = ControllerAdaptationStrategy(spec, genesis_mode=mode, adapt_config=cfg,
	                                     seed=0, batch_evaluator=ev,
	                                     logger=lambda m: None)
	res = strat.optimize(initial_population=[strat.random_genome() for _ in range(pop)])
	return res


def test_synaptogenesis_loop_runs():
	spec = _spec()
	res = _run_adapt("synaptogenesis", spec)
	assert isinstance(res.best_genome, RecurrentArchGenome)
	res.best_genome.assert_valid()
	assert len(res.history) >= 1
	print(f"✓ synaptogenesis_loop_runs (best state_bits={res.best_genome.state_bits_per_neuron}, "
	      f"fitness={res.final_fitness:.3f})")


def test_neurogenesis_loop_runs():
	spec = _spec()
	res = _run_adapt("neurogenesis", spec)
	res.best_genome.assert_valid()
	cfg = default_controller_arch_config(spec)
	assert cfg.min_state_neurons <= res.best_genome.state_neurons <= cfg.max_state_neurons
	print(f"✓ neurogenesis_loop_runs (best state_neurons={res.best_genome.state_neurons})")


def test_factory_lamarckian_wiring():
	spec = _spec()
	neuro = create_strategy(WnnType.CONTROLLER, StrategyKind.LAMARCKIAN, Dim.NEURONS, spec=spec, seed=0)
	assert isinstance(neuro, ControllerAdaptationStrategy) and neuro._cfg.genesis_mode == "neurogenesis"
	synap = create_strategy(WnnType.CONTROLLER, StrategyKind.LAMARCKIAN, Dim.BITS, spec=spec, seed=0)
	assert synap._cfg.genesis_mode == "synaptogenesis"
	# axonogenesis (CONNECTIONS Lamarckian) is deferred to 4b-5
	try:
		create_strategy(WnnType.CONTROLLER, StrategyKind.LAMARCKIAN, Dim.CONNECTIONS, spec=spec)
		raise SystemExit("expected NotImplementedError for Lamarckian CONNECTIONS")
	except NotImplementedError:
		pass
	print("✓ factory_lamarckian_wiring")


def test_memory_dimension_paradigm_b():
	"""MEMORY dimension via the factory: paradigm B on the unified genome —
	fixed arch, evolve QSR cell values over a recorded universe, scored WITHOUT
	training (evaluator.score_genomes)."""
	from wnn.control.arch_strategy import ControllerMemoryGAStrategy
	from wnn.control.ga_strategy import default_controller_ga_config
	spec = _spec()
	ev = _make_eval(spec)
	strat = create_strategy(WnnType.CONTROLLER, StrategyKind.GA, Dim.MEMORY, spec=spec,
	                        seed=0, batch_evaluator=ev, record_episodes=2, record_steps=40,
	                        ga_config=default_controller_ga_config(population_size=4, generations=2))
	assert isinstance(strat, ControllerMemoryGAStrategy)
	# create_random_genome records the universe and seeds cells over it.
	pop = [strat.create_random_genome() for _ in range(4)]
	for g in pop:
		g.assert_valid()
		assert g.cells is not None, "MEMORY genome must carry cells"
		# all genomes share ONE arch + universe (paradigm B)
		assert g.state_neurons == pop[0].state_neurons
		assert g.cells.state_universe == pop[0].cells.state_universe
	# No-train scoring path + GA loop run end-to-end.
	res = strat.optimize(evaluate_fn=lambda gg: ev.score_genomes([gg])[0].ce,
	                     batch_evaluate_fn=ev.score_genomes, initial_population=pop)
	best = res.best_genome
	best.assert_valid()
	assert best.cells is not None
	# arch never changed (MEMORY only nudges values)
	assert best.state_neurons == spec.state_neurons
	print(f"✓ memory_dimension_paradigm_b (universe: {len(pop[0].cells.state_universe)} state, "
	      f"{len(pop[0].cells.output_universe)} output cells; fitness={res.final_fitness:.3f})")


if __name__ == "__main__":
	test_adaptive_eval_stats()
	test_lamarckian_write_back()
	test_suffix_target_log2()
	test_synaptogenesis_loop_runs()
	test_neurogenesis_loop_runs()
	test_factory_lamarckian_wiring()
	test_memory_dimension_paradigm_b()
	print("\nAll controller Lamarckian (4b-1..4) tests passed.")
