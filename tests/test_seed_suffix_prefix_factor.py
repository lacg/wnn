"""Regression: the arch-GA/TS/adaptation strategies must compute the seed
sampled-suffix as `bits - prefix_factor*state_neurons`, NOT the stale 2-bit-era
`bits - 2*state_neurons`. The literal 2· (left over from the 08/06/2026 1-bit
state migration) computed seed suffix ~0, which collapsed connectivity into
NON-UNIFORM suffix widths and produced genomes whose to_connections() length
no longer matched num_motors*levels*output_bits_per_neuron — crashing
reward_gated_train / the Rust dagger with a length-mismatch ValueError.

This guards the invariant for genomes the strategies emit, AND through several
generations of NEURONS-stage mutation + crossover (the path that first failed).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "wnn"))

import numpy as np

from wnn.control.evaluator import arch_shape_from_spec
from wnn.control.flow_adapter import controller_spec_from_params
from wnn.control.arch_strategy import (
	ControllerArchGAStrategy,
	default_controller_arch_config,
)
from wnn.control.recurrent_genome import RecurrentArchGenome
from wnn.ram.strategies.optimization_dimension import OptimizationDimension


def _spec(sn=12, sbits=24, obits=24, levels=16):
	return controller_spec_from_params({
		"controller_state_neurons": sn,
		"controller_state_bits": sbits,
		"controller_output_bits": obits,
		"controller_levels_per_motor": levels,
		"controller_num_motors": 4,
	})


def _assert_uniform_and_consistent(g: RecurrentArchGenome, tag: str):
	sc, oc = g.to_connections()
	exp_s = g.state_neurons * g.state_bits_per_neuron
	exp_o = g.output_neurons * g.output_bits_per_neuron
	sw = set(len(s) for s in g.state_sampled)
	ow = set(len(s) for s in g.output_sampled)
	assert len(sw) <= 1, f"[{tag}] non-uniform state suffix widths {sorted(sw)}"
	assert len(ow) <= 1, f"[{tag}] non-uniform output suffix widths {sorted(ow)}"
	assert len(sc) == exp_s, f"[{tag}] state conn {len(sc)} != {exp_s}"
	assert len(oc) == exp_o, f"[{tag}] output conn {len(oc)} != {exp_o}"


def test_seed_suffix_uses_prefix_factor_not_two():
	spec = _spec(sn=12, sbits=24, obits=24)
	pf = arch_shape_from_spec(spec).prefix_factor
	assert pf == 1, "1-bit migration should give prefix_factor=1"
	strat = ControllerArchGAStrategy(spec, OptimizationDimension.NEURONS, seed=7)
	# Seed suffix MUST be bits - pf*sn = 24 - 12 = 12, not 24 - 2*12 = 0.
	assert strat._seed_state_suffix == 24 - pf * 12 == 12
	assert strat._seed_output_suffix == 24 - pf * 12 == 12
	# default_controller_arch_config must also reflect the real suffix (max_suffix).
	cfg = default_controller_arch_config(spec)
	assert cfg.max_suffix >= 12, f"max_suffix {cfg.max_suffix} collapsed (stale 2·sn)"


def test_random_seed_genome_is_valid():
	spec = _spec(sn=12, sbits=24, obits=24)
	strat = ControllerArchGAStrategy(spec, OptimizationDimension.NEURONS, seed=7)
	for i in range(20):
		_assert_uniform_and_consistent(strat.create_random_genome(), f"random{i}")


def test_neurons_stage_offspring_stay_valid_through_generations():
	"""The exact failure path: seed → mutate(NEURONS) + crossover over many gens."""
	spec = _spec(sn=12, sbits=24, obits=24)
	cfg = default_controller_arch_config(spec)
	strat = ControllerArchGAStrategy(spec, OptimizationDimension.NEURONS, seed=7)
	rng = np.random.default_rng(7)
	pop = [strat.create_random_genome() for _ in range(8)]
	for g in pop:
		_assert_uniform_and_consistent(g, "init")
	for gen in range(40):
		nxt = []
		for _ in range(8):
			a, b = pop[int(rng.integers(0, len(pop)))], pop[int(rng.integers(0, len(pop)))]
			child = strat.crossover_genomes(a, b)
			_assert_uniform_and_consistent(child, f"xover_g{gen}")
			m = strat.mutate_genome(child, 0.5)
			_assert_uniform_and_consistent(m, f"mutate_g{gen}")
			nxt.append(m)
		pop = nxt


if __name__ == "__main__":
	test_seed_suffix_uses_prefix_factor_not_two()
	test_random_seed_genome_is_valid()
	test_neurons_stage_offspring_stay_valid_through_generations()
	print("OK — seed suffix uses prefix_factor; genomes uniform + connection-consistent")
