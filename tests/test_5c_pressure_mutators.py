"""Phase 5c — GA mutators consume the splitting trainer's pressure.

The trainer emits (saturation, wish_bits); the eval path stamps it onto the
genome's `pressure`; the mutators bias offspring:
  - saturation > 0  -> _mutate_neurons grows state_neurons
  - wish_bits       -> _mutate_connections routes a state neuron to each wished bit

This is a unit test of the consumption (no full GA needed).
"""

import numpy as np
from wnn.control.recurrent_genome import (
	RecurrentArchGenome, RecurrentArchShape, RecurrentArchConfig,
)
from wnn.ram.strategies.optimization_dimension import OptimizationDimension


def _genome():
	shape = RecurrentArchShape(prefix_factor=1, state_input_space=288,
	                           output_input_space=72, output_quantum=4)
	return RecurrentArchGenome.random(shape, state_neurons=4, output_neurons=8,
	                                   state_suffix=20, output_suffix=20,
	                                   rng=np.random.default_rng(0))


def test_saturation_grows_state():
	cfg = RecurrentArchConfig(state_neuron_delta=2, min_state_neurons=2, max_state_neurons=16)
	rng = np.random.default_rng(1)
	# rate=0: without pressure NEURONS mutation does nothing; with saturation it grows.
	g = _genome()
	g.pressure = (0, ())
	child_nopress = g.mutate(OptimizationDimension.NEURONS, 0.0, cfg, rng)
	g.pressure = (50, ())
	grew = [g.mutate(OptimizationDimension.NEURONS, 0.0, cfg, np.random.default_rng(s)).state_neurons
	        for s in range(8)]
	print(f"  no-pressure state_neurons : {child_nopress.state_neurons} (== {g.state_neurons})")
	print(f"  saturated  state_neurons  : {grew}  (all > {g.state_neurons})")
	ok = (child_nopress.state_neurons == g.state_neurons) and all(n > g.state_neurons for n in grew)
	print("  -> saturation grows state  " + ("OK" if ok else "FAIL"))
	return ok


def test_wish_routes_connection():
	cfg = RecurrentArchConfig()
	g = _genome()
	wishes = (5, 17, 142, 200)
	g.pressure = (0, wishes)
	# rate=0: resampling does nothing; only the wish-injection changes connectivity.
	child = g.mutate(OptimizationDimension.CONNECTIONS, 0.0, cfg, np.random.default_rng(2))
	observed = set().union(*[set(s) for s in child.state_sampled])
	missing = [w for w in wishes if w not in observed]
	print(f"  wished bits {wishes} -> all observed after mutate? missing={missing}")
	# the parent should NOT already observe them all (so the test is meaningful)
	parent_obs = set().union(*[set(s) for s in g.state_sampled])
	pre = [w for w in wishes if w not in parent_obs]
	print(f"  (parent was missing {len(pre)}/{len(wishes)} of them before)")
	ok = (not missing) and (len(pre) > 0)
	print("  -> wishes routed into connectivity  " + ("OK" if ok else "FAIL"))
	return ok


def main():
	print("=" * 70)
	print("  PHASE 5c — pressure-driven mutation (saturation->grow, wish->route)")
	print("=" * 70)
	print("\n[neurons]"); a = test_saturation_grows_state()
	print("\n[connections]"); b = test_wish_routes_connection()
	ok = a and b
	print("\n" + "-" * 70)
	print("  PHASE 5c PASS — mutators consume trainer pressure." if ok else "  PHASE 5c FAIL")
	return ok


if __name__ == "__main__":
	raise SystemExit(0 if main() else 1)
