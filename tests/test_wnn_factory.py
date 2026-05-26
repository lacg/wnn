"""Tests for the WnnType strategy factory (Phase B step 5).

Verifies the registry-based unification: importing the controller module
self-registers WnnType.CONTROLLER, after which create_strategy builds the right
GA/TS strategy per dimension; capability validation and the not-yet-wired combos
(MEMORY / Lamarckian / unregistered IDS) raise clearly.

Run: PYTHONPATH=src/wnn python tests/test_wnn_factory.py
"""

from __future__ import annotations

import wnn.control.arch_strategy  # noqa: F401 — import side effect registers CONTROLLER
from wnn.control.arch_strategy import ControllerArchGAStrategy, ControllerArchTSStrategy
from wnn.control.evaluator import ControllerSpec
from wnn.ram.strategies.wnn_factory import (
	WnnType, StrategyKind, create_strategy, supports, supported_dimensions, is_registered,
)
from wnn.ram.strategies.optimization_dimension import OptimizationDimension as Dim


def _spec():
	bits = 2 * 3 + 16
	return ControllerSpec(num_motors=4, levels_per_motor=4, bits_per_feature=8,
	                      input_window_k=4, state_neurons=3,
	                      state_bits_per_neuron=bits, output_bits_per_neuron=bits)


def test_controller_registered():
	assert is_registered(WnnType.CONTROLLER), "importing arch_strategy must self-register CONTROLLER"
	print("✓ controller_registered")


def test_capability_map():
	# Controller is the only family that optimizes MEMORY (content / paradigm B).
	assert Dim.MEMORY in supported_dimensions(WnnType.CONTROLLER)
	assert Dim.MEMORY not in supported_dimensions(WnnType.IDS)
	assert Dim.MEMORY not in supported_dimensions(WnnType.LM)
	for d in (Dim.NEURONS, Dim.BITS, Dim.CONNECTIONS):
		assert supports(WnnType.CONTROLLER, StrategyKind.GA, d)
		assert supports(WnnType.CONTROLLER, StrategyKind.TS, d)
	assert not supports(WnnType.LM, StrategyKind.LAMARCKIAN, Dim.NEURONS)
	print("✓ capability_map")


def test_build_ga_and_ts_per_dimension():
	spec = _spec()
	for dim in (Dim.NEURONS, Dim.BITS, Dim.CONNECTIONS):
		ga = create_strategy(WnnType.CONTROLLER, StrategyKind.GA, dim, spec=spec, seed=0)
		assert isinstance(ga, ControllerArchGAStrategy) and ga._dimension == dim
		ga.create_random_genome().assert_valid()       # genome creation works
		ts = create_strategy(WnnType.CONTROLLER, StrategyKind.TS, dim, spec=spec, seed=0)
		assert isinstance(ts, ControllerArchTSStrategy) and ts._dimension == dim
		ts.seed_genome().assert_valid()
	print("✓ build_ga_and_ts_per_dimension")


def test_unsupported_and_pending_raise():
	spec = _spec()
	# Every declared CONTROLLER combo now BUILDS (axonogenesis only needs the Rust
	# rebuild at RUN time, guarded). Spot-check the paradigm-B + axonogenesis cells.
	from wnn.control.arch_strategy import ControllerMemoryGAStrategy, ControllerMemoryTSStrategy
	assert isinstance(create_strategy(WnnType.CONTROLLER, StrategyKind.GA, Dim.MEMORY, spec=spec, seed=0),
	                  ControllerMemoryGAStrategy)
	assert isinstance(create_strategy(WnnType.CONTROLLER, StrategyKind.TS, Dim.MEMORY, spec=spec, seed=0),
	                  ControllerMemoryTSStrategy)
	create_strategy(WnnType.CONTROLLER, StrategyKind.LAMARCKIAN, Dim.CONNECTIONS, spec=spec, seed=0)  # builds
	# Undeclared capability → ValueError (LM has no MEMORY).
	try:
		create_strategy(WnnType.LM, StrategyKind.GA, Dim.MEMORY, spec=spec)
		raise SystemExit("expected ValueError for LM/MEMORY")
	except ValueError:
		pass
	# Declared but no builder registered → NotImplementedError (extension point).
	try:
		create_strategy(WnnType.IDS, StrategyKind.GA, Dim.NEURONS)
		raise SystemExit("expected NotImplementedError for unregistered IDS")
	except NotImplementedError:
		pass
	print("✓ unsupported_and_pending_raise")


if __name__ == "__main__":
	test_controller_registered()
	test_capability_map()
	test_build_ga_and_ts_per_dimension()
	test_unsupported_and_pending_raise()
	print("\nAll WnnType-factory tests passed.")
