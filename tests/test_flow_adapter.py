"""Tests for the controller flow→strategy adapter (worker-path wiring core).

Verifies a dashboard flow's params dict maps to a ControllerSpec + evaluator, and
each experiment phase_type resolves to the right strategy across the full matrix —
without touching worker.py/experiment.py.

Run: PYTHONPATH=src/wnn python tests/test_flow_adapter.py
"""

from __future__ import annotations

import wnn.control.arch_strategy  # noqa: F401 — registers CONTROLLER in the factory
from wnn.control.flow_adapter import (
	controller_spec_from_params, build_controller_evaluator, resolve_phase,
	build_controller_strategy, batch_eval_fn_for, PHASE_TO_KIND_DIM,
)
from wnn.control.evaluator import ControllerSpec, ControllerEvaluator
from wnn.control.arch_strategy import (
	ControllerArchGAStrategy, ControllerArchTSStrategy, ControllerMemoryGAStrategy,
	ControllerMemoryTSStrategy,
)
from wnn.control.arch_adaptation import ControllerAdaptationStrategy
from wnn.ram.strategies.optimization_dimension import OptimizationDimension as Dim
from wnn.ram.strategies.wnn_factory import StrategyKind


def test_spec_from_params():
	p = {"controller_state_neurons": 5, "controller_levels_per_motor": 32,
	     "controller_bits_per_feature": 8, "controller_state_bits": 24, "controller_output_bits": 24}
	spec = controller_spec_from_params(p)
	assert isinstance(spec, ControllerSpec)
	assert spec.state_neurons == 5 and spec.levels_per_motor == 32
	# Floor: bits must hold the forced full-state prefix + >=1 sampled bit. The
	# prefix is prefix_factor*state_neurons, and prefix_factor has been 1 since
	# the 08/06/2026 1-bit migration (7837d31a fixed the other stale 2* sites;
	# this assertion was missed by that sweep and asserted the dead contract).
	p2 = {"controller_state_neurons": 20, "controller_state_bits": 10}  # 10 < 20+1
	assert controller_spec_from_params(p2).state_bits_per_neuron >= 21
	# defaults
	d = controller_spec_from_params({})
	assert d.num_motors == 4 and d.state_neurons == 4
	print("✓ spec_from_params")


def test_resolve_phase_full_matrix():
	expect = {
		"ga_neurons": (StrategyKind.GA, Dim.NEURONS),
		"ga_bits": (StrategyKind.GA, Dim.BITS),
		"ga_connections": (StrategyKind.GA, Dim.CONNECTIONS),
		"ga_memory": (StrategyKind.GA, Dim.MEMORY),
		"ts_neurons": (StrategyKind.TS, Dim.NEURONS),
		"ts_memory": (StrategyKind.TS, Dim.MEMORY),
		"neurogenesis": (StrategyKind.LAMARCKIAN, Dim.NEURONS),
		"synaptogenesis": (StrategyKind.LAMARCKIAN, Dim.BITS),
		"axonogenesis": (StrategyKind.LAMARCKIAN, Dim.CONNECTIONS),
		"lamarckian_memory": (StrategyKind.LAMARCKIAN, Dim.MEMORY),
	}
	for phase, kd in expect.items():
		assert resolve_phase(phase) == kd, f"{phase} -> {resolve_phase(phase)} != {kd}"
	assert resolve_phase("GA_Neurons") == (StrategyKind.GA, Dim.NEURONS)  # case-insensitive
	try:
		resolve_phase("bogus")
		raise SystemExit("expected ValueError")
	except ValueError:
		pass
	print(f"✓ resolve_phase_full_matrix ({len(PHASE_TO_KIND_DIM)} phases)")


def test_build_evaluator_and_strategies():
	p = {"controller_state_neurons": 4, "controller_levels_per_motor": 16,
	     "controller_eval_episodes": 5, "controller_steps": 50, "seed": 0}
	ev = build_controller_evaluator(p)
	assert isinstance(ev, ControllerEvaluator) and ev.num_eval == 5

	cases = {
		"ga_neurons": ControllerArchGAStrategy,
		"ga_bits": ControllerArchGAStrategy,
		"ts_connections": ControllerArchTSStrategy,
		"ga_memory": ControllerMemoryGAStrategy,
		"ts_memory": ControllerMemoryTSStrategy,
		"neurogenesis": ControllerAdaptationStrategy,
		"synaptogenesis": ControllerAdaptationStrategy,
		"axonogenesis": ControllerAdaptationStrategy,
		"lamarckian_memory": ControllerAdaptationStrategy,
	}
	for phase, cls in cases.items():
		strat = build_controller_strategy(p, phase, ev)
		assert isinstance(strat, cls), f"{phase} -> {type(strat).__name__}, expected {cls.__name__}"
	print(f"✓ build_evaluator_and_strategies ({len(cases)} phases build)")


def test_batch_eval_fn_selection():
	p = {"controller_eval_episodes": 5, "controller_steps": 50}
	ev = build_controller_evaluator(p)
	# MEMORY (paradigm B) → no-train scorer; everything else → trains
	assert batch_eval_fn_for("ga_memory", ev) == ev.score_genomes
	assert batch_eval_fn_for("ts_memory", ev) == ev.score_genomes
	assert batch_eval_fn_for("ga_neurons", ev) == ev.evaluate_batch
	assert batch_eval_fn_for("ts_bits", ev) == ev.evaluate_batch
	print("✓ batch_eval_fn_selection")


def test_conn_policy_wired_through():
	"""controller_conn_policy reaches the GA's RecurrentArchConfig (it was registered
	in KNOWN_PARAMS but never read, 08/11/2026); framed1 and typos are refused."""
	from wnn.control.arch_strategy import default_controller_arch_config
	for name, want in (("spread", ("spread", 2)), ("min1", ("min_per_cluster", 1)),
	                   ("min3", ("min_per_cluster", 3))):
		spec = controller_spec_from_params({"controller_conn_policy": name})
		cfg = default_controller_arch_config(spec)
		assert (cfg.conn_policy, cfg.conn_policy_min) == want, (name, cfg.conn_policy, cfg.conn_policy_min)
	p = {"controller_conn_policy": "min2", "controller_eval_episodes": 2, "controller_steps": 20}
	strat = build_controller_strategy(p, "ga_neurons", build_controller_evaluator(p))
	assert (strat._arch_config.conn_policy, strat._arch_config.conn_policy_min) == ("min_per_cluster", 2)
	for bad in ("framed1", "min9"):
		try:
			controller_spec_from_params({"controller_conn_policy": bad})
		except ValueError:
			continue
		raise AssertionError(f"controller_conn_policy={bad} must be refused")
	print("✓ conn_policy wired (spread/min1-3), framed1 + typos refused")


def test_unbuilt_controller_keys_refused():
	"""Every registered controller_* key is either built by the adapter or refused —
	none is silently dropped."""
	from wnn.control.flow_adapter import ADAPTER_KEYS
	from wnn.ram.experiments.params import KNOWN_PARAMS
	ctl = sorted(k for k in KNOWN_PARAMS if k.startswith("controller_"))
	assert ADAPTER_KEYS <= set(KNOWN_PARAMS), f"unregistered: {sorted(ADAPTER_KEYS - set(KNOWN_PARAMS))}"
	unbuilt = [k for k in ctl if k not in ADAPTER_KEYS]
	assert unbuilt, "expected phased_ga-only keys in the registry"
	for k in unbuilt:
		try:
			controller_spec_from_params({k: 1})
		except ValueError:
			continue
		raise AssertionError(f"{k} silently ignored by the flow adapter")
	for unset in (None, False):   # an explicitly-off flag is the default, not a request
		assert controller_spec_from_params({"controller_translation": unset}).num_motors == 4
	print(f"✓ {len(unbuilt)} phased_ga-only controller_* keys refused, {len(ctl) - len(unbuilt)} built")


if __name__ == "__main__":
	test_conn_policy_wired_through()
	test_unbuilt_controller_keys_refused()
	test_spec_from_params()
	test_resolve_phase_full_matrix()
	test_build_evaluator_and_strategies()
	test_batch_eval_fn_selection()
	print("\nAll flow-adapter tests passed.")
