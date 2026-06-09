"""flow_adapter — bridge a dashboard flow into a controller strategy.

This is the worker-side wiring for `architecture_type='controller'` flows: it maps
a flow's flat `params` dict → a `ControllerSpec` + `ControllerEvaluator`, and each
experiment's `phase_type` → `(WnnType, StrategyKind, OptimizationDimension)` →
`wnn_factory.create_strategy`. Isolated here (and unit-tested) so the actual
worker.py / experiment.py hookup is a 2-call insertion against verified code.

HOOKUP (the two live-path insertions, done in an attended session):
  worker.py  (evaluator dispatch, ~line 506):
      elif architecture_type == "controller":
          evaluator = build_controller_evaluator(params)
  experiment.py (before OptimizerStrategyFactory.create):
      if self.architecture_type == "controller":
          strategy = build_controller_strategy(params, cfg, self.evaluator)
      else:
          strategy = OptimizerStrategyFactory.create(**strategy_kwargs)

Param keys (all optional, sensible defaults; namespaced `controller_*`):
  controller_num_motors(4) controller_levels_per_motor(16) controller_bits_per_feature(8)
  controller_input_window_k(4) controller_state_neurons(4) controller_state_bits(24)
  controller_output_bits(24) controller_delta_control(False)
  controller_eval_episodes(20) controller_steps(1500) controller_tilt_deg(15.0) seed(0)
"""

from __future__ import annotations

import math
from typing import Any, Optional

from wnn.ram.strategies.wnn_factory import WnnType, StrategyKind, create_strategy
from wnn.ram.strategies.optimization_dimension import OptimizationDimension as Dim

# Import side effect: registers WnnType.CONTROLLER in the factory. Importing the
# adapter (the controller-flow bridge) therefore guarantees the builder exists.
import wnn.control.arch_strategy  # noqa: F401

from .evaluator import ControllerSpec, ControllerEvaluator
from .training import EpisodeConfig


# Experiment phase_type → (strategy kind, optimization dimension). Covers the full
# {GA,TS,Lamarckian} × {neurons,bits,connections,memory} matrix. Aliases included
# so dashboard naming variants resolve. Lamarckian genesis modes map to their
# dimension (neurogenesis→NEURONS, synaptogenesis→BITS, axonogenesis→CONNECTIONS).
PHASE_TO_KIND_DIM: dict[str, tuple] = {
	# GA
	"ga_neurons": (StrategyKind.GA, Dim.NEURONS),
	"ga_bits": (StrategyKind.GA, Dim.BITS),
	"ga_connections": (StrategyKind.GA, Dim.CONNECTIONS),
	"ga_connectivity": (StrategyKind.GA, Dim.CONNECTIONS),
	"ga_memory": (StrategyKind.GA, Dim.MEMORY),
	# TS
	"ts_neurons": (StrategyKind.TS, Dim.NEURONS),
	"ts_bits": (StrategyKind.TS, Dim.BITS),
	"ts_connections": (StrategyKind.TS, Dim.CONNECTIONS),
	"ts_connectivity": (StrategyKind.TS, Dim.CONNECTIONS),
	"ts_memory": (StrategyKind.TS, Dim.MEMORY),
	# Lamarckian (genesis-mode named OR dimension-named)
	"neurogenesis": (StrategyKind.LAMARCKIAN, Dim.NEURONS),
	"synaptogenesis": (StrategyKind.LAMARCKIAN, Dim.BITS),
	"axonogenesis": (StrategyKind.LAMARCKIAN, Dim.CONNECTIONS),
	"lamarckian_neurons": (StrategyKind.LAMARCKIAN, Dim.NEURONS),
	"lamarckian_bits": (StrategyKind.LAMARCKIAN, Dim.BITS),
	"lamarckian_connections": (StrategyKind.LAMARCKIAN, Dim.CONNECTIONS),
	"lamarckian_memory": (StrategyKind.LAMARCKIAN, Dim.MEMORY),
}


def _p(params: dict, key: str, default: Any) -> Any:
	v = params.get(key)
	return default if v is None else v


def controller_spec_from_params(params: dict) -> ControllerSpec:
	"""Build a ControllerSpec from a flow's flat params dict."""
	sn = int(_p(params, "controller_state_neurons", 4))
	# Floor: bits must hold the forced full-state prefix + ≥1 sampled. The prefix is
	# prefix_factor·state_neurons = 1·sn since the 08/06/2026 1-bit migration (was 2·sn).
	sbits = max(int(_p(params, "controller_state_bits", 24)), sn + 1)
	obits = max(int(_p(params, "controller_output_bits", 24)), sn + 1)
	return ControllerSpec(
		num_motors=int(_p(params, "controller_num_motors", 4)),
		levels_per_motor=int(_p(params, "controller_levels_per_motor", 16)),
		bits_per_feature=int(_p(params, "controller_bits_per_feature", 8)),
		input_window_k=int(_p(params, "controller_input_window_k", 4)),
		state_neurons=sn,
		state_bits_per_neuron=sbits,
		output_bits_per_neuron=obits,
		delta_control=bool(_p(params, "controller_delta_control", False)),
	)


def _episode_config(params: dict) -> EpisodeConfig:
	tilt = math.radians(float(_p(params, "controller_tilt_deg", 15.0)))
	return EpisodeConfig(
		dt=0.001, steps_per_episode=int(_p(params, "controller_steps", 1500)),
		max_initial_tilt_rad=tilt, max_initial_yaw_rad=tilt,
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)


def build_controller_evaluator(params: dict) -> ControllerEvaluator:
	"""ControllerEvaluator for a controller flow (worker evaluator-dispatch hook)."""
	return ControllerEvaluator(
		controller_spec_from_params(params),
		num_eval_episodes=int(_p(params, "controller_eval_episodes", 20)),
		seed=int(_p(params, "seed", 0)),
		episode_config=_episode_config(params),
	)


def resolve_phase(phase_type: str) -> tuple:
	"""phase_type string → (StrategyKind, OptimizationDimension). Raises ValueError
	with the known set if unrecognized."""
	key = (phase_type or "").strip().lower()
	if key not in PHASE_TO_KIND_DIM:
		raise ValueError(
			f"unknown controller phase_type {phase_type!r}; known: {sorted(PHASE_TO_KIND_DIM)}")
	return PHASE_TO_KIND_DIM[key]


def build_controller_strategy(params: dict, phase_type: str, evaluator: ControllerEvaluator,
                              *, seed: Optional[int] = None, **extra) -> Any:
	"""Build the controller strategy for one experiment phase (experiment.py hook).

	`extra` forwards loop knobs (e.g. ga_config / ts_config / adapt_config) to the
	underlying strategy. The no-train MEMORY strategies must be driven with
	`batch_evaluate_fn=evaluator.score_genomes`; the rest use evaluator.evaluate_batch
	(GA/TS) or evaluator.evaluate_for_adaptation (Lamarckian)."""
	kind, dim = resolve_phase(phase_type)
	if seed is None:
		seed = int(_p(params, "seed", 0))
	return create_strategy(WnnType.CONTROLLER, kind, dim,
	                       spec=controller_spec_from_params(params),
	                       seed=seed, batch_evaluator=evaluator, **extra)


def batch_eval_fn_for(phase_type: str, evaluator: ControllerEvaluator):
	"""The correct batch_evaluate_fn for optimize(): MEMORY (paradigm B) scores
	WITHOUT training (score_genomes); everything else trains (evaluate_batch)."""
	_kind, dim = resolve_phase(phase_type)
	return evaluator.score_genomes if dim == Dim.MEMORY else evaluator.evaluate_batch


__all__ = [
	"PHASE_TO_KIND_DIM", "controller_spec_from_params", "build_controller_evaluator",
	"resolve_phase", "build_controller_strategy", "batch_eval_fn_for",
]
