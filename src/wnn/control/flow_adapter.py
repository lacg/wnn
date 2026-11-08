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
          strategy = OptimizerStrategyFactory.create(strategy_type, opt_config, ...)

Param keys (all optional, sensible defaults; namespaced `controller_*`):
  controller_num_motors(4) controller_levels_per_motor(16) controller_bits_per_feature(8)
  controller_input_window_k(4) controller_state_neurons(4) controller_state_bits(24)
  controller_output_bits(24) controller_delta_control(False)
  controller_eval_episodes(20) controller_steps(1500) controller_tilt_deg(15.0) seed(0)
  controller_memory_mode(QUAD_WEIGHTED) controller_disturbance_level(OFF)
  controller_conn_policy(spread | min1 | min2 | min3 — framed1 refused)

Every OTHER controller_* key in KNOWN_PARAMS is a phased_ga knob this adapter does
not build (vertical/horizontal channel, cascade, arm D, ...). It is REFUSED with a
pointer to phased_ga instead of silently dropped (08/11/2026: 23 keys were dropped).
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
from .recurrent_genome import conn_policy_from_name
from .training import DisturbanceConfig, EpisodeConfig


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


# The controller_* keys this adapter honours. Any other controller_* key with a
# value is refused (see _refuse_unbuilt_keys).
ADAPTER_KEYS = frozenset({
	"controller_num_motors", "controller_levels_per_motor", "controller_bits_per_feature",
	"controller_input_window_k", "controller_state_neurons", "controller_state_bits",
	"controller_output_bits", "controller_delta_control", "controller_memory_mode",
	"controller_output_mode", "controller_conn_policy",
	"controller_eval_episodes", "controller_steps", "controller_tilt_deg",
	"controller_disturbance_level",
})


def _refuse_unbuilt_keys(params: dict) -> None:
	"""Refuse controller_* keys this adapter does not build (the silently-ignored-
	key failure): a flow asking for e.g. controller_translation must not run a
	controller without it."""
	bad = sorted(k for k, v in params.items()   # None / False = the flag's off default
	             if k.startswith("controller_") and v is not None and v is not False
	             and k not in ADAPTER_KEYS)
	if bad:
		raise ValueError(f"{bad}: not built by the dashboard flow adapter (motor-level, legacy "
		                 "plant only) — run these through phased_ga.")


def _conn_policy(params: dict) -> tuple[str, int]:
	"""controller_conn_policy -> (policy, m), the same parser as phased_ga --conn-policy.
	framed1 is refused: it needs arm D (output_full_window), which this adapter
	does not build — without it every neuron would silently draw as min1."""
	name = str(_p(params, "controller_conn_policy", "spread"))
	if name == "framed1":
		raise ValueError("controller_conn_policy=framed1 needs output_full_window (arm D), which "
		                 "the dashboard flow adapter does not build — run it through phased_ga.")
	return conn_policy_from_name(name)


def controller_spec_from_params(params: dict) -> ControllerSpec:
	"""Build a ControllerSpec from a flow's flat params dict."""
	# ABI 38: the cascade needs the airframe's firmware rate loop, which this
	# adapter's EpisodeConfig does not carry — refuse rather than build a motor-level
	# controller under a "ctbr" flow (the silently-ignored-key failure).
	mode = str(_p(params, "controller_output_mode", "pwm"))
	# CTRL-38: the cascade input blocks only exist under ctbr — refuse them here too
	# rather than build a controller that silently lacks them.
	for key in ("controller_obs_rate_i", "controller_obs_cascade_dhat"):
		if bool(_p(params, key, False)):
			raise ValueError(f"{key}: a cascade (ctbr) input block — the dashboard flow adapter builds "
			                 "motor-level controllers only; run it through phased_ga --output-mode ctbr.")
	_refuse_unbuilt_keys(params)
	if mode != "pwm":
		raise ValueError(
			f"controller_output_mode={mode!r}: the dashboard flow adapter builds motor-level "
			"controllers only (no airframe cascade to be the CTBR inner loop) — run the cascade "
			"through phased_ga --output-mode ctbr.")
	sn = int(_p(params, "controller_state_neurons", 4))
	# Floor: bits must hold the forced full-state prefix + ≥1 sampled. The prefix is
	# prefix_factor·state_neurons = 1·sn since the 08/06/2026 1-bit migration (was 2·sn).
	sbits = max(int(_p(params, "controller_state_bits", 24)), sn + 1)
	obits = max(int(_p(params, "controller_output_bits", 24)), sn + 1)
	policy, policy_min = _conn_policy(params)
	return ControllerSpec(
		num_motors=int(_p(params, "controller_num_motors", 4)),
		levels_per_motor=int(_p(params, "controller_levels_per_motor", 16)),
		bits_per_feature=int(_p(params, "controller_bits_per_feature", 8)),
		input_window_k=int(_p(params, "controller_input_window_k", 4)),
		state_neurons=sn,
		state_bits_per_neuron=sbits,
		output_bits_per_neuron=obits,
		delta_control=bool(_p(params, "controller_delta_control", False)),
		memory_mode=str(_p(params, "controller_memory_mode", "QUAD_WEIGHTED")),
		conn_policy=policy, conn_policy_min=policy_min,
	)


def _episode_config(params: dict) -> EpisodeConfig:
	tilt = math.radians(float(_p(params, "controller_tilt_deg", 15.0)))
	# W2 disturbance passthrough: level preset name (OFF/L1/L2/L3/L2D/L3D).
	level = str(_p(params, "controller_disturbance_level", "OFF"))
	return EpisodeConfig(
		dt=0.001, steps_per_episode=int(_p(params, "controller_steps", 1500)),
		max_initial_tilt_rad=tilt, max_initial_yaw_rad=tilt,
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=DisturbanceConfig.preset(level, seed=int(_p(params, "seed", 0))),
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
