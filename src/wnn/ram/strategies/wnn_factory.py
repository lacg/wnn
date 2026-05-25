"""WnnType factory — one entry point to build an optimization strategy for any
WNN family (IDS, controller, LM) across the GA / TS / Lamarckian kinds and the
NEURONS / BITS / CONNECTIONS / MEMORY dimensions.

WHY A REGISTRY (not direct imports)
-----------------------------------
The controller code (src/wnn/control/) is *additive*: nothing in src/wnn/ram/ or
src/wnn/ids/ imports from it (see control/__init__.py). To respect that while
still letting one factory build controller strategies, each domain SELF-REGISTERS
a builder here when its module is imported — the factory never imports a domain.
Import direction stays one-way (control → ram), and the running IDS flow never
pulls in the controller's heavier deps (sim, ram_accelerator controller).

Usage:
    import wnn.control.arch_strategy            # registers WnnType.CONTROLLER
    from wnn.ram.strategies.wnn_factory import create_strategy, WnnType, StrategyKind
    from wnn.ram.strategies.optimization_dimension import OptimizationDimension as Dim
    strat = create_strategy(WnnType.CONTROLLER, StrategyKind.GA, Dim.NEURONS, spec=spec)

CAPABILITIES is the declared (WnnType → supported dimensions × kinds) map — the
single source of truth for what each family CAN optimize. `create_strategy`
validates against it before dispatching to the registered builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from wnn.ram.strategies.optimization_dimension import OptimizationDimension as Dim


class WnnType(Enum):
	"""WNN family — selects which genome + strategy classes a build resolves to."""
	IDS = "ids"
	CONTROLLER = "controller"
	LM = "lm"


class StrategyKind(Enum):
	"""Optimization meta-strategy (orthogonal to the dimension being optimized)."""
	GA = "ga"               # population search
	TS = "ts"               # local search with tabu list
	LAMARCKIAN = "lamarckian"  # stats-guided genesis (inherits acquired traits)


@dataclass(frozen=True)
class WnnCapability:
	"""What a WNN family can optimize: which dimensions × which strategy kinds."""
	dimensions: frozenset
	kinds: frozenset


# Declared conceptual capability per family. MEMORY (content) is controller-only
# today (paradigm B); IDS/LM trains cells rather than evolving them. This is the
# source of truth `create_strategy` validates against — a builder may still raise
# NotImplementedError for a declared-but-not-yet-wired (kind, dimension) combo.
CAPABILITIES: dict[WnnType, WnnCapability] = {
	WnnType.IDS: WnnCapability(
		frozenset({Dim.NEURONS, Dim.BITS, Dim.CONNECTIONS}),
		frozenset({StrategyKind.GA, StrategyKind.TS, StrategyKind.LAMARCKIAN})),
	WnnType.CONTROLLER: WnnCapability(
		frozenset({Dim.NEURONS, Dim.BITS, Dim.CONNECTIONS, Dim.MEMORY}),
		frozenset({StrategyKind.GA, StrategyKind.TS, StrategyKind.LAMARCKIAN})),
	WnnType.LM: WnnCapability(
		frozenset({Dim.NEURONS, Dim.BITS, Dim.CONNECTIONS}),
		frozenset({StrategyKind.GA, StrategyKind.TS})),
}

# A builder is (kind, dimension, **kwargs) -> strategy. Domains register theirs.
StrategyBuilder = Callable[..., Any]
_BUILDERS: dict[WnnType, StrategyBuilder] = {}


def register_wnn_type(wnn_type: WnnType, builder: StrategyBuilder) -> None:
	"""Register (or replace) a family's strategy builder. Called by each domain
	module at import time, so the factory itself imports no domain code."""
	_BUILDERS[wnn_type] = builder


def is_registered(wnn_type: WnnType) -> bool:
	return wnn_type in _BUILDERS


def supported_dimensions(wnn_type: WnnType) -> frozenset:
	return CAPABILITIES[wnn_type].dimensions


def supported_kinds(wnn_type: WnnType) -> frozenset:
	return CAPABILITIES[wnn_type].kinds


def supports(wnn_type: WnnType, kind: StrategyKind, dimension) -> bool:
	cap = CAPABILITIES.get(wnn_type)
	return bool(cap) and kind in cap.kinds and dimension in cap.dimensions


def create_strategy(wnn_type: WnnType, kind: StrategyKind, dimension, **kwargs) -> Any:
	"""Build the strategy for (family, kind, dimension). Validates against the
	declared CAPABILITIES, then delegates to the family's registered builder.

	Raises ValueError for an undeclared combo, and a clear error if the family's
	module hasn't been imported (so its builder isn't registered yet)."""
	cap = CAPABILITIES.get(wnn_type)
	if cap is None:
		raise ValueError(f"unknown WnnType: {wnn_type!r}")
	if dimension not in cap.dimensions:
		raise ValueError(
			f"{wnn_type.value} does not optimize {getattr(dimension, 'name', dimension)} "
			f"(supported: {sorted(d.name for d in cap.dimensions)})")
	if kind not in cap.kinds:
		raise ValueError(
			f"{wnn_type.value} does not support {kind.value} "
			f"(supported: {sorted(k.value for k in cap.kinds)})")
	builder = _BUILDERS.get(wnn_type)
	if builder is None:
		raise NotImplementedError(
			f"no builder registered for {wnn_type.value}; import its strategy module "
			f"first (e.g. `import wnn.control.arch_strategy` for CONTROLLER)")
	return builder(kind, dimension, **kwargs)


__all__ = [
	"WnnType", "StrategyKind", "WnnCapability", "CAPABILITIES",
	"register_wnn_type", "is_registered", "supported_dimensions",
	"supported_kinds", "supports", "create_strategy",
]
