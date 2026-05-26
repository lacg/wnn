"""ControllerArchGAStrategy — phase-aware architecture GA for the controller.

The variable-shape counterpart to ControllerGAStrategy (ga_strategy.py). Where
that one evolves a fixed-shape FiniteStateGenome's connectivity only, this drives
a RecurrentArchGenome through ONE optimization dimension at a time — NEURONS,
BITS, or CONNECTIONS — mirroring how the IDS ArchitectureGAStrategy uses
optimize_neurons / optimize_bits / optimize_connections.

Phase gating is structural, not config-flag based: the genome's own
`mutate(dimension, …)` dispatch isolates each dimension, so a GA-Neurons phase
physically cannot drift bits or connections. `create_random_genome` likewise
randomizes ONLY the optimized dimension and pins the rest to the seed spec, so
the initial population explores the right axis.

Reuses the whole proven GenericGAStrategy loop (tournament, elitism, early-stop,
dashboard tracker) and the controller's reward-based ranking + disabled
accuracy-floor (see default_controller_ga_config). The evaluator
(ControllerEvaluator) is now variable-shape-aware: it materializes each genome's
own spec + connectivity and scores shape-uniform groups on the GPU.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from wnn.ram.strategies.connectivity.generic_strategies import (
	GenericGAStrategy, GenericTSStrategy, GAConfig, TSConfig,
)
from wnn.ram.strategies.optimization_dimension import OptimizationDimension
from wnn.ram.fitness import FitnessCalculatorType

from .evaluator import ControllerSpec, arch_shape_from_spec
from .ga_strategy import default_controller_ga_config
from .recurrent_genome import RecurrentArchGenome, RecurrentArchShape, RecurrentArchConfig


# ---- shared genome-op helpers (used by both GA and TS controller strategies) --

def _random_arch_genome(shape: RecurrentArchShape, dimension: OptimizationDimension,
                        cfg: RecurrentArchConfig, base: tuple[int, int, int, int],
                        rng: np.random.Generator) -> RecurrentArchGenome:
	"""Build a random genome that varies ONLY the optimized dimension, pinning the
	rest to the seed dims `base = (state_neurons, output_neurons, state_suffix,
	output_suffix)`. This is what makes a GA-Bits population vary bits but share
	neuron counts, etc. (the IDS optimize_* gating, applied to initialization)."""
	sn, on, ssuf, osuf = base
	q = shape.output_quantum
	if dimension in (OptimizationDimension.NEURONS, OptimizationDimension.CLUSTER):
		sn = int(rng.integers(cfg.min_state_neurons, cfg.max_state_neurons + 1))
		lo_b, hi_b = max(1, cfg.min_output_neurons // q), max(1, cfg.max_output_neurons // q)
		on = int(rng.integers(lo_b, hi_b + 1)) * q
	if dimension in (OptimizationDimension.BITS, OptimizationDimension.CLUSTER):
		ssuf = int(rng.integers(cfg.min_suffix, min(cfg.max_suffix, shape.state_input_space) + 1))
		osuf = int(rng.integers(cfg.min_suffix, min(cfg.max_suffix, shape.output_input_space) + 1))
	return RecurrentArchGenome.random(shape, sn, on, ssuf, osuf, rng)


def _arch_move_info(dimension: OptimizationDimension, before: RecurrentArchGenome,
                    after: RecurrentArchGenome):
	"""Tabu move descriptor. NEURONS/BITS → directional axis tokens (axis, ±1) so
	only the REVERSE is tabu; CONNECTIONS → the set of changed neuron tokens
	(overlap-based tabu, mirroring IDS). None if nothing changed."""
	tok: list = []
	if dimension == OptimizationDimension.NEURONS:
		if after.state_neurons != before.state_neurons:
			tok.append(("SN", 1 if after.state_neurons > before.state_neurons else -1))
		if after.output_neurons != before.output_neurons:
			tok.append(("ON", 1 if after.output_neurons > before.output_neurons else -1))
	elif dimension == OptimizationDimension.BITS:
		if after.state_suffix_width != before.state_suffix_width:
			tok.append(("SB", 1 if after.state_suffix_width > before.state_suffix_width else -1))
		if after.output_suffix_width != before.output_suffix_width:
			tok.append(("OB", 1 if after.output_suffix_width > before.output_suffix_width else -1))
	else:  # CONNECTIONS — counts unchanged, compare per-neuron suffixes
		tok += [("S", i) for i in range(before.state_neurons)
		        if before.state_sampled[i] != after.state_sampled[i]]
		tok += [("O", i) for i in range(before.output_neurons)
		        if before.output_sampled[i] != after.output_sampled[i]]
	return tuple(tok) if tok else None


def _arch_is_tabu(dimension: OptimizationDimension, move, tabu_list: list) -> bool:
	if move is None or not move:
		return False
	if dimension in (OptimizationDimension.NEURONS, OptimizationDimension.BITS):
		# Tabu iff this move reverses a recent one (opposite sign on a shared axis).
		reversed_axes = {(axis, -sign) for (axis, sign) in move}
		return any(tm and reversed_axes & set(tm) for tm in tabu_list)
	# CONNECTIONS: tabu iff >50% of changed neurons overlap a recent move.
	move_set = set(move)
	return any(tm and len(move_set & set(tm)) > len(move_set) * 0.5 for tm in tabu_list)


def default_controller_arch_config(spec: ControllerSpec) -> RecurrentArchConfig:
	"""Mutation bounds centered on `spec`, scaled to a sensible search box.

	state_neurons ∈ [2, 4×spec], output levels ∈ [¼×, 4×] spec, suffix widths
	bounded by the per-layer input space. Deltas are small (the small-neighborhood
	rule) so each generation is a local move, not a random jump."""
	q = spec.num_motors
	levels = spec.levels_per_motor
	state_suffix = spec.state_bits_per_neuron - 2 * spec.state_neurons
	out_suffix = spec.output_bits_per_neuron - 2 * spec.state_neurons
	max_suffix = max(state_suffix, out_suffix, 8) * 2
	return RecurrentArchConfig(
		min_state_neurons=2,
		max_state_neurons=max(4, 4 * spec.state_neurons),
		min_output_neurons=max(q, (levels // 4) * q),
		max_output_neurons=max(4 * levels, levels) * q,
		min_suffix=1,
		max_suffix=max_suffix,
		state_neuron_delta=1,
		output_block_delta=1,
		suffix_delta=2,
	)


class ControllerArchGAStrategy(GenericGAStrategy):
	"""Single-dimension architecture GA over RecurrentArchGenome, ranked by reward."""

	def __init__(
		self,
		spec: ControllerSpec,
		dimension: OptimizationDimension,
		arch_config: Optional[RecurrentArchConfig] = None,
		ga_config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional[Any] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		# Set before super().__init__ — the base reads self.name (→ _dimension).
		self._dimension = dimension
		super().__init__(config=ga_config or default_controller_ga_config(),
		                 seed=seed, logger=logger)
		self._spec = spec
		self._arch_config = arch_config or default_controller_arch_config(spec)
		self._shape: RecurrentArchShape = arch_shape_from_spec(spec)
		self._batch_evaluator = batch_evaluator
		self._cached_evaluator = None        # no Rust search_offspring → batch_fn path
		self._checkpoint_config = None
		self._shutdown_check = shutdown_check
		self._np_rng = np.random.default_rng(0 if seed is None else seed)
		self._random_counter = 0
		# Seed-spec dims (the pinned values for non-optimized dimensions).
		self._seed_state_neurons = spec.state_neurons
		self._seed_output_neurons = spec.num_motors * spec.levels_per_motor
		self._seed_state_suffix = spec.state_bits_per_neuron - 2 * spec.state_neurons
		self._seed_output_suffix = spec.output_bits_per_neuron - 2 * spec.state_neurons

	@property
	def name(self) -> str:
		return f"ControllerGA-{self._dimension.name.title()}"

	# ---- the four genome operations the loop calls as black boxes -----------

	def clone_genome(self, genome: RecurrentArchGenome) -> RecurrentArchGenome:
		return genome.clone()

	def mutate_genome(self, genome: RecurrentArchGenome, mutation_rate: float) -> RecurrentArchGenome:
		return genome.mutate(self._dimension, mutation_rate, self._arch_config, self._np_rng)

	def crossover_genomes(self, parent1: RecurrentArchGenome,
	                      parent2: RecurrentArchGenome) -> RecurrentArchGenome:
		return RecurrentArchGenome.crossover(parent1, parent2, self._np_rng)

	def _seed_dims(self) -> tuple[int, int, int, int]:
		return (self._seed_state_neurons, self._seed_output_neurons,
		        self._seed_state_suffix, self._seed_output_suffix)

	def create_random_genome(self) -> RecurrentArchGenome:
		"""Randomize ONLY the optimized dimension; pin the rest to the seed spec
		(so a GA-Bits population varies bits but shares neuron counts, etc.)."""
		self._random_counter += 1
		rng = np.random.default_rng(
			(0 if self._seed is None else self._seed) * 100_000 + self._random_counter)
		return _random_arch_genome(self._shape, self._dimension, self._arch_config,
		                           self._seed_dims(), rng)


def default_controller_ts_config(
	iterations: int = 6,
	neighbors_per_iter: int = 12,
	tabu_size: int = 10,
) -> TSConfig:
	"""TSConfig wired for the controller: reward ranking, no accuracy floor (the
	TS analog of default_controller_ga_config). Iterations/neighbors scaled down
	from the IDS 100/50 — the controller's K-seed inner loop is far costlier."""
	return TSConfig(
		iterations=iterations,
		neighbors_per_iter=neighbors_per_iter,
		tabu_size=tabu_size,
		progressive_threshold=False,
		min_accuracy=0.0,
		min_accuracy_floor=0.0,
		fitness_calculator_type=FitnessCalculatorType.CONTROLLER,
	)


class ControllerArchTSStrategy(GenericTSStrategy):
	"""Single-dimension architecture Tabu Search over RecurrentArchGenome.

	Same phase-isolated mutation as the GA, but local search with a tabu list.
	Uses the base GenericTSStrategy Python neighbor loop (no Rust search) — it
	calls mutate_genome (→ neighbor + move) and is_tabu_move per candidate.
	"""

	def __init__(
		self,
		spec: ControllerSpec,
		dimension: OptimizationDimension,
		arch_config: Optional[RecurrentArchConfig] = None,
		ts_config: Optional[TSConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional[Any] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		self._dimension = dimension   # set before super (base reads self.name)
		super().__init__(config=ts_config or default_controller_ts_config(),
		                 seed=seed, logger=logger)
		self._spec = spec
		self._arch_config = arch_config or default_controller_arch_config(spec)
		self._shape: RecurrentArchShape = arch_shape_from_spec(spec)
		self._batch_evaluator = batch_evaluator
		self._cached_evaluator = None     # no Rust search_neighbors → Python loop
		self._checkpoint_config = None
		self._shutdown_check = shutdown_check
		self._np_rng = np.random.default_rng(0 if seed is None else seed)
		self._seed_state_neurons = spec.state_neurons
		self._seed_output_neurons = spec.num_motors * spec.levels_per_motor
		self._seed_state_suffix = spec.state_bits_per_neuron - 2 * spec.state_neurons
		self._seed_output_suffix = spec.output_bits_per_neuron - 2 * spec.state_neurons

	@property
	def name(self) -> str:
		return f"ControllerTS-{self._dimension.name.title()}"

	def clone_genome(self, genome: RecurrentArchGenome) -> RecurrentArchGenome:
		return genome.clone()

	def mutate_genome(self, genome: RecurrentArchGenome, mutation_rate: float):
		"""Neighbor + move descriptor (for tabu tracking)."""
		neighbor = genome.mutate(self._dimension, mutation_rate, self._arch_config, self._np_rng)
		return neighbor, _arch_move_info(self._dimension, genome, neighbor)

	def is_tabu_move(self, move, tabu_list: list) -> bool:
		return _arch_is_tabu(self._dimension, move, tabu_list)

	def seed_genome(self) -> RecurrentArchGenome:
		"""The seed-spec genome — a convenient TS starting point / initial_genome."""
		return _random_arch_genome(
			self._shape, self._dimension, self._arch_config,
			(self._seed_state_neurons, self._seed_output_neurons,
			 self._seed_state_suffix, self._seed_output_suffix), self._np_rng)


def controller_ts_neurons(spec: ControllerSpec, **kw) -> ControllerArchTSStrategy:
	"""TS-Neurons: local search over state-neuron count + output levels."""
	return ControllerArchTSStrategy(spec, OptimizationDimension.NEURONS, **kw)


def controller_ts_bits(spec: ControllerSpec, **kw) -> ControllerArchTSStrategy:
	"""TS-Bits: local search over sampled-suffix width per layer."""
	return ControllerArchTSStrategy(spec, OptimizationDimension.BITS, **kw)


def controller_ts_connections(spec: ControllerSpec, **kw) -> ControllerArchTSStrategy:
	"""TS-Connectivity: local search over which sampled input bits each neuron reads."""
	return ControllerArchTSStrategy(spec, OptimizationDimension.CONNECTIONS, **kw)


def controller_ga_neurons(spec: ControllerSpec, **kw) -> ControllerArchGAStrategy:
	"""GA-Neurons: evolve state-neuron count (memory capacity) + output levels
	(PWM resolution); bits + connectivity held at the seed."""
	return ControllerArchGAStrategy(spec, OptimizationDimension.NEURONS, **kw)


def controller_ga_bits(spec: ControllerSpec, **kw) -> ControllerArchGAStrategy:
	"""GA-Bits: evolve sampled-suffix width per layer (synaptogenesis); neuron
	counts + connectivity held at the seed."""
	return ControllerArchGAStrategy(spec, OptimizationDimension.BITS, **kw)


def controller_ga_connections(spec: ControllerSpec, **kw) -> ControllerArchGAStrategy:
	"""GA-Connectivity: evolve which sampled input bits each neuron reads
	(axonogenesis); shape held at the seed. Equivalent role to the legacy
	ControllerGAStrategy but on the variable-shape genome."""
	return ControllerArchGAStrategy(spec, OptimizationDimension.CONNECTIONS, **kw)


# ---- WnnType-factory registration -------------------------------------------
# Self-register the controller family so wnn_factory.create_strategy can build
# these without the ram-level factory ever importing control (registry pattern).

from wnn.ram.strategies.wnn_factory import WnnType, StrategyKind, register_wnn_type

_GA_BY_DIM = {
	OptimizationDimension.NEURONS: controller_ga_neurons,
	OptimizationDimension.BITS: controller_ga_bits,
	OptimizationDimension.CONNECTIONS: controller_ga_connections,
}
_TS_BY_DIM = {
	OptimizationDimension.NEURONS: controller_ts_neurons,
	OptimizationDimension.BITS: controller_ts_bits,
	OptimizationDimension.CONNECTIONS: controller_ts_connections,
}


# Lamarckian dimension → genesis mode. CONNECTIONS (axonogenesis) needs per-input-
# bit entropy from new Rust instrumentation and is deferred (step 4b-5).
_GENESIS_MODE_BY_DIM = {
	OptimizationDimension.NEURONS: "neurogenesis",
	OptimizationDimension.BITS: "synaptogenesis",
}


def _controller_strategy_builder(kind: StrategyKind, dimension: OptimizationDimension,
                                 *, spec: ControllerSpec, **kwargs):
	"""WnnType.CONTROLLER builder. GA/TS over the architecture dimensions and
	Lamarckian neuro/synapto-genesis are wired; MEMORY (paradigm B) and
	Lamarckian axonogenesis are pending."""
	if kind == StrategyKind.GA and dimension in _GA_BY_DIM:
		return _GA_BY_DIM[dimension](spec, **kwargs)
	if kind == StrategyKind.TS and dimension in _TS_BY_DIM:
		return _TS_BY_DIM[dimension](spec, **kwargs)
	if kind == StrategyKind.LAMARCKIAN:
		mode = _GENESIS_MODE_BY_DIM.get(dimension)
		if mode is not None:
			from .arch_adaptation import ControllerAdaptationStrategy  # lazy: avoid import cycle
			return ControllerAdaptationStrategy(spec, genesis_mode=mode, **kwargs)
		if dimension == OptimizationDimension.CONNECTIONS:
			raise NotImplementedError(
				"controller axonogenesis (Lamarckian CONNECTIONS) needs per-input-bit "
				"entropy stats from new Rust instrumentation — Phase B step 4b-5.")
	if dimension == OptimizationDimension.MEMORY:
		raise NotImplementedError(
			"controller MEMORY dimension = paradigm B (ga_memory.ControllerMemoryGAStrategy); "
			"factory wiring needs a recorded address universe — pending step 4b-4.")
	raise ValueError(f"unsupported controller strategy: kind={kind}, dimension={dimension}")


register_wnn_type(WnnType.CONTROLLER, _controller_strategy_builder)


__all__ = [
	"ControllerArchGAStrategy",
	"ControllerArchTSStrategy",
	"default_controller_arch_config",
	"default_controller_ts_config",
	"controller_ga_neurons",
	"controller_ga_bits",
	"controller_ga_connections",
	"controller_ts_neurons",
	"controller_ts_bits",
	"controller_ts_connections",
]
