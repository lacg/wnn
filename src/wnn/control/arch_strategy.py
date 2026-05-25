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

from wnn.ram.strategies.connectivity.generic_strategies import GenericGAStrategy, GAConfig
from wnn.ram.strategies.optimization_dimension import OptimizationDimension

from .evaluator import ControllerSpec, arch_shape_from_spec
from .ga_strategy import default_controller_ga_config
from .recurrent_genome import RecurrentArchGenome, RecurrentArchShape, RecurrentArchConfig


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

	def create_random_genome(self) -> RecurrentArchGenome:
		"""Randomize ONLY the optimized dimension; pin the rest to the seed spec
		(so a GA-Bits population varies bits but shares neuron counts, etc.)."""
		self._random_counter += 1
		rng = np.random.default_rng(
			(0 if self._seed is None else self._seed) * 100_000 + self._random_counter)
		cfg, q = self._arch_config, self._shape.output_quantum
		dim = self._dimension

		sn = self._seed_state_neurons
		on = self._seed_output_neurons
		ssuf = self._seed_state_suffix
		osuf = self._seed_output_suffix

		if dim in (OptimizationDimension.NEURONS, OptimizationDimension.CLUSTER):
			sn = int(rng.integers(cfg.min_state_neurons, cfg.max_state_neurons + 1))
			lo_b, hi_b = max(1, cfg.min_output_neurons // q), max(1, cfg.max_output_neurons // q)
			on = int(rng.integers(lo_b, hi_b + 1)) * q
		if dim in (OptimizationDimension.BITS, OptimizationDimension.CLUSTER):
			cap_s = min(cfg.max_suffix, self._shape.state_input_space)
			cap_o = min(cfg.max_suffix, self._shape.output_input_space)
			ssuf = int(rng.integers(cfg.min_suffix, cap_s + 1))
			osuf = int(rng.integers(cfg.min_suffix, cap_o + 1))
		# CONNECTIONS / any: suffixes are random-sampled regardless, dims pinned.
		return RecurrentArchGenome.random(self._shape, sn, on, ssuf, osuf, rng)


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


__all__ = [
	"ControllerArchGAStrategy",
	"default_controller_arch_config",
	"controller_ga_neurons",
	"controller_ga_bits",
	"controller_ga_connections",
]
