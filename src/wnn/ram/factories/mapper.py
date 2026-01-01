"""
Mapper Factory

Factory for creating mapper instances based on MapperStrategy.
"""

from torch.nn import Module

from wnn.ram.enums import MapperStrategy, ContextMode, BitMapperMode


class MapperFactory:
	"""
	Factory for creating mappers based on strategy.

	This provides a clean interface for creating the appropriate mapper
	without needing to import individual mapper classes.

	Example:
		mapper = MapperFactory.create(MapperStrategy.BIT_LEVEL, n_bits=8)
		mapper = MapperFactory.create("bit_level", n_bits=8)  # string also works
	"""

	@staticmethod
	def create(
		strategy: MapperStrategy | str,
		n_bits: int,
		n_groups: int = 2,
		context_mode: ContextMode | str = ContextMode.CUMULATIVE,
		output_mode: BitMapperMode | str = BitMapperMode.FLIP,
		local_window: int = 3,
		cross_group_context: bool = True,
		rng: int | None = None,
	) -> Module:
		"""
		Create a mapper based on the specified strategy.

		Args:
			strategy: MapperStrategy enum or string
			n_bits: Number of bits to process
			n_groups: Number of groups for compositional strategies
			context_mode: Context mode for bit-level mapper
			output_mode: Output mode for bit-level mapper
			local_window: Window size for LOCAL context mode
			cross_group_context: Whether groups can see each other
			rng: Random seed

		Returns:
			Configured mapper module
		"""
		# Lazy imports to avoid circular dependencies
		from wnn.ram.core import RAMLayer
		from wnn.ram.core import (
			BitLevelMapper,
			CompositionalMapper,
			GeneralizingProjection,
		)

		# Convert string to enum if needed
		if isinstance(strategy, str):
			strategy = MapperStrategy[strategy.upper()]
		if isinstance(context_mode, str):
			context_mode = ContextMode[context_mode.upper()]
		if isinstance(output_mode, str):
			output_mode = BitMapperMode[output_mode.upper()]

		match strategy:
			case MapperStrategy.DIRECT:
				return RAMLayer(
					total_input_bits=n_bits,
					num_neurons=n_bits,
					n_bits_per_neuron=min(n_bits, 12),
					rng=rng,
				)
			case MapperStrategy.BIT_LEVEL:
				return BitLevelMapper(
					n_bits=n_bits,
					context_mode=context_mode,
					output_mode=output_mode,
					local_window=local_window,
					rng=rng,
				)
			case MapperStrategy.COMPOSITIONAL:
				return CompositionalMapper(
					n_bits=n_bits,
					n_groups=n_groups,
					cross_group_context=cross_group_context,
					rng=rng,
				)
			case MapperStrategy.HYBRID:
				return GeneralizingProjection(
					input_bits=n_bits,
					output_bits=n_bits,
					strategy=MapperStrategy.HYBRID,
					n_groups=n_groups,
					rng=rng,
				)
			case _:
				raise ValueError(f"Unknown strategy: {strategy}")

	@staticmethod
	def create_projection(
		input_bits: int,
		output_bits: int,
		strategy: MapperStrategy = MapperStrategy.BIT_LEVEL,
		n_groups: int = 2,
		rng: int | None = None,
	) -> Module:
		"""
		Create a generalizing projection.

		Args:
			input_bits: Number of input bits
			output_bits: Number of output bits
			strategy: Generalization strategy
			n_groups: Number of groups for compositional
			rng: Random seed

		Returns:
			GeneralizingProjection instance
		"""
		from wnn.ram.core import GeneralizingProjection

		return GeneralizingProjection(
			input_bits=input_bits,
			output_bits=output_bits,
			strategy=strategy,
			n_groups=n_groups,
			rng=rng,
		)
