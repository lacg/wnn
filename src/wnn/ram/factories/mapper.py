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
		mapper = MapperFactory.create("hash", n_bits=8, hash_bits=6)
		mapper = MapperFactory.create("residual", n_bits=8)
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
		hash_bits: int = 6,
		n_hash_functions: int = 3,
		rng: int | None = None,
	) -> Module:
		"""
		Create a mapper based on the specified strategy.

		Args:
			strategy: MapperStrategy enum or string
			n_bits: Number of bits to process
			n_groups: Number of groups for compositional strategies
			context_mode: Context mode for bit-level/residual mapper
			output_mode: Output mode for bit-level mapper
			local_window: Window size for LOCAL/BIDIRECTIONAL context modes
			cross_group_context: Whether groups can see each other
			hash_bits: Number of hash bits for HASH strategy
			n_hash_functions: Number of hash functions for HASH strategy
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
			HashMapper,
			ResidualMapper,
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
			case MapperStrategy.HASH:
				return HashMapper(
					n_bits=n_bits,
					hash_bits=hash_bits,
					n_hash_functions=n_hash_functions,
					rng=rng,
				)
			case MapperStrategy.RESIDUAL:
				return ResidualMapper(
					n_bits=n_bits,
					context_mode=context_mode,
					local_window=local_window,
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
