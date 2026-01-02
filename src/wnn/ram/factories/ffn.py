"""
FFN Factory

Factory for creating feed-forward network layers.
Uses match-case for clean dispatch.
"""

from torch.nn import Module

from wnn.ram.enums import FFNType, ArithmeticOp


class FFNFactory:
	"""
	Factory for creating FFN layers.

	Supports both learned and computed FFN types.
	Computed types achieve 100% generalization with no training.
	"""

	@staticmethod
	def create(
		ffn_type: FFNType,
		input_bits: int,
		hidden_bits: int | None = None,
		constant: int = 1,
		modulo: int | None = None,
		rng: int | None = None,
	) -> Module | None:
		"""
		Create an FFN layer based on type.

		Args:
			ffn_type: Type of FFN to create
			input_bits: Bits per token
			hidden_bits: Hidden dimension for TWO_LAYER
			constant: Constant for ADD_MOD/SUBTRACT_MOD
			modulo: Modulo for ADD_MOD/SUBTRACT_MOD
			rng: Random seed

		Returns:
			FFN module or None if NONE type
		"""
		# Lazy imports to avoid circular dependencies
		from wnn.ram.core.RAMLayer import RAMLayer
		from wnn.ram.core.RAMGeneralization import GeneralizingProjection, MapperStrategy
		from wnn.ram.core.models.two_layer_ffn import TwoLayerFFN
		from wnn.ram.core.models.computed_arithmetic import ComputedArithmeticFFN

		match ffn_type:
			case FFNType.NONE:
				return None

			# Learned FFN types
			case FFNType.SINGLE:
				return RAMLayer(
					total_input_bits=input_bits,
					num_neurons=input_bits,
					n_bits_per_neuron=min(input_bits, 8),
					rng=rng,
				)

			case FFNType.BIT_LEVEL:
				return GeneralizingProjection(
					input_bits=input_bits,
					output_bits=input_bits,
					strategy=MapperStrategy.BIT_LEVEL,
					rng=rng,
				)

			case FFNType.TWO_LAYER:
				return TwoLayerFFN(
					input_bits=input_bits,
					hidden_bits=hidden_bits if hidden_bits else input_bits * 2,
					output_bits=input_bits,
					rng=rng,
				)

			# Computed FFN types (100% generalization)
			case FFNType.INCREMENT:
				return ComputedArithmeticFFN(
					input_bits=input_bits,
					operation=ArithmeticOp.INCREMENT,
					rng=rng,
				)

			case FFNType.DECREMENT:
				return ComputedArithmeticFFN(
					input_bits=input_bits,
					operation=ArithmeticOp.DECREMENT,
					rng=rng,
				)

			case FFNType.ADD_MOD:
				return ComputedArithmeticFFN(
					input_bits=input_bits,
					operation=ArithmeticOp.ADD_MOD,
					constant=constant,
					modulo=modulo if modulo else 26,
					rng=rng,
				)

			case FFNType.SUBTRACT_MOD:
				return ComputedArithmeticFFN(
					input_bits=input_bits,
					operation=ArithmeticOp.SUBTRACT_MOD,
					constant=constant,
					modulo=modulo if modulo else 26,
					rng=rng,
				)

			case FFNType.ROT13:
				return ComputedArithmeticFFN(
					input_bits=input_bits,
					operation=ArithmeticOp.ROT13,
					rng=rng,
				)

			case FFNType.NEGATE:
				return ComputedArithmeticFFN(
					input_bits=input_bits,
					operation=ArithmeticOp.NEGATE,
					rng=rng,
				)

			case _:
				raise ValueError(f"Unknown FFN type: {ffn_type}")
