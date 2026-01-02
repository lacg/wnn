"""
RAM Transformer Factory

Factory for creating pre-configured RAM Transformers.
Uses match-case for clean dispatch.
"""

from wnn.ram.enums import (
	RAMTransformerType,
	AttentionType,
	FFNType,
	ContentMatchMode,
	Step,
)
from wnn.ram.encoders_decoders import PositionMode
from wnn.ram.factories.config import StepConfigurationFactory


class RAMTransformerFactory:
	"""
	Factory for creating pre-configured RAM Transformers.

	Each transformer type is optimized for a specific task
	with appropriate attention and FFN configurations.
	"""

	@staticmethod
	def create(
		transformer_type: RAMTransformerType,
		input_bits: int,
		max_seq_len: int = 16,
		# Caesar cipher parameters
		caesar_shift: int = 3,
		rng: int | None = None,
	):
		"""
		Create a pre-configured RAM Transformer.

		Args:
			transformer_type: Type of transformer to create
			input_bits: Bits per token
			max_seq_len: Maximum sequence length
			caesar_shift: Shift amount for CAESAR type
			rng: Random seed

		Returns:
			Configured RAMTransformer instance
		"""
		# Lazy import to avoid circular dependency
		from wnn.ram.core.models.transformer import RAMTransformer

		match transformer_type:
			# Position-based transformers (100% generalization)
			case RAMTransformerType.COPY:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.POSITION_ONLY,
					position_mode=PositionMode.RELATIVE,
					causal=False,
					ffn_type=FFNType.NONE,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case RAMTransformerType.SHIFT:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.POSITION_ONLY,
					position_mode=PositionMode.RELATIVE,
					causal=True,
					ffn_type=FFNType.NONE,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case RAMTransformerType.REVERSE:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.POSITION_ONLY,
					position_mode=PositionMode.BINARY,
					causal=False,
					ffn_type=FFNType.NONE,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			# Computed attention transformers
			case RAMTransformerType.SORTING:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.SORTING,
					ffn_type=FFNType.NONE,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case RAMTransformerType.SELF_MATCHING:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.CONTENT_MATCH,
					content_match=ContentMatchMode.XOR_EQUAL,
					causal=False,
					ffn_type=FFNType.BIT_LEVEL,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			# Computed FFN transformers (100% generalization)
			case RAMTransformerType.INCREMENT:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.POSITION_ONLY,
					position_mode=PositionMode.RELATIVE,
					causal=False,
					ffn_type=FFNType.INCREMENT,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case RAMTransformerType.DECREMENT:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.POSITION_ONLY,
					position_mode=PositionMode.RELATIVE,
					causal=False,
					ffn_type=FFNType.DECREMENT,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case RAMTransformerType.ROT13:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.POSITION_ONLY,
					position_mode=PositionMode.RELATIVE,
					causal=False,
					ffn_type=FFNType.ROT13,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case RAMTransformerType.CAESAR:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.POSITION_ONLY,
					position_mode=PositionMode.RELATIVE,
					causal=False,
					ffn_type=FFNType.ADD_MOD,
					ffn_constant=caesar_shift,
					ffn_modulo=26,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case RAMTransformerType.NEGATE:
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=1,
					attention_type=AttentionType.POSITION_ONLY,
					position_mode=PositionMode.RELATIVE,
					causal=False,
					ffn_type=FFNType.NEGATE,
					use_residual=False,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case _:
				raise ValueError(f"Unknown transformer type: {transformer_type}")

	@staticmethod
	def create_multi_step(
		input_bits: int,
		steps: list[Step],
		max_seq_len: int = 16,
		rng: int | None = None,
	):
		"""
		Create a multi-step transformer from a list of steps.

		Args:
			input_bits: Bits per token
			steps: List of Step enums defining the pipeline
			max_seq_len: Maximum sequence length
			rng: Random seed

		Returns:
			RAMTransformer with one block per step
		"""
		from wnn.ram.core.models.transformer import RAMTransformer

		block_configs = StepConfigurationFactory.create_many(steps)

		return RAMTransformer(
			input_bits=input_bits,
			num_blocks=len(steps),
			ffn_type=FFNType.NONE,
			use_residual=False,
			max_seq_len=max_seq_len,
			block_configs=block_configs,
			rng=rng,
		)
