"""
Unified Models Factory

Single entry point for creating all RAM model types.
Uses match-case dispatch on ModelType enum with explicit parameters.

Usage:
	from wnn.ram.factories import ModelsFactory
	from wnn.ram.core.models import ModelType, RAMTransformerType

	# Create a transformer (preset type)
	model = ModelsFactory.create(
		ModelType.TRANSFORMER,
		input_bits=8,
		transformer_type=RAMTransformerType.COPY,
	)

	# Create a recurrent network
	model = ModelsFactory.create(
		ModelType.RECURRENT,
		input_bits=8,
		n_state_neurons=16,
		n_output_neurons=8,
	)
"""

from typing import Optional

from wnn.ram.core.models import (
	ModelType,
	RAMTransformerType,
	AttentionType,
	FFNType,
	ContentMatchMode,
	Step,
)
from wnn.ram.encoders_decoders import OutputMode, PositionMode


class ModelsFactory:
	"""
	Unified factory for creating RAM models.

	Single `create()` method dispatches on ModelType enum.
	All parameters are explicit with sensible defaults.
	"""

	@staticmethod
	def create(
		model_type: ModelType,
		# === Universal parameters ===
		input_bits: int = 8,
		rng: Optional[int] = None,
		# === Transformer parameters ===
		transformer_type: Optional[RAMTransformerType] = None,
		steps: Optional[list[Step]] = None,
		num_blocks: int = 2,
		num_heads: int = 8,
		attention_type: AttentionType = AttentionType.POSITION_ONLY,
		content_match: ContentMatchMode = ContentMatchMode.NONE,
		position_mode: PositionMode = PositionMode.RELATIVE,
		causal: bool = True,
		ffn_type: FFNType = FFNType.BIT_LEVEL,
		ffn_constant: int = 1,
		ffn_modulo: Optional[int] = None,
		use_residual: bool = True,
		max_seq_len: int = 16,
		caesar_shift: int = 3,
		block_configs: Optional[list[dict]] = None,
		# === Recurrent parameters ===
		n_state_neurons: int = 16,
		n_output_neurons: int = 8,
		n_bits_per_state_neuron: int = 4,
		n_bits_per_output_neuron: int = 4,
		use_hashing: bool = False,
		hash_size: int = 1024,
		max_iters: int = 4,
		output_mode: OutputMode = OutputMode.HAMMING,
		# === KV Memory parameters ===
		spec=None,  # KVSpec - no type hint to avoid circular import
		neurons_per_head: int = 8,
		# === Seq2Seq parameters ===
		output_bits: Optional[int] = None,
		num_encoder_blocks: int = 2,
		num_decoder_blocks: int = 2,
	):
		"""
		Create a RAM model of the specified type.

		Args:
			model_type: The type of model to create (ModelType enum)

			Universal:
				input_bits: Bits per input token (default: 8)
				rng: Random seed (default: None)

			Transformer:
				transformer_type: Preset type (COPY, SHIFT, etc.)
				steps: Multi-step pipeline steps
				num_blocks: Number of transformer blocks
				num_heads: Number of attention heads
				attention_type: Attention mechanism type
				content_match: Content matching mode
				position_mode: Position encoding mode
				causal: Use causal masking
				ffn_type: Feed-forward network type
				ffn_constant: Constant for arithmetic FFN
				ffn_modulo: Modulo for arithmetic FFN
				use_residual: Use residual connections
				max_seq_len: Maximum sequence length
				caesar_shift: Caesar cipher shift
				block_configs: Per-block overrides

			Recurrent:
				n_state_neurons: Number of state neurons
				n_output_neurons: Number of output neurons
				n_bits_per_state_neuron: Bits per state connection
				n_bits_per_output_neuron: Bits per output connection
				use_hashing: Use hash-based memory
				hash_size: Hash table size
				max_iters: Maximum EDRA iterations
				output_mode: Output decoding mode

			KV_Memory (extends Recurrent):
				spec: KVSpec configuration
				neurons_per_head: Neurons per attention head

			Seq2Seq/EncoderDecoder:
				output_bits: Bits per output token
				num_encoder_blocks: Encoder blocks
				num_decoder_blocks: Decoder blocks

		Returns:
			Configured model instance
		"""
		match model_type:
			case ModelType.TRANSFORMER:
				# Preset type takes priority
				if transformer_type is not None:
					from wnn.ram.factories.transformer import RAMTransformerFactory
					return RAMTransformerFactory.create(
						transformer_type=transformer_type,
						input_bits=input_bits,
						max_seq_len=max_seq_len,
						caesar_shift=caesar_shift,
						rng=rng,
					)

				# Multi-step pipeline
				if steps is not None:
					from wnn.ram.factories.transformer import RAMTransformerFactory
					return RAMTransformerFactory.create_multi_step(
						input_bits=input_bits,
						steps=steps,
						max_seq_len=max_seq_len,
						rng=rng,
					)

				# Custom configuration
				from wnn.ram.core.models.transformer import RAMTransformer
				return RAMTransformer(
					input_bits=input_bits,
					num_blocks=num_blocks,
					attention_type=attention_type,
					num_heads=num_heads,
					content_match=content_match,
					position_mode=position_mode,
					causal=causal,
					ffn_type=ffn_type,
					ffn_constant=ffn_constant,
					ffn_modulo=ffn_modulo,
					use_residual=use_residual,
					max_seq_len=max_seq_len,
					block_configs=block_configs,
					rng=rng,
				)

			case ModelType.RECURRENT:
				from wnn.ram.core.recurrent_network import RAMRecurrentNetwork
				return RAMRecurrentNetwork(
					input_bits=input_bits,
					n_state_neurons=n_state_neurons,
					n_output_neurons=n_output_neurons,
					n_bits_per_state_neuron=n_bits_per_state_neuron,
					n_bits_per_output_neuron=n_bits_per_output_neuron,
					use_hashing=use_hashing,
					hash_size=hash_size,
					rng=rng,
					max_iters=max_iters,
					output_mode=output_mode,
				)

			case ModelType.KV_MEMORY:
				if spec is None:
					raise ValueError("KV_MEMORY requires 'spec' parameter (KVSpec)")
				from wnn.ram.core.kv_transformer import RAMKVMemory
				return RAMKVMemory(
					spec=spec,
					neurons_per_head=neurons_per_head,
					n_bits_per_state_neuron=n_bits_per_state_neuron,
					n_bits_per_output_neuron=n_bits_per_output_neuron,
					use_hashing=use_hashing,
					hash_size=hash_size,
					rng=rng,
					max_iters=max_iters,
					output_mode=output_mode,
				)

			case ModelType.SEQ2SEQ:
				from wnn.ram.core.models.seq2seq import RAMSeq2Seq
				return RAMSeq2Seq(
					input_bits=input_bits,
					output_bits=output_bits,
					num_encoder_blocks=num_encoder_blocks,
					num_decoder_blocks=num_decoder_blocks,
					num_heads=num_heads,
					ffn_type=ffn_type,
					use_residual=use_residual,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case ModelType.ENCODER_DECODER:
				from wnn.ram.core.models.encoder_decoder import RAMEncoderDecoder
				return RAMEncoderDecoder(
					input_bits=input_bits,
					output_bits=output_bits,
					num_encoder_blocks=num_encoder_blocks,
					num_decoder_blocks=num_decoder_blocks,
					num_heads=num_heads,
					position_mode=position_mode,
					ffn_type=ffn_type,
					use_residual=use_residual,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case _:
				raise ValueError(f"Unknown model type: {model_type}")
