"""
RAM Transformer Components

Classes for building RAM-based transformer architectures.
"""

from enum import IntEnum, auto


# =============================================================================
# Model Enums (self-contained)
# =============================================================================

class CrossAttentionMode(IntEnum):
	"""How to encode positions in cross-attention."""
	NONE = 0          # No position info (content-only)
	ENCODER_ONLY = 1  # Only encode key/value positions (encoder side)
	BOTH = 2          # Encode both query (decoder) and key (encoder) positions


class AttentionType(IntEnum):
	"""Type of attention mechanism to use in RAMTransformerBlock."""
	SOFT_RAM = 0        # Standard SoftRAMAttention (configurable, learned)
	SORTING = 1         # SortingAttention (computed, 100% generalization)
	MIN_MAX = 2         # MinMaxAttention (computed, 100% generalization)
	POSITION_ONLY = 3   # Position-only attention (100% generalization)
	CONTENT_MATCH = 4   # Content-matching attention (XOR_EQUAL, etc.)


class ContentMatchMode(IntEnum):
	"""Content-based attention matching modes (computed operations)."""
	NONE = 0           # No content matching (use learned attention)
	XOR_EQUAL = 1      # Attend if query == key (XOR is all zeros)


class AttentionCombineMode(IntEnum):
	"""How to combine content matching with position patterns."""
	CONTENT_ONLY = 0    # Only use content matching (ignore position)
	POSITION_ONLY = 1   # Only use position patterns (ignore content)
	CONTENT_AND_POS = 2 # Attend if BOTH content AND position match


class AggregationStrategy(IntEnum):
	"""How to aggregate attention votes into final weights."""
	TOP_1 = 0       # Winner-take-all (best for retrieval)
	MAJORITY = 1    # Per-bit weighted voting (best for combining)
	TOP_K = 2       # XOR top K highest-voted values


class PositionPattern(IntEnum):
	"""Pre-defined position attention patterns."""
	COPY = 0        # Position i attends to position i
	SHIFT_LEFT = 1  # Position i attends to position i-1
	SHIFT_RIGHT = 2 # Position i attends to position i+1
	REVERSE = 3     # Position i attends to position n-1-i
	FIRST = 4       # All positions attend to position 0
	LAST = 5        # All positions attend to position n-1
	BROADCAST = 6   # All positions attend to all positions


class FFNMode(IntEnum):
	"""Feed-forward network mode."""
	STANDARD = 0     # Two RAMLayer projections
	GENERALIZED = 1  # Uses GeneralizingProjection for better generalization
	GATED = 2        # Gated variant: output = gate * up_proj


class FFNType(IntEnum):
	"""
	Type of feed-forward network in RAMTransformerBlock.

	Learned types may not generalize to unseen tokens.
	Computed types achieve 100% generalization with no training.
	"""
	# Learned FFN types (may not generalize to unseen tokens)
	NONE = 0            # No FFN (attention only)
	SINGLE = 1          # Single projection layer
	TWO_LAYER = 2       # Two-layer MLP (expand then contract)
	BIT_LEVEL = 3       # BIT_LEVEL generalization (partial)

	# Computed FFN types (100% generalization - no training needed)
	INCREMENT = 10      # Add 1 to value
	DECREMENT = 11      # Subtract 1 from value
	ADD_MOD = 12        # Add constant with modulo
	SUBTRACT_MOD = 13   # Subtract constant with modulo
	ROT13 = 14          # ROT13 cipher (add 13 mod 26)
	NEGATE = 15         # Bitwise complement (max - value)

	def is_computed(self) -> bool:
		"""Return True if this FFN type uses computed operations."""
		return self.value >= 10

	def is_learned(self) -> bool:
		"""Return True if this FFN type uses learned operations."""
		return self.value < 10 and self != FFNType.NONE


class ArithmeticOp(IntEnum):
	"""Arithmetic operations for ComputedArithmeticFFN."""
	INCREMENT = 0      # value + 1
	DECREMENT = 1      # value - 1
	ADD = 2            # value + constant
	SUBTRACT = 3       # value - constant
	ADD_MOD = 4        # (value + constant) mod N
	SUBTRACT_MOD = 5   # (value - constant) mod N
	ROT13 = 6          # (value + 13) mod 26
	NEGATE = 7         # max_value - value


class Step(IntEnum):
	"""
	Steps for multi-step transformer pipelines.

	Each step represents an operation that can be composed
	in a multi-block transformer.
	"""
	# Position-based operations (100% generalization)
	COPY = auto()       # Copy input to output (identity)
	SHIFT = auto()      # Shift right (causal)
	REVERSE = auto()    # Reverse sequence

	# Computed attention operations (100% generalization)
	SORT = auto()       # Sort by token value

	# Computed FFN operations (100% generalization)
	INCREMENT = auto()  # Add 1 to each token
	DECREMENT = auto()  # Subtract 1 from each token
	ROT13 = auto()      # Apply ROT13 cipher
	NEGATE = auto()     # Negate each token (max - value)


class RAMTransformerType(IntEnum):
	"""
	Pre-configured RAM Transformer architectures.

	Each type represents a specific task with optimal configuration.
	"""
	# Position-based transformers (100% generalization)
	COPY = auto()           # Copy task
	SHIFT = auto()          # Shift right
	REVERSE = auto()        # Reverse sequence

	# Computed attention transformers (100% generalization)
	SORTING = auto()        # Sort by value
	SELF_MATCHING = auto()  # Find matching tokens

	# Computed FFN transformers (100% generalization)
	INCREMENT = auto()      # Add 1 to each token
	DECREMENT = auto()      # Subtract 1 from each token
	ROT13 = auto()          # ROT13 cipher
	CAESAR = auto()         # Caesar cipher (configurable shift)
	NEGATE = auto()         # Negate each token


class PositionEncoding(IntEnum):
	"""How to encode position information in embeddings."""
	NONE = 0       # No position encoding
	BINARY = 1     # Binary representation of position
	LEARNED = 2    # Learned position embeddings
	SINUSOIDAL = 3 # Discrete approximation of sinusoidal


class ModelType(IntEnum):
	"""
	High-level model architecture types.

	These represent the major model families in the RAM architecture.
	For transformer subtypes, see RAMTransformerType.
	"""
	# Recurrent models
	RECURRENT = auto()          # Basic RAMRecurrentNetwork
	KV_MEMORY = auto()          # RAMKVMemory with head-based routing

	# Transformer models
	TRANSFORMER = auto()        # RAMTransformer (use with RAMTransformerType)

	# Sequence-to-sequence models
	SEQ2SEQ = auto()            # RAMSeq2Seq
	ENCODER_DECODER = auto()    # RAMEncoderDecoder


class NormStrategy(IntEnum):
	"""
	Discrete normalization strategies for RAM networks.

	Since RAM networks operate on boolean values, traditional layer
	normalization isn't applicable. These strategies provide discrete
	equivalents:

	- NONE: No normalization
	- ENSEMBLE_VOTE: Multiple sub-networks with majority voting
		(provides stability through redundancy)
	- BIT_BALANCE: Learn to transform toward ~50% ones
		(keeps information content high)
	"""
	NONE = 0
	ENSEMBLE_VOTE = 1  # Majority voting across sub-networks
	BIT_BALANCE = 2    # Learn toward 50% ones (max entropy)


# =============================================================================
# Component Imports (lazy to avoid circular imports)
# =============================================================================
# Components import core enums but also depend on core/RAMLayer which depends
# on factories which depend on these enums. To break the cycle, components
# are imported lazily via __getattr__.
#
# Usage:
#   from wnn.ram.core.models import AttentionType  # Works (enum)
#   from wnn.ram.core.models import SoftRAMAttention  # Works (lazy import)
#   from wnn.ram.core.models.soft_ram_attention import SoftRAMAttention  # Also works


def __getattr__(name: str):
	"""Lazy import for component classes to avoid circular imports."""
	# Attention mechanisms
	if name == 'SoftRAMAttention':
		from wnn.ram.core.models.soft_ram_attention import SoftRAMAttention
		return SoftRAMAttention
	if name in ('ComputedSortingAttention', 'SortingAttention'):
		from wnn.ram.core.models.sorting_attention import ComputedSortingAttention
		return ComputedSortingAttention
	if name == 'LearnedComparator':
		from wnn.ram.core.models.learned_sorting import LearnedComparator
		return LearnedComparator
	if name == 'LearnedSortingAttention':
		from wnn.ram.core.models.learned_sorting import LearnedSortingAttention
		return LearnedSortingAttention
	if name == 'BitLevelComparator':
		from wnn.ram.core.models.learned_sorting import BitLevelComparator
		return BitLevelComparator
	if name in ('ComputedMinMaxAttention', 'MinMaxAttention'):
		from wnn.ram.core.models.minmax_attention import ComputedMinMaxAttention
		return ComputedMinMaxAttention
	if name == 'RAMAttention':
		from wnn.ram.core.models.attention import RAMAttention
		return RAMAttention
	if name == 'RAMCrossAttention':
		from wnn.ram.core.models.attention import RAMCrossAttention
		return RAMCrossAttention
	if name == 'AttentionBase':
		from wnn.ram.core.models.attention_base import AttentionBase
		return AttentionBase
	if name == 'LearnableAttention':
		from wnn.ram.core.models.attention_base import LearnableAttention
		return LearnableAttention
	if name == 'ComputedAttention':
		from wnn.ram.core.models.attention_base import ComputedAttention
		return ComputedAttention
	if name == 'PositionOnlyAttention':
		from wnn.ram.core.models.position_attention import PositionOnlyAttention
		return PositionOnlyAttention

	# Computed attention operations
	if name == 'ComputedMedianAttention':
		from wnn.ram.core.models.computed_attention import ComputedMedianAttention
		return ComputedMedianAttention
	if name == 'ComputedArgMaxAttention':
		from wnn.ram.core.models.computed_attention import ComputedArgMaxAttention
		return ComputedArgMaxAttention
	if name == 'ComputedCountDistinctAttention':
		from wnn.ram.core.models.computed_attention import ComputedCountDistinctAttention
		return ComputedCountDistinctAttention
	if name == 'ComputedSumAttention':
		from wnn.ram.core.models.computed_attention import ComputedSumAttention
		return ComputedSumAttention
	if name == 'ComputedMeanAttention':
		from wnn.ram.core.models.computed_attention import ComputedMeanAttention
		return ComputedMeanAttention
	if name == 'ComputedShiftAttention':
		from wnn.ram.core.models.computed_attention import ComputedShiftAttention
		return ComputedShiftAttention

	# XOR attention
	if name == 'XORCrossAttention':
		from wnn.ram.core.models.xor_attention import XORCrossAttention
		return XORCrossAttention
	if name == 'XORContentAddressableMemory':
		from wnn.ram.core.models.xor_attention import XORContentAddressableMemory
		return XORContentAddressableMemory
	if name == 'TopKAggregation':
		from wnn.ram.core.models.xor_attention import TopKAggregation
		return TopKAggregation

	# Attention masking
	if name == 'AttentionMask':
		from wnn.ram.core.models.attention_mask import AttentionMask
		return AttentionMask
	if name == 'MaskStrategy':
		from wnn.ram.core.models.attention_mask import MaskStrategy
		return MaskStrategy
	if name == 'can_attend':
		from wnn.ram.core.models.attention_mask import can_attend
		return can_attend

	# Computed operations
	if name == 'ComputedArithmeticFFN':
		from wnn.ram.core.models.computed_arithmetic import ComputedArithmeticFFN
		return ComputedArithmeticFFN
	if name == 'ComputedCopyFFN':
		from wnn.ram.core.models.computed_arithmetic import ComputedCopyFFN
		return ComputedCopyFFN
	if name == 'bits_to_int':
		from wnn.ram.core.models.computed_arithmetic import bits_to_int
		return bits_to_int
	if name == 'int_to_bits':
		from wnn.ram.core.models.computed_arithmetic import int_to_bits
		return int_to_bits

	# FFN
	if name == 'TwoLayerFFN':
		from wnn.ram.core.models.two_layer_ffn import TwoLayerFFN
		return TwoLayerFFN
	if name == 'RAMFeedForward':
		from wnn.ram.core.models.feedforward import RAMFeedForward
		return RAMFeedForward

	# Embeddings
	if name == 'RAMEmbedding':
		from wnn.ram.core.models.embedding import RAMEmbedding
		return RAMEmbedding

	# Transformer blocks
	if name == 'RAMTransformerBlock':
		from wnn.ram.core.models.transformer_block import RAMTransformerBlock
		return RAMTransformerBlock
	if name == 'RAMTransformer':
		from wnn.ram.core.models.transformer import RAMTransformer
		return RAMTransformer

	# Normalization
	if name == 'DiscreteNormalization':
		from wnn.ram.core.models.discrete_normalization import DiscreteNormalization
		return DiscreteNormalization

	# Seq2Seq models
	if name == 'RAMSeq2Seq':
		from wnn.ram.core.models.seq2seq import RAMSeq2Seq
		return RAMSeq2Seq
	if name == 'RAMEncoderDecoder':
		from wnn.ram.core.models.encoder_decoder import RAMEncoderDecoder
		return RAMEncoderDecoder

	# Factories (self-contained in models)
	if name == 'FFNFactory':
		from wnn.ram.core.models.ffn_factory import FFNFactory
		return FFNFactory
	if name == 'AttentionFactory':
		from wnn.ram.core.models.attention_factory import AttentionFactory
		return AttentionFactory
	if name == 'StepConfigurationFactory':
		from wnn.ram.core.models.step_config_factory import StepConfigurationFactory
		return StepConfigurationFactory
	if name == 'RAMTransformerFactory':
		from wnn.ram.core.models.transformer_factory import RAMTransformerFactory
		return RAMTransformerFactory
	if name == 'ModelsFactory':
		from wnn.ram.core.models.models_factory import ModelsFactory
		return ModelsFactory

	# Language Models
	if name == 'RAMLM':
		from wnn.ram.core.models.ramlm import RAMLM
		return RAMLM

	raise AttributeError(f"module 'wnn.ram.core.models' has no attribute '{name}'")


__all__ = [
	# ==== Enums (self-contained, available immediately) ====
	'CrossAttentionMode',
	'AttentionType',
	'ContentMatchMode',
	'AttentionCombineMode',
	'AggregationStrategy',
	'PositionPattern',
	'FFNMode',
	'FFNType',
	'ArithmeticOp',
	'Step',
	'RAMTransformerType',
	'PositionEncoding',
	'ModelType',
	'NormStrategy',
	# ==== Components (lazy-loaded to avoid circular imports) ====
	# Computed operations
	'ComputedArithmeticFFN',
	'ComputedCopyFFN',
	'bits_to_int',
	'int_to_bits',
	# Attention base classes
	'AttentionBase',
	'LearnableAttention',
	'ComputedAttention',
	# Attention mechanisms
	'SoftRAMAttention',
	'ComputedSortingAttention',
	'SortingAttention',  # Alias for ComputedSortingAttention
	'LearnedComparator',
	'LearnedSortingAttention',
	'BitLevelComparator',
	'ComputedMinMaxAttention',
	'MinMaxAttention',  # Alias for ComputedMinMaxAttention
	'RAMAttention',
	'RAMCrossAttention',  # Alias for RAMAttention (cross-attention mode)
	'PositionOnlyAttention',
	'ComputedMedianAttention',
	'ComputedArgMaxAttention',
	'ComputedCountDistinctAttention',
	'ComputedSumAttention',
	'ComputedMeanAttention',
	'ComputedShiftAttention',
	# XOR attention
	'XORCrossAttention',
	'XORContentAddressableMemory',
	'TopKAggregation',
	# Attention masking
	'AttentionMask',
	'MaskStrategy',
	'can_attend',
	# FFN
	'TwoLayerFFN',
	'RAMFeedForward',
	# Embeddings
	'RAMEmbedding',
	# Transformer
	'RAMTransformerBlock',
	'RAMTransformer',
	# Normalization
	'DiscreteNormalization',
	# Seq2Seq
	'RAMSeq2Seq',
	'RAMEncoderDecoder',
	# Factories (lazy-loaded)
	'FFNFactory',
	'AttentionFactory',
	'StepConfigurationFactory',
	'RAMTransformerFactory',
	'ModelsFactory',
	# Language Models
	'RAMLM',
]
