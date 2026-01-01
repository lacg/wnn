"""
RAM Transformer Components

Classes for building RAM-based transformer architectures.
"""

# Computed operations
from wnn.ram.core.transformers.computed_arithmetic import (
    ComputedArithmeticFFN,
    ComputedCopyFFN,
    bits_to_int,
    int_to_bits,
)

# Attention mechanisms
from wnn.ram.core.transformers.soft_ram_attention import SoftRAMAttention
from wnn.ram.core.transformers.sorting_attention import ComputedSortingAttention, SortingAttention
from wnn.ram.core.transformers.minmax_attention import ComputedMinMaxAttention, MinMaxAttention
from wnn.ram.core.transformers.attention import RAMAttention, RAMCrossAttention, CrossAttentionMode
from wnn.ram.core.transformers.attention_base import AttentionBase, LearnableAttention, ComputedAttention
from wnn.ram.core.transformers.position_attention import PositionOnlyAttention, PositionPattern
from wnn.ram.core.transformers.computed_attention import (
    ComputedMedianAttention,
    ComputedArgMaxAttention,
    ComputedCountDistinctAttention,
    ComputedSumAttention,
    ComputedMeanAttention,
    ComputedShiftAttention,
)
from wnn.ram.core.transformers.xor_attention import (
    XORCrossAttention,
    XORContentAddressableMemory,
    TopKAggregation,
)

# Attention masking
from wnn.ram.core.transformers.attention_mask import (
    AttentionMask,
    MaskStrategy,
    can_attend,
)

# FFN
from wnn.ram.core.transformers.two_layer_ffn import TwoLayerFFN
from wnn.ram.core.transformers.feedforward import RAMFeedForward, FFNMode

# Embeddings
from wnn.ram.core.transformers.embedding import RAMEmbedding, PositionEncoding

# Transformer blocks
from wnn.ram.core.transformers.transformer_block import RAMTransformerBlock
from wnn.ram.core.transformers.transformer import RAMTransformer

# Seq2Seq models
from wnn.ram.core.transformers.seq2seq import RAMSeq2Seq
from wnn.ram.core.transformers.encoder_decoder import RAMEncoderDecoder


__all__ = [
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
    'ComputedMinMaxAttention',
    'MinMaxAttention',  # Alias for ComputedMinMaxAttention
    'RAMAttention',
    'RAMCrossAttention',  # Alias for RAMAttention (cross-attention mode)
    'CrossAttentionMode',
    'PositionOnlyAttention',
    'PositionPattern',
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
    'FFNMode',
    # Embeddings
    'RAMEmbedding',
    'PositionEncoding',
    # Transformer
    'RAMTransformerBlock',
    'RAMTransformer',
    # Seq2Seq
    'RAMSeq2Seq',
    'RAMEncoderDecoder',
]
