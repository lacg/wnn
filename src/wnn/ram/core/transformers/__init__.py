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
from wnn.ram.core.transformers.sorting_attention import SortingAttention
from wnn.ram.core.transformers.minmax_attention import MinMaxAttention
from wnn.ram.core.transformers.attention import RAMAttention
from wnn.ram.core.transformers.cross_attention import RAMCrossAttention

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
    # Attention
    'SoftRAMAttention',
    'SortingAttention',
    'MinMaxAttention',
    'RAMAttention',
    'RAMCrossAttention',
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
