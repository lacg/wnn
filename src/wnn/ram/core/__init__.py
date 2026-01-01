"""
RAM Core Components

Fundamental building blocks for RAM neural networks.
"""

# Fundamental storage and layer
from wnn.ram.core.Memory import Memory
from wnn.ram.core.RAMLayer import RAMLayer

# Generalization components
from wnn.ram.core.RAMGeneralization import (
    BitLevelMapper,
    CompositionalMapper,
    GeneralizingProjection,
)

# Aggregation
from wnn.ram.core.RAMAggregator import RAMAggregator

# Recurrent networks
from wnn.ram.core.recurrent_network import RAMRecurrentNetwork
from wnn.ram.core.sequence import RAMSequence
from wnn.ram.core.multihead_sequence import RAMMultiHeadSequence
from wnn.ram.core.multihead_kv import RAMMultiHeadKV
from wnn.ram.core.multihead_shared import RAMMultiHeadShared
from wnn.ram.core.kv_transformer import RAMTransformer as RAMKVTransformer
from wnn.ram.core.automaton import RAMAutomaton
from wnn.ram.core.trainer import RAMTrainer, TrainingStats, LayerState, LayerType

# Transformer components (submodule)
from wnn.ram.core import transformers

# Re-export transformer components at core level for convenience
from wnn.ram.core.transformers import (
    # Computed operations
    ComputedArithmeticFFN,
    ComputedCopyFFN,
    bits_to_int,
    int_to_bits,
    # Attention mechanisms
    SoftRAMAttention,
    SortingAttention,
    MinMaxAttention,
    RAMAttention,
    RAMCrossAttention,
    # FFN
    TwoLayerFFN,
    RAMFeedForward,
    # Embeddings
    RAMEmbedding,
    PositionEncoding,
    # Transformer blocks
    RAMTransformerBlock,
    RAMTransformer,
    # Seq2Seq
    RAMSeq2Seq,
    RAMEncoderDecoder,
)


__all__ = [
    # Fundamental
    'Memory',
    'RAMLayer',
    # Generalization
    'BitLevelMapper',
    'CompositionalMapper',
    'GeneralizingProjection',
    # Aggregation
    'RAMAggregator',
    # Recurrent networks
    'RAMRecurrentNetwork',
    'RAMSequence',
    'RAMMultiHeadSequence',
    'RAMMultiHeadKV',
    'RAMMultiHeadShared',
    'RAMKVTransformer',
    'RAMAutomaton',
    'RAMTrainer',
    'TrainingStats',
    'LayerState',
    'LayerType',
    # Submodules
    'transformers',
    # Transformer components (re-exported)
    'ComputedArithmeticFFN',
    'ComputedCopyFFN',
    'bits_to_int',
    'int_to_bits',
    'SoftRAMAttention',
    'SortingAttention',
    'MinMaxAttention',
    'RAMAttention',
    'RAMCrossAttention',
    'TwoLayerFFN',
    'RAMFeedForward',
    'RAMEmbedding',
    'PositionEncoding',
    'RAMTransformerBlock',
    'RAMTransformer',
    'RAMSeq2Seq',
    'RAMEncoderDecoder',
]
