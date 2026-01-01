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
    HashMapper,
    ResidualMapper,
)

# Aggregation
from wnn.ram.core.RAMAggregator import RAMAggregator

# Recurrent networks
from wnn.ram.core.recurrent_network import RAMRecurrentNetwork
from wnn.ram.enums import StateMode
from wnn.ram.core.sequence import RAMSequence
from wnn.ram.core.multihead_sequence import RAMMultiHeadSequence
from wnn.ram.core.multihead_kv import RAMMultiHeadKV
from wnn.ram.core.multihead_shared import RAMMultiHeadShared
from wnn.ram.core.kv_transformer import RAMKVMemory
from wnn.ram.core.automaton import RAMAutomaton
from wnn.ram.core.trainer import (
    RAMTrainer,
    TrainingStats,
    LayerState,
    EpochStats,
    TrainingCallback,
)
from wnn.ram.enums import LayerType, TrainingMode, TrainingPhase

# Transformer components (submodule)
from wnn.ram.core import transformers

# Re-export transformer components at core level for convenience
from wnn.ram.core.transformers import (
    # Computed operations
    ComputedArithmeticFFN,
    ComputedCopyFFN,
    bits_to_int,
    int_to_bits,
    # Attention base classes
    AttentionBase,
    LearnableAttention,
    ComputedAttention,
    # Attention mechanisms
    SoftRAMAttention,
    ComputedSortingAttention,
    SortingAttention,  # Alias
    ComputedMinMaxAttention,
    MinMaxAttention,  # Alias
    RAMAttention,
    RAMCrossAttention,  # Alias
    CrossAttentionMode,
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
    'HashMapper',
    'ResidualMapper',
    # Aggregation
    'RAMAggregator',
    # Recurrent networks
    'RAMRecurrentNetwork',
    'StateMode',
    'RAMSequence',
    'RAMMultiHeadSequence',
    'RAMMultiHeadKV',
    'RAMMultiHeadShared',
    'RAMKVMemory',
    'RAMAutomaton',
    'RAMTrainer',
    'TrainingStats',
    'LayerState',
    'EpochStats',
    'TrainingCallback',
    'LayerType',
    'TrainingMode',
    'TrainingPhase',
    # Submodules
    'transformers',
    # Transformer components (re-exported)
    'ComputedArithmeticFFN',
    'ComputedCopyFFN',
    'bits_to_int',
    'int_to_bits',
    'AttentionBase',
    'LearnableAttention',
    'ComputedAttention',
    'SoftRAMAttention',
    'ComputedSortingAttention',
    'SortingAttention',
    'ComputedMinMaxAttention',
    'MinMaxAttention',
    'RAMAttention',
    'RAMCrossAttention',
    'CrossAttentionMode',
    'TwoLayerFFN',
    'RAMFeedForward',
    'RAMEmbedding',
    'PositionEncoding',
    'RAMTransformerBlock',
    'RAMTransformer',
    'RAMSeq2Seq',
    'RAMEncoderDecoder',
]
