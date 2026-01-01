"""
RAM Core Components

Fundamental building blocks for RAM neural networks.
"""

# Base classes for RAM components
from wnn.ram.core.base import RAMComponent, RAMSequenceModel, RAMTrainable

# Fundamental storage and layer
from wnn.ram.core.Memory import Memory
from wnn.ram.core.SparseMemory import SparseMemory
from wnn.ram.core.RAMLayer import RAMLayer

# Generalization components
from wnn.ram.core.RAMGeneralization import (
    BitLevelMapper,
    CompositionalMapper,
    GeneralizingProjection,
    HashMapper,
    ResidualMapper,
    RecurrentParityMapper,
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
    Checkpoint,
    # Enhanced curriculum learning
    CurriculumTrainer,
    CurriculumSchedule,
    length_difficulty,
    bit_count_difficulty,
    hamming_difficulty,
    combined_difficulty,
    # Multi-task learning
    MultiTaskTrainer,
    Task,
    MixingStrategy,
)
from wnn.ram.enums import LayerType, TrainingMode, TrainingPhase

# Serialization (use model.save() / Model.load() instead)
from wnn.ram.core.serialization import SERIALIZATION_VERSION

# Batch processing utilities
from wnn.ram.core.batch import (
    BatchProcessor,
    BatchResult,
    pad_sequences,
    collate_sequences,
    uncollate_batch,
)

# Generation result types and streaming functions
from wnn.ram.core.generation import (
    GenerationResult,
    BeamCandidate,
    StreamToken,
    stream_greedy_decode,
    stream_sample_decode,
    stream_top_k_decode,
    collect_stream,
)

# Sequence generator wrapper
from wnn.ram.core.sequence_generator import SequenceGenerator

# Transformer components (submodule)
from wnn.ram.core import models

# Re-export transformer components at core level for convenience
from wnn.ram.core.models import (
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
    PositionOnlyAttention,
    PositionPattern,
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
    # Base classes
    'RAMComponent',
    'RAMSequenceModel',
    'RAMTrainable',
    # Fundamental
    'Memory',
    'SparseMemory',
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
    'Checkpoint',
    'LayerType',
    'TrainingMode',
    'TrainingPhase',
    # Enhanced curriculum learning
    'CurriculumTrainer',
    'CurriculumSchedule',
    'length_difficulty',
    'bit_count_difficulty',
    'hamming_difficulty',
    'combined_difficulty',
    # Multi-task learning
    'MultiTaskTrainer',
    'Task',
    'MixingStrategy',
    # Serialization (use model.save() / Model.load())
    'SERIALIZATION_VERSION',
    # Batch processing utilities
    'BatchProcessor',
    'BatchResult',
    'pad_sequences',
    'collate_sequences',
    'uncollate_batch',
    # Generation types and wrapper
    'GenerationResult',
    'BeamCandidate',
    'SequenceGenerator',
    # Submodules
    'models',
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
    'PositionOnlyAttention',
    'PositionPattern',
    'TwoLayerFFN',
    'RAMFeedForward',
    'RAMEmbedding',
    'PositionEncoding',
    'RAMTransformerBlock',
    'RAMTransformer',
    'RAMSeq2Seq',
    'RAMEncoderDecoder',
]
