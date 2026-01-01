"""
RAM Transformer Enums

All enumeration types for the RAM architecture, consolidated from across the codebase.
"""

# Memory
from wnn.ram.enums.memory import MemoryVal

# Decoder
from wnn.ram.enums.decoder import OutputMode, PositionMode

# Cost
from wnn.ram.enums.cost import CostCalculatorType

# Generalization
from wnn.ram.enums.generalization import (
    ContextMode,
    BitMapperMode,
    MapperStrategy,
)

# Embedding
from wnn.ram.enums.embedding import PositionEncoding

# Trainer
from wnn.ram.enums.trainer import LayerType, TrainingMode, TrainingPhase, MixingStrategy

# Attention
from wnn.ram.enums.attention import (
    CrossAttentionMode,
    AttentionType,
    ContentMatchMode,
    AttentionCombineMode,
    AggregationStrategy,
    PositionPattern,
)

# FFN
from wnn.ram.enums.ffn import (
    FFNMode,
    FFNType,
    ArithmeticOp,
)

# Transformer
from wnn.ram.enums.transformer import (
    Step,
    RAMTransformerType,
)

# Model types
from wnn.ram.enums.model import ModelType

# Recurrent
from wnn.ram.enums.recurrent import StateMode

# Normalization
from wnn.ram.enums.normalization import NormStrategy


__all__ = [
    # Memory
    'MemoryVal',
    # Decoder
    'OutputMode',
    'PositionMode',
    # Cost
    'CostCalculatorType',
    # Generalization
    'ContextMode',
    'BitMapperMode',
    'MapperStrategy',
    # Embedding
    'PositionEncoding',
    # Trainer
    'LayerType',
    'TrainingMode',
    'TrainingPhase',
    'MixingStrategy',
    # Attention
    'CrossAttentionMode',
    'AttentionType',
    'ContentMatchMode',
    'AttentionCombineMode',
    'AggregationStrategy',
    'PositionPattern',
    # FFN
    'FFNMode',
    'FFNType',
    'ArithmeticOp',
    # Transformer
    'Step',
    'RAMTransformerType',
    # Model types
    'ModelType',
    # Recurrent
    'StateMode',
    # Normalization
    'NormStrategy',
]
