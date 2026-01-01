"""
Strategy classes for RAM neural networks.

Contains strategy pattern implementations for:
- Attention masking strategies
"""

from .attention_mask import (
    MaskStrategy,
    AttentionMaskStrategy,
    CausalMask,
    BidirectionalMask,
    SlidingWindowMask,
    BlockMask,
    PrefixMask,
    StridedMask,
    DilatedMask,
    LocalGlobalMask,
    CustomMask,
    MaskStrategyFactory,
    combine_masks,
)

__all__ = [
    'MaskStrategy',
    'AttentionMaskStrategy',
    'CausalMask',
    'BidirectionalMask',
    'SlidingWindowMask',
    'BlockMask',
    'PrefixMask',
    'StridedMask',
    'DilatedMask',
    'LocalGlobalMask',
    'CustomMask',
    'MaskStrategyFactory',
    'combine_masks',
]
