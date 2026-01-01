"""
Attention Factory

Factory for creating attention layers.
Uses match-case for clean dispatch.
"""

from torch.nn import Module

from wnn.ram.enums import (
    AttentionType,
    ContentMatchMode,
    AttentionCombineMode,
    AggregationStrategy,
)
from wnn.ram.encoders_decoders import PositionMode
from wnn.ram.enums import MapperStrategy


class AttentionFactory:
    """
    Factory for creating attention layers.

    Supports both learned and computed attention types.
    Computed types achieve 100% generalization.
    """

    @staticmethod
    def create(
        attention_type: AttentionType,
        input_bits: int,
        num_heads: int = 8,
        content_match: ContentMatchMode = ContentMatchMode.NONE,
        attention_combine: AttentionCombineMode = AttentionCombineMode.CONTENT_ONLY,
        position_mode: PositionMode = PositionMode.RELATIVE,
        causal: bool = True,
        max_seq_len: int = 16,
        rng: int | None = None,
    ) -> Module:
        """
        Create an attention layer based on type.

        Args:
            attention_type: Type of attention to create
            input_bits: Bits per token
            num_heads: Number of attention heads
            content_match: Content matching mode
            attention_combine: How to combine content and position
            position_mode: Position encoding mode
            causal: Use causal attention mask
            max_seq_len: Maximum sequence length
            rng: Random seed

        Returns:
            Attention module
        """
        # Lazy imports to avoid circular dependencies
        from wnn.ram.core.models.soft_ram_attention import SoftRAMAttention
        from wnn.ram.core.models.sorting_attention import SortingAttention
        from wnn.ram.core.models.minmax_attention import MinMaxAttention

        match attention_type:
            case AttentionType.SORTING:
                return SortingAttention(
                    input_bits=input_bits,
                    descending=False,
                    rng=rng,
                )

            case AttentionType.MIN_MAX:
                return MinMaxAttention(
                    input_bits=input_bits,
                    find_max=False,
                    rng=rng,
                )

            case AttentionType.POSITION_ONLY:
                return SoftRAMAttention(
                    input_bits=input_bits,
                    num_heads=num_heads,
                    aggregation=AggregationStrategy.TOP_1,
                    value_strategy=MapperStrategy.BIT_LEVEL,
                    position_mode=position_mode,
                    max_seq_len=max_seq_len,
                    causal=causal,
                    position_only=True,
                    rng=rng,
                )

            case AttentionType.CONTENT_MATCH:
                return SoftRAMAttention(
                    input_bits=input_bits,
                    num_heads=num_heads,
                    aggregation=AggregationStrategy.TOP_1,
                    value_strategy=MapperStrategy.BIT_LEVEL,
                    position_mode=position_mode,
                    max_seq_len=max_seq_len,
                    causal=causal,
                    position_only=False,
                    content_match=content_match,
                    attention_combine=attention_combine,
                    rng=rng,
                )

            case AttentionType.SOFT_RAM:
                return SoftRAMAttention(
                    input_bits=input_bits,
                    num_heads=num_heads,
                    aggregation=AggregationStrategy.TOP_1,
                    value_strategy=MapperStrategy.BIT_LEVEL,
                    position_mode=position_mode,
                    max_seq_len=max_seq_len,
                    causal=causal,
                    position_only=False,
                    content_match=content_match,
                    attention_combine=attention_combine,
                    rng=rng,
                )

            case _:
                raise ValueError(f"Unknown attention type: {attention_type}")
