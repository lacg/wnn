"""
Attention Mask Strategy Classes

Strategy pattern implementation for attention masking.
Each strategy encapsulates its mask generation logic and parameters.

Usage:
    strategy = MaskStrategyFactory.create(MaskStrategy.CAUSAL)
    mask = strategy.create_mask(seq_len=10)

    # Or with parameters
    strategy = MaskStrategyFactory.create(
        MaskStrategy.SLIDING_WINDOW,
        window_size=3,
        causal=True,
    )
    mask = strategy.create_mask(seq_len=10)
"""

from abc import ABC, abstractmethod
from typing import Callable, Protocol
from enum import Enum, auto
from torch import Tensor, zeros, ones, tril, bool as tbool


class MaskStrategy(Enum):
    """Predefined attention masking strategies."""
    CAUSAL = auto()           # j <= i
    BIDIRECTIONAL = auto()    # All positions
    SLIDING_WINDOW = auto()   # |i - j| <= window_size
    BLOCK = auto()            # Same block only
    PREFIX = auto()           # Prefix bidirectional, rest causal
    CUSTOM = auto()           # User-defined
    # Sparse patterns
    STRIDED = auto()          # Every k-th position
    DILATED = auto()          # Exponentially increasing gaps
    LOCAL_GLOBAL = auto()     # Local window + global tokens


class AttentionMaskStrategy(ABC):
    """
    Abstract base class for attention mask strategies.

    Each strategy defines how to generate an attention mask
    for a given sequence length.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> MaskStrategy:
        """Return the strategy type enum."""
        pass

    @abstractmethod
    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        """
        Create attention mask for given sequence length.

        Args:
            seq_len: Query sequence length
            key_len: Key sequence length (default: same as seq_len)

        Returns:
            Boolean mask [seq_len, key_len] where True = can attend
        """
        pass

    def can_attend(self, mask: Tensor, query_pos: int, key_pos: int) -> bool:
        """Check if query position can attend to key position."""
        return bool(mask[query_pos, key_pos].item())


class CausalMask(AttentionMaskStrategy):
    """Causal (autoregressive) attention mask: position i attends to j <= i."""

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.CAUSAL

    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        if key_len != seq_len:
            # Cross-attention: full mask (causal doesn't apply)
            return ones(seq_len, key_len, dtype=tbool)
        return tril(ones(seq_len, seq_len, dtype=tbool))


class BidirectionalMask(AttentionMaskStrategy):
    """Bidirectional attention mask: all positions can attend to all."""

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.BIDIRECTIONAL

    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        return ones(seq_len, key_len, dtype=tbool)


class SlidingWindowMask(AttentionMaskStrategy):
    """Sliding window attention: attend within fixed window around each position."""

    def __init__(self, window_size: int = 3, causal: bool = False):
        self.window_size = window_size
        self.causal = causal

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.SLIDING_WINDOW

    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        mask = zeros(seq_len, key_len, dtype=tbool)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = i + 1 if self.causal else min(key_len, i + self.window_size + 1)
            mask[i, start:end] = True
        return mask


class BlockMask(AttentionMaskStrategy):
    """Block-diagonal attention: positions attend within their block only."""

    def __init__(self, block_size: int = 4):
        self.block_size = block_size

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.BLOCK

    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        mask = zeros(seq_len, key_len, dtype=tbool)
        for i in range(seq_len):
            block_start = (i // self.block_size) * self.block_size
            block_end = min(block_start + self.block_size, key_len)
            mask[i, block_start:block_end] = True
        return mask


class PrefixMask(AttentionMaskStrategy):
    """Prefix attention: first k positions bidirectional, rest causal."""

    def __init__(self, prefix_len: int = 2):
        self.prefix_len = prefix_len

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.PREFIX

    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        mask = zeros(seq_len, key_len, dtype=tbool)
        for i in range(seq_len):
            if i < self.prefix_len:
                # Prefix: bidirectional
                mask[i, :] = True
            else:
                # Rest: causal, but can always see prefix
                mask[i, :self.prefix_len] = True
                mask[i, self.prefix_len:i+1] = True
        return mask


class StridedMask(AttentionMaskStrategy):
    """Strided attention: attend every k-th position."""

    def __init__(self, stride: int = 2, causal: bool = False):
        self.stride = stride
        self.causal = causal

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.STRIDED

    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        mask = zeros(seq_len, key_len, dtype=tbool)
        for i in range(seq_len):
            for j in range(0, key_len, self.stride):
                if not self.causal or j <= i:
                    mask[i, j] = True
            # Always attend to self
            if i < key_len:
                mask[i, i] = True
        return mask


class DilatedMask(AttentionMaskStrategy):
    """Dilated attention: attend at exponentially increasing distances."""

    def __init__(
        self,
        dilation_rates: list[int] | None = None,
        window_per_rate: int = 2,
        causal: bool = False,
    ):
        self.dilation_rates = dilation_rates or [1, 2, 4]
        self.window_per_rate = window_per_rate
        self.causal = causal

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.DILATED

    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        mask = zeros(seq_len, key_len, dtype=tbool)
        for i in range(seq_len):
            # Always attend to self
            if i < key_len:
                mask[i, i] = True

            for rate in self.dilation_rates:
                for step in range(1, self.window_per_rate + 1):
                    dist = rate * step
                    # Look backward
                    if i - dist >= 0:
                        mask[i, i - dist] = True
                    # Look forward (if not causal)
                    if not self.causal and i + dist < key_len:
                        mask[i, i + dist] = True
        return mask


class LocalGlobalMask(AttentionMaskStrategy):
    """Local-global attention: local window + global positions visible to all."""

    def __init__(
        self,
        local_window: int = 3,
        global_positions: list[int] | None = None,
        causal: bool = False,
    ):
        self.local_window = local_window
        self.global_positions = global_positions or [0]
        self.causal = causal

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.LOCAL_GLOBAL

    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        mask = zeros(seq_len, key_len, dtype=tbool)
        for i in range(seq_len):
            # Local window
            start = max(0, i - self.local_window)
            end = i + 1 if self.causal else min(key_len, i + self.local_window + 1)
            mask[i, start:end] = True

            # Global positions
            for g in self.global_positions:
                if 0 <= g < key_len:
                    mask[i, g] = True
                if 0 <= g < seq_len:
                    mask[g, i] = True
        return mask


class CustomMask(AttentionMaskStrategy):
    """Custom attention mask from user-provided function."""

    def __init__(self, can_attend_fn: Callable[[int, int], bool]):
        self.can_attend_fn = can_attend_fn

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.CUSTOM

    def create_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        mask = zeros(seq_len, key_len, dtype=tbool)
        for i in range(seq_len):
            for j in range(key_len):
                if self.can_attend_fn(i, j):
                    mask[i, j] = True
        return mask


class MaskStrategyFactory:
    """Factory for creating attention mask strategies."""

    @staticmethod
    def create(
        strategy: MaskStrategy,
        window_size: int = 3,
        block_size: int = 4,
        prefix_len: int = 2,
        stride: int = 2,
        dilation_rates: list[int] | None = None,
        window_per_rate: int = 2,
        global_positions: list[int] | None = None,
        causal: bool = False,
        can_attend_fn: Callable[[int, int], bool] | None = None,
    ) -> AttentionMaskStrategy:
        """
        Create an attention mask strategy.

        Args:
            strategy: The mask strategy type
            window_size: For SLIDING_WINDOW and LOCAL_GLOBAL
            block_size: For BLOCK
            prefix_len: For PREFIX
            stride: For STRIDED
            dilation_rates: For DILATED
            window_per_rate: For DILATED
            global_positions: For LOCAL_GLOBAL
            causal: For strategies that support causal variant
            can_attend_fn: For CUSTOM strategy

        Returns:
            Configured AttentionMaskStrategy instance
        """
        match strategy:
            case MaskStrategy.CAUSAL:
                return CausalMask()
            case MaskStrategy.BIDIRECTIONAL:
                return BidirectionalMask()
            case MaskStrategy.SLIDING_WINDOW:
                return SlidingWindowMask(window_size, causal)
            case MaskStrategy.BLOCK:
                return BlockMask(block_size)
            case MaskStrategy.PREFIX:
                return PrefixMask(prefix_len)
            case MaskStrategy.STRIDED:
                return StridedMask(stride, causal)
            case MaskStrategy.DILATED:
                return DilatedMask(dilation_rates, window_per_rate, causal)
            case MaskStrategy.LOCAL_GLOBAL:
                return LocalGlobalMask(window_size, global_positions, causal)
            case MaskStrategy.CUSTOM:
                if can_attend_fn is None:
                    raise ValueError("CUSTOM strategy requires can_attend_fn")
                return CustomMask(can_attend_fn)
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")


# Utility functions
def combine_masks(masks: list[Tensor], mode: str = "and") -> Tensor:
    """
    Combine multiple masks.

    Args:
        masks: List of boolean masks (same shape)
        mode: "and" (intersection) or "or" (union)

    Returns:
        Combined mask
    """
    if not masks:
        raise ValueError("Need at least one mask")

    result = masks[0].clone()
    for m in masks[1:]:
        if mode == "and":
            result = result & m
        elif mode == "or":
            result = result | m
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return result
