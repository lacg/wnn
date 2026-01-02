"""
Attention Mask Strategy Classes

Strategy pattern implementation for attention masking.
Each strategy encapsulates its mask generation logic and parameters.

Usage:
    strategy = MaskStrategyFactory.create(MaskStrategy.CAUSAL)
    mask = strategy.create(seq_len=10)

    # Or with parameters
    strategy = MaskStrategyFactory.create(
        MaskStrategy.SLIDING_WINDOW,
        window_size=3,
        causal=True,
    )
    mask = strategy.create(seq_len=10)
"""

from abc import ABC, abstractmethod
from typing import Callable
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
    LEARNED = auto()          # RAM-learned position-pair patterns


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
    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
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

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
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

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
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

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
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

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
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

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
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

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
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

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
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

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
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

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
        key_len = key_len or seq_len
        mask = zeros(seq_len, key_len, dtype=tbool)
        for i in range(seq_len):
            for j in range(key_len):
                if self.can_attend_fn(i, j):
                    mask[i, j] = True
        return mask


class LearnedSparseMask(AttentionMaskStrategy):
    """
    Learned sparse attention: RAMLayer learns which position pairs should attend.

    Uses a RAMLayer to learn (query_pos, key_pos) → should_attend mapping.
    Enables automatic pattern discovery without hand-designed sparsity.

    Usage:
        mask_strategy = LearnedSparseMask(max_seq_len=32, position_bits=5)

        # Train on desired patterns
        mask_strategy.train_pattern(query_pos=5, key_pos=3, should_attend=True)
        mask_strategy.train_pattern(query_pos=5, key_pos=10, should_attend=False)

        # Or train from an existing mask
        causal_mask = CausalMask().create(seq_len=16)
        mask_strategy.train_from_mask(causal_mask)

        # Use in attention
        mask = mask_strategy.create(seq_len=16)
    """

    def __init__(
        self,
        max_seq_len: int = 32,
        position_bits: int | None = None,
        causal_init: bool = False,
        rng: int | None = None,
    ):
        """
        Args:
            max_seq_len: Maximum sequence length to support
            position_bits: Bits per position (default: ceil(log2(max_seq_len)))
            causal_init: Initialize with causal pattern (vs empty)
            rng: Random seed for RAMLayer
        """
        from wnn.ram.core import RAMLayer
        import math

        self.max_seq_len = max_seq_len
        self.position_bits = position_bits or max(1, math.ceil(math.log2(max_seq_len + 1)))

        # Input: [query_pos_bits, key_pos_bits]
        total_input_bits = self.position_bits * 2

        # RAMLayer learns: (query_pos, key_pos) → attend (1 bit)
        self.pattern_layer = RAMLayer(
            total_input_bits=total_input_bits,
            num_neurons=1,  # Single output: attend or not
            n_bits_per_neuron=min(total_input_bits, 12),
            rng=rng,
        )

        # Optionally initialize with causal pattern
        if causal_init:
            self._init_causal()

    def _init_causal(self) -> None:
        """Initialize with causal attention pattern."""
        for i in range(self.max_seq_len):
            for j in range(i + 1):  # j <= i for causal
                self.train_pattern(i, j, should_attend=True)

    def _encode_positions(self, query_pos: int, key_pos: int) -> Tensor:
        """Encode position pair as binary tensor."""
        from torch import tensor, uint8
        # Binary encode query position
        q_bits = [(query_pos >> b) & 1 for b in range(self.position_bits)]
        # Binary encode key position
        k_bits = [(key_pos >> b) & 1 for b in range(self.position_bits)]
        # Concatenate
        bits = q_bits + k_bits
        return tensor(bits, dtype=uint8)

    @property
    def strategy_type(self) -> MaskStrategy:
        return MaskStrategy.LEARNED

    def train_pattern(
        self,
        query_pos: int,
        key_pos: int,
        should_attend: bool,
    ) -> None:
        """
        Train a single position pair.

        Args:
            query_pos: Query position index
            key_pos: Key position index
            should_attend: Whether query should attend to key
        """
        import torch
        if query_pos >= self.max_seq_len or key_pos >= self.max_seq_len:
            raise ValueError(f"Position exceeds max_seq_len={self.max_seq_len}")

        inputs = self._encode_positions(query_pos, key_pos).unsqueeze(0)
        target = torch.tensor([[1 if should_attend else 0]], dtype=torch.uint8)
        self.pattern_layer.commit(inputs, target)

    def train_from_mask(self, mask: Tensor) -> int:
        """
        Train from an existing attention mask.

        Args:
            mask: Boolean mask [seq_len, key_len]

        Returns:
            Number of patterns trained
        """
        seq_len, key_len = mask.shape
        count = 0
        for i in range(min(seq_len, self.max_seq_len)):
            for j in range(min(key_len, self.max_seq_len)):
                should_attend = bool(mask[i, j].item())
                self.train_pattern(i, j, should_attend)
                count += 1
        return count

    def create(self, seq_len: int, key_len: int | None = None) -> Tensor:
        """
        Create attention mask using learned patterns.

        Args:
            seq_len: Query sequence length
            key_len: Key sequence length (default: same as seq_len)

        Returns:
            Boolean mask [seq_len, key_len]
        """
        import torch
        key_len = key_len or seq_len
        mask = zeros(seq_len, key_len, dtype=tbool)

        for i in range(seq_len):
            for j in range(key_len):
                if i < self.max_seq_len and j < self.max_seq_len:
                    inputs = self._encode_positions(i, j).unsqueeze(0)
                    output = self.pattern_layer(inputs)
                    mask[i, j] = output[0, 0].item() == 1
                # Positions beyond max_seq_len default to False

        return mask

    def get_sparsity(self, seq_len: int) -> float:
        """Get sparsity ratio (fraction of True values) for a sequence length."""
        mask = self.create(seq_len)
        return mask.sum().item() / mask.numel()


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
        max_seq_len: int = 32,
        position_bits: int | None = None,
        causal_init: bool = False,
        rng: int | None = None,
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
            max_seq_len: For LEARNED strategy
            position_bits: For LEARNED strategy
            causal_init: For LEARNED strategy (initialize with causal pattern)
            rng: Random seed for LEARNED strategy

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
            case MaskStrategy.LEARNED:
                return LearnedSparseMask(max_seq_len, position_bits, causal_init, rng)
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
