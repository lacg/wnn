"""
Attention Masking Strategies

Provides flexible attention mask generation for different attention patterns:
- CAUSAL: Position i can only attend to positions j <= i
- BIDIRECTIONAL: All positions can attend to all positions
- SLIDING_WINDOW: Attend within a fixed window around each position
- BLOCK: Attention within fixed-size blocks only
- PREFIX: First k positions can attend bidirectionally, rest are causal
- CUSTOM: User-provided mask function
- STRIDED: Attend every k-th position (reduces O(n²) to O(n²/k))
- DILATED: Exponentially increasing gaps [1, 2, 4, ...]
- LOCAL_GLOBAL: Local window + global tokens (e.g., [CLS])

Usage:
    mask = AttentionMask.causal(seq_len=10)
    mask = AttentionMask.sliding_window(seq_len=10, window_size=3)
    mask = AttentionMask.strided(seq_len=10, stride=2)
    mask = AttentionMask.dilated(seq_len=10, dilation_rates=[1, 2, 4])
    mask = AttentionMask.local_global(seq_len=10, local_window=2, global_positions=[0])
    mask = AttentionMask.from_strategy(MaskStrategy.CAUSAL, seq_len=10)
"""

from enum import Enum, auto
from typing import Callable
from torch import Tensor, zeros, ones, tril, triu, bool as tbool, float32


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


class AttentionMask:
    """
    Factory and container for attention masks.

    Masks are boolean tensors where True = can attend, False = cannot attend.
    Shape: [query_len, key_len]

    For self-attention: query_len == key_len
    For cross-attention: typically bidirectional (no masking)
    """

    @staticmethod
    def causal(seq_len: int) -> Tensor:
        """
        Create causal (lower triangular) mask.

        Position i can attend to positions 0..i (inclusive).
        Used for autoregressive decoding.

        Args:
            seq_len: Sequence length

        Returns:
            Boolean mask [seq_len, seq_len] where mask[i,j] = (j <= i)
        """
        mask = tril(ones(seq_len, seq_len, dtype=tbool))
        return mask

    @staticmethod
    def bidirectional(query_len: int, key_len: int | None = None) -> Tensor:
        """
        Create bidirectional (full) mask - all positions can attend.

        Args:
            query_len: Query sequence length
            key_len: Key sequence length (default: same as query_len)

        Returns:
            Boolean mask [query_len, key_len] with all True
        """
        key_len = key_len or query_len
        return ones(query_len, key_len, dtype=tbool)

    @staticmethod
    def sliding_window(
        seq_len: int,
        window_size: int,
        causal: bool = False,
    ) -> Tensor:
        """
        Create sliding window mask.

        Position i can attend to positions max(0, i-window_size) to min(seq_len, i+window_size).
        Optionally combined with causal masking.

        Args:
            seq_len: Sequence length
            window_size: How many positions to look left and right
            causal: If True, also apply causal mask (only look left)

        Returns:
            Boolean mask [seq_len, seq_len]
        """
        mask = zeros(seq_len, seq_len, dtype=tbool)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = i + 1 if causal else min(seq_len, i + window_size + 1)
            mask[i, start:end] = True
        return mask

    @staticmethod
    def block(seq_len: int, block_size: int) -> Tensor:
        """
        Create block-diagonal mask.

        Positions can only attend within their block.
        Block 0: positions 0..block_size-1
        Block 1: positions block_size..2*block_size-1
        etc.

        Args:
            seq_len: Sequence length
            block_size: Size of each block

        Returns:
            Boolean mask [seq_len, seq_len]
        """
        mask = zeros(seq_len, seq_len, dtype=tbool)
        for i in range(seq_len):
            block_start = (i // block_size) * block_size
            block_end = min(block_start + block_size, seq_len)
            mask[i, block_start:block_end] = True
        return mask

    @staticmethod
    def prefix(seq_len: int, prefix_len: int) -> Tensor:
        """
        Create prefix-bidirectional mask.

        First prefix_len positions can attend bidirectionally.
        Remaining positions are causal (can attend to prefix + past).

        Args:
            seq_len: Sequence length
            prefix_len: Number of prefix positions

        Returns:
            Boolean mask [seq_len, seq_len]
        """
        mask = zeros(seq_len, seq_len, dtype=tbool)
        for i in range(seq_len):
            if i < prefix_len:
                # Prefix: bidirectional
                mask[i, :] = True
            else:
                # Rest: causal, but can always see prefix
                mask[i, :prefix_len] = True  # Always see prefix
                mask[i, prefix_len:i+1] = True  # Causal for rest
        return mask

    @staticmethod
    def strided(
        seq_len: int,
        stride: int = 2,
        causal: bool = False,
    ) -> Tensor:
        """
        Create strided attention mask.

        Position i attends to positions 0, stride, 2*stride, ...
        Reduces complexity from O(n²) to O(n²/stride).

        Args:
            seq_len: Sequence length
            stride: Attend every stride-th position
            causal: If True, also apply causal constraint (j <= i)

        Returns:
            Boolean mask [seq_len, seq_len]
        """
        mask = zeros(seq_len, seq_len, dtype=tbool)
        for i in range(seq_len):
            for j in range(0, seq_len, stride):
                if not causal or j <= i:
                    mask[i, j] = True
            # Always attend to self
            mask[i, i] = True
        return mask

    @staticmethod
    def dilated(
        seq_len: int,
        dilation_rates: list[int] | None = None,
        window_per_rate: int = 2,
        causal: bool = False,
    ) -> Tensor:
        """
        Create dilated attention mask with multiple dilation rates.

        Each rate defines a sparse pattern: attend to positions at multiples
        of that rate within a window. Captures local + global context.

        Example with rates=[1,2,4], window_per_rate=2:
        - Rate 1: attend at distances 0, 1, 2 (dense local)
        - Rate 2: attend at distances 0, 2, 4 (sparse medium)
        - Rate 4: attend at distances 0, 4, 8 (sparse global)

        Args:
            seq_len: Sequence length
            dilation_rates: List of dilation rates (default: [1, 2, 4])
            window_per_rate: How many steps per rate (default: 2)
            causal: If True, only attend to positions j <= i

        Returns:
            Boolean mask [seq_len, seq_len]
        """
        if dilation_rates is None:
            dilation_rates = [1, 2, 4]

        mask = zeros(seq_len, seq_len, dtype=tbool)
        for i in range(seq_len):
            # Always attend to self
            mask[i, i] = True

            for rate in dilation_rates:
                # Attend to positions at multiples of rate
                for step in range(1, window_per_rate + 1):
                    dist = rate * step
                    # Look backward (always allowed)
                    if i - dist >= 0:
                        mask[i, i - dist] = True
                    # Look forward (only if not causal)
                    if not causal and i + dist < seq_len:
                        mask[i, i + dist] = True

        return mask

    @staticmethod
    def local_global(
        seq_len: int,
        local_window: int = 3,
        global_positions: list[int] | None = None,
        causal: bool = False,
    ) -> Tensor:
        """
        Create local-global attention mask.

        Combines local sliding window with global token access.
        Global positions (e.g., [CLS], first/last) are visible to all.

        Args:
            seq_len: Sequence length
            local_window: Local window size (left and right)
            global_positions: Positions that all can attend to/from (default: [0])
            causal: If True, apply causal constraint to local window

        Returns:
            Boolean mask [seq_len, seq_len]
        """
        if global_positions is None:
            global_positions = [0]  # First position is global by default

        mask = zeros(seq_len, seq_len, dtype=tbool)
        for i in range(seq_len):
            # Local window
            start = max(0, i - local_window)
            end = i + 1 if causal else min(seq_len, i + local_window + 1)
            mask[i, start:end] = True

            # Global positions: can attend to them and they attend to all
            for g in global_positions:
                if 0 <= g < seq_len:
                    mask[i, g] = True  # i can attend to global
                    mask[g, i] = True  # global can attend to i

        return mask

    @staticmethod
    def custom(
        query_len: int,
        key_len: int,
        can_attend: Callable[[int, int], bool],
    ) -> Tensor:
        """
        Create custom mask from a function.

        Args:
            query_len: Query sequence length
            key_len: Key sequence length
            can_attend: Function(query_pos, key_pos) -> bool

        Returns:
            Boolean mask [query_len, key_len]
        """
        mask = zeros(query_len, key_len, dtype=tbool)
        for i in range(query_len):
            for j in range(key_len):
                if can_attend(i, j):
                    mask[i, j] = True
        return mask

    @staticmethod
    def from_strategy(
        strategy: MaskStrategy,
        seq_len: int,
        key_len: int | None = None,
        window_size: int = 3,
        block_size: int = 4,
        prefix_len: int = 2,
        can_attend: Callable[[int, int], bool] | None = None,
        stride: int = 2,
        dilation_rates: list[int] | None = None,
        window_per_rate: int = 2,
        global_positions: list[int] | None = None,
        causal: bool = False,
    ) -> Tensor:
        """
        Create mask from a predefined strategy.

        Args:
            strategy: The masking strategy to use
            seq_len: Query sequence length
            key_len: Key sequence length (for cross-attention)
            window_size: Window size for SLIDING_WINDOW and LOCAL_GLOBAL
            block_size: Block size for BLOCK
            prefix_len: Prefix length for PREFIX
            can_attend: Function for CUSTOM strategy
            stride: Stride for STRIDED strategy
            dilation_rates: Dilation rates for DILATED strategy
            window_per_rate: Steps per rate for DILATED strategy
            global_positions: Global positions for LOCAL_GLOBAL strategy
            causal: Whether to apply causal constraint (for sparse patterns)

        Returns:
            Boolean mask tensor
        """
        if strategy == MaskStrategy.CAUSAL:
            return AttentionMask.causal(seq_len)
        elif strategy == MaskStrategy.BIDIRECTIONAL:
            return AttentionMask.bidirectional(seq_len, key_len)
        elif strategy == MaskStrategy.SLIDING_WINDOW:
            return AttentionMask.sliding_window(seq_len, window_size, causal)
        elif strategy == MaskStrategy.BLOCK:
            return AttentionMask.block(seq_len, block_size)
        elif strategy == MaskStrategy.PREFIX:
            return AttentionMask.prefix(seq_len, prefix_len)
        elif strategy == MaskStrategy.CUSTOM:
            if can_attend is None:
                raise ValueError("CUSTOM strategy requires can_attend function")
            key_len = key_len or seq_len
            return AttentionMask.custom(seq_len, key_len, can_attend)
        elif strategy == MaskStrategy.STRIDED:
            return AttentionMask.strided(seq_len, stride, causal)
        elif strategy == MaskStrategy.DILATED:
            return AttentionMask.dilated(seq_len, dilation_rates, window_per_rate, causal)
        elif strategy == MaskStrategy.LOCAL_GLOBAL:
            return AttentionMask.local_global(seq_len, window_size, global_positions, causal)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def combine(masks: list[Tensor], mode: str = "and") -> Tensor:
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

    @staticmethod
    def to_float(mask: Tensor, fill_value: float = float('-inf')) -> Tensor:
        """
        Convert boolean mask to float mask for softmax.

        True positions get 0.0, False positions get fill_value (typically -inf).

        Args:
            mask: Boolean mask
            fill_value: Value for masked positions

        Returns:
            Float mask
        """
        result = zeros(mask.shape, dtype=float32)
        result[~mask] = fill_value
        return result


def can_attend(mask: Tensor, query_pos: int, key_pos: int) -> bool:
    """
    Check if a query position can attend to a key position.

    Args:
        mask: Boolean attention mask [query_len, key_len]
        query_pos: Query position
        key_pos: Key position

    Returns:
        True if query can attend to key
    """
    return bool(mask[query_pos, key_pos].item())
