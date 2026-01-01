"""
Computed Attention Mechanisms

Additional computed (non-learnable) attention mechanisms that achieve
100% generalization by using direct computation instead of learned lookup.

These extend the existing ComputedSortingAttention and ComputedMinMaxAttention
with additional useful operations.
"""

from torch import Tensor, zeros, uint8, float32

from wnn.ram.core.transformers.attention_base import ComputedAttention
from wnn.ram.core.transformers.computed_arithmetic import bits_to_int, int_to_bits


class ComputedMedianAttention(ComputedAttention):
    """
    Find the median token using computed comparisons.

    Every position outputs the median token from the sequence.
    For even-length sequences, uses the lower median.
    Generalizes 100% because comparison is computed, not learned.
    """

    def __init__(
        self,
        input_bits: int,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            rng: Random seed (unused, for API compatibility)
        """
        super().__init__()
        self.input_bits = input_bits
        print(f"[ComputedMedianAttention] input={input_bits}b")

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """
        Get attention weights for median finding.

        Returns a [num_queries, num_keys] tensor where every row has 1.0 at
        the position of the median token.
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        # Get values and find median
        values = [bits_to_int(t) for t in tokens]
        sorted_vals = sorted(values)
        median_idx = (n - 1) // 2  # Lower median for even n
        median_val = sorted_vals[median_idx]

        # Find first occurrence of median value
        median_pos = values.index(median_val)

        # Build attention weights
        weights = zeros(n, n, dtype=float32)
        for i in range(n):
            weights[i, median_pos] = 1.0

        return weights

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Output median token at every position."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        values = [bits_to_int(t) for t in tokens]
        sorted_vals = sorted(values)
        median_idx = (n - 1) // 2
        median_val = sorted_vals[median_idx]

        # Find first token with median value
        median_pos = values.index(median_val)
        median_token = tokens[median_pos].clone()

        return [median_token.clone() for _ in range(n)]


class ComputedArgMaxAttention(ComputedAttention):
    """
    Find the position (index) of the max (or min) token.

    Outputs the INDEX of the max/min token as a binary number at every position.
    Useful for pointer networks and selection operations.
    """

    def __init__(
        self,
        input_bits: int,
        output_bits: int | None = None,
        find_max: bool = True,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            output_bits: Bits for output index (default: ceil(log2(max_seq_len)))
            find_max: If True, find argmax; else argmin
            rng: Random seed (unused)
        """
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits or 8  # Default: support up to 256 positions
        self.find_max = find_max
        mode = "argmax" if find_max else "argmin"
        print(f"[ComputedArgMaxAttention] input={input_bits}b, output={self.output_bits}b, mode={mode}")

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """
        Get attention weights showing which position has max/min.
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        values = [bits_to_int(t) for t in tokens]
        if self.find_max:
            target_val = max(values)
        else:
            target_val = min(values)

        target_pos = values.index(target_val)

        weights = zeros(n, n, dtype=float32)
        for i in range(n):
            weights[i, target_pos] = 1.0

        return weights

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Output the INDEX of max/min as binary at every position."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        values = [bits_to_int(t) for t in tokens]
        if self.find_max:
            target_val = max(values)
        else:
            target_val = min(values)

        target_pos = values.index(target_val)

        # Convert position to binary
        pos_bits = int_to_bits(target_pos, self.output_bits)

        return [pos_bits.clone() for _ in range(n)]


class ComputedCountDistinctAttention(ComputedAttention):
    """
    Count the number of distinct values in the sequence.

    Outputs the COUNT of unique tokens as a binary number at every position.
    Useful for aggregation and cardinality operations.
    """

    def __init__(
        self,
        input_bits: int,
        output_bits: int | None = None,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            output_bits: Bits for output count (default: same as input)
            rng: Random seed (unused)
        """
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits or input_bits
        print(f"[ComputedCountDistinctAttention] input={input_bits}b, output={self.output_bits}b")

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """
        Get attention weights (uniform over all positions for count).
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        # Uniform attention (count considers all tokens equally)
        weights = zeros(n, n, dtype=float32)
        weights.fill_(1.0 / n)

        return weights

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Output the COUNT of distinct values at every position."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        # Count distinct values
        values = set(bits_to_int(t) for t in tokens)
        count = len(values)

        # Convert count to binary
        count_bits = int_to_bits(count, self.output_bits)

        return [count_bits.clone() for _ in range(n)]


class ComputedSumAttention(ComputedAttention):
    """
    Sum all token values in the sequence.

    Outputs the SUM of all token values (mod 2^output_bits) at every position.
    Useful for aggregation operations.
    """

    def __init__(
        self,
        input_bits: int,
        output_bits: int | None = None,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            output_bits: Bits for output sum (default: input_bits + 4 for overflow)
            rng: Random seed (unused)
        """
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits or (input_bits + 4)
        print(f"[ComputedSumAttention] input={input_bits}b, output={self.output_bits}b")

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """Uniform attention weights (sum considers all tokens equally)."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        weights = zeros(n, n, dtype=float32)
        weights.fill_(1.0 / n)

        return weights

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Output the SUM of all values at every position."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        # Compute sum
        total = sum(bits_to_int(t) for t in tokens)
        max_val = 2 ** self.output_bits
        total = total % max_val  # Mod to fit in output bits

        # Convert to binary
        sum_bits = int_to_bits(total, self.output_bits)

        return [sum_bits.clone() for _ in range(n)]


class ComputedShiftAttention(ComputedAttention):
    """
    Shift attention: each position attends to a fixed offset position.

    This solves DOUBLE (x * 2) and other permutation tasks:
    - SHIFT_LEFT (offset=1): output[i] = input[i+1], solves DOUBLE
    - SHIFT_RIGHT (offset=-1): output[i] = input[i-1], solves HALVE (integer div)

    100% generalization because routing is computed, not learned.
    """

    def __init__(
        self,
        input_bits: int,
        offset: int = 1,
        fill_value: int = 0,
        wrap: bool = False,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token (also determines sequence length)
            offset: How many positions to shift (positive=left, negative=right)
            fill_value: Value to fill shifted-in positions (0 or 1)
            wrap: If True, wrap around (rotate); if False, fill with fill_value
            rng: Random seed (unused)
        """
        super().__init__()
        self.input_bits = input_bits
        self.offset = offset
        self.fill_value = fill_value
        self.wrap = wrap

        direction = "left" if offset > 0 else "right"
        mode = "rotate" if wrap else "shift"
        print(f"[ComputedShiftAttention] bits={input_bits}, {mode}_{direction} by {abs(offset)}")

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """Get attention weights for shift pattern."""
        n = len(tokens)
        weights = zeros(n, n, dtype=float32)

        for i in range(n):
            src = i + self.offset
            if self.wrap:
                src = src % n
            if 0 <= src < n:
                weights[i, src] = 1.0
            # else: no attention (will use fill_value)

        return weights

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Apply shift to the token sequence."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)
        token_bits = len(tokens[0]) if tokens else self.input_bits

        outputs = []
        for i in range(n):
            src = i + self.offset
            if self.wrap:
                src = src % n
            if 0 <= src < n:
                outputs.append(tokens[src].clone())
            else:
                # Fill with fill_value
                fill = zeros(token_bits, dtype=uint8)
                if self.fill_value:
                    fill.fill_(1)
                outputs.append(fill)

        return outputs


class ComputedMeanAttention(ComputedAttention):
    """
    Compute the mean of all token values in the sequence.

    Outputs the integer MEAN (floor) of all token values at every position.
    """

    def __init__(
        self,
        input_bits: int,
        output_bits: int | None = None,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            output_bits: Bits for output mean (default: same as input)
            rng: Random seed (unused)
        """
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits or input_bits
        print(f"[ComputedMeanAttention] input={input_bits}b, output={self.output_bits}b")

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """Uniform attention weights."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        weights = zeros(n, n, dtype=float32)
        weights.fill_(1.0 / n)

        return weights

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Output the MEAN of all values at every position."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        # Compute mean (floor division)
        total = sum(bits_to_int(t) for t in tokens)
        mean = total // n

        # Convert to binary
        mean_bits = int_to_bits(mean, self.output_bits)

        return [mean_bits.clone() for _ in range(n)]
