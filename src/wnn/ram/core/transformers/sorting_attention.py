"""
Computed Sorting Attention

Attention mechanism that sorts tokens by their numeric value.
Uses computed comparisons for 100% generalization.

This is a COMPUTED (non-learnable) attention mechanism - no training required.
"""

from torch import Tensor, zeros, uint8, float32

from wnn.ram.core.transformers.attention_base import ComputedAttention
from wnn.ram.core.transformers.computed_arithmetic import bits_to_int


class ComputedSortingAttention(ComputedAttention):
    """
    Sorting attention using computed comparisons.

    Key insight: Sorting can be done with COMPUTED attention (no learning needed)
    if we can compare tokens numerically via their bit patterns.

    For output position i, we attend to the token that has exactly i tokens
    smaller than it (i.e., the (i+1)th smallest token).

    This generalizes 100% to unseen tokens because comparison is computed,
    not learned from a lookup table.
    """

    def __init__(
        self,
        input_bits: int,
        descending: bool = False,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            descending: If True, sort largest first (descending order)
            rng: Random seed (unused, for API compatibility)
        """
        super().__init__()

        self.input_bits = input_bits
        self.descending = descending

        print(f"[ComputedSortingAttention] input={input_bits}b, "
              f"order={'descending' if descending else 'ascending'}")

    def _count_smaller(self, token: Tensor, all_tokens: list[Tensor]) -> int:
        """Count how many tokens in the list are smaller than this token."""
        token_val = bits_to_int(token)
        count = 0
        for other in all_tokens:
            other_val = bits_to_int(other)
            if other_val < token_val:
                count += 1
        return count

    def _count_larger(self, token: Tensor, all_tokens: list[Tensor]) -> int:
        """Count how many tokens in the list are larger than this token."""
        token_val = bits_to_int(token)
        count = 0
        for other in all_tokens:
            other_val = bits_to_int(other)
            if other_val > token_val:
                count += 1
        return count

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """
        Get attention weights for sorting (implements AttentionBase interface).

        Returns a [num_queries, num_keys] tensor where each row has exactly
        one 1.0 at the position of the token that belongs at that output position.
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        # Get values and create (value, original_index) pairs
        values = [bits_to_int(t) for t in tokens]
        indexed = [(val, i) for i, val in enumerate(values)]

        # Sort by value (stable sort preserves order for ties)
        if self.descending:
            indexed.sort(key=lambda x: -x[0])
        else:
            indexed.sort(key=lambda x: x[0])

        # Build full attention matrix
        weights = zeros(n, n, dtype=float32)
        for query_pos in range(n):
            target_input_pos = indexed[query_pos][1]
            weights[query_pos, target_input_pos] = 1.0

        return weights

    def _get_weights_for_position(self, tokens: list[Tensor], query_pos: int) -> list[float]:
        """Get attention weights for a single output position (internal helper)."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        values = [bits_to_int(t) for t in tokens]
        indexed = [(val, i) for i, val in enumerate(values)]

        if self.descending:
            indexed.sort(key=lambda x: -x[0])
        else:
            indexed.sort(key=lambda x: x[0])

        target_input_pos = indexed[query_pos][1]
        weights = [0.0] * n
        weights[target_input_pos] = 1.0
        return weights

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """
        Sort the sequence (implements AttentionBase interface).

        No learning required - attention is computed from token comparisons.
        context parameter is ignored (sorting is always self-attention).
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        outputs = []

        for i in range(seq_len):
            weights = self._get_weights_for_position(tokens, i)

            # Find the attended position (should be exactly one)
            for j, w in enumerate(weights):
                if w > 0:
                    outputs.append(tokens[j].clone())
                    break
            else:
                # Fallback if no match (shouldn't happen with valid input)
                outputs.append(zeros(self.input_bits, dtype=uint8))

        return outputs

    def visualize_attention(self, tokens: list[Tensor]) -> str:
        """Visualize sorting attention weights."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        lines = [f"ComputedSortingAttention ({'descending' if self.descending else 'ascending'}):"]
        lines.append("     " + " ".join(f"{j:4d}" for j in range(seq_len)))

        for i in range(seq_len):
            weights = self._get_weights_for_position(tokens, i)
            row = f"{i:2d}: "
            for w in weights:
                row += f"{w:4.2f} "
            lines.append(row)

        return "\n".join(lines)


# Backwards-compatible alias
SortingAttention = ComputedSortingAttention
