"""
Position-Only RAM Attention

Attention that routes based ONLY on positions, not token content.
This enables pre-training of position patterns (COPY, SHIFT, REVERSE, etc.)
with guaranteed generalization to any token values.

Key insight: If attention only sees positions, pre-training on position pairs
works for ALL possible tokens - achieving 100% generalization.

Pattern Categories:
- Length-Invariant: COPY, SHIFT_LEFT, SHIFT_RIGHT, FIRST
  These patterns can be trained once on max_seq_len and work for any shorter sequence.
  Example: COPY(i→i) - position 3 always attends to position 3 regardless of sequence length.

- Length-Dependent: REVERSE, LAST
  These patterns depend on sequence length and must be trained at the target length.
  Example: REVERSE(i→n-1-i) - for n=8, position 0 attends to 7; for n=4, to 3.
"""

from enum import IntEnum
from torch import Tensor, zeros, uint8, tensor, cat
from torch.nn import Module, ModuleList

from wnn.ram.core import RAMLayer


class PositionPattern(IntEnum):
    """Pre-defined position attention patterns."""
    COPY = 0       # Position i attends to position i
    SHIFT_LEFT = 1   # Position i attends to position i-1
    SHIFT_RIGHT = 2  # Position i attends to position i+1
    REVERSE = 3    # Position i attends to position n-1-i
    FIRST = 4      # All positions attend to position 0
    LAST = 5       # All positions attend to position n-1
    BROADCAST = 6  # All positions attend to all positions


class PositionOnlyAttention(Module):
    """
    Attention that routes based ONLY on query and key positions.

    Unlike RAMAttention which sees [query, key, positions], this only sees
    [query_position, key_position]. This enables:
    - Pre-training position patterns that generalize to ANY tokens
    - 100% generalization for COPY, SHIFT, REVERSE, etc.

    The value projection still sees token content for transformation.
    """

    def __init__(
        self,
        token_bits: int,
        max_seq_len: int = 16,
        num_heads: int = 1,
        causal: bool = False,
        rng: int | None = None,
    ):
        """
        Args:
            token_bits: Bits per token
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            causal: If True, position i can only attend to j <= i
            rng: Random seed
        """
        super().__init__()

        self.token_bits = token_bits
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.causal = causal

        # Position encoding
        self.n_position_bits = max_seq_len.bit_length()

        # Routing network: [query_pos, key_pos] -> attend? (binary)
        # Only 2 * n_position_bits input - NO token content
        routing_input_bits = 2 * self.n_position_bits

        self.routing_heads = ModuleList([
            RAMLayer(
                total_input_bits=routing_input_bits,
                num_neurons=1,  # Binary: attend or not
                n_bits_per_neuron=routing_input_bits,  # See ALL position bits
                rng=rng + i if rng else None,
            )
            for i in range(num_heads)
        ])

        # Value projection: [token, position] -> projected_value
        value_input_bits = token_bits + self.n_position_bits
        self.value_heads = ModuleList([
            RAMLayer(
                total_input_bits=value_input_bits,
                num_neurons=token_bits,
                n_bits_per_neuron=min(value_input_bits, 10),
                rng=rng + num_heads + i if rng else None,
            )
            for i in range(num_heads)
        ])

        # Output combination
        self.output_layer = RAMLayer(
            total_input_bits=num_heads * token_bits,
            num_neurons=token_bits,
            n_bits_per_neuron=min(num_heads * token_bits, 12),
            rng=rng + 2 * num_heads if rng else None,
        )

        print(f"[PositionOnlyAttention] heads={num_heads}, "
              f"tokens={token_bits}b, pos={self.n_position_bits}b, "
              f"causal={causal}")

    def _encode_position(self, pos: int) -> Tensor:
        """Encode position as binary bits."""
        pos_bits = zeros(self.n_position_bits, dtype=uint8)
        for b in range(self.n_position_bits):
            pos_bits[self.n_position_bits - 1 - b] = (pos >> b) & 1
        return pos_bits

    def _get_routing(self, query_pos: int, key_pos: int, head_idx: int) -> bool:
        """Check if query_pos should attend to key_pos."""
        q_pos = self._encode_position(query_pos)
        k_pos = self._encode_position(key_pos)
        routing_input = cat([q_pos, k_pos]).unsqueeze(0)
        return bool(self.routing_heads[head_idx](routing_input).squeeze().item())

    def _set_routing(self, query_pos: int, key_pos: int, attend: bool, head_idx: int):
        """Set whether query_pos should attend to key_pos."""
        q_pos = self._encode_position(query_pos)
        k_pos = self._encode_position(key_pos)
        routing_input = cat([q_pos, k_pos]).unsqueeze(0)
        target = tensor([[1 if attend else 0]], dtype=uint8)
        self.routing_heads[head_idx].commit(routing_input, target)

    # ------------------------------------------------------------------
    # Pre-training Methods
    # ------------------------------------------------------------------

    def pretrain_pattern(self, pattern: PositionPattern, seq_len: int, head_idx: int = 0) -> int:
        """
        Pre-train a position pattern on a head.

        Args:
            pattern: Which pattern to train
            seq_len: Sequence length
            head_idx: Which head to train

        Returns:
            Number of patterns written
        """
        count = 0
        for i in range(seq_len):
            for j in range(seq_len):
                if self.causal and j > i:
                    continue

                # Determine if this (i, j) pair should attend
                match pattern:
                    case PositionPattern.COPY:
                        attend = (i == j)
                    case PositionPattern.SHIFT_LEFT:
                        attend = (j == i - 1) if i > 0 else False
                    case PositionPattern.SHIFT_RIGHT:
                        attend = (j == i + 1) if i < seq_len - 1 else False
                    case PositionPattern.REVERSE:
                        attend = (j == seq_len - 1 - i)
                    case PositionPattern.FIRST:
                        attend = (j == 0)
                    case PositionPattern.LAST:
                        attend = (j == seq_len - 1)
                    case PositionPattern.BROADCAST:
                        attend = True
                    case _:
                        attend = False

                self._set_routing(i, j, attend, head_idx)
                count += 1
        return count

    def pretrain_copy(self, seq_len: int, head_idx: int = 0) -> int:
        """Pre-train COPY pattern: position i attends to position i."""
        return self.pretrain_pattern(PositionPattern.COPY, seq_len, head_idx)

    def pretrain_shift(self, seq_len: int, offset: int = -1, head_idx: int = 0) -> int:
        """Pre-train SHIFT pattern: position i attends to position i+offset."""
        count = 0
        for i in range(seq_len):
            for j in range(seq_len):
                if self.causal and j > i:
                    continue
                target = i + offset
                attend = (j == target) and (0 <= target < seq_len)
                self._set_routing(i, j, attend, head_idx)
                count += 1
        return count

    def pretrain_reverse(self, seq_len: int, head_idx: int = 0) -> int:
        """Pre-train REVERSE pattern: position i attends to position n-1-i."""
        return self.pretrain_pattern(PositionPattern.REVERSE, seq_len, head_idx)

    def pretrain_first(self, seq_len: int, head_idx: int = 0) -> int:
        """Pre-train FIRST pattern: all positions attend to position 0."""
        return self.pretrain_pattern(PositionPattern.FIRST, seq_len, head_idx)

    def pretrain_last(self, seq_len: int, head_idx: int = 0) -> int:
        """Pre-train LAST pattern: all positions attend to position n-1."""
        return self.pretrain_pattern(PositionPattern.LAST, seq_len, head_idx)

    def pretrain_identity_values(self, seq_len: int) -> int:
        """Pre-train value heads to be identity (output = input)."""
        count = 0
        for pos in range(seq_len):
            pos_bits = self._encode_position(pos)
            for h in range(self.num_heads):
                # Train identity for a few representative token values
                for val in range(min(4, 2 ** self.token_bits)):
                    token = zeros(self.token_bits, dtype=uint8)
                    for b in range(self.token_bits):
                        token[b] = (val >> b) & 1
                    val_input = cat([token, pos_bits]).unsqueeze(0)
                    self.value_heads[h].commit(val_input, token.unsqueeze(0))
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """
        Apply position-only attention.

        Args:
            tokens: List of [token_bits] tensors

        Returns:
            Transformed tokens
        """
        seq_len = len(tokens)
        outputs = []

        for i in range(seq_len):
            head_outputs = []

            for h in range(self.num_heads):
                # Find which positions to attend to
                attended_values = []
                for j in range(seq_len):
                    if self.causal and j > i:
                        continue
                    if self._get_routing(i, j, h):
                        # Project value
                        tok = tokens[j].squeeze()
                        pos_bits = self._encode_position(j)
                        val_input = cat([tok, pos_bits]).unsqueeze(0)
                        projected = self.value_heads[h](val_input).squeeze()
                        attended_values.append(projected)

                # Simple aggregation: take first attended value (or zeros)
                if attended_values:
                    head_out = attended_values[0]  # Could use voting/average
                else:
                    head_out = zeros(self.token_bits, dtype=uint8)

                head_outputs.append(head_out)

            # Combine heads
            if self.num_heads == 1:
                output = head_outputs[0]
            else:
                combined = cat(head_outputs).unsqueeze(0)
                output = self.output_layer(combined).squeeze()

            outputs.append(output)

        return outputs

    def get_attention_weights(self, tokens: list[Tensor]) -> Tensor:
        """Get attention pattern (position-based, ignores token content)."""
        seq_len = len(tokens)
        weights = zeros(seq_len, seq_len)

        for i in range(seq_len):
            for j in range(seq_len):
                if self.causal and j > i:
                    continue
                if self._get_routing(i, j, head_idx=0):
                    weights[i, j] = 1.0

        return weights
