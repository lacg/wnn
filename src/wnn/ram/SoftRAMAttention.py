"""
Soft RAM Attention via Voting

Traditional soft attention:
  weights = softmax(Q·K^T / √d)    # Continuous [0,1]
  output = Σ (weights_i × V_i)     # Weighted sum

Hard RAM attention (current):
  attend = 1 or 0                   # Binary decision
  output = aggregate(attended_values)  # Discrete selection

Soft RAM Attention (this module):
  vote_count[j] = Σ heads_attending[h][j]  # Count votes
  weight[j] = vote_count[j] / num_heads    # Approximate [0,1]
  output = weighted_aggregate(values, weights)

Key insight: With enough heads, voting approximates continuous weights.
  8 heads → weights in {0, 0.125, 0.25, ..., 1.0}
  16 heads → weights in {0, 0.0625, 0.125, ..., 1.0}

Aggregation Strategies:
  - TOP_1: Winner-take-all (best for retrieval)
  - MAJORITY: Per-bit weighted voting (best for combining)
  - THRESHOLD: Include only above 50% votes
"""

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMGeneralization import GeneralizingProjection, MapperStrategy
from wnn.ram.encoders_decoders import PositionMode, PositionEncoderFactory

from torch import Tensor, zeros, uint8, cat, tensor
from torch.nn import Module, ModuleList
from enum import IntEnum


class AggregationStrategy(IntEnum):
    """Aggregation strategy for soft attention."""
    TOP_1 = 0       # Winner-take-all (most robust)
    MAJORITY = 1    # Per-bit weighted majority
    THRESHOLD = 2   # XOR values above 50% threshold
    TOP_K = 3       # XOR top K values


class SoftRAMAttention(Module):
    """
    Soft attention approximation using head voting.

    Each head votes on which positions to attend.
    Vote counts create pseudo-continuous attention weights.
    Weighted aggregation combines values based on votes.

    Aggregation strategies:
        TOP_1: Return highest-voted value (best for retrieval)
        MAJORITY: Per-bit weighted voting (best for combining)
        THRESHOLD: XOR values above 50% votes
        TOP_K: XOR top K highest-voted values
    """

    def __init__(
        self,
        input_bits: int,
        num_heads: int = 8,  # More heads = finer weight granularity
        aggregation: AggregationStrategy = AggregationStrategy.TOP_1,
        value_strategy: MapperStrategy = MapperStrategy.BIT_LEVEL,
        position_mode: PositionMode = PositionMode.RELATIVE,
        max_seq_len: int = 16,
        causal: bool = True,
        top_k: int = 2,  # For TOP_K strategy
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            num_heads: Number of voting heads (more = finer weights)
            aggregation: Aggregation strategy (TOP_1, MAJORITY, THRESHOLD, TOP_K)
            value_strategy: Generalization strategy for value projection (BIT_LEVEL recommended)
            position_mode: How to encode positions
            max_seq_len: Maximum sequence length
            causal: Only attend to past positions
            top_k: Number of values to combine for TOP_K strategy
            rng: Random seed
        """
        super().__init__()

        self.input_bits = input_bits
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.value_strategy = value_strategy
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.position_mode = position_mode

        # Position encoding setup
        if position_mode == PositionMode.NONE:
            self.position_encoder = None
            self.n_position_bits = 0
        elif position_mode == PositionMode.RELATIVE:
            self.position_encoder = PositionEncoderFactory.create(
                PositionMode.RELATIVE,
                max_distance=max_seq_len - 1,
            )
            self.n_position_bits = self.position_encoder.n_bits
        elif position_mode == PositionMode.BINARY:
            self.position_encoder = PositionEncoderFactory.create(
                PositionMode.BINARY,
                max_seq_len=max_seq_len,
            )
            self.n_position_bits = self.position_encoder.n_bits * 2  # query + key pos
        else:
            raise ValueError(f"Unsupported position_mode: {position_mode}")

        # Similarity input: [query, key, position]
        similarity_input_bits = 2 * input_bits + self.n_position_bits

        # Voting heads: each decides attend/don't-attend
        self.voting_heads = ModuleList([
            RAMLayer(
                total_input_bits=similarity_input_bits,
                num_neurons=1,
                n_bits_per_neuron=min(similarity_input_bits, 12),
                rng=rng + i if rng else None,
            )
            for i in range(num_heads)
        ])

        # Value projection using GeneralizingProjection for better generalization
        self.value_projection = GeneralizingProjection(
            input_bits=input_bits,
            output_bits=input_bits,
            strategy=value_strategy,
            rng=rng + num_heads if rng else None,
        )

        # Weighted aggregation layers (one per weight level)
        # With N heads, we have N+1 weight levels (0/N, 1/N, ..., N/N)
        # We learn how to combine values at each weight level
        self.weight_levels = num_heads + 1

        # Aggregation: given [value, weight_level], produce contribution
        # weight_level encoded as bits
        weight_bits = self.weight_levels.bit_length()
        self.weight_aggregator = RAMLayer(
            total_input_bits=input_bits + weight_bits,
            num_neurons=input_bits,
            n_bits_per_neuron=min(input_bits + weight_bits, 12),
            rng=rng + num_heads + 1 if rng else None,
        )
        self.weight_bits = weight_bits

        # Final output combination
        self.output_layer = RAMLayer(
            total_input_bits=input_bits * max_seq_len,  # Combine all weighted values
            num_neurons=input_bits,
            n_bits_per_neuron=min(input_bits * 2, 12),  # Can't look at all
            rng=rng + num_heads + 2 if rng else None,
        )

        print(f"[SoftRAMAttention] heads={num_heads}, input={input_bits}b, "
              f"aggregation={aggregation.name}, value={value_strategy.name}, causal={causal}")

    def _encode_weight(self, vote_count: int) -> Tensor:
        """Encode vote count as bits."""
        bits = zeros(self.weight_bits, dtype=uint8)
        for i in range(self.weight_bits - 1, -1, -1):
            bits[self.weight_bits - 1 - i] = (vote_count >> i) & 1
        return bits

    def _compute_votes(
        self,
        query: Tensor,
        keys: list[Tensor],
        query_pos: int,
    ) -> list[int]:
        """
        Count votes for each key position.

        Returns:
            votes: List of vote counts [0, num_heads] for each key
        """
        votes = []

        for j, key in enumerate(keys):
            # Causal mask
            if self.causal and j > query_pos:
                votes.append(0)
                continue

            # Count how many heads vote to attend
            vote_count = 0

            for head in self.voting_heads:
                # Build similarity input
                parts = [query, key]

                if self.position_mode == PositionMode.RELATIVE:
                    rel_dist = self.position_encoder.encode_relative(query_pos, j)
                    parts.append(rel_dist)
                elif self.position_mode == PositionMode.BINARY:
                    q_pos = self.position_encoder.encode(query_pos)
                    k_pos = self.position_encoder.encode(j)
                    parts.extend([q_pos, k_pos])

                similarity_input = cat(parts).unsqueeze(0)
                attend = head(similarity_input).squeeze().item()
                vote_count += attend

            votes.append(vote_count)

        return votes

    def _aggregate_top_1(self, values: list[Tensor], votes: list[int]) -> Tensor:
        """Winner-take-all: return highest-voted value."""
        if not values or not votes:
            return zeros(self.input_bits, dtype=uint8)

        max_vote = max(votes)
        if max_vote == 0:
            return zeros(self.input_bits, dtype=uint8)

        for val, vote in zip(values, votes):
            if vote == max_vote:
                return val.clone()

        return zeros(self.input_bits, dtype=uint8)

    def _aggregate_majority(self, values: list[Tensor], votes: list[int]) -> Tensor:
        """Per-bit weighted majority voting."""
        if not values:
            return zeros(self.input_bits, dtype=uint8)

        result = zeros(self.input_bits, dtype=uint8)

        for bit_pos in range(self.input_bits):
            weighted_ones = sum(
                vote for val, vote in zip(values, votes)
                if val[bit_pos] == 1
            )
            weighted_zeros = sum(
                vote for val, vote in zip(values, votes)
                if val[bit_pos] == 0
            )
            result[bit_pos] = 1 if weighted_ones > weighted_zeros else 0

        return result

    def _aggregate_threshold(self, values: list[Tensor], votes: list[int]) -> Tensor:
        """XOR values with votes above 50% threshold."""
        if not values:
            return zeros(self.input_bits, dtype=uint8)

        threshold = self.num_heads // 2
        result = zeros(self.input_bits, dtype=uint8)
        included = 0

        for val, vote in zip(values, votes):
            if vote >= threshold:
                result = result ^ val
                included += 1

        # Fallback to top-1 if nothing above threshold
        if included == 0:
            return self._aggregate_top_1(values, votes)

        return result

    def _aggregate_top_k(self, values: list[Tensor], votes: list[int]) -> Tensor:
        """XOR top K highest-voted values."""
        if not values:
            return zeros(self.input_bits, dtype=uint8)

        # Sort by votes descending
        sorted_pairs = sorted(zip(votes, values), key=lambda x: -x[0])

        result = zeros(self.input_bits, dtype=uint8)
        for i, (vote, val) in enumerate(sorted_pairs[:self.top_k]):
            if vote > 0:
                result = result ^ val

        return result

    def _aggregate(self, values: list[Tensor], votes: list[int]) -> Tensor:
        """Apply selected aggregation strategy."""
        if self.aggregation == AggregationStrategy.TOP_1:
            return self._aggregate_top_1(values, votes)
        elif self.aggregation == AggregationStrategy.MAJORITY:
            return self._aggregate_majority(values, votes)
        elif self.aggregation == AggregationStrategy.THRESHOLD:
            return self._aggregate_threshold(values, votes)
        elif self.aggregation == AggregationStrategy.TOP_K:
            return self._aggregate_top_k(values, votes)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """
        Apply soft attention to sequence.

        Args:
            tokens: List of token tensors

        Returns:
            outputs: Attended output for each position
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        # Project values using GeneralizingProjection
        values = []
        for tok in tokens:
            proj = self.value_projection(tok)
            if proj.ndim > 1:
                proj = proj.squeeze()
            values.append(proj)

        outputs = []

        for i in range(seq_len):
            query = tokens[i]

            # Get vote counts for each key position
            votes = self._compute_votes(query, tokens, i)

            # Aggregate using selected strategy
            output = self._aggregate(values, votes)
            outputs.append(output)

        return outputs

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        query_pos: int,
    ) -> list[float]:
        """
        Get soft attention weights for a query position.

        Returns:
            weights: List of approximate attention weights [0.0, 1.0]
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        query = tokens[query_pos]
        votes = self._compute_votes(query, tokens, query_pos)

        # Normalize to [0, 1]
        return [v / self.num_heads for v in votes]

    def visualize_attention(self, tokens: list[Tensor]) -> str:
        """Visualize soft attention weights."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        lines = ["Soft Attention Weights (vote counts / num_heads):"]
        lines.append("     " + " ".join(f"{j:4d}" for j in range(seq_len)))

        for i in range(seq_len):
            weights = self.get_attention_weights(tokens, i)
            row = f"{i:2d}: "
            for j, w in enumerate(weights):
                if self.causal and j > i:
                    row += "   - "
                else:
                    # Show weight as fraction
                    row += f"{w:4.2f} "
            lines.append(row)

        return "\n".join(lines)

    def train_attention_weights(
        self,
        tokens: list[Tensor],
        query_pos: int,
        target_weights: list[float],  # Target weights [0.0, 1.0]
    ) -> int:
        """
        Train voting heads to produce target attention weights.

        Strategy: For each (query, key) pair, decide how many heads
        should vote "attend" to approximate target weight.
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        query = tokens[query_pos]
        corrections = 0

        for j, (key, target_w) in enumerate(zip(tokens, target_weights)):
            if self.causal and j > query_pos:
                continue

            # Target vote count
            target_votes = int(target_w * self.num_heads + 0.5)
            target_votes = max(0, min(self.num_heads, target_votes))

            # Build similarity input
            parts = [query, key]
            if self.position_mode == PositionMode.RELATIVE:
                rel_dist = self.position_encoder.encode_relative(query_pos, j)
                parts.append(rel_dist)
            elif self.position_mode == PositionMode.BINARY:
                q_pos = self.position_encoder.encode(query_pos)
                k_pos = self.position_encoder.encode(j)
                parts.extend([q_pos, k_pos])

            similarity_input = cat(parts).unsqueeze(0)

            # Train heads to achieve target vote count
            # First target_votes heads should attend, rest shouldn't
            for h in range(self.num_heads):
                should_attend = 1 if h < target_votes else 0
                current = self.voting_heads[h](similarity_input).squeeze().item()

                if current != should_attend:
                    corrections += 1
                    target = tensor([[should_attend]], dtype=uint8)
                    self.voting_heads[h].commit(similarity_input, target)

        return corrections

    def train_value_projection(
        self,
        tokens: list[Tensor],
        target_values: list[Tensor] | None = None,
    ) -> int:
        """
        Train value projection layer using GeneralizingProjection.

        If target_values is None, trains identity mapping (value = input).
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        if target_values is None:
            target_values = tokens  # Identity mapping

        target_values = [t.squeeze() if t.ndim > 1 else t for t in target_values]
        corrections = 0

        for tok, target in zip(tokens, target_values):
            # GeneralizingProjection uses train_mapping(input, output)
            corrections += self.value_projection.train_mapping(tok, target)

        return corrections

    def train_step(
        self,
        input_tokens: list[Tensor],
        target_tokens: list[Tensor],
    ) -> int:
        """
        Train both attention and value projection for sequence-to-sequence task.

        This learns:
        1. Value projection: input -> target (what each position should output)
        2. Attention weights: query should attend to positions that have the answer

        For simple copy task: attention is diagonal, values are identity.
        """
        input_tokens = [t.squeeze() if t.ndim > 1 else t for t in input_tokens]
        target_tokens = [t.squeeze() if t.ndim > 1 else t for t in target_tokens]
        corrections = 0

        # Train value projection: each input should project to its target
        corrections += self.train_value_projection(input_tokens, target_tokens)

        # For now, assume diagonal attention (copy task)
        # More sophisticated training would infer attention from input/target
        for pos in range(len(input_tokens)):
            target_weights = [0.0] * len(input_tokens)
            target_weights[pos] = 1.0  # Self-attention for copy
            corrections += self.train_attention_weights(input_tokens, pos, target_weights)

        return corrections


class SoftRAMAttentionV2(Module):
    """
    Alternative soft attention using bit-level voting.

    Instead of voting on attend/don't-attend, each head
    contributes a "partial" value, and we combine based
    on agreement.

    More heads agreeing on a bit = stronger signal.
    """

    def __init__(
        self,
        input_bits: int,
        num_heads: int = 8,
        position_mode: PositionMode = PositionMode.RELATIVE,
        max_seq_len: int = 16,
        causal: bool = True,
        rng: int | None = None,
    ):
        super().__init__()

        self.input_bits = input_bits
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.position_mode = position_mode

        # Position encoding
        if position_mode == PositionMode.RELATIVE:
            self.position_encoder = PositionEncoderFactory.create(
                PositionMode.RELATIVE,
                max_distance=max_seq_len - 1,
            )
            self.n_position_bits = self.position_encoder.n_bits
        else:
            self.position_encoder = None
            self.n_position_bits = 0

        # Each head produces a full value output (not binary attend)
        # Input: [query, key, position]
        head_input_bits = 2 * input_bits + self.n_position_bits

        self.value_heads = ModuleList([
            RAMLayer(
                total_input_bits=head_input_bits,
                num_neurons=input_bits,  # Each head outputs full value
                n_bits_per_neuron=min(head_input_bits, 12),
                rng=rng + i if rng else None,
            )
            for i in range(num_heads)
        ])

        # Majority voting: count 1s per bit position
        # Then threshold to get final output
        self.threshold = num_heads // 2

        print(f"[SoftRAMAttentionV2] heads={num_heads}, input={input_bits}b, "
              f"threshold={self.threshold}, causal={causal}")

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """
        Apply bit-level voting attention.

        For each output position:
        1. Each head produces a value for each (query, key) pair
        2. Aggregate across keys using XOR
        3. Aggregate across heads using majority voting
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        outputs = []

        for i in range(seq_len):
            query = tokens[i]

            # Collect outputs from all heads
            head_outputs = []

            for head in self.value_heads:
                # Aggregate across all attended positions
                head_result = zeros(self.input_bits, dtype=uint8)

                for j in range(seq_len):
                    if self.causal and j > i:
                        continue

                    key = tokens[j]

                    # Build input
                    parts = [query, key]
                    if self.position_mode == PositionMode.RELATIVE:
                        rel_dist = self.position_encoder.encode_relative(i, j)
                        parts.append(rel_dist)

                    head_input = cat(parts).unsqueeze(0)
                    value = head(head_input).squeeze()

                    # XOR aggregate (order-invariant)
                    head_result = head_result ^ value

                head_outputs.append(head_result)

            # Majority voting across heads (per bit)
            output = zeros(self.input_bits, dtype=uint8)
            for bit in range(self.input_bits):
                ones_count = sum(h[bit].item() for h in head_outputs)
                output[bit] = 1 if ones_count > self.threshold else 0

            outputs.append(output)

        return outputs

    def __repr__(self):
        return f"SoftRAMAttentionV2(heads={self.num_heads}, bits={self.input_bits})"
