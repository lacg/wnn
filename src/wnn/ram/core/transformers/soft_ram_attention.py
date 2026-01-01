"""
Soft RAM Attention via Voting

Soft RAM Attention approximates continuous attention weights using
discrete voting across multiple heads.

Key insight: With enough heads, voting approximates continuous weights.
  8 heads -> weights in {0, 0.125, 0.25, ..., 1.0}
  16 heads -> weights in {0, 0.0625, 0.125, ..., 1.0}

Aggregation Strategies:
  - TOP_1: Winner-take-all (best for retrieval)
  - MAJORITY: Per-bit weighted voting (best for combining)
  - THRESHOLD: Include only above 50% votes
  - TOP_K: XOR top K highest-voted values
"""

from wnn.ram.core.RAMLayer import RAMLayer
from wnn.ram.core.RAMGeneralization import GeneralizingProjection
from wnn.ram.core.transformers.attention_base import LearnableAttention
from wnn.ram.enums import (
    ContentMatchMode,
    AttentionCombineMode,
    AggregationStrategy,
    MapperStrategy,
)
from wnn.ram.encoders_decoders import PositionMode, PositionEncoderFactory
from wnn.ram.core.transformers.computed_arithmetic import bits_to_int

from torch import Tensor, zeros, uint8, cat, tensor, randperm, manual_seed, float32
from torch.nn import Module, ModuleList
import random


def _hamming_distance(a: Tensor, b: Tensor) -> int:
    """Count differing bits between two tensors."""
    return (a != b).sum().item()


def _xor_match(query: Tensor, key: Tensor) -> bool:
    """Check if query equals key using XOR (all zeros = equal)."""
    xor_result = query.squeeze() ^ key.squeeze()
    return (xor_result == 0).all().item()


def _hamming_match(query: Tensor, key: Tensor, threshold: int) -> bool:
    """Check if Hamming distance is within threshold."""
    dist = _hamming_distance(query.squeeze(), key.squeeze())
    return dist <= threshold


def _less_than(a: Tensor, b: Tensor) -> bool:
    """Check if a < b by comparing bit patterns as integers."""
    return bits_to_int(a) < bits_to_int(b)


def _less_equal(a: Tensor, b: Tensor) -> bool:
    """Check if a <= b by comparing bit patterns as integers."""
    return bits_to_int(a) <= bits_to_int(b)


def _greater_than(a: Tensor, b: Tensor) -> bool:
    """Check if a > b by comparing bit patterns as integers."""
    return bits_to_int(a) > bits_to_int(b)


def _greater_equal(a: Tensor, b: Tensor) -> bool:
    """Check if a >= b by comparing bit patterns as integers."""
    return bits_to_int(a) >= bits_to_int(b)


class EnsembleVotingHead(Module):
    """
    Ensemble of RAMs with diverse projections for attention voting.

    Each sub-RAM sees a different subset of input bits.
    Majority voting provides better generalization for boolean decisions.
    """

    def __init__(
        self,
        input_bits: int,
        num_sub_rams: int = 4,
        bits_per_ram: int | None = None,
        rng: int | None = None,
    ):
        super().__init__()

        self.input_bits = input_bits
        self.num_sub_rams = num_sub_rams

        if bits_per_ram is None:
            bits_per_ram = max(4, int(input_bits * 0.6))
        self.bits_per_ram = min(bits_per_ram, input_bits)

        if rng is not None:
            manual_seed(rng)
            random.seed(rng)

        self.projections = []
        for i in range(num_sub_rams):
            perm = randperm(input_bits)[:self.bits_per_ram].sort().values
            self.projections.append(perm)

        self.sub_rams = ModuleList([
            RAMLayer(
                total_input_bits=self.bits_per_ram,
                num_neurons=1,
                n_bits_per_neuron=min(self.bits_per_ram, 8),
                rng=rng + i * 100 if rng else None,
            )
            for i in range(num_sub_rams)
        ])

    def _project(self, x: Tensor, projection: Tensor) -> Tensor:
        x = x.squeeze()
        return x[projection]

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze()
        votes = 0
        for ram, proj in zip(self.sub_rams, self.projections):
            projected = self._project(x, proj).unsqueeze(0)
            out = ram(projected).squeeze()
            votes += out.item()
        attend = 1 if votes > self.num_sub_rams // 2 else 0
        return tensor([attend], dtype=uint8)

    def commit(self, x: Tensor, target: Tensor) -> None:
        x = x.squeeze()
        target_val = target.squeeze()
        for ram, proj in zip(self.sub_rams, self.projections):
            projected = self._project(x, proj).unsqueeze(0)
            ram.commit(projected, target_val.unsqueeze(0))


class SoftRAMAttention(LearnableAttention):
    """
    Soft attention approximation using head voting.

    Each head votes on which positions to attend.
    Vote counts create pseudo-continuous attention weights.

    Recommended Configurations:
        Simple (position-based):
            SoftRAMAttention(input_bits=8, num_heads=8)

        Content matching (for retrieval):
            SoftRAMAttention(input_bits=8, num_heads=8,
                           content_match=ContentMatchMode.XOR_EQUAL)

        Position-only (100% generalization):
            SoftRAMAttention(input_bits=8, num_heads=8, position_only=True)
    """

    def __init__(
        self,
        input_bits: int,
        num_heads: int = 8,
        aggregation: AggregationStrategy = AggregationStrategy.TOP_1,
        value_strategy: MapperStrategy = MapperStrategy.BIT_LEVEL,
        position_mode: PositionMode = PositionMode.RELATIVE,
        max_seq_len: int = 16,
        causal: bool = True,
        top_k: int = 2,
        use_ensemble: bool = False,
        ensemble_sub_rams: int = 4,
        position_only: bool = False,
        content_match: ContentMatchMode = ContentMatchMode.NONE,
        attention_combine: AttentionCombineMode = AttentionCombineMode.CONTENT_ONLY,
        rng: int | None = None,
    ):
        super().__init__()

        self.input_bits = input_bits
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.value_strategy = value_strategy
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.position_mode = position_mode
        self.use_ensemble = use_ensemble
        self.ensemble_sub_rams = ensemble_sub_rams
        self.position_only = position_only
        self.content_match = content_match
        self.attention_combine = attention_combine

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
            self.n_position_bits = self.position_encoder.n_bits * 2
        else:
            raise ValueError(f"Unsupported position_mode: {position_mode}")

        # Similarity input size
        if position_only:
            similarity_input_bits = self.n_position_bits
            if similarity_input_bits == 0:
                raise ValueError("position_only=True requires position_mode != NONE")
        else:
            similarity_input_bits = 2 * input_bits + self.n_position_bits
        self.similarity_input_bits = similarity_input_bits

        # Voting heads
        if use_ensemble:
            self.voting_heads = ModuleList([
                EnsembleVotingHead(
                    input_bits=similarity_input_bits,
                    num_sub_rams=ensemble_sub_rams,
                    bits_per_ram=max(4, int(similarity_input_bits * 0.6)),
                    rng=rng + i * 1000 if rng else None,
                )
                for i in range(num_heads)
            ])
        else:
            self.voting_heads = ModuleList([
                RAMLayer(
                    total_input_bits=similarity_input_bits,
                    num_neurons=1,
                    n_bits_per_neuron=min(similarity_input_bits, 12),
                    rng=rng + i if rng else None,
                )
                for i in range(num_heads)
            ])

        # Value projection
        self.value_projection = GeneralizingProjection(
            input_bits=input_bits,
            output_bits=input_bits,
            strategy=value_strategy,
            rng=rng + num_heads if rng else None,
        )

        # Weight aggregation
        self.weight_levels = num_heads + 1
        weight_bits = self.weight_levels.bit_length()
        self.weight_aggregator = RAMLayer(
            total_input_bits=input_bits + weight_bits,
            num_neurons=input_bits,
            n_bits_per_neuron=min(input_bits + weight_bits, 12),
            rng=rng + num_heads + 1 if rng else None,
        )
        self.weight_bits = weight_bits

        # Final output
        self.output_layer = RAMLayer(
            total_input_bits=input_bits * max_seq_len,
            num_neurons=input_bits,
            n_bits_per_neuron=min(input_bits * 2, 12),
            rng=rng + num_heads + 2 if rng else None,
        )

        ensemble_str = f", ensemble={ensemble_sub_rams}x" if use_ensemble else ""
        pos_only_str = ", pos_only" if position_only else ""
        content_str = f", content={content_match.name}" if content_match != ContentMatchMode.NONE else ""
        combine_str = f", combine={attention_combine.name}" if attention_combine != AttentionCombineMode.CONTENT_ONLY else ""
        print(f"[SoftRAMAttention] heads={num_heads}, input={input_bits}b, "
              f"aggregation={aggregation.name}, value={value_strategy.name}{ensemble_str}{pos_only_str}{content_str}{combine_str}, causal={causal}")

    def _encode_weight(self, vote_count: int) -> Tensor:
        bits = zeros(self.weight_bits, dtype=uint8)
        for i in range(self.weight_bits - 1, -1, -1):
            bits[self.weight_bits - 1 - i] = (vote_count >> i) & 1
        return bits

    def _compute_content_match(self, query: Tensor, key: Tensor) -> bool:
        """Check if query and key match based on content_match mode."""
        match self.content_match:
            case ContentMatchMode.XOR_EQUAL:
                return _xor_match(query, key)
            case ContentMatchMode.HAMMING_1:
                return _hamming_match(query, key, threshold=1)
            case ContentMatchMode.HAMMING_2:
                return _hamming_match(query, key, threshold=2)
            case ContentMatchMode.LESS_THAN:
                return _less_than(key, query)
            case ContentMatchMode.LESS_EQUAL:
                return _less_equal(key, query)
            case ContentMatchMode.GREATER_THAN:
                return _greater_than(key, query)
            case ContentMatchMode.GREATER_EQUAL:
                return _greater_equal(key, query)
            case _:
                return False

    def _compute_position_votes(self, query: Tensor, key: Tensor, query_pos: int, key_pos: int) -> int:
        vote_count = 0
        for head in self.voting_heads:
            if self.position_only:
                parts = []
            else:
                parts = [query, key]

            if self.position_mode == PositionMode.RELATIVE:
                rel_dist = self.position_encoder.encode_relative(query_pos, key_pos)
                parts.append(rel_dist)
            elif self.position_mode == PositionMode.BINARY:
                q_pos = self.position_encoder.encode(query_pos)
                k_pos = self.position_encoder.encode(key_pos)
                parts.extend([q_pos, k_pos])

            similarity_input = cat(parts).unsqueeze(0)
            attend = head(similarity_input).squeeze().item()
            vote_count += attend
        return vote_count

    def _combine_votes(
        self,
        content_matches: bool,
        position_votes: int,
        query_pos: int,
        key_pos: int,
        num_keys: int,
    ) -> int:
        """Combine content matching and position votes based on combine mode."""
        match self.attention_combine:
            case AttentionCombineMode.CONTENT_ONLY:
                if self.content_match != ContentMatchMode.NONE:
                    return self.num_heads if content_matches else 0
                return position_votes

            case AttentionCombineMode.POSITION_ONLY:
                return position_votes

            case AttentionCombineMode.CONTENT_AND_POS:
                if self.content_match != ContentMatchMode.NONE:
                    return position_votes if content_matches and position_votes > 0 else 0
                return position_votes

            case AttentionCombineMode.CONTENT_OR_POS:
                if self.content_match != ContentMatchMode.NONE:
                    if content_matches:
                        return self.num_heads
                    return position_votes if position_votes > 0 else 0
                return position_votes

            case AttentionCombineMode.CONTENT_BIASED:
                if self.content_match != ContentMatchMode.NONE and content_matches:
                    distance = abs(query_pos - key_pos)
                    max_dist = num_keys - 1
                    bias = 1.0 - (distance / max_dist) * 0.5 if max_dist > 0 else 1.0
                    return int(self.num_heads * bias)
                return 0

            case _:
                return position_votes

    def _compute_votes(self, query: Tensor, keys: list[Tensor], query_pos: int) -> list[int]:
        """Compute vote counts for each key position."""
        votes = []
        for j, key in enumerate(keys):
            # Causal mask
            if self.causal and j > query_pos:
                votes.append(0)
                continue

            # Content matching (if enabled)
            content_matches = False
            if self.content_match != ContentMatchMode.NONE:
                content_matches = self._compute_content_match(query, key)

            # Position-based votes
            position_votes = self._compute_position_votes(query, key, query_pos, j)

            # Combine content and position
            vote_count = self._combine_votes(
                content_matches, position_votes, query_pos, j, len(keys)
            )
            votes.append(vote_count)

        return votes

    def _aggregate_top_1(self, values: list[Tensor], votes: list[int]) -> Tensor:
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
        if not values:
            return zeros(self.input_bits, dtype=uint8)
        result = zeros(self.input_bits, dtype=uint8)
        for bit_pos in range(self.input_bits):
            weighted_ones = sum(vote for val, vote in zip(values, votes) if val[bit_pos] == 1)
            weighted_zeros = sum(vote for val, vote in zip(values, votes) if val[bit_pos] == 0)
            result[bit_pos] = 1 if weighted_ones > weighted_zeros else 0
        return result

    def _aggregate_threshold(self, values: list[Tensor], votes: list[int]) -> Tensor:
        if not values:
            return zeros(self.input_bits, dtype=uint8)
        threshold = self.num_heads // 2
        result = zeros(self.input_bits, dtype=uint8)
        included = 0
        for val, vote in zip(values, votes):
            if vote >= threshold:
                result = result ^ val
                included += 1
        if included == 0:
            return self._aggregate_top_1(values, votes)
        return result

    def _aggregate_top_k(self, values: list[Tensor], votes: list[int]) -> Tensor:
        if not values:
            return zeros(self.input_bits, dtype=uint8)
        sorted_pairs = sorted(zip(votes, values), key=lambda x: -x[0])
        result = zeros(self.input_bits, dtype=uint8)
        for i, (vote, val) in enumerate(sorted_pairs[:self.top_k]):
            if vote > 0:
                result = result ^ val
        return result

    def _aggregate(self, values: list[Tensor], votes: list[int]) -> Tensor:
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

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """
        Forward pass (implements LearnableAttention interface).

        Args:
            tokens: Input tokens (queries)
            context: Ignored (SoftRAMAttention is self-attention only)
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        # Project values
        values = []
        for tok in tokens:
            proj = self.value_projection(tok)
            if proj.ndim > 1:
                proj = proj.squeeze()
            values.append(proj)

        # Compute outputs
        outputs = []
        for i in range(seq_len):
            query = tokens[i]
            votes = self._compute_votes(query, tokens, i)
            output = self._aggregate(values, votes)
            outputs.append(output)
        return outputs

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """
        Get attention weights matrix (implements LearnableAttention interface).

        Returns:
            Tensor [num_queries, num_keys] with normalized vote weights
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        weights = zeros(n, n, dtype=float32)
        for i in range(n):
            query = tokens[i]
            votes = self._compute_votes(query, tokens, i)
            for j, v in enumerate(votes):
                weights[i, j] = v / self.num_heads

        return weights

    def _get_weights_for_position(self, tokens: list[Tensor], query_pos: int) -> list[float]:
        """Get attention weights for a single query position (internal helper)."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        query = tokens[query_pos]
        votes = self._compute_votes(query, tokens, query_pos)
        return [v / self.num_heads for v in votes]

    def train_step(
        self,
        tokens: list[Tensor],
        targets: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> int:
        """
        Single training step (implements LearnableAttention interface).

        Trains the attention mechanism to produce target outputs.

        Args:
            tokens: Input tokens
            targets: Target outputs
            context: Ignored (self-attention only)

        Returns:
            Number of updates made
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        targets = [t.squeeze() if t.ndim > 1 else t for t in targets]

        # Simple training: just train value projection
        corrections = 0
        for tok, target in zip(tokens, targets):
            corrections += self.value_projection.train_mapping(tok, target)

        return corrections

    def train_attention_weights(self, tokens: list[Tensor], query_pos: int, target_weights: list[float]) -> int:
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        query = tokens[query_pos]
        corrections = 0
        for j, (key, target_w) in enumerate(zip(tokens, target_weights)):
            if self.causal and j > query_pos:
                continue
            target_votes = int(target_w * self.num_heads + 0.5)
            target_votes = max(0, min(self.num_heads, target_votes))
            if self.position_only:
                parts = []
            else:
                parts = [query, key]
            if self.position_mode == PositionMode.RELATIVE:
                rel_dist = self.position_encoder.encode_relative(query_pos, j)
                parts.append(rel_dist)
            elif self.position_mode == PositionMode.BINARY:
                q_pos = self.position_encoder.encode(query_pos)
                k_pos = self.position_encoder.encode(j)
                parts.extend([q_pos, k_pos])
            similarity_input = cat(parts).unsqueeze(0)
            for h in range(self.num_heads):
                should_attend = 1 if h < target_votes else 0
                current = self.voting_heads[h](similarity_input).squeeze().item()
                if current != should_attend:
                    corrections += 1
                    target = tensor([[should_attend]], dtype=uint8)
                    self.voting_heads[h].commit(similarity_input, target)
        return corrections

    def train_value_projection(self, tokens: list[Tensor], target_values: list[Tensor] | None = None) -> int:
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        if target_values is None:
            target_values = tokens
        target_values = [t.squeeze() if t.ndim > 1 else t for t in target_values]
        corrections = 0
        for tok, target in zip(tokens, target_values):
            corrections += self.value_projection.train_mapping(tok, target)
        return corrections


class SoftRAMAttentionV2(Module):
    """Alternative soft attention using bit-level voting."""

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

        if position_mode == PositionMode.RELATIVE:
            self.position_encoder = PositionEncoderFactory.create(
                PositionMode.RELATIVE,
                max_distance=max_seq_len - 1,
            )
            self.n_position_bits = self.position_encoder.n_bits
        else:
            self.position_encoder = None
            self.n_position_bits = 0

        head_input_bits = 2 * input_bits + self.n_position_bits
        self.value_heads = ModuleList([
            RAMLayer(
                total_input_bits=head_input_bits,
                num_neurons=input_bits,
                n_bits_per_neuron=min(head_input_bits, 12),
                rng=rng + i if rng else None,
            )
            for i in range(num_heads)
        ])
        self.threshold = num_heads // 2
        print(f"[SoftRAMAttentionV2] heads={num_heads}, input={input_bits}b, "
              f"threshold={self.threshold}, causal={causal}")

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)
        outputs = []
        for i in range(seq_len):
            query = tokens[i]
            head_outputs = []
            for head in self.value_heads:
                head_result = zeros(self.input_bits, dtype=uint8)
                for j in range(seq_len):
                    if self.causal and j > i:
                        continue
                    key = tokens[j]
                    parts = [query, key]
                    if self.position_mode == PositionMode.RELATIVE:
                        rel_dist = self.position_encoder.encode_relative(i, j)
                        parts.append(rel_dist)
                    head_input = cat(parts).unsqueeze(0)
                    value = head(head_input).squeeze()
                    head_result = head_result ^ value
                head_outputs.append(head_result)
            output = zeros(self.input_bits, dtype=uint8)
            for bit in range(self.input_bits):
                ones_count = sum(h[bit].item() for h in head_outputs)
                output[bit] = 1 if ones_count > self.threshold else 0
            outputs.append(output)
        return outputs


__all__ = [
    'SoftRAMAttention',
    'SoftRAMAttentionV2',
    'EnsembleVotingHead',
]
