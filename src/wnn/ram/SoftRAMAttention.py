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

from torch import Tensor, zeros, uint8, cat, tensor, randperm, manual_seed
from torch.nn import Module, ModuleList
from enum import IntEnum
import random


class ContentMatchMode(IntEnum):
    """Content-based attention matching modes."""
    NONE = 0           # No content matching (use learned attention)
    XOR_EQUAL = 1      # Attend if query == key (XOR is all zeros)
    HAMMING_1 = 2      # Attend if Hamming distance <= 1
    HAMMING_2 = 3      # Attend if Hamming distance <= 2
    LESS_THAN = 4      # Attend if key < query (for sorting: find smaller tokens)
    LESS_EQUAL = 5     # Attend if key <= query
    GREATER_THAN = 6   # Attend if key > query
    GREATER_EQUAL = 7  # Attend if key >= query


class AttentionCombineMode(IntEnum):
    """How to combine content matching with position patterns."""
    CONTENT_ONLY = 0    # Only use content matching (ignore position)
    POSITION_ONLY = 1   # Only use position patterns (ignore content)
    CONTENT_AND_POS = 2 # Attend if BOTH content matches AND position pattern matches
    CONTENT_OR_POS = 3  # Attend if EITHER content matches OR position pattern matches
    CONTENT_BIASED = 4  # Content match with position bias (closer positions get more votes)


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


def _bits_to_int(bits: Tensor) -> int:
    """Convert bit tensor to integer (MSB first)."""
    bits = bits.squeeze()
    val = 0
    for b in bits:
        val = val * 2 + int(b.item())
    return val


def _less_than(a: Tensor, b: Tensor) -> bool:
    """Check if a < b by comparing bit patterns as integers."""
    return _bits_to_int(a) < _bits_to_int(b)


def _less_equal(a: Tensor, b: Tensor) -> bool:
    """Check if a <= b by comparing bit patterns as integers."""
    return _bits_to_int(a) <= _bits_to_int(b)


def _greater_than(a: Tensor, b: Tensor) -> bool:
    """Check if a > b by comparing bit patterns as integers."""
    return _bits_to_int(a) > _bits_to_int(b)


def _greater_equal(a: Tensor, b: Tensor) -> bool:
    """Check if a >= b by comparing bit patterns as integers."""
    return _bits_to_int(a) >= _bits_to_int(b)


class EnsembleVotingHead(Module):
    """
    Ensemble of RAMs with diverse projections for attention voting.

    Each sub-RAM sees a different subset of input bits.
    Majority voting provides better generalization for boolean decisions.

    Key insight: An unseen (query, key, position) pattern might be
    "familiar" to some projections even if new to the full input space.
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

        # Default: each sub-RAM sees ~60% of input bits
        if bits_per_ram is None:
            bits_per_ram = max(4, int(input_bits * 0.6))
        self.bits_per_ram = min(bits_per_ram, input_bits)

        # Generate diverse random projections
        if rng is not None:
            manual_seed(rng)
            random.seed(rng)

        self.projections = []
        for i in range(num_sub_rams):
            perm = randperm(input_bits)[:self.bits_per_ram].sort().values
            self.projections.append(perm)

        # Create sub-RAMs
        self.sub_rams = ModuleList([
            RAMLayer(
                total_input_bits=self.bits_per_ram,
                num_neurons=1,  # Binary attend/don't decision
                n_bits_per_neuron=min(self.bits_per_ram, 8),
                rng=rng + i * 100 if rng else None,
            )
            for i in range(num_sub_rams)
        ])

    def _project(self, x: Tensor, projection: Tensor) -> Tensor:
        """Extract bits specified by projection."""
        x = x.squeeze()
        return x[projection]

    def forward(self, x: Tensor) -> Tensor:
        """Forward with majority voting across sub-RAMs."""
        x = x.squeeze()

        votes = 0
        for ram, proj in zip(self.sub_rams, self.projections):
            projected = self._project(x, proj).unsqueeze(0)
            out = ram(projected).squeeze()
            votes += out.item()

        # Majority vote: attend if more than half say yes
        attend = 1 if votes > self.num_sub_rams // 2 else 0
        return tensor([attend], dtype=uint8)

    def commit(self, x: Tensor, target: Tensor) -> None:
        """Train all sub-RAMs on the same target."""
        x = x.squeeze()
        target_val = target.squeeze()

        for ram, proj in zip(self.sub_rams, self.projections):
            projected = self._project(x, proj).unsqueeze(0)
            ram.commit(projected, target_val.unsqueeze(0))


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
        use_ensemble: bool = False,  # Use ensemble voting heads for better generalization
        ensemble_sub_rams: int = 4,  # Number of sub-RAMs per voting head
        position_only: bool = False,  # If True, attention ignores token content (only positions)
        content_match: ContentMatchMode = ContentMatchMode.NONE,  # Content-based matching mode
        attention_combine: AttentionCombineMode = AttentionCombineMode.CONTENT_ONLY,  # How to combine content+position
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
            use_ensemble: If True, each voting head uses ensemble projections for better generalization
            ensemble_sub_rams: Number of sub-RAMs per ensemble voting head
            position_only: If True, attention decision ignores token content (generalizes across tokens)
            content_match: Content matching mode (XOR_EQUAL, HAMMING_1, HAMMING_2) for direct matching
            attention_combine: How to combine content matching with position patterns:
                - CONTENT_ONLY: Only use content matching
                - POSITION_ONLY: Only use position patterns (learned or position_only)
                - CONTENT_AND_POS: Attend if BOTH content AND position match
                - CONTENT_OR_POS: Attend if EITHER content OR position matches
                - CONTENT_BIASED: Content match with position-based vote weighting
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
            self.n_position_bits = self.position_encoder.n_bits * 2  # query + key pos
        else:
            raise ValueError(f"Unsupported position_mode: {position_mode}")

        # Similarity input: [query, key, position] or [position] if position_only
        if position_only:
            # Position-only: attention ignores token content
            similarity_input_bits = self.n_position_bits
            if similarity_input_bits == 0:
                raise ValueError("position_only=True requires position_mode != NONE")
        else:
            # Standard: attention sees query, key, and position
            similarity_input_bits = 2 * input_bits + self.n_position_bits
        self.similarity_input_bits = similarity_input_bits

        # Voting heads: each decides attend/don't-attend
        if use_ensemble:
            # Use ensemble voting heads for better generalization
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
            # Standard single-RAM voting heads
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

        ensemble_str = f", ensemble={ensemble_sub_rams}x" if use_ensemble else ""
        pos_only_str = ", pos_only" if position_only else ""
        content_str = f", content={content_match.name}" if content_match != ContentMatchMode.NONE else ""
        combine_str = f", combine={attention_combine.name}" if attention_combine != AttentionCombineMode.CONTENT_ONLY else ""
        print(f"[SoftRAMAttention] heads={num_heads}, input={input_bits}b, "
              f"aggregation={aggregation.name}, value={value_strategy.name}{ensemble_str}{pos_only_str}{content_str}{combine_str}, causal={causal}")

    def _encode_weight(self, vote_count: int) -> Tensor:
        """Encode vote count as bits."""
        bits = zeros(self.weight_bits, dtype=uint8)
        for i in range(self.weight_bits - 1, -1, -1):
            bits[self.weight_bits - 1 - i] = (vote_count >> i) & 1
        return bits

    def _compute_content_match(self, query: Tensor, key: Tensor) -> bool:
        """Compute content-based match (XOR/Hamming/Comparison)."""
        if self.content_match == ContentMatchMode.XOR_EQUAL:
            return _xor_match(query, key)
        elif self.content_match == ContentMatchMode.HAMMING_1:
            return _hamming_match(query, key, threshold=1)
        elif self.content_match == ContentMatchMode.HAMMING_2:
            return _hamming_match(query, key, threshold=2)
        elif self.content_match == ContentMatchMode.LESS_THAN:
            return _less_than(key, query)  # key < query (find smaller keys)
        elif self.content_match == ContentMatchMode.LESS_EQUAL:
            return _less_equal(key, query)  # key <= query
        elif self.content_match == ContentMatchMode.GREATER_THAN:
            return _greater_than(key, query)  # key > query
        elif self.content_match == ContentMatchMode.GREATER_EQUAL:
            return _greater_equal(key, query)  # key >= query
        return False

    def _compute_position_votes(
        self,
        query: Tensor,
        key: Tensor,
        query_pos: int,
        key_pos: int,
    ) -> int:
        """Compute position-based votes from voting heads."""
        vote_count = 0

        for head in self.voting_heads:
            # Build similarity input
            if self.position_only:
                # Position-only: don't include token content
                parts = []
            else:
                # Standard: include query and key tokens
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

    def _compute_votes(
        self,
        query: Tensor,
        keys: list[Tensor],
        query_pos: int,
    ) -> list[int]:
        """
        Count votes for each key position.

        Combines content matching and position patterns based on attention_combine mode.

        Returns:
            votes: List of vote counts [0, num_heads] for each key
        """
        votes = []

        for j, key in enumerate(keys):
            # Causal mask
            if self.causal and j > query_pos:
                votes.append(0)
                continue

            # Compute content match if enabled
            content_matches = False
            if self.content_match != ContentMatchMode.NONE:
                content_matches = self._compute_content_match(query, key)

            # Compute position votes from voting heads
            position_votes = self._compute_position_votes(query, key, query_pos, j)

            # Combine based on attention_combine mode
            if self.attention_combine == AttentionCombineMode.CONTENT_ONLY:
                # Only use content matching
                if self.content_match != ContentMatchMode.NONE:
                    vote_count = self.num_heads if content_matches else 0
                else:
                    # Fallback to position if no content mode set
                    vote_count = position_votes

            elif self.attention_combine == AttentionCombineMode.POSITION_ONLY:
                # Only use position patterns (ignore content)
                vote_count = position_votes

            elif self.attention_combine == AttentionCombineMode.CONTENT_AND_POS:
                # Both must match: content AND position
                if self.content_match != ContentMatchMode.NONE:
                    # Content must match AND position votes must be positive
                    if content_matches and position_votes > 0:
                        vote_count = position_votes  # Use position votes as weight
                    else:
                        vote_count = 0
                else:
                    # No content mode, just use position
                    vote_count = position_votes

            elif self.attention_combine == AttentionCombineMode.CONTENT_OR_POS:
                # Either matches: content OR position
                if self.content_match != ContentMatchMode.NONE:
                    if content_matches:
                        # Content matches - full votes
                        vote_count = self.num_heads
                    elif position_votes > 0:
                        # Position matches - use position votes
                        vote_count = position_votes
                    else:
                        vote_count = 0
                else:
                    # No content mode, just use position
                    vote_count = position_votes

            elif self.attention_combine == AttentionCombineMode.CONTENT_BIASED:
                # Content match with position-based weighting
                # Closer positions get more votes
                if self.content_match != ContentMatchMode.NONE and content_matches:
                    # Base: content matches
                    distance = abs(query_pos - j)
                    # Bias: closer = more votes (linear decay)
                    max_dist = len(keys) - 1
                    if max_dist > 0:
                        bias = 1.0 - (distance / max_dist) * 0.5  # 1.0 for same pos, 0.5 for furthest
                    else:
                        bias = 1.0
                    vote_count = int(self.num_heads * bias)
                else:
                    vote_count = 0

            else:
                # Default fallback
                vote_count = position_votes

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
            if self.position_only:
                # Position-only: don't include token content
                parts = []
            else:
                # Standard: include query and key tokens
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


class SortingAttention(Module):
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

        print(f"[SortingAttention] input={input_bits}b, "
              f"order={'descending' if descending else 'ascending'}")

    def _count_smaller(self, token: Tensor, all_tokens: list[Tensor]) -> int:
        """Count how many tokens in the list are smaller than this token."""
        token_val = _bits_to_int(token)
        count = 0
        for other in all_tokens:
            other_val = _bits_to_int(other)
            if other_val < token_val:
                count += 1
        return count

    def _count_larger(self, token: Tensor, all_tokens: list[Tensor]) -> int:
        """Count how many tokens in the list are larger than this token."""
        token_val = _bits_to_int(token)
        count = 0
        for other in all_tokens:
            other_val = _bits_to_int(other)
            if other_val > token_val:
                count += 1
        return count

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        query_pos: int,
    ) -> list[float]:
        """
        Get attention weights for sorting.

        For output position query_pos, we want the (query_pos+1)th smallest token.
        Handles duplicates by using stable sorting (preserving original order for ties).
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        # Get values and create (value, original_index) pairs
        values = [_bits_to_int(t) for t in tokens]
        indexed = [(val, i) for i, val in enumerate(values)]

        # Sort by value (stable sort preserves order for ties)
        if self.descending:
            indexed.sort(key=lambda x: -x[0])
        else:
            indexed.sort(key=lambda x: x[0])

        # For output position query_pos, find which input position should go there
        target_input_pos = indexed[query_pos][1]

        # Create weights: 1.0 for the target position, 0.0 elsewhere
        weights = [0.0] * n
        weights[target_input_pos] = 1.0

        return weights

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """
        Sort the sequence.

        No learning required - attention is computed from token comparisons.
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        outputs = []

        for i in range(seq_len):
            weights = self.get_attention_weights(tokens, i)

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

        lines = [f"Sorting Attention ({'descending' if self.descending else 'ascending'}):"]
        lines.append("     " + " ".join(f"{j:4d}" for j in range(seq_len)))

        for i in range(seq_len):
            weights = self.get_attention_weights(tokens, i)
            row = f"{i:2d}: "
            for w in weights:
                row += f"{w:4.2f} "
            lines.append(row)

        return "\n".join(lines)


class MinMaxAttention(Module):
    """
    Find minimum or maximum token using computed comparisons.

    Every position outputs the min (or max) token from the sequence.
    Generalizes 100% because comparison is computed, not learned.
    """

    def __init__(
        self,
        input_bits: int,
        find_max: bool = False,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            find_max: If True, find maximum instead of minimum
            rng: Random seed (unused, for API compatibility)
        """
        super().__init__()

        self.input_bits = input_bits
        self.find_max = find_max

        print(f"[MinMaxAttention] input={input_bits}b, "
              f"mode={'max' if find_max else 'min'}")

    def get_attention_weights(self, tokens: list[Tensor]) -> list[float]:
        """
        Get attention weights for min/max finding.

        All positions attend to the min (or max) token.
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]

        # Find min or max value
        values = [_bits_to_int(t) for t in tokens]
        if self.find_max:
            target_val = max(values)
        else:
            target_val = min(values)

        # Attend to position(s) with target value
        weights = []
        for val in values:
            if val == target_val:
                weights.append(1.0)
            else:
                weights.append(0.0)

        return weights

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """
        Output min (or max) at every position.
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        weights = self.get_attention_weights(tokens)

        # Find the min/max token
        min_max_token = None
        for j, w in enumerate(weights):
            if w > 0:
                min_max_token = tokens[j].clone()
                break

        if min_max_token is None:
            min_max_token = zeros(self.input_bits, dtype=uint8)

        # Output same token at every position
        return [min_max_token.clone() for _ in range(seq_len)]
