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
  - TOP_K: XOR top K highest-voted values
"""

from wnn.ram.core.RAMLayer import RAMLayer
from wnn.ram.core.RAMGeneralization import GeneralizingProjection
from wnn.ram.core.models.attention_base import LearnableAttention
from wnn.ram.core.models.attention_mask import AttentionMask, MaskStrategy
from wnn.ram.enums import (
    ContentMatchMode,
    AttentionCombineMode,
    AggregationStrategy,
    MapperStrategy,
    CrossAttentionMode,
)
from wnn.ram.encoders_decoders import PositionMode, PositionEncoderFactory

from torch import Tensor, zeros, uint8, cat, tensor, randperm, manual_seed, float32
from torch.nn import Module, ModuleList
import random


def _xor_match(query: Tensor, key: Tensor) -> bool:
    """Check if query equals key using XOR (all zeros = equal)."""
    xor_result = query.squeeze() ^ key.squeeze()
    return (xor_result == 0).all().item()


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
        key_bits: int | None = None,  # None = self-attention, else cross-attention
        num_heads: int = 8,
        aggregation: AggregationStrategy = AggregationStrategy.TOP_1,
        value_strategy: MapperStrategy = MapperStrategy.BIT_LEVEL,
        position_mode: PositionMode = PositionMode.RELATIVE,
        cross_attention_mode: CrossAttentionMode = CrossAttentionMode.ENCODER_ONLY,
        max_seq_len: int = 16,
        max_context_len: int | None = None,  # For cross-attention
        causal: bool = True,
        mask_strategy: MaskStrategy | None = None,  # Override causal
        window_size: int = 3,  # For SLIDING_WINDOW
        block_size: int = 4,  # For BLOCK
        prefix_len: int = 2,  # For PREFIX
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
        self.key_bits = key_bits or input_bits  # Default to self-attention
        self.is_cross_attention = key_bits is not None
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.value_strategy = value_strategy
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.max_context_len = max_context_len or max_seq_len
        self.position_mode = position_mode
        self.cross_attention_mode = cross_attention_mode
        self.use_ensemble = use_ensemble
        self.ensemble_sub_rams = ensemble_sub_rams
        self.position_only = position_only
        self.content_match = content_match
        self.attention_combine = attention_combine

        # Mask configuration
        self.window_size = window_size
        self.block_size = block_size
        self.prefix_len = prefix_len

        # Determine mask strategy
        # Cross-attention is always bidirectional (no causal masking)
        if self.is_cross_attention:
            self.mask_strategy = MaskStrategy.BIDIRECTIONAL
            self.causal = False
        elif mask_strategy is not None:
            self.mask_strategy = mask_strategy
            self.causal = self.mask_strategy == MaskStrategy.CAUSAL
        else:
            self.mask_strategy = MaskStrategy.CAUSAL if causal else MaskStrategy.BIDIRECTIONAL
            self.causal = causal

        # Position encoding setup
        match position_mode:
            case PositionMode.NONE:
                self.position_encoder = None
                self.n_position_bits = 0

            case PositionMode.RELATIVE:
                self.position_encoder = PositionEncoderFactory.create(
                    PositionMode.RELATIVE,
                    max_distance=max_seq_len - 1,
                )
                self.n_position_bits = self.position_encoder.n_bits

            case PositionMode.BINARY:
                self.position_encoder = PositionEncoderFactory.create(
                    PositionMode.BINARY,
                    max_seq_len=max_seq_len,
                )
                self.n_position_bits = self.position_encoder.n_bits * 2

            case PositionMode.LEARNED:
                # Learned position embeddings via RAMLayer
                n_base_bits = max(4, max_seq_len.bit_length())
                self.position_encoder = PositionEncoderFactory.create(
                    PositionMode.LEARNED,
                    n_position_bits=n_base_bits,
                    max_seq_len=max_seq_len,
                    rng=rng + 5000 if rng else None,
                )
                # Enable relative position learning
                self.position_encoder.enable_relative(
                    max_distance=max_seq_len - 1,
                    rng=rng + 5001 if rng else None,
                )
                self.n_position_bits = self.position_encoder.n_bits

            case _:
                raise ValueError(f"Unsupported position_mode: {position_mode}")

        # Similarity input size: [query, key, position]
        if position_only:
            similarity_input_bits = self.n_position_bits
            if similarity_input_bits == 0:
                raise ValueError("position_only=True requires position_mode != NONE")
        else:
            # Cross-attention: query_bits + key_bits
            # Self-attention: input_bits + input_bits
            similarity_input_bits = input_bits + self.key_bits + self.n_position_bits
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

        # Value projection: transforms context values (key_bits) to query space (input_bits)
        # For cross-attention with different dimensions, use DIRECT strategy
        effective_value_strategy = value_strategy
        if self.is_cross_attention and self.key_bits != input_bits:
            # BIT_LEVEL/RESIDUAL require same input/output dims
            if value_strategy in (MapperStrategy.BIT_LEVEL, MapperStrategy.RESIDUAL):
                effective_value_strategy = MapperStrategy.DIRECT

        self.value_projection = GeneralizingProjection(
            input_bits=self.key_bits,  # Values come from context
            output_bits=input_bits,    # Output in query space
            strategy=effective_value_strategy,
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
        cross_str = f", cross={self.key_bits}b" if self.is_cross_attention else ""
        mask_name = self.mask_strategy.name
        print(f"[SoftRAMAttention] heads={num_heads}, input={input_bits}b{cross_str}, "
              f"aggregation={aggregation.name}, value={value_strategy.name}{ensemble_str}{pos_only_str}{content_str}{combine_str}, mask={mask_name}")

    def _get_mask(self, seq_len: int, key_len: int | None = None) -> Tensor:
        """Get attention mask for the given sequence/key lengths."""
        key_len = key_len or seq_len
        return AttentionMask.from_strategy(
            strategy=self.mask_strategy,
            seq_len=seq_len,
            key_len=key_len,
            window_size=self.window_size,
            block_size=self.block_size,
            prefix_len=self.prefix_len,
        )

    def _can_attend(self, mask: Tensor, query_pos: int, key_pos: int) -> bool:
        """Check if query position can attend to key position."""
        return bool(mask[query_pos, key_pos].item())

    def get_mask(self, seq_len: int) -> Tensor:
        """Public method to get attention mask."""
        return self._get_mask(seq_len)

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
            case _:
                return False

    def _compute_position_votes(self, query: Tensor, key: Tensor, query_pos: int, key_pos: int) -> int:
        vote_count = 0
        for head in self.voting_heads:
            parts = [] if self.position_only else [query, key]

            match self.position_mode:
                case PositionMode.RELATIVE:
                    rel_dist = self.position_encoder.encode_relative(query_pos, key_pos)
                    parts.append(rel_dist)

                case PositionMode.BINARY:
                    q_pos = self.position_encoder.encode(query_pos)
                    k_pos = self.position_encoder.encode(key_pos)
                    parts.extend([q_pos, k_pos])

                case PositionMode.LEARNED:
                    # Use learned relative position encoding
                    rel_embed = self.position_encoder.encode_relative(query_pos, key_pos)
                    parts.append(rel_embed)

                case PositionMode.NONE:
                    pass  # No position encoding

            similarity_input = cat(parts).unsqueeze(0)
            attend = head(similarity_input).squeeze().item()
            vote_count += attend
        return vote_count

    def _combine_votes(
        self,
        content_matches: bool,
        position_votes: int,
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

            case _:
                return position_votes

    def _compute_votes(self, query: Tensor, keys: list[Tensor], query_pos: int, mask: Tensor | None = None) -> list[int]:
        """Compute vote counts for each key position."""
        # Get mask if not provided
        if mask is None:
            mask = self._get_mask(len(keys))

        votes = []
        for j, key in enumerate(keys):
            # Check mask instead of inline causal check
            if not self._can_attend(mask, query_pos, j):
                votes.append(0)
                continue

            # Content matching (if enabled)
            content_matches = False
            if self.content_match != ContentMatchMode.NONE:
                content_matches = self._compute_content_match(query, key)

            # Position-based votes
            position_votes = self._compute_position_votes(query, key, query_pos, j)

            # Combine content and position
            vote_count = self._combine_votes(content_matches, position_votes)
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
        match self.aggregation:
            case AggregationStrategy.TOP_1:
                return self._aggregate_top_1(values, votes)
            case AggregationStrategy.MAJORITY:
                return self._aggregate_majority(values, votes)
            case AggregationStrategy.TOP_K:
                return self._aggregate_top_k(values, votes)
            case _:
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
            context: Context tokens for cross-attention (keys/values).
                     None for self-attention.
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        # Determine keys: context for cross-attention, tokens for self-attention
        if context is not None:
            keys = [t.squeeze() if t.ndim > 1 else t for t in context]
            key_len = len(keys)
        else:
            keys = tokens
            key_len = seq_len

        # Pre-compute mask once
        mask = self._get_mask(seq_len, key_len)

        # Project values (from keys, not queries)
        values = []
        for key in keys:
            proj = self.value_projection(key)
            if proj.ndim > 1:
                proj = proj.squeeze()
            values.append(proj)

        # Compute outputs
        outputs = []
        for i in range(seq_len):
            query = tokens[i]
            votes = self._compute_votes(query, keys, i, mask=mask)
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

        # Determine keys: context for cross-attention, tokens for self-attention
        if context is not None:
            keys = [t.squeeze() if t.ndim > 1 else t for t in context]
            m = len(keys)
        else:
            keys = tokens
            m = n

        # Pre-compute mask once
        mask = self._get_mask(n, m)

        weights = zeros(n, m, dtype=float32)
        for i in range(n):
            query = tokens[i]
            votes = self._compute_votes(query, keys, i, mask=mask)
            for j, v in enumerate(votes):
                weights[i, j] = v / self.num_heads

        return weights

    def _get_weights_for_position(self, tokens: list[Tensor], query_pos: int, mask: Tensor | None = None) -> list[float]:
        """Get attention weights for a single query position (internal helper)."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        if mask is None:
            mask = self._get_mask(len(tokens))
        query = tokens[query_pos]
        votes = self._compute_votes(query, tokens, query_pos, mask=mask)
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

        # Pre-compute mask
        mask = self._get_mask(len(tokens))

        corrections = 0
        for j, (key, target_w) in enumerate(zip(tokens, target_weights)):
            # Use mask instead of inline causal check
            if not self._can_attend(mask, query_pos, j):
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
        mask_strategy: MaskStrategy | None = None,
        rng: int | None = None,
    ):
        super().__init__()
        self.input_bits = input_bits
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.position_mode = position_mode

        # Mask configuration
        if mask_strategy is not None:
            self.mask_strategy = mask_strategy
        else:
            self.mask_strategy = MaskStrategy.CAUSAL if causal else MaskStrategy.BIDIRECTIONAL

        # Keep causal for backwards compatibility
        self.causal = self.mask_strategy == MaskStrategy.CAUSAL

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
        mask_name = self.mask_strategy.name
        print(f"[SoftRAMAttentionV2] heads={num_heads}, input={input_bits}b, "
              f"threshold={self.threshold}, mask={mask_name}")

    def _get_mask(self, seq_len: int) -> Tensor:
        """Get attention mask for the given sequence length."""
        return AttentionMask.from_strategy(
            strategy=self.mask_strategy,
            seq_len=seq_len,
        )

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        seq_len = len(tokens)

        # Pre-compute mask
        mask = self._get_mask(seq_len)

        outputs = []
        for i in range(seq_len):
            query = tokens[i]
            head_outputs = []
            for head in self.value_heads:
                head_result = zeros(self.input_bits, dtype=uint8)
                for j in range(seq_len):
                    # Use mask instead of inline causal check
                    if not mask[i, j]:
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
