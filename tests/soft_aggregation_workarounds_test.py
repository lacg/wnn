#!/usr/bin/env python3
"""
Soft Attention Aggregation Workarounds

The problem: We can express discrete attention weights via voting,
but we can't do weighted sums (multiplication is continuous).

Workarounds explored:
1. THRESHOLD: Only include values above vote threshold in XOR
2. TIERED: Different aggregation per weight tier (high/medium/low)
3. SELECTIVE: Top-K selection based on votes
4. WEIGHTED XOR: Weight-dependent bit masking before XOR
5. MAJORITY VOTING: Per-bit majority across weighted values
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMGeneralization import MapperStrategy, GeneralizingProjection
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode, PositionEncoderFactory
from torch import zeros, ones, uint8, Tensor, cat
from torch.nn import Module, ModuleList

print("=" * 70)
print("Soft Attention Aggregation Workarounds")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_char(c: str) -> Tensor:
    return decoder.encode(c).squeeze()

def decode_bits(bits: Tensor) -> str:
    return decoder.decode(bits.unsqueeze(0))

def decode_sequence(tokens: list) -> str:
    return ''.join(decode_bits(t) for t in tokens)


# =============================================================
# Aggregation Strategies
# =============================================================

def aggregate_threshold(values: list[Tensor], votes: list[int], threshold: int) -> Tensor:
    """
    THRESHOLD: Only XOR values with votes above threshold.

    Higher threshold = stricter selection = fewer values combined.
    """
    if not values:
        return zeros(values[0].shape[0] if values else 5, dtype=uint8)

    result = zeros(values[0].shape[0], dtype=uint8)
    included = 0

    for val, vote in zip(values, votes):
        if vote >= threshold:
            result = result ^ val
            included += 1

    # Fallback: if nothing included, use highest voted
    if included == 0:
        max_vote = max(votes)
        for val, vote in zip(values, votes):
            if vote == max_vote:
                return val.clone()

    return result


def aggregate_tiered(values: list[Tensor], votes: list[int], num_heads: int) -> Tensor:
    """
    TIERED: Split into high/medium/low tiers, aggregate each differently.

    - High (>66%): Include directly in XOR
    - Medium (33-66%): Include with bit mask (partial)
    - Low (<33%): Exclude
    """
    if not values:
        return zeros(5, dtype=uint8)

    bits = values[0].shape[0]
    high_threshold = int(num_heads * 0.66)
    medium_threshold = int(num_heads * 0.33)

    result = zeros(bits, dtype=uint8)

    for val, vote in zip(values, votes):
        if vote >= high_threshold:
            # High: full inclusion
            result = result ^ val
        elif vote >= medium_threshold:
            # Medium: only include upper half of bits
            masked = val.clone()
            masked[bits//2:] = 0  # Zero out lower bits
            result = result ^ masked
        # Low: exclude

    return result


def aggregate_top_k(values: list[Tensor], votes: list[int], k: int) -> Tensor:
    """
    TOP-K: Select K highest-voted values, XOR them.

    Approximates "attend to top K positions".
    """
    if not values:
        return zeros(5, dtype=uint8)

    # Sort by votes descending
    sorted_pairs = sorted(zip(votes, values), key=lambda x: -x[0])

    # Take top K
    result = zeros(values[0].shape[0], dtype=uint8)
    for i, (vote, val) in enumerate(sorted_pairs):
        if i >= k:
            break
        if vote > 0:  # Only include if any votes
            result = result ^ val

    return result


def aggregate_weighted_mask(values: list[Tensor], votes: list[int], num_heads: int) -> Tensor:
    """
    WEIGHTED MASK: Higher votes = more bits included.

    vote=8/8 → all bits included
    vote=4/8 → half bits included (upper half)
    vote=2/8 → quarter bits included (top quarter)
    """
    if not values:
        return zeros(5, dtype=uint8)

    bits = values[0].shape[0]
    result = zeros(bits, dtype=uint8)

    for val, vote in zip(values, votes):
        if vote == 0:
            continue

        # Calculate how many bits to include
        fraction = vote / num_heads
        bits_to_include = max(1, int(bits * fraction))

        # Create mask (include first N bits)
        masked = val.clone()
        masked[bits_to_include:] = 0

        result = result ^ masked

    return result


def aggregate_majority_voting(values: list[Tensor], votes: list[int], num_heads: int) -> Tensor:
    """
    MAJORITY VOTING: Per-bit majority, weighted by votes.

    For each bit position:
    - Count weighted 1s: sum(vote * bit_value)
    - Count weighted 0s: sum(vote * (1 - bit_value))
    - Output = 1 if weighted_1s > weighted_0s
    """
    if not values:
        return zeros(5, dtype=uint8)

    bits = values[0].shape[0]
    result = zeros(bits, dtype=uint8)

    for bit_pos in range(bits):
        weighted_ones = 0
        weighted_zeros = 0

        for val, vote in zip(values, votes):
            if val[bit_pos] == 1:
                weighted_ones += vote
            else:
                weighted_zeros += vote

        result[bit_pos] = 1 if weighted_ones > weighted_zeros else 0

    return result


# =============================================================
# Test Model with Selectable Aggregation
# =============================================================
class SoftAttentionWithAggregation(Module):
    """
    Soft attention with configurable aggregation strategy.
    """

    def __init__(
        self,
        input_bits: int,
        num_heads: int = 8,
        aggregation: str = "threshold",
        max_seq_len: int = 8,
        causal: bool = True,
        rng: int | None = None,
    ):
        super().__init__()

        self.input_bits = input_bits
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.max_seq_len = max_seq_len
        self.causal = causal

        # Position encoding
        self.position_encoder = PositionEncoderFactory.create(
            PositionMode.RELATIVE,
            max_distance=max_seq_len - 1,
        )
        self.n_position_bits = self.position_encoder.n_bits

        # Voting heads
        similarity_bits = 2 * input_bits + self.n_position_bits
        self.voting_heads = ModuleList([
            RAMLayer(
                total_input_bits=similarity_bits,
                num_neurons=1,
                n_bits_per_neuron=min(similarity_bits, 12),
                rng=rng + i if rng else None,
            )
            for i in range(num_heads)
        ])

        print(f"[SoftAttention] heads={num_heads}, aggregation={aggregation}")

    def _compute_votes(self, query: Tensor, keys: list[Tensor], query_pos: int) -> list[int]:
        votes = []
        for j, key in enumerate(keys):
            if self.causal and j > query_pos:
                votes.append(0)
                continue

            rel_dist = self.position_encoder.encode_relative(query_pos, j)
            similarity_input = cat([query, key, rel_dist]).unsqueeze(0)

            vote_count = sum(
                head(similarity_input).squeeze().item()
                for head in self.voting_heads
            )
            votes.append(vote_count)

        return votes

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        outputs = []

        for i, query in enumerate(tokens):
            votes = self._compute_votes(query, tokens, i)

            # Apply selected aggregation strategy
            if self.aggregation == "threshold":
                output = aggregate_threshold(tokens, votes, threshold=self.num_heads // 2)
            elif self.aggregation == "tiered":
                output = aggregate_tiered(tokens, votes, self.num_heads)
            elif self.aggregation == "top_k":
                output = aggregate_top_k(tokens, votes, k=2)
            elif self.aggregation == "weighted_mask":
                output = aggregate_weighted_mask(tokens, votes, self.num_heads)
            elif self.aggregation == "majority":
                output = aggregate_majority_voting(tokens, votes, self.num_heads)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

            outputs.append(output)

        return outputs

    def train_self_attention(self, tokens: list[Tensor]) -> int:
        """Train heads to attend to self (position i attends to position i)."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        corrections = 0

        for i, query in enumerate(tokens):
            for j, key in enumerate(tokens):
                if self.causal and j > i:
                    continue

                rel_dist = self.position_encoder.encode_relative(i, j)
                similarity_input = cat([query, key, rel_dist]).unsqueeze(0)

                # Self-attention: attend only to same position
                should_attend = 1 if i == j else 0

                for head in self.voting_heads:
                    current = head(similarity_input).squeeze().item()
                    if current != should_attend:
                        corrections += 1
                        head.commit(similarity_input, Tensor([[should_attend]]).to(uint8))

        return corrections


# =============================================================
# Test 1: Compare Aggregation Strategies on Copy Task
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Aggregation Strategies on Copy Task")
print("=" * 60)
print("Task: output[i] = input[i] (requires attending only to self)")

strategies = ["threshold", "tiered", "top_k", "weighted_mask", "majority"]

for strategy in strategies:
    print(f"\n--- {strategy.upper()} ---")

    model = SoftAttentionWithAggregation(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=strategy,
        causal=False,
        rng=42,
    )

    # Train self-attention pattern
    train_seqs = ["ABCD", "EFGH", "IJKL"]
    for epoch in range(5):
        for seq_str in train_seqs:
            tokens = [encode_char(c) for c in seq_str]
            model.train_self_attention(tokens)

    # Test
    test_seqs = ["ABCD", "MNOP", "WXYZ"]
    correct = 0
    total = 0

    for seq_str in test_seqs:
        tokens = [encode_char(c) for c in seq_str]
        outputs = model.forward(tokens)
        result = decode_sequence(outputs)

        for r, e in zip(result, seq_str):
            total += 1
            if r == e:
                correct += 1

        status = "OK" if result == seq_str else "X"
        print(f"  [{status}] '{seq_str}' -> '{result}'")

    print(f"  Accuracy: {100*correct/total:.0f}%")


# =============================================================
# Test 2: Multi-Position Attention (Last + Current)
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Attend to Current AND Previous Position")
print("=" * 60)
print("Task: output[i] = input[i] XOR input[i-1]")

class XORWithPrevious(Module):
    """Output = current XOR previous."""

    def __init__(self, bits, num_heads=8, aggregation="threshold", rng=None):
        super().__init__()
        self.bits = bits
        self.num_heads = num_heads
        self.aggregation = aggregation

        self.position_encoder = PositionEncoderFactory.create(
            PositionMode.RELATIVE,
            max_distance=8,
        )
        self.n_position_bits = self.position_encoder.n_bits

        similarity_bits = 2 * bits + self.n_position_bits
        self.voting_heads = ModuleList([
            RAMLayer(
                total_input_bits=similarity_bits,
                num_neurons=1,
                n_bits_per_neuron=min(similarity_bits, 12),
                rng=rng + i if rng else None,
            )
            for i in range(num_heads)
        ])

    def train_pattern(self, tokens: list[Tensor]) -> int:
        """Train to attend to current (i) and previous (i-1)."""
        tokens = [t.squeeze() for t in tokens]
        corrections = 0

        for i, query in enumerate(tokens):
            for j, key in enumerate(tokens):
                if j > i:  # Causal
                    continue

                rel_dist = self.position_encoder.encode_relative(i, j)
                sim_input = cat([query, key, rel_dist]).unsqueeze(0)

                # Attend to current (i==j) and previous (j==i-1)
                should_attend = 1 if (j == i or j == i - 1) else 0

                for head in self.voting_heads:
                    current = head(sim_input).squeeze().item()
                    if current != should_attend:
                        corrections += 1
                        head.commit(sim_input, Tensor([[should_attend]]).to(uint8))

        return corrections

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        tokens = [t.squeeze() for t in tokens]
        outputs = []

        for i, query in enumerate(tokens):
            votes = []
            for j, key in enumerate(tokens):
                if j > i:
                    votes.append(0)
                    continue

                rel_dist = self.position_encoder.encode_relative(i, j)
                sim_input = cat([query, key, rel_dist]).unsqueeze(0)

                vote_count = sum(
                    head(sim_input).squeeze().item()
                    for head in self.voting_heads
                )
                votes.append(vote_count)

            # Use threshold aggregation (XOR values with high votes)
            output = aggregate_threshold(tokens, votes, threshold=self.num_heads // 2)
            outputs.append(output)

        return outputs

print("\nTraining XOR with previous model...")
model = XORWithPrevious(bits=bits_per_token, num_heads=8, rng=42)

for epoch in range(5):
    for seq_str in ["ABCD", "EFGH", "IJKL"]:
        tokens = [encode_char(c) for c in seq_str]
        model.train_pattern(tokens)

# Test
print("\nResults:")
for seq_str in ["ABCD", "MNOP"]:
    tokens = [encode_char(c) for c in seq_str]
    outputs = model.forward(tokens)
    result = decode_sequence(outputs)

    # Expected: each position XORs with previous
    expected = []
    for i, c in enumerate(seq_str):
        if i == 0:
            expected.append(c)  # First has no previous
        else:
            # XOR current with previous
            curr_bits = encode_char(c)
            prev_bits = encode_char(seq_str[i-1])
            xor_bits = curr_bits ^ prev_bits
            expected.append(decode_bits(xor_bits))
    expected_str = ''.join(expected)

    status = "OK" if result == expected_str else "~"
    print(f"  [{status}] '{seq_str}' -> '{result}' (expected '{expected_str}')")


# =============================================================
# Test 3: Weighted Majority for "Averaging"
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Weighted Majority (Pseudo-Averaging)")
print("=" * 60)
print("Use majority voting to approximate weighted combination")

def test_majority_averaging():
    # Create values with known bit patterns
    val_a = encode_char('A')  # 00000
    val_b = encode_char('B')  # 00001
    val_c = encode_char('C')  # 00010
    val_d = encode_char('D')  # 00011

    values = [val_a, val_b, val_c, val_d]

    print(f"\nValues:")
    print(f"  A = {val_a.tolist()}")
    print(f"  B = {val_b.tolist()}")
    print(f"  C = {val_c.tolist()}")
    print(f"  D = {val_d.tolist()}")

    # Test different vote distributions
    test_cases = [
        ([8, 0, 0, 0], "100% A"),
        ([4, 4, 0, 0], "50% A, 50% B"),
        ([2, 2, 2, 2], "25% each"),
        ([6, 2, 0, 0], "75% A, 25% B"),
        ([0, 0, 4, 4], "50% C, 50% D"),
    ]

    print(f"\nMajority voting results:")
    for votes, desc in test_cases:
        result = aggregate_majority_voting(values, votes, num_heads=8)
        result_char = decode_bits(result)
        print(f"  {desc}: votes={votes} -> {result.tolist()} = '{result_char}'")

test_majority_averaging()


# =============================================================
# Test 4: Comparing All Strategies on Same Input
# =============================================================
print("\n" + "=" * 60)
print("Test 4: All Strategies on Same Vote Distribution")
print("=" * 60)

values = [encode_char(c) for c in "ABCD"]
votes = [6, 4, 2, 0]  # Descending votes

print(f"Values: A, B, C, D")
print(f"Votes:  {votes}")
print(f"(8 heads total)")

print("\nResults:")
print(f"  Threshold (>=4):    {decode_bits(aggregate_threshold(values, votes, threshold=4))}")
print(f"  Tiered:             {decode_bits(aggregate_tiered(values, votes, num_heads=8))}")
print(f"  Top-2:              {decode_bits(aggregate_top_k(values, votes, k=2))}")
print(f"  Weighted Mask:      {decode_bits(aggregate_weighted_mask(values, votes, num_heads=8))}")
print(f"  Majority Voting:    {decode_bits(aggregate_majority_voting(values, votes, num_heads=8))}")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Aggregation Workarounds")
print("=" * 60)

print("""
STRATEGY COMPARISON:

1. THRESHOLD
   - Simple: include values with votes >= threshold
   - Best for: selecting top contributors
   - Limitation: binary include/exclude

2. TIERED
   - Split into high/medium/low tiers
   - Different treatment per tier (full/partial/exclude)
   - More nuanced than threshold

3. TOP-K
   - Always select K highest-voted values
   - Predictable number of values combined
   - Good for "attend to top K positions"

4. WEIGHTED MASK
   - Higher votes = more bits included
   - Approximates weighted contribution
   - Per-value bit masking

5. MAJORITY VOTING
   - Per-bit majority weighted by votes
   - Closest to true weighted averaging!
   - Each bit decided independently

RECOMMENDATION:
  - For selection tasks: THRESHOLD or TOP-K
  - For combining values: MAJORITY VOTING
  - For nuanced control: TIERED or WEIGHTED MASK

KEY INSIGHT:
  Majority voting is the best discrete approximation of
  weighted averaging because it considers vote weights
  at the bit level, not the value level.
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
