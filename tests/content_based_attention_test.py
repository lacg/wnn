#!/usr/bin/env python3
"""
Content-Based Attention Test

Explore attention patterns that depend on token CONTENT rather than position.

Examples of content-based attention:
1. Self-match: attend to tokens that match the query
2. Similarity: attend to tokens similar to query (low hamming distance)
3. Pattern match: attend to specific token patterns
4. Key-value lookup: query token determines which key to attend to

Challenge: RAM networks need to cover (query, key) combinations.
Can we find content patterns that generalize?
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.SoftRAMAttention import SoftRAMAttention, AggregationStrategy
from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import Tensor, zeros, uint8, cat
from torch.nn import Module

print("=" * 70)
print("Content-Based Attention Test")
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

def hamming_distance(a: Tensor, b: Tensor) -> int:
    """Count differing bits."""
    return (a != b).sum().item()


# =============================================================
# Test 1: Self-Match Attention (attend to identical tokens)
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Self-Match Attention")
print("=" * 60)
print("Attend to positions with identical token content")
print("E.g., 'ABAB' query@0 should attend to positions 0,2 (both 'A')")

class SelfMatchAttention(Module):
    """
    Content-based attention: attend to tokens matching the query.

    This is a simple form of content-based attention that can
    potentially generalize if we learn the "equality" pattern.
    """

    def __init__(self, input_bits: int, rng: int = None):
        super().__init__()
        self.input_bits = input_bits

        # Learn: given (query, key), should we attend?
        # Input: 2 * input_bits, Output: 1 bit (attend/don't)
        self.match_detector = RAMLayer(
            total_input_bits=2 * input_bits,
            num_neurons=1,
            n_bits_per_neuron=min(2 * input_bits, 10),
            rng=rng,
        )

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """For each position, aggregate tokens that match the query."""
        tokens = [t.squeeze() for t in tokens]
        outputs = []

        for i, query in enumerate(tokens):
            # Find matching tokens
            matches = []
            for j, key in enumerate(tokens):
                inp = cat([query, key]).unsqueeze(0)
                match = self.match_detector(inp).squeeze().item()
                if match:
                    matches.append(key)

            # Output: XOR of all matches (or zeros if none)
            if matches:
                result = matches[0].clone()
                for m in matches[1:]:
                    result = result ^ m
            else:
                result = zeros(self.input_bits, dtype=uint8)

            outputs.append(result)

        return outputs

    def train_equality(self, tokens: list[Tensor]):
        """Train to detect equal tokens."""
        tokens = [t.squeeze() for t in tokens]

        for query in tokens:
            for key in tokens:
                inp = cat([query, key]).unsqueeze(0)
                # Should attend if query == key
                should_match = 1 if (query == key).all() else 0
                target = Tensor([[should_match]]).to(uint8)
                self.match_detector.commit(inp, target)


# Train on some tokens
print("\nTraining equality detector on A, B, C, D...")
model = SelfMatchAttention(bits_per_token, rng=42)
train_tokens = [encode_char(c) for c in "ABCD"]
model.train_equality(train_tokens)

# Test on sequence with repeats
test_cases = [
    ("ABAB", "Self-match should find A at 0,2 and B at 1,3"),
    ("AAAA", "All positions match all others"),
    ("ABCD", "Each position only matches itself"),
]

for seq, desc in test_cases:
    print(f"\n  {seq}: {desc}")
    tokens = [encode_char(c) for c in seq]

    for i, query_char in enumerate(seq):
        query = tokens[i]
        matches = []
        for j, key in enumerate(tokens):
            inp = cat([query, key]).unsqueeze(0)
            match = model.match_detector(inp).squeeze().item()
            if match:
                matches.append(j)
        print(f"    Query {query_char}@{i} matches positions: {matches}")

# Test generalization to unseen tokens
print("\n  Testing on unseen tokens (W, X, Y, Z)...")
unseen_tokens = [encode_char(c) for c in "WXYZ"]
model.train_equality(unseen_tokens)  # Need to train equality for new tokens

for seq in ["WXWX", "WWWW"]:
    print(f"\n  {seq}:")
    tokens = [encode_char(c) for c in seq]
    for i, query_char in enumerate(seq):
        query = tokens[i]
        matches = []
        for j, key in enumerate(tokens):
            inp = cat([query, key]).unsqueeze(0)
            match = model.match_detector(inp).squeeze().item()
            if match:
                matches.append(j)
        print(f"    Query {query_char}@{i} matches positions: {matches}")


# =============================================================
# Test 2: Bit-Level Equality Detection
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Bit-Level Equality Detection")
print("=" * 60)
print("Can we learn equality using BIT_LEVEL generalization?")

from wnn.ram.RAMGeneralization import GeneralizingProjection

class BitLevelEqualityDetector(Module):
    """
    Detect if two tokens are equal using per-bit XOR comparison.

    Idea: Compute XOR of corresponding bits. If all zero, tokens are equal.
    This is computed directly (no learning needed).
    """

    def __init__(self, input_bits: int, rng: int = None):
        super().__init__()
        self.input_bits = input_bits

    def check_equal(self, query: Tensor, key: Tensor) -> bool:
        """Check if query equals key using XOR."""
        query = query.squeeze()
        key = key.squeeze()

        # XOR: if any bit differs, not equal
        xor_result = query ^ key
        return (xor_result == 0).all().item()

    def get_xor_bits(self, query: Tensor, key: Tensor) -> Tensor:
        """Get XOR of corresponding bits (shows which bits differ)."""
        return query.squeeze() ^ key.squeeze()


print("\nBit-level XOR detector (no training needed - direct computation):")
detector = BitLevelEqualityDetector(bits_per_token, rng=42)

# Test equality detection
print("\nTesting equality detection:")
for q_char in "ABCD":
    for k_char in "ABCD":
        query = encode_char(q_char)
        key = encode_char(k_char)
        equal = detector.check_equal(query, key)
        expected = q_char == k_char
        status = "✓" if equal == expected else "✗"
        print(f"  {status} {q_char} == {k_char}? {equal} (expected {expected})")

# Test on unseen tokens
print("\nTesting on unseen tokens (W, X):")
for q_char in "WX":
    for k_char in "WX":
        query = encode_char(q_char)
        key = encode_char(k_char)
        equal = detector.check_equal(query, key)
        expected = q_char == k_char
        status = "✓" if equal == expected else "✗"
        print(f"  {status} {q_char} == {k_char}? {equal} (expected {expected})")


# =============================================================
# Test 3: Similarity-Based Attention (Hamming Distance)
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Similarity-Based Attention")
print("=" * 60)
print("Attend to tokens within Hamming distance threshold")

class SimilarityAttention(Module):
    """
    Attend to tokens similar to query (low Hamming distance).
    """

    def __init__(self, input_bits: int, threshold: int = 1, rng: int = None):
        super().__init__()
        self.input_bits = input_bits
        self.threshold = threshold

    def get_attention(self, query: Tensor, keys: list[Tensor]) -> list[float]:
        """Return attention weights based on similarity."""
        query = query.squeeze()
        weights = []

        for key in keys:
            key = key.squeeze()
            dist = hamming_distance(query, key)
            # Attend if within threshold
            weight = 1.0 if dist <= self.threshold else 0.0
            weights.append(weight)

        return weights


print("\nCharacter bit patterns:")
for c in "ABCDEFGH":
    bits = encode_char(c)
    print(f"  {c} = {bits.tolist()}")

print("\nHamming distances from 'A':")
a_bits = encode_char('A')
for c in "ABCDEFGH":
    c_bits = encode_char(c)
    dist = hamming_distance(a_bits, c_bits)
    print(f"  A ↔ {c}: {dist}")

print("\nSimilarity attention (threshold=1):")
model = SimilarityAttention(bits_per_token, threshold=1)
tokens = [encode_char(c) for c in "ABCD"]
for i, c in enumerate("ABCD"):
    weights = model.get_attention(tokens[i], tokens)
    print(f"  Query {c}: weights = {weights}")


# =============================================================
# Test 4: Key-Value Content Lookup
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Key-Value Content Lookup")
print("=" * 60)
print("Query content determines which position to attend")
print("E.g., 'A' always attends to position 0, 'B' to position 1, etc.")

class ContentPositionMapper(Module):
    """
    Map token content to attended position using a simple RAM lookup.
    """

    def __init__(self, input_bits: int, max_positions: int = 4, rng: int = None):
        super().__init__()
        self.input_bits = input_bits
        self.max_positions = max_positions
        self.pos_bits = max_positions.bit_length()

        # Simple RAM layer: token bits → position bits
        self.ram = RAMLayer(
            total_input_bits=input_bits,
            num_neurons=self.pos_bits,
            n_bits_per_neuron=min(input_bits, 8),
            rng=rng,
        )

    def get_attended_position(self, query: Tensor) -> int:
        """Get position that this query content should attend to."""
        query = query.squeeze().unsqueeze(0)
        pos_enc = self.ram(query).squeeze()
        pos = 0
        for b in pos_enc:
            pos = pos * 2 + b.item()
        return min(pos, self.max_positions - 1)

    def train_mapping(self, token: Tensor, position: int):
        """Train: this token content should attend to this position."""
        token = token.squeeze().unsqueeze(0)
        pos_enc = zeros(self.pos_bits, dtype=uint8)
        for i in range(self.pos_bits - 1, -1, -1):
            pos_enc[self.pos_bits - 1 - i] = (position >> i) & 1
        self.ram.commit(token, pos_enc.unsqueeze(0))


print("\nTraining content → position mapping:")
print("  A → position 0")
print("  B → position 1")
print("  C → position 2")
print("  D → position 3")

mapper = ContentPositionMapper(bits_per_token, max_positions=4, rng=42)
for i, c in enumerate("ABCD"):
    mapper.train_mapping(encode_char(c), i)

print("\nTesting trained tokens:")
for c in "ABCD":
    pos = mapper.get_attended_position(encode_char(c))
    expected = "ABCD".index(c)
    status = "✓" if pos == expected else "✗"
    print(f"  {status} {c} → position {pos} (expected {expected})")

print("\nTesting unseen tokens (does BIT_LEVEL generalize?):")
for c in "EFGH":
    pos = mapper.get_attended_position(encode_char(c))
    print(f"  {c} → position {pos}")


# =============================================================
# Test 5: Hybrid Content + Position Attention
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Hybrid Content + Position Attention")
print("=" * 60)
print("Combine content-based routing with position patterns")

print("\nUsing SoftRAMAttention with content (standard mode):")
model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    position_only=False,  # Include content
    rng=42,
)

# Train: 'A' queries attend to position 0, 'B' to position 1, etc.
train_seqs = ["ABCD", "DCBA", "BADC"]  # Various arrangements

for seq in train_seqs:
    tokens = [encode_char(c) for c in seq]
    model.train_value_projection(tokens)

    for pos in range(len(tokens)):
        query_char = seq[pos]
        # This query char should attend to position = ord(query_char) - ord('A')
        target_pos = ord(query_char) - ord('A')
        target_weights = [0.0, 0.0, 0.0, 0.0]
        target_weights[target_pos] = 1.0
        model.train_attention_weights(tokens, pos, target_weights)

print("\nTrained: A→pos0, B→pos1, C→pos2, D→pos3")
print("Testing on various arrangements:")

for seq in ["ABCD", "DCBA", "CDAB"]:
    tokens = [encode_char(c) for c in seq]
    print(f"\n  Sequence: {seq}")
    for pos in range(len(tokens)):
        weights = model.get_attention_weights(tokens, pos)
        max_pos = max(range(len(weights)), key=lambda i: weights[i])
        query_char = seq[pos]
        expected_pos = ord(query_char) - ord('A')
        status = "✓" if max_pos == expected_pos else "✗"
        print(f"    {status} {query_char}@{pos} attends to pos {max_pos} (expected {expected_pos})")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Content-Based Attention")
print("=" * 60)

print("""
CONTENT-BASED ATTENTION PATTERNS:

1. SELF-MATCH (attend to identical tokens):
   - Requires learning equality for each token pair
   - Doesn't generalize: (A==A) doesn't help learn (W==W)
   - Need to train on all token pairs

2. BIT-LEVEL EQUALITY:
   - Learn XOR pattern: query[i] != key[i]
   - BIT_LEVEL can potentially generalize XOR
   - Still limited by bit pattern coverage

3. SIMILARITY (Hamming distance):
   - No learning needed - computed directly
   - Works for any tokens
   - Limited expressiveness (can't learn arbitrary patterns)

4. CONTENT → POSITION MAPPING:
   - Learn which content should attend where
   - BIT_LEVEL can partially generalize
   - Good for fixed routing rules

5. HYBRID (Content + Position):
   - Full SoftRAMAttention with content
   - Most flexible but least generalizable
   - Requires training on content×position combinations

GENERALIZATION HIERARCHY:
  Position-only    > Content→Position > Full content+position
  (Best)                                (Worst)

RECOMMENDATIONS:
  - If attention pattern is position-based: use position_only=True
  - If routing is content-based but fixed: use content→position mapper
  - If pattern is arbitrary: train on all combinations
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
