#!/usr/bin/env python3
"""
Pure Aggregation Strategy Test

Isolate aggregation behavior from attention learning.
Use fixed vote patterns to test how each strategy combines values.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode
from torch import zeros, uint8, Tensor

print("=" * 70)
print("Pure Aggregation Strategy Comparison")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)

def encode_char(c: str) -> Tensor:
    return decoder.encode(c).squeeze()

def decode_bits(bits: Tensor) -> str:
    return decoder.decode(bits.unsqueeze(0))


# =============================================================
# Aggregation Functions (from previous test)
# =============================================================

def agg_xor_all(values, votes, num_heads):
    """XOR all values with any votes."""
    result = zeros(values[0].shape[0], dtype=uint8)
    for val, vote in zip(values, votes):
        if vote > 0:
            result = result ^ val
    return result

def agg_threshold(values, votes, num_heads):
    """XOR only values above 50% threshold."""
    threshold = num_heads // 2
    result = zeros(values[0].shape[0], dtype=uint8)
    for val, vote in zip(values, votes):
        if vote >= threshold:
            result = result ^ val
    return result

def agg_top_1(values, votes, num_heads):
    """Return only the highest-voted value."""
    max_vote = max(votes)
    for val, vote in zip(values, votes):
        if vote == max_vote:
            return val.clone()
    return zeros(values[0].shape[0], dtype=uint8)

def agg_top_2(values, votes, num_heads):
    """XOR top 2 highest-voted values."""
    sorted_pairs = sorted(zip(votes, values), key=lambda x: -x[0])
    result = zeros(values[0].shape[0], dtype=uint8)
    for i, (vote, val) in enumerate(sorted_pairs[:2]):
        if vote > 0:
            result = result ^ val
    return result

def agg_majority(values, votes, num_heads):
    """Per-bit weighted majority voting."""
    bits = values[0].shape[0]
    result = zeros(bits, dtype=uint8)
    for bit_pos in range(bits):
        weighted_ones = sum(vote for val, vote in zip(values, votes) if val[bit_pos] == 1)
        weighted_zeros = sum(vote for val, vote in zip(values, votes) if val[bit_pos] == 0)
        result[bit_pos] = 1 if weighted_ones > weighted_zeros else 0
    return result

def agg_weighted_first(values, votes, num_heads):
    """Use first value above threshold, else highest."""
    threshold = num_heads // 2
    for val, vote in zip(values, votes):
        if vote >= threshold:
            return val.clone()
    return agg_top_1(values, votes, num_heads)


STRATEGIES = {
    "XOR All": agg_xor_all,
    "Threshold (50%)": agg_threshold,
    "Top-1 (Winner)": agg_top_1,
    "Top-2 (XOR)": agg_top_2,
    "Majority Vote": agg_majority,
    "First Above 50%": agg_weighted_first,
}


# =============================================================
# Test Scenarios
# =============================================================

print("\n" + "=" * 60)
print("Test Scenarios with Fixed Vote Patterns")
print("=" * 60)

scenarios = [
    {
        "name": "Single Winner (100% to A)",
        "values": ["A", "B", "C", "D"],
        "votes": [8, 0, 0, 0],
        "expected_behavior": "Should return A",
    },
    {
        "name": "Two Equal (50% A, 50% B)",
        "values": ["A", "B", "C", "D"],
        "votes": [4, 4, 0, 0],
        "expected_behavior": "A XOR B for combining strategies",
    },
    {
        "name": "Dominant + Minor (75% A, 25% B)",
        "values": ["A", "B", "C", "D"],
        "votes": [6, 2, 0, 0],
        "expected_behavior": "A should dominate",
    },
    {
        "name": "Three Way Split",
        "values": ["A", "B", "C", "D"],
        "votes": [4, 3, 1, 0],
        "expected_behavior": "A wins or A XOR B",
    },
    {
        "name": "All Equal (25% each)",
        "values": ["A", "B", "C", "D"],
        "votes": [2, 2, 2, 2],
        "expected_behavior": "XOR all or tie-breaker",
    },
    {
        "name": "Last Two Only",
        "values": ["A", "B", "C", "D"],
        "votes": [0, 0, 5, 3],
        "expected_behavior": "C or C XOR D",
    },
]

num_heads = 8

for scenario in scenarios:
    print(f"\n--- {scenario['name']} ---")
    print(f"Values: {scenario['values']}")
    print(f"Votes:  {scenario['votes']} (out of {num_heads})")
    print(f"Expected: {scenario['expected_behavior']}")

    values = [encode_char(c) for c in scenario["values"]]
    votes = scenario["votes"]

    print("\nResults:")
    for name, func in STRATEGIES.items():
        result = func(values, votes, num_heads)
        char = decode_bits(result)
        print(f"  {name:20s} -> '{char}' {result.tolist()}")


# =============================================================
# Bit Pattern Analysis
# =============================================================
print("\n" + "=" * 60)
print("Bit Pattern Reference")
print("=" * 60)

print("\nCharacter bit patterns (5 bits):")
for c in "ABCDEFGH":
    bits = encode_char(c)
    print(f"  {c} = {bits.tolist()} (decimal {sum(b.item() << (4-i) for i, b in enumerate(bits))})")

print("\nXOR relationships:")
print(f"  A XOR B = {decode_bits(encode_char('A') ^ encode_char('B'))} = {(encode_char('A') ^ encode_char('B')).tolist()}")
print(f"  A XOR C = {decode_bits(encode_char('A') ^ encode_char('C'))} = {(encode_char('A') ^ encode_char('C')).tolist()}")
print(f"  B XOR C = {decode_bits(encode_char('B') ^ encode_char('C'))} = {(encode_char('B') ^ encode_char('C')).tolist()}")
print(f"  A XOR B XOR C = {decode_bits(encode_char('A') ^ encode_char('B') ^ encode_char('C'))} = {(encode_char('A') ^ encode_char('B') ^ encode_char('C')).tolist()}")


# =============================================================
# Strategy Recommendation Matrix
# =============================================================
print("\n" + "=" * 60)
print("Strategy Recommendation Matrix")
print("=" * 60)

print("""
┌─────────────────────┬────────────────┬─────────────────────────────┐
│ Use Case            │ Best Strategy  │ Why                         │
├─────────────────────┼────────────────┼─────────────────────────────┤
│ Select single best  │ Top-1 (Winner) │ Returns highest-voted only  │
│ Combine top choices │ Top-2 (XOR)    │ XOR gives difference        │
│ Weighted average    │ Majority Vote  │ Per-bit weighted decision   │
│ Selective inclusion │ Threshold      │ Include if above cutoff     │
│ First match         │ First Above    │ Position-aware selection    │
│ Combine all voted   │ XOR All        │ Order-independent merge     │
└─────────────────────┴────────────────┴─────────────────────────────┘
""")


# =============================================================
# Practical Example: Attention-based Selection
# =============================================================
print("\n" + "=" * 60)
print("Practical Example: Position-based Retrieval")
print("=" * 60)
print("Scenario: Query wants to retrieve value from position 2")

values = [encode_char(c) for c in "ABCD"]
position_labels = ["pos0", "pos1", "pos2", "pos3"]

# Ideal: all votes to position 2
ideal_votes = [0, 0, 8, 0]
print(f"\nIdeal attention: {list(zip(position_labels, ideal_votes))}")

for name, func in STRATEGIES.items():
    result = func(values, ideal_votes, num_heads)
    char = decode_bits(result)
    expected = "C"
    status = "✓" if char == expected else "✗"
    print(f"  {status} {name:20s} -> '{char}' (want 'C')")

# Noisy: some votes leak to adjacent positions
print(f"\nNoisy attention (some leak): ")
noisy_votes = [0, 2, 5, 1]
print(f"  Votes: {list(zip(position_labels, noisy_votes))}")

for name, func in STRATEGIES.items():
    result = func(values, noisy_votes, num_heads)
    char = decode_bits(result)
    expected = "C"
    status = "✓" if char == expected else "~"
    print(f"  {status} {name:20s} -> '{char}' (want 'C')")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print("""
KEY FINDINGS:

1. TOP-1 (Winner-take-all):
   - Most robust to noise
   - Always returns a valid value
   - Best for retrieval/lookup tasks

2. MAJORITY VOTE:
   - Best approximation of weighted averaging
   - Handles ties gracefully (defaults to 0)
   - Good for combining similar values

3. THRESHOLD:
   - Good middle ground
   - Excludes low-confidence values
   - May return XOR of multiple (unexpected)

4. XOR-BASED (Top-2, XOR All):
   - Creates "difference" between values
   - Good for encoding relationships
   - May produce unexpected characters

RECOMMENDATION FOR RAM ATTENTION:
  - Use TOP-1 for retrieval (copy, lookup)
  - Use MAJORITY for combining (averaging)
  - Use THRESHOLD for selective fusion
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
