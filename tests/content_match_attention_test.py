#!/usr/bin/env python3
"""
Content Match Attention Test

Test the integrated ContentMatchMode in SoftRAMAttention.

Key insight: XOR-based matching generalizes 100% because it's computed
directly (no training needed). The attention decision is:
  - XOR_EQUAL: attend if query == key
  - HAMMING_1: attend if Hamming distance <= 1
  - HAMMING_2: attend if Hamming distance <= 2

This enables tasks like "attend to all matching tokens" to work
on completely unseen tokens.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.SoftRAMAttention import SoftRAMAttention, AggregationStrategy, ContentMatchMode
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import Tensor

print("=" * 70)
print("Content Match Attention Integration Test")
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
# Test 1: XOR_EQUAL - Self-Match Attention
# =============================================================
print("\n" + "=" * 60)
print("Test 1: XOR_EQUAL - Self-Match Attention")
print("=" * 60)
print("Attend to positions with identical tokens (query == key)")

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    content_match=ContentMatchMode.XOR_EQUAL,
    rng=42,
)

# Test: sequence with repeated characters
test_seqs = [
    ("ABAB", "Positions 0,2 should match (both A), 1,3 should match (both B)"),
    ("AAAA", "All positions should match all others"),
    ("ABCD", "Each position only matches itself"),
]

for seq, desc in test_seqs:
    print(f"\n  {seq}: {desc}")
    tokens = [encode_char(c) for c in seq]

    for i, query_char in enumerate(seq):
        weights = model.get_attention_weights(tokens, i)
        matching = [j for j, w in enumerate(weights) if w > 0]
        expected = [j for j, c in enumerate(seq) if c == query_char]
        status = "✓" if matching == expected else "✗"
        print(f"    {status} Query {query_char}@{i} attends to: {matching} (expected {expected})")

# Test on unseen tokens (should work 100%)
print("\n  Testing on unseen tokens (W, X, Y, Z)...")
for seq in ["WXWX", "WWWW", "WXYZ"]:
    print(f"\n  {seq}:")
    tokens = [encode_char(c) for c in seq]
    for i, query_char in enumerate(seq):
        weights = model.get_attention_weights(tokens, i)
        matching = [j for j, w in enumerate(weights) if w > 0]
        expected = [j for j, c in enumerate(seq) if c == query_char]
        status = "✓" if matching == expected else "✗"
        print(f"    {status} Query {query_char}@{i} attends to: {matching} (expected {expected})")


# =============================================================
# Test 2: XOR_EQUAL with Value Projection
# =============================================================
print("\n" + "=" * 60)
print("Test 2: XOR_EQUAL with Value Projection")
print("=" * 60)
print("Combine XOR matching with trained value projection")

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    content_match=ContentMatchMode.XOR_EQUAL,
    rng=42,
)

# Train value projection on identity for some chars
print("\nTraining value projection on A, B, C, D...")
train_tokens = [encode_char(c) for c in "ABCD"]
model.train_value_projection(train_tokens)

# Test: "ABAB" should output first matching char at each position
print("\nTesting ABAB:")
tokens = [encode_char(c) for c in "ABAB"]
outputs = model.forward(tokens)
result = decode_sequence(outputs)
print(f"  ABAB → {result}")

# For XOR_EQUAL + TOP_1: at position 0 (A), both positions 0,2 match
# TOP_1 picks first match (position 0), outputs A
# Expected: ABAB (each position outputs itself since self matches first)
for i, (out_c, exp_c) in enumerate(zip(result, "ABAB")):
    status = "✓" if out_c == exp_c else "~"
    print(f"  {status} Position {i}: output={out_c}, expected={exp_c}")


# =============================================================
# Test 3: HAMMING_1 - Similarity Attention
# =============================================================
print("\n" + "=" * 60)
print("Test 3: HAMMING_1 - Similarity Attention")
print("=" * 60)
print("Attend to tokens within Hamming distance 1")

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    content_match=ContentMatchMode.HAMMING_1,
    rng=42,
)

# Show bit patterns for reference
print("\nCharacter bit patterns:")
for c in "ABCDEFGH":
    bits = encode_char(c)
    print(f"  {c} = {bits.tolist()}")

# Compute Hamming distances from A
print("\nHamming distances from 'A':")
a_bits = encode_char('A')
distances = {}
for c in "ABCDEFGH":
    c_bits = encode_char(c)
    dist = (a_bits != c_bits).sum().item()
    distances[c] = dist
    print(f"  A ↔ {c}: {dist}")

# Test attention
print("\nAttention weights for 'ABCD':")
tokens = [encode_char(c) for c in "ABCD"]
for i, c in enumerate("ABCD"):
    weights = model.get_attention_weights(tokens, i)
    matching = [j for j, w in enumerate(weights) if w > 0]
    print(f"  Query {c}@{i}: attends to positions {matching}")


# =============================================================
# Test 4: Comparison - Learned vs XOR_EQUAL
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Comparison - Learned vs XOR_EQUAL")
print("=" * 60)
print("Task: Attend to all matching tokens")

# Learned attention: needs training on each token pair
print("\n--- Learned Attention ---")
model_learned = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    content_match=ContentMatchMode.NONE,  # Standard learned
    rng=42,
)

# Train on ABAB to match identical tokens
train_seq = "ABAB"
tokens = [encode_char(c) for c in train_seq]
model_learned.train_value_projection(tokens)

# Train attention: each position attends to matching positions
for i, query_char in enumerate(train_seq):
    target_weights = [0.0] * len(tokens)
    for j, key_char in enumerate(train_seq):
        if query_char == key_char:
            target_weights[j] = 1.0 / train_seq.count(query_char)  # Distribute weight
    model_learned.train_attention_weights(tokens, i, target_weights)

print("Trained on ABAB (matching attention)")
print("\nTesting on trained sequence (ABAB):")
for i, c in enumerate("ABAB"):
    weights = model_learned.get_attention_weights(tokens, i)
    matching = [j for j, w in enumerate(weights) if w > 0]
    expected = [j for j, x in enumerate("ABAB") if x == c]
    status = "✓" if matching == expected else "~"
    print(f"  {status} {c}@{i} → positions {matching} (expected {expected})")

print("\nTesting on unseen sequence (WXWX):")
unseen_tokens = [encode_char(c) for c in "WXWX"]
for i, c in enumerate("WXWX"):
    weights = model_learned.get_attention_weights(unseen_tokens, i)
    matching = [j for j, w in enumerate(weights) if w > 0]
    expected = [j for j, x in enumerate("WXWX") if x == c]
    status = "✓" if matching == expected else "✗"
    print(f"  {status} {c}@{i} → positions {matching} (expected {expected})")

# XOR_EQUAL attention: no training needed
print("\n--- XOR_EQUAL Attention ---")
model_xor = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    content_match=ContentMatchMode.XOR_EQUAL,  # Direct XOR matching
    rng=42,
)

print("No training needed for XOR matching")
print("\nTesting on ABAB:")
tokens = [encode_char(c) for c in "ABAB"]
for i, c in enumerate("ABAB"):
    weights = model_xor.get_attention_weights(tokens, i)
    matching = [j for j, w in enumerate(weights) if w > 0]
    expected = [j for j, x in enumerate("ABAB") if x == c]
    status = "✓" if matching == expected else "✗"
    print(f"  {status} {c}@{i} → positions {matching} (expected {expected})")

print("\nTesting on unseen WXWX:")
unseen_tokens = [encode_char(c) for c in "WXWX"]
for i, c in enumerate("WXWX"):
    weights = model_xor.get_attention_weights(unseen_tokens, i)
    matching = [j for j, w in enumerate(weights) if w > 0]
    expected = [j for j, x in enumerate("WXWX") if x == c]
    status = "✓" if matching == expected else "✗"
    print(f"  {status} {c}@{i} → positions {matching} (expected {expected})")


# =============================================================
# Test 5: Causal XOR Matching
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Causal XOR Matching")
print("=" * 60)
print("XOR matching with causal mask (only attend to past)")

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,  # Causal mask
    content_match=ContentMatchMode.XOR_EQUAL,
    rng=42,
)

print("\nTesting ABAB with causal mask:")
tokens = [encode_char(c) for c in "ABAB"]
for i, c in enumerate("ABAB"):
    weights = model.get_attention_weights(tokens, i)
    matching = [j for j, w in enumerate(weights) if w > 0]
    # Expected: only matching positions <= current position
    expected = [j for j, x in enumerate("ABAB") if x == c and j <= i]
    status = "✓" if matching == expected else "✗"
    print(f"  {status} {c}@{i} → positions {matching} (expected {expected})")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Content Match Modes")
print("=" * 60)

print("""
CONTENT MATCH MODES:

  ContentMatchMode.NONE:
    - Standard learned attention
    - Requires training on each (query, key) pattern
    - Doesn't generalize to unseen tokens

  ContentMatchMode.XOR_EQUAL:
    - Attend if query == key (XOR is all zeros)
    - 100% generalization - works on ANY tokens
    - No training needed for attention decision
    - Best for: self-matching, copy with repetition

  ContentMatchMode.HAMMING_1:
    - Attend if Hamming distance <= 1
    - Generalizes to similar (1-bit different) tokens
    - Best for: fuzzy matching, error tolerance

  ContentMatchMode.HAMMING_2:
    - Attend if Hamming distance <= 2
    - Even broader similarity matching
    - Best for: very fuzzy matching

KEY INSIGHT:
  XOR-based matching is COMPUTED, not LEARNED.
  This bypasses RAM's lookup table limitation for the
  attention decision while still using RAM for value projection.

COMBINATION STRATEGIES:
  1. XOR_EQUAL attention + BIT_LEVEL values = Best for self-matching tasks
  2. position_only attention + XOR_EQUAL = Position AND content patterns
  3. Learned attention + XOR hints = Hybrid approach
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
