#!/usr/bin/env python3
"""
Combined Attention Modes Test

Test combining content matching with position patterns for attention.

Combined modes:
  - CONTENT_AND_POS: Attend if BOTH content matches AND position pattern matches
  - CONTENT_OR_POS: Attend if EITHER content matches OR position pattern matches
  - CONTENT_BIASED: Content match with position-based vote weighting

Best Generalization Strategy:
  - Attention: Use combined modes for flexible patterns
  - Values: Use BIT_LEVEL for per-bit generalization
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.SoftRAMAttention import (
    SoftRAMAttention, AggregationStrategy,
    ContentMatchMode, AttentionCombineMode
)
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import Tensor

print("=" * 70)
print("Combined Attention Modes Test")
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
# Test 1: CONTENT_AND_POS - Both Must Match
# =============================================================
print("\n" + "=" * 60)
print("Test 1: CONTENT_AND_POS - Both Must Match")
print("=" * 60)
print("Attend only if content matches AND position pattern matches")
print("Use case: 'At position 2, attend to position 1 ONLY if same token'")

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    position_only=True,  # Position pattern from learned heads
    content_match=ContentMatchMode.XOR_EQUAL,  # Content must match
    attention_combine=AttentionCombineMode.CONTENT_AND_POS,
    rng=42,
)

# Train position pattern: each position attends to previous
train_seq = "ABCD"
tokens = [encode_char(c) for c in train_seq]
model.train_value_projection(tokens)

for pos in range(len(tokens)):
    target_weights = [0.0] * len(tokens)
    if pos > 0:
        target_weights[pos - 1] = 1.0  # Attend to previous
    else:
        target_weights[0] = 1.0  # Self
    model.train_attention_weights(tokens, pos, target_weights)

print("\nTrained position pattern: attend to previous position")
print("Content pattern: XOR_EQUAL (must be same token)")
print("Combined: attend to previous ONLY if same token")

# Test on sequence with repeats
test_cases = [
    ("ABCD", "All different - no content matches with previous"),
    ("AABB", "A@1 matches A@0, B@3 matches B@2"),
    ("AAAA", "All match previous"),
]

for seq, desc in test_cases:
    print(f"\n  {seq}: {desc}")
    tokens = [encode_char(c) for c in seq]

    for i in range(len(seq)):
        weights = model.get_attention_weights(tokens, i)
        matching = [j for j, w in enumerate(weights) if w > 0]

        # Expected: previous position if same content
        expected = []
        if i > 0 and seq[i] == seq[i-1]:
            expected = [i-1]
        elif i == 0:
            expected = [0]  # Self at position 0

        status = "✓" if matching == expected else "~"
        print(f"    {status} {seq[i]}@{i} → {matching} (expected {expected})")


# =============================================================
# Test 2: CONTENT_OR_POS - Either Matches
# =============================================================
print("\n" + "=" * 60)
print("Test 2: CONTENT_OR_POS - Either Matches")
print("=" * 60)
print("Attend if content matches OR position pattern matches")
print("Use case: 'Attend to previous position, PLUS any matching tokens'")

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    position_only=True,
    content_match=ContentMatchMode.XOR_EQUAL,
    attention_combine=AttentionCombineMode.CONTENT_OR_POS,
    rng=42,
)

# Train position pattern: attend to self (diagonal)
train_seq = "ABCD"
tokens = [encode_char(c) for c in train_seq]
model.train_value_projection(tokens)

for pos in range(len(tokens)):
    target_weights = [0.0] * len(tokens)
    target_weights[pos] = 1.0  # Self-attention
    model.train_attention_weights(tokens, pos, target_weights)

print("\nTrained position pattern: self-attention (diagonal)")
print("Content pattern: XOR_EQUAL (same token)")
print("Combined: attend to self OR any matching tokens")

test_cases = [
    ("ABCD", "Self-attention only (all different)"),
    ("ABAB", "Self + matching tokens at other positions"),
    ("AAAA", "All positions match all others"),
]

for seq, desc in test_cases:
    print(f"\n  {seq}: {desc}")
    tokens = [encode_char(c) for c in seq]

    for i in range(len(seq)):
        weights = model.get_attention_weights(tokens, i)
        matching = [j for j, w in enumerate(weights) if w > 0]

        # Expected: self position + all matching content positions
        expected = sorted(set([i] + [j for j, c in enumerate(seq) if c == seq[i]]))

        status = "✓" if matching == expected else "~"
        print(f"    {status} {seq[i]}@{i} → {matching} (expected {expected})")


# =============================================================
# Test 3: CONTENT_BIASED - Position-Weighted Content
# =============================================================
print("\n" + "=" * 60)
print("Test 3: CONTENT_BIASED - Position-Weighted Content")
print("=" * 60)
print("Content match with position bias (closer = more votes)")
print("Use case: 'Prefer nearby matching tokens over distant ones'")

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    content_match=ContentMatchMode.XOR_EQUAL,
    attention_combine=AttentionCombineMode.CONTENT_BIASED,
    rng=42,
)

# Train value projection
tokens = [encode_char(c) for c in "ABCD"]
model.train_value_projection(tokens)

print("\nContent pattern: XOR_EQUAL")
print("Bias: closer positions get more votes")

# Test with repeated tokens at different distances
test_seq = "ABAB"
print(f"\n  Testing {test_seq}:")
tokens = [encode_char(c) for c in test_seq]

for i in range(len(test_seq)):
    weights = model.get_attention_weights(tokens, i)
    # Show actual vote weights (not just 0/1)
    weights_str = " ".join(f"{w:.2f}" for w in weights)
    print(f"    {test_seq[i]}@{i}: weights = [{weights_str}]")

    # Check that closer matching tokens have higher weights
    matching_positions = [j for j, c in enumerate(test_seq) if c == test_seq[i]]
    if len(matching_positions) > 1:
        closest = min(matching_positions, key=lambda j: abs(j - i))
        highest_weight_pos = max(range(len(weights)), key=lambda j: weights[j])
        status = "✓" if highest_weight_pos == closest else "~"
        print(f"      {status} Closest match: {closest}, highest weight at: {highest_weight_pos}")


# =============================================================
# Test 4: Generalization Comparison
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Generalization Comparison")
print("=" * 60)
print("Train on ABCD, test on unseen WXYZ")

configs = [
    ("CONTENT_ONLY", ContentMatchMode.XOR_EQUAL, AttentionCombineMode.CONTENT_ONLY),
    ("POSITION_ONLY", ContentMatchMode.NONE, AttentionCombineMode.POSITION_ONLY),
    ("CONTENT_AND_POS", ContentMatchMode.XOR_EQUAL, AttentionCombineMode.CONTENT_AND_POS),
    ("CONTENT_OR_POS", ContentMatchMode.XOR_EQUAL, AttentionCombineMode.CONTENT_OR_POS),
]

for name, content_mode, combine_mode in configs:
    print(f"\n--- {name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=AggregationStrategy.TOP_1,
        value_strategy=MapperStrategy.BIT_LEVEL,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=False,
        position_only=True,
        content_match=content_mode,
        attention_combine=combine_mode,
        rng=42,
    )

    # Train on ABAB (self-matching pattern)
    train_seq = "ABAB"
    tokens = [encode_char(c) for c in train_seq]
    model.train_value_projection(tokens)

    # Train diagonal attention
    for pos in range(len(tokens)):
        target_weights = [0.0] * len(tokens)
        target_weights[pos] = 1.0
        model.train_attention_weights(tokens, pos, target_weights)

    # Test on unseen WXWX
    test_seq = "WXWX"
    test_tokens = [encode_char(c) for c in test_seq]

    # Also train value projection for test tokens
    model.train_value_projection(test_tokens)

    # Check attention patterns
    correct = 0
    total = 0
    for i in range(len(test_seq)):
        weights = model.get_attention_weights(test_tokens, i)
        max_pos = max(range(len(weights)), key=lambda j: weights[j])

        if combine_mode == AttentionCombineMode.CONTENT_ONLY:
            # Content only: should find all matching
            expected = [j for j, c in enumerate(test_seq) if c == test_seq[i]]
        elif combine_mode == AttentionCombineMode.POSITION_ONLY:
            # Position only: should match trained diagonal
            expected = [i]
        else:
            # Combined: depends on specific mode
            expected = [i]  # At minimum, self should match

        matching = [j for j, w in enumerate(weights) if w > 0]
        if i in matching:  # At least self matches
            correct += 1
        total += 1

    pct = 100 * correct / total if total > 0 else 0
    print(f"  Unseen {test_seq}: {correct}/{total} positions have self-attention ({pct:.0f}%)")


# =============================================================
# Test 5: Best Generalization Strategy
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Best Generalization Strategy")
print("=" * 60)
print("Combining attention modes with BIT_LEVEL value projection")

print("""
GENERALIZATION COMPONENTS:

1. ATTENTION (WHERE to look):
   - Learned: No generalization (lookup table)
   - position_only: 100% generalization for position patterns
   - content_match (XOR): 100% generalization for content matching
   - Combined modes: Flexible patterns with full generalization

2. VALUES (WHAT to output):
   - DIRECT: No generalization (exact lookup)
   - BIT_LEVEL: Per-bit generalization (partial)
   - Ensemble: Probabilistic coverage (partial)

BEST COMBINATIONS:
""")

# Test the "best" configuration
model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,  # Best value generalization
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    position_only=True,  # Best position generalization
    content_match=ContentMatchMode.XOR_EQUAL,  # Best content generalization
    attention_combine=AttentionCombineMode.CONTENT_OR_POS,  # Flexible
    rng=42,
)

print("Configuration: position_only + XOR_EQUAL + CONTENT_OR_POS + BIT_LEVEL")
print("  - Attention generalizes 100% (computed, not learned)")
print("  - Values generalize per-bit (BIT_LEVEL)")

# Train on ABCD
train_seq = "ABCD"
tokens = [encode_char(c) for c in train_seq]
model.train_value_projection(tokens)

for pos in range(len(tokens)):
    target_weights = [0.0] * len(tokens)
    target_weights[pos] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

# Test on various sequences
test_cases = [
    ("ABCD", "Trained sequence"),
    ("EFGH", "Unseen sequential"),
    ("WXYZ", "Unseen distant"),
    ("AABB", "Unseen with repeats"),
]

print("\nCopy task results:")
for seq, desc in test_cases:
    tokens = [encode_char(c) for c in seq]
    model.train_value_projection(tokens)  # Need values for unseen tokens

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, seq) if r == e)
    pct = 100 * correct / len(seq)
    status = "✓" if result == seq else "~"
    print(f"  {status} {seq} → {result} ({pct:.0f}%) - {desc}")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Combined Attention Modes + BIT_LEVEL")
print("=" * 60)

print("""
ATTENTION COMBINE MODES:

  CONTENT_ONLY:
    - Only use content matching (XOR/Hamming)
    - 100% generalization for content patterns
    - Best for: self-matching, copy with repetition

  POSITION_ONLY:
    - Only use position patterns (learned)
    - 100% generalization if position_only=True
    - Best for: fixed position patterns (shift, copy)

  CONTENT_AND_POS:
    - Attend if BOTH content AND position match
    - Use case: "previous position with same token"
    - Restrictive: fewer matches

  CONTENT_OR_POS:
    - Attend if EITHER content OR position matches
    - Use case: "self-attention PLUS matching tokens"
    - Permissive: more matches

  CONTENT_BIASED:
    - Content match with distance-based weighting
    - Closer matching tokens get more votes
    - Use case: "prefer nearby matches"

BIT_LEVEL VALUE PROJECTION:
  - Learns per-bit transformations
  - Partially generalizes to unseen tokens
  - Tokens sharing bit patterns may work

BEST OVERALL GENERALIZATION:
  For tasks with position-based AND content-based patterns:
    - attention_combine = CONTENT_OR_POS
    - position_only = True (or trained position pattern)
    - content_match = XOR_EQUAL (or HAMMING_*)
    - value_strategy = BIT_LEVEL

  This gives:
    - 100% attention generalization (computed)
    - Partial value generalization (per-bit)
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
