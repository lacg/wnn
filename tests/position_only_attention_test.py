#!/usr/bin/env python3
"""
Position-Only Attention Test

Test attention that ignores token content and only considers positions.

Key insight: If attention is based only on (query_pos, key_pos), then:
- "Position 2 should attend to position 1" applies to ALL tokens
- Training on "ABCD" generalizes to "WXYZ" automatically
- The attention pattern is position-dependent, not content-dependent
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.SoftRAMAttention import SoftRAMAttention, AggregationStrategy
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import Tensor

print("=" * 70)
print("Position-Only Attention Test")
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
# Test 1: Copy Task - Position-Only vs Standard
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Copy Task (diagonal attention)")
print("=" * 60)
print("Train on ABCD, test on WXYZ - position patterns should generalize")

for position_only in [False, True]:
    name = "Position-Only" if position_only else "Standard"
    print(f"\n--- {name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=AggregationStrategy.TOP_1,
        value_strategy=MapperStrategy.BIT_LEVEL,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=False,
        position_only=position_only,
        rng=42,
    )

    # Train on ABCD only (diagonal attention pattern)
    train_seq = "ABCD"
    tokens = [encode_char(c) for c in train_seq]

    # Train value projection (identity)
    model.train_value_projection(tokens)

    # Train attention: each position attends to itself
    for pos in range(len(tokens)):
        target_weights = [0.0] * len(tokens)
        target_weights[pos] = 1.0
        model.train_attention_weights(tokens, pos, target_weights)

    # Test on trained sequence
    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    print(f"  Trained: {train_seq} → {result}")

    # Test on unseen sequences
    for test_seq in ["EFGH", "MNOP", "WXYZ"]:
        test_tokens = [encode_char(c) for c in test_seq]

        # Also train value projection for test tokens
        model.train_value_projection(test_tokens)

        outputs = model.forward(test_tokens)
        result = decode_sequence(outputs)
        correct = sum(1 for r, e in zip(result, test_seq) if r == e)
        pct = 100 * correct / len(test_seq)
        status = "✓" if result == test_seq else "~"
        print(f"  {status} Unseen: {test_seq} → {result} ({pct:.0f}%)")


# =============================================================
# Test 2: Shift Task - Position-Only
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Shift Task (output[i] = input[i-1])")
print("=" * 60)
print("Train shift pattern on ABCD, should generalize to all sequences")

for position_only in [False, True]:
    name = "Position-Only" if position_only else "Standard"
    print(f"\n--- {name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=AggregationStrategy.TOP_1,
        value_strategy=MapperStrategy.BIT_LEVEL,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=True,
        position_only=position_only,
        rng=42,
    )

    # Train on ABCD only
    train_seq = "ABCD"
    tokens = [encode_char(c) for c in train_seq]

    # Train value projection (identity)
    model.train_value_projection(tokens)

    # Train shift attention: position i attends to position i-1
    for pos in range(len(tokens)):
        target_weights = [0.0] * len(tokens)
        if pos > 0:
            target_weights[pos - 1] = 1.0
        else:
            target_weights[0] = 1.0  # First position attends to self
        model.train_attention_weights(tokens, pos, target_weights)

    # Test on various sequences
    test_cases = [
        ("ABCD", "AABC"),
        ("EFGH", "EEFG"),
        ("MNOP", "MMNO"),
        ("WXYZ", "WWXY"),
    ]

    for test_seq, expected in test_cases:
        test_tokens = [encode_char(c) for c in test_seq]

        # Train value projection for this sequence
        model.train_value_projection(test_tokens)

        outputs = model.forward(test_tokens)
        result = decode_sequence(outputs)
        correct = sum(1 for r, e in zip(result, expected) if r == e)
        pct = 100 * correct / len(expected)
        status = "✓" if result == expected else "~"
        print(f"  {status} {test_seq} → {result} (want {expected}, {pct:.0f}%)")


# =============================================================
# Test 3: Attention Pattern Visualization
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Position-Only Attention Patterns")
print("=" * 60)

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    position_only=True,
    rng=42,
)

# Train diagonal attention on ABCD
tokens = [encode_char(c) for c in "ABCD"]
model.train_value_projection(tokens)
for pos in range(4):
    target_weights = [0.0, 0.0, 0.0, 0.0]
    target_weights[pos] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

print("\nTrained on ABCD (diagonal):")
print(model.visualize_attention(tokens))

print("\nApplied to WXYZ (should be same pattern!):")
wxyz_tokens = [encode_char(c) for c in "WXYZ"]
print(model.visualize_attention(wxyz_tokens))


# =============================================================
# Test 4: First-Position Attention (Retrieval)
# =============================================================
print("\n" + "=" * 60)
print("Test 4: First-Position Attention (Retrieval)")
print("=" * 60)
print("All positions attend to position 0")

for position_only in [False, True]:
    name = "Position-Only" if position_only else "Standard"
    print(f"\n--- {name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=AggregationStrategy.TOP_1,
        value_strategy=MapperStrategy.BIT_LEVEL,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=False,
        position_only=position_only,
        rng=42,
    )

    # Train: all positions attend to position 0
    train_seq = "ABCD"
    tokens = [encode_char(c) for c in train_seq]
    model.train_value_projection(tokens)

    for pos in range(len(tokens)):
        target_weights = [1.0, 0.0, 0.0, 0.0]  # Always attend to position 0
        model.train_attention_weights(tokens, pos, target_weights)

    # Test: should output first char repeated
    test_cases = [
        ("ABCD", "AAAA"),
        ("WXYZ", "WWWW"),
        ("EFGH", "EEEE"),
    ]

    for test_seq, expected in test_cases:
        test_tokens = [encode_char(c) for c in test_seq]
        model.train_value_projection(test_tokens)

        outputs = model.forward(test_tokens)
        result = decode_sequence(outputs)
        correct = sum(1 for r, e in zip(result, expected) if r == e)
        pct = 100 * correct / len(expected)
        status = "✓" if result == expected else "~"
        print(f"  {status} {test_seq} → {result} (want {expected}, {pct:.0f}%)")


# =============================================================
# Test 5: Position-Only + Ensemble
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Position-Only + Ensemble (Best of Both)")
print("=" * 60)

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    position_only=True,
    use_ensemble=True,
    ensemble_sub_rams=4,
    rng=42,
)

# Train shift on one sequence
train_seq = "ABCD"
tokens = [encode_char(c) for c in train_seq]
model.train_value_projection(tokens)

for pos in range(len(tokens)):
    target_weights = [0.0] * len(tokens)
    if pos > 0:
        target_weights[pos - 1] = 1.0
    else:
        target_weights[0] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

# Test on many sequences
test_cases = [
    ("ABCD", "AABC"),
    ("EFGH", "EEFG"),
    ("MNOP", "MMNO"),
    ("WXYZ", "WWXY"),
]

print("\nShift task (trained on ABCD only):")
total_correct = 0
total_chars = 0

for test_seq, expected in test_cases:
    test_tokens = [encode_char(c) for c in test_seq]
    model.train_value_projection(test_tokens)

    outputs = model.forward(test_tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    total_correct += correct
    total_chars += len(expected)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    print(f"  {status} {test_seq} → {result} (want {expected}, {pct:.0f}%)")

overall_pct = 100 * total_correct / total_chars
print(f"\n  Overall: {overall_pct:.0f}%")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Position-Only Attention")
print("=" * 60)

print("""
POSITION-ONLY ATTENTION:
  Input to voting heads: only (query_pos, key_pos)
  Token content is NOT used for attention decision

WHY IT GENERALIZES:
  Standard: learns (query='A', key='B', pos=1) → attend
            This doesn't help for (query='W', key='X', pos=1)

  Position-Only: learns (pos=1) → attend to prev
                 This applies to ALL tokens at position 1

TRADE-OFF:
  + Perfect generalization of attention patterns
  - Cannot do content-based routing (e.g., "attend to all 'X' tokens")

  Best for: Fixed position patterns (shift, copy, first-position retrieval)
  Not for: Content-based attention (query-key matching)

VALUE PROJECTION STILL MATTERS:
  - Attention tells us WHERE to look
  - Value projection tells us WHAT to output
  - For copy tasks: need to train value projection on each token
  - BIT_LEVEL helps value projection generalize

COMBINED APPROACH:
  Position-Only attention + BIT_LEVEL values = Best generalization
  for position-based tasks
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
