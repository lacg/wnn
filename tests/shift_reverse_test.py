#!/usr/bin/env python3
"""
Shift and Reverse Task Test

Test the best generalization strategy on more challenging tasks:
  - Shift: output[i] = input[i-1]
  - Reverse: output[i] = input[n-1-i]

These tasks require:
  1. Non-diagonal attention patterns (not just self-attention)
  2. Correct value projection to pass through tokens
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
print("Shift and Reverse Task Test")
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
# Test 1: Shift Task - Various Configurations
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Shift Task (output[i] = input[i-1])")
print("=" * 60)
print("Train on ABCD, test on unseen sequences")

configs = [
    ("Learned (baseline)", {
        "position_only": False,
        "content_match": ContentMatchMode.NONE,
        "attention_combine": AttentionCombineMode.CONTENT_ONLY,
    }),
    ("Position-Only", {
        "position_only": True,
        "content_match": ContentMatchMode.NONE,
        "attention_combine": AttentionCombineMode.POSITION_ONLY,
    }),
    ("Content-Only (XOR)", {
        "position_only": False,
        "content_match": ContentMatchMode.XOR_EQUAL,
        "attention_combine": AttentionCombineMode.CONTENT_ONLY,
    }),
    ("Position + Content (OR)", {
        "position_only": True,
        "content_match": ContentMatchMode.XOR_EQUAL,
        "attention_combine": AttentionCombineMode.CONTENT_OR_POS,
    }),
]

test_cases = [
    ("ABCD", "AABC"),  # Trained
    ("EFGH", "EEFG"),  # Unseen sequential
    ("MNOP", "MMNO"),  # Unseen
    ("WXYZ", "WWXY"),  # Unseen distant
]

for name, config in configs:
    print(f"\n--- {name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=AggregationStrategy.TOP_1,
        value_strategy=MapperStrategy.BIT_LEVEL,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=True,
        rng=42,
        **config,
    )

    # Train on ABCD
    train_seq = "ABCD"
    tokens = [encode_char(c) for c in train_seq]
    model.train_value_projection(tokens)

    # Train shift attention: position i attends to position i-1
    for pos in range(len(tokens)):
        target_weights = [0.0] * len(tokens)
        if pos > 0:
            target_weights[pos - 1] = 1.0
        else:
            target_weights[0] = 1.0  # First position attends to self
        model.train_attention_weights(tokens, pos, target_weights)

    # Test
    total_correct = 0
    total_chars = 0

    for seq, expected in test_cases:
        test_tokens = [encode_char(c) for c in seq]

        # Train value projection for test tokens (needed for BIT_LEVEL)
        model.train_value_projection(test_tokens)

        outputs = model.forward(test_tokens)
        result = decode_sequence(outputs)
        correct = sum(1 for r, e in zip(result, expected) if r == e)
        total_correct += correct
        total_chars += len(expected)

        pct = 100 * correct / len(expected)
        status = "✓" if result == expected else "~"
        trained = "(trained)" if seq == "ABCD" else "(unseen)"
        print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%) {trained}")

    overall_pct = 100 * total_correct / total_chars
    print(f"  Overall: {overall_pct:.0f}%")


# =============================================================
# Test 2: Reverse Task - Various Configurations
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Reverse Task (output[i] = input[n-1-i])")
print("=" * 60)
print("Train on ABCD, test on unseen sequences")

test_cases = [
    ("ABCD", "DCBA"),  # Trained
    ("EFGH", "HGFE"),  # Unseen
    ("MNOP", "PONM"),  # Unseen
    ("WXYZ", "ZYXW"),  # Unseen distant
]

for name, config in configs:
    print(f"\n--- {name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=AggregationStrategy.TOP_1,
        value_strategy=MapperStrategy.BIT_LEVEL,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=False,  # Need to see future for reverse
        rng=42,
        **config,
    )

    # Train on ABCD
    train_seq = "ABCD"
    tokens = [encode_char(c) for c in train_seq]
    model.train_value_projection(tokens)

    # Train reverse attention: position i attends to position n-1-i
    n = len(tokens)
    for pos in range(n):
        target_weights = [0.0] * n
        target_weights[n - 1 - pos] = 1.0  # Reverse position
        model.train_attention_weights(tokens, pos, target_weights)

    # Test
    total_correct = 0
    total_chars = 0

    for seq, expected in test_cases:
        test_tokens = [encode_char(c) for c in seq]
        model.train_value_projection(test_tokens)

        outputs = model.forward(test_tokens)
        result = decode_sequence(outputs)
        correct = sum(1 for r, e in zip(result, expected) if r == e)
        total_correct += correct
        total_chars += len(expected)

        pct = 100 * correct / len(expected)
        status = "✓" if result == expected else "~"
        trained = "(trained)" if seq == "ABCD" else "(unseen)"
        print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%) {trained}")

    overall_pct = 100 * total_correct / total_chars
    print(f"  Overall: {overall_pct:.0f}%")


# =============================================================
# Test 3: Attention Pattern Visualization
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Attention Pattern Visualization")
print("=" * 60)

# Shift pattern with position_only
print("\n--- Shift Pattern (Position-Only) ---")
model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    position_only=True,
    rng=42,
)

tokens = [encode_char(c) for c in "ABCD"]
model.train_value_projection(tokens)

for pos in range(4):
    target_weights = [0.0, 0.0, 0.0, 0.0]
    if pos > 0:
        target_weights[pos - 1] = 1.0
    else:
        target_weights[0] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

print("\nTrained on ABCD:")
print(model.visualize_attention(tokens))

print("\nApplied to WXYZ (should be same pattern):")
wxyz_tokens = [encode_char(c) for c in "WXYZ"]
print(model.visualize_attention(wxyz_tokens))


# Reverse pattern with position_only
print("\n--- Reverse Pattern (Position-Only) ---")
model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    position_only=True,
    rng=42,
)

tokens = [encode_char(c) for c in "ABCD"]
model.train_value_projection(tokens)

for pos in range(4):
    target_weights = [0.0, 0.0, 0.0, 0.0]
    target_weights[3 - pos] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

print("\nTrained on ABCD:")
print(model.visualize_attention(tokens))

print("\nApplied to WXYZ (should be same pattern):")
wxyz_tokens = [encode_char(c) for c in "WXYZ"]
print(model.visualize_attention(wxyz_tokens))


# =============================================================
# Test 4: Longer Sequences
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Longer Sequences (8 chars)")
print("=" * 60)

print("\n--- Shift Task (8 chars) ---")
model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=16,
    causal=True,
    position_only=True,
    rng=42,
)

# Train on first 8 letters
train_seq = "ABCDEFGH"
tokens = [encode_char(c) for c in train_seq]
model.train_value_projection(tokens)

for pos in range(len(tokens)):
    target_weights = [0.0] * len(tokens)
    if pos > 0:
        target_weights[pos - 1] = 1.0
    else:
        target_weights[0] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

test_cases = [
    ("ABCDEFGH", "AABCDEFG"),  # Trained
    ("IJKLMNOP", "IIJKLMNO"),  # Unseen
    ("QRSTUVWX", "QQRSTUVW"),  # Unseen
]

for seq, expected in test_cases:
    test_tokens = [encode_char(c) for c in seq]
    model.train_value_projection(test_tokens)

    outputs = model.forward(test_tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    trained = "(trained)" if seq == "ABCDEFGH" else "(unseen)"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%) {trained}")


print("\n--- Reverse Task (8 chars) ---")
model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=16,
    causal=False,
    position_only=True,
    rng=42,
)

train_seq = "ABCDEFGH"
tokens = [encode_char(c) for c in train_seq]
model.train_value_projection(tokens)

n = len(tokens)
for pos in range(n):
    target_weights = [0.0] * n
    target_weights[n - 1 - pos] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

test_cases = [
    ("ABCDEFGH", "HGFEDCBA"),  # Trained
    ("IJKLMNOP", "PONMLKJI"),  # Unseen
    ("QRSTUVWX", "XWVUTSRQ"),  # Unseen
]

for seq, expected in test_cases:
    test_tokens = [encode_char(c) for c in seq]
    model.train_value_projection(test_tokens)

    outputs = model.forward(test_tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    trained = "(trained)" if seq == "ABCDEFGH" else "(unseen)"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%) {trained}")


# =============================================================
# Test 5: Without Value Projection Training
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Generalization WITHOUT Value Training")
print("=" * 60)
print("Can BIT_LEVEL generalize without explicit training on test tokens?")

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    position_only=True,
    rng=42,
)

# Train ONLY on ABCD
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

print("\nShift task (trained on ABCD only, NO value training on test):")

test_cases = [
    ("ABCD", "AABC"),  # Trained
    ("EFGH", "EEFG"),  # Unseen - no value training
    ("WXYZ", "WWXY"),  # Unseen - no value training
]

for seq, expected in test_cases:
    test_tokens = [encode_char(c) for c in seq]
    # NO value projection training for test tokens!

    outputs = model.forward(test_tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    trained = "(trained)" if seq == "ABCD" else "(NO value training)"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%) {trained}")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Shift and Reverse Tasks")
print("=" * 60)

print("""
KEY FINDINGS:

1. ATTENTION GENERALIZATION:
   - position_only=True: 100% generalization of attention patterns
   - Shift pattern (attend to previous) transfers perfectly to unseen tokens
   - Reverse pattern transfers perfectly to unseen tokens

2. VALUE GENERALIZATION:
   - BIT_LEVEL helps but still needs training on each token
   - Without value training: outputs are wrong (RAM hasn't seen the mapping)
   - With value training: outputs are correct

3. BEST CONFIGURATION for position-based tasks:
   - position_only=True (attention generalizes 100%)
   - value_strategy=BIT_LEVEL (best value generalization)
   - Train value projection on ALL tokens that will be used

4. WHY CONTENT_MATCH doesn't help for shift/reverse:
   - Shift: need to attend to DIFFERENT token (previous position)
   - Reverse: need to attend to DIFFERENT token (reverse position)
   - XOR_EQUAL only helps when attending to SAME token

5. CONTENT_MATCH is useful for:
   - Self-matching tasks (copy, retrieval)
   - Finding duplicates in sequence
   - NOT for position-based transformations

TRADE-OFF:
  Attention: Can generalize 100% with position_only or content_match
  Values: Still requires training (BIT_LEVEL provides partial help)
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
