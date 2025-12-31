#!/usr/bin/env python3
"""
Ensemble Attention Test

Compare SoftRAMAttention with and without ensemble voting heads.

Hypothesis: Ensemble voting heads should generalize better to unseen
(query, key, position) combinations because different projections
provide coverage of the input space.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.SoftRAMAttention import SoftRAMAttention, AggregationStrategy
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import Tensor

print("=" * 70)
print("Ensemble Attention Integration Test")
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
# Test 1: Copy Task - Standard vs Ensemble
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Copy Task Generalization")
print("=" * 60)
print("Train on A-L, test on M-Z")

# Train on first half of alphabet
train_chars = "ABCDEFGHIJKL"
test_chars = "MNOPQRSTUVWXYZ"

for use_ensemble in [False, True]:
    name = "Ensemble" if use_ensemble else "Standard"
    print(f"\n--- {name} Voting Heads ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=AggregationStrategy.TOP_1,
        value_strategy=MapperStrategy.BIT_LEVEL,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=False,
        use_ensemble=use_ensemble,
        ensemble_sub_rams=4,
        rng=42,
    )

    # Train on copy task for A-L
    for epoch in range(3):
        for i in range(0, len(train_chars), 4):
            seq = train_chars[i:i+4]
            tokens = [encode_char(c) for c in seq]
            model.train_step(tokens, tokens)

    # Test on trained data
    train_correct = 0
    for i in range(0, 8, 4):  # Test first two groups
        seq = train_chars[i:i+4]
        tokens = [encode_char(c) for c in seq]
        outputs = model.forward(tokens)
        result = decode_sequence(outputs)
        train_correct += sum(1 for r, e in zip(result, seq) if r == e)

    # Test on unseen data
    test_correct = 0
    for i in range(0, 8, 4):
        seq = test_chars[i:i+4]
        tokens = [encode_char(c) for c in seq]
        outputs = model.forward(tokens)
        result = decode_sequence(outputs)
        test_correct += sum(1 for r, e in zip(result, seq) if r == e)
        print(f"  {seq} → {result}")

    train_pct = 100 * train_correct / 8
    test_pct = 100 * test_correct / 8
    print(f"  Trained: {train_pct:.0f}%  |  Unseen: {test_pct:.0f}%")


# =============================================================
# Test 2: Shift Task - Standard vs Ensemble
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Shift Task (output[i] = input[i-1])")
print("=" * 60)

train_seqs = ["ABCD", "EFGH", "IJKL"]
test_seqs = [("MNOP", "MMNO"), ("QRST", "QQRS"), ("WXYZ", "WWXY")]

for use_ensemble in [False, True]:
    name = "Ensemble" if use_ensemble else "Standard"
    print(f"\n--- {name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=AggregationStrategy.TOP_1,
        value_strategy=MapperStrategy.BIT_LEVEL,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=True,
        use_ensemble=use_ensemble,
        ensemble_sub_rams=4,
        rng=42,
    )

    # Train shift task
    for epoch in range(3):
        for seq_str in train_seqs:
            tokens = [encode_char(c) for c in seq_str]
            model.train_value_projection(tokens)

            for pos in range(len(tokens)):
                target_weights = [0.0] * len(tokens)
                if pos > 0:
                    target_weights[pos - 1] = 1.0
                else:
                    target_weights[pos] = 1.0
                model.train_attention_weights(tokens, pos, target_weights)

    # Test
    for seq, expected in test_seqs:
        tokens = [encode_char(c) for c in seq]
        outputs = model.forward(tokens)
        result = decode_sequence(outputs)
        correct = sum(1 for r, e in zip(result, expected) if r == e)
        pct = 100 * correct / len(expected)
        status = "✓" if result == expected else "~"
        print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%)")


# =============================================================
# Test 3: Attention Pattern Visualization
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Attention Patterns (trained vs unseen)")
print("=" * 60)

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    use_ensemble=True,
    ensemble_sub_rams=4,
    rng=42,
)

# Train on ABCD
tokens = [encode_char(c) for c in "ABCD"]
model.train_value_projection(tokens)
for pos in range(4):
    target_weights = [0.0, 0.0, 0.0, 0.0]
    target_weights[pos] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

print("\nTrained sequence (ABCD):")
print(model.visualize_attention(tokens))

# Test on unseen sequence
unseen_tokens = [encode_char(c) for c in "WXYZ"]
print("\nUnseen sequence (WXYZ):")
print(model.visualize_attention(unseen_tokens))


# =============================================================
# Test 4: Varying Ensemble Size
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Effect of Ensemble Size")
print("=" * 60)

train_chars = "ABCDEFGH"
test_seqs = ["IJKL", "QRST"]

for sub_rams in [2, 4, 8]:
    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=AggregationStrategy.TOP_1,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=False,
        use_ensemble=True,
        ensemble_sub_rams=sub_rams,
        rng=42,
    )

    # Train
    for i in range(0, len(train_chars), 4):
        seq = train_chars[i:i+4]
        tokens = [encode_char(c) for c in seq]
        model.train_step(tokens, tokens)

    # Test
    total_correct = 0
    total_chars = 0
    for seq in test_seqs:
        tokens = [encode_char(c) for c in seq]
        outputs = model.forward(tokens)
        result = decode_sequence(outputs)
        total_correct += sum(1 for r, e in zip(result, seq) if r == e)
        total_chars += len(seq)

    pct = 100 * total_correct / total_chars
    print(f"  {sub_rams} sub-RAMs per head: {pct:.0f}% on unseen")


# =============================================================
# Test 5: Coverage Analysis
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Attention Decision Coverage")
print("=" * 60)

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=4,  # Fewer heads to analyze
    aggregation=AggregationStrategy.TOP_1,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    use_ensemble=True,
    ensemble_sub_rams=4,
    rng=42,
)

# Train on ABCD
tokens = [encode_char(c) for c in "ABCD"]
model.train_value_projection(tokens)
for pos in range(4):
    target_weights = [0.0, 0.0, 0.0, 0.0]
    target_weights[pos] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

print("\nAfter training on ABCD (diagonal attention):")
print("Checking attention weights for various sequences...")

for seq in ["ABCD", "EFGH", "WXYZ"]:
    tokens = [encode_char(c) for c in seq]
    print(f"\n  {seq}:")
    for query_pos in range(4):
        weights = model.get_attention_weights(tokens, query_pos)
        weights_str = " ".join(f"{w:.2f}" for w in weights)
        expected_pos = query_pos  # Diagonal attention
        actual_max = max(range(len(weights)), key=lambda i: weights[i])
        status = "✓" if actual_max == expected_pos else "✗"
        print(f"    q={query_pos}: [{weights_str}] max@{actual_max} {status}")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Ensemble Voting Heads")
print("=" * 60)

print("""
ENSEMBLE VOTING HEAD MECHANISM:
  - Each voting head has multiple sub-RAMs with different projections
  - Each sub-RAM sees ~60% of input bits (different subset)
  - Majority vote across sub-RAMs decides attend/don't-attend

WHY IT HELPS:
  - An unseen (query, key, position) pattern might match known
    patterns in some projections even if the full pattern is new
  - More robust to noise in individual RAM lookups

TRADE-OFFS:
  - More memory: num_heads × ensemble_sub_rams sub-RAMs
  - Still requires training data coverage
  - Best for boolean decisions (attend/don't), not value reconstruction

RECOMMENDATIONS:
  - Use ensemble=True when attention generalization matters
  - Start with ensemble_sub_rams=4
  - Increase for more coverage at cost of memory
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
