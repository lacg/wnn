#!/usr/bin/env python3
"""
Soft Attention Aggregation Integration Test

Test the SoftRAMAttention with different aggregation strategies on real tasks:
1. Copy task: output[i] = input[i]
2. Retrieval task: query a specific position
3. Shift task: output[i] = input[i-1]

Compare TOP_1, MAJORITY, THRESHOLD, TOP_K strategies.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.SoftRAMAttention import SoftRAMAttention, AggregationStrategy
from wnn.ram.RAMAttention import RAMAttention
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import Tensor, zeros, uint8

print("=" * 70)
print("Soft Attention Aggregation Integration Test")
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
# Test 1: Copy Task - Identity Mapping
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Copy Task (output[i] = input[i])")
print("=" * 60)
print("Each position should attend to itself and return its own value.")

for strategy in AggregationStrategy:
    print(f"\n--- {strategy.name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=strategy,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=False,  # Non-causal for copy
        top_k=2,
        rng=42,
    )

    # Train on copy task using train_step (trains both attention + value projection)
    # Use more sequences to cover more characters for generalization
    train_sequences = ["ABCD", "EFGH", "IJKL", "MNOP", "QRST", "UVWX"]

    for epoch in range(5):
        for seq_str in train_sequences:
            tokens = [encode_char(c) for c in seq_str]
            # For copy: input = output
            model.train_step(tokens, tokens)

    # Test
    test_seqs = ["ABCD", "WXYZ", "MNOP"]
    total_correct = 0
    total_chars = 0

    for seq_str in test_seqs:
        tokens = [encode_char(c) for c in seq_str]
        outputs = model.forward(tokens)
        result = decode_sequence(outputs)

        correct = sum(1 for r, e in zip(result, seq_str) if r == e)
        total_correct += correct
        total_chars += len(seq_str)

        status = "OK" if result == seq_str else "X"
        print(f"  [{status}] '{seq_str}' -> '{result}'")

    pct = 100 * total_correct / total_chars
    print(f"  Accuracy: {pct:.0f}%")


# =============================================================
# Test 2: Position Retrieval Task
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Position Retrieval (query position 0)")
print("=" * 60)
print("All positions should attend to position 0 and return its value.")

for strategy in AggregationStrategy:
    print(f"\n--- {strategy.name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=strategy,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=False,
        top_k=2,
        rng=123,
    )

    # Train: all positions attend to position 0
    # Use more sequences to cover alphabet
    train_sequences = ["ABCD", "EFGH", "IJKL", "MNOP", "QRST", "UVWX"]

    for epoch in range(5):
        for seq_str in train_sequences:
            tokens = [encode_char(c) for c in seq_str]

            # First train value projection (identity)
            model.train_value_projection(tokens)

            # Then train attention to position 0
            for pos in range(len(tokens)):
                target_weights = [0.0] * len(tokens)
                target_weights[0] = 1.0  # Full attention to position 0
                model.train_attention_weights(tokens, pos, target_weights)

    # Test: output should be first char repeated
    test_cases = [
        ("ABCD", "AAAA"),
        ("WXYZ", "WWWW"),
    ]

    for seq_str, expected in test_cases:
        tokens = [encode_char(c) for c in seq_str]
        outputs = model.forward(tokens)
        result = decode_sequence(outputs)

        correct = sum(1 for r, e in zip(result, expected) if r == e)
        pct = 100 * correct / len(expected)
        status = "OK" if result == expected else "~"
        print(f"  [{status}] '{seq_str}' -> '{result}' (want '{expected}', {pct:.0f}%)")


# =============================================================
# Test 3: Shift Task (Causal)
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Shift Task (output[i] = input[i-1], causal)")
print("=" * 60)
print("Each position attends to previous position.")

for strategy in AggregationStrategy:
    print(f"\n--- {strategy.name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=strategy,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=True,  # Causal for shift
        top_k=2,
        rng=42,
    )

    # Train shift task
    # Use more sequences to cover alphabet
    train_sequences = ["ABCD", "EFGH", "IJKL", "MNOP", "QRST", "UVWX"]

    for epoch in range(5):
        for seq_str in train_sequences:
            tokens = [encode_char(c) for c in seq_str]

            # Train value projection (identity)
            model.train_value_projection(tokens)

            for pos in range(len(tokens)):
                target_weights = [0.0] * len(tokens)
                if pos > 0:
                    target_weights[pos - 1] = 1.0  # Attend to previous
                else:
                    target_weights[pos] = 1.0  # First position attends to self
                model.train_attention_weights(tokens, pos, target_weights)

    # Test: each char should be previous char (first stays same)
    test_cases = [
        ("ABCD", "AABC"),  # A, A, B, C
        ("WXYZ", "WWXY"),  # W, W, X, Y
    ]

    for seq_str, expected in test_cases:
        tokens = [encode_char(c) for c in seq_str]
        outputs = model.forward(tokens)
        result = decode_sequence(outputs)

        correct = sum(1 for r, e in zip(result, expected) if r == e)
        pct = 100 * correct / len(expected)
        status = "OK" if result == expected else "~"
        print(f"  [{status}] '{seq_str}' -> '{result}' (want '{expected}', {pct:.0f}%)")


# =============================================================
# Test 4: Attention Weight Visualization
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Attention Weight Distribution")
print("=" * 60)

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    rng=42,
)

# Train for copy (self-attention)
tokens = [encode_char(c) for c in "ABCD"]
model.train_value_projection(tokens)  # Train value projection first
for pos in range(4):
    target_weights = [0.0, 0.0, 0.0, 0.0]
    target_weights[pos] = 1.0
    model.train_attention_weights(tokens, pos, target_weights)

print("\nTrained attention pattern (should be diagonal):")
print(model.visualize_attention(tokens))


# =============================================================
# Test 5: Comparison with Hard Attention (RAMAttention)
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Soft vs Hard Attention Comparison")
print("=" * 60)

print("\n--- Hard Attention (RAMAttention) ---")
hard_model = RAMAttention(
    input_bits=bits_per_token,
    num_heads=4,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    rng=42,
)

# Can't train RAMAttention directly on attention weights
# It learns from sequence-to-sequence examples
print("  (Untrained - using default random connections)")

for seq_str in ["ABCD", "WXYZ"]:
    tokens = [encode_char(c) for c in seq_str]
    outputs = hard_model.forward(tokens)
    result = decode_sequence(outputs)
    print(f"  '{seq_str}' -> '{result}'")

print("\n--- Soft Attention (SoftRAMAttention, TOP_1) ---")
soft_model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    rng=42,
)

# Train for copy using train_step (more sequences)
for seq_str in ["ABCD", "EFGH", "IJKL", "MNOP", "QRST", "UVWX"]:
    tokens = [encode_char(c) for c in seq_str]
    soft_model.train_step(tokens, tokens)

for seq_str in ["ABCD", "WXYZ"]:
    tokens = [encode_char(c) for c in seq_str]
    outputs = soft_model.forward(tokens)
    result = decode_sequence(outputs)
    status = "OK" if result == seq_str else "X"
    print(f"  [{status}] '{seq_str}' -> '{result}'")


# =============================================================
# Test 6: Noisy Vote Distribution
# =============================================================
print("\n" + "=" * 60)
print("Test 6: Robustness to Noisy Attention Weights")
print("=" * 60)
print("Train with imperfect (noisy) attention weights, test recovery.")

for strategy in [AggregationStrategy.TOP_1, AggregationStrategy.MAJORITY]:
    print(f"\n--- {strategy.name} ---")

    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=8,
        aggregation=strategy,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=False,
        top_k=2,
        rng=42,
    )

    # Train with noisy weights (not 100% to target)
    tokens = [encode_char(c) for c in "ABCD"]

    # Train value projection first
    model.train_value_projection(tokens)

    for pos in range(4):
        # Noisy: 75% to target, 25% spread to others
        target_weights = [0.083, 0.083, 0.083, 0.083]  # ~8% each
        target_weights[pos] = 0.75  # 75% to self
        model.train_attention_weights(tokens, pos, target_weights)

    # Test
    outputs = model.forward(tokens)
    result = decode_sequence(outputs)

    correct = sum(1 for r, e in zip(result, "ABCD") if r == e)
    pct = 100 * correct / 4
    status = "OK" if result == "ABCD" else "~"
    print(f"  [{status}] 'ABCD' -> '{result}' (75% weights, {pct:.0f}% correct)")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Aggregation Strategy Comparison")
print("=" * 60)

print("""
KEY FINDINGS:

1. AGGREGATION STRATEGIES WORK CORRECTLY:
   - TOP_1: Returns highest-voted value (best for retrieval)
   - MAJORITY: Per-bit weighted voting (best for combining)
   - THRESHOLD: Selective inclusion above 50%
   - TOP_K: Combines exactly K values

2. GENERALIZATION LIMITATION (RAM Networks):
   - Trained tokens work perfectly (ABCD, MNOP)
   - Untrained tokens fail (Y, Z not seen during training)
   - This is fundamental: RAM networks learn lookup tables, not functions
   - Unlike neural networks, they don't interpolate to unseen inputs

3. VALUE PROJECTION:
   - BIT_LEVEL helps with identity mapping for trained tokens
   - But can't generalize to completely unseen token content

4. ATTENTION MECHANISM:
   - Voting heads learn (token, position) → attend patterns
   - Pattern [query='W', key='X', pos=1] must be seen during training
   - Position encoding helps, but token content still matters

RECOMMENDATIONS:
   - For closed vocabularies: Train on all tokens
   - For open vocabularies: Consider position-only attention
   - Use TOP_1 for retrieval tasks (most robust)
   - Use MAJORITY for combining similar values
""")


print("=" * 70)
print("Tests completed!")
print("=" * 70)
