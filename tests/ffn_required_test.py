#!/usr/bin/env python3
"""
FFN Required Test

Test tasks that ACTUALLY NEED the FFN layer - where attention alone isn't enough.

These tasks require:
1. Attention: Route to correct position
2. FFN: Transform the value

Examples:
- Increment: A→B, B→C, C→D (copy position, transform value)
- Double: Output each value twice with transformation
- XOR combination: Combine values from different positions
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMTransformerBlock import (
    RAMTransformerBlock, RAMTransformer,
    AttentionType, FFNType,
)
from wnn.ram.SoftRAMAttention import ContentMatchMode
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import Tensor

print("=" * 70)
print("FFN Required Test - Tasks That Need Value Transformation")
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

def encode_sequence(s: str) -> list[Tensor]:
    return [encode_char(c) for c in s]


# =============================================================
# Test 1: Increment Task (A→B, B→C, etc.)
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Increment Task (Each char + 1)")
print("=" * 60)

print("\n--- Without FFN (attention only) ---")
model_no_ffn = RAMTransformer(
    input_bits=bits_per_token,
    num_blocks=1,
    attention_type=AttentionType.POSITION_ONLY,
    position_mode=PositionMode.RELATIVE,
    causal=False,
    ffn_type=FFNType.NONE,  # No FFN
    use_residual=False,
    max_seq_len=8,
    rng=42,
)

# Train on ABCD → BCDE
train_input = "ABCD"
train_target = "BCDE"
input_tokens = encode_sequence(train_input)
target_tokens = encode_sequence(train_target)

# Train attention (copy pattern)
model_no_ffn.blocks[0].attention.train_value_projection(input_tokens)
for pos in range(4):
    weights = [0.0] * 4
    weights[pos] = 1.0  # Copy: attend to same position
    model_no_ffn.blocks[0].attention.train_attention_weights(input_tokens, pos, weights)

# Test
test_cases = [
    ("ABCD", "BCDE"),
    ("EFGH", "FGHI"),
    ("WXYZ", "XYZ["),  # Z+1 goes beyond alphabet
]

print("\nWithout FFN (should just copy, not increment):")
for seq, expected in test_cases:
    tokens = encode_sequence(seq)
    model_no_ffn.blocks[0].attention.train_value_projection(tokens)

    outputs = model_no_ffn.forward(tokens)
    result = decode_sequence(outputs)
    status = "✓" if result == expected else "~"
    print(f"  {status} {seq} → {result} (want {expected})")


print("\n--- With BIT_LEVEL FFN ---")
model_with_ffn = RAMTransformer(
    input_bits=bits_per_token,
    num_blocks=1,
    attention_type=AttentionType.POSITION_ONLY,
    position_mode=PositionMode.RELATIVE,
    causal=False,
    ffn_type=FFNType.BIT_LEVEL,  # BIT_LEVEL FFN for transformation
    use_residual=False,
    max_seq_len=8,
    rng=42,
)

# Train attention (copy pattern)
input_tokens = encode_sequence(train_input)
model_with_ffn.blocks[0].attention.train_value_projection(input_tokens)
for pos in range(4):
    weights = [0.0] * 4
    weights[pos] = 1.0
    model_with_ffn.blocks[0].attention.train_attention_weights(input_tokens, pos, weights)

# Train FFN to do increment transformation
print("\nTraining FFN on increment mappings:")
for c in "ABCDEFGHIJKLMNOPQRSTUVWXY":  # Train A→B through Y→Z
    next_c = chr(ord(c) + 1)
    inp = encode_char(c)
    tgt = encode_char(next_c)
    model_with_ffn.blocks[0].ffn.train_mapping(inp, tgt)
    print(f"  {c} → {next_c}")

print("\nWith BIT_LEVEL FFN (trained on A-Y increments):")
for seq, expected in test_cases[:2]:  # Skip Z case for now
    tokens = encode_sequence(seq)
    model_with_ffn.blocks[0].attention.train_value_projection(tokens)

    outputs = model_with_ffn.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    trained = "(trained)" if set(seq).issubset(set("ABCD")) else "(unseen)"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%) {trained}")


# =============================================================
# Test 2: Decrement Task (B→A, C→B, etc.)
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Decrement Task (Each char - 1)")
print("=" * 60)

model = RAMTransformer(
    input_bits=bits_per_token,
    num_blocks=1,
    attention_type=AttentionType.POSITION_ONLY,
    position_mode=PositionMode.RELATIVE,
    causal=False,
    ffn_type=FFNType.BIT_LEVEL,
    use_residual=False,
    max_seq_len=8,
    rng=42,
)

# Train attention (copy pattern)
input_tokens = encode_sequence("BCDE")
model.blocks[0].attention.train_value_projection(input_tokens)
for pos in range(4):
    weights = [0.0] * 4
    weights[pos] = 1.0
    model.blocks[0].attention.train_attention_weights(input_tokens, pos, weights)

# Train FFN for decrement
print("\nTraining FFN on decrement mappings:")
for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ":  # B→A through Z→Y
    prev_c = chr(ord(c) - 1)
    inp = encode_char(c)
    tgt = encode_char(prev_c)
    model.blocks[0].ffn.train_mapping(inp, tgt)

print("\nDecrement results:")
test_cases = [
    ("BCDE", "ABCD"),
    ("FGHI", "EFGH"),
    ("MNOP", "LMNO"),  # Unseen
]

for seq, expected in test_cases:
    tokens = encode_sequence(seq)
    model.blocks[0].attention.train_value_projection(tokens)

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    trained = "(trained)" if set(seq).issubset(set("BCDE")) else "(generalized)"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%) {trained}")


# =============================================================
# Test 3: ROT13 Cipher (more complex transformation)
# =============================================================
print("\n" + "=" * 60)
print("Test 3: ROT13 Cipher (shift by 13)")
print("=" * 60)

def rot13(c: str) -> str:
    if 'A' <= c <= 'Z':
        return chr((ord(c) - ord('A') + 13) % 26 + ord('A'))
    return c

model = RAMTransformer(
    input_bits=bits_per_token,
    num_blocks=1,
    attention_type=AttentionType.POSITION_ONLY,
    position_mode=PositionMode.RELATIVE,
    causal=False,
    ffn_type=FFNType.BIT_LEVEL,
    use_residual=False,
    max_seq_len=8,
    rng=42,
)

# Train attention (copy pattern)
input_tokens = encode_sequence("ABCD")
model.blocks[0].attention.train_value_projection(input_tokens)
for pos in range(4):
    weights = [0.0] * 4
    weights[pos] = 1.0
    model.blocks[0].attention.train_attention_weights(input_tokens, pos, weights)

# Train FFN for ROT13
print("\nTraining FFN on ROT13 mappings:")
for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    rot_c = rot13(c)
    inp = encode_char(c)
    tgt = encode_char(rot_c)
    model.blocks[0].ffn.train_mapping(inp, tgt)
    if c in "ABCN":
        print(f"  {c} → {rot_c}")

print("\nROT13 results:")
test_cases = [
    ("ABCD", "NOPQ"),
    ("HELLO", "URYYB"),
    ("NOPQ", "ABCD"),  # ROT13 is its own inverse
    ("WXYZ", "JKLM"),
]

for seq, expected in test_cases:
    tokens = encode_sequence(seq)
    model.blocks[0].attention.train_value_projection(tokens)

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%)")


# =============================================================
# Test 4: Shift + Transform (multi-block)
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Shift + Increment (2 blocks)")
print("=" * 60)

print("\nTask: Shift right, then increment each character")
print("  ABCD → AABC (shift) → BBCD (increment)")

model = RAMTransformer(
    input_bits=bits_per_token,
    num_blocks=2,
    attention_type=AttentionType.POSITION_ONLY,
    position_mode=PositionMode.RELATIVE,
    causal=True,  # For shift
    ffn_type=FFNType.NONE,  # First block: no FFN
    use_residual=False,
    max_seq_len=8,
    block_configs=[
        {'ffn_type': FFNType.NONE, 'causal': True},      # Block 0: Shift only
        {'ffn_type': FFNType.BIT_LEVEL, 'causal': False}, # Block 1: Copy + Increment
    ],
    rng=42,
)

# Train block 0: shift
input_tokens = encode_sequence("ABCD")
model.blocks[0].attention.train_value_projection(input_tokens)
for pos in range(4):
    weights = [0.0] * 4
    if pos > 0:
        weights[pos - 1] = 1.0
    else:
        weights[0] = 1.0
    model.blocks[0].attention.train_attention_weights(input_tokens, pos, weights)

# Train block 1: copy + increment
shifted_tokens = encode_sequence("AABC")
model.blocks[1].attention.train_value_projection(shifted_tokens)
for pos in range(4):
    weights = [0.0] * 4
    weights[pos] = 1.0
    model.blocks[1].attention.train_attention_weights(shifted_tokens, pos, weights)

# Train FFN for increment
for c in "ABCDEFGHIJKLMNOPQRSTUVWXY":
    next_c = chr(ord(c) + 1)
    inp = encode_char(c)
    tgt = encode_char(next_c)
    model.blocks[1].ffn.train_mapping(inp, tgt)

def shift_then_increment(s: str) -> str:
    shifted = s[0] + s[:-1]  # Shift right
    return ''.join(chr(min(ord(c) + 1, ord('Z'))) for c in shifted)  # Increment

print("\nShift + Increment results:")
test_cases = [
    "ABCD",
    "EFGH",
    "WXYZ",
]

for seq in test_cases:
    expected = shift_then_increment(seq)
    tokens = encode_sequence(seq)

    # Train value projections
    model.blocks[0].attention.train_value_projection(tokens)
    shifted = seq[0] + seq[:-1]
    model.blocks[1].attention.train_value_projection(encode_sequence(shifted))

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    trained = "(trained)" if seq == "ABCD" else "(unseen)"
    print(f"  {status} {seq} → shift → {shifted} → inc → {result} (want {expected}, {pct:.0f}%) {trained}")


# =============================================================
# Test 5: XOR Combination (requires FFN to combine)
# =============================================================
print("\n" + "=" * 60)
print("Test 5: XOR with Previous (attention + residual)")
print("=" * 60)

print("\nTask: XOR each position with the previous position")
print("  Uses attention to get previous + XOR residual")

model = RAMTransformer(
    input_bits=bits_per_token,
    num_blocks=1,
    attention_type=AttentionType.POSITION_ONLY,
    position_mode=PositionMode.RELATIVE,
    causal=True,
    ffn_type=FFNType.NONE,
    use_residual=True,  # XOR residual: output = input XOR attention_output
    max_seq_len=8,
    rng=42,
)

# Train to attend to previous position
input_tokens = encode_sequence("ABCD")
model.blocks[0].attention.train_value_projection(input_tokens)
for pos in range(4):
    weights = [0.0] * 4
    if pos > 0:
        weights[pos - 1] = 1.0  # Attend to previous
    else:
        weights[0] = 1.0  # First position attends to itself
    model.blocks[0].attention.train_attention_weights(input_tokens, pos, weights)

print("\nXOR with previous (residual connection):")
test_cases = ["ABCD", "AAAA", "ABAB"]

for seq in test_cases:
    tokens = encode_sequence(seq)
    model.blocks[0].attention.train_value_projection(tokens)

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)

    # Compute expected: each position XOR with previous
    expected_bits = []
    for i, t in enumerate(tokens):
        if i == 0:
            prev = t  # XOR with self = 0
        else:
            prev = tokens[i-1]
        expected_bits.append(t ^ prev)
    expected = decode_sequence(expected_bits)

    print(f"  {seq} → {result}")
    print(f"    (each pos XOR prev: {expected})")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: When FFN is Required")
print("=" * 60)

print("""
TASKS THAT NEED FFN:

1. VALUE TRANSFORMATION (attention routes, FFN transforms):
   - Increment: A→B, B→C, ... (trained 100%, generalization varies)
   - Decrement: B→A, C→B, ...
   - ROT13: Complex bit pattern transformation
   - Any character mapping/cipher

2. MULTI-STEP WITH TRANSFORMATION:
   - Shift + Increment: Block 1 routes, Block 2 transforms
   - Sort + Transform: Computed sort, then FFN transforms values

3. TASKS THAT DON'T NEED FFN:
   - Copy, Shift, Reverse (position routing only)
   - Sorting (computed comparison)
   - Self-matching (computed XOR equality)

KEY INSIGHT:
  FFN is needed when the OUTPUT VALUE differs from the INPUT VALUE
  at the attended position. If you just need to route tokens to
  different positions, attention alone suffices.

BIT_LEVEL FFN GENERALIZATION:
  - Trains per-bit transformations
  - May generalize if transformation is bit-regular
  - Increment: A=00000→B=00001 is just adding 1 to LSB (partial generalization)
  - ROT13: More complex bit patterns (less generalization)
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
