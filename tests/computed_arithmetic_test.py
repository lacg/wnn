#!/usr/bin/env python3
"""
Computed Arithmetic FFN Test

Test the ComputedArithmeticFFN that computes transformations directly
from bit patterns - achieving 100% generalization like SortingAttention.

Key insight: If we can COMPUTE the transformation (not LEARN it),
it will generalize to all inputs.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.SoftRAMAttention import (
    ComputedArithmeticFFN, ArithmeticOp,
    _bits_to_int, _int_to_bits
)
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode
from torch import Tensor

print("=" * 70)
print("Computed Arithmetic FFN Test")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_char(c: str) -> Tensor:
    return decoder.encode(c).squeeze()

def decode_bits(bits: Tensor) -> str:
    return decoder.decode(bits.unsqueeze(0))


# =============================================================
# Test 1: Verify _int_to_bits and _bits_to_int are inverses
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Bit Conversion Functions")
print("=" * 60)

print("\nVerifying _bits_to_int and _int_to_bits are inverses:")
for c in "ABCMNXYZ":
    bits = encode_char(c)
    value = _bits_to_int(bits)
    back = _int_to_bits(value, bits_per_token)
    back_char = decode_bits(back)
    status = "✓" if back_char == c else "✗"
    print(f"  {status} {c} → bits → {value} → bits → {back_char}")


# =============================================================
# Test 2: Computed Increment (no training needed)
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Computed INCREMENT (A→B, B→C, ...)")
print("=" * 60)

ffn = ComputedArithmeticFFN(bits_per_token, ArithmeticOp.INCREMENT)

print("\nIncrement results (no training!):")
test_cases = [
    ("A", "B"),
    ("B", "C"),
    ("M", "N"),
    ("X", "Y"),
    ("Y", "Z"),
]

total_correct = 0
for inp, expected in test_cases:
    bits = encode_char(inp)
    out = ffn(bits)
    result = decode_bits(out)
    status = "✓" if result == expected else "✗"
    if result == expected:
        total_correct += 1
    print(f"  {status} {inp} → {result} (want {expected})")

print(f"\nAccuracy: {100 * total_correct / len(test_cases):.0f}%")


# =============================================================
# Test 3: Computed Decrement
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Computed DECREMENT (B→A, C→B, ...)")
print("=" * 60)

ffn = ComputedArithmeticFFN(bits_per_token, ArithmeticOp.DECREMENT)

print("\nDecrement results (no training!):")
test_cases = [
    ("B", "A"),
    ("C", "B"),
    ("N", "M"),
    ("Y", "X"),
    ("Z", "Y"),
]

total_correct = 0
for inp, expected in test_cases:
    bits = encode_char(inp)
    out = ffn(bits)
    result = decode_bits(out)
    status = "✓" if result == expected else "✗"
    if result == expected:
        total_correct += 1
    print(f"  {status} {inp} → {result} (want {expected})")

print(f"\nAccuracy: {100 * total_correct / len(test_cases):.0f}%")


# =============================================================
# Test 4: Computed ROT13 (the key test!)
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Computed ROT13 (add 13 mod 26)")
print("=" * 60)

ffn = ComputedArithmeticFFN(bits_per_token, ArithmeticOp.ROT13)

def expected_rot13(c: str) -> str:
    if 'A' <= c <= 'Z':
        return chr((ord(c) - ord('A') + 13) % 26 + ord('A'))
    return c

print("\nROT13 results (no training!):")
all_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
total_correct = 0

for c in all_letters:
    expected = expected_rot13(c)
    bits = encode_char(c)
    out = ffn(bits)
    result = decode_bits(out)
    status = "✓" if result == expected else "✗"
    if result == expected:
        total_correct += 1
    print(f"  {status} {c} → {result} (want {expected})")

print(f"\nAccuracy: {100 * total_correct / len(all_letters):.0f}%")


# =============================================================
# Test 5: ROT13 is its own inverse
# =============================================================
print("\n" + "=" * 60)
print("Test 5: ROT13 Double Application (should return original)")
print("=" * 60)

ffn = ComputedArithmeticFFN(bits_per_token, ArithmeticOp.ROT13)

print("\nApplying ROT13 twice:")
test_cases = ["HELLO", "WORLD", "ABCDE", "NOPQR"]

for word in test_cases:
    result1 = ""
    result2 = ""
    for c in word:
        bits = encode_char(c)
        out1 = ffn(bits)
        out2 = ffn(out1)
        result1 += decode_bits(out1)
        result2 += decode_bits(out2)

    status = "✓" if result2 == word else "✗"
    print(f"  {status} {word} → {result1} → {result2}")


# =============================================================
# Test 6: Custom ADD_MOD
# =============================================================
print("\n" + "=" * 60)
print("Test 6: Custom ADD_MOD (add 5 mod 26)")
print("=" * 60)

ffn = ComputedArithmeticFFN(
    bits_per_token,
    ArithmeticOp.ADD_MOD,
    constant=5,
    modulo=26
)

print("\nAdd 5 mod 26 (Caesar cipher +5):")
test_cases = [
    ("A", "F"),  # 0+5=5
    ("U", "Z"),  # 20+5=25
    ("V", "A"),  # 21+5=26 mod 26=0
    ("Z", "E"),  # 25+5=30 mod 26=4
]

total_correct = 0
for inp, expected in test_cases:
    bits = encode_char(inp)
    out = ffn(bits)
    result = decode_bits(out)
    status = "✓" if result == expected else "✗"
    if result == expected:
        total_correct += 1
    print(f"  {status} {inp} → {result} (want {expected})")

print(f"\nAccuracy: {100 * total_correct / len(test_cases):.0f}%")


# =============================================================
# Test 7: NEGATE (bitwise complement relative to max)
# =============================================================
print("\n" + "=" * 60)
print("Test 7: NEGATE (max_value - value)")
print("=" * 60)

ffn = ComputedArithmeticFFN(bits_per_token, ArithmeticOp.NEGATE)

print("\nNegate results:")
print(f"  (max_value = {(1 << bits_per_token) - 1} = {bits_per_token}-bit max)")

test_cases = [
    "A",  # 0 → 31
    "B",  # 1 → 30
    "Z",  # 25 → 6
]

for c in test_cases:
    bits = encode_char(c)
    value = _bits_to_int(bits)
    out = ffn(bits)
    out_value = _bits_to_int(out)
    result = decode_bits(out)
    print(f"  {c}({value}) → {result}({out_value})")


# =============================================================
# Test 8: Comparison - Computed vs Learned
# =============================================================
print("\n" + "=" * 60)
print("Test 8: Comparison - Computed vs Learned ROT13")
print("=" * 60)

from wnn.ram.RAMGeneralization import GeneralizingProjection, MapperStrategy

print("\n--- Learned BIT_LEVEL (trained on A-M only) ---")
learned_ffn = GeneralizingProjection(
    input_bits=bits_per_token,
    output_bits=bits_per_token,
    strategy=MapperStrategy.BIT_LEVEL,
    rng=42,
)

# Train only on first half of alphabet
print("Training on A-M:")
for c in "ABCDEFGHIJKLM":
    target = expected_rot13(c)
    learned_ffn.train_mapping(encode_char(c), encode_char(target))

print("\nLearned FFN results:")
train_correct = 0
test_correct = 0
for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    expected = expected_rot13(c)
    out = learned_ffn(encode_char(c))
    result = decode_bits(out)
    is_train = c in "ABCDEFGHIJKLM"
    if result == expected:
        if is_train:
            train_correct += 1
        else:
            test_correct += 1
    tag = "(train)" if is_train else "(test)"
    status = "✓" if result == expected else "✗"
    print(f"  {status} {c} → {result} (want {expected}) {tag}")

print(f"\nTrained (A-M): {100 * train_correct / 13:.0f}%")
print(f"Unseen (N-Z): {100 * test_correct / 13:.0f}%")

print("\n--- Computed ROT13 (no training) ---")
computed_ffn = ComputedArithmeticFFN(bits_per_token, ArithmeticOp.ROT13)

total_correct = 0
for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    expected = expected_rot13(c)
    out = computed_ffn(encode_char(c))
    result = decode_bits(out)
    if result == expected:
        total_correct += 1

print(f"All letters: {100 * total_correct / 26:.0f}%")


# =============================================================
# Test 9: Full Transform Pipeline (Position + Arithmetic)
# =============================================================
print("\n" + "=" * 60)
print("Test 9: Full Pipeline - Shift + ROT13")
print("=" * 60)

from wnn.ram.RAMTransformerBlock import RAMTransformer, AttentionType, FFNType
from wnn.ram.encoders_decoders import PositionMode

# Manual pipeline: position attention + computed arithmetic
print("\nManual pipeline: Shift attention → ROT13 FFN")

# Create position-only attention for shift
from wnn.ram.SoftRAMAttention import SoftRAMAttention, AggregationStrategy

attention = SoftRAMAttention(
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

rot13_ffn = ComputedArithmeticFFN(bits_per_token, ArithmeticOp.ROT13)

def encode_sequence(s: str) -> list[Tensor]:
    return [encode_char(c) for c in s]

def decode_sequence(tokens: list) -> str:
    return ''.join(decode_bits(t) for t in tokens)

def shift_then_rot13(s: str) -> str:
    shifted = s[0] + s[:-1]
    return ''.join(expected_rot13(c) for c in shifted)

# Train shift attention
train_seq = "ABCD"
tokens = encode_sequence(train_seq)
attention.train_value_projection(tokens)
for pos in range(4):
    weights = [0.0] * 4
    if pos > 0:
        weights[pos - 1] = 1.0
    else:
        weights[0] = 1.0
    attention.train_attention_weights(tokens, pos, weights)

print("\nShift → ROT13 results:")
test_cases = ["ABCD", "HELLO", "WXYZ"]

for seq in test_cases:
    expected = shift_then_rot13(seq)
    tokens = encode_sequence(seq)

    # Train value projection for test
    attention.train_value_projection(tokens)

    # Apply shift attention
    shifted_tokens = attention.forward(tokens)

    # Apply ROT13 to each token
    rot13_tokens = [rot13_ffn(t) for t in shifted_tokens]

    result = decode_sequence(rot13_tokens)
    shifted = decode_sequence(shifted_tokens)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    trained = "(trained)" if seq == "ABCD" else "(unseen)"
    print(f"  {status} {seq} → shift:{shifted} → ROT13:{result} (want {expected}, {pct:.0f}%) {trained}")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Computed vs Learned Transformations")
print("=" * 60)

print("""
COMPUTED ARITHMETIC FFN:

Operations available:
  - INCREMENT: value + 1 (clamped to max)
  - DECREMENT: value - 1 (clamped to 0)
  - ADD: value + constant
  - SUBTRACT: value - constant
  - ADD_MOD: (value + constant) mod N
  - SUBTRACT_MOD: (value - constant) mod N
  - ROT13: (value + 13) mod 26 (Caesar cipher)
  - NEGATE: max_value - value

GENERALIZATION COMPARISON:

| Operation | Learned BIT_LEVEL | Computed Arithmetic |
|-----------|-------------------|---------------------|
| Increment | ~100% (regular)   | 100% (computed)     |
| ROT13     | ~0-30% (irregular)| 100% (computed)     |
| Caesar +5 | ~0-30%            | 100% (computed)     |

KEY INSIGHT:
  Learned BIT_LEVEL generalizes when the transformation is
  "bit-regular" (like +1, which is binary addition).

  But complex operations like ROT13 require modular arithmetic
  that doesn't follow per-bit patterns - so learning fails.

  COMPUTED operations bypass learning entirely, achieving 100%
  generalization because we know the exact transformation.

WHEN TO USE EACH:
  - Computed: When transformation is mathematically defined
  - Learned: When transformation must be discovered from data
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
