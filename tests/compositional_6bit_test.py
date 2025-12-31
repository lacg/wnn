#!/usr/bin/env python3
"""
COMPOSITIONAL Strategy with 6-bit Encoding

The issue: 5 bits can't divide evenly into meaningful groups.
Solution: Use 6 bits (2 groups of 3, or 3 groups of 2).

This allows COMPOSITIONAL to properly handle carry propagation.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMGeneralization import (
    MapperStrategy, GeneralizingProjection, CompositionalMapper,
    ContextMode, OutputMode
)
from torch import zeros, uint8, tensor, Tensor
from torch.nn import Module

print("=" * 70)
print("COMPOSITIONAL Strategy with 6-bit Encoding")
print("=" * 70)

# =============================================================
# Custom 6-bit Encoder/Decoder
# =============================================================
class SixBitEncoder:
    """
    Encodes A-Z using 6 bits (allows 64 values, we use 26).
    This enables proper COMPOSITIONAL grouping (2x3 or 3x2).
    """
    def __init__(self):
        self.vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.bits_per_token = 6
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

    def encode(self, char: str) -> Tensor:
        """Encode character to 6-bit tensor."""
        idx = self.char_to_idx[char]
        bits = zeros(6, dtype=uint8)
        for i in range(5, -1, -1):
            bits[5 - i] = (idx >> i) & 1
        return bits

    def decode(self, bits: Tensor) -> str:
        """Decode 6-bit tensor to character."""
        bits = bits.squeeze()
        idx = 0
        for bit in bits:
            idx = (idx << 1) | int(bit)
        if idx >= 26:
            return '?'
        return self.idx_to_char[idx]

encoder = SixBitEncoder()
bits_per_token = encoder.bits_per_token

def encode_char(c: str) -> Tensor:
    return encoder.encode(c)

def decode_bits(bits: Tensor) -> str:
    return encoder.decode(bits)

def decode_sequence(tokens: list) -> str:
    return ''.join(decode_bits(t) for t in tokens)

print(f"Using {bits_per_token}-bit encoding")
print(f"  A = {encode_char('A').tolist()}")
print(f"  B = {encode_char('B').tolist()}")
print(f"  Z = {encode_char('Z').tolist()}")


# =============================================================
# Autoregressive Model
# =============================================================
class AutoregressiveModel(Module):
    def __init__(
        self,
        bits: int,
        strategy: MapperStrategy,
        n_groups: int = 2,
        rng: int | None = None,
    ):
        super().__init__()
        self.bits = bits
        self.strategy = strategy

        self.transition = GeneralizingProjection(
            input_bits=bits,
            output_bits=bits,
            strategy=strategy,
            n_groups=n_groups,
            rng=rng,
        )

        print(f"[Model] bits={bits}, strategy={strategy.name}, n_groups={n_groups}")

    def generate(self, start: Tensor, length: int) -> list[Tensor]:
        start = start.squeeze()
        outputs = [start.clone()]
        current = start

        for _ in range(length - 1):
            next_token = self.transition(current)
            if next_token.ndim > 1:
                next_token = next_token.squeeze()
            outputs.append(next_token.clone())
            current = next_token

        return outputs

    def train_transition(self, from_token: Tensor, to_token: Tensor) -> int:
        return self.transition.train_mapping(from_token, to_token)


# =============================================================
# Test configurations
# =============================================================
strategies_to_test = [
    ("DIRECT", MapperStrategy.DIRECT, 2),
    ("BIT_LEVEL", MapperStrategy.BIT_LEVEL, 2),
    ("COMPOSITIONAL (2x3)", MapperStrategy.COMPOSITIONAL, 2),  # 2 groups of 3 bits
    ("COMPOSITIONAL (3x2)", MapperStrategy.COMPOSITIONAL, 3),  # 3 groups of 2 bits
]


# =============================================================
# Test 1: Counting (Increment)
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Counting (A→B→C→...) with 8 Training Examples")
print("=" * 60)

for name, strategy, n_groups in strategies_to_test:
    print(f"\n--- {name} ---")

    model = AutoregressiveModel(
        bits=bits_per_token,
        strategy=strategy,
        n_groups=n_groups,
        rng=42,
    )

    # Train on A→B, B→C, ..., H→I (8 transitions)
    train_chars = "ABCDEFGHI"
    for epoch in range(3):
        for i in range(len(train_chars) - 1):
            model.train_transition(
                encode_char(train_chars[i]),
                encode_char(train_chars[i + 1])
            )

    # Test from A (trained path)
    generated = model.generate(encode_char('A'), length=10)
    result = decode_sequence(generated)
    expected = "ABCDEFGHIJ"
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    print(f"  From 'A': '{result}' ({100*correct/len(expected):.0f}%)")

    # Test from M (unseen)
    generated = model.generate(encode_char('M'), length=6)
    result = decode_sequence(generated)
    expected = "MNOPQR"
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    print(f"  From 'M': '{result}' ({100*correct/len(expected):.0f}%) <- UNSEEN")


# =============================================================
# Test 2: Identity (Copy)
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Identity (Copy) with 3 Training Examples")
print("=" * 60)

for name, strategy, n_groups in strategies_to_test:
    print(f"\n--- {name} ---")

    model = AutoregressiveModel(
        bits=bits_per_token,
        strategy=strategy,
        n_groups=n_groups,
        rng=123,
    )

    # Train on just A→A, M→M, Z→Z
    for c in ['A', 'M', 'Z']:
        for _ in range(3):
            model.train_transition(encode_char(c), encode_char(c))

    # Test on all 26 letters
    correct = 0
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        generated = model.generate(encode_char(c), length=4)
        result = decode_sequence(generated)
        if result == c * 4:
            correct += 1

    print(f"  Accuracy on all 26 letters: {100*correct/26:.0f}%")

    # Show examples
    for c in ['B', 'K', 'X']:
        generated = model.generate(encode_char(c), length=5)
        result = decode_sequence(generated)
        status = "OK" if result == c * 5 else "X"
        print(f"    [{status}] '{c}' -> '{result}'")


# =============================================================
# Test 3: Decrement
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Decrement (Z→Y→X→...) with 8 Training Examples")
print("=" * 60)

for name, strategy, n_groups in strategies_to_test:
    print(f"\n--- {name} ---")

    model = AutoregressiveModel(
        bits=bits_per_token,
        strategy=strategy,
        n_groups=n_groups,
        rng=456,
    )

    # Train on Z→Y, Y→X, ..., S→R
    train_chars = "ZYXWVUTSR"
    for epoch in range(3):
        for i in range(len(train_chars) - 1):
            model.train_transition(
                encode_char(train_chars[i]),
                encode_char(train_chars[i + 1])
            )

    # Test from Z (trained)
    generated = model.generate(encode_char('Z'), length=8)
    result = decode_sequence(generated)
    expected = "ZYXWVUTS"
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    print(f"  From 'Z': '{result}' ({100*correct/len(expected):.0f}%)")

    # Test from M (unseen)
    generated = model.generate(encode_char('M'), length=6)
    result = decode_sequence(generated)
    expected = "MLKJIH"
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    print(f"  From 'M': '{result}' ({100*correct/len(expected):.0f}%) <- UNSEEN")


# =============================================================
# Test 4: Skip-2
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Skip-2 (A→C→E→...) with 6 Training Examples")
print("=" * 60)

for name, strategy, n_groups in strategies_to_test:
    print(f"\n--- {name} ---")

    model = AutoregressiveModel(
        bits=bits_per_token,
        strategy=strategy,
        n_groups=n_groups,
        rng=789,
    )

    # Train A→C, C→E, E→G, G→I, I→K, K→M
    train_pairs = [('A','C'), ('C','E'), ('E','G'), ('G','I'), ('I','K'), ('K','M')]
    for epoch in range(3):
        for from_c, to_c in train_pairs:
            model.train_transition(encode_char(from_c), encode_char(to_c))

    # Test from A (trained)
    generated = model.generate(encode_char('A'), length=6)
    result = decode_sequence(generated)
    expected = "ACEGIK"
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    print(f"  From 'A': '{result}' ({100*correct/len(expected):.0f}%)")

    # Test from B (unseen - odd positions)
    generated = model.generate(encode_char('B'), length=6)
    result = decode_sequence(generated)
    expected = "BDFHJL"
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    print(f"  From 'B': '{result}' ({100*correct/len(expected):.0f}%) <- UNSEEN")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: 6-bit Encoding Results")
print("=" * 60)

print("""
With 6-bit encoding, COMPOSITIONAL can use proper groups:
  - 2 groups of 3 bits each
  - 3 groups of 2 bits each

This enables meaningful carry propagation between groups.

Key Questions Answered:
1. Does COMPOSITIONAL improve with proper grouping?
2. Which grouping (2x3 vs 3x2) works better?
3. Is BIT_LEVEL still the best for arithmetic?
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
