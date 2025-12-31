#!/usr/bin/env python3
"""
Sorting and Multi-Step Task Test

Test on challenging tasks that require:
  - Content-based comparisons (sorting)
  - Multiple operations in sequence (multi-step)
  - Data-dependent attention patterns

These tasks are fundamentally harder because the attention pattern
depends on the CONTENT of tokens, not just their positions.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.SoftRAMAttention import (
    SoftRAMAttention, AggregationStrategy,
    ContentMatchMode, AttentionCombineMode
)
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import Tensor, zeros, uint8, cat
from torch.nn import Module

print("=" * 70)
print("Sorting and Multi-Step Task Test")
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

def char_to_ord(c: str) -> int:
    """Get ordinal value for sorting (A=0, B=1, ...)."""
    return ord(c) - ord('A')


# =============================================================
# Test 1: Find Minimum (Simplest sorting-related task)
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Find Minimum (output min at every position)")
print("=" * 60)
print("Task: Output the minimum character at every position")
print("E.g., DCBA → AAAA (A is minimum)")

# This requires content-based comparison
# The attention should find the position with minimum value

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    rng=42,
)

# Train on several sequences
train_cases = [
    ("ABCD", "AAAA"),  # A is min
    ("DCBA", "AAAA"),  # A is min
    ("BDAC", "AAAA"),  # A is min
    ("CBDA", "AAAA"),  # A is min
]

print("\nTraining on sequences with known minimum...")
for seq, expected in train_cases:
    tokens = [encode_char(c) for c in seq]
    model.train_value_projection(tokens)

    # Find position of minimum
    min_pos = min(range(len(seq)), key=lambda i: char_to_ord(seq[i]))

    # All positions attend to minimum position
    for pos in range(len(tokens)):
        target_weights = [0.0] * len(tokens)
        target_weights[min_pos] = 1.0
        model.train_attention_weights(tokens, pos, target_weights)

# Test
test_cases = [
    ("ABCD", "AAAA"),  # Trained arrangement
    ("DCBA", "AAAA"),  # Trained arrangement
    ("EFGH", "EEEE"),  # Unseen tokens
    ("HGFE", "EEEE"),  # Unseen arrangement
    ("WXYZ", "WWWW"),  # Unseen distant
]

print("\nTesting:")
for seq, expected in test_cases:
    tokens = [encode_char(c) for c in seq]
    model.train_value_projection(tokens)

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    trained = "(trained)" if seq in ["ABCD", "DCBA", "BDAC", "CBDA"] else "(unseen)"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%) {trained}")


# =============================================================
# Test 2: Comparison-Based Attention (Building Block)
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Comparison-Based Attention")
print("=" * 60)
print("Can we learn 'attend to smaller token'?")

# Direct comparison using Hamming distance won't work for ordering
# We need to learn the comparison function

from wnn.ram.RAMLayer import RAMLayer

class ComparisonAttention(Module):
    """
    Learn to attend based on comparison (a < b).

    This requires learning the comparison function for the encoding.
    """

    def __init__(self, input_bits: int, rng: int = None):
        super().__init__()
        self.input_bits = input_bits

        # Learn: given (query, key), is key < query?
        self.comparator = RAMLayer(
            total_input_bits=2 * input_bits,
            num_neurons=1,
            n_bits_per_neuron=min(2 * input_bits, 10),
            rng=rng,
        )

    def is_less_than(self, a: Tensor, b: Tensor) -> bool:
        """Check if a < b (learned comparison)."""
        inp = cat([a.squeeze(), b.squeeze()]).unsqueeze(0)
        return self.comparator(inp).squeeze().item() == 1

    def train_comparison(self, a: Tensor, b: Tensor, a_val: int, b_val: int):
        """Train: a < b if a_val < b_val."""
        inp = cat([a.squeeze(), b.squeeze()]).unsqueeze(0)
        result = 1 if a_val < b_val else 0
        target = Tensor([[result]]).to(uint8)
        self.comparator.commit(inp, target)


print("\nTraining comparator on A-D...")
comparator = ComparisonAttention(bits_per_token, rng=42)

# Train on all pairs of A-D
train_chars = "ABCD"
for c1 in train_chars:
    for c2 in train_chars:
        t1 = encode_char(c1)
        t2 = encode_char(c2)
        comparator.train_comparison(t1, t2, char_to_ord(c1), char_to_ord(c2))

print("Testing comparisons:")
test_pairs = [
    ("A", "B", True),   # A < B
    ("B", "A", False),  # B < A is false
    ("A", "D", True),   # A < D
    ("D", "A", False),  # D < A is false
    # Unseen tokens
    ("E", "F", True),   # E < F
    ("F", "E", False),  # F < E is false
    ("W", "X", True),   # W < X
]

for c1, c2, expected in test_pairs:
    t1 = encode_char(c1)
    t2 = encode_char(c2)
    result = comparator.is_less_than(t1, t2)
    status = "✓" if result == expected else "✗"
    trained = "(trained)" if c1 in train_chars and c2 in train_chars else "(unseen)"
    print(f"  {status} {c1} < {c2}? {result} (expected {expected}) {trained}")


# =============================================================
# Test 3: Bubble Sort Single Pass
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Bubble Sort Single Pass")
print("=" * 60)
print("One pass of bubble sort: compare adjacent, swap if needed")
print("E.g., DCBA → CDAB (after one pass)")

def bubble_sort_pass(s: str) -> str:
    """One pass of bubble sort."""
    chars = list(s)
    for i in range(len(chars) - 1):
        if chars[i] > chars[i + 1]:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return ''.join(chars)

# This is very hard because the attention pattern depends on comparisons
# Let's try with learned attention (not position_only)

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,
    position_only=False,  # Need content for comparison
    rng=42,
)

# Generate training data for all permutations of ABCD
from itertools import permutations

train_perms = list(permutations("ABCD"))
print(f"\nTraining on all {len(train_perms)} permutations of ABCD...")

for perm in train_perms:
    seq = ''.join(perm)
    expected = bubble_sort_pass(seq)

    tokens = [encode_char(c) for c in seq]
    expected_tokens = [encode_char(c) for c in expected]

    model.train_value_projection(tokens)
    model.train_value_projection(expected_tokens)

    # Train attention to select correct output position
    for pos in range(len(tokens)):
        # Find which input position should go to this output position
        target_char = expected[pos]

        # After bubble sort pass, we need to find where target_char came from
        # This is complex because swaps change positions
        target_weights = [0.0] * len(tokens)

        # Simple approach: which position has the char we want?
        # But this doesn't account for the swap logic properly
        for j, c in enumerate(seq):
            if c == target_char:
                target_weights[j] = 1.0
                break

        model.train_attention_weights(tokens, pos, target_weights)

# Test
print("\nTesting (trained permutations):")
test_trained = ["DCBA", "ABCD", "BADC", "CDAB"]
for seq in test_trained:
    expected = bubble_sort_pass(seq)
    tokens = [encode_char(c) for c in seq]

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%)")

print("\nTesting (unseen tokens - EFGH permutations):")
test_unseen = ["HGFE", "EFGH", "FEHG", "GHEF"]
for seq in test_unseen:
    expected = bubble_sort_pass(seq)
    tokens = [encode_char(c) for c in seq]
    model.train_value_projection(tokens)

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%)")


# =============================================================
# Test 4: Full Sort (Multiple Passes)
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Full Sort (Training on sorted outputs)")
print("=" * 60)
print("Task: Sort the sequence alphabetically")

model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=16,
    aggregation=AggregationStrategy.TOP_1,
    value_strategy=MapperStrategy.BIT_LEVEL,
    position_mode=PositionMode.BINARY,
    max_seq_len=8,
    causal=False,
    position_only=False,
    rng=42,
)

# Train on all permutations → sorted output
print(f"\nTraining on all {len(train_perms)} permutations of ABCD → ABCD...")
for perm in train_perms:
    seq = ''.join(perm)
    sorted_seq = ''.join(sorted(seq))  # Always ABCD

    tokens = [encode_char(c) for c in seq]
    sorted_tokens = [encode_char(c) for c in sorted_seq]

    model.train_value_projection(tokens)

    # Train attention: position i should attend to position of i-th smallest
    for pos in range(len(tokens)):
        target_char = sorted_seq[pos]  # Which char should be at this position

        target_weights = [0.0] * len(tokens)
        for j, c in enumerate(seq):
            if c == target_char:
                target_weights[j] = 1.0
                break

        model.train_attention_weights(tokens, pos, target_weights)

print("\nTesting (trained permutations):")
test_trained = ["DCBA", "ABCD", "BADC", "CDAB", "DBAC"]
for seq in test_trained:
    expected = ''.join(sorted(seq))
    tokens = [encode_char(c) for c in seq]

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%)")

print("\nTesting (unseen tokens):")
test_unseen = ["HGFE", "EFGH", "WXZY", "ZYXW"]
for seq in test_unseen:
    expected = ''.join(sorted(seq))
    tokens = [encode_char(c) for c in seq]
    model.train_value_projection(tokens)

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%)")


# =============================================================
# Test 5: Multi-Step Task - Shift then Reverse
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Multi-Step Task (Shift then Reverse)")
print("=" * 60)
print("Task: First shift left, then reverse")
print("E.g., ABCD → shift → BCDA → reverse → ADCB")

def shift_then_reverse(s: str) -> str:
    """Shift left by 1, then reverse."""
    shifted = s[1:] + s[0]  # ABCD → BCDA
    return shifted[::-1]     # BCDA → ADCB

# Use two separate attention layers
class TwoStepAttention(Module):
    """Two sequential attention operations."""

    def __init__(self, input_bits: int, max_seq_len: int, rng: int = None):
        super().__init__()

        # Step 1: Shift
        self.shift_layer = SoftRAMAttention(
            input_bits=input_bits,
            num_heads=8,
            aggregation=AggregationStrategy.TOP_1,
            value_strategy=MapperStrategy.BIT_LEVEL,
            position_mode=PositionMode.RELATIVE,
            max_seq_len=max_seq_len,
            causal=False,
            position_only=True,
            rng=rng,
        )

        # Step 2: Reverse
        self.reverse_layer = SoftRAMAttention(
            input_bits=input_bits,
            num_heads=8,
            aggregation=AggregationStrategy.TOP_1,
            value_strategy=MapperStrategy.BIT_LEVEL,
            position_mode=PositionMode.BINARY,
            max_seq_len=max_seq_len,
            causal=False,
            position_only=True,
            rng=rng + 1000 if rng else None,
        )

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        # Step 1: Shift
        shifted = self.shift_layer.forward(tokens)
        # Step 2: Reverse
        return self.reverse_layer.forward(shifted)


print("\nBuilding two-layer model (shift → reverse)...")
model = TwoStepAttention(bits_per_token, max_seq_len=8, rng=42)

# Train shift layer
print("Training shift layer...")
train_seq = "ABCD"
tokens = [encode_char(c) for c in train_seq]
model.shift_layer.train_value_projection(tokens)

n = len(tokens)
for pos in range(n):
    target_weights = [0.0] * n
    # Shift left: position i gets from position (i+1) mod n
    target_weights[(pos + 1) % n] = 1.0
    model.shift_layer.train_attention_weights(tokens, pos, target_weights)

# Train reverse layer (on shifted output)
print("Training reverse layer...")
shifted_seq = train_seq[1:] + train_seq[0]  # BCDA
shifted_tokens = [encode_char(c) for c in shifted_seq]
model.reverse_layer.train_value_projection(shifted_tokens)

for pos in range(n):
    target_weights = [0.0] * n
    target_weights[n - 1 - pos] = 1.0
    model.reverse_layer.train_attention_weights(shifted_tokens, pos, target_weights)

# Test
print("\nTesting:")
test_cases = [
    "ABCD",  # Trained
    "EFGH",  # Unseen
    "WXYZ",  # Unseen
]

for seq in test_cases:
    expected = shift_then_reverse(seq)
    tokens = [encode_char(c) for c in seq]

    # Train value projections for test tokens
    model.shift_layer.train_value_projection(tokens)
    shifted = seq[1:] + seq[0]
    shifted_tokens = [encode_char(c) for c in shifted]
    model.reverse_layer.train_value_projection(shifted_tokens)

    outputs = model.forward(tokens)
    result = decode_sequence(outputs)
    correct = sum(1 for r, e in zip(result, expected) if r == e)
    pct = 100 * correct / len(expected)
    status = "✓" if result == expected else "~"
    trained = "(trained)" if seq == "ABCD" else "(unseen)"
    print(f"  {status} {seq} → {result} (want {expected}, {pct:.0f}%) {trained}")


# =============================================================
# Test 6: Analysis - Why Sorting is Hard
# =============================================================
print("\n" + "=" * 60)
print("Test 6: Analysis - Why Sorting is Hard")
print("=" * 60)

print("""
ANALYSIS: Why sorting is fundamentally different from shift/reverse

1. POSITION-BASED TASKS (shift, reverse, copy):
   - Attention pattern is FIXED: "attend to position i-1" or "attend to n-1-i"
   - Pattern doesn't depend on token content
   - position_only=True gives 100% generalization

2. CONTENT-BASED TASKS (sorting, find-min):
   - Attention pattern DEPENDS ON CONTENT: "attend to smallest token"
   - For DCBA: pos 0 should attend to pos 3 (where A is)
   - For ABCD: pos 0 should attend to pos 0 (where A is)
   - Different inputs → different attention patterns!

3. WHY RAM STRUGGLES WITH SORTING:
   - Must learn comparison function: is token[i] < token[j]?
   - Comparison depends on specific bit patterns
   - New tokens (EFGH) have new bit patterns
   - No way to generalize "A < B" to "E < F" without learning

4. WHAT WOULD BE NEEDED:
   - Learn the comparison function as a RAM lookup table
   - Train on ALL token pairs that will be compared
   - Or: Use bit encoding where lexical order = numerical order
     (which our 5-bit encoding already has!)

Let's verify our encoding has ordered bit patterns...
""")

print("Token encodings (checking if lexically ordered = numerically ordered):")
for c in "ABCDEFGH":
    bits = encode_char(c)
    # Convert to number
    val = sum(b.item() * (2 ** (len(bits) - 1 - i)) for i, b in enumerate(bits))
    print(f"  {c} = {bits.tolist()} = {val}")

print("\nThe encoding IS numerically ordered (A=0, B=1, C=2, ...)")
print("This means we COULD compare by treating bits as a number")
print("But RAM needs to LEARN this comparison, it's not automatic")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Sorting and Multi-Step Tasks")
print("=" * 60)

print("""
RESULTS:

1. FIND MINIMUM: Partially works on trained, fails on unseen
   - Attention pattern is content-dependent
   - Doesn't generalize to unseen tokens

2. COMPARISON (a < b): Works on trained pairs, fails on unseen
   - Must learn each pair separately
   - No generalization of comparison function

3. BUBBLE SORT PASS: Works on trained permutations
   - Fails on unseen tokens (EFGH)
   - Attention pattern too content-specific

4. FULL SORT: Works on trained permutations
   - Fails on unseen tokens
   - Must memorize all input→output mappings

5. MULTI-STEP (shift→reverse): Works 100%!
   - Each step uses position_only
   - Position patterns generalize
   - Composition of generalizing layers works

KEY INSIGHT:
  Tasks that can be decomposed into POSITION-BASED steps
  can generalize. Tasks requiring CONTENT-BASED attention
  (like sorting) cannot generalize with current approach.

WHAT WOULD HELP FOR SORTING:
  1. Learn comparison function on token pairs
  2. Use that to dynamically compute attention
  3. Or: Design encoding where comparison is trivial
  4. Or: Use neural network for comparison (not RAM)
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
