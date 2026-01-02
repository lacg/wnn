"""
Generalization Test - Consolidated

Tests for RAM network generalization strategies:
- Parity/shift-left analysis (why some strategies fail)
- Learned sorting via comparators
- BitLevelComparator for 100% generalization

Key insight: Decompose complex operations into small learnable primitives.

| Approach | Patterns | Generalization |
|----------|----------|----------------|
| Memorization (LearnedComparator) | n² pairs | ~50% on unseen |
| BitLevel decomposition | 8 + prefix patterns | 100% |

Summary:
- Parity fails with CUMULATIVE (needs all bits) → use RECURRENT XOR
- Shift-left fails (wrong direction) → use SHIFTED context mode
- Sorting: BitLevelComparator achieves 100% with 70 patterns (6-bit)
"""

import random
from datetime import datetime

import torch
from torch import tensor, uint8

from wnn.ram.core.RAMGeneralization import BitLevelMapper
from wnn.ram.enums import ContextMode
from wnn.ram.enums.generalization import BitMapperMode
from wnn.ram.core.models import (
	LearnedComparator,
	LearnedSortingAttention,
	ComputedSortingAttention,
	BitLevelComparator,
)
from wnn.ram.core.models.computed_arithmetic import int_to_bits, bits_to_int


# =============================================================================
# UTILITIES
# =============================================================================

def compute_parity(n: int) -> int:
	"""Compute parity (XOR of all bits)."""
	parity = 0
	while n:
		parity ^= (n & 1)
		n >>= 1
	return parity


# =============================================================================
# PARITY AND SHIFT-LEFT ANALYSIS
# =============================================================================

def analyze_parity():
	"""Analyze why parity fails with different context modes."""
	print("\n" + "=" * 60)
	print("PARITY ANALYSIS")
	print("=" * 60)

	n_bits = 6
	max_val = 2 ** n_bits

	def make_parity_pair(n):
		bits = []
		for i in range(n_bits - 1, -1, -1):
			bits.append((n >> i) & 1)
		inp = tensor(bits, dtype=uint8)
		parity = compute_parity(n)
		out = inp.clone()
		out[-1] = parity  # Last bit is parity
		return inp, out

	train_indices = list(range(0, max_val, 2))  # Even indices
	test_indices = list(range(1, max_val, 2))   # Odd indices

	train_data = [make_parity_pair(i) for i in train_indices]
	test_data = [make_parity_pair(i) for i in test_indices]

	print(f"\nDataset: {len(train_data)} train, {len(test_data)} test")
	print(f"\nParity examples:")
	for i in range(min(4, len(train_data))):
		inp, out = train_data[i]
		print(f"  {inp.tolist()} → {out.tolist()} (parity={out[-1].item()})")

	for context_mode in [ContextMode.CUMULATIVE, ContextMode.FULL, ContextMode.CAUSAL]:
		print(f"\n--- Testing {context_mode.name} context mode ---")

		mapper = BitLevelMapper(
			n_bits=n_bits,
			context_mode=context_mode,
			output_mode=BitMapperMode.OUTPUT,
			rng=42,
		)

		for epoch in range(10):
			errors = 0
			for inp, out in train_data:
				errors += mapper.train_mapping(inp, out)
			if errors == 0:
				print(f"  Converged at epoch {epoch + 1}")
				break

		train_correct = sum(1 for inp, out in train_data if (mapper.forward(inp) == out).all())
		test_correct = sum(1 for inp, out in test_data if (mapper.forward(inp) == out).all())

		print(f"  Train accuracy: {100 * train_correct / len(train_data):.1f}%")
		print(f"  Test accuracy: {100 * test_correct / len(test_data):.1f}%")


def analyze_shift_left():
	"""Analyze why shift-left fails with different context modes."""
	print("\n" + "=" * 60)
	print("SHIFT-LEFT ANALYSIS")
	print("=" * 60)

	n_bits = 6
	max_val = 2 ** n_bits

	def make_shift_pair(n):
		bits = []
		for i in range(n_bits - 1, -1, -1):
			bits.append((n >> i) & 1)
		inp = tensor(bits, dtype=uint8)
		shifted = ((n << 1) | (n >> (n_bits - 1))) & (max_val - 1)
		out_bits = []
		for i in range(n_bits - 1, -1, -1):
			out_bits.append((shifted >> i) & 1)
		out = tensor(out_bits, dtype=uint8)
		return inp, out

	train_indices = list(range(0, max_val, 2))
	test_indices = list(range(1, max_val, 2))

	train_data = [make_shift_pair(i) for i in train_indices]
	test_data = [make_shift_pair(i) for i in test_indices]

	print(f"\nDataset: {len(train_data)} train, {len(test_data)} test")
	print(f"\nShift-left examples (output[i] = input[i+1]):")
	for i in range(min(4, len(train_data))):
		inp, out = train_data[i]
		print(f"  {inp.tolist()} → {out.tolist()}")

	print("\n  Pattern: Each output bit depends on the NEXT input bit:")
	print("    out[0] = inp[1], out[1] = inp[2], ..., out[n-1] = inp[0] (wrap)")

	# Only test CUMULATIVE and FULL - LOCAL/BIDIRECTIONAL have edge-case issues
	for context_mode in [ContextMode.CUMULATIVE, ContextMode.FULL]:
		print(f"\n--- Testing {context_mode.name} context mode ---")

		mapper = BitLevelMapper(
			n_bits=n_bits,
			context_mode=context_mode,
			output_mode=BitMapperMode.OUTPUT,
			rng=42,
		)

		for epoch in range(10):
			errors = 0
			for inp, out in train_data:
				errors += mapper.train_mapping(inp, out)
			if errors == 0:
				print(f"  Converged at epoch {epoch + 1}")
				break

		train_correct = sum(1 for inp, out in train_data if (mapper.forward(inp) == out).all())
		test_correct = sum(1 for inp, out in test_data if (mapper.forward(inp) == out).all())

		print(f"  Train accuracy: {100 * train_correct / len(train_data):.1f}%")
		print(f"  Test accuracy: {100 * test_correct / len(test_data):.1f}%")


def propose_solutions():
	"""Propose solutions for parity and shift-left problems."""
	print("\n" + "=" * 60)
	print("PROPOSED SOLUTIONS")
	print("=" * 60)

	print("\n--- PARITY SOLUTION ---")
	print("Problem: Parity bit needs ALL input bits, but CUMULATIVE only provides lower bits.")
	print("Solution: Use RECURRENT parity computation")
	print("  - State = running XOR")
	print("  - At each step: state = state XOR input_bit")
	print("  - Only 4 patterns to learn: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0")
	print("  - This generalizes perfectly!")

	print("\n--- SHIFT-LEFT SOLUTION ---")
	print("Problem: Output bit i depends on input bit (i+1), not lower bits.")
	print("Solution: Add SHIFTED context mode")
	print("  - SHIFTED: output bit i sees input bit (i+k) mod n")
	print("  - For shift-left: k=1")
	print("  - Each bit only needs 1-bit context → only 2 patterns")


# =============================================================================
# LEARNED SORTING TESTS
# =============================================================================

def test_learned_comparator(n_bits: int = 4, verbose: bool = True):
	"""Test the learned comparator on pairwise comparison."""
	print(f"\n{'='*60}")
	print(f"Testing LearnedComparator with {n_bits} bits")
	print(f"{'='*60}")

	max_value = (1 << n_bits) - 1
	total_pairs = (max_value + 1) ** 2

	print(f"Value range: 0 to {max_value}")
	print(f"Total comparison pairs: {total_pairs}")

	comparator = LearnedComparator(n_bits=n_bits, rng=42)

	accuracy_before = comparator.test_accuracy()
	print(f"\nAccuracy before training: {accuracy_before:.1%}")

	print("\nTraining on all pairs...")
	pairs_trained, errors = comparator.train_all_pairs()
	print(f"Trained on {pairs_trained} pairs, {errors} corrections")

	accuracy_after = comparator.test_accuracy()
	print(f"Accuracy after training: {accuracy_after:.1%}")

	if verbose:
		print("\nExample comparisons:")
		test_pairs = [(0, 1), (5, 3), (7, 7), (max_value, 0), (max_value//2, max_value//2 + 1)]
		for a_val, b_val in test_pairs:
			a = int_to_bits(a_val, n_bits)
			b = int_to_bits(b_val, n_bits)
			result = comparator(a, b)
			expected = 1 if a_val < b_val else 0
			status = "✓" if result == expected else "✗"
			print(f"  {a_val} < {b_val}? Predicted: {result}, Expected: {expected} {status}")

	return accuracy_after


def test_bit_level_comparator(n_bits: int = 4, verbose: bool = True):
	"""Test the BitLevelComparator which decomposes comparison into bit-level ops."""
	print(f"\n{'='*60}")
	print(f"Testing BitLevelComparator with {n_bits} bits")
	print(f"{'='*60}")

	max_value = (1 << n_bits) - 1

	print(f"Value range: 0 to {max_value}")
	print("This comparator learns:")
	print("  - Per-bit less-than (4 patterns)")
	print("  - Per-bit equality (4 patterns)")
	print("  - Prefix AND logic (2^i patterns per position)")

	comparator = BitLevelComparator(n_bits=n_bits, rng=42)

	accuracy_before = comparator.test_accuracy()
	print(f"\nAccuracy before training: {accuracy_before:.1%}")

	print("\nTraining bit-level operations...")
	errors = comparator.train_all()
	print(f"Training corrections: {errors}")

	accuracy_after = comparator.test_accuracy()
	print(f"Accuracy after training: {accuracy_after:.1%}")

	if verbose:
		print("\nExample comparisons:")
		test_pairs = [(0, 1), (5, 3), (7, 7), (max_value, 0), (max_value//2, max_value//2 + 1)]
		for a_val, b_val in test_pairs:
			a = int_to_bits(a_val, n_bits)
			b = int_to_bits(b_val, n_bits)
			result = comparator(a, b)
			expected = 1 if a_val < b_val else 0
			status = "✓" if result == expected else "✗"
			print(f"  {a_val} < {b_val}? Predicted: {result}, Expected: {expected} {status}")

	return accuracy_after


def test_bit_level_generalization(n_bits: int = 6, verbose: bool = True):
	"""Test BitLevelComparator generalization to larger bit widths."""
	print(f"\n{'='*60}")
	print(f"Testing BitLevelComparator Generalization ({n_bits} bits)")
	print(f"{'='*60}")

	max_value = (1 << n_bits) - 1
	total_pairs = (max_value + 1) ** 2

	print(f"Value range: 0 to {max_value}")
	print(f"Total possible pairs: {total_pairs}")

	basic_patterns = 8  # 4 for less_at + 4 for equal_at
	prefix_patterns = sum(1 << i for i in range(1, n_bits))
	total_training = basic_patterns + prefix_patterns
	print(f"Training patterns needed: {total_training} (vs {total_pairs} for memorization)")
	print(f"Compression ratio: {total_pairs / total_training:.1f}x")

	comparator = BitLevelComparator(n_bits=n_bits, rng=42)
	comparator.train_all()

	accuracy = comparator.test_accuracy()
	print(f"\nAccuracy on ALL {total_pairs} pairs: {accuracy:.1%}")

	if verbose and accuracy == 1.0:
		print("\n✓ BitLevelComparator achieves 100% generalization!")

	return accuracy


def test_bit_level_sorting(n_bits: int = 6, seq_len: int = 5, n_test: int = 30, verbose: bool = True):
	"""Test full sorting with BitLevelComparator (100% generalization)."""
	print(f"\n{'='*60}")
	print(f"Testing BitLevel Sorting ({n_bits} bits, seq_len={seq_len})")
	print(f"{'='*60}")

	max_value = (1 << n_bits) - 1

	print(f"Value range: 0 to {max_value}")
	print(f"Using BitLevelComparator for 100% generalization")

	sorter = LearnedSortingAttention(
		input_bits=n_bits,
		comparator_mode="bit_level",
		rng=42,
	)

	print("\nTraining bit-level comparator...")
	patterns, errors = sorter.train_comparator()
	total_pairs = (max_value + 1) ** 2
	print(f"Trained on {patterns} patterns (vs {total_pairs} for memorization)")
	print(f"Compression ratio: {total_pairs / patterns:.1f}x")

	random.seed(789)
	test_sequences = [
		[random.randint(0, max_value) for _ in range(seq_len)]
		for _ in range(n_test)
	]

	accuracy, correct, total = sorter.test_sorting_accuracy(test_sequences)
	print(f"\nSorting accuracy: {accuracy:.1%} ({correct}/{total})")

	if verbose:
		print("\nExample sorts:")
		for seq in test_sequences[:3]:
			tokens = [int_to_bits(v, n_bits) for v in seq]
			sorted_tokens = sorter.forward(tokens)
			sorted_values = [bits_to_int(t) for t in sorted_tokens]
			expected = sorted(seq)
			status = "✓" if sorted_values == expected else "✗"
			print(f"  {seq} → {sorted_values} {status}")

	return accuracy


def compare_with_computed(n_bits: int = 4, seq_len: int = 4, n_test: int = 50):
	"""Compare learned vs computed sorting."""
	print(f"\n{'='*60}")
	print(f"Comparing Learned vs Computed Sorting")
	print(f"{'='*60}")

	max_value = (1 << n_bits) - 1

	learned = LearnedSortingAttention(input_bits=n_bits, rng=42)
	computed = ComputedSortingAttention(input_bits=n_bits)

	print("\nTraining learned comparator...")
	pairs, _ = learned.train_comparator()
	print(f"Trained on {pairs} pairs")

	random.seed(456)
	test_sequences = [
		[random.randint(0, max_value) for _ in range(seq_len)]
		for _ in range(n_test)
	]

	learned_correct = 0
	computed_correct = 0

	for seq in test_sequences:
		tokens = [int_to_bits(v, n_bits) for v in seq]
		expected = sorted(seq)

		learned_out = learned.forward(tokens)
		if [bits_to_int(t) for t in learned_out] == expected:
			learned_correct += 1

		computed_out = computed.forward(tokens)
		if [bits_to_int(t) for t in computed_out] == expected:
			computed_correct += 1

	print(f"\nResults on {n_test} random sequences:")
	print(f"  Learned sorting:  {learned_correct}/{n_test} = {learned_correct/n_test:.1%}")
	print(f"  Computed sorting: {computed_correct}/{n_test} = {computed_correct/n_test:.1%}")

	return learned_correct / n_test, computed_correct / n_test


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
	print(f"\n{'='*60}")
	print("Generalization Test - All Approaches")
	print(f"Started at: {datetime.now()}")
	print(f"{'='*60}")

	# Part 1: Parity and Shift-Left Analysis
	analyze_parity()
	analyze_shift_left()
	propose_solutions()

	# Part 2: Sorting Tests
	comp_acc = test_learned_comparator(n_bits=4)
	bit_level_acc = test_bit_level_comparator(n_bits=4)
	bit_level_gen_acc = test_bit_level_generalization(n_bits=6)
	bit_level_sort_acc = test_bit_level_sorting(n_bits=6, seq_len=5, n_test=30)
	learned_acc, computed_acc = compare_with_computed(n_bits=4, seq_len=4, n_test=50)

	# Summary
	print(f"\n{'='*60}")
	print("SUMMARY")
	print(f"{'='*60}")

	print("\nParity/Shift-left:")
	print("  - Parity: Needs RECURRENT XOR (4 patterns) for 100%")
	print("  - Shift-left: Needs SHIFTED context mode (2 patterns) for 100%")

	print("\nSorting:")
	print(f"  Memorization (4-bit):           {comp_acc:.1%}")
	print(f"  BitLevel (4-bit):               {bit_level_acc:.1%}")
	print(f"  BitLevel generalization (6-bit):{bit_level_gen_acc:.1%}")
	print(f"  BitLevel sorting (6-bit):       {bit_level_sort_acc:.1%}")
	print(f"  Learned vs Computed:            {learned_acc:.1%} vs {computed_acc:.1%}")

	print("""
Key insight: Decomposition > Memorization for generalization.
- BitLevelComparator: 70 patterns for 6-bit (vs 4096 for memorization)
- RecurrentParityMapper: 4 patterns for any bit width
- SHIFTED context: 2 patterns for any bit width
""")
	print(f"Finished at: {datetime.now()}")
	print(f"{'='*60}")
