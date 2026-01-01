"""
Learned Sorting Test

Tests whether RAM networks can learn to sort by learning pairwise comparison.

Key questions:
1. Can a RAM network learn the comparison function (a < b)?
2. If comparison is learned, does sorting generalize?
3. How does learned sorting compare to computed sorting?
"""

import random
from datetime import datetime

from wnn.ram.core.models import (
    LearnedComparator,
    LearnedSortingAttention,
    ComputedSortingAttention,
    BitLevelComparator,
)
from wnn.ram.core.models.computed_arithmetic import int_to_bits, bits_to_int


def test_learned_comparator(n_bits: int = 4, verbose: bool = True):
    """Test the learned comparator on pairwise comparison."""
    print(f"\n{'='*60}")
    print(f"Testing LearnedComparator with {n_bits} bits")
    print(f"{'='*60}")

    max_value = (1 << n_bits) - 1
    total_pairs = (max_value + 1) ** 2

    print(f"Value range: 0 to {max_value}")
    print(f"Total comparison pairs: {total_pairs}")

    # Create comparator
    comparator = LearnedComparator(n_bits=n_bits, rng=42)
    print(f"\nComparator: {comparator}")

    # Test before training
    accuracy_before = comparator.test_accuracy()
    print(f"\nAccuracy before training: {accuracy_before:.1%}")

    # Train on all pairs
    print("\nTraining on all pairs...")
    pairs_trained, errors = comparator.train_all_pairs()
    print(f"Trained on {pairs_trained} pairs, {errors} corrections")

    # Test after training
    accuracy_after = comparator.test_accuracy()
    print(f"Accuracy after training: {accuracy_after:.1%}")

    # Show some examples
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


def test_learned_sorting(n_bits: int = 4, seq_len: int = 4, n_test: int = 20, verbose: bool = True):
    """Test learned sorting on random sequences."""
    print(f"\n{'='*60}")
    print(f"Testing LearnedSortingAttention")
    print(f"{'='*60}")

    max_value = (1 << n_bits) - 1

    print(f"Bits per token: {n_bits}")
    print(f"Sequence length: {seq_len}")
    print(f"Value range: 0 to {max_value}")

    # Create sorting attention
    sorter = LearnedSortingAttention(input_bits=n_bits, rng=42)

    # First, train the comparator
    print("\nPhase 1: Training comparator on all pairs...")
    pairs, errors = sorter.train_comparator()
    print(f"Trained on {pairs} pairs, {errors} corrections")

    # Test comparator accuracy
    comp_accuracy = sorter.comparator.test_accuracy()
    print(f"Comparator accuracy: {comp_accuracy:.1%}")

    # Generate test sequences
    print(f"\nPhase 2: Testing sorting on {n_test} random sequences...")
    random.seed(123)

    test_sequences = []
    for _ in range(n_test):
        # Generate random sequence (allow duplicates)
        seq = [random.randint(0, max_value) for _ in range(seq_len)]
        test_sequences.append(seq)

    # Test sorting
    accuracy, correct, total = sorter.test_sorting_accuracy(test_sequences)
    print(f"Sorting accuracy: {accuracy:.1%} ({correct}/{total} sequences correct)")

    # Show some examples
    if verbose:
        print("\nExample sorts:")
        for seq in test_sequences[:5]:
            tokens = [int_to_bits(v, n_bits) for v in seq]
            sorted_tokens = sorter.forward(tokens)
            sorted_values = [bits_to_int(t) for t in sorted_tokens]
            expected = sorted(seq)
            status = "✓" if sorted_values == expected else "✗"
            print(f"  Input:    {seq}")
            print(f"  Output:   {sorted_values}")
            print(f"  Expected: {expected} {status}")
            print()

    return accuracy


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

    # Create comparator
    comparator = BitLevelComparator(n_bits=n_bits, rng=42)
    print(f"\nComparator: {comparator}")

    # Test before training
    accuracy_before = comparator.test_accuracy()
    print(f"\nAccuracy before training: {accuracy_before:.1%}")

    # Train (only needs to learn 4+4 + sum(2^i for i in 1..n-1) patterns)
    # For n=4: 4 + 4 + 2 + 4 + 8 = 22 patterns (vs 256 for full pairs)
    print("\nTraining bit-level operations...")
    errors = comparator.train_all()
    print(f"Training corrections: {errors}")

    # Test after training
    accuracy_after = comparator.test_accuracy()
    print(f"Accuracy after training: {accuracy_after:.1%}")

    # Show some examples
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


def test_bit_level_sorting(n_bits: int = 6, seq_len: int = 5, n_test: int = 30, verbose: bool = True):
    """Test full sorting with BitLevelComparator (100% generalization)."""
    print(f"\n{'='*60}")
    print(f"Testing BitLevel Sorting ({n_bits} bits, seq_len={seq_len})")
    print(f"{'='*60}")

    max_value = (1 << n_bits) - 1

    print(f"Value range: 0 to {max_value}")
    print(f"Using BitLevelComparator for 100% generalization")

    # Create sorting attention with bit_level mode (default)
    sorter = LearnedSortingAttention(
        input_bits=n_bits,
        comparator_mode="bit_level",
        rng=42,
    )

    # Train comparator (only bit-level patterns)
    print("\nTraining bit-level comparator...")
    patterns, errors = sorter.train_comparator()
    total_pairs = (max_value + 1) ** 2
    print(f"Trained on {patterns} patterns (vs {total_pairs} for memorization)")
    print(f"Compression ratio: {total_pairs / patterns:.1f}x")

    # Test sorting
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


def test_bit_level_generalization(n_bits: int = 6, verbose: bool = True):
    """Test BitLevelComparator generalization to larger bit widths."""
    print(f"\n{'='*60}")
    print(f"Testing BitLevelComparator Generalization ({n_bits} bits)")
    print(f"{'='*60}")

    max_value = (1 << n_bits) - 1
    total_pairs = (max_value + 1) ** 2

    print(f"Value range: 0 to {max_value}")
    print(f"Total possible pairs: {total_pairs}")

    # Count training patterns
    basic_patterns = 8  # 4 for less_at + 4 for equal_at
    prefix_patterns = sum(1 << i for i in range(1, n_bits))  # 2 + 4 + 8 + ... + 2^(n-1)
    total_training = basic_patterns + prefix_patterns
    print(f"Training patterns needed: {total_training} (vs {total_pairs} for memorization)")
    print(f"Compression ratio: {total_pairs / total_training:.1f}x")

    # Create and train
    comparator = BitLevelComparator(n_bits=n_bits, rng=42)
    errors = comparator.train_all()

    # Test on ALL pairs (including unseen during training!)
    # The bit-level approach should generalize because:
    # - Per-bit ops are shared across all pairs
    # - Prefix AND logic is learned separately
    accuracy = comparator.test_accuracy()
    print(f"\nAccuracy on ALL {total_pairs} pairs: {accuracy:.1%}")

    if verbose and accuracy == 1.0:
        print("\n✓ BitLevelComparator achieves 100% generalization!")
        print("  Unlike memorization-based approach, this learns the LOGIC")

    return accuracy


def test_generalization(n_bits: int = 4, train_frac: float = 0.8, verbose: bool = True):
    """Test if comparison generalizes to unseen pairs."""
    print(f"\n{'='*60}")
    print(f"Testing Generalization (train on {train_frac:.0%} of pairs)")
    print(f"{'='*60}")

    max_value = (1 << n_bits) - 1
    all_pairs = [(a, b) for a in range(max_value + 1) for b in range(max_value + 1)]

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_pairs)

    n_train = int(len(all_pairs) * train_frac)
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:]

    print(f"Total pairs: {len(all_pairs)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    # Create and train comparator
    comparator = LearnedComparator(n_bits=n_bits, rng=42)

    print("\nTraining...")
    errors = 0
    for a_val, b_val in train_pairs:
        a = int_to_bits(a_val, n_bits)
        b = int_to_bits(b_val, n_bits)
        expected = 1 if a_val < b_val else 0
        errors += comparator.train_pair(a, b, expected)
    print(f"Training corrections: {errors}")

    # Test on training set
    train_correct = 0
    for a_val, b_val in train_pairs:
        a = int_to_bits(a_val, n_bits)
        b = int_to_bits(b_val, n_bits)
        expected = 1 if a_val < b_val else 0
        if comparator(a, b) == expected:
            train_correct += 1
    train_acc = train_correct / len(train_pairs)
    print(f"Training accuracy: {train_acc:.1%}")

    # Test on held-out set
    test_correct = 0
    for a_val, b_val in test_pairs:
        a = int_to_bits(a_val, n_bits)
        b = int_to_bits(b_val, n_bits)
        expected = 1 if a_val < b_val else 0
        if comparator(a, b) == expected:
            test_correct += 1
    test_acc = test_correct / len(test_pairs)
    print(f"Test accuracy (generalization): {test_acc:.1%}")

    # Show some test examples
    if verbose and test_pairs:
        print("\nExample test pairs (unseen during training):")
        for a_val, b_val in test_pairs[:5]:
            a = int_to_bits(a_val, n_bits)
            b = int_to_bits(b_val, n_bits)
            expected = 1 if a_val < b_val else 0
            predicted = comparator(a, b)
            status = "✓" if predicted == expected else "✗"
            print(f"  {a_val} < {b_val}? Predicted: {predicted}, Expected: {expected} {status}")

    return train_acc, test_acc


def compare_with_computed(n_bits: int = 4, seq_len: int = 4, n_test: int = 50):
    """Compare learned vs computed sorting."""
    print(f"\n{'='*60}")
    print(f"Comparing Learned vs Computed Sorting")
    print(f"{'='*60}")

    max_value = (1 << n_bits) - 1

    # Create both sorters
    learned = LearnedSortingAttention(input_bits=n_bits, rng=42)
    computed = ComputedSortingAttention(input_bits=n_bits)

    # Train learned comparator
    print("\nTraining learned comparator...")
    pairs, _ = learned.train_comparator()
    print(f"Trained on {pairs} pairs")

    # Generate test sequences
    random.seed(456)
    test_sequences = []
    for _ in range(n_test):
        seq = [random.randint(0, max_value) for _ in range(seq_len)]
        test_sequences.append(seq)

    # Test both
    learned_correct = 0
    computed_correct = 0

    for seq in test_sequences:
        tokens = [int_to_bits(v, n_bits) for v in seq]
        expected = sorted(seq)

        # Learned
        learned_out = learned.forward(tokens)
        learned_values = [bits_to_int(t) for t in learned_out]
        if learned_values == expected:
            learned_correct += 1

        # Computed
        computed_out = computed.forward(tokens)
        computed_values = [bits_to_int(t) for t in computed_out]
        if computed_values == expected:
            computed_correct += 1

    print(f"\nResults on {n_test} random sequences:")
    print(f"  Learned sorting:  {learned_correct}/{n_test} = {learned_correct/n_test:.1%}")
    print(f"  Computed sorting: {computed_correct}/{n_test} = {computed_correct/n_test:.1%}")

    return learned_correct / n_test, computed_correct / n_test


def test_larger_values(n_bits: int = 6, seq_len: int = 5, n_test: int = 30):
    """Test with larger bit widths to see scaling behavior."""
    print(f"\n{'='*60}")
    print(f"Testing with Larger Values ({n_bits} bits)")
    print(f"{'='*60}")

    max_value = (1 << n_bits) - 1
    total_pairs = (max_value + 1) ** 2

    print(f"Value range: 0 to {max_value}")
    print(f"Total possible pairs: {total_pairs}")

    # Create sorting attention
    sorter = LearnedSortingAttention(input_bits=n_bits, rng=42)

    # Train comparator
    print("\nTraining comparator...")
    pairs, errors = sorter.train_comparator()
    print(f"Trained on {pairs} pairs, {errors} corrections")

    # Test comparator
    comp_accuracy = sorter.comparator.test_accuracy()
    print(f"Comparator accuracy: {comp_accuracy:.1%}")

    # Test sorting
    random.seed(789)
    test_sequences = [
        [random.randint(0, max_value) for _ in range(seq_len)]
        for _ in range(n_test)
    ]

    accuracy, correct, total = sorter.test_sorting_accuracy(test_sequences)
    print(f"Sorting accuracy: {accuracy:.1%} ({correct}/{total})")

    return accuracy


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Learned Sorting Test")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    # Test 1: Basic memorization-based comparator
    comp_acc = test_learned_comparator(n_bits=4)

    # Test 2: Full sorting with memorization
    sort_acc = test_learned_sorting(n_bits=4, seq_len=4, n_test=20)

    # Test 3: Generalization failure with memorization
    train_acc, test_acc = test_generalization(n_bits=4, train_frac=0.8)

    # Test 4: BitLevelComparator (structured approach)
    bit_level_acc = test_bit_level_comparator(n_bits=4)

    # Test 5: BitLevelComparator generalization
    bit_level_gen_acc = test_bit_level_generalization(n_bits=6)

    # Test 6: Full sorting with BitLevelComparator
    bit_level_sort_acc = test_bit_level_sorting(n_bits=6, seq_len=5, n_test=30)

    # Test 7: Compare with computed
    learned_acc, computed_acc = compare_with_computed(n_bits=4, seq_len=4, n_test=50)

    # Test 7: Scale to larger values (memorization)
    large_acc = test_larger_values(n_bits=6, seq_len=5, n_test=30)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nMemorization-based (LearnedComparator):")
    print(f"  Comparator accuracy (4-bit, full training):  {comp_acc:.1%}")
    print(f"  Sorting accuracy (4-bit, full training):     {sort_acc:.1%}")
    print(f"  Generalization (4-bit, 80% train):           {test_acc:.1%}")
    print(f"  Larger values (6-bit, full training):        {large_acc:.1%}")

    print("\nBit-level structured (BitLevelComparator):")
    print(f"  Comparator accuracy (4-bit):                 {bit_level_acc:.1%}")
    print(f"  Comparator generalization (6-bit):           {bit_level_gen_acc:.1%}")
    print(f"  Full sorting accuracy (6-bit):               {bit_level_sort_acc:.1%}")

    print("\nComparison:")
    print(f"  Learned vs Computed (4-bit sorting):         {learned_acc:.1%} vs {computed_acc:.1%}")

    if bit_level_gen_acc == 1.0:
        print("\n✓ BitLevelComparator achieves 100% generalization!")
        print("  Key insight: Decompose into learnable bit-level operations")

    print(f"\n{'='*60}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
