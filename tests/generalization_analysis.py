"""
Analysis script for parity and shift-left generalization failures.

This investigates WHY the current strategies fail and proposes solutions.
"""

import sys
sys.path.insert(0, 'src')

import torch
from torch import tensor, uint8, zeros

from wnn.ram.core.RAMGeneralization import BitLevelMapper
from wnn.ram.enums import ContextMode
from wnn.ram.enums.generalization import BitMapperMode


def int_to_bits(n: int, n_bits: int) -> tensor:
    """Convert integer to bit tensor (MSB first)."""
    bits = []
    for i in range(n_bits - 1, -1, -1):
        bits.append((n >> i) & 1)
    return tensor(bits, dtype=uint8)


def bits_to_int(bits: tensor) -> int:
    """Convert bit tensor to integer."""
    bits = bits.squeeze()
    n_bits = len(bits)
    return sum(int(bits[i].item()) << (n_bits - 1 - i) for i in range(n_bits))


def compute_parity(n: int) -> int:
    """Compute parity (XOR of all bits)."""
    parity = 0
    while n:
        parity ^= (n & 1)
        n >>= 1
    return parity


def analyze_parity():
    """Analyze why parity fails with different context modes."""
    print("\n" + "=" * 70)
    print("PARITY ANALYSIS")
    print("=" * 70)

    n_bits = 6
    max_val = 2 ** n_bits

    # For parity task: output = input with last bit = XOR of all bits
    def make_parity_pair(n):
        inp = int_to_bits(n, n_bits)
        parity = compute_parity(n)
        out = inp.clone()
        out[-1] = parity  # Last bit is parity
        return inp, out

    # Create train/test split
    train_indices = list(range(0, max_val, 2))  # Even indices
    test_indices = list(range(1, max_val, 2))   # Odd indices

    train_data = [make_parity_pair(i) for i in train_indices]
    test_data = [make_parity_pair(i) for i in test_indices]

    print(f"\nDataset: {len(train_data)} train, {len(test_data)} test")
    print(f"\nParity examples:")
    for i in range(min(4, len(train_data))):
        inp, out = train_data[i]
        print(f"  {inp.tolist()} → {out.tolist()} (parity={out[-1].item()})")

    # Test different context modes
    for context_mode in [ContextMode.CUMULATIVE, ContextMode.FULL, ContextMode.CAUSAL]:
        print(f"\n--- Testing {context_mode.name} context mode ---")

        mapper = BitLevelMapper(
            n_bits=n_bits,
            context_mode=context_mode,
            output_mode=BitMapperMode.OUTPUT,  # Direct output, not flip
            rng=42,
        )

        # Train
        for epoch in range(10):
            errors = 0
            for inp, out in train_data:
                errors += mapper.train_mapping(inp, out)
            if errors == 0:
                print(f"  Converged at epoch {epoch + 1}")
                break

        # Test on training set
        train_correct = 0
        for inp, out in train_data:
            pred = mapper.forward(inp)
            if (pred == out).all():
                train_correct += 1

        # Test on test set
        test_correct = 0
        for inp, out in test_data:
            pred = mapper.forward(inp)
            if (pred == out).all():
                test_correct += 1

        train_acc = 100 * train_correct / len(train_data)
        test_acc = 100 * test_correct / len(test_data)

        print(f"  Train accuracy: {train_acc:.1f}%")
        print(f"  Test accuracy: {test_acc:.1f}%")

        # Analyze what the parity bit mapper learned
        parity_mapper = mapper.bit_mappers[-1]  # Last bit is parity
        context_size = mapper._compute_context_size(n_bits - 1)
        print(f"  Parity bit context size: {context_size} bits")


def analyze_shift_left():
    """Analyze why shift-left fails with different context modes."""
    print("\n" + "=" * 70)
    print("SHIFT-LEFT ANALYSIS")
    print("=" * 70)

    n_bits = 6
    max_val = 2 ** n_bits

    # Shift left: output[i] = input[i+1] (with wraparound)
    def make_shift_pair(n):
        inp = int_to_bits(n, n_bits)
        shifted = ((n << 1) | (n >> (n_bits - 1))) & (max_val - 1)
        out = int_to_bits(shifted, n_bits)
        return inp, out

    # Create train/test split
    train_indices = list(range(0, max_val, 2))
    test_indices = list(range(1, max_val, 2))

    train_data = [make_shift_pair(i) for i in train_indices]
    test_data = [make_shift_pair(i) for i in test_indices]

    print(f"\nDataset: {len(train_data)} train, {len(test_data)} test")
    print(f"\nShift-left examples (output[i] = input[i+1]):")
    for i in range(min(4, len(train_data))):
        inp, out = train_data[i]
        print(f"  {inp.tolist()} → {out.tolist()}")

    # Explain the pattern
    print("\n  Pattern: Each output bit depends on the NEXT input bit:")
    print("    out[0] = inp[1], out[1] = inp[2], ..., out[n-1] = inp[0] (wrap)")

    # Test different context modes
    for context_mode in [ContextMode.CUMULATIVE, ContextMode.FULL, ContextMode.LOCAL, ContextMode.BIDIRECTIONAL]:
        print(f"\n--- Testing {context_mode.name} context mode ---")

        # Use local_window=2 for LOCAL/BIDIRECTIONAL to capture neighboring bits
        local_window = 2 if context_mode in [ContextMode.LOCAL, ContextMode.BIDIRECTIONAL] else 3

        mapper = BitLevelMapper(
            n_bits=n_bits,
            context_mode=context_mode,
            output_mode=BitMapperMode.OUTPUT,
            local_window=local_window,
            rng=42,
        )

        # Train
        for epoch in range(10):
            errors = 0
            for inp, out in train_data:
                errors += mapper.train_mapping(inp, out)
            if errors == 0:
                print(f"  Converged at epoch {epoch + 1}")
                break

        # Test
        train_correct = sum(1 for inp, out in train_data if (mapper.forward(inp) == out).all())
        test_correct = sum(1 for inp, out in test_data if (mapper.forward(inp) == out).all())

        train_acc = 100 * train_correct / len(train_data)
        test_acc = 100 * test_correct / len(test_data)

        print(f"  Train accuracy: {train_acc:.1f}%")
        print(f"  Test accuracy: {test_acc:.1f}%")

        # Show what context each bit sees
        print(f"  Context sizes: ", end="")
        for i in range(n_bits):
            ctx_size = mapper._compute_context_size(i)
            print(f"bit{i}:{ctx_size} ", end="")
        print()


def propose_solutions():
    """Propose and test solutions for both problems."""
    print("\n" + "=" * 70)
    print("PROPOSED SOLUTIONS")
    print("=" * 70)

    print("\n--- PARITY SOLUTION ---")
    print("Problem: Parity bit needs ALL input bits, but CUMULATIVE only provides lower bits.")
    print("Solution: Use FULL context mode for the parity bit.")
    print("  - FULL context: each output bit sees all input bits")
    print("  - The parity bit mapper learns: XOR(all inputs)")
    print("  - With 6 bits and FULL context: 2^6 = 64 patterns to learn")
    print("  - But only 32 training examples → 50% coverage")
    print("  - Need a COMPOSITIONAL approach for parity!")

    print("\n  Better solution: RECURRENT parity computation")
    print("  - State = running XOR")
    print("  - At each step: state = state XOR input_bit")
    print("  - Only 4 patterns to learn: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0")
    print("  - This generalizes perfectly!")

    print("\n--- SHIFT-LEFT SOLUTION ---")
    print("Problem: Output bit i depends on input bit (i+1), not lower bits.")
    print("Current modes:")
    print("  - CUMULATIVE: bit i sees 0..i-1 (wrong direction)")
    print("  - FULL: sees all (too much, won't generalize)")
    print("  - LOCAL: centered window (not shifted)")

    print("\nSolution: Add SHIFTED context mode")
    print("  - SHIFTED: output bit i sees input bit (i+k) mod n")
    print("  - For shift-left: k=1")
    print("  - Each bit only needs 1-bit context → only 2 patterns")
    print("  - Generalizes perfectly!")

    print("\nAlternative: Use computed operations (already exist)")
    print("  - shift_left is a purely positional operation")
    print("  - Can be computed exactly without learning")


if __name__ == "__main__":
    analyze_parity()
    analyze_shift_left()
    propose_solutions()
