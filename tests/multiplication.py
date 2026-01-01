"""
Multiplication Learning Test

Tests whether RAM networks can learn multiplication by decomposition.

Binary multiplication (shift-and-add):
- For each bit of b: if bit=1, add (a << position) to result
- Reuses LearnedFullAdder from addition
- No new primitives needed!

Decimal multiplication (grade-school):
- Learn single-digit multiplication table (100 patterns)
- Multiply each digit pair, shift by position, accumulate
- Reuses multi-digit addition
"""

import random
from datetime import datetime

from torch import zeros, uint8, tensor
from torch.nn import Module

from wnn.ram.core import RAMLayer


# Import from arithmetic test
import sys
sys.path.insert(0, '/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn/tests')
from arithmetic import LearnedFullAdder, MultiDigitAdder


class BinaryMultiplier(Module):
    """
    Binary multiplication using shift-and-add algorithm.

    For each bit position i of multiplier b:
        if b[i] == 1: result += a << i

    This reuses LearnedFullAdder - no new primitives needed!
    Only 8 patterns for the full adder, generalizes to any bit width.
    """

    def __init__(self, rng: int | None = None):
        super().__init__()
        self.adder = MultiDigitAdder(base=2, rng=rng)

    def train(self) -> int:
        """Train the underlying full adder (8 patterns)."""
        return self.adder.train()

    def multiply(self, a: int, b: int, max_bits: int = 16) -> int:
        """
        Multiply two integers using shift-and-add.

        Args:
            a: Multiplicand
            b: Multiplier
            max_bits: Maximum bits for intermediate results

        Returns:
            Product a * b
        """
        result = 0

        # For each bit of b
        bit_pos = 0
        temp_b = b
        while temp_b > 0:
            if temp_b & 1:  # If bit is 1
                # Add a << bit_pos to result
                shifted_a = a << bit_pos
                result = self.adder.add_int(result, shifted_a)

            temp_b >>= 1
            bit_pos += 1

        return result

    def __repr__(self):
        return f"BinaryMultiplier(adder={self.adder})"


class LearnedDigitMultiplier(Module):
    """
    Learn single-digit multiplication: a * b for a,b in 0-9.

    This is a lookup table with 100 entries.
    Output can be 0-81 (9*9), needs 7 bits.
    """

    def __init__(self, rng: int | None = None):
        super().__init__()

        # Input: 4 bits (a) + 4 bits (b) = 8 bits
        # Output: 7 bits (0-81)
        self.multiplier = RAMLayer(
            total_input_bits=8,
            num_neurons=7,
            n_bits_per_neuron=8,
            rng=rng,
        )
        self._trained = False

    def train_all(self) -> int:
        """Train on all 100 digit pairs."""
        errors = 0

        for a in range(10):
            for b in range(10):
                product = a * b

                # Encode input: 4 bits each
                a_bits = [(a >> i) & 1 for i in range(3, -1, -1)]
                b_bits = [(b >> i) & 1 for i in range(3, -1, -1)]
                inp = tensor(a_bits + b_bits, dtype=uint8)

                # Encode output: 7 bits
                out_bits = [(product >> i) & 1 for i in range(6, -1, -1)]
                out = tensor(out_bits, dtype=uint8)

                errors += self.multiplier.commit(inp.unsqueeze(0), out.unsqueeze(0))

        self._trained = True
        return errors

    def forward(self, a: int, b: int) -> int:
        """Multiply two single digits."""
        a_bits = [(a >> i) & 1 for i in range(3, -1, -1)]
        b_bits = [(b >> i) & 1 for i in range(3, -1, -1)]
        inp = tensor(a_bits + b_bits, dtype=uint8)

        out = self.multiplier(inp.unsqueeze(0)).squeeze()
        result = sum(int(out[i].item()) << (6 - i) for i in range(7))
        return result

    def test_accuracy(self) -> float:
        """Test accuracy on all digit pairs."""
        correct = 0
        for a in range(10):
            for b in range(10):
                if self.forward(a, b) == a * b:
                    correct += 1
        return correct / 100

    def __repr__(self):
        return f"LearnedDigitMultiplier(trained={self._trained})"


class DecimalMultiplier(Module):
    """
    Decimal multiplication using grade-school algorithm.

    For each digit of multiplier b:
        partial = a * b[i]  (using digit multiplier)
        result += partial << (i decimal positions)

    Needs:
    - LearnedDigitMultiplier: 100 patterns
    - MultiDigitAdder: 200 patterns (for accumulation)
    """

    def __init__(self, rng: int | None = None):
        super().__init__()
        self.digit_mult = LearnedDigitMultiplier(rng)
        self.adder = MultiDigitAdder(base=10, rng=rng + 1000 if rng else None)

    def train(self) -> tuple[int, int]:
        """Train both components."""
        mult_errors = self.digit_mult.train_all()
        add_errors = self.adder.train()
        return mult_errors, add_errors

    def _int_to_digits(self, n: int) -> list[int]:
        """Convert to digit list (LSD first)."""
        if n == 0:
            return [0]
        digits = []
        while n > 0:
            digits.append(n % 10)
            n //= 10
        return digits

    def _digits_to_int(self, digits: list[int]) -> int:
        """Convert from digit list (LSD first)."""
        result = 0
        for i, d in enumerate(digits):
            result += d * (10 ** i)
        return result

    def multiply(self, a: int, b: int) -> int:
        """
        Multiply using grade-school algorithm.

        For each digit of b, multiply by a, shift, and accumulate.
        """
        a_digits = self._int_to_digits(a)
        b_digits = self._int_to_digits(b)

        result_digits = [0]

        for b_pos, b_digit in enumerate(b_digits):
            if b_digit == 0:
                continue

            # Multiply a by single digit b_digit
            partial = []
            carry = 0

            for a_digit in a_digits:
                # Single-digit multiply
                prod = self.digit_mult(a_digit, b_digit) + carry
                partial.append(prod % 10)
                carry = prod // 10

            if carry:
                partial.append(carry)

            # Shift by b_pos (prepend zeros)
            shifted = [0] * b_pos + partial

            # Add to result
            result_digits = self.adder.add(result_digits, shifted)

        return self._digits_to_int(result_digits)

    def __repr__(self):
        return f"DecimalMultiplier(digit_mult={self.digit_mult}, adder={self.adder})"


def test_binary_multiplier():
    """Test binary multiplication."""
    print(f"\n{'='*60}")
    print("Testing Binary Multiplication (Shift-and-Add)")
    print(f"{'='*60}")

    mult = BinaryMultiplier(rng=42)

    # Train full adder (8 patterns)
    errors = mult.train()
    print(f"Full adder trained with {errors} corrections")
    print("No new primitives needed - reuses addition!")

    # Test on random multiplications
    random.seed(123)
    n_test = 30
    correct = 0

    print(f"\nTesting on {n_test} random multiplications...")
    for _ in range(n_test):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        expected = a * b
        result = mult.multiply(a, b)

        if result == expected:
            correct += 1

    accuracy = correct / n_test
    print(f"Accuracy: {accuracy:.1%} ({correct}/{n_test})")

    # Show examples
    print("\nExamples:")
    examples = [(5, 3), (12, 11), (255, 2), (0, 100), (15, 15)]
    for a, b in examples:
        expected = a * b
        result = mult.multiply(a, b)
        ok = "✓" if result == expected else "✗"
        print(f"  {a} × {b} = {result} (expected {expected}) {ok}")

    return accuracy


def test_digit_multiplier():
    """Test single-digit multiplication table."""
    print(f"\n{'='*60}")
    print("Testing Digit Multiplier (100 patterns)")
    print(f"{'='*60}")

    mult = LearnedDigitMultiplier(rng=42)

    # Test before training
    acc_before = mult.test_accuracy()
    print(f"Accuracy before training: {acc_before:.1%}")

    # Train
    errors = mult.train_all()
    print(f"Training corrections: {errors}")

    # Test after
    acc_after = mult.test_accuracy()
    print(f"Accuracy after training: {acc_after:.1%}")

    # Show multiplication table
    print("\nMultiplication table (sample):")
    print("    0  1  2  3  4  5  6  7  8  9")
    for a in range(10):
        row = f"{a}: "
        for b in range(10):
            result = mult(a, b)
            expected = a * b
            if result == expected:
                row += f"{result:2d} "
            else:
                row += f"X{result} "
        print(row)

    return acc_after


def test_decimal_multiplier():
    """Test decimal multiplication."""
    print(f"\n{'='*60}")
    print("Testing Decimal Multiplication (Grade-school)")
    print(f"{'='*60}")

    mult = DecimalMultiplier(rng=42)

    # Train
    print("Training components...")
    mult_err, add_err = mult.train()
    print(f"  Digit multiplier: {mult_err} corrections (100 patterns)")
    print(f"  Decimal adder: {add_err} corrections (200 patterns)")
    print(f"  Total: 300 patterns")

    # Test
    random.seed(456)
    n_test = 30
    correct = 0

    print(f"\nTesting on {n_test} random multiplications...")
    for _ in range(n_test):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        expected = a * b
        result = mult.multiply(a, b)

        if result == expected:
            correct += 1

    accuracy = correct / n_test
    print(f"Accuracy: {accuracy:.1%} ({correct}/{n_test})")

    # Examples
    print("\nExamples:")
    examples = [(12, 34), (99, 99), (7, 8), (0, 50), (25, 4)]
    for a, b in examples:
        expected = a * b
        result = mult.multiply(a, b)
        ok = "✓" if result == expected else "✗"
        print(f"  {a} × {b} = {result} (expected {expected}) {ok}")

    return accuracy


def test_generalization():
    """Test that multiplication generalizes to larger numbers."""
    print(f"\n{'='*60}")
    print("Testing Generalization")
    print(f"{'='*60}")

    # Binary: train on 8 patterns, test on 16-bit multiplication
    print("\nBinary multiplication:")
    print("  Training: 8 full-adder patterns")

    mult = BinaryMultiplier(rng=42)
    mult.train()

    sizes = [(8, 50), (12, 30), (16, 20)]
    for bits, n_test in sizes:
        max_val = (1 << bits) - 1
        random.seed(789 + bits)

        correct = 0
        for _ in range(n_test):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            if mult.multiply(a, b) == a * b:
                correct += 1

        print(f"  {bits}-bit × {bits}-bit: {correct}/{n_test} = {correct/n_test:.1%}")

    # Decimal: train on 300 patterns, test on larger numbers
    print("\nDecimal multiplication:")
    print("  Training: 100 digit-mult + 200 adder = 300 patterns")

    mult = DecimalMultiplier(rng=42)
    mult.train()

    sizes = [(2, 30), (3, 20), (4, 15)]
    for digits, n_test in sizes:
        max_val = 10 ** digits - 1
        random.seed(999 + digits)

        correct = 0
        for _ in range(n_test):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            if mult.multiply(a, b) == a * b:
                correct += 1

        print(f"  {digits}-digit × {digits}-digit: {correct}/{n_test} = {correct/n_test:.1%}")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Multiplication Learning Test")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    # Test 1: Binary multiplication (reuses addition)
    binary_acc = test_binary_multiplier()

    # Test 2: Digit multiplication table
    digit_acc = test_digit_multiplier()

    # Test 3: Decimal multiplication
    decimal_acc = test_decimal_multiplier()

    # Test 4: Generalization
    test_generalization()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\nPrimitives needed:")
    print("  Binary: 8 patterns (full adder only)")
    print("  Decimal: 300 patterns (100 digit-mult + 200 adder)")

    print("\nAccuracy:")
    print(f"  Binary (8-bit):  {binary_acc:.1%}")
    print(f"  Digit table:     {digit_acc:.1%}")
    print(f"  Decimal (2-dig): {decimal_acc:.1%}")

    if binary_acc == 1.0:
        print("\n✓ Binary multiplication works with just 8 patterns!")
        print("  Completely reuses addition - no new primitives!")

    print(f"\n{'='*60}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
