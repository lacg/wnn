"""
Arithmetic Learning Test

Tests whether RAM networks can learn multi-digit addition.

Key insight: Addition is a recurrent process (carry propagation)
with a small state space (carry bit). The individual operation
(full adder) only needs 8 patterns for binary or 200 for decimal.

This is analogous to parity (which uses 1-bit XOR state).
"""

import random
from datetime import datetime

from torch import zeros, uint8, tensor
from torch.nn import Module

from wnn.ram.core import RAMLayer


class LearnedFullAdder(Module):
    """
    Learn a full adder: (a, b, carry_in) → (sum, carry_out)

    For binary (base 2):
    - Input: 3 bits (a, b, carry_in)
    - Output: 2 bits (sum, carry_out)
    - Only 8 patterns to learn

    For decimal (base 10):
    - Input: a (0-9), b (0-9), carry (0-1)
    - Output: sum (0-9), carry (0-1)
    - 200 patterns to learn
    """

    def __init__(self, base: int = 2, rng: int | None = None):
        """
        Args:
            base: Number base (2 for binary, 10 for decimal)
            rng: Random seed
        """
        super().__init__()

        self.base = base

        if base == 2:
            # Binary: 3 bits input → 2 bits output
            self.input_bits = 3
            self.output_bits = 2
        else:
            # Decimal: need to encode digits 0-9 (4 bits each) + carry
            # Input: 4 bits (a) + 4 bits (b) + 1 bit (carry) = 9 bits
            # Output: 4 bits (sum) + 1 bit (carry) = 5 bits
            self.input_bits = 9
            self.output_bits = 5

        # RAM layer for the adder function
        # Use all input bits for addressing (9 bits = 512 addresses, but only 200 used)
        self.adder = RAMLayer(
            total_input_bits=self.input_bits,
            num_neurons=self.output_bits,
            n_bits_per_neuron=self.input_bits,  # Use all bits to avoid collisions
            rng=rng,
        )

        self._trained = False

    def _encode_input(self, a: int, b: int, carry: int) -> list[int]:
        """Encode inputs as bits."""
        if self.base == 2:
            return [a, b, carry]
        else:
            # Decimal: encode each digit as 4 bits
            a_bits = [(a >> i) & 1 for i in range(3, -1, -1)]
            b_bits = [(b >> i) & 1 for i in range(3, -1, -1)]
            return a_bits + b_bits + [carry]

    def _decode_output(self, bits: list[int]) -> tuple[int, int]:
        """Decode output bits to (sum, carry)."""
        if self.base == 2:
            return bits[0], bits[1]
        else:
            # Decimal: first 4 bits are sum, last bit is carry
            sum_val = sum(b << (3 - i) for i, b in enumerate(bits[:4]))
            return sum_val, bits[4]

    def train_all(self) -> int:
        """Train the full adder on all input combinations."""
        errors = 0

        for a in range(self.base):
            for b in range(self.base):
                for carry in range(2):
                    # Compute expected result
                    total = a + b + carry
                    sum_digit = total % self.base
                    carry_out = 1 if total >= self.base else 0

                    # Encode input and output
                    inp_bits = self._encode_input(a, b, carry)
                    inp = tensor(inp_bits, dtype=uint8)

                    if self.base == 2:
                        out_bits = [sum_digit, carry_out]
                    else:
                        out_bits = [(sum_digit >> i) & 1 for i in range(3, -1, -1)] + [carry_out]
                    out = tensor(out_bits, dtype=uint8)

                    errors += self.adder.commit(inp.unsqueeze(0), out.unsqueeze(0))

        self._trained = True
        return errors

    def forward(self, a: int, b: int, carry: int) -> tuple[int, int]:
        """
        Add two digits with carry.

        Args:
            a: First digit (0 to base-1)
            b: Second digit (0 to base-1)
            carry: Carry from previous position (0 or 1)

        Returns:
            (sum_digit, carry_out)
        """
        inp_bits = self._encode_input(a, b, carry)
        inp = tensor(inp_bits, dtype=uint8)

        out = self.adder(inp.unsqueeze(0)).squeeze()
        out_list = [int(b.item()) for b in out]

        return self._decode_output(out_list)

    def test_accuracy(self) -> float:
        """Test accuracy on all input combinations."""
        correct = 0
        total = 0

        for a in range(self.base):
            for b in range(self.base):
                for carry in range(2):
                    # Expected
                    expected_total = a + b + carry
                    expected_sum = expected_total % self.base
                    expected_carry = 1 if expected_total >= self.base else 0

                    # Predicted
                    pred_sum, pred_carry = self.forward(a, b, carry)

                    if pred_sum == expected_sum and pred_carry == expected_carry:
                        correct += 1
                    total += 1

        return correct / total if total > 0 else 0.0

    def __repr__(self):
        patterns = self.base * self.base * 2
        return f"LearnedFullAdder(base={self.base}, patterns={patterns}, trained={self._trained})"


class MultiDigitAdder(Module):
    """
    Multi-digit addition using learned full adder.

    Processes digits from LSB to MSB, propagating carry.
    """

    def __init__(self, base: int = 2, rng: int | None = None):
        """
        Args:
            base: Number base (2 for binary, 10 for decimal)
            rng: Random seed
        """
        super().__init__()

        self.base = base
        self.full_adder = LearnedFullAdder(base, rng)

    def train(self) -> int:
        """Train the full adder."""
        return self.full_adder.train_all()

    def add(self, a_digits: list[int], b_digits: list[int]) -> list[int]:
        """
        Add two multi-digit numbers.

        Args:
            a_digits: First number as list of digits (LSB first)
            b_digits: Second number as list of digits (LSB first)

        Returns:
            Sum as list of digits (LSB first)
        """
        # Pad to same length
        max_len = max(len(a_digits), len(b_digits))
        a_padded = a_digits + [0] * (max_len - len(a_digits))
        b_padded = b_digits + [0] * (max_len - len(b_digits))

        result = []
        carry = 0

        # Process from LSB to MSB
        for i in range(max_len):
            sum_digit, carry = self.full_adder(a_padded[i], b_padded[i], carry)
            result.append(sum_digit)

        # Handle final carry
        if carry:
            result.append(carry)

        return result

    def add_int(self, a: int, b: int) -> int:
        """Add two integers using learned addition."""
        # Convert to digit lists (LSB first)
        a_digits = self._int_to_digits(a)
        b_digits = self._int_to_digits(b)

        # Add
        sum_digits = self.add(a_digits, b_digits)

        # Convert back
        return self._digits_to_int(sum_digits)

    def _int_to_digits(self, n: int) -> list[int]:
        """Convert integer to digit list (LSB first)."""
        if n == 0:
            return [0]
        digits = []
        while n > 0:
            digits.append(n % self.base)
            n //= self.base
        return digits

    def _digits_to_int(self, digits: list[int]) -> int:
        """Convert digit list (LSB first) to integer."""
        result = 0
        for i, d in enumerate(digits):
            result += d * (self.base ** i)
        return result

    def __repr__(self):
        return f"MultiDigitAdder(base={self.base}, full_adder={self.full_adder})"


def test_binary_full_adder():
    """Test binary full adder."""
    print(f"\n{'='*60}")
    print("Testing Binary Full Adder (8 patterns)")
    print(f"{'='*60}")

    adder = LearnedFullAdder(base=2, rng=42)
    print(f"Adder: {adder}")

    # Test before training
    acc_before = adder.test_accuracy()
    print(f"\nAccuracy before training: {acc_before:.1%}")

    # Train
    errors = adder.train_all()
    print(f"Training corrections: {errors}")

    # Test after training
    acc_after = adder.test_accuracy()
    print(f"Accuracy after training: {acc_after:.1%}")

    # Show all patterns
    print("\nFull adder truth table:")
    print("  a  b  c_in | sum c_out | correct")
    print("  " + "-" * 35)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                expected_sum = (a + b + c) % 2
                expected_carry = 1 if (a + b + c) >= 2 else 0
                pred_sum, pred_carry = adder.forward(a, b, c)
                ok = "✓" if (pred_sum == expected_sum and pred_carry == expected_carry) else "✗"
                print(f"  {a}  {b}    {c}  |  {pred_sum}     {pred_carry}    | {ok}")

    return acc_after


def test_decimal_full_adder():
    """Test decimal full adder."""
    print(f"\n{'='*60}")
    print("Testing Decimal Full Adder (200 patterns)")
    print(f"{'='*60}")

    adder = LearnedFullAdder(base=10, rng=42)
    print(f"Adder: {adder}")

    # Test before training
    acc_before = adder.test_accuracy()
    print(f"\nAccuracy before training: {acc_before:.1%}")

    # Train
    errors = adder.train_all()
    print(f"Training corrections: {errors}")

    # Test after training
    acc_after = adder.test_accuracy()
    print(f"Accuracy after training: {acc_after:.1%}")

    # Show some examples
    print("\nExample additions:")
    examples = [(5, 3, 0), (9, 9, 0), (9, 9, 1), (0, 0, 1), (7, 8, 0)]
    for a, b, c in examples:
        expected_sum = (a + b + c) % 10
        expected_carry = 1 if (a + b + c) >= 10 else 0
        pred_sum, pred_carry = adder.forward(a, b, c)
        ok = "✓" if (pred_sum == expected_sum and pred_carry == expected_carry) else "✗"
        print(f"  {a} + {b} + {c} = {pred_sum} carry {pred_carry} (expected {expected_sum} carry {expected_carry}) {ok}")

    return acc_after


def test_multi_digit_binary():
    """Test multi-digit binary addition."""
    print(f"\n{'='*60}")
    print("Testing Multi-digit Binary Addition")
    print(f"{'='*60}")

    adder = MultiDigitAdder(base=2, rng=42)
    errors = adder.train()
    print(f"Full adder trained with {errors} corrections")

    # Test on random additions
    random.seed(123)
    n_test = 20
    correct = 0

    print(f"\nTesting on {n_test} random additions...")
    for _ in range(n_test):
        a = random.randint(0, 255)  # 8-bit numbers
        b = random.randint(0, 255)
        expected = a + b
        result = adder.add_int(a, b)

        if result == expected:
            correct += 1

    accuracy = correct / n_test
    print(f"Accuracy: {accuracy:.1%} ({correct}/{n_test})")

    # Show some examples
    print("\nExamples:")
    examples = [(5, 3), (127, 128), (255, 1), (0, 0), (100, 155)]
    for a, b in examples:
        expected = a + b
        result = adder.add_int(a, b)
        ok = "✓" if result == expected else "✗"
        print(f"  {a} + {b} = {result} (expected {expected}) {ok}")

    return accuracy


def test_multi_digit_decimal():
    """Test multi-digit decimal addition."""
    print(f"\n{'='*60}")
    print("Testing Multi-digit Decimal Addition")
    print(f"{'='*60}")

    adder = MultiDigitAdder(base=10, rng=42)
    errors = adder.train()
    print(f"Full adder trained with {errors} corrections")

    # Test on random additions
    random.seed(456)
    n_test = 30
    correct = 0

    print(f"\nTesting on {n_test} random additions...")
    for _ in range(n_test):
        a = random.randint(0, 9999)  # 4-digit numbers
        b = random.randint(0, 9999)
        expected = a + b
        result = adder.add_int(a, b)

        if result == expected:
            correct += 1

    accuracy = correct / n_test
    print(f"Accuracy: {accuracy:.1%} ({correct}/{n_test})")

    # Show some examples
    print("\nExamples:")
    examples = [(123, 456), (999, 1), (9999, 9999), (0, 0), (1234, 8765)]
    for a, b in examples:
        expected = a + b
        result = adder.add_int(a, b)
        ok = "✓" if result == expected else "✗"
        print(f"  {a} + {b} = {result} (expected {expected}) {ok}")

    return accuracy


def test_generalization():
    """Test that trained full adder generalizes to unseen multi-digit additions."""
    print(f"\n{'='*60}")
    print("Testing Generalization (train on 8, test on thousands)")
    print(f"{'='*60}")

    # Binary adder only needs 8 patterns
    adder = MultiDigitAdder(base=2, rng=42)
    adder.train()

    print("Binary full adder trained on 8 patterns")
    print("Testing generalization to multi-digit additions...\n")

    # Test on various sizes
    sizes = [(8, 100), (16, 100), (32, 50), (64, 30)]

    for bits, n_test in sizes:
        max_val = (1 << bits) - 1
        random.seed(789 + bits)

        correct = 0
        for _ in range(n_test):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            expected = a + b
            result = adder.add_int(a, b)
            if result == expected:
                correct += 1

        accuracy = correct / n_test
        print(f"  {bits}-bit numbers: {accuracy:.1%} ({correct}/{n_test})")

    return


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Arithmetic Learning Test")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    # Test 1: Binary full adder
    binary_fa_acc = test_binary_full_adder()

    # Test 2: Decimal full adder
    decimal_fa_acc = test_decimal_full_adder()

    # Test 3: Multi-digit binary
    binary_multi_acc = test_multi_digit_binary()

    # Test 4: Multi-digit decimal
    decimal_multi_acc = test_multi_digit_decimal()

    # Test 5: Generalization
    test_generalization()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nFull Adder Learning:")
    print(f"  Binary (8 patterns):    {binary_fa_acc:.1%}")
    print(f"  Decimal (200 patterns): {decimal_fa_acc:.1%}")
    print(f"\nMulti-digit Addition:")
    print(f"  Binary (8-bit):         {binary_multi_acc:.1%}")
    print(f"  Decimal (4-digit):      {decimal_multi_acc:.1%}")

    if binary_fa_acc == 1.0 and decimal_fa_acc == 1.0:
        print("\n✓ Full adder learned perfectly!")
        print("  Key insight: Only need to learn individual digit addition")
        print("  Carry propagation comes for free via recurrence")

    print(f"\n{'='*60}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
