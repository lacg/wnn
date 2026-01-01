"""
Division Learning Test

Tests whether RAM networks can learn division by decomposition.

Binary division (shift-and-subtract):
- For each bit position, compare and conditionally subtract
- Reuses: BitLevelComparator (comparison) + LearnedFullAdder (subtraction)
- Produces both quotient and remainder

This is the inverse of multiplication's shift-and-add.
"""

import random
from datetime import datetime

from torch import zeros, uint8, tensor
from torch.nn import Module

from wnn.ram.core import RAMLayer

# Import from other tests
import sys
sys.path.insert(0, '/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn/tests')
from arithmetic import LearnedFullAdder, MultiDigitAdder


class LearnedSubtractor(Module):
    """
    Learn subtraction using full adder with borrow.

    a - b = a + (~b) + 1  (two's complement)

    Or directly learn (a, b, borrow_in) → (diff, borrow_out)
    """

    def __init__(self, base: int = 2, rng: int | None = None):
        super().__init__()
        self.base = base

        if base == 2:
            self.input_bits = 3  # a, b, borrow_in
            self.output_bits = 2  # diff, borrow_out
        else:
            self.input_bits = 9  # 4+4+1 for decimal
            self.output_bits = 5  # 4+1 for decimal

        self.subtractor = RAMLayer(
            total_input_bits=self.input_bits,
            num_neurons=self.output_bits,
            n_bits_per_neuron=self.input_bits,
            rng=rng,
        )
        self._trained = False

    def _encode_input(self, a: int, b: int, borrow: int) -> list[int]:
        if self.base == 2:
            return [a, b, borrow]
        else:
            a_bits = [(a >> i) & 1 for i in range(3, -1, -1)]
            b_bits = [(b >> i) & 1 for i in range(3, -1, -1)]
            return a_bits + b_bits + [borrow]

    def _decode_output(self, bits: list[int]) -> tuple[int, int]:
        if self.base == 2:
            return bits[0], bits[1]
        else:
            diff = sum(b << (3 - i) for i, b in enumerate(bits[:4]))
            return diff, bits[4]

    def train_all(self) -> int:
        errors = 0
        for a in range(self.base):
            for b in range(self.base):
                for borrow in range(2):
                    # a - b - borrow
                    result = a - b - borrow
                    if result < 0:
                        diff = result + self.base
                        borrow_out = 1
                    else:
                        diff = result
                        borrow_out = 0

                    inp_bits = self._encode_input(a, b, borrow)
                    inp = tensor(inp_bits, dtype=uint8)

                    if self.base == 2:
                        out_bits = [diff, borrow_out]
                    else:
                        out_bits = [(diff >> i) & 1 for i in range(3, -1, -1)] + [borrow_out]
                    out = tensor(out_bits, dtype=uint8)

                    errors += self.subtractor.commit(inp.unsqueeze(0), out.unsqueeze(0))

        self._trained = True
        return errors

    def forward(self, a: int, b: int, borrow: int) -> tuple[int, int]:
        inp_bits = self._encode_input(a, b, borrow)
        inp = tensor(inp_bits, dtype=uint8)
        out = self.subtractor(inp.unsqueeze(0)).squeeze()
        out_list = [int(x.item()) for x in out]
        return self._decode_output(out_list)

    def test_accuracy(self) -> float:
        correct = 0
        total = 0
        for a in range(self.base):
            for b in range(self.base):
                for borrow in range(2):
                    result = a - b - borrow
                    if result < 0:
                        exp_diff = result + self.base
                        exp_borrow = 1
                    else:
                        exp_diff = result
                        exp_borrow = 0

                    pred_diff, pred_borrow = self.forward(a, b, borrow)
                    if pred_diff == exp_diff and pred_borrow == exp_borrow:
                        correct += 1
                    total += 1
        return correct / total

    def __repr__(self):
        return f"LearnedSubtractor(base={self.base}, trained={self._trained})"


class MultiDigitSubtractor(Module):
    """Multi-digit subtraction with borrow propagation."""

    def __init__(self, base: int = 2, rng: int | None = None):
        super().__init__()
        self.base = base
        self.subtractor = LearnedSubtractor(base, rng)

    def train(self) -> int:
        return self.subtractor.train_all()

    def subtract(self, a_digits: list[int], b_digits: list[int]) -> tuple[list[int], bool]:
        """
        Subtract b from a (a - b).

        Returns (result_digits, negative) where negative=True if a < b.
        """
        max_len = max(len(a_digits), len(b_digits))
        a_padded = a_digits + [0] * (max_len - len(a_digits))
        b_padded = b_digits + [0] * (max_len - len(b_digits))

        result = []
        borrow = 0

        for i in range(max_len):
            diff, borrow = self.subtractor(a_padded[i], b_padded[i], borrow)
            result.append(diff)

        # If borrow remains, result is negative
        return result, borrow == 1

    def subtract_int(self, a: int, b: int) -> tuple[int, bool]:
        """Subtract integers, return (result, negative)."""
        a_digits = self._int_to_digits(a)
        b_digits = self._int_to_digits(b)
        result_digits, negative = self.subtract(a_digits, b_digits)
        result = self._digits_to_int(result_digits)
        return result, negative

    def _int_to_digits(self, n: int) -> list[int]:
        if n == 0:
            return [0]
        digits = []
        while n > 0:
            digits.append(n % self.base)
            n //= self.base
        return digits

    def _digits_to_int(self, digits: list[int]) -> int:
        result = 0
        for i, d in enumerate(digits):
            result += d * (self.base ** i)
        return result


class LearnedComparator(Module):
    """
    Simple comparator: is a >= b?

    For division, we need to know if we can subtract.
    Uses subtraction: a >= b iff (a - b) doesn't borrow.
    """

    def __init__(self, base: int = 2, rng: int | None = None):
        super().__init__()
        self.subtractor = MultiDigitSubtractor(base, rng)

    def train(self) -> int:
        return self.subtractor.train()

    def compare(self, a: int, b: int) -> bool:
        """Return True if a >= b."""
        _, negative = self.subtractor.subtract_int(a, b)
        return not negative


class BinaryDivider(Module):
    """
    Binary division using restoring division algorithm.

    For dividend / divisor:
    1. Start with remainder = 0
    2. For each bit of dividend (MSB to LSB):
       - Shift remainder left, bring in next dividend bit
       - If remainder >= divisor: subtract, quotient bit = 1
       - Else: quotient bit = 0
    3. Final remainder is the modulo result

    Reuses: subtraction (8 patterns) + comparison (via subtraction)
    """

    def __init__(self, rng: int | None = None):
        super().__init__()
        self.subtractor = MultiDigitSubtractor(base=2, rng=rng)

    def train(self) -> int:
        return self.subtractor.train()

    def divide(self, dividend: int, divisor: int, max_bits: int = 16) -> tuple[int, int]:
        """
        Divide dividend by divisor.

        Returns (quotient, remainder).
        Raises ValueError if divisor is 0.
        """
        if divisor == 0:
            raise ValueError("Division by zero")

        if dividend < divisor:
            return 0, dividend

        # Find bit width of dividend
        n_bits = dividend.bit_length()

        quotient = 0
        remainder = 0

        # Process from MSB to LSB
        for i in range(n_bits - 1, -1, -1):
            # Shift remainder left and bring in next bit of dividend
            remainder = (remainder << 1) | ((dividend >> i) & 1)

            # Try to subtract divisor
            diff, negative = self.subtractor.subtract_int(remainder, divisor)

            if not negative:  # remainder >= divisor
                remainder = diff
                quotient = (quotient << 1) | 1
            else:
                quotient = quotient << 1

        return quotient, remainder

    def __repr__(self):
        return f"BinaryDivider(subtractor={self.subtractor})"


def test_subtractor():
    """Test learned subtraction."""
    print(f"\n{'='*60}")
    print("Testing Binary Subtractor (8 patterns)")
    print(f"{'='*60}")

    sub = LearnedSubtractor(base=2, rng=42)

    acc_before = sub.test_accuracy()
    print(f"Accuracy before training: {acc_before:.1%}")

    errors = sub.train_all()
    print(f"Training corrections: {errors}")

    acc_after = sub.test_accuracy()
    print(f"Accuracy after training: {acc_after:.1%}")

    print("\nSubtractor truth table:")
    print("  a  b  borrow | diff borrow_out")
    for a in range(2):
        for b in range(2):
            for borrow in range(2):
                diff, bout = sub.forward(a, b, borrow)
                print(f"  {a}  {b}    {borrow}    |   {diff}      {bout}")

    return acc_after


def test_binary_division():
    """Test binary division."""
    print(f"\n{'='*60}")
    print("Testing Binary Division")
    print(f"{'='*60}")

    div = BinaryDivider(rng=42)
    errors = div.train()
    print(f"Subtractor trained with {errors} corrections")

    # Test on random divisions
    random.seed(123)
    n_test = 30
    correct = 0

    print(f"\nTesting on {n_test} random divisions...")
    for _ in range(n_test):
        dividend = random.randint(1, 255)
        divisor = random.randint(1, 15)

        expected_q = dividend // divisor
        expected_r = dividend % divisor

        try:
            result_q, result_r = div.divide(dividend, divisor)
            if result_q == expected_q and result_r == expected_r:
                correct += 1
        except:
            pass

    accuracy = correct / n_test
    print(f"Accuracy: {accuracy:.1%} ({correct}/{n_test})")

    # Examples
    print("\nExamples:")
    examples = [(15, 3), (100, 7), (255, 16), (17, 5), (8, 2)]
    for dividend, divisor in examples:
        expected_q = dividend // divisor
        expected_r = dividend % divisor
        try:
            result_q, result_r = div.divide(dividend, divisor)
            ok = "✓" if (result_q == expected_q and result_r == expected_r) else "✗"
            print(f"  {dividend} ÷ {divisor} = {result_q} R {result_r} "
                  f"(expected {expected_q} R {expected_r}) {ok}")
        except Exception as e:
            print(f"  {dividend} ÷ {divisor} = Error: {e}")

    return accuracy


def test_generalization():
    """Test division generalization to larger numbers."""
    print(f"\n{'='*60}")
    print("Testing Division Generalization")
    print(f"{'='*60}")

    div = BinaryDivider(rng=42)
    div.train()
    print("Trained with 8 subtractor patterns")

    sizes = [(8, 4, 50), (12, 6, 30), (16, 8, 20)]

    for div_bits, divisor_bits, n_test in sizes:
        max_dividend = (1 << div_bits) - 1
        max_divisor = (1 << divisor_bits) - 1
        random.seed(789 + div_bits)

        correct = 0
        for _ in range(n_test):
            dividend = random.randint(1, max_dividend)
            divisor = random.randint(1, max_divisor)

            expected_q = dividend // divisor
            expected_r = dividend % divisor

            try:
                result_q, result_r = div.divide(dividend, divisor)
                if result_q == expected_q and result_r == expected_r:
                    correct += 1
            except:
                pass

        print(f"  {div_bits}-bit ÷ {divisor_bits}-bit: {correct}/{n_test} = {correct/n_test:.1%}")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Division Learning Test")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    # Test 1: Subtractor
    sub_acc = test_subtractor()

    # Test 2: Binary division
    div_acc = test_binary_division()

    # Test 3: Generalization
    test_generalization()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nSubtractor accuracy: {sub_acc:.1%}")
    print(f"Division accuracy:   {div_acc:.1%}")
    print("\nPrimitives needed: 8 patterns (subtractor only)")
    print("Division reuses subtraction like multiplication reuses addition!")

    print(f"\n{'='*60}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
