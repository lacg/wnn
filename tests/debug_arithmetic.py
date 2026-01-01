"""
Debug Arithmetic Edge Cases

Investigate why hybrid seq2seq doesn't achieve 100% on all arithmetic tests.
"""

import random
from datetime import datetime

from torch import tensor, uint8

import sys
sys.path.insert(0, '/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn/tests')
from arithmetic import LearnedFullAdder, MultiDigitAdder
from division import LearnedSubtractor, MultiDigitSubtractor


def test_adder_edge_cases():
    """Test the full adder on edge cases."""
    print(f"\n{'='*60}")
    print("Debugging LearnedFullAdder")
    print(f"{'='*60}")

    adder = MultiDigitAdder(base=2, rng=42)
    adder.train()

    # Test specific edge cases
    edge_cases = [
        (0, 0),
        (1, 1),
        (255, 255),  # Max 8-bit
        (255, 1),    # Overflow
        (128, 128),  # Power of 2
        (127, 128),  # Just under overflow
        (0, 255),
        (1, 254),
    ]

    print("\nAddition edge cases:")
    failures = []
    for a, b in edge_cases:
        expected = a + b
        result = adder.add_int(a, b)
        ok = "✓" if result == expected else "✗"
        if result != expected:
            failures.append((a, b, expected, result))
        print(f"  {a:3d} + {b:3d} = {result:5d} (expected {expected:5d}) {ok}")

    return failures


def test_subtractor_edge_cases():
    """Test the subtractor on edge cases."""
    print(f"\n{'='*60}")
    print("Debugging LearnedSubtractor")
    print(f"{'='*60}")

    subtractor = MultiDigitSubtractor(base=2, rng=42)
    subtractor.train()

    # Test specific edge cases
    edge_cases = [
        (0, 0),
        (1, 1),
        (255, 0),
        (255, 255),
        (0, 1),      # Negative result
        (100, 200),  # Negative result
        (200, 100),
        (128, 64),
        (1, 255),    # Large negative
    ]

    print("\nSubtraction edge cases:")
    failures = []
    for a, b in edge_cases:
        result, negative = subtractor.subtract_int(a, b)

        # What we expect
        if a >= b:
            expected_result = a - b
            expected_negative = False
        else:
            expected_result = b - a  # Magnitude
            expected_negative = True

        ok = "✓" if (result == expected_result and negative == expected_negative) else "✗"
        if result != expected_result or negative != expected_negative:
            failures.append((a, b, expected_result, expected_negative, result, negative))

        sign = "-" if negative else "+"
        exp_sign = "-" if expected_negative else "+"
        print(f"  {a:3d} - {b:3d} = {sign}{result:3d} (expected {exp_sign}{expected_result:3d}) {ok}")

    return failures


def test_modular_arithmetic():
    """Test modular arithmetic handling."""
    print(f"\n{'='*60}")
    print("Testing Modular Arithmetic (mod 256)")
    print(f"{'='*60}")

    adder = MultiDigitAdder(base=2, rng=42)
    adder.train()

    subtractor = MultiDigitSubtractor(base=2, rng=42)
    subtractor.train()

    print("\nAddition mod 256:")
    add_failures = []
    random.seed(999)
    for _ in range(20):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        expected = (a + b) % 256
        result = adder.add_int(a, b) % 256  # Apply mod

        ok = "✓" if result == expected else "✗"
        if result != expected:
            add_failures.append((a, b, expected, result))
            print(f"  {a:3d} + {b:3d} mod 256 = {result:3d} (expected {expected:3d}) {ok}")

    print(f"  Addition failures: {len(add_failures)}/20")

    print("\nSubtraction mod 256:")
    sub_failures = []
    random.seed(888)
    for _ in range(20):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        expected = (a - b) % 256

        result, negative = subtractor.subtract_int(a, b)
        if negative:
            # Convert magnitude to two's complement
            actual = (256 - result) % 256
        else:
            actual = result % 256

        ok = "✓" if actual == expected else "✗"
        if actual != expected:
            sub_failures.append((a, b, expected, actual, result, negative))
            print(f"  {a:3d} - {b:3d} mod 256 = {actual:3d} (expected {expected:3d}) {ok}")
            print(f"       raw: result={result}, negative={negative}")

    print(f"  Subtraction failures: {len(sub_failures)}/20")

    return add_failures, sub_failures


def trace_subtraction_bug():
    """Trace through a specific subtraction to find the bug."""
    print(f"\n{'='*60}")
    print("Tracing Subtraction Bug")
    print(f"{'='*60}")

    subtractor = MultiDigitSubtractor(base=2, rng=42)
    subtractor.train()

    # Test case: 100 - 200
    a, b = 100, 200
    print(f"\nTracing {a} - {b}:")

    # Convert to binary digits (LSB first)
    a_digits = []
    temp_a = a
    while temp_a > 0 or len(a_digits) < 8:
        a_digits.append(temp_a % 2)
        temp_a //= 2
        if len(a_digits) >= 8 and temp_a == 0:
            break

    b_digits = []
    temp_b = b
    while temp_b > 0 or len(b_digits) < 8:
        b_digits.append(temp_b % 2)
        temp_b //= 2
        if len(b_digits) >= 8 and temp_b == 0:
            break

    # Pad to same length
    max_len = max(len(a_digits), len(b_digits))
    a_digits += [0] * (max_len - len(a_digits))
    b_digits += [0] * (max_len - len(b_digits))

    print(f"  a = {a} = {a_digits} (LSB first)")
    print(f"  b = {b} = {b_digits} (LSB first)")

    # Manual subtraction
    result_digits = []
    borrow = 0

    print("\n  Step-by-step subtraction:")
    for i in range(max_len):
        a_bit = a_digits[i]
        b_bit = b_digits[i]

        # Use the learned subtractor
        diff, new_borrow = subtractor.subtractor(a_bit, b_bit, borrow)

        # Expected result
        raw = a_bit - b_bit - borrow
        if raw < 0:
            exp_diff = raw + 2
            exp_borrow = 1
        else:
            exp_diff = raw
            exp_borrow = 0

        ok = "✓" if (diff == exp_diff and new_borrow == exp_borrow) else "✗"
        print(f"    bit {i}: {a_bit} - {b_bit} - {borrow} = {diff} borrow {new_borrow} (expected {exp_diff} borrow {exp_borrow}) {ok}")

        result_digits.append(diff)
        borrow = new_borrow

    # Convert result back
    result = sum(d * (2 ** i) for i, d in enumerate(result_digits))
    print(f"\n  Result digits: {result_digits}")
    print(f"  Result value: {result}")
    print(f"  Final borrow: {borrow} (1 means negative)")

    # What we expect
    expected = abs(a - b)
    print(f"\n  Expected magnitude: {expected}")
    print(f"  Expected negative: {a < b}")


def fix_and_retest():
    """Test with corrected modular handling."""
    print(f"\n{'='*60}")
    print("Testing Corrected Modular Handling")
    print(f"{'='*60}")

    adder = MultiDigitAdder(base=2, rng=42)
    adder.train()

    subtractor = MultiDigitSubtractor(base=2, rng=42)
    subtractor.train()

    def safe_add_mod256(a, b):
        """Addition mod 256 with proper handling."""
        result = adder.add_int(a, b)
        return result % 256

    def safe_sub_mod256(a, b):
        """Subtraction mod 256 with proper handling."""
        result, negative = subtractor.subtract_int(a, b)
        if negative:
            # a - b is negative, so we need 256 - |a - b|
            return (256 - result) % 256
        else:
            return result % 256

    print("\nTesting corrected functions:")
    random.seed(777)

    add_correct = 0
    sub_correct = 0
    total = 50

    for _ in range(total):
        a = random.randint(0, 255)
        b = random.randint(0, 255)

        # Addition
        expected_add = (a + b) % 256
        result_add = safe_add_mod256(a, b)
        if result_add == expected_add:
            add_correct += 1

        # Subtraction
        expected_sub = (a - b) % 256
        result_sub = safe_sub_mod256(a, b)
        if result_sub == expected_sub:
            sub_correct += 1

    print(f"  Addition:    {add_correct}/{total} = {100*add_correct/total:.0f}%")
    print(f"  Subtraction: {sub_correct}/{total} = {100*sub_correct/total:.0f}%")

    # Show any failures
    print("\nShowing any failures:")
    random.seed(777)
    shown = 0
    for _ in range(total):
        a = random.randint(0, 255)
        b = random.randint(0, 255)

        expected_sub = (a - b) % 256
        result_sub = safe_sub_mod256(a, b)

        if result_sub != expected_sub and shown < 5:
            print(f"  {a} - {b} mod 256: got {result_sub}, expected {expected_sub}")
            result, negative = subtractor.subtract_int(a, b)
            print(f"    raw: result={result}, negative={negative}")
            shown += 1


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Debugging Arithmetic Edge Cases")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    # Test adder
    add_failures = test_adder_edge_cases()

    # Test subtractor
    sub_failures = test_subtractor_edge_cases()

    # Test modular arithmetic
    mod_add_failures, mod_sub_failures = test_modular_arithmetic()

    # Trace a specific bug
    trace_subtraction_bug()

    # Test corrected handling
    fix_and_retest()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nAdder edge case failures: {len(add_failures)}")
    print(f"Subtractor edge case failures: {len(sub_failures)}")
    print(f"Modular addition failures: {len(mod_add_failures)}")
    print(f"Modular subtraction failures: {len(mod_sub_failures)}")

    print(f"\n{'='*60}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
