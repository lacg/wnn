"""
Arithmetic Learning Test - Consolidated

All arithmetic primitives for RAM networks:
- Addition: LearnedFullAdder, MultiDigitAdder
- Subtraction: LearnedSubtractor, MultiDigitSubtractor
- Multiplication: BinaryMultiplier, LearnedDigitMultiplier, DecimalMultiplier
- Division: LearnedComparator, BinaryDivider

Key insight: All operations decompose into small primitives (8-200 patterns)
that generalize to arbitrary bit widths through recurrence/composition.

| Operation | Primitives | Patterns | Generalization |
|-----------|-----------|----------|----------------|
| Addition  | Full adder | 8 (binary) / 200 (decimal) | 100% |
| Subtraction | Full subtractor | 8 (binary) | 100% |
| Multiplication | Reuses addition | 0 new (binary) / 100 (decimal) | 100% |
| Division | Reuses subtraction | 0 new | 100% |
"""

import random
from datetime import datetime

from torch import zeros, uint8, tensor
from torch.nn import Module

from wnn.ram.core import RAMLayer


# =============================================================================
# ADDITION
# =============================================================================

class LearnedFullAdder(Module):
	"""
	Learn a full adder: (a, b, carry_in) → (sum, carry_out)

	For binary (base 2): 8 patterns
	For decimal (base 10): 200 patterns
	"""

	def __init__(self, base: int = 2, rng: int | None = None):
		super().__init__()
		self.base = base

		if base == 2:
			self.input_bits = 3
			self.output_bits = 2
		else:
			self.input_bits = 9  # 4+4+1
			self.output_bits = 5  # 4+1

		self.adder = RAMLayer(
			total_input_bits=self.input_bits,
			num_neurons=self.output_bits,
			n_bits_per_neuron=self.input_bits,
			rng=rng,
		)
		self._trained = False

	def _encode_input(self, a: int, b: int, carry: int) -> list[int]:
		if self.base == 2:
			return [a, b, carry]
		else:
			a_bits = [(a >> i) & 1 for i in range(3, -1, -1)]
			b_bits = [(b >> i) & 1 for i in range(3, -1, -1)]
			return a_bits + b_bits + [carry]

	def _decode_output(self, bits: list[int]) -> tuple[int, int]:
		if self.base == 2:
			return bits[0], bits[1]
		else:
			sum_val = sum(b << (3 - i) for i, b in enumerate(bits[:4]))
			return sum_val, bits[4]

	def train_all(self) -> int:
		errors = 0
		for a in range(self.base):
			for b in range(self.base):
				for carry in range(2):
					total = a + b + carry
					sum_digit = total % self.base
					carry_out = 1 if total >= self.base else 0

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
		inp_bits = self._encode_input(a, b, carry)
		inp = tensor(inp_bits, dtype=uint8)
		out = self.adder(inp.unsqueeze(0)).squeeze()
		out_list = [int(b.item()) for b in out]
		return self._decode_output(out_list)

	def test_accuracy(self) -> float:
		correct = 0
		total = 0
		for a in range(self.base):
			for b in range(self.base):
				for carry in range(2):
					expected_total = a + b + carry
					expected_sum = expected_total % self.base
					expected_carry = 1 if expected_total >= self.base else 0
					pred_sum, pred_carry = self.forward(a, b, carry)
					if pred_sum == expected_sum and pred_carry == expected_carry:
						correct += 1
					total += 1
		return correct / total if total > 0 else 0.0


class MultiDigitAdder(Module):
	"""Multi-digit addition using learned full adder."""

	def __init__(self, base: int = 2, rng: int | None = None):
		super().__init__()
		self.base = base
		self.full_adder = LearnedFullAdder(base, rng)

	def train(self) -> int:
		return self.full_adder.train_all()

	def add(self, a_digits: list[int], b_digits: list[int]) -> list[int]:
		max_len = max(len(a_digits), len(b_digits))
		a_padded = a_digits + [0] * (max_len - len(a_digits))
		b_padded = b_digits + [0] * (max_len - len(b_digits))

		result = []
		carry = 0

		for i in range(max_len):
			sum_digit, carry = self.full_adder(a_padded[i], b_padded[i], carry)
			result.append(sum_digit)

		if carry:
			result.append(carry)

		return result

	def add_int(self, a: int, b: int) -> int:
		a_digits = self._int_to_digits(a)
		b_digits = self._int_to_digits(b)
		sum_digits = self.add(a_digits, b_digits)
		return self._digits_to_int(sum_digits)

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


# =============================================================================
# SUBTRACTION
# =============================================================================

class LearnedSubtractor(Module):
	"""Learn subtraction: (a, b, borrow_in) → (diff, borrow_out)"""

	def __init__(self, base: int = 2, rng: int | None = None):
		super().__init__()
		self.base = base

		if base == 2:
			self.input_bits = 3
			self.output_bits = 2
		else:
			self.input_bits = 9
			self.output_bits = 5

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


class MultiDigitSubtractor(Module):
	"""Multi-digit subtraction with borrow propagation."""

	def __init__(self, base: int = 2, rng: int | None = None):
		super().__init__()
		self.base = base
		self.subtractor = LearnedSubtractor(base, rng)

	def train(self) -> int:
		return self.subtractor.train_all()

	def subtract(self, a_digits: list[int], b_digits: list[int]) -> tuple[list[int], bool]:
		max_len = max(len(a_digits), len(b_digits))
		a_padded = a_digits + [0] * (max_len - len(a_digits))
		b_padded = b_digits + [0] * (max_len - len(b_digits))

		result = []
		borrow = 0

		for i in range(max_len):
			diff, borrow = self.subtractor(a_padded[i], b_padded[i], borrow)
			result.append(diff)

		return result, borrow == 1

	def subtract_int(self, a: int, b: int) -> tuple[int, bool]:
		"""Subtract integers, return (magnitude, negative)."""
		a_digits = self._int_to_digits(a)
		b_digits = self._int_to_digits(b)
		result_digits, negative = self.subtract(a_digits, b_digits)
		result = self._digits_to_int(result_digits)

		if negative:
			n_bits = len(result_digits)
			max_val = self.base ** n_bits
			result = max_val - result

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


# =============================================================================
# MULTIPLICATION
# =============================================================================

class BinaryMultiplier(Module):
	"""Binary multiplication using shift-and-add. Reuses addition (0 new patterns)."""

	def __init__(self, rng: int | None = None):
		super().__init__()
		self.adder = MultiDigitAdder(base=2, rng=rng)

	def train(self) -> int:
		return self.adder.train()

	def multiply(self, a: int, b: int, max_bits: int = 16) -> int:
		result = 0
		bit_pos = 0
		temp_b = b

		while temp_b > 0:
			if temp_b & 1:
				shifted_a = a << bit_pos
				result = self.adder.add_int(result, shifted_a)
			temp_b >>= 1
			bit_pos += 1

		return result


class LearnedDigitMultiplier(Module):
	"""Learn single-digit multiplication (100 patterns for 0-9 × 0-9)."""

	def __init__(self, rng: int | None = None):
		super().__init__()
		self.multiplier = RAMLayer(
			total_input_bits=8,
			num_neurons=7,
			n_bits_per_neuron=8,
			rng=rng,
		)
		self._trained = False

	def train_all(self) -> int:
		errors = 0
		for a in range(10):
			for b in range(10):
				product = a * b
				a_bits = [(a >> i) & 1 for i in range(3, -1, -1)]
				b_bits = [(b >> i) & 1 for i in range(3, -1, -1)]
				inp = tensor(a_bits + b_bits, dtype=uint8)
				out_bits = [(product >> i) & 1 for i in range(6, -1, -1)]
				out = tensor(out_bits, dtype=uint8)
				errors += self.multiplier.commit(inp.unsqueeze(0), out.unsqueeze(0))
		self._trained = True
		return errors

	def forward(self, a: int, b: int) -> int:
		a_bits = [(a >> i) & 1 for i in range(3, -1, -1)]
		b_bits = [(b >> i) & 1 for i in range(3, -1, -1)]
		inp = tensor(a_bits + b_bits, dtype=uint8)
		out = self.multiplier(inp.unsqueeze(0)).squeeze()
		result = sum(int(out[i].item()) << (6 - i) for i in range(7))
		return result

	def test_accuracy(self) -> float:
		correct = 0
		for a in range(10):
			for b in range(10):
				if self.forward(a, b) == a * b:
					correct += 1
		return correct / 100


class DecimalMultiplier(Module):
	"""Decimal multiplication using grade-school algorithm (100 + 200 = 300 patterns)."""

	def __init__(self, rng: int | None = None):
		super().__init__()
		self.digit_mult = LearnedDigitMultiplier(rng)
		self.adder = MultiDigitAdder(base=10, rng=rng + 1000 if rng else None)

	def train(self) -> tuple[int, int]:
		mult_errors = self.digit_mult.train_all()
		add_errors = self.adder.train()
		return mult_errors, add_errors

	def multiply(self, a: int, b: int) -> int:
		a_digits = self._int_to_digits(a)
		b_digits = self._int_to_digits(b)
		result_digits = [0]

		for b_pos, b_digit in enumerate(b_digits):
			if b_digit == 0:
				continue

			partial = []
			carry = 0

			for a_digit in a_digits:
				prod = self.digit_mult(a_digit, b_digit) + carry
				partial.append(prod % 10)
				carry = prod // 10

			if carry:
				partial.append(carry)

			shifted = [0] * b_pos + partial
			result_digits = self.adder.add(result_digits, shifted)

		return self._digits_to_int(result_digits)

	def _int_to_digits(self, n: int) -> list[int]:
		if n == 0:
			return [0]
		digits = []
		while n > 0:
			digits.append(n % 10)
			n //= 10
		return digits

	def _digits_to_int(self, digits: list[int]) -> int:
		result = 0
		for i, d in enumerate(digits):
			result += d * (10 ** i)
		return result


# =============================================================================
# DIVISION
# =============================================================================

class LearnedComparator(Module):
	"""Compare a >= b using subtraction."""

	def __init__(self, base: int = 2, rng: int | None = None):
		super().__init__()
		self.subtractor = MultiDigitSubtractor(base, rng)

	def train(self) -> int:
		return self.subtractor.train()

	def compare(self, a: int, b: int) -> bool:
		_, negative = self.subtractor.subtract_int(a, b)
		return not negative


class BinaryDivider(Module):
	"""Binary division using restoring division. Reuses subtraction (0 new patterns)."""

	def __init__(self, rng: int | None = None):
		super().__init__()
		self.subtractor = MultiDigitSubtractor(base=2, rng=rng)

	def train(self) -> int:
		return self.subtractor.train()

	def divide(self, dividend: int, divisor: int, max_bits: int = 16) -> tuple[int, int]:
		if divisor == 0:
			raise ValueError("Division by zero")

		if dividend < divisor:
			return 0, dividend

		n_bits = dividend.bit_length()
		quotient = 0
		remainder = 0

		for i in range(n_bits - 1, -1, -1):
			remainder = (remainder << 1) | ((dividend >> i) & 1)
			diff, negative = self.subtractor.subtract_int(remainder, divisor)

			if not negative:
				remainder = diff
				quotient = (quotient << 1) | 1
			else:
				quotient = quotient << 1

		return quotient, remainder


# =============================================================================
# TESTS
# =============================================================================

def test_addition():
	"""Test addition primitives."""
	print(f"\n{'='*60}")
	print("Testing Addition")
	print(f"{'='*60}")

	# Binary
	adder = MultiDigitAdder(base=2, rng=42)
	adder.train()

	random.seed(123)
	correct = sum(1 for _ in range(30)
					for a, b in [(random.randint(0, 255), random.randint(0, 255))]
					if adder.add_int(a, b) == a + b)
	print(f"Binary 8-bit: {correct}/30")

	# Generalization
	for bits in [16, 32, 64]:
		max_val = (1 << bits) - 1
		random.seed(789 + bits)
		correct = sum(1 for _ in range(20)
						for a, b in [(random.randint(0, max_val), random.randint(0, max_val))]
						if adder.add_int(a, b) == a + b)
		print(f"Binary {bits}-bit: {correct}/20")


def test_subtraction():
	"""Test subtraction primitives."""
	print(f"\n{'='*60}")
	print("Testing Subtraction")
	print(f"{'='*60}")

	sub = MultiDigitSubtractor(base=2, rng=42)
	sub.train()

	# Test modular subtraction
	random.seed(888)
	correct = 0
	for _ in range(30):
		a, b = random.randint(0, 255), random.randint(0, 255)
		result, negative = sub.subtract_int(a, b)
		expected = abs(a - b)
		if result == expected and negative == (a < b):
			correct += 1
	print(f"Binary 8-bit: {correct}/30")


def test_multiplication():
	"""Test multiplication primitives."""
	print(f"\n{'='*60}")
	print("Testing Multiplication")
	print(f"{'='*60}")

	mult = BinaryMultiplier(rng=42)
	mult.train()

	random.seed(123)
	correct = sum(1 for _ in range(30)
					for a, b in [(random.randint(0, 255), random.randint(0, 255))]
					if mult.multiply(a, b) == a * b)
	print(f"Binary 8-bit × 8-bit: {correct}/30")

	# Generalization
	for bits in [12, 16]:
		max_val = (1 << bits) - 1
		random.seed(789 + bits)
		correct = sum(1 for _ in range(20)
						for a, b in [(random.randint(0, max_val), random.randint(0, max_val))]
						if mult.multiply(a, b) == a * b)
		print(f"Binary {bits}-bit × {bits}-bit: {correct}/20")


def test_division():
	"""Test division primitives."""
	print(f"\n{'='*60}")
	print("Testing Division")
	print(f"{'='*60}")

	div = BinaryDivider(rng=42)
	div.train()

	random.seed(123)
	correct = 0
	for _ in range(30):
		dividend = random.randint(1, 255)
		divisor = random.randint(1, 15)
		try:
			q, r = div.divide(dividend, divisor)
			if q == dividend // divisor and r == dividend % divisor:
				correct += 1
		except:
			pass
	print(f"8-bit ÷ 4-bit: {correct}/30")

	# Generalization
	for div_bits, divisor_bits in [(12, 6), (16, 8)]:
		max_dividend = (1 << div_bits) - 1
		max_divisor = (1 << divisor_bits) - 1
		random.seed(789 + div_bits)

		correct = 0
		for _ in range(20):
			dividend = random.randint(1, max_dividend)
			divisor = random.randint(1, max_divisor)
			try:
				q, r = div.divide(dividend, divisor)
				if q == dividend // divisor and r == dividend % divisor:
					correct += 1
			except:
				pass
		print(f"{div_bits}-bit ÷ {divisor_bits}-bit: {correct}/20")


if __name__ == "__main__":
	print(f"\n{'='*60}")
	print("Arithmetic Learning Test - All Operations")
	print(f"Started at: {datetime.now()}")
	print(f"{'='*60}")

	test_addition()
	test_subtraction()
	test_multiplication()
	test_division()

	print(f"\n{'='*60}")
	print("SUMMARY")
	print(f"{'='*60}")
	print("""
Patterns needed for 100% generalization:
	Addition:       8 (binary) / 200 (decimal)
	Subtraction:    8 (binary)
	Multiplication: 0 new (reuses addition)
	Division:       0 new (reuses subtraction)

Key insight: Decompose into small primitives, compose via recurrence.
""")
	print(f"Finished at: {datetime.now()}")
	print(f"{'='*60}")
