"""
Computed Arithmetic FFN

Arithmetic operations computed directly from bit patterns.
Achieves 100% generalization with no training.
"""

from torch import Tensor, zeros, uint8
from torch.nn import Module

from wnn.ram.core.models import ArithmeticOp


def bits_to_int(bits: Tensor) -> int:
	"""Convert bit tensor to integer (MSB first)."""
	bits = bits.squeeze()
	val = 0
	for b in bits:
		val = val * 2 + int(b.item())
	return val


def int_to_bits(value: int, num_bits: int) -> Tensor:
	"""Convert integer to bit tensor (MSB first)."""
	bits = zeros(num_bits, dtype=uint8)
	for i in range(num_bits - 1, -1, -1):
		bits[num_bits - 1 - i] = (value >> i) & 1
	return bits


class ComputedArithmeticFFN(Module):
	"""
	Computed arithmetic FFN - transforms values using arithmetic operations.

	Key insight: Like SortingAttention computes comparisons from bits,
	ComputedArithmeticFFN computes arithmetic from bits.

	Generalizes 100% because the transformation is computed, not learned.

	Examples:
		- INCREMENT: A(0) -> B(1), B(1) -> C(2), ...
		- ROT13: A(0) -> N(13), B(1) -> O(14), N(13) -> A(0)
		- ADD_MOD: With constant=3, modulo=26: A->D, X->A
	"""

	def __init__(
		self,
		input_bits: int,
		operation: ArithmeticOp = ArithmeticOp.INCREMENT,
		constant: int = 1,
		modulo: int | None = None,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Bits per token
			operation: Arithmetic operation to apply
			constant: Constant for ADD/SUBTRACT operations
			modulo: Modulo for ADD_MOD/SUBTRACT_MOD (None = no modulo)
			rng: Random seed (unused, for API compatibility)
		"""
		super().__init__()

		self.input_bits = input_bits
		self.operation = operation
		self.constant = constant
		self.modulo = modulo
		self.max_value = (1 << input_bits) - 1

		# Set defaults for specific operations
		if operation == ArithmeticOp.ROT13:
			self.constant = 13
			self.modulo = 26

		op_name = operation.name
		mod_str = f" mod {self.modulo}" if self.modulo else ""
		const_str = f" {self.constant}" if operation in [
			ArithmeticOp.ADD, ArithmeticOp.SUBTRACT,
			ArithmeticOp.ADD_MOD, ArithmeticOp.SUBTRACT_MOD
		] else ""
		print(f"[ComputedArithmeticFFN] {input_bits}b, op={op_name}{const_str}{mod_str}")

	def _apply_operation(self, value: int) -> int:
		"""Apply the arithmetic operation to a value using match-case."""
		match self.operation:
			case ArithmeticOp.INCREMENT:
				result = value + 1

			case ArithmeticOp.DECREMENT:
				result = value - 1

			case ArithmeticOp.ADD:
				result = value + self.constant

			case ArithmeticOp.SUBTRACT:
				result = value - self.constant

			case ArithmeticOp.ADD_MOD:
				result = (value + self.constant) % self.modulo

			case ArithmeticOp.SUBTRACT_MOD:
				result = (value - self.constant) % self.modulo

			case ArithmeticOp.ROT13:
				result = (value + 13) % 26

			case ArithmeticOp.NEGATE:
				result = self.max_value - value

			case _:
				result = value

		# Clamp to valid range (unless modulo is used)
		if self.modulo is None:
			result = max(0, min(result, self.max_value))

		return result

	def forward(self, x: Tensor) -> Tensor:
		"""Apply arithmetic operation to input tensor."""
		x = x.squeeze()
		value = bits_to_int(x)
		result = self._apply_operation(value)
		return int_to_bits(result, self.input_bits)

	def __call__(self, x: Tensor) -> Tensor:
		"""Make it callable like other FFN modules."""
		return self.forward(x)

	def train_mapping(self, inp: Tensor, target: Tensor) -> int:
		"""No training needed - operation is computed."""
		return 0


class ComputedCopyFFN(Module):
	"""
	Identity FFN - just passes through the input unchanged.

	Useful as a placeholder or when only attention transformation is needed.
	"""

	def __init__(self, input_bits: int, rng: int | None = None):
		super().__init__()
		self.input_bits = input_bits
		print(f"[ComputedCopyFFN] {input_bits}b (identity)")

	def forward(self, x: Tensor) -> Tensor:
		return x.squeeze().clone()

	def __call__(self, x: Tensor) -> Tensor:
		return self.forward(x)

	def train_mapping(self, inp: Tensor, target: Tensor) -> int:
		return 0
