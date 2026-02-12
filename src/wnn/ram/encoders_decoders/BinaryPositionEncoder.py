"""
Binary position encoder.

Encodes sequence positions as binary bit vectors:
  Position 0 -> [0, 0, 0]
  Position 1 -> [0, 0, 1]
  Position 5 -> [1, 0, 1]
  Position 7 -> [1, 1, 1]

The number of bits determines max sequence length:
  3 bits -> max 8 positions (0-7)
  4 bits -> max 16 positions (0-15)
  8 bits -> max 256 positions (0-255)
"""

from wnn.ram.encoders_decoders.PositionEncoder import PositionEncoder

from torch import Tensor
from torch import uint8
from torch import zeros


class BinaryPositionEncoder(PositionEncoder):
	"""
	Binary position encoder for RAM networks.

	Simple and efficient: positions encoded as binary numbers.
	Natural for RAM neurons since they work directly with bit patterns.
	"""

	def __init__(
		self,
		n_position_bits: int,
		max_seq_len: int | None = None,
	):
		"""
		Args:
			n_position_bits: Number of bits for position encoding
			max_seq_len: Optional explicit max length (default: 2^n_position_bits)
		"""
		self._n_bits = n_position_bits
		self._max_seq_len = max_seq_len or (1 << n_position_bits)

		# Validate
		if self._max_seq_len > (1 << n_position_bits):
			required_bits = self.bits_needed(max_seq_len)
			raise ValueError(
				f"max_seq_len={max_seq_len} requires at least "
				f"{required_bits} bits, got {n_position_bits}"
			)

		# Cache for efficiency
		self._cache: dict[int, Tensor] = {}

	@property
	def n_bits(self) -> int:
		return self._n_bits

	@property
	def max_seq_len(self) -> int:
		return self._max_seq_len

	def encode(self, position: int, device=None) -> Tensor:
		"""
		Encode a single position as binary bits.

		Args:
			position: Sequence position (0-indexed)
			device: Target device for tensor

		Returns:
			Tensor of shape [n_bits] with uint8 dtype
		"""
		if position < 0:
			raise ValueError(f"Position must be non-negative, got {position}")
		if position >= self._max_seq_len:
			raise ValueError(
				f"Position {position} exceeds max_seq_len {self._max_seq_len}"
			)

		# Check cache
		if position in self._cache:
			cached = self._cache[position]
			if device is not None and cached.device != device:
				return cached.to(device)
			return cached.clone()

		# Binary encoding: MSB first
		bits = zeros(self._n_bits, dtype=uint8)
		val = position
		for i in range(self._n_bits - 1, -1, -1):
			bits[i] = val & 1
			val >>= 1

		self._cache[position] = bits

		if device is not None:
			return bits.to(device)
		return bits.clone()

	def decode(self, bits: Tensor) -> int:
		"""
		Decode position bits back to integer.

		Args:
			bits: Tensor of shape [n_bits]

		Returns:
			Position as integer
		"""
		if bits.ndim != 1 or bits.numel() != self._n_bits:
			raise ValueError(
				f"Expected shape [{self._n_bits}], got {bits.shape}"
			)

		position = 0
		for bit in bits:
			position = (position << 1) | int(bit)
		return position

	@staticmethod
	def bits_needed(max_seq_len: int) -> int:
		"""Calculate minimum bits needed for a given max sequence length."""
		if max_seq_len <= 1:
			return 1
		# (max_seq_len - 1).bit_length() gives bits needed to represent 0 to max_seq_len-1
		return (max_seq_len - 1).bit_length()

	def __repr__(self):
		return f"BinaryPositionEncoder(bits={self._n_bits}, max_len={self._max_seq_len})"
