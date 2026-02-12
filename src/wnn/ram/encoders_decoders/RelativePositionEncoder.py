"""
Relative position encoder.

Instead of absolute positions (0, 1, 2, ...), encodes relative distances
between query and key positions.

Useful for patterns like "attend to 2 positions back" which is
position-invariant (works the same regardless of absolute position).

Distance encoding uses sign-magnitude format:
  Same position (d=0)  -> [0, 0, 0, 0]
  1 ahead (d=+1)       -> [0, 0, 0, 1]
  1 behind (d=-1)      -> [1, 0, 0, 1]  (sign bit + magnitude)
  3 behind (d=-3)      -> [1, 0, 1, 1]
"""

from wnn.ram.encoders_decoders.PositionEncoder import PositionEncoder

from torch import Tensor
from torch import uint8
from torch import zeros


class RelativePositionEncoder(PositionEncoder):
	"""
	Relative position encoder for attention patterns.

	Encodes the distance between two positions rather than absolute positions.
	First bit is sign (0=positive/zero, 1=negative), remaining bits are magnitude.
	"""

	def __init__(
		self,
		n_distance_bits: int,
		max_distance: int | None = None,
	):
		"""
		Args:
			n_distance_bits: Total bits (1 sign bit + n-1 magnitude bits)
			max_distance: Max relative distance (default: 2^(n-1) - 1)
		"""
		self._n_bits = n_distance_bits
		self._n_magnitude_bits = n_distance_bits - 1  # Reserve 1 for sign
		self._max_distance = max_distance or ((1 << self._n_magnitude_bits) - 1)

		if self._max_distance > (1 << self._n_magnitude_bits) - 1:
			raise ValueError(
				f"max_distance={max_distance} requires more magnitude bits"
			)

		# For PositionEncoder interface, max_seq_len is 2*max_distance+1
		# (positions from -max_distance to +max_distance)
		self._max_seq_len = 2 * self._max_distance + 1

	@property
	def n_bits(self) -> int:
		return self._n_bits

	@property
	def max_seq_len(self) -> int:
		return self._max_seq_len

	@property
	def max_distance(self) -> int:
		return self._max_distance

	def encode(self, position: int, device=None) -> Tensor:
		"""
		Encode a relative distance as bits.

		For RelativePositionEncoder, 'position' is interpreted as a signed distance.
		Use encode_relative() for the two-position interface.

		Args:
			position: Relative distance (can be negative)
			device: Target device

		Returns:
			Tensor of shape [n_bits] with [sign_bit, magnitude_bits...]
		"""
		distance = position

		# Clamp to max distance
		distance = max(-self._max_distance, min(self._max_distance, distance))

		bits = zeros(self._n_bits, dtype=uint8)

		# Sign bit: 0 = positive/zero, 1 = negative
		if distance < 0:
			bits[0] = 1
			distance = -distance

		# Magnitude bits (LSB first after sign)
		for i in range(self._n_bits - 1, 0, -1):
			bits[i] = distance & 1
			distance >>= 1

		if device is not None:
			return bits.to(device)
		return bits

	def encode_relative(self, query_pos: int, key_pos: int, device=None) -> Tensor:
		"""
		Encode relative distance between query and key positions.

		Args:
			query_pos: Position of query
			key_pos: Position of key
			device: Target device

		Returns:
			Tensor of shape [n_bits]
		"""
		distance = key_pos - query_pos
		return self.encode(distance, device)

	def decode(self, bits: Tensor) -> int:
		"""
		Decode distance bits back to signed integer.

		Args:
			bits: Tensor of shape [n_bits]

		Returns:
			Signed distance
		"""
		if bits.ndim != 1 or bits.numel() != self._n_bits:
			raise ValueError(
				f"Expected shape [{self._n_bits}], got {bits.shape}"
			)

		# Extract sign
		is_negative = bool(bits[0])

		# Extract magnitude
		magnitude = 0
		for bit in bits[1:]:
			magnitude = (magnitude << 1) | int(bit)

		return -magnitude if is_negative else magnitude

	def __repr__(self):
		return f"RelativePositionEncoder(bits={self._n_bits}, max_dist={self._max_distance})"
