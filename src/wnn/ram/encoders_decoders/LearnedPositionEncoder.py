"""
Learned position encoder.

Uses RAMLayer to learn position embeddings rather than using fixed encodings.
The learned embeddings can capture task-specific position patterns.

Unlike BINARY (fixed bit pattern) or RELATIVE (fixed distance encoding),
LEARNED mode trains position → embedding mappings during model training.
"""

from wnn.ram.encoders_decoders.PositionEncoder import PositionEncoder
from wnn.ram.encoders_decoders.BinaryPositionEncoder import BinaryPositionEncoder

from torch import Tensor, zeros, cat, uint8
from torch.nn import Module


class LearnedPositionEncoder(PositionEncoder, Module):
	"""
	Learned position encoder using RAMLayer.

	Positions are first converted to binary bits, then passed through
	a RAMLayer that learns optimal embeddings for each position.

	This allows the model to learn task-specific position representations
	rather than relying on fixed encodings.
	"""

	def __init__(
		self,
		n_output_bits: int,
		max_seq_len: int,
		n_input_bits: int | None = None,
		rng: int | None = None,
	):
		"""
		Args:
			n_output_bits: Size of learned position embedding
			max_seq_len: Maximum sequence length supported
			n_input_bits: Input bits for binary position (auto if None)
			rng: Random seed for RAMLayer initialization
		"""
		Module.__init__(self)

		self._n_output_bits = n_output_bits
		self._max_seq_len = max_seq_len

		# Binary encoder for position → bits
		if n_input_bits is None:
			n_input_bits = BinaryPositionEncoder.bits_needed(max_seq_len)
		self._n_input_bits = n_input_bits
		self._binary_encoder = BinaryPositionEncoder(n_input_bits, max_seq_len)

		# Lazy import to avoid circular dependency
		from wnn.ram.core import RAMLayer

		# RAMLayer learns position → embedding
		self._position_ram = RAMLayer(
			total_input_bits=n_input_bits,
			num_neurons=n_output_bits,
			n_bits_per_neuron=n_input_bits,
			rng=rng,
		)

		# Cache for efficiency
		self._cache: dict[int, Tensor] = {}

		# For relative position learning
		self._relative_ram = None
		self._max_distance = None

	@property
	def n_bits(self) -> int:
		"""Number of output bits for position embedding."""
		return self._n_output_bits

	@property
	def max_seq_len(self) -> int:
		return self._max_seq_len

	@property
	def position_ram(self) -> Module:
		"""Access to underlying RAMLayer for training."""
		return self._position_ram

	def encode(self, position: int, device=None) -> Tensor:
		"""
		Encode a position to learned embedding.

		Args:
			position: Sequence position (0-indexed)
			device: Target device for tensor

		Returns:
			Tensor of shape [n_output_bits] with uint8 dtype
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

		# Binary encode position
		pos_bits = self._binary_encoder.encode(position, device)

		# Pass through RAMLayer
		embedding = self._position_ram(pos_bits.unsqueeze(0)).squeeze()

		# Cache result
		self._cache[position] = embedding.clone()

		return embedding

	def decode(self, bits: Tensor) -> int:
		"""
		Decode position from embedding (best-effort reverse lookup).

		Note: This is approximate since learned embeddings may not be unique.
		Returns the position whose embedding is closest (hamming distance).

		Args:
			bits: Tensor of shape [n_output_bits]

		Returns:
			Best-matching position as integer
		"""
		if bits.ndim != 1 or bits.numel() != self._n_output_bits:
			raise ValueError(
				f"Expected shape [{self._n_output_bits}], got {bits.shape}"
			)

		# Find closest cached embedding
		best_pos = 0
		best_dist = float('inf')

		for pos in range(self._max_seq_len):
			embedding = self.encode(pos, bits.device)
			dist = (embedding != bits).sum().item()
			if dist < best_dist:
				best_dist = dist
				best_pos = pos
			if dist == 0:
				break

		return best_pos

	def clear_cache(self):
		"""Clear embedding cache (useful after training updates)."""
		self._cache.clear()

	def enable_relative(self, max_distance: int, rng: int | None = None):
		"""
		Enable relative position encoding.

		Creates a separate RAMLayer for learning relative position embeddings.

		Args:
			max_distance: Maximum relative distance to support
			rng: Random seed
		"""
		from wnn.ram.core import RAMLayer

		self._max_distance = max_distance
		# Input: sign bit + magnitude bits
		n_rel_input_bits = 1 + max_distance.bit_length()

		self._relative_ram = RAMLayer(
			total_input_bits=n_rel_input_bits,
			num_neurons=self._n_output_bits,
			n_bits_per_neuron=n_rel_input_bits,
			rng=rng,
		)

	def encode_relative(self, query_pos: int, key_pos: int, device=None) -> Tensor:
		"""
		Encode relative position (distance between query and key).

		Args:
			query_pos: Query position
			key_pos: Key position
			device: Target device

		Returns:
			Tensor of shape [n_output_bits] with learned relative embedding
		"""
		if self._relative_ram is None:
			raise RuntimeError(
				"Relative encoding not enabled. Call enable_relative() first."
			)

		distance = query_pos - key_pos

		# Clamp to max distance
		if abs(distance) > self._max_distance:
			distance = self._max_distance if distance > 0 else -self._max_distance

		# Encode: sign bit + magnitude bits
		sign_bit = 0 if distance >= 0 else 1
		magnitude = abs(distance)

		n_mag_bits = self._max_distance.bit_length()
		bits = zeros(1 + n_mag_bits, dtype=uint8, device=device)
		bits[0] = sign_bit

		# Binary encode magnitude
		for i in range(n_mag_bits - 1, -1, -1):
			bits[1 + i] = magnitude & 1
			magnitude >>= 1

		# Pass through relative RAMLayer
		embedding = self._relative_ram(bits.unsqueeze(0)).squeeze()
		return embedding

	def train_position(self, position: int, target_embedding: Tensor) -> int:
		"""
		Train embedding for a specific position.

		Args:
			position: Position to train
			target_embedding: Desired embedding

		Returns:
			Number of errors (bits that couldn't be set)
		"""
		pos_bits = self._binary_encoder.encode(position, target_embedding.device)
		errors = self._position_ram.commit(pos_bits.unsqueeze(0), target_embedding.unsqueeze(0))

		# Invalidate cache for this position
		if position in self._cache:
			del self._cache[position]

		return errors

	def __repr__(self):
		rel_info = f", relative=True" if self._relative_ram else ""
		return (
			f"LearnedPositionEncoder(output_bits={self._n_output_bits}, "
			f"max_len={self._max_seq_len}{rel_info})"
		)
