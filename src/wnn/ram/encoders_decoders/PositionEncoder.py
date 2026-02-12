"""
Base class for position encoders.

Position encoders convert sequence positions (integers) into bit vectors
that can be concatenated with token encodings for position-aware processing.
"""

from abc import ABC, abstractmethod
from torch import Tensor


class PositionEncoder(ABC):
	"""Abstract base class for position encoders."""

	@property
	@abstractmethod
	def n_bits(self) -> int:
		"""Number of bits used for position encoding."""
		pass

	@property
	@abstractmethod
	def max_seq_len(self) -> int:
		"""Maximum sequence length supported."""
		pass

	@abstractmethod
	def encode(self, position: int, device=None) -> Tensor:
		"""
		Encode a single position to bits.

		Args:
			position: Sequence position (0-indexed)
			device: Target device for tensor

		Returns:
			Tensor of shape [n_bits] with uint8 dtype
		"""
		pass

	@abstractmethod
	def decode(self, bits: Tensor) -> int:
		"""
		Decode position bits back to integer.

		Args:
			bits: Tensor of shape [n_bits]

		Returns:
			Position as integer
		"""
		pass

	def encode_sequence(self, seq_len: int, device=None) -> list[Tensor]:
		"""
		Encode all positions for a sequence.

		Args:
			seq_len: Length of sequence
			device: Target device

		Returns:
			List of position bit tensors
		"""
		return [self.encode(i, device) for i in range(seq_len)]
