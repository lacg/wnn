"""
Factory for creating position encoders.
"""

from wnn.ram.enums import PositionMode
from wnn.ram.encoders_decoders.PositionEncoder import PositionEncoder
from wnn.ram.encoders_decoders.BinaryPositionEncoder import BinaryPositionEncoder
from wnn.ram.encoders_decoders.RelativePositionEncoder import RelativePositionEncoder
from wnn.ram.encoders_decoders.LearnedPositionEncoder import LearnedPositionEncoder


class PositionEncoderFactory:
	"""Factory for creating position encoders based on mode."""

	@staticmethod
	def create(
		mode: PositionMode,
		n_position_bits: int | None = None,
		max_seq_len: int | None = None,
		max_distance: int | None = None,
		rng: int | None = None,
	) -> PositionEncoder | None:
		"""
		Create a position encoder.

		Args:
			mode: Position encoding mode (NONE, BINARY, RELATIVE, LEARNED)
			n_position_bits: Number of bits for encoding (auto-calculated if None)
			max_seq_len: Maximum sequence length (for BINARY/LEARNED modes)
			max_distance: Maximum relative distance (for RELATIVE mode)
			rng: Random seed (for LEARNED mode)

		Returns:
			PositionEncoder instance, or None if mode is NONE
		"""
		match mode:
			case PositionMode.NONE:
				return None

			case PositionMode.BINARY:
				# Auto-calculate bits if not specified
				if n_position_bits is None:
					if max_seq_len is None:
						max_seq_len = 64  # Default
					n_position_bits = BinaryPositionEncoder.bits_needed(max_seq_len)
				return BinaryPositionEncoder(n_position_bits, max_seq_len)

			case PositionMode.RELATIVE:
				# Auto-calculate bits if not specified
				if n_position_bits is None:
					if max_distance is None:
						max_distance = 31  # Default
					# Need 1 sign bit + magnitude bits
					# max_distance.bit_length() gives bits to represent max_distance
					n_position_bits = 1 + max_distance.bit_length()
				return RelativePositionEncoder(n_position_bits, max_distance)

			case PositionMode.LEARNED:
				# Learned position embeddings via RAMLayer
				if max_seq_len is None:
					max_seq_len = 64  # Default
				if n_position_bits is None:
					# Default: same size as binary encoding
					n_position_bits = BinaryPositionEncoder.bits_needed(max_seq_len)
				return LearnedPositionEncoder(
					n_output_bits=n_position_bits,
					max_seq_len=max_seq_len,
					rng=rng,
				)

			case _:
				raise ValueError(f"Unsupported PositionMode: {mode}")

	@staticmethod
	def bits_for_seq_len(seq_len: int) -> int:
		"""Calculate minimum bits needed for a sequence length."""
		return BinaryPositionEncoder.bits_needed(seq_len)
