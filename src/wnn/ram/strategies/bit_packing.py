"""
Bit Packing Utilities - Consistent encoding between Python and Rust.

This module provides utilities for packing word bits into u64 arrays,
matching the exact format used by the Rust accelerator for HashMap lookups.

The format is:
- Each word is encoded to BITS_PER_WORD bits (default 12)
- Words are packed sequentially into u64 values
- For n words: total_bits = n * BITS_PER_WORD, num_u64s = ceil(total_bits / 64)

Example for n=6 words with 12 bits each:
- total_bits = 72
- num_u64s = 2
- Word 0 bits 0-11 → u64[0] bits 0-11
- Word 1 bits 0-11 → u64[0] bits 12-23
- Word 2 bits 0-11 → u64[0] bits 24-35
- Word 3 bits 0-11 → u64[0] bits 36-47
- Word 4 bits 0-11 → u64[0] bits 48-59, u64[1] bits 0-3
- Word 5 bits 0-11 → u64[1] bits 4-15
"""

from typing import Sequence


class ContextBitPacker:
	"""
	Packs word bit codes into u64 arrays for Rust-compatible HashMap keys.

	This class ensures Python and Rust use identical key formats for
	exact RAM context lookups.

	Usage:
		packer = ContextBitPacker(bits_per_word=12)
		packed = packer.pack([0x123, 0x456, 0x789])  # Returns tuple of u64s
	"""

	def __init__(self, bits_per_word: int = 12):
		"""
		Initialize the packer.

		Args:
			bits_per_word: Number of bits per word code (default 12).
		"""
		self.bits_per_word = bits_per_word

	def pack(self, word_bits: Sequence[int]) -> tuple[int, ...]:
		"""
		Pack a sequence of word bit codes into u64 tuple.

		Args:
			word_bits: Sequence of word bit codes (each should be < 2^bits_per_word)

		Returns:
			Tuple of packed u64 values matching Rust's encode_context format.
		"""
		if not word_bits:
			return ()

		total_bits = len(word_bits) * self.bits_per_word
		num_u64s = (total_bits + 63) // 64
		packed = [0] * num_u64s

		for word_idx, bits in enumerate(word_bits):
			bit_offset = word_idx * self.bits_per_word
			# Pack bits_per_word bits starting at bit_offset
			for bit in range(self.bits_per_word):
				if (bits >> bit) & 1:
					global_bit = bit_offset + bit
					word_pos = global_bit // 64
					bit_pos = global_bit % 64
					if word_pos < num_u64s:
						packed[word_pos] |= (1 << bit_pos)

		return tuple(packed)

	def unpack(self, packed: Sequence[int], n_words: int) -> tuple[int, ...]:
		"""
		Unpack u64 tuple back to word bit codes.

		Args:
			packed: Tuple of packed u64 values
			n_words: Number of words to extract

		Returns:
			Tuple of word bit codes.
		"""
		if not packed or n_words <= 0:
			return ()

		word_bits = []
		for word_idx in range(n_words):
			bits = 0
			bit_offset = word_idx * self.bits_per_word
			for bit in range(self.bits_per_word):
				global_bit = bit_offset + bit
				word_pos = global_bit // 64
				bit_pos = global_bit % 64
				if word_pos < len(packed) and (packed[word_pos] >> bit_pos) & 1:
					bits |= (1 << bit)
			word_bits.append(bits)

		return tuple(word_bits)


# Default packer instance (12 bits per word, matching Rust)
DEFAULT_PACKER = ContextBitPacker(bits_per_word=12)


def pack_context_bits(word_bits: Sequence[int], bits_per_word: int = 12) -> tuple[int, ...]:
	"""
	Convenience function to pack word bits into u64 tuple.

	Args:
		word_bits: Sequence of word bit codes
		bits_per_word: Bits per word (default 12)

	Returns:
		Tuple of packed u64 values.
	"""
	if bits_per_word == 12:
		return DEFAULT_PACKER.pack(word_bits)
	return ContextBitPacker(bits_per_word).pack(word_bits)


def unpack_context_bits(packed: Sequence[int], n_words: int, bits_per_word: int = 12) -> tuple[int, ...]:
	"""
	Convenience function to unpack u64 tuple back to word bits.

	Args:
		packed: Tuple of packed u64 values
		n_words: Number of words to extract
		bits_per_word: Bits per word (default 12)

	Returns:
		Tuple of word bit codes.
	"""
	if bits_per_word == 12:
		return DEFAULT_PACKER.unpack(packed, n_words)
	return ContextBitPacker(bits_per_word).unpack(packed, n_words)
