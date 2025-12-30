from wnn.ram.encoders_decoders.TransformerDecoder import TransformerDecoder

from torch import Tensor
from torch import tensor
from torch import uint8

class TransformerTokenDecoder(TransformerDecoder):
	"""
	Decoder for single token outputs.
	Encodes/decodes characters (A-Z) to/from binary bits.
	"""

	def __init__(self, vocab: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
		"""
		Args:
			vocab: String of characters in vocabulary (default: A-Z)
		"""
		self.vocab = vocab
		self.vocab_size = len(vocab)
		self.bits_per_token = (self.vocab_size - 1).bit_length()

		# Create mappings
		self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
		self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}

	def _encode_char(self, char: str) -> Tensor:
		"""Encode single character to binary bits."""
		if char not in self.char_to_idx:
			raise ValueError(f"Character '{char}' not in vocabulary")

		idx = self.char_to_idx[char]

		# Convert to binary
		bits = []
		for i in range(self.bits_per_token):
			bit = (idx >> (self.bits_per_token - 1 - i)) & 1
			bits.append(bit)

		return tensor(bits, dtype=uint8)

	def _decode_bits(self, bits: Tensor) -> str:
		"""Decode binary bits to character."""
		if bits.ndim == 2:
			bits = bits[0]  # Remove batch dimension

		# Convert binary to index
		idx = 0
		for bit in bits:
			idx = (idx << 1) | int(bit)

		if idx >= self.vocab_size:
			return '?'  # Out of vocabulary

		return self.idx_to_char[idx]

	def encode(self, target: str) -> Tensor:
		"""
		Encode a single character to output layer target bits.

		Args:
			target: Single character from vocabulary

		Returns:
			Binary tensor [1, bits_per_token]
		"""
		if len(target) != 1:
			raise ValueError(f"Expected single character, got '{target}' (len={len(target)})")
		bits = self._encode_char(target)
		return bits.unsqueeze(0)

	def decode(self, output_bits: Tensor) -> str:
		"""
		Decode output layer bits to character.

		Args:
			output_bits: Output bits [1, bits_per_token] or [bits_per_token]

		Returns:
			Decoded character
		"""
		return self._decode_bits(output_bits)
