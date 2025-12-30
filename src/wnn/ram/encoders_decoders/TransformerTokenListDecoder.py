from wnn.ram.encoders_decoders.TransformerTokenDecoder import TransformerTokenDecoder

from torch import Tensor

class TransformerTokenListDecoder:
	"""
	Encoder for token sequences (input side).
	Uses TransformerTokenDecoder internally for character encoding.
	"""

	def __init__(self, vocab: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
		"""
		Args:
			vocab: String of characters in vocabulary (default: A-Z)
		"""
		self.token_decoder = TransformerTokenDecoder(vocab)
		self.vocab = self.token_decoder.vocab
		self.vocab_size = self.token_decoder.vocab_size
		self.bits_per_token = self.token_decoder.bits_per_token

	def encode(self, sequence: str) -> list[Tensor]:
		"""
		Encode a string to a list of token tensors.

		Args:
			sequence: String of characters from vocabulary

		Returns:
			List of tensors, each [1, bits_per_token]
		"""
		if len(sequence) == 0:
			raise ValueError("Empty sequence string")
		return [self.token_decoder.encode(char) for char in sequence]

	def decode(self, token_list: list[Tensor]) -> str:
		"""
		Decode a list of token tensors to string.

		Args:
			token_list: List of tensors, each [1, bits_per_token]

		Returns:
			Decoded string
		"""
		return ''.join(self.token_decoder.decode(bits) for bits in token_list)
