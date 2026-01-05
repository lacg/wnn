"""
Transformer Decoder Factory

Factory for creating decoder instances based on OutputMode.
"""

from wnn.ram.encoders_decoders import OutputMode


class TransformerDecoderFactory:
	"""Factory for creating transformer decoders based on output mode."""

	@staticmethod
	def create(mode: OutputMode, n_output_neurons: int = 0):
		"""
		Create a decoder instance based on the output mode.

		Args:
			mode: The output mode (BITWISE, HAMMING, RAW, TOKEN, TOKEN_LIST)
			n_output_neurons: Number of output neurons (for HAMMING mode)

		Returns:
			Appropriate TransformerDecoder instance
		"""
		# Lazy imports to avoid circular dependencies
		from wnn.ram.encoders_decoders.TransformerBitWiseDecoder import TransformerBitWiseDecoder
		from wnn.ram.encoders_decoders.TransformerHammingDecoder import TransformerHammingDecoder
		from wnn.ram.encoders_decoders.TransformerRawDecoder import TransformerRawDecoder
		from wnn.ram.encoders_decoders.TransformerTokenDecoder import TransformerTokenDecoder
		from wnn.ram.encoders_decoders.TransformerTokenListDecoder import TransformerTokenListDecoder

		match mode:
			case OutputMode.BITWISE:
				return TransformerBitWiseDecoder()
			case OutputMode.HAMMING:
				return TransformerHammingDecoder(n_output_neurons)
			case OutputMode.RAW:
				return TransformerRawDecoder()
			case OutputMode.TOKEN:
				return TransformerTokenDecoder()
			case OutputMode.TOKEN_LIST:
				return TransformerTokenListDecoder()
			case _:
				raise ValueError(f"Unsupported OutputMode: {mode}")
