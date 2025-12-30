from wnn.ram.encoders_decoders.DecoderEnums import OutputMode
from wnn.ram.encoders_decoders.TransformerBitWiseDecoder import TransformerBitWiseDecoder
from wnn.ram.encoders_decoders.TransformerDecoder import TransformerDecoder
from wnn.ram.encoders_decoders.TransformerHammingDecoder import TransformerHammingDecoder
from wnn.ram.encoders_decoders.TransformerRawDecoder import TransformerRawDecoder
from wnn.ram.encoders_decoders.TransformerTokenDecoder import TransformerTokenDecoder
from wnn.ram.encoders_decoders.TransformerTokenListDecoder import TransformerTokenListDecoder

class TransformerDecoderFactory:

	@staticmethod
	def create(mode: OutputMode, n_output_neurons: int = 0):
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