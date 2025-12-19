from wnn.ram.decoders.DecoderEnums import OutputMode
from wnn.ram.decoders.TransformerBitWiseDecoder import TransformerBitWiseDecoder
from wnn.ram.decoders.TransformerDecoder import TransformerDecoder
from wnn.ram.decoders.TransformerHammingDecoder import TransformerHammingDecoder

class TransformerDecoderFactory:

	@staticmethod
	def create(mode: OutputMode, n_output_neurons: int) -> TransformerDecoder:
		match mode:
			case OutputMode.BITWISE:
				return TransformerBitWiseDecoder()
			case OutputMode.HAMMING:
				return TransformerHammingDecoder(n_output_neurons)
			case _:
				raise ValueError(f"Unsupported OutputMode: {mode}")