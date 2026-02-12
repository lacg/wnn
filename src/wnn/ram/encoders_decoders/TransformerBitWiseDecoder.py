from wnn.ram.encoders_decoders.TransformerDecoder import TransformerDecoder

from torch import Tensor
from torch import uint8

class TransformerBitWiseDecoder(TransformerDecoder):

	def encode(self, target: Tensor) -> Tensor:
		return target.unsqueeze(0).to(uint8) if target.ndim == 1 else target.to(uint8)

	def decode(self, output_bits: Tensor) -> Tensor:
		return output_bits.unsqueeze(0) if output_bits.ndim == 1 else output_bits