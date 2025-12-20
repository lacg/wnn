from wnn.ram.decoders.TransformerDecoder import TransformerDecoder

from torch import bool as tbool
from torch import Tensor


class TransformerRawDecoder(TransformerDecoder):

    def encode(self, target_bits: Tensor) -> Tensor:
        # Expect [B, n_bits]
        return target_bits.to(tbool)

    def decode(self, output_bits: Tensor) -> Tensor:
        # Identity
        return output_bits.to(tbool)