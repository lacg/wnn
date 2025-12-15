from wnn.ram.decoders.TransformerDecoder import TransformerDecoder

from torch import bool as tbool
from torch import full
from torch import Tensor
from torch import uint8

class TransformerHammingDecoder(TransformerDecoder):

	def __init__(self, n_output_neurons: int):
		self.n_output_neurons = n_output_neurons

	def encode(self, target: Tensor) -> Tensor:
		# Expect scalar class {0,1}
		if target.ndim == 2 and target.shape[1] == 1:
			label = bool(target[0, 0].item())
		elif target.ndim == 1 and target.numel() == 1:
			label = bool(target.item())
		else:
			raise ValueError("Hamming decoder expects scalar target")

		return full((1, self.n_output_neurons), 1 if label else 0, dtype=uint8, device=target.device)

	def decode(self, output_bits: Tensor) -> Tensor:
		if output_bits.ndim == 1:
			output_bits = output_bits.unsqueeze(0)

		n = output_bits.shape[1]
		ones = output_bits.sum(dim=1)
		zeros = n - ones

		# Majority vote
		return (ones > zeros).to(tbool).unsqueeze(1)