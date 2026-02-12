"""
Two-Layer Feed-Forward Network

MLP-style FFN with hidden expansion.
"""

from torch import Tensor
from torch.nn import Module

from wnn.ram.core.RAMGeneralization import GeneralizingProjection, MapperStrategy


class TwoLayerFFN(Module):
	"""Two-layer feed-forward network with hidden expansion."""

	def __init__(
		self,
		input_bits: int,
		hidden_bits: int,
		output_bits: int,
		rng: int | None = None,
	):
		super().__init__()

		self.input_bits = input_bits
		self.hidden_bits = hidden_bits
		self.output_bits = output_bits

		# Up projection: input -> hidden
		self.up_proj = GeneralizingProjection(
			input_bits=input_bits,
			output_bits=hidden_bits,
			strategy=MapperStrategy.BIT_LEVEL,
			rng=rng,
		)

		# Down projection: hidden -> output
		self.down_proj = GeneralizingProjection(
			input_bits=hidden_bits,
			output_bits=output_bits,
			strategy=MapperStrategy.BIT_LEVEL,
			rng=rng + 500 if rng else None,
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Forward pass: up -> activation (none for binary) -> down."""
		x = x.squeeze()
		hidden = self.up_proj(x)
		output = self.down_proj(hidden)
		return output

	def train_mapping(self, inp: Tensor, target: Tensor) -> int:
		"""Train both projections."""
		inp = inp.squeeze()
		target = target.squeeze()

		# Forward to get hidden
		hidden = self.up_proj(inp)

		# Train down_proj
		return self.down_proj.train_mapping(hidden, target)
