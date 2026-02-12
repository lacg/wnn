"""
Discrete Normalization for RAM Networks

Since RAM networks operate on boolean values, traditional layer normalization
(which normalizes to zero mean and unit variance) isn't applicable.

This module provides discrete equivalents:
- ENSEMBLE_VOTE: Multiple sub-networks vote, majority wins
	(analogous to dropout + averaging for stability)
- BIT_BALANCE: Learn transformations toward balanced bit patterns
	(maximizes entropy, prevents saturation)

Usage:
	norm = DiscreteNormalization(input_bits=8, strategy=NormStrategy.ENSEMBLE_VOTE)
	output = norm(input_tensor)  # [8] -> [8]
"""

from torch import Tensor, zeros, uint8
from torch.nn import Module, ModuleList

from wnn.ram.core.models import NormStrategy


class DiscreteNormalization(Module):
	"""
	Discrete normalization for RAM networks.

	Provides stability through redundant sub-networks and voting.
	Trainable via RAM commit/explore operations.
	"""

	def __init__(
		self,
		input_bits: int,
		strategy: NormStrategy = NormStrategy.ENSEMBLE_VOTE,
		num_sub_networks: int = 4,
		bits_per_sub: int | None = None,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Number of input/output bits
			strategy: Normalization strategy to use
			num_sub_networks: Number of sub-networks for ENSEMBLE_VOTE
			bits_per_sub: Bits per sub-network (default: 60% of input)
			rng: Random seed for initialization
		"""
		super().__init__()

		self.input_bits = input_bits
		self.strategy = strategy
		self.num_sub_networks = num_sub_networks

		# Lazy import to avoid circular dependency
		from wnn.ram.core import RAMLayer

		match strategy:
			case NormStrategy.NONE:
				self.sub_networks = None

			case NormStrategy.ENSEMBLE_VOTE:
				# Multiple sub-networks with voting for stability
				if bits_per_sub is None:
					bits_per_sub = max(4, int(input_bits * 0.6))
				bits_per_sub = min(bits_per_sub, input_bits)

				self.sub_networks = ModuleList([
					RAMLayer(
						total_input_bits=input_bits,
						num_neurons=input_bits,
						n_bits_per_neuron=bits_per_sub,
						rng=rng + i * 100 if rng else None,
					)
					for i in range(num_sub_networks)
				])

			case NormStrategy.BIT_BALANCE:
				# Single network that learns balanced transformations
				from wnn.ram.core import BitLevelMapper, ContextMode, BitMapperMode

				self.bit_transform = BitLevelMapper(
					n_bits=input_bits,
					context_mode=ContextMode.FULL,
					output_mode=BitMapperMode.FLIP,  # Learn to flip toward balance
					rng=rng,
				)
				self.sub_networks = None

			case _:
				raise ValueError(f"Unknown strategy: {strategy}")

	def forward(self, x: Tensor) -> Tensor:
		"""
		Apply discrete normalization.

		Args:
			x: Input tensor of shape [input_bits]

		Returns:
			Normalized tensor of shape [input_bits]
		"""
		x = x.squeeze() if x.ndim > 1 else x

		match self.strategy:
			case NormStrategy.NONE:
				return x.clone()

			case NormStrategy.ENSEMBLE_VOTE:
				return self._ensemble_forward(x)

			case NormStrategy.BIT_BALANCE:
				return self._bit_balance_forward(x)

			case _:
				return x.clone()

	def _ensemble_forward(self, x: Tensor) -> Tensor:
		"""Forward pass with ensemble voting."""
		# Get outputs from all sub-networks
		outputs = [net(x.unsqueeze(0)).squeeze() for net in self.sub_networks]

		# Majority vote per bit
		result = zeros(self.input_bits, dtype=uint8)
		threshold = self.num_sub_networks / 2

		for bit in range(self.input_bits):
			ones = sum(o[bit].item() for o in outputs)
			result[bit] = 1 if ones > threshold else 0

		return result

	def _bit_balance_forward(self, x: Tensor) -> Tensor:
		"""Forward pass with bit balancing."""
		return self.bit_transform(x)

	def commit_ensemble(self, x: Tensor, target: Tensor) -> int:
		"""
		Train ensemble sub-networks on a target output.

		Args:
			x: Input tensor
			target: Desired output

		Returns:
			Total errors across all sub-networks
		"""
		if self.strategy != NormStrategy.ENSEMBLE_VOTE or self.sub_networks is None:
			return 0

		x = x.squeeze() if x.ndim > 1 else x
		target = target.squeeze() if target.ndim > 1 else target

		total_errors = 0
		for net in self.sub_networks:
			errors = net.commit(x.unsqueeze(0), target.unsqueeze(0))
			total_errors += errors

		return total_errors

	def __repr__(self):
		return (
			f"DiscreteNormalization("
			f"bits={self.input_bits}, "
			f"strategy={self.strategy.name}, "
			f"subs={self.num_sub_networks if self.sub_networks else 0})"
		)
