from wnn.ram.RAMSequence import RAMSequence
from wnn.ram.encoders_decoders import OutputMode
from wnn.ram.cost import CostCalculatorFactory
from wnn.ram.cost import CostCalculatorType

from torch import Tensor
from torch import tensor
from torch.nn import Module
from torch.nn import ModuleList

class RAMMultiHeadSequence(Module):
	"""
	Multi-head sequence-to-sequence model using parallel RAMSequence heads.

	Similar to multi-head attention in Transformers, but with discrete RAM neurons:
	- Multiple heads process inputs in parallel
	- Each head can specialize in different patterns
	- Outputs are combined via CostCalculator (voting/consensus)

	This enables the model to attend to different aspects of the input
	(e.g., different character ranges, linguistic features, positional patterns).
	"""

	def __init__(
		self,
		num_heads: int,
		input_bits: int,
		n_state_neurons_per_head: int,
		n_output_neurons: int,
		n_bits_per_state_neuron: int,
		n_bits_per_output_neuron: int,
		output_mode: OutputMode = OutputMode.RAW,
		use_hashing: bool = False,
		hash_size: int = 1024,
		routing_mode: str = "vote",  # "vote", "partition", or "learn"
		cost_calculator_type: CostCalculatorType = CostCalculatorType.VOTE,
		rng: int | None = None,
	):
		"""
		Args:
			num_heads: Number of parallel heads
			input_bits: Input size (e.g., 5 for A-Z tokens)
			n_state_neurons_per_head: State neurons per head
			n_output_neurons: Output neurons (shared across heads or per-head)
			n_bits_per_state_neuron: Connections per state neuron
			n_bits_per_output_neuron: Connections per output neuron
			output_mode: Output decoder mode
			use_hashing: Whether to use hash-based addressing
			hash_size: Hash table size if using hashing
			routing_mode: How to combine head outputs
				- "vote": All heads vote on output (ensemble)
				- "partition": Route to specific head based on input
				- "learn": Learn routing with RAM (future)
			cost_calculator_type: Cost calculator for combining outputs
			rng: Random seed
		"""
		super().__init__()

		self.num_heads = num_heads
		self.input_bits = input_bits
		self.routing_mode = routing_mode

		# Create cost calculator for combining head outputs
		self.cost_calculator = CostCalculatorFactory.create(cost_calculator_type)

		# Create multiple heads
		self.heads = ModuleList([
			RAMSequence(
				input_bits=input_bits,
				n_state_neurons=n_state_neurons_per_head,
				n_output_neurons=n_output_neurons,
				n_bits_per_state_neuron=n_bits_per_state_neuron,
				n_bits_per_output_neuron=n_bits_per_output_neuron,
				output_mode=output_mode,
				use_hashing=use_hashing,
				hash_size=hash_size,
				rng=rng + i if rng is not None else None,
			)
			for i in range(num_heads)
		])

		# Store decoder from first head (all heads share same decoder type)
		self.decoder = self.heads[0].decoder

	def _route_partition(self, char: str) -> int:
		"""Partition routing: map character to specific head."""
		# A-Z split into num_heads ranges
		char_idx = ord(char.upper()) - ord('A')
		if char_idx < 0 or char_idx >= 26:
			return 0  # Default to first head for non-alphabet
		head_idx = (char_idx * self.num_heads) // 26
		return min(head_idx, self.num_heads - 1)

	def _combine_outputs(self, head_outputs: list[str]) -> str:
		"""
		Combine head outputs using CostCalculator.

		Counts votes for each character and uses cost calculator
		to select the winner (highest votes).
		"""
		if not head_outputs:
			return '?'

		# Build vocabulary of unique outputs
		unique_chars = list(set(head_outputs))
		if len(unique_chars) == 1:
			return unique_chars[0]

		# Count votes for each character
		votes = tensor([head_outputs.count(char) for char in unique_chars])

		# Use cost calculator to select winner (argmax for VOTE)
		winner_idx = self.cost_calculator.calculate_index(votes)

		return unique_chars[winner_idx]

	def train(self, windows: list[Tensor], targets: str | list[str]) -> None:
		"""
		Train heads based on routing mode.

		Args:
			windows: List of input tensors, one per timestep [1, input_bits]
			targets: Either a string (one char per timestep) or list of strings
		"""
		if self.routing_mode == "vote":
			# Train all heads on all data (ensemble)
			for head in self.heads:
				head.train(windows, targets)

		elif self.routing_mode == "partition":
			# Route to specific head based on first character
			# Convert targets to list
			if isinstance(targets, str):
				target_list = list(targets)
			else:
				target_list = targets

			# Determine which head based on first target character
			first_target = target_list[0] if target_list else 'A'
			head_idx = self._route_partition(first_target)

			# Train only the selected head
			self.heads[head_idx].train(windows, targets)

		else:
			raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

	def forward(self, input_bits: Tensor) -> str:
		"""
		Forward pass through heads based on routing mode.

		Args:
			input_bits: Input tensor [batch_size, total_bits] or [total_bits]

		Returns:
			Predicted character string
		"""
		if self.routing_mode == "vote":
			# Get predictions from all heads
			head_outputs = [head.forward(input_bits) for head in self.heads]
			# Combine using cost calculator
			return self._combine_outputs(head_outputs)

		elif self.routing_mode == "partition":
			# For partition mode during inference, use voting across heads
			# (we don't know which partition to route to a priori)
			head_outputs = [head.forward(input_bits) for head in self.heads]
			return self._combine_outputs(head_outputs)

		else:
			raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

	def __repr__(self):
		return (
			f"RAMMultiHeadSequence("
			f"num_heads={self.num_heads}, "
			f"routing_mode={self.routing_mode}, "
			f"input_bits={self.input_bits})"
		)
