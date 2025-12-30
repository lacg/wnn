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
	- Outputs are combined via CostCalculator (VOTE, RAM, etc.)

	Automatically calculates optimal state neurons per head based on
	vocabulary partitioning: vocab_size / num_heads.
	"""

	def __init__(
		self,
		num_heads: int,
		input_bits: int,
		vocab_size: int = 26,  # Vocabulary size for automatic calculation
		n_state_neurons_per_head: int | None = None,  # Auto-calculated if None
		n_output_neurons: int = 5,
		n_bits_per_state_neuron: int | None = None,  # Auto-calculated if None
		n_bits_per_output_neuron: int = 5,
		output_mode: OutputMode = OutputMode.RAW,
		use_hashing: bool = False,
		hash_size: int = 1024,
		cost_calculator_type: CostCalculatorType = CostCalculatorType.VOTE,
		rng: int | None = None,
	):
		"""
		Args:
			num_heads: Number of parallel heads
			input_bits: Input size (e.g., 5 for A-Z tokens)
			vocab_size: Vocabulary size (default 26 for A-Z)
			n_state_neurons_per_head: State neurons per head (auto if None)
			n_output_neurons: Output neurons
			n_bits_per_state_neuron: Connections per state neuron (auto if None)
			n_bits_per_output_neuron: Connections per output neuron
			output_mode: Output decoder mode
			use_hashing: Whether to use hash-based addressing
			hash_size: Hash table size if using hashing
			cost_calculator_type: How to combine head outputs (VOTE, RAM, etc.)
			rng: Random seed
		"""
		super().__init__()

		self.num_heads = num_heads
		self.input_bits = input_bits
		self.vocab_size = vocab_size
		self.cost_calculator_type = cost_calculator_type

		# Auto-calculate optimal state neurons per head
		# Each head handles vocab_size / num_heads characters
		if n_state_neurons_per_head is None:
			chars_per_head = (vocab_size + num_heads - 1) // num_heads  # ceiling division
			n_state_neurons_per_head = (chars_per_head - 1).bit_length()  # bits needed
			print(f"[MultiHead] Auto: {chars_per_head} chars/head → {n_state_neurons_per_head} state neurons/head")

		self.n_state_neurons_per_head = n_state_neurons_per_head

		# Auto-calculate bits per state neuron (full connectivity)
		if n_bits_per_state_neuron is None:
			n_bits_per_state_neuron = input_bits + n_state_neurons_per_head

		# Cap output neuron connections at available input bits (state neurons)
		# Output layer sees only state bits, so can't connect to more than that
		if n_bits_per_output_neuron > n_state_neurons_per_head:
			n_bits_per_output_neuron = n_state_neurons_per_head

		# Create cost calculator for combining head outputs
		match cost_calculator_type:
			case CostCalculatorType.RAM:
				# RAM calculator needs input context
				self.cost_calculator = CostCalculatorFactory.create(
					mode=cost_calculator_type,
					input_bits=input_bits,
					num_options=num_heads,
					n_bits_per_neuron=min(n_bits_per_state_neuron, 10),  # Reasonable default
					use_hashing=use_hashing,
					hash_size=hash_size,
					rng=rng,
				)
			case _:
				# VOTE, ARGMIN, etc. don't need parameters
				self.cost_calculator = CostCalculatorFactory.create(mode=cost_calculator_type)

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

	def _combine_outputs(self, head_outputs: list[str], input_bits: Tensor | None = None) -> str:
		"""
		Combine head outputs using CostCalculator.

		Args:
			head_outputs: List of character predictions from each head
			input_bits: Optional input context for RAM calculator

		Returns:
			Combined prediction
		"""
		if not head_outputs:
			return '?'

		# Build vocabulary of unique outputs
		unique_chars = list(set(head_outputs))
		if len(unique_chars) == 1:
			return unique_chars[0]

		# Count votes for each character
		votes = tensor([head_outputs.count(char) for char in unique_chars])

		# For RAM calculator, use input_bits as context
		if self.cost_calculator_type == CostCalculatorType.RAM and input_bits is not None:
			# RAM calculator uses input as context, not votes
			winner_idx = self.cost_calculator.calculate_index(input_bits.squeeze())
			# Map to actual head output
			return head_outputs[winner_idx % len(head_outputs)]
		else:
			# Use votes as cost
			winner_idx = self.cost_calculator.calculate_index(votes)
			return unique_chars[winner_idx]

	def train(self, windows: list[Tensor], targets: str | list[str]) -> None:
		"""
		Train heads (all heads trained on all data for ensemble).

		Args:
			windows: List of input tensors, one per timestep [1, input_bits]
			targets: Either a string (one char per timestep) or list of strings
		"""
		# Train all heads on all data
		for head in self.heads:
			head.train(windows, targets)

	def forward(self, input_bits: Tensor) -> str:
		"""
		Forward pass through all heads and combine outputs.

		Args:
			input_bits: Input tensor [batch_size, total_bits] or [total_bits]

		Returns:
			Predicted character string
		"""
		# Get predictions from all heads
		head_outputs = [head.forward(input_bits) for head in self.heads]

		# Combine using cost calculator
		return self._combine_outputs(head_outputs, input_bits)

	def __repr__(self):
		return (
			f"RAMMultiHeadSequence("
			f"num_heads={self.num_heads}, "
			f"cost_calculator={self.cost_calculator_type.name}, "
			f"state_neurons/head={self.n_state_neurons_per_head})"
		)
