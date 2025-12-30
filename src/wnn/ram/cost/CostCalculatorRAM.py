from wnn.ram.cost.CostCalculator import CostCalculator
from wnn.ram.RAMLayer import RAMLayer

from torch import Tensor
from torch import zeros
from torch import uint8

class CostCalculatorRAM(CostCalculator):
	"""
	RAM-based cost calculator that learns to select options.

	Uses a RAM layer to compute attention scores (0-7) for each option,
	then selects the option with the highest score.

	This enables learned routing/selection where the model decides
	which option should be chosen based on the input context.
	"""

	def __init__(
		self,
		input_bits: int,
		num_options: int,
		n_bits_per_neuron: int = 8,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Size of input context
			num_options: Number of options to score (e.g., num_heads)
			n_bits_per_neuron: Connections per RAM neuron
			use_hashing: Whether to use hash-based addressing
			hash_size: Hash table size
			rng: Random seed
		"""
		super().__init__()

		self.input_bits = input_bits
		self.num_options = num_options
		self.bits_per_score = 3  # 3 bits → scores 0-7

		# Total neurons: 3 bits per option
		self.total_neurons = num_options * self.bits_per_score

		# RAM layer for scoring
		self.scorer = RAMLayer(
			total_input_bits=input_bits,
			num_neurons=self.total_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=rng,
		)

		# Store context for training
		self.last_input_bits = None

	def _decode_scores(self, outputs: Tensor) -> Tensor:
		"""
		Decode RAM outputs to scores.

		Args:
			outputs: RAM layer outputs [total_neurons]

		Returns:
			Scores [num_options] with values 0-7
		"""
		scores = zeros(self.num_options, dtype=uint8)
		for option_idx in range(self.num_options):
			# Get 3 bits for this option
			bit_start = option_idx * self.bits_per_score
			bit_end = bit_start + self.bits_per_score
			option_bits = outputs[bit_start:bit_end]

			# Convert 3 bits to score (0-7)
			score = 0
			for bit in option_bits:
				score = (score << 1) | int(bit)

			scores[option_idx] = score

		return scores

	def _calculate_index(self, total_cost: Tensor) -> Tensor:
		"""
		Calculate the index using RAM-based scoring.

		Note: total_cost is actually the input context (not costs).
		For RAM calculator, we use the input to compute scores.

		Args:
			total_cost: Input context tensor [input_bits]

		Returns:
			Index of option with highest score
		"""
		# Store for potential training
		self.last_input_bits = total_cost.unsqueeze(0) if total_cost.ndim == 1 else total_cost

		# Get RAM outputs
		outputs = self.scorer.forward(self.last_input_bits).squeeze(0)

		# Decode to scores
		scores = self._decode_scores(outputs)

		# Return argmax (highest score wins)
		return scores.argmax()

	def train_scores(self, input_bits: Tensor, target_scores: Tensor) -> None:
		"""
		Train the RAM scorer to produce target scores.

		Args:
			input_bits: Input context [1, input_bits] or [input_bits]
			target_scores: Target scores [num_options] with values 0-7
		"""
		# Normalize input
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		# Encode target scores to bits
		target_bits = zeros(self.total_neurons, dtype=uint8)

		for option_idx in range(self.num_options):
			score = int(target_scores[option_idx])
			# Convert score to 3 bits
			bit_start = option_idx * self.bits_per_score
			for i in range(self.bits_per_score):
				bit = (score >> (self.bits_per_score - 1 - i)) & 1
				target_bits[bit_start + i] = bit

		# Train RAM layer
		target_bits = target_bits.unsqueeze(0)  # [1, total_neurons]
		self.scorer.commit(input_bits, target_bits)
