"""
Cost Calculator Factory

Factory for creating cost calculators based on CostCalculatorType.
"""

from wnn.ram.enums import CostCalculatorType


class CostCalculatorFactory:
	"""Factory for creating cost calculators based on type."""

	@staticmethod
	def create(
		mode: CostCalculatorType,
		epochs: int = 0,
		num_neurons: int = 0,
		# Parameters for CostCalculatorRAM
		input_bits: int = 0,
		num_options: int = 0,
		n_bits_per_neuron: int = 8,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: int | None = None,
	):
		"""
		Create a cost calculator instance.

		Args:
			mode: Type of cost calculator
			epochs: Number of epochs (for STOCHASTIC)
			num_neurons: Number of neurons (for STOCHASTIC)
			input_bits: Input bits (for RAM)
			num_options: Number of options (for RAM)
			n_bits_per_neuron: Bits per neuron (for RAM)
			use_hashing: Use hashing (for RAM)
			hash_size: Hash size (for RAM)
			rng: Random seed (for RAM)

		Returns:
			CostCalculator instance
		"""
		# Lazy imports to avoid circular dependencies
		from wnn.ram.cost.CostCalculatorArgMin import CostCalculatorArgMin
		from wnn.ram.cost.CostCalculatorStochastic import CostCalculatorStochastic
		from wnn.ram.cost.CostCalculatorVote import CostCalculatorVote

		match mode:
			case CostCalculatorType.STOCHASTIC:
				return CostCalculatorStochastic(epochs, num_neurons)
			case CostCalculatorType.ARGMIN:
				return CostCalculatorArgMin()
			case CostCalculatorType.VOTE:
				return CostCalculatorVote()
			case CostCalculatorType.RAM:
				# Lazy import to avoid circular dependency
				from wnn.ram.cost.CostCalculatorRAM import CostCalculatorRAM
				return CostCalculatorRAM(
					input_bits=input_bits,
					num_options=num_options,
					n_bits_per_neuron=n_bits_per_neuron,
					use_hashing=use_hashing,
					hash_size=hash_size,
					rng=rng,
				)
			case _:
				raise ValueError(f"Unsupported CostCalculatorType: {mode}")
