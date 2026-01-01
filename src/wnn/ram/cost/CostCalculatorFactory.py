from wnn.ram.cost.CostCalculator import CostCalculator
from wnn.ram.cost.CostCalculatorArgMin import CostCalculatorArgMin
from wnn.ram.cost.CostCalculatorStochastic import CostCalculatorStochastic
from wnn.ram.enums import CostCalculatorType
from wnn.ram.cost.CostCalculatorVote import CostCalculatorVote

class CostCalculatorFactory:

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
	) -> CostCalculator:
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
				raise ValueError(f"Unsupported OutputMode: {mode}")