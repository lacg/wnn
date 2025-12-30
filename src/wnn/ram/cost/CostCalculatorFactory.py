from wnn.ram.cost.CostCalculator import CostCalculator
from wnn.ram.cost.CostCalculatorArgMin import CostCalculatorArgMin
from wnn.ram.cost.CostCalculatorStochastic import CostCalculatorStochastic
from wnn.ram.cost.CostCalculatorType import CostCalculatorType
from wnn.ram.cost.CostCalculatorVote import CostCalculatorVote

class CostCalculatorFactory:

	@staticmethod
	def create(mode: CostCalculatorType, epochs: int = 0, num_neurons: int = 0) -> CostCalculator:
		match mode:
			case CostCalculatorType.STOCHASTIC:
				return CostCalculatorStochastic(epochs, num_neurons)
			case CostCalculatorType.ARGMIN:
				return CostCalculatorArgMin()
			case CostCalculatorType.VOTE:
				return CostCalculatorVote()
			case _:
				raise ValueError(f"Unsupported OutputMode: {mode}")