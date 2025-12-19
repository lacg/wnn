from wnn.ram.cost.CostCalculator import CostCalculator

from torch import multinomial
from torch import softmax
from torch import Tensor

class CostCalculatorStochastic(CostCalculator):

	def __init__(self, epochs: int, num_neurons: int) -> None:
		super().__init__()
		self.epoch = 0
		self.maximum_epochs = epochs * num_neurons

	def _calculate_index(self, total_cost: Tensor) -> Tensor:
		"""
		Calculate the index based on the total cost.
		"""
		temperature = max(0.1, 1.0 - self.epoch / self.maximum_epochs)
		self.epoch += 1
		probability = softmax(-total_cost / temperature, dim=0)
		return multinomial(probability, 1)

