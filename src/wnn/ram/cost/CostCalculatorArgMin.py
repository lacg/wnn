from wnn.ram.cost.CostCalculator import CostCalculator

from torch import Tensor

class CostCalculatorArgMin(CostCalculator):

	def __init__(self) -> None:
		super().__init__()

	def _calculate_index(self, total_cost: Tensor) -> Tensor:
		"""
		Calculate the index based on the total cost.
		"""
		return total_cost.argmin()

