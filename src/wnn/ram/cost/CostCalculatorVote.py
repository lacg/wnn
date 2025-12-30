from wnn.ram.cost.CostCalculator import CostCalculator

from torch import Tensor

class CostCalculatorVote(CostCalculator):
	"""
	Vote-based cost calculator for multi-head consensus.

	Treats the cost tensor as vote counts and returns the index
	with the maximum votes (most popular choice).
	"""

	def __init__(self) -> None:
		super().__init__()

	def _calculate_index(self, total_cost: Tensor) -> Tensor:
		"""
		Calculate the index with maximum votes.

		Args:
			total_cost: Tensor of vote counts [num_options]

		Returns:
			Index of the option with most votes (argmax)
		"""
		return total_cost.argmax()
