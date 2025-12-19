from abc import ABC, abstractmethod

from torch import Tensor

class CostCalculator(ABC):

	def __init__(self) -> None:
		super().__init__()

	@abstractmethod
	def _calculate_index(self, total_cost: Tensor) -> Tensor:
		"""
		Calculate the index based on the total cost.
		"""
		pass


	def calculate_index(self, total_cost: Tensor) -> int:
		"""
		Calculate the index based on the total cost.
		"""
		index = self._calculate_index(total_cost)
		return index.item() if index.numel() == 1 else int(index)


