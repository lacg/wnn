"""
Fitness calculator wrapper that enforces a minimum accuracy floor.

Genomes with accuracy below the floor get fitness = infinity.
"""

import math

from wnn.ram.metrics import Metrics
from .FitnessCalculator import FitnessCalculator


class FitnessCalculatorWithAccuracyFloor(FitnessCalculator):
	"""
	Wrapper: genomes below min_accuracy get fitness = inf.

	Prevents pathological optimization where CE improves but accuracy degrades.
	"""

	def __init__(self, base_calculator: FitnessCalculator, min_accuracy: float = 0.003):
		if min_accuracy < 0 or min_accuracy > 1:
			raise ValueError(f"min_accuracy must be in [0, 1], got {min_accuracy}")
		self._base = base_calculator
		self._min_accuracy = min_accuracy

	@property
	def min_accuracy(self) -> float:
		return self._min_accuracy

	@property
	def base_calculator(self) -> FitnessCalculator:
		return self._base

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		if not metrics_list:
			return []

		base_scores = self._base.fitness(metrics_list)
		return [
			math.inf if m.acc < self._min_accuracy else score
			for m, score in zip(metrics_list, base_scores)
		]

	@property
	def name(self) -> str:
		return f"{self._base.name}(floor={self._min_accuracy:.4f})"
