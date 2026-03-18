"""
CE-only fitness calculator.

Ranks genomes purely by cross-entropy loss, ignoring all other metrics.
"""

from wnn.ram.metrics import Metrics
from .FitnessCalculator import FitnessCalculator


class FitnessCalculatorCE(FitnessCalculator):
	"""Fitness = CE (lower is better)."""

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		return [m.ce for m in metrics_list]

	@property
	def name(self) -> str:
		return "CE"
