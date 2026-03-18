"""
IDS Security fitness calculator.

Fitness = 1 - F1 × (1 - FPR)^2
Strongly penalizes high false positive rates.
"""

from wnn.ram.metrics import Metrics
from .FitnessCalculator import FitnessCalculator


class FitnessCalculatorIDSSecurity(FitnessCalculator):
	"""
	Fitness = 1 - F1 × (1 - FPR)^2

	Lower fitness = better. Falls back to CE if F1/FPR not available.
	"""

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		n = len(metrics_list)
		if n == 0:
			return []
		if n == 1:
			return [0.0]

		has_ids = any(m.f1 is not None for m in metrics_list)

		if has_ids:
			return [
				1.0 - (m.f1 or 0.0) * (1.0 - (m.fpr or 1.0)) ** 2
				for m in metrics_list
			]

		return [m.ce for m in metrics_list]

	@property
	def name(self) -> str:
		return "IDSSecurity(F1×(1-FPR)²)"
