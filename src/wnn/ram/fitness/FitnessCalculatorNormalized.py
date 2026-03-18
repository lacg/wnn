"""
Normalized fitness calculator.

Normalizes CE and accuracy to [0, 1] before combining with weights.
"""

from wnn.ram.metrics import Metrics
from .FitnessCalculator import FitnessCalculator


class FitnessCalculatorNormalized(FitnessCalculator):
	"""
	Fitness = weighted sum of normalized CE and accuracy.

	Both metrics normalized to [0, 1] where 0 = best, 1 = worst.
	Lower fitness = better.
	"""

	def __init__(self, weight_ce: float = 1.0, weight_acc: float = 1.0):
		self.weight_ce = weight_ce
		self.weight_acc = weight_acc

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		n = len(metrics_list)
		if n == 0:
			return []
		if n == 1:
			return [0.0]

		ce_values = [m.ce for m in metrics_list]
		acc_values = [m.acc for m in metrics_list]

		min_ce, max_ce = min(ce_values), max(ce_values)
		ce_range = max_ce - min_ce
		norm_ce = [(ce - min_ce) / ce_range for ce in ce_values] if ce_range > 0 else [0.0] * n

		min_acc, max_acc = min(acc_values), max(acc_values)
		acc_range = max_acc - min_acc
		norm_acc = [(max_acc - acc) / acc_range for acc in acc_values] if acc_range > 0 else [0.0] * n

		w_total = self.weight_ce + self.weight_acc
		return [
			(self.weight_ce * norm_ce[i] + self.weight_acc * norm_acc[i]) / w_total
			for i in range(n)
		]

	@property
	def name(self) -> str:
		if self.weight_ce == 1.0 and self.weight_acc == 1.0:
			return "Normalized"
		return f"Normalized(ce={self.weight_ce}, acc={self.weight_acc})"
