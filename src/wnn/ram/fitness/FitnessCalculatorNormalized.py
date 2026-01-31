"""
Normalized fitness calculator.

Normalizes both CE and accuracy to [0, 1] scale before combining with weights.
This allows equivalent influence from both metrics regardless of their original scales.
"""

from typing import TypeVar

from .FitnessCalculator import FitnessCalculator

G = TypeVar('G')


class FitnessCalculatorNormalized(FitnessCalculator[G]):
	"""
	Fitness calculator using normalized values on [0, 1] scale.

	Both CE and accuracy are normalized to [0, 1] before combining:
	- CE:  normalized_ce = (max_ce - ce) / (max_ce - min_ce)  → 0=worst, 1=best
	- Acc: normalized (already 0-1, but scaled to population min/max for better discrimination)

	Final fitness = w_ce * (1 - norm_ce) + w_acc * (1 - norm_acc)
	               = weighted sum where lower is better

	With default weights (1.0, 1.0), both metrics have equal influence.

	Example:
		| Genome | CE    | Acc   | Norm_CE | Norm_Acc | Fitness |
		|--------|-------|-------|---------|----------|---------|
		| A      | 10.32 | 0.02% | 1.00    | 0.33     | 0.33    | <- Best CE
		| B      | 10.35 | 0.04% | 0.00    | 1.00     | 0.50    | <- Best Acc
		| C      | 10.33 | 0.03% | 0.67    | 0.67     | 0.33    | <- Balanced

	Properties:
	- Equal scale: 1% CE improvement = same as 1% accuracy improvement (with equal weights)
	- Interpretable: fitness in [0, 1] range
	- Lower fitness = better (for minimization)
	"""

	def __init__(self, weight_ce: float = 1.0, weight_acc: float = 1.0):
		"""
		Initialize with optional weights.

		Args:
			weight_ce: Weight for normalized CE (default 1.0)
			weight_acc: Weight for normalized accuracy (default 1.0)
		"""
		self.weight_ce = weight_ce
		self.weight_acc = weight_acc

	def fitness(self, population: list[tuple[G, float, float]]) -> list[float]:
		"""
		Compute fitness as weighted sum of normalized CE and accuracy.

		Args:
			population: List of (genome, ce, accuracy) tuples

		Returns:
			List of fitness values (lower = better, range [0, 1])
		"""
		n = len(population)
		if n == 0:
			return []
		if n == 1:
			return [0.0]  # Single genome is "best"

		# Extract metrics
		ce_values = [ce for _, ce, _ in population]
		acc_values = [acc for _, _, acc in population]

		# Normalize CE to [0, 1] where 0 = best (lowest CE), 1 = worst (highest CE)
		min_ce = min(ce_values)
		max_ce = max(ce_values)
		ce_range = max_ce - min_ce
		if ce_range > 0:
			# Invert so lower CE = lower normalized value
			norm_ce = [(ce - min_ce) / ce_range for ce in ce_values]
		else:
			norm_ce = [0.0] * n  # All same CE

		# Normalize accuracy to [0, 1] where 0 = best (highest acc), 1 = worst (lowest acc)
		min_acc = min(acc_values)
		max_acc = max(acc_values)
		acc_range = max_acc - min_acc
		if acc_range > 0:
			# Invert so higher accuracy = lower normalized value
			norm_acc = [(max_acc - acc) / acc_range for acc in acc_values]
		else:
			norm_acc = [0.0] * n  # All same accuracy

		# Compute weighted sum (lower = better)
		w_total = self.weight_ce + self.weight_acc
		fitness_scores = []
		for i in range(n):
			# Weighted average of normalized values
			f = (self.weight_ce * norm_ce[i] + self.weight_acc * norm_acc[i]) / w_total
			fitness_scores.append(f)

		return fitness_scores

	@property
	def name(self) -> str:
		if self.weight_ce == 1.0 and self.weight_acc == 1.0:
			return "Normalized"
		return f"Normalized(ce={self.weight_ce}, acc={self.weight_acc})"
