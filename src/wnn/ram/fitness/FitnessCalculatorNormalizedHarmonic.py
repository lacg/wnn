"""
Normalized Harmonic fitness calculator.

Normalizes both CE and accuracy to [0, 1] scale, then combines using weighted harmonic mean.
This penalizes imbalance more strongly than arithmetic mean - a genome must be good at BOTH
metrics to get a good score.
"""

from typing import TypeVar

from .FitnessCalculator import FitnessCalculator

G = TypeVar('G')


class FitnessCalculatorNormalizedHarmonic(FitnessCalculator[G]):
	"""
	Fitness calculator using normalized values with weighted harmonic mean.

	Both CE and accuracy are normalized to [0, 1] (where 0=best, 1=worst),
	then combined using weighted harmonic mean:

		fitness = (w_ce + w_acc) / (w_ce / norm_ce + w_acc / norm_acc)

	The harmonic mean penalizes imbalance more strongly than arithmetic mean:
	- A genome that's best in CE (0.0) but worst in accuracy (1.0) gets ~0.0 (best!)
	  with arithmetic mean, but with harmonic mean it gets a worse score.

	To handle the 0 case (best in one metric), we add a small epsilon.

	Example with w_ce=1, w_acc=2 (accuracy 2x more important):
		| Genome | Norm_CE | Norm_Acc | Arith Mean | Harmonic Mean |
		|--------|---------|----------|------------|---------------|
		| A      | 0.0     | 1.0      | 0.67       | 0.00 (eps)    |
		| B      | 1.0     | 0.0      | 0.33       | 0.00 (eps)    |
		| C      | 0.3     | 0.3      | 0.30       | 0.30          |

	Properties:
	- Strongly penalizes being bad at either metric
	- Lower fitness = better
	- Weights control relative importance (higher weight = more important)
	"""

	def __init__(self, weight_ce: float = 1.0, weight_acc: float = 1.0):
		"""
		Initialize with weights.

		Args:
			weight_ce: Weight for CE (default 1.0)
			weight_acc: Weight for accuracy (default 1.0)
		"""
		self.weight_ce = weight_ce
		self.weight_acc = weight_acc

	def fitness(self, population: list[tuple[G, float, float]]) -> list[float]:
		"""
		Compute fitness as weighted harmonic mean of normalized CE and accuracy.

		Args:
			population: List of (genome, ce, accuracy) tuples

		Returns:
			List of fitness values (lower = better)
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
			norm_ce = [(ce - min_ce) / ce_range for ce in ce_values]
		else:
			norm_ce = [0.0] * n  # All same CE

		# Normalize accuracy to [0, 1] where 0 = best (highest acc), 1 = worst (lowest acc)
		min_acc = min(acc_values)
		max_acc = max(acc_values)
		acc_range = max_acc - min_acc
		if acc_range > 0:
			norm_acc = [(max_acc - acc) / acc_range for acc in acc_values]
		else:
			norm_acc = [0.0] * n  # All same accuracy

		# Compute weighted harmonic mean
		# HM = (w1 + w2) / (w1/x1 + w2/x2)
		# Add small epsilon to avoid division by zero
		eps = 1e-6
		w_total = self.weight_ce + self.weight_acc

		fitness_scores = []
		for i in range(n):
			# Add epsilon to prevent division by zero
			nc = norm_ce[i] + eps
			na = norm_acc[i] + eps

			# Weighted harmonic mean
			hm = w_total / (self.weight_ce / nc + self.weight_acc / na)
			fitness_scores.append(hm)

		return fitness_scores

	@property
	def name(self) -> str:
		if self.weight_ce == 1.0 and self.weight_acc == 1.0:
			return "NormalizedHarmonic"
		return f"NormalizedHarmonic(ce={self.weight_ce}, acc={self.weight_acc})"
