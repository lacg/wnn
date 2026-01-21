"""
Harmonic Rank fitness calculator.

Ranks genomes by harmonic mean of their CE rank and accuracy rank.
This balances both objectives without requiring weight tuning.
"""

from typing import TypeVar

from .FitnessCalculator import FitnessCalculator

G = TypeVar('G')


class FitnessCalculatorHarmonicRank(FitnessCalculator[G]):
	"""
	Fitness calculator using weighted harmonic mean of ranks.

	fitness = (w_ce + w_acc) / (w_ce/rank_ce + w_acc/rank_acc)

	Where:
	- rank_ce: Position when sorted by CE ascending (1 = lowest CE = best)
	- rank_acc: Position when sorted by accuracy descending (1 = highest acc = best)
	- w_ce, w_acc: Weights for each metric (higher = more important)

	Properties:
	- Lower harmonic mean = better (closer to rank 1 in both metrics)
	- Penalizes extreme imbalances (good in one, bad in other)
	- Weights allow tuning the CE vs accuracy trade-off
	- Default weights (1, 1) = standard harmonic mean

	Example with w_ce=1.2, w_acc=1.0 (CE 20% more important):
		| Genome | CE    | Acc   | Rank_CE | Rank_Acc | WHM  |
		|--------|-------|-------|---------|----------|------|
		| A      | 10.32 | 0.02% | 1       | 4        | 1.57 | <- Now wins
		| B      | 10.35 | 0.04% | 2       | 1        | 1.47 |
	"""

	def __init__(self, weight_ce: float = 1.0, weight_acc: float = 1.0):
		"""
		Initialize with optional weights.

		Args:
			weight_ce: Weight for CE rank (default 1.0)
			weight_acc: Weight for accuracy rank (default 1.0)
		"""
		self.weight_ce = weight_ce
		self.weight_acc = weight_acc

	def fitness(self, population: list[tuple[G, float, float]]) -> list[float]:
		"""
		Compute fitness as harmonic mean of CE and accuracy ranks.

		Args:
			population: List of (genome, ce, accuracy) tuples

		Returns:
			List of harmonic mean values (lower = better)
		"""
		n = len(population)
		if n == 0:
			return []
		if n == 1:
			return [1.0]  # Single genome is rank 1

		# Extract metrics
		ce_values = [ce for _, ce, _ in population]
		acc_values = [acc for _, _, acc in population]

		# Compute CE ranks (lower CE = better = rank 1)
		ce_sorted_indices = sorted(range(n), key=lambda i: ce_values[i])
		rank_ce = [0] * n
		for rank, idx in enumerate(ce_sorted_indices, start=1):
			rank_ce[idx] = rank

		# Compute accuracy ranks (higher acc = better = rank 1)
		acc_sorted_indices = sorted(range(n), key=lambda i: -acc_values[i])
		rank_acc = [0] * n
		for rank, idx in enumerate(acc_sorted_indices, start=1):
			rank_acc[idx] = rank

		# Compute weighted harmonic mean of ranks
		# WHM = (w_ce + w_acc) / (w_ce/r_ce + w_acc/r_acc)
		#     = (w_ce + w_acc) * r_ce * r_acc / (w_ce * r_acc + w_acc * r_ce)
		w_ce = self.weight_ce
		w_acc = self.weight_acc
		w_sum = w_ce + w_acc

		fitness_scores = []
		for i in range(n):
			r_ce = rank_ce[i]
			r_acc = rank_acc[i]
			whm = w_sum * r_ce * r_acc / (w_ce * r_acc + w_acc * r_ce)
			fitness_scores.append(whm)

		return fitness_scores

	@property
	def name(self) -> str:
		if self.weight_ce == 1.0 and self.weight_acc == 1.0:
			return "HarmonicRank"
		return f"HarmonicRank(ce={self.weight_ce}, acc={self.weight_acc})"
