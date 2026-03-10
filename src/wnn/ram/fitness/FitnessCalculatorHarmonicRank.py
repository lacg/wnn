"""
Harmonic Rank fitness calculator.

Ranks genomes by weighted harmonic mean of their CE, accuracy, and
optionally F1-macro ranks. Supports tunable knobs for IDS use cases
(FPR vs recall tradeoff via F1 weight).
"""

from typing import TypeVar

from .FitnessCalculator import FitnessCalculator

G = TypeVar('G')


class FitnessCalculatorHarmonicRank(FitnessCalculator[G]):
	"""
	Fitness calculator using weighted harmonic mean of ranks.

	With 2 metrics (default):
		WHM = (w_ce + w_acc) / (w_ce/rank_ce + w_acc/rank_acc)

	With 3 metrics (when weight_f1 > 0):
		WHM = (w_ce + w_acc + w_f1) / (w_ce/rank_ce + w_acc/rank_acc + w_f1/rank_f1)

	F1-macro rank is computed from the 4th element of population tuples
	(genome, ce, accuracy, f1_macro). If tuples have only 3 elements,
	F1 weight is ignored gracefully.

	Knobs for IDS:
	- weight_f1=0.0 (default): CE+accuracy only (LM behavior)
	- weight_f1=0.5: F1 as tiebreaker (mild F1 preference)
	- weight_f1=1.0: F1 equal to accuracy (balanced classification)
	- weight_f1=2.0: F1-dominant (prioritize per-class balance)
	"""

	def __init__(
		self,
		weight_ce: float = 1.0,
		weight_acc: float = 1.0,
		weight_f1: float = 0.0,
	):
		self.weight_ce = weight_ce
		self.weight_acc = weight_acc
		self.weight_f1 = weight_f1

	def fitness(self, population: list[tuple]) -> list[float]:
		"""
		Compute fitness as harmonic mean of CE, accuracy, and optionally F1 ranks.

		Args:
			population: List of (genome, ce, accuracy) or (genome, ce, accuracy, f1) tuples

		Returns:
			List of harmonic mean values (lower = better)
		"""
		n = len(population)
		if n == 0:
			return []
		if n == 1:
			return [1.0]

		# Extract metrics
		ce_values = [t[1] for t in population]
		acc_values = [t[2] for t in population]

		# Check if F1 is available and requested
		use_f1 = self.weight_f1 > 0 and len(population[0]) > 3
		f1_values = [t[3] for t in population] if use_f1 else None

		# Compute CE ranks (lower CE = better = rank 1)
		ce_sorted = sorted(range(n), key=lambda i: ce_values[i])
		rank_ce = [0] * n
		for rank, idx in enumerate(ce_sorted, start=1):
			rank_ce[idx] = rank

		# Compute accuracy ranks (higher acc = better = rank 1)
		acc_sorted = sorted(range(n), key=lambda i: -acc_values[i])
		rank_acc = [0] * n
		for rank, idx in enumerate(acc_sorted, start=1):
			rank_acc[idx] = rank

		# Compute F1 ranks (higher F1 = better = rank 1)
		rank_f1 = None
		if use_f1:
			f1_sorted = sorted(range(n), key=lambda i: -f1_values[i])
			rank_f1 = [0] * n
			for rank, idx in enumerate(f1_sorted, start=1):
				rank_f1[idx] = rank

		# Compute weighted harmonic mean
		w_ce = self.weight_ce
		w_acc = self.weight_acc
		w_f1 = self.weight_f1 if use_f1 else 0.0
		w_sum = w_ce + w_acc + w_f1

		fitness_scores = []
		for i in range(n):
			inv_sum = w_ce / rank_ce[i] + w_acc / rank_acc[i]
			if use_f1:
				inv_sum += w_f1 / rank_f1[i]
			whm = w_sum / inv_sum
			fitness_scores.append(whm)

		return fitness_scores

	@property
	def name(self) -> str:
		parts = []
		if self.weight_ce != 1.0:
			parts.append(f"ce={self.weight_ce}")
		if self.weight_acc != 1.0:
			parts.append(f"acc={self.weight_acc}")
		if self.weight_f1 > 0:
			parts.append(f"f1={self.weight_f1}")
		if not parts:
			return "HarmonicRank"
		return f"HarmonicRank({', '.join(parts)})"
