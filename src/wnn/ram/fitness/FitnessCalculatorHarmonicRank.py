"""
Harmonic Rank fitness calculator.

Ranks genomes by weighted harmonic mean of their CE, accuracy, and
optionally F1-macro and FPR ranks. Supports tunable knobs for IDS use
cases (FPR vs recall tradeoff via independent metric weights).

Population tuples: (genome, ce, accuracy[, f1_macro[, fpr]])
"""

from typing import TypeVar

from .FitnessCalculator import FitnessCalculator

G = TypeVar('G')


class FitnessCalculatorHarmonicRank(FitnessCalculator[G]):
	"""
	Fitness calculator using weighted harmonic mean of ranks.

	WHM = sum(weights) / sum(weight_i / rank_i)

	Up to 4 metrics: CE, accuracy, F1-macro, FPR.
	Metrics with weight=0 or missing data are excluded.

	Population tuples: (genome, ce, accuracy[, f1_macro[, fpr]])

	Knobs for IDS:
	- weight_f1=0, weight_fpr=0 (default): CE+accuracy only (LM behavior)
	- weight_f1=0.3, weight_fpr=0.4: IDS-balanced (F1 + low false alarms)
	- weight_fpr=1.0: FPR-dominant (minimize false positives at all costs)
	"""

	def __init__(
		self,
		weight_ce: float = 1.0,
		weight_acc: float = 1.0,
		weight_f1: float = 0.0,
		weight_fpr: float = 0.0,
	):
		self.weight_ce = weight_ce
		self.weight_acc = weight_acc
		self.weight_f1 = weight_f1
		self.weight_fpr = weight_fpr

	def _compute_ranks(self, values: list[float], ascending: bool = True) -> list[int]:
		"""Compute ranks (1 = best). ascending=True means lower value = rank 1."""
		n = len(values)
		order = sorted(range(n), key=lambda i: values[i] if ascending else -values[i])
		ranks = [0] * n
		for rank, idx in enumerate(order, start=1):
			ranks[idx] = rank
		return ranks

	def fitness(self, population: list[tuple]) -> list[float]:
		"""
		Compute fitness as weighted harmonic mean of metric ranks.

		Args:
			population: List of (genome, ce, accuracy[, f1[, fpr]]) tuples

		Returns:
			List of harmonic mean values (lower = better)
		"""
		n = len(population)
		if n == 0:
			return []
		if n == 1:
			return [1.0]

		tuple_len = len(population[0])

		# Build list of (ranks, weight) for active metrics
		active = []

		if self.weight_ce > 0:
			ranks = self._compute_ranks([t[1] for t in population], ascending=True)
			active.append((ranks, self.weight_ce))

		if self.weight_acc > 0:
			ranks = self._compute_ranks([t[2] for t in population], ascending=False)
			active.append((ranks, self.weight_acc))

		if self.weight_f1 > 0 and tuple_len > 3:
			f1_vals = [t[3] if t[3] is not None else 0.0 for t in population]
			ranks = self._compute_ranks(f1_vals, ascending=False)
			active.append((ranks, self.weight_f1))

		if self.weight_fpr > 0 and tuple_len > 4:
			fpr_vals = [t[4] if t[4] is not None else 1.0 for t in population]
			ranks = self._compute_ranks(fpr_vals, ascending=True)  # lower FPR = better
			active.append((ranks, self.weight_fpr))

		if not active:
			return [1.0] * n

		w_sum = sum(w for _, w in active)

		fitness_scores = []
		for i in range(n):
			inv_sum = sum(w / ranks[i] for ranks, w in active)
			whm = w_sum / inv_sum
			fitness_scores.append(whm)

		return fitness_scores

	@property
	def name(self) -> str:
		parts = []
		if self.weight_ce != 1.0 or self.weight_fpr > 0:
			parts.append(f"ce={self.weight_ce}")
		if self.weight_acc != 1.0 or self.weight_fpr > 0:
			parts.append(f"acc={self.weight_acc}")
		if self.weight_f1 > 0:
			parts.append(f"f1={self.weight_f1}")
		if self.weight_fpr > 0:
			parts.append(f"fpr={self.weight_fpr}")
		if not parts:
			return "HarmonicRank"
		return f"HarmonicRank({', '.join(parts)})"
