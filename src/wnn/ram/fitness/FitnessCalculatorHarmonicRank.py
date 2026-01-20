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
	Fitness calculator using harmonic mean of ranks.

	fitness = 2 / (1/rank_ce + 1/rank_acc)

	Where:
	- rank_ce: Position when sorted by CE ascending (1 = lowest CE = best)
	- rank_acc: Position when sorted by accuracy descending (1 = highest acc = best)

	Properties:
	- Lower harmonic mean = better (closer to rank 1 in both metrics)
	- Penalizes extreme imbalances (good in one, bad in other)
	- No weight parameter needed - naturally balances both objectives
	- A genome must be good at BOTH metrics to rank high

	Example:
		| Genome | CE    | Acc   | Rank_CE | Rank_Acc | HM   |
		|--------|-------|-------|---------|----------|------|
		| A      | 10.32 | 0.02% | 1       | 4        | 1.60 |
		| B      | 10.35 | 0.04% | 2       | 1        | 1.33 | <- Winner
		| C      | 10.38 | 0.03% | 3       | 2        | 2.40 |

		Genome B wins: not best at either metric alone, but best combined.
	"""

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

		# Compute harmonic mean of ranks
		# HM(a, b) = 2 / (1/a + 1/b) = 2ab / (a + b)
		fitness_scores = []
		for i in range(n):
			r_ce = rank_ce[i]
			r_acc = rank_acc[i]
			hm = 2.0 * r_ce * r_acc / (r_ce + r_acc)
			fitness_scores.append(hm)

		return fitness_scores

	@property
	def name(self) -> str:
		return "HarmonicRank"
