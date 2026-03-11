"""
IDS Security fitness calculator.

Optimizes for the IDS security metric: F1 × (1 - FPR)^2

This fitness function strongly penalizes high false positive rates while
rewarding balanced per-class performance (F1-macro). In production IDS
deployments, even 1% FPR at scale = thousands of false alerts per day.

The (1 - FPR)^2 term makes FPR reduction exponentially more valuable:
- FPR 0.10 → penalty 0.81 (mild)
- FPR 0.05 → penalty 0.90 (good)
- FPR 0.01 → penalty 0.98 (excellent)

Population tuples: (genome, ce, accuracy, f1_macro, fpr)
"""

from typing import TypeVar

from .FitnessCalculator import FitnessCalculator

G = TypeVar('G')


class FitnessCalculatorIDSSecurity(FitnessCalculator[G]):
	"""
	Fitness = 1 - F1 × (1 - FPR)^2

	Lower fitness = better (convention: all fitness calculators minimize).
	Ranges from 0.0 (perfect: F1=1.0, FPR=0.0) to 1.0 (worst).

	Falls back to CE-based ranking if F1/FPR are not available in tuples.
	"""

	def fitness(self, population: list[tuple]) -> list[float]:
		n = len(population)
		if n == 0:
			return []
		if n == 1:
			return [0.0]

		# Check if F1 and FPR are available (5-element tuples)
		has_ids = len(population[0]) >= 5

		if has_ids:
			scores = []
			for t in population:
				f1 = t[3] if t[3] is not None else 0.0
				fpr = t[4] if t[4] is not None else 1.0
				# F1 × (1 - FPR)^2, inverted for minimization
				security_score = f1 * (1 - fpr) ** 2
				scores.append(1.0 - security_score)
			return scores

		# Fallback: rank by CE if IDS metrics not available
		return [t[1] for t in population]

	def bests(self, population):
		return self._ids_bests(population)

	@property
	def name(self) -> str:
		return "IDSSecurity(F1×(1-FPR)²)"
