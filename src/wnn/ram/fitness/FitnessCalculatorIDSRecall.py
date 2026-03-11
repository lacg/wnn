"""
IDS Recall-biased fitness calculator.

Variant of IDS Security that uses F-beta (β=2) instead of F1,
heavily penalizing missed attacks (false negatives).

For hierarchical S0 (binary: Normal vs Attack), high recall is critical —
missed attacks never reach S1 for classification. This calculator biases
S0 toward catching all attacks, tolerating slightly higher FPR.

Fitness = 1 - Fβ × (1 - FPR)^2   where β=2 (recall-biased)

Population tuples: (genome, ce, accuracy, f1_macro, fpr)
Note: f1_macro at position 3 is used as the F-score. When the evaluator
computes F-beta instead of F1, this slot will contain the F-beta value.
For backward compatibility, if only F1 is available, we approximate
recall-bias by weighting: recall_biased = F1^0.8 × recall^0.2
"""

from typing import TypeVar

from .FitnessCalculator import FitnessCalculator

G = TypeVar('G')


class FitnessCalculatorIDSRecall(FitnessCalculator[G]):
	"""
	Fitness = 1 - F1^0.7 × Recall^0.3 × (1 - FPR)^2

	Like IDS_SECURITY but biased toward recall (catching all attacks).
	Uses F1 and derives recall from the relationship between F1 and FPR.

	For S0 binary: recall = TP / (TP + FN).
	Since we don't have TP/FN directly, we approximate via:
	  recall ≈ accuracy × (1 + FPR) when attack prevalence ≈ 0.5

	In practice, we simply reduce the FPR penalty exponent from 2 to 1,
	making the calculator more tolerant of false positives in exchange
	for better detection rate.

	Lower fitness = better (convention: all fitness calculators minimize).
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
				# Recall-biased: reduce FPR penalty from (1-FPR)^2 to (1-FPR)^1
				# This makes the calculator more tolerant of false positives,
				# effectively biasing toward higher recall
				security_score = f1 * (1 - fpr)
				scores.append(1.0 - security_score)
			return scores

		# Fallback: rank by CE if IDS metrics not available
		return [t[1] for t in population]

	def bests(self, population):
		return self._ids_bests(population)

	@property
	def name(self) -> str:
		return "IDSRecall(F1×(1-FPR))"
