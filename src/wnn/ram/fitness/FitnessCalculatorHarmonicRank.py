"""
Harmonic Rank fitness calculator.

Ranks genomes by weighted harmonic mean of their per-metric ranks.
Supports all 4 metrics: CE, accuracy, F1-macro, FPR.
"""

from wnn.ram.metrics import Metrics, FitnessWeights, MetricType
from .FitnessCalculator import FitnessCalculator, compute_ranks


class FitnessCalculatorHarmonicRank(FitnessCalculator):
	"""
	WHM = sum(weights) / sum(weight_i / rank_i)

	Lower WHM = better (closer to rank 1 in all metrics).
	Metrics with weight=0 are excluded.

	Ranks come from the shared `compute_ranks` (fractional, tie-aware) — this class
	carried its own positional-rank copy until 09/08/2026; see compute_ranks's
	docstring for the tie bias that fixed.
	"""

	def __init__(
		self,
		weight_ce: float = 1.0,
		weight_acc: float = 1.0,
		weight_f1: float = 0.0,
		weight_fpr: float = 0.0,
	):
		self.weights = FitnessWeights(ce=weight_ce, acc=weight_acc, f1=weight_f1, fpr=weight_fpr)

	_compute_ranks = staticmethod(compute_ranks)

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		n = len(metrics_list)
		if n == 0:
			return []
		if n == 1:
			return [1.0]

		# Build (ranks, weight) for each active metric
		active: list[tuple[list[int], float]] = []

		if self.weights.ce > 0:
			ranks = self._compute_ranks([m.ce for m in metrics_list], ascending=True)
			active.append((ranks, self.weights.ce))

		if self.weights.acc > 0:
			ranks = self._compute_ranks([m.acc for m in metrics_list], ascending=False)
			active.append((ranks, self.weights.acc))

		if self.weights.f1 > 0:
			f1_vals = [m.f1 if m.f1 is not None else 0.0 for m in metrics_list]
			ranks = self._compute_ranks(f1_vals, ascending=False)
			active.append((ranks, self.weights.f1))

		if self.weights.fpr > 0:
			fpr_vals = [m.fpr if m.fpr is not None else 1.0 for m in metrics_list]
			ranks = self._compute_ranks(fpr_vals, ascending=True)
			active.append((ranks, self.weights.fpr))

		if not active:
			return [1.0] * n

		w_sum = sum(w for _, w in active)
		return [
			w_sum / sum(w / ranks[i] for ranks, w in active)
			for i in range(n)
		]

	@property
	def name(self) -> str:
		parts = []
		if self.weights.ce != 1.0 or self.weights.fpr > 0:
			parts.append(f"ce={self.weights.ce}")
		if self.weights.acc != 1.0 or self.weights.fpr > 0:
			parts.append(f"acc={self.weights.acc}")
		if self.weights.f1 > 0:
			parts.append(f"f1={self.weights.f1}")
		if self.weights.fpr > 0:
			parts.append(f"fpr={self.weights.fpr}")
		if not parts:
			return "HarmonicRank"
		return f"HarmonicRank({', '.join(parts)})"
