"""
Abstract base class for fitness calculators.

Fitness calculators combine metrics (CE, accuracy, F1, FPR) into a single
fitness score for ranking genomes during GA/TS optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TypeVar, Generic, Optional

from wnn.ram.metrics import IDSMetrics, Metrics, GenomeType

G = TypeVar('G')  # Genome type


@dataclass
class GenomeBest(Generic[G]):
	"""A genome identified as best by a specific criterion."""
	genome: G
	metrics: Metrics
	genome_type: GenomeType


@dataclass
class PopulationBests(Generic[G]):
	"""Best genomes by different criteria from a population.

	Each is a different genome selected by a different metric:
	- best_ce: lowest cross-entropy
	- best_acc: highest accuracy
	- best_f1: highest F1-macro (IDS) or same as best_ce (LM)
	- best_fpr: lowest FPR (IDS) or same as best_acc (LM)
	- best_fitness: best combined fitness score
	"""
	best_ce: GenomeBest[G]
	best_acc: GenomeBest[G]
	best_f1: GenomeBest[G]
	best_fpr: GenomeBest[G]
	best_fitness: GenomeBest[G]


def _lower_better_scalar(m) -> float:
	"""Real CE where CE exists (IDS/LM); -reward on ControllerMetrics. Diagnostic and
	slot-ranking use ONLY — never display this as CE."""
	ce = getattr(m, "ce", None)
	return float(ce) if ce is not None else -float(m.reward)


def compute_ranks(values: list[float], ascending: bool = True) -> list[float]:
	"""1-based fractional ranks; tied values share the AVERAGE of their positions.

	THE ONE ranking helper for every rank-based fitness calculator (IDS harmonic-rank
	and controller harmonic alike). It replaces two identical per-class copies whose
	shared defect was found 09/08/2026: `enumerate(sorted(...))` hands TIED values
	distinct ranks by list position, so among genomes that are exactly equal on a
	metric the ordering is arbitrary — and systematically biased, because populations
	arrive sorted by the previous generation's fitness, silently favoring incumbent
	elites on precisely the metric where genomes are indistinguishable. Measured on
	the live IDS cohort: 42% of populations carry an fpr tie, worst case 20 genomes
	sharing the best fpr yet ranked 1..20. In the controller the dominant tie is
	stable_rate=100%. Fractional ranks (standard average-rank / "1224.5" scheme) make
	the ranking a pure function of the VALUES: order-independent, equal values equal
	rank, and Σranks preserved so WHM weights keep their meaning.

	ascending=True → lower value gets rank 1 (costs); False → higher value gets
	rank 1 (scores). Ties group on exact value equality, matching the equality the
	old code silently broke.
	"""
	n = len(values)
	order = sorted(range(n), key=lambda i: values[i] if ascending else -values[i])
	ranks = [0.0] * n
	i = 0
	while i < n:
		j = i
		while j + 1 < n and values[order[j + 1]] == values[order[i]]:
			j += 1
		avg = (i + j) / 2 + 1          # positions i..j hold ranks i+1..j+1 → their mean
		for k in range(i, j + 1):
			ranks[order[k]] = avg
		i = j + 1
	return ranks


class FitnessCalculator(ABC, Generic[G]):
	"""
	Abstract base class for fitness calculation and ranking.

	All calculators accept list[Metrics] and return list[float] scores
	where lower = better.
	"""

	@abstractmethod
	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		"""
		Compute fitness scores for a list of metrics.

		Args:
			metrics_list: List of Metrics objects (one per genome)

		Returns:
			List of fitness scores (lower = better), same order as input
		"""
		pass

	def rank(self, genomes: list[G], metrics_list: list[Metrics]) -> list[tuple[G, float]]:
		"""
		Rank genomes by fitness.

		Returns:
			List of (genome, fitness_score) sorted by fitness (lower = better)
		"""
		scores = self.fitness(metrics_list)
		ranked = list(zip(genomes, scores))
		ranked.sort(key=lambda x: x[1])
		return ranked

	def bests(self, genomes: list[G], metrics_list: list[Metrics]) -> PopulationBests[G]:
		"""
		Extract the best genomes by each criterion.

		Returns the genome with best CE, best accuracy, best F1, best FPR,
		and best fitness. These are typically different genomes.
		"""
		if not genomes or not metrics_list:
			raise ValueError("Cannot compute bests on empty population")

		scores = self.fitness(metrics_list)

		# Best by each metric. The "ce" SLOT is legacy vocabulary (DB columns, GenomeType
		# names): on IDS/LM it ranks real CE; on the controller — which has NO ce — it
		# ranks -reward, labeled as such nowhere user-facing. acc is IDSMetrics.acc or
		# ControllerMetrics.acc (a read-only alias of stable_rate).
		best_ce_idx = min(range(len(metrics_list)), key=lambda i: _lower_better_scalar(metrics_list[i]))
		best_acc_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i].acc)

		# F1: highest (fallback to best_ce if not available)
		if any(getattr(m, "f1", None) is not None for m in metrics_list):
			best_f1_idx = max(range(len(metrics_list)),
				key=lambda i: getattr(metrics_list[i], "f1", None) if getattr(metrics_list[i], "f1", None) is not None else -1.0)
		else:
			best_f1_idx = best_ce_idx

		# FPR: lowest (fallback to best_acc if not available)
		if any(getattr(m, "fpr", None) is not None for m in metrics_list):
			best_fpr_idx = min(range(len(metrics_list)),
				key=lambda i: getattr(metrics_list[i], "fpr", None) if getattr(metrics_list[i], "fpr", None) is not None else 2.0)
		else:
			best_fpr_idx = best_acc_idx

		# Best fitness
		best_fit_idx = min(range(len(scores)), key=lambda i: scores[i])

		def _make(idx: int, gtype: GenomeType) -> GenomeBest[G]:
			# Type-preserving copy with the composite fitness stamped on — works for
			# IDSMetrics AND ControllerMetrics (the old field-by-field rebuild hardcoded
			# the IDS shape and silently dropped every field it didn't name).
			m = replace(metrics_list[idx], fitness=scores[idx])
			return GenomeBest(genome=genomes[idx], metrics=m, genome_type=gtype)

		return PopulationBests(
			best_ce=_make(best_ce_idx, GenomeType.BEST_CE),
			best_acc=_make(best_acc_idx, GenomeType.BEST_ACC),
			best_f1=_make(best_f1_idx, GenomeType.BEST_F1),
			best_fpr=_make(best_fpr_idx, GenomeType.BEST_FPR),
			best_fitness=_make(best_fit_idx, GenomeType.BEST_FITNESS),
		)

	@property
	@abstractmethod
	def name(self) -> str:
		"""Human-readable name for this calculator."""
		pass
