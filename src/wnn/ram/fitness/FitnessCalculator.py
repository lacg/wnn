"""
Abstract base class for fitness calculators.

Fitness calculators combine metrics (CE, accuracy, F1, FPR) into a single
fitness score for ranking genomes during GA/TS optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional

from wnn.ram.metrics import Metrics, GenomeType

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

		# Best by each metric
		best_ce_idx = min(range(len(metrics_list)), key=lambda i: metrics_list[i].ce)
		best_acc_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i].acc)

		# F1: highest (fallback to best_ce if not available)
		if any(m.f1 is not None for m in metrics_list):
			best_f1_idx = max(range(len(metrics_list)),
				key=lambda i: metrics_list[i].f1 if metrics_list[i].f1 is not None else -1.0)
		else:
			best_f1_idx = best_ce_idx

		# FPR: lowest (fallback to best_acc if not available)
		if any(m.fpr is not None for m in metrics_list):
			best_fpr_idx = min(range(len(metrics_list)),
				key=lambda i: metrics_list[i].fpr if metrics_list[i].fpr is not None else 2.0)
		else:
			best_fpr_idx = best_acc_idx

		# Best fitness
		best_fit_idx = min(range(len(scores)), key=lambda i: scores[i])

		def _make(idx: int, gtype: GenomeType) -> GenomeBest[G]:
			m = metrics_list[idx]
			return GenomeBest(genome=genomes[idx], metrics=Metrics(
				ce=m.ce, acc=m.acc, f1=m.f1, fpr=m.fpr,
				threshold=m.threshold, fitness=scores[idx],
				bit_accuracy=m.bit_accuracy,
			), genome_type=gtype)

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
