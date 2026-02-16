"""
Abstract base class for fitness calculators.

Fitness calculators combine CE and accuracy into a single fitness score
for ranking genomes during GA/TS optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional

G = TypeVar('G')  # Genome type


@dataclass
class GenomeBest(Generic[G]):
	"""A genome identified as best by a specific metric."""
	genome: G
	ce: float
	accuracy: float
	fitness_score: float


@dataclass
class PopulationBests(Generic[G]):
	"""Three independent bests from potentially different genomes.

	In a well-optimized population, these are typically three DIFFERENT genomes:
	- best_ce: lowest cross-entropy (best language model)
	- best_acc: highest accuracy (best classifier)
	- best_fitness: best combined score (harmonic rank or similar)
	"""
	best_ce: GenomeBest[G]
	best_acc: GenomeBest[G]
	best_fitness: GenomeBest[G]


class FitnessCalculator(ABC, Generic[G]):
	"""
	Abstract base class for fitness calculation and ranking.

	Subclasses implement different strategies for combining CE and accuracy
	into a fitness score. Lower fitness = better.

	Usage:
		calculator = FitnessCalculatorFactory.create(FitnessCalculatorType.HARMONIC_RANK)
		ranked = calculator.rank(population)
		# ranked is sorted by fitness (best first)

		bests = calculator.bests(population)
		# bests.best_ce, bests.best_acc, bests.best_fitness — three independent genomes
	"""

	@abstractmethod
	def fitness(self, population: list[tuple[G, float, float]]) -> list[float]:
		"""
		Compute fitness scores for a population.

		Args:
			population: List of (genome, ce, accuracy) tuples
				- ce: Cross-entropy loss (lower is better)
				- accuracy: Prediction accuracy 0.0-1.0 (higher is better)

		Returns:
			List of fitness scores (lower = better), same order as input
		"""
		pass

	def rank(self, population: list[tuple[G, float, float]]) -> list[tuple[G, float]]:
		"""
		Rank population by fitness.

		Calls fitness() to compute scores, then sorts by fitness ascending.

		Args:
			population: List of (genome, ce, accuracy) tuples

		Returns:
			List of (genome, fitness_score) sorted by fitness (lower = better)
		"""
		fitness_scores = self.fitness(population)
		# Pair genomes with their fitness scores
		ranked = [(genome, score) for (genome, _, _), score in zip(population, fitness_scores)]
		# Sort by fitness (lower = better)
		ranked.sort(key=lambda x: x[1])
		return ranked

	def bests(self, population: list[tuple[G, float, Optional[float]]]) -> PopulationBests[G]:
		"""
		Extract the three independent bests from a population.

		Returns the genome with the best CE, best accuracy, and best fitness
		score. These are typically three DIFFERENT genomes.

		Handles None accuracy values by treating them as 0.0.

		Args:
			population: List of (genome, ce, accuracy) tuples.
				Accuracy may be None (treated as 0.0).

		Returns:
			PopulationBests with best_ce, best_acc, and best_fitness.
		"""
		if not population:
			raise ValueError("Cannot compute bests on empty population")

		# Normalize None accuracy to 0.0 for fitness calculation
		normalized = [
			(g, ce, acc if acc is not None else 0.0)
			for g, ce, acc in population
		]
		scores = self.fitness(normalized)

		best_ce_idx = min(range(len(normalized)), key=lambda i: normalized[i][1])
		best_acc_idx = max(range(len(normalized)), key=lambda i: normalized[i][2])
		best_fit_idx = min(range(len(scores)), key=lambda i: scores[i])

		def _make(idx: int) -> GenomeBest[G]:
			g, ce, acc = normalized[idx]
			return GenomeBest(genome=g, ce=ce, accuracy=acc, fitness_score=scores[idx])

		return PopulationBests(
			best_ce=_make(best_ce_idx),
			best_acc=_make(best_acc_idx),
			best_fitness=_make(best_fit_idx),
		)

	@property
	@abstractmethod
	def name(self) -> str:
		"""Human-readable name for this calculator."""
		pass
