"""
Abstract base class for fitness calculators.

Fitness calculators combine CE and accuracy into a single fitness score
for ranking genomes during GA/TS optimization.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

G = TypeVar('G')  # Genome type


class FitnessCalculator(ABC, Generic[G]):
	"""
	Abstract base class for fitness calculation and ranking.

	Subclasses implement different strategies for combining CE and accuracy
	into a fitness score. Lower fitness = better.

	Usage:
		calculator = FitnessCalculatorFactory.create(FitnessCalculatorType.HARMONIC_RANK)
		ranked = calculator.rank(population)
		# ranked is sorted by fitness (best first)
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

	@property
	@abstractmethod
	def name(self) -> str:
		"""Human-readable name for this calculator."""
		pass
