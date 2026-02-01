"""
Fitness calculator wrapper that enforces a minimum accuracy floor.

Genomes with accuracy below the floor get worst fitness (infinity), preventing
the pathological optimization direction where CE improves but accuracy degrades.
"""

from typing import TypeVar
import math

from .FitnessCalculator import FitnessCalculator

G = TypeVar('G')


class FitnessCalculatorWithAccuracyFloor(FitnessCalculator[G]):
	"""
	Wrapper that enforces minimum accuracy threshold.

	Genomes with accuracy below min_accuracy get fitness = infinity,
	effectively removing them from consideration regardless of CE.

	This prevents the pathological case where the GA discovers that
	reducing neuron activity produces more uniform (lower CE) but
	near-random (near-zero accuracy) predictions.

	Usage:
		base = FitnessCalculatorNormalizedHarmonic(weight_ce=1.0, weight_acc=2.0)
		calculator = FitnessCalculatorWithAccuracyFloor(base, min_accuracy=0.003)
		# Genomes with acc < 0.3% get fitness = inf
	"""

	def __init__(
		self,
		base_calculator: FitnessCalculator[G],
		min_accuracy: float = 0.003,  # 0.3% default
	):
		"""
		Initialize with base calculator and minimum accuracy threshold.

		Args:
			base_calculator: Underlying fitness calculator to use
			min_accuracy: Minimum accuracy (0.0-1.0) below which fitness = inf
		"""
		if min_accuracy < 0 or min_accuracy > 1:
			raise ValueError(f"min_accuracy must be in [0, 1], got {min_accuracy}")
		self._base = base_calculator
		self._min_accuracy = min_accuracy

	@property
	def min_accuracy(self) -> float:
		"""Minimum accuracy threshold."""
		return self._min_accuracy

	@property
	def base_calculator(self) -> FitnessCalculator[G]:
		"""Underlying fitness calculator."""
		return self._base

	def fitness(self, population: list[tuple[G, float, float]]) -> list[float]:
		"""
		Compute fitness with accuracy floor enforcement.

		Args:
			population: List of (genome, ce, accuracy) tuples

		Returns:
			List of fitness values where genomes below min_accuracy get inf
		"""
		if not population:
			return []

		# Get base fitness scores
		base_scores = self._base.fitness(population)

		# Override with infinity for genomes below accuracy floor
		result = []
		for (_, _, acc), score in zip(population, base_scores):
			if acc < self._min_accuracy:
				result.append(math.inf)
			else:
				result.append(score)

		return result

	@property
	def name(self) -> str:
		base_name = self._base.name
		return f"{base_name}(floor={self._min_accuracy:.4f})"
