"""Overfitting detection over train/eval fitness gaps."""

from dataclasses import dataclass, field
from typing import Callable, Optional


# =============================================================================
# Overfitting Detection
# =============================================================================

class OverfitDetector:
	"""
	Detects overfitting by comparing current performance against a fixed baseline.

	The baseline is the mean fitness of top-K elites evaluated on FULL validation
	data at initialization. Each tick compares current top-K on FULL validation
	against this baseline.

	Delta = (current_mean - baseline_mean) / baseline_mean × 100
	- Positive delta = overfitting (worse on full validation than baseline)
	- Negative delta = generalizing (better on full validation than baseline)

	Usage:
		# Initialize with top-K fitness values on FULL data
		detector = OverfitDetector(initial_fitness_values)

		# Each check interval, pass current top-K fitness on FULL data
		delta = detector.tick(current_fitness_values)
		# delta > 0 means overfitting, delta < 0 means improving
	"""

	def __init__(self, initial_fitness: list[float]):
		"""
		Initialize with baseline fitness values.

		Args:
			initial_fitness: Fitness values of top-K elites on FULL validation at init
		"""
		if not initial_fitness:
			raise ValueError("initial_fitness cannot be empty")
		self._baseline_mean = sum(initial_fitness) / len(initial_fitness)
		self._k = len(initial_fitness)

	@property
	def baseline_mean(self) -> float:
		"""The fixed baseline mean from initialization."""
		return self._baseline_mean

	@property
	def k(self) -> int:
		"""Number of elites used for baseline."""
		return self._k

	def tick(self, current_fitness: list[float]) -> float:
		"""
		Compute delta against baseline.

		Args:
			current_fitness: Fitness values of top-K elites on FULL validation NOW

		Returns:
			Delta percentage: positive = overfitting, negative = improving
		"""
		if not current_fitness:
			return 0.0
		current_mean = sum(current_fitness) / len(current_fitness)
		if self._baseline_mean == 0:
			return 0.0
		return (current_mean - self._baseline_mean) / self._baseline_mean * 100

	def tick_with_mean(self, current_fitness: list[float]) -> tuple[float, float]:
		"""
		Compute delta and return both delta and current mean.

		Args:
			current_fitness: Fitness values of top-K elites on FULL validation NOW

		Returns:
			Tuple of (delta_percentage, current_mean)
		"""
		if not current_fitness:
			return 0.0, 0.0
		current_mean = sum(current_fitness) / len(current_fitness)
		if self._baseline_mean == 0:
			return 0.0, current_mean
		delta = (current_mean - self._baseline_mean) / self._baseline_mean * 100
		return delta, current_mean
