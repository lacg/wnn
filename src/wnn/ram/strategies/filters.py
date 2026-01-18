"""
Population filters for optimization strategies.

Provides reusable filters that can be applied to populations during
optimization to control selection pressure on various metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar('T')  # Genome type


class FilterMode(IntEnum):
	"""How to interpret the percentile threshold."""
	LOWER_IS_BETTER = auto()  # Keep genomes with LOWEST values (e.g., CE)
	HIGHER_IS_BETTER = auto()  # Keep genomes with HIGHEST values (e.g., Accuracy)


@dataclass
class FilterResult(Generic[T]):
	"""Result of applying a filter to a population."""
	kept: list[tuple[T, float]]  # Genomes that passed the filter
	filtered: list[tuple[T, float]]  # Genomes that were filtered out
	threshold_value: float  # The computed threshold
	kept_count: int
	total_count: int

	@property
	def kept_ratio(self) -> float:
		"""Ratio of kept genomes."""
		return self.kept_count / self.total_count if self.total_count > 0 else 0.0


class PercentileFilter(Generic[T]):
	"""
	Filters a population to keep only genomes within a percentile threshold.

	Can be used for CE filtering (lower is better) or accuracy filtering
	(higher is better). Supports optional absolute floor/ceiling constraints.

	Example usage:
		# Keep top 75% by CE (lower is better)
		ce_filter = PercentileFilter(
			percentile=0.75,
			mode=FilterMode.LOWER_IS_BETTER,
			metric_name="CE",
		)

		# Filter population
		result = ce_filter.apply(population, key=lambda g, f: f)
		viable_population = result.kept

		# With absolute floor (never accept CE > 10.5)
		ce_filter = PercentileFilter(
			percentile=0.75,
			mode=FilterMode.LOWER_IS_BETTER,
			absolute_bound=10.5,
			metric_name="CE",
		)
	"""

	def __init__(
		self,
		percentile: float = 0.75,
		mode: FilterMode = FilterMode.LOWER_IS_BETTER,
		absolute_bound: Optional[float] = None,
		metric_name: str = "metric",
	):
		"""
		Initialize the filter.

		Args:
			percentile: Fraction of population to keep (0.0-1.0).
				0.75 means keep top 75%.
			mode: Whether lower or higher values are better.
			absolute_bound: Optional absolute threshold.
				For LOWER_IS_BETTER: values above this are always filtered.
				For HIGHER_IS_BETTER: values below this are always filtered.
			metric_name: Name for logging purposes.
		"""
		if not 0.0 < percentile <= 1.0:
			raise ValueError(f"Percentile must be in (0, 1], got {percentile}")

		self.percentile = percentile
		self.mode = mode
		self.absolute_bound = absolute_bound
		self.metric_name = metric_name

	def compute_threshold(self, values: list[float]) -> float:
		"""
		Compute the percentile threshold from a list of values.

		Args:
			values: List of metric values

		Returns:
			The threshold value at the specified percentile
		"""
		if not values:
			return float('inf') if self.mode == FilterMode.LOWER_IS_BETTER else float('-inf')

		sorted_values = sorted(values, reverse=(self.mode == FilterMode.HIGHER_IS_BETTER))
		# Index for the percentile cutoff
		cutoff_idx = int(len(sorted_values) * self.percentile) - 1
		cutoff_idx = max(0, min(cutoff_idx, len(sorted_values) - 1))

		threshold = sorted_values[cutoff_idx]

		# Apply absolute bound if specified
		if self.absolute_bound is not None:
			if self.mode == FilterMode.LOWER_IS_BETTER:
				# For CE: threshold should not exceed absolute_bound
				threshold = min(threshold, self.absolute_bound)
			else:
				# For Acc: threshold should not go below absolute_bound
				threshold = max(threshold, self.absolute_bound)

		return threshold

	def apply(
		self,
		population: list[tuple[T, float]],
		key: Optional[Callable[[T, float], float]] = None,
	) -> FilterResult[T]:
		"""
		Apply the filter to a population.

		Args:
			population: List of (genome, fitness) tuples
			key: Function to extract the metric value from (genome, fitness).
				Defaults to using fitness directly.

		Returns:
			FilterResult with kept/filtered genomes and statistics
		"""
		if key is None:
			key = lambda g, f: f

		if not population:
			return FilterResult(
				kept=[],
				filtered=[],
				threshold_value=0.0,
				kept_count=0,
				total_count=0,
			)

		# Extract metric values
		values = [key(g, f) for g, f in population]

		# Compute threshold
		threshold = self.compute_threshold(values)

		# Filter population
		kept = []
		filtered = []

		for (genome, fitness), value in zip(population, values):
			passes = self._passes_threshold(value, threshold)
			if passes:
				kept.append((genome, fitness))
			else:
				filtered.append((genome, fitness))

		return FilterResult(
			kept=kept,
			filtered=filtered,
			threshold_value=threshold,
			kept_count=len(kept),
			total_count=len(population),
		)

	def _passes_threshold(self, value: float, threshold: float) -> bool:
		"""Check if a value passes the threshold."""
		if self.mode == FilterMode.LOWER_IS_BETTER:
			return value <= threshold
		else:
			return value >= threshold

	def __repr__(self) -> str:
		bound_str = f", bound={self.absolute_bound}" if self.absolute_bound else ""
		mode_str = "lower_better" if self.mode == FilterMode.LOWER_IS_BETTER else "higher_better"
		return f"PercentileFilter({self.metric_name}, {self.percentile:.0%}, {mode_str}{bound_str})"


class DualPercentileFilter(Generic[T]):
	"""
	Filters a population using two metrics simultaneously.

	Useful for filtering by both CE and accuracy, keeping genomes that
	pass BOTH thresholds.

	Example:
		dual_filter = DualPercentileFilter(
			primary=PercentileFilter(0.75, FilterMode.LOWER_IS_BETTER, metric_name="CE"),
			secondary=PercentileFilter(0.5, FilterMode.HIGHER_IS_BETTER, metric_name="Acc"),
		)

		# Population items are (genome, (ce, acc)) tuples
		result = dual_filter.apply(
			population,
			primary_key=lambda g, f: f[0],  # CE
			secondary_key=lambda g, f: f[1],  # Acc
		)
	"""

	def __init__(
		self,
		primary: PercentileFilter[T],
		secondary: PercentileFilter[T],
		require_both: bool = True,
	):
		"""
		Initialize dual filter.

		Args:
			primary: First filter (e.g., CE filter)
			secondary: Second filter (e.g., Acc filter)
			require_both: If True, genome must pass both filters.
				If False, genome passes if it passes either filter.
		"""
		self.primary = primary
		self.secondary = secondary
		self.require_both = require_both

	def apply(
		self,
		population: list[tuple[T, float]],
		primary_key: Callable[[T, float], float],
		secondary_key: Callable[[T, float], float],
	) -> FilterResult[T]:
		"""
		Apply both filters to the population.

		Args:
			population: List of (genome, fitness) tuples
			primary_key: Function to extract primary metric
			secondary_key: Function to extract secondary metric

		Returns:
			FilterResult with genomes passing the combined filter
		"""
		if not population:
			return FilterResult(
				kept=[],
				filtered=[],
				threshold_value=0.0,
				kept_count=0,
				total_count=0,
			)

		# Compute thresholds
		primary_values = [primary_key(g, f) for g, f in population]
		secondary_values = [secondary_key(g, f) for g, f in population]

		primary_threshold = self.primary.compute_threshold(primary_values)
		secondary_threshold = self.secondary.compute_threshold(secondary_values)

		# Filter
		kept = []
		filtered = []

		for (genome, fitness), pval, sval in zip(population, primary_values, secondary_values):
			passes_primary = self.primary._passes_threshold(pval, primary_threshold)
			passes_secondary = self.secondary._passes_threshold(sval, secondary_threshold)

			if self.require_both:
				passes = passes_primary and passes_secondary
			else:
				passes = passes_primary or passes_secondary

			if passes:
				kept.append((genome, fitness))
			else:
				filtered.append((genome, fitness))

		return FilterResult(
			kept=kept,
			filtered=filtered,
			threshold_value=primary_threshold,  # Return primary threshold
			kept_count=len(kept),
			total_count=len(population),
		)

	def __repr__(self) -> str:
		op = "AND" if self.require_both else "OR"
		return f"DualPercentileFilter({self.primary} {op} {self.secondary})"
