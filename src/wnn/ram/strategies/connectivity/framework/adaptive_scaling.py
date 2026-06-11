"""Adaptive parameter scaling driven by improvement levels."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional


# =============================================================================
# Adaptive Parameter Scaling
# =============================================================================

class AdaptiveLevel(IntEnum):
	"""Optimization health levels for adaptive scaling."""
	HEALTHY = 0    # Improving well, use base parameters
	NEUTRAL = 1    # Small improvement, use base parameters
	WARNING = 2    # Stalled/mild regression, boost parameters
	CRITICAL = 3   # Significant regression, max boost


@dataclass


@dataclass
class AdaptiveScalerConfig:
	"""Configuration for adaptive parameter scaling.

	Scale factors are applied to base values:
	- WARNING: base × (1 + warning_*_boost)
	- CRITICAL: base × (1 + critical_*_boost)

	Note: CRITICAL boosts are over BASE, not compounded over WARNING.
	"""
	# Population scaling (GA: population_size, TS: neighbors_per_iter)
	warning_population_boost: float = 0.15    # +15% at WARNING
	critical_population_boost: float = 0.30   # +30% at CRITICAL (over base)

	# Mutation rate scaling
	warning_mutation_boost: float = 0.50      # +50% at WARNING (0.1 → 0.15)
	critical_mutation_boost: float = 1.00     # +100% at CRITICAL (0.1 → 0.2)


# =============================================================================
# Progressive Accuracy Threshold
# =============================================================================

@dataclass


class AdaptiveScaler:
	"""
	Scales optimization parameters based on health level.

	Use with EarlyStoppingTracker to detect WARNING/CRITICAL states and
	automatically adjust population size and mutation rate to escape
	local optima or increase exploration.

	Usage:
		scaler = AdaptiveScaler(base_population=50, base_mutation=0.1)

		# In optimization loop, after early_stopper.check():
		scaler.update(early_stopper.current_level)
		if scaler.level_changed:
			# Apply new parameters
			cfg.population_size = scaler.population
			cfg.mutation_rate = scaler.mutation_rate
			scaler.log_transition(logger)

	Transition rules:
		- HEALTHY/NEUTRAL → WARNING: boost to warning levels
		- WARNING → CRITICAL: boost to critical levels (over base)
		- CRITICAL → WARNING: de-escalate to warning levels
		- WARNING → HEALTHY: restore base levels
	"""

	def __init__(
		self,
		base_population: int,
		base_mutation: float,
		config: Optional[AdaptiveScalerConfig] = None,
		name: str = "Optimizer",
	):
		self._config = config or AdaptiveScalerConfig()
		self._name = name

		# Base values (never change)
		self._base_population = base_population
		self._base_mutation = base_mutation

		# Current level and scaled values
		self._level = AdaptiveLevel.HEALTHY
		self._prev_level = AdaptiveLevel.HEALTHY
		self._population = base_population
		self._mutation_rate = base_mutation

	@property
	def level(self) -> AdaptiveLevel:
		"""Current adaptive level."""
		return self._level

	@property
	def level_changed(self) -> bool:
		"""True if level changed on last update."""
		return self._level != self._prev_level

	@property
	def population(self) -> int:
		"""Current scaled population size."""
		return self._population

	@property
	def mutation_rate(self) -> float:
		"""Current scaled mutation rate."""
		return self._mutation_rate

	@property
	def base_population(self) -> int:
		"""Original base population size."""
		return self._base_population

	@property
	def base_mutation(self) -> float:
		"""Original base mutation rate."""
		return self._base_mutation

	def update(self, new_level: AdaptiveLevel) -> AdaptiveLevel:
		"""
		Update to new level and recalculate scaled parameters.

		Transition rules:
		- Only boost when entering WARNING or CRITICAL
		- Only de-escalate to WARNING when leaving CRITICAL
		- Only restore base when reaching HEALTHY (not just NEUTRAL)
		"""
		self._prev_level = self._level
		cfg = self._config

		# Determine if we should change scaling
		if new_level == AdaptiveLevel.CRITICAL:
			# Always use critical scaling when CRITICAL
			self._level = AdaptiveLevel.CRITICAL
			self._population = int(self._base_population * (1 + cfg.critical_population_boost))
			self._mutation_rate = self._base_mutation * (1 + cfg.critical_mutation_boost)

		elif new_level == AdaptiveLevel.WARNING:
			# Use warning scaling
			self._level = AdaptiveLevel.WARNING
			self._population = int(self._base_population * (1 + cfg.warning_population_boost))
			self._mutation_rate = self._base_mutation * (1 + cfg.warning_mutation_boost)

		elif new_level == AdaptiveLevel.HEALTHY:
			# Restore base parameters (only on HEALTHY, not NEUTRAL)
			self._level = AdaptiveLevel.HEALTHY
			self._population = self._base_population
			self._mutation_rate = self._base_mutation

		else:  # NEUTRAL
			# Keep current level/scaling (don't change on NEUTRAL)
			# This prevents oscillation between base and boosted
			pass

		return self._level

	def log_transition(self, log_fn: Callable[[str], None]) -> None:
		"""Log the level transition if it changed."""
		if not self.level_changed:
			return

		prev_name = self._prev_level.name
		curr_name = self._level.name

		if self._level > self._prev_level:
			# Escalating
			log_fn(
				f"[{self._name}] Adaptive ESCALATE {prev_name} → {curr_name}: "
				f"pop={self._population} (+{(self._population/self._base_population - 1)*100:.0f}%), "
				f"mut={self._mutation_rate:.3f} (+{(self._mutation_rate/self._base_mutation - 1)*100:.0f}%)"
			)
		else:
			# De-escalating
			if self._level == AdaptiveLevel.HEALTHY:
				log_fn(
					f"[{self._name}] Adaptive RESTORE {prev_name} → {curr_name}: "
					f"pop={self._population}, mut={self._mutation_rate:.3f} (base)"
				)
			else:
				log_fn(
					f"[{self._name}] Adaptive DE-ESCALATE {prev_name} → {curr_name}: "
					f"pop={self._population} (+{(self._population/self._base_population - 1)*100:.0f}%), "
					f"mut={self._mutation_rate:.3f} (+{(self._mutation_rate/self._base_mutation - 1)*100:.0f}%)"
				)
