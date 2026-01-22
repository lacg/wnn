"""
Generic GA and TS strategy base classes.

These provide genome-agnostic optimization algorithms that can be specialized
for different genome types (connectivity patterns, architecture configurations, etc.)
through abstract genome operations.

The core GA/TS loops are implemented here, subclasses provide:
- clone_genome: Copy a genome
- mutate_genome: Generate a neighbor by mutation
- crossover_genomes: Combine two parents (GA only)

Supports:
- Early stopping with patience and delta logging (EarlyStoppingTracker)
- Overfitting detection via callbacks (OverfittingCallback)
- Diversity mode for escaping local optima
"""

import logging
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, Generic, Optional, TypeVar, Any

from wnn.ram.fitness import FitnessCalculatorType

# Generic genome type
T = TypeVar('T')

# Custom TRACE level (below DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class OptimizationLogger:
	"""
	Logger wrapper with TRACE, DEBUG, INFO, ERROR levels.

	TRACE: Filtered candidates, very verbose per-candidate info (stdout only)
	DEBUG: Individual genome info (elites, init genomes)
	INFO: Progress summaries, phase transitions
	ERROR: Errors and warnings

	Usage:
		logger = OptimizationLogger("ArchitectureGA", level=logging.DEBUG)
		logger.debug("Elite details...")
		logger.trace("Filtered candidate...")
		logger.info("Generation complete")

	With file logging:
		file_log = lambda msg: print(msg, file=open("log.txt", "a"))
		logger = OptimizationLogger("GA", file_logger=file_log)
	"""

	def __init__(
		self,
		name: str,
		level: int = logging.DEBUG,
		file_logger: Optional[Callable[[str], None]] = None,
	):
		self._logger = logging.getLogger(f"wnn.optimizer.{name}")
		# Only add StreamHandler if no file_logger (file_logger handles stdout+file)
		if not file_logger and not self._logger.handlers:
			handler = logging.StreamHandler()
			handler.setFormatter(logging.Formatter("%(message)s"))
			self._logger.addHandler(handler)
		self._logger.setLevel(level)
		self._name = name
		self._file_logger = file_logger  # Handles stdout + file when provided

	def trace(self, msg: str) -> None:
		"""Log at TRACE level (filtered candidates, stdout only)."""
		if self._logger.isEnabledFor(TRACE):
			if self._file_logger:
				# file_logger handles stdout+file, but TRACE goes to stdout only
				print(msg)
			else:
				self._logger.log(TRACE, msg)

	def debug(self, msg: str) -> None:
		"""Log at DEBUG level (individual genome info)."""
		if self._logger.isEnabledFor(logging.DEBUG):
			if self._file_logger:
				self._file_logger(msg)  # file_logger handles stdout + file
			else:
				self._logger.debug(msg)

	def info(self, msg: str) -> None:
		"""Log at INFO level (progress summaries)."""
		if self._logger.isEnabledFor(logging.INFO):
			if self._file_logger:
				self._file_logger(msg)
			else:
				self._logger.info(msg)

	def warning(self, msg: str) -> None:
		"""Log at WARNING level."""
		if self._logger.isEnabledFor(logging.WARNING):
			if self._file_logger:
				self._file_logger(msg)
			else:
				self._logger.warning(msg)

	def error(self, msg: str) -> None:
		"""Log at ERROR level."""
		if self._logger.isEnabledFor(logging.ERROR):
			if self._file_logger:
				self._file_logger(msg)
			else:
				self._logger.error(msg)

	def __call__(self, msg: str) -> None:
		"""Default: INFO level (backward compatible with print-style logging)."""
		self.info(msg)

	def set_level(self, level: int) -> None:
		"""Change log level dynamically."""
		self._logger.setLevel(level)


class StopReason(IntEnum):
	"""Reason why optimization stopped early."""
	CONVERGENCE = auto()  # No improvement for patience iterations
	OVERFITTING = auto()  # Overfitting callback triggered early stop
	MAX_ITERATIONS = auto()  # Reached maximum iterations (not early stopped)


@dataclass
class OptimizerResult(Generic[T]):
	"""
	Unified result from optimization (GA, TS, SA).

	This is a generic result type that works with any genome type (Tensor, ClusterGenome, etc.)
	through the type parameter T.

	Naming conventions:
	- Uses 'genome' terminology (more generic than 'connections')
	- Uses 'fitness' terminology (minimization by default, lower is better)

	Attributes:
		initial_genome: Starting genome before optimization
		best_genome: Best genome found during optimization
		initial_fitness: Fitness of initial genome (lower is better)
		final_fitness: Fitness of best genome
		improvement_percent: Percentage improvement ((initial - final) / initial * 100)
		iterations_run: Number of iterations/generations run
		method_name: Name of the optimization method (e.g., "ArchitectureGA")
		history: List of (iteration, best_fitness) tuples for plotting
		early_stopped: Whether optimization stopped early (due to convergence or overfitting)
		stop_reason: Why optimization stopped (StopReason enum)
		final_population: Final population for seeding next phase (GA/TS)
		initial_accuracy: Optional accuracy at start
		final_accuracy: Optional accuracy at end
		final_threshold: Final accuracy threshold (pass to next phase for continuity)
	"""
	initial_genome: T
	best_genome: T
	initial_fitness: float
	final_fitness: float
	improvement_percent: float
	iterations_run: int
	method_name: str
	history: list[tuple[int, float]] = field(default_factory=list)
	early_stopped: bool = False
	stop_reason: Optional[StopReason] = None
	# For population seeding between phases
	final_population: Optional[list[T]] = None
	# Accuracy tracking
	initial_accuracy: Optional[float] = None
	final_accuracy: Optional[float] = None
	# Threshold continuity: pass to next phase (no hardcoded phase_index jumps)
	final_threshold: Optional[float] = None

	def __repr__(self) -> str:
		stop_str = f", stop={self.stop_reason.name}" if self.stop_reason else ""
		return (
			f"OptimizerResult("
			f"method={self.method_name}, "
			f"initial={self.initial_fitness:.4f}, "
			f"final={self.final_fitness:.4f}, "
			f"improvement={self.improvement_percent:.2f}%{stop_str})"
		)


@dataclass
class EarlyStoppingConfig:
	"""Configuration for early stopping with patience."""
	patience: int = 5              # Number of checks without improvement before stopping
	check_interval: int = 5        # Check every N iterations/generations
	min_improvement_pct: float = 0.02  # Minimum % improvement required to reset patience


class EarlyStoppingTracker:
	"""
	Reusable early stopping tracker with patience and delta logging.

	Checks improvement at regular intervals (default: every 5 iterations).
	Logs delta improvement and patience counter (e.g., "Δ=0.15%, patience=3/5").
	Stops when patience is exhausted (no improvement for patience * check_interval iterations).

	Usage:
		tracker = EarlyStoppingTracker(config, logger)
		for iteration in range(max_iterations):
			# ... do work ...
			if tracker.check(iteration, current_best_fitness):
				break  # Early stop
	"""

	# Level display formatting (emoji + name)
	_LEVEL_DISPLAY = {
		'HEALTHY': "🟢 HEALTHY",
		'NEUTRAL': "⚪ NEUTRAL",
		'WARNING': "🟡 WARNING",
		'CRITICAL': "🔴 CRITICAL",
	}

	def __init__(
		self,
		config: EarlyStoppingConfig,
		logger: Callable[[str], None],
		method_name: str = "Optimizer",
	):
		self._config = config
		self._log = logger
		self._method_name = method_name
		self._patience_counter = 0
		self._prev_best: Optional[float] = None
		self._baseline: Optional[float] = None
		# Import here to avoid circular import at module level
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveLevel
		self._last_level: 'AdaptiveLevel' = AdaptiveLevel.NEUTRAL

	def reset(self, initial_fitness: float) -> None:
		"""Reset tracker with initial fitness value."""
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveLevel
		self._patience_counter = 0
		self._prev_best = initial_fitness
		self._baseline = initial_fitness
		self._last_level = AdaptiveLevel.NEUTRAL

	def check(self, iteration: int, current_best: float) -> bool:
		"""
		Check if early stopping should occur.

		Args:
			iteration: Current iteration (0-indexed)
			current_best: Current best fitness value (lower is better)

		Returns:
			True if should stop, False otherwise
		"""
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveLevel
		cfg = self._config

		# Only check at specified intervals (1-indexed iteration)
		if (iteration + 1) % cfg.check_interval != 0:
			return False

		# Compute improvement from last check
		if self._prev_best is not None and self._prev_best > 0:
			improvement_pct = (self._prev_best - current_best) / self._prev_best * 100
		else:
			improvement_pct = 0.0

		# Check if improvement meets threshold
		if improvement_pct >= cfg.min_improvement_pct:
			# Recover 1 patience (not full reset) - prevents dragging out with small improvements
			self._patience_counter = max(0, self._patience_counter - 1)
			self._prev_best = current_best
		else:
			self._patience_counter += 1

		# Determine level using OverfitThreshold values (negate since improvement is opposite sign)
		# improvement_pct > 0 = improving, OverfitThreshold delta < 0 = healthy
		from wnn.core.thresholds import OverfitThreshold
		delta = -improvement_pct  # Convert to OverfitThreshold convention
		if delta < OverfitThreshold.HEALTHY:  # < -1% (big improvement)
			level = AdaptiveLevel.HEALTHY
		elif delta < OverfitThreshold.WARNING:  # -1% to 0% (small improvement)
			level = AdaptiveLevel.NEUTRAL
		elif delta < OverfitThreshold.CRITICAL:  # 0% to 3% (stalled/mild regression)
			level = AdaptiveLevel.WARNING
		else:  # >= 3% (significant regression)
			level = AdaptiveLevel.CRITICAL

		# Save level for adaptive scaling
		self._last_level = level

		# Log progress with delta, patience, and status display
		remaining = cfg.patience - self._patience_counter
		display = self._LEVEL_DISPLAY[level.name]
		self._log(
			f"[{self._method_name}] Early stop check: "
			f"Δ={improvement_pct:+.2f}%, patience={remaining}/{cfg.patience} {display}"
		)

		# Check if patience exhausted
		if self._patience_counter >= cfg.patience:
			total_iters_without_improvement = self._patience_counter * cfg.check_interval
			self._log(
				f"[{self._method_name}] Early stop: no improvement >= {cfg.min_improvement_pct}% "
				f"for {total_iters_without_improvement} iterations"
			)
			return True

		return False

	@property
	def patience_exhausted(self) -> bool:
		"""Check if patience is exhausted."""
		return self._patience_counter >= self._config.patience

	@property
	def current_level(self) -> 'AdaptiveLevel':
		"""Return the current AdaptiveLevel enum."""
		return self._last_level

	def reset_trend(self, top_k_fitness: list[float]) -> None:
		"""
		Reset tracker for trend-based early stopping.

		Args:
			top_k_fitness: Fitness values of top-K% genomes (lower is better).
			               For CE mode, pass CE values. For HARMONIC_RANK mode,
			               pass harmonic rank values.
		"""
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveLevel
		self._patience_counter = 0
		self._prev_trend_mean = sum(top_k_fitness) / len(top_k_fitness) if top_k_fitness else 0.0
		self._baseline = self._prev_trend_mean
		self._last_level = AdaptiveLevel.NEUTRAL

	def check_trend(self, iteration: int, top_k_fitness: list[float]) -> bool:
		"""
		Check early stopping using mean of top-K% fitness values.

		More robust than single-best comparison because it tracks the trend
		of the elite population rather than a single potentially-noisy genome.

		Args:
			iteration: Current iteration (0-indexed)
			top_k_fitness: Fitness values of top-K% genomes (lower is better).
			               For CE mode, pass CE values. For HARMONIC_RANK mode,
			               pass harmonic rank values.

		Returns:
			True if should stop, False otherwise
		"""
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveLevel
		cfg = self._config

		# Only check at specified intervals (1-indexed iteration)
		if (iteration + 1) % cfg.check_interval != 0:
			return False

		# Calculate current mean of top-K%
		if not top_k_fitness:
			return False
		current_mean = sum(top_k_fitness) / len(top_k_fitness)

		# Compute improvement from last check (using _prev_trend_mean if available, else _prev_best)
		prev_mean = getattr(self, '_prev_trend_mean', self._prev_best)
		if prev_mean is not None and prev_mean > 0:
			improvement_pct = (prev_mean - current_mean) / prev_mean * 100
		else:
			improvement_pct = 0.0

		# Check if improvement meets threshold
		if improvement_pct >= cfg.min_improvement_pct:
			# Recover 1 patience (not full reset)
			self._patience_counter = max(0, self._patience_counter - 1)
			self._prev_trend_mean = current_mean
		else:
			self._patience_counter += 1

		# Determine level using OverfitThreshold values
		from wnn.core.thresholds import OverfitThreshold
		delta = -improvement_pct
		if delta < OverfitThreshold.HEALTHY:
			level = AdaptiveLevel.HEALTHY
		elif delta < OverfitThreshold.WARNING:
			level = AdaptiveLevel.NEUTRAL
		elif delta < OverfitThreshold.CRITICAL:
			level = AdaptiveLevel.WARNING
		else:
			level = AdaptiveLevel.CRITICAL

		self._last_level = level

		# Log progress with trend info
		remaining = cfg.patience - self._patience_counter
		display = self._LEVEL_DISPLAY[level.name]
		self._log(
			f"[{self._method_name}] Early stop check (top-{len(top_k_fitness)} trend): "
			f"mean={current_mean:.4f}, Δ={improvement_pct:+.2f}%, patience={remaining}/{cfg.patience} {display}"
		)

		# Check if patience exhausted
		if self._patience_counter >= cfg.patience:
			total_iters = self._patience_counter * cfg.check_interval
			self._log(
				f"[{self._method_name}] Early stop: no trend improvement >= {cfg.min_improvement_pct}% "
				f"for {total_iters} iterations"
			)
			return True

		return False


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
class ProgressiveThresholdConfig:
	"""Configuration for progressive accuracy threshold.

	The threshold increases both within a phase (as progress goes 0→1) and
	across phases (each phase starts where the previous ended).

	Formula: threshold = base + (phase_index + progress) * delta

	Example with base=0.0001 (0.01%), delta=0.0002 (0.02%):
		Phase 1a (idx=0): 0.01% → 0.03%
		Phase 1b (idx=1): 0.03% → 0.05%
		Phase 2a (idx=2): 0.05% → 0.07%
		...
	"""
	base: float = 0.0001      # Starting threshold (0.01%)
	delta: float = 0.0002     # Increase per phase (0.02%)


class ProgressiveThreshold:
	"""
	Computes accuracy threshold that increases with optimization progress.

	The threshold gets stricter as optimization progresses, both within
	a single phase and across phases (curriculum learning).

	Usage:
		threshold = ProgressiveThreshold(phase_index=0)

		# In optimization loop:
		progress = iteration / total_iterations  # 0.0 to 1.0
		min_accuracy = threshold.get(progress)

		# For next phase:
		threshold = ProgressiveThreshold(phase_index=1)
	"""

	def __init__(
		self,
		phase_index: int = 0,
		config: Optional[ProgressiveThresholdConfig] = None,
	):
		"""
		Initialize progressive threshold.

		Args:
			phase_index: Current phase (0=1a, 1=1b, 2=2a, 3=2b, 4=3a, 5=3b)
			config: Optional configuration for base and delta values
		"""
		self._config = config or ProgressiveThresholdConfig()
		self._phase_index = phase_index

	@property
	def phase_index(self) -> int:
		"""Current phase index."""
		return self._phase_index

	@property
	def start_threshold(self) -> float:
		"""Threshold at start of this phase (progress=0)."""
		return self._config.base + self._phase_index * self._config.delta

	@property
	def end_threshold(self) -> float:
		"""Threshold at end of this phase (progress=1)."""
		return self._config.base + (self._phase_index + 1) * self._config.delta

	def get(self, progress: float) -> float:
		"""
		Get threshold for current progress within the phase.

		Args:
			progress: Progress through current phase (0.0 to 1.0)

		Returns:
			Accuracy threshold for filtering candidates
		"""
		progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
		return self._config.base + (self._phase_index + progress) * self._config.delta

	def format_range(self) -> str:
		"""Format the threshold range for logging."""
		return f"{self.start_threshold:.2%} → {self.end_threshold:.2%}"


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


@dataclass
class GAConfig:
	"""Configuration for Genetic Algorithm."""
	population_size: int = 30
	generations: int = 50
	mutation_rate: float = 0.1
	crossover_rate: float = 0.7
	tournament_size: int = 3  # Tournament selection size
	# Dual elitism: keep top N% by CE AND top N% by accuracy (unique)
	# Total elites = 10-20% of population depending on overlap
	elitism_pct: float = 0.1       # 10% by CE + 10% by accuracy
	# Threshold continuity: start threshold passed from previous phase
	# If None, uses min_accuracy as the base (first phase)
	initial_threshold: Optional[float] = None
	min_accuracy: float = 0.002    # 0.2% base threshold (used if initial_threshold is None)
	threshold_delta: float = 0.002   # 0.2% increase over full phase
	progressive_threshold: bool = True  # Enable progressive threshold within phase
	# CE percentile filter: keep only offspring in top X% by CE (None = disabled)
	# Example: 0.75 keeps top 75% by CE. Applied after accuracy threshold.
	ce_percentile: Optional[float] = None
	# Fitness calculator: how to combine CE and accuracy for ranking
	# CE = pure CE ranking, uses dual elites (10% CE + 10% Acc)
	# HARMONIC_RANK = harmonic mean of ranks (default), uses single elite (20% by rank)
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	# Weights for HARMONIC_RANK mode (higher weight = more important)
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	# Early stopping (all configurable via parameters)
	patience: int = 5              # Checks without improvement before stopping
	check_interval: int = 5        # Check every N generations
	min_improvement_pct: float = 0.05  # GA needs diversity, lower threshold (0.05%)


@dataclass
class TSConfig:
	"""Configuration for Tabu Search with dual-path CE/Acc optimization."""
	iterations: int = 100
	neighbors_per_iter: int = 20   # Total neighbors per iteration (split: N/2 from CE, N/2 from Acc)
	tabu_size: int = 10
	mutation_rate: float = 0.1     # Fraction of genome elements to mutate per neighbor
	# Total neighbors cache for seeding next phase (top K/2 by CE + top K/2 by Acc)
	# None = use neighbors_per_iter (legacy), set to GA population size to preserve diversity
	total_neighbors_size: Optional[int] = None
	# Threshold continuity: start threshold passed from previous phase
	# If None, uses min_accuracy as the base (first phase)
	initial_threshold: Optional[float] = None
	min_accuracy: float = 0.002    # 0.2% base threshold (used if initial_threshold is None)
	threshold_delta: float = 0.002   # 0.2% increase over full phase
	progressive_threshold: bool = True  # Enable progressive threshold within phase
	# CE percentile filter: keep only neighbors in top X% by CE (None = disabled)
	# Example: 0.75 keeps top 75% by CE. Applied after accuracy threshold.
	ce_percentile: Optional[float] = None
	# Fitness calculator: how to combine CE and accuracy for ranking
	# CE = pure CE ranking, uses dual paths (25 neighbors from best_ce + 25 from best_acc)
	# HARMONIC_RANK = harmonic mean of ranks (default), uses single path (50 from best_harmonic)
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	# Weights for HARMONIC_RANK mode (higher weight = more important)
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	# Early stopping (all configurable via parameters)
	patience: int = 5              # Checks without improvement before stopping
	check_interval: int = 5        # Check every N iterations
	min_improvement_pct: float = 0.5  # TS is focused, higher threshold (0.5%)


class GenericGAStrategy(ABC, Generic[T]):
	"""
	Generic Genetic Algorithm strategy.

	Subclasses must implement genome operations:
	- clone_genome: Copy a genome
	- mutate_genome: Generate a mutated variant
	- crossover_genomes: Combine two parents
	- create_random_genome: Create a new random genome

	The core GA loop (selection, crossover, mutation, elitism) is implemented here.
	"""

	def __init__(
		self,
		config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		log_level: int = logging.DEBUG,
	):
		self._config = config or GAConfig()
		self._seed = seed
		# Use OptimizationLogger with optional file logging (DEBUG+ goes to file)
		self._log = OptimizationLogger(self.name, level=log_level, file_logger=logger)
		self._rng: Optional[random.Random] = None

	@property
	def config(self) -> GAConfig:
		return self._config

	@property
	def name(self) -> str:
		return "GenericGA"

	def _ensure_rng(self) -> None:
		if self._rng is None:
			self._rng = random.Random(self._seed)

	# =========================================================================
	# Abstract genome operations - subclasses must implement
	# =========================================================================

	@abstractmethod
	def clone_genome(self, genome: T) -> T:
		"""Create a deep copy of the genome."""
		...

	@abstractmethod
	def mutate_genome(self, genome: T, mutation_rate: float) -> T:
		"""Create a mutated variant of the genome."""
		...

	@abstractmethod
	def crossover_genomes(self, parent1: T, parent2: T) -> T:
		"""Create a child by combining two parents."""
		...

	@abstractmethod
	def create_random_genome(self) -> T:
		"""Create a new random genome (for population initialization)."""
		...

	# =========================================================================
	# Core GA loop
	# =========================================================================

	def optimize(
		self,
		evaluate_fn: Callable[[T], float],
		initial_genome: Optional[T] = None,
		initial_population: Optional[list[T]] = None,
		batch_evaluate_fn: Optional[Callable[[list[T]], list[tuple[float, float]]]] = None,
		overfitting_callback: Optional[Callable[[T, float], Any]] = None,
	) -> OptimizerResult[T]:
		"""
		Run Genetic Algorithm optimization.

		Args:
			evaluate_fn: Function to evaluate a single genome (lower is better)
			initial_genome: Optional seed genome (used if no initial_population)
			initial_population: Optional seed population from previous phase
			batch_evaluate_fn: Optional batch evaluation function returning list[(CE, accuracy)]
			overfitting_callback: Optional callback for overfitting detection.
				Called every check_interval generations with (best_genome, best_fitness).
				Returns OverfittingControl (or any object with early_stop attribute).
				If early_stop=True, optimization stops with stop_reason=StopReason.OVERFITTING.

		Returns:
			OptimizerResult with best genome and statistics
		"""
		self._ensure_rng()
		cfg = self._config

		# Threshold continuity: use initial_threshold from config if set (passed from previous phase)
		# Otherwise, fall back to min_accuracy (first phase)
		start_threshold = cfg.initial_threshold if cfg.initial_threshold is not None else cfg.min_accuracy
		end_threshold = start_threshold + cfg.threshold_delta

		# Helper to get current threshold based on progress
		def get_threshold(progress: float = 0.0) -> float:
			if not cfg.progressive_threshold:
				return start_threshold
			progress = max(0.0, min(1.0, progress))
			return start_threshold + progress * cfg.threshold_delta

		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.2%} → {end_threshold:.2%}")

		# Build initial population with viable candidates only (accuracy >= threshold at start)
		initial_threshold = get_threshold(0.0)
		if initial_population:
			self._log.info(f"[{self.name}] Seeding population from {len(initial_population)} genomes")
			# Generator creates mutations of best seed genome
			def seed_generator() -> T:
				return self.mutate_genome(self.clone_genome(initial_population[0]), cfg.mutation_rate * 3)
			population = self._build_viable_population(
				target_size=cfg.population_size,
				generator_fn=seed_generator,
				batch_fn=batch_evaluate_fn,
				single_fn=evaluate_fn,
				min_accuracy=initial_threshold,
				seed_genomes=initial_population,
			)
		elif initial_genome is not None:
			# Generator creates mutations of seed genome
			def single_seed_generator() -> T:
				return self.mutate_genome(self.clone_genome(initial_genome), cfg.mutation_rate * 3)
			population = self._build_viable_population(
				target_size=cfg.population_size,
				generator_fn=single_seed_generator,
				batch_fn=batch_evaluate_fn,
				single_fn=evaluate_fn,
				min_accuracy=initial_threshold,
				seed_genomes=[initial_genome],
			)
		else:
			# Random initialization
			population = self._build_viable_population(
				target_size=cfg.population_size,
				generator_fn=self.create_random_genome,
				batch_fn=batch_evaluate_fn,
				single_fn=evaluate_fn,
				min_accuracy=initial_threshold,
			)

		fitness_values = [f for _, f, _ in population]
		accuracy_values = [a for _, _, a in population]

		# Find initial best
		best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
		best = self.clone_genome(population[best_idx][0])
		best_fitness = fitness_values[best_idx]
		initial_fitness = fitness_values[0] if initial_genome else best_fitness
		initial_accuracy = accuracy_values[best_idx]

		history = [(0, best_fitness)]

		# Initialize early stopping tracker
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stopper = EarlyStoppingTracker(early_stop_config, self._log, self.name)
		early_stopper.reset(best_fitness)

		# Initialize adaptive scaler for dynamic parameter adjustment
		adaptive_scaler = AdaptiveScaler(
			base_population=cfg.population_size,
			base_mutation=cfg.mutation_rate,
			name=self.name,
		)

		# Track initial diversity (CE spread)
		initial_ce_spread = max(fitness_values) - min(fitness_values) if fitness_values else 0.0

		# Log config and initial best
		self._log.info(f"[{self.name}] Config: pop={cfg.population_size}, gens={cfg.generations}, "
					   f"elitism={cfg.elitism_pct:.0%} per metric, "
					   f"patience={cfg.patience}, check_interval={cfg.check_interval}, min_delta={cfg.min_improvement_pct}%")
		self._log.info(f"[{self.name}] Initial best: {best_fitness:.4f}, diversity (CE spread): {initial_ce_spread:.4f}")

		# Tracking for analysis
		elite_wins = 0  # Iterations where elite beat new offspring
		improved_iterations = 0
		# Track elites from first generation for survival analysis
		initial_elite_genomes = None  # Will be set after first generation
		# Track progressive threshold changes
		prev_threshold: Optional[float] = None

		generation = 0
		for generation in range(cfg.generations):
			# Selection and reproduction
			new_population: list[tuple[T, Optional[float], Optional[float]]] = []

			# Dual elitism: keep top N% by CE AND top N% by accuracy (unique)
			# Use target population_size (not len(population)) so we don't lose elites
			# when we couldn't generate enough viable candidates
			n_per_metric = max(1, int(cfg.population_size * cfg.elitism_pct))

			# Top by CE (lower is better)
			ce_sorted = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
			ce_elite_indices = set(ce_sorted[:n_per_metric])

			# Top by accuracy (higher is better) - MUTUALLY EXCLUSIVE from CE elites
			has_accuracy = any(a is not None for a in accuracy_values)
			if has_accuracy:
				# Exclude CE-selected indices when selecting by accuracy
				acc_candidates = [i for i in range(len(accuracy_values)) if i not in ce_elite_indices]
				acc_sorted = sorted(acc_candidates, key=lambda i: -(accuracy_values[i] or 0))
				acc_elite_indices = set(acc_sorted[:n_per_metric])
			else:
				acc_elite_indices = set()

			# Combine elites (mutually exclusive, no overlap)
			all_elite_indices = list(ce_elite_indices) + list(acc_elite_indices)
			total_elites = len(all_elite_indices)

			# Track initial elites (first generation only) for survival analysis
			if generation == 0:
				initial_elite_genomes = [
					(self.clone_genome(population[idx][0]), fitness_values[idx])
					for idx in all_elite_indices
				]
				# Log elite composition (INFO level)
				self._log.info(f"[{self.name}] Elitism: {len(ce_elite_indices)} by CE + "
							   f"{len(acc_elite_indices)} by Acc = {total_elites} unique elites")

			# Add elites to new population
			elite_width = len(str(total_elites))
			for i, elite_idx in enumerate(all_elite_indices):
				elite_genome = self.clone_genome(population[elite_idx][0])
				elite_fitness = fitness_values[elite_idx]
				elite_accuracy = accuracy_values[elite_idx]
				new_population.append((elite_genome, elite_fitness, elite_accuracy))

				# Log elite source (DEBUG level)
				source = "CE" if elite_idx in ce_elite_indices else "Acc"
				acc_str = f", Acc={elite_accuracy:.2%}" if elite_accuracy is not None else ""
				self._log.debug(f"[Elite {i + 1:0{elite_width}d}/{total_elites}] CE={elite_fitness:.4f}{acc_str} ({source})")

			# Generate viable offspring to fill the rest of the population
			needed_offspring = cfg.population_size - len(new_population)

			def offspring_generator() -> T:
				# Tournament selection and crossover/mutation
				p1 = self._tournament_select(population)
				p2 = self._tournament_select(population)
				if self._rng.random() < cfg.crossover_rate:
					child = self.crossover_genomes(p1, p2)
				else:
					child = self.clone_genome(p1)
				return self.mutate_genome(child, cfg.mutation_rate)

			# Progressive threshold: gets stricter as generations progress
			current_threshold = get_threshold(generation / cfg.generations)
			# Only log if formatted values differ (avoid noise from tiny internal differences)
			if prev_threshold is not None and f"{prev_threshold:.4%}" != f"{current_threshold:.4%}":
				self._log.debug(f"[{self.name}] Threshold changed: {prev_threshold:.4%} → {current_threshold:.4%}")
			prev_threshold = current_threshold
			offspring = self._build_viable_population(
				target_size=needed_offspring,
				generator_fn=offspring_generator,
				batch_fn=batch_evaluate_fn,
				single_fn=evaluate_fn,
				min_accuracy=current_threshold,
			)

			# Combine elites with viable offspring
			population = new_population + offspring
			fitness_values = [f for _, f, _ in population]
			accuracy_values = [a for _, _, a in population]

			# Update best
			gen_best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
			if fitness_values[gen_best_idx] < best_fitness:
				best = self.clone_genome(population[gen_best_idx][0])
				best_fitness = fitness_values[gen_best_idx]

			history.append((generation + 1, best_fitness))

			# Compute new_best (best among new candidates, excluding elites)
			new_candidate_fitness = fitness_values[total_elites:]  # Non-elite fitness values
			new_best = min(new_candidate_fitness) if new_candidate_fitness else float('inf')

			# Track elite wins: did any elite beat the best new candidate?
			elite_fitness = fitness_values[:total_elites]
			best_elite = min(elite_fitness) if elite_fitness else float('inf')
			if best_elite <= new_best:
				elite_wins += 1

			# Track improvement
			prev_best = history[-2][1] if len(history) >= 2 else history[-1][1]
			if best_fitness < prev_best:
				improved_iterations += 1

			# Log progress
			gen_avg = sum(fitness_values) / len(fitness_values)
			gen_width = len(str(cfg.generations))
			self._log.info(f"[{self.name}] Gen {generation + 1:0{gen_width}d}/{cfg.generations}: "
						   f"best={best_fitness:.4f}, new_best={new_best:.4f}, avg={gen_avg:.4f}")

			# Early stopping check (checks at configured intervals)
			if early_stopper.check(generation, best_fitness):
				break

			# Adaptive parameter scaling based on health status
			adaptive_scaler.update(early_stopper.current_level)
			if adaptive_scaler.level_changed:
				adaptive_scaler.log_transition(self._log)
				old_pop_size = cfg.population_size
				cfg.population_size = adaptive_scaler.population
				cfg.mutation_rate = adaptive_scaler.mutation_rate

				# Adjust population size if needed
				if cfg.population_size > old_pop_size:
					# Need more individuals - generate random ones
					needed = cfg.population_size - len(population)
					if needed > 0:
						new_individuals = self._build_viable_population(
							target_size=needed,
							generator_fn=self.create_random_genome,
							batch_fn=batch_evaluate_fn,
							single_fn=evaluate_fn,
							min_accuracy=current_threshold,
						)
						population.extend(new_individuals)
						fitness_values.extend([f for _, f, _ in new_individuals])
						accuracy_values.extend([a for _, _, a in new_individuals])
				elif cfg.population_size < old_pop_size:
					# Shrink population - keep best
					combined = list(zip(population, fitness_values, accuracy_values))
					combined.sort(key=lambda x: x[1])  # Sort by fitness
					combined = combined[:cfg.population_size]
					population = [p for p, _, _ in combined]
					fitness_values = [f for _, f, _ in combined]
					accuracy_values = [a for _, _, a in combined]

			# Overfitting callback check (same interval as early stopping)
			if overfitting_callback is not None and (generation + 1) % cfg.check_interval == 0:
				control = overfitting_callback(best, best_fitness)
				if hasattr(control, 'early_stop') and control.early_stop:
					self._log.warning(f"[{self.name}] Overfitting early stop at gen {generation + 1}")
					# Return early with overfitting stop reason
					final_population = [self.clone_genome(g) for g, _, _ in sorted(population, key=lambda x: x[1])]
					improvement_pct = (initial_fitness - best_fitness) / initial_fitness * 100 if initial_fitness > 0 else 0
					# Compute final_threshold at current progress for next phase continuity
					current_final_threshold = get_threshold(generation / cfg.generations)
					return OptimizerResult(
						initial_genome=initial_genome if initial_genome else population[0][0],
						best_genome=best,
						initial_fitness=initial_fitness,
						final_fitness=best_fitness,
						improvement_percent=improvement_pct,
						iterations_run=generation + 1,
						method_name=self.name,
						history=history,
						early_stopped=True,
						stop_reason=StopReason.OVERFITTING,
						final_population=final_population,
						initial_accuracy=initial_accuracy,
						final_accuracy=accuracy_values[gen_best_idx] if accuracy_values else None,
						final_threshold=current_final_threshold,
					)

		# Get final accuracy from best genome's cached accuracy
		best_idx_final = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
		final_accuracy = accuracy_values[best_idx_final] if accuracy_values else None

		# Extract final population for seeding next phase (sorted by fitness)
		final_population = [self.clone_genome(g) for g, _, _ in sorted(population, key=lambda x: x[1])]

		# Compute final diversity
		final_ce_spread = max(fitness_values) - min(fitness_values) if fitness_values else 0.0

		# Count elite survivals (how many initial elites made it to final population)
		elite_survivals = 0
		if initial_elite_genomes:
			final_fitness_set = set(f for _, f, _ in population)
			for _, elite_fit in initial_elite_genomes:
				if elite_fit in final_fitness_set:
					elite_survivals += 1

		# Log analysis summary
		total_gens = generation + 1
		elite_win_rate = elite_wins / total_gens * 100 if total_gens > 0 else 0
		improvement_rate = improved_iterations / total_gens * 100 if total_gens > 0 else 0
		diversity_change = final_ce_spread - initial_ce_spread

		# Compute final threshold for next phase continuity
		final_threshold = get_threshold(generation / cfg.generations) if cfg.generations > 0 else start_threshold

		self._log.info(f"\n[{self.name}] Analysis Summary:")
		self._log.info(f"  CE improvement: {initial_fitness:.4f} → {best_fitness:.4f} ({(1 - best_fitness/initial_fitness)*100:+.2f}%)")
		self._log.info(f"  CE spread: {initial_ce_spread:.4f} → {final_ce_spread:.4f} ({diversity_change:+.4f})")
		self._log.info(f"  Elite survivals: {elite_survivals}/{len(initial_elite_genomes) if initial_elite_genomes else 0}")
		self._log.info(f"  Elite win rate: {elite_wins}/{total_gens} ({elite_win_rate:.1f}%)")
		self._log.info(f"  Improvement rate: {improved_iterations}/{total_gens} ({improvement_rate:.1f}%)")
		self._log.info(f"  Final threshold: {final_threshold:.2%} (for next phase)")

		improvement_pct = (initial_fitness - best_fitness) / initial_fitness * 100 if initial_fitness > 0 else 0

		return OptimizerResult(
			initial_genome=initial_genome if initial_genome else population[0][0],
			best_genome=best,
			initial_fitness=initial_fitness,
			final_fitness=best_fitness,
			improvement_percent=improvement_pct,
			iterations_run=generation + 1,
			method_name=self.name,
			history=history,
			early_stopped=early_stopper.patience_exhausted,
			stop_reason=StopReason.CONVERGENCE if early_stopper.patience_exhausted else None,
			final_population=final_population,
			initial_accuracy=initial_accuracy,
			final_accuracy=final_accuracy,
			final_threshold=final_threshold,
		)

	def _evaluate_population(
		self,
		population: list[tuple[T, Optional[float], Optional[float]]],
		batch_fn: Optional[Callable[[list[T]], list[tuple[float, float]]]],
		single_fn: Callable[[T], float],
		generation: int = 0,
		total_generations: int = 0,
	) -> list[tuple[T, float, Optional[float]]]:
		"""
		Evaluate individuals with None fitness, tracking accuracy.

		Args:
			population: List of (genome, fitness, accuracy) tuples
			batch_fn: Optional batch evaluation function returning list[(CE, accuracy)]
			single_fn: Single genome evaluation function (CE only, for fallback)
			generation: Current generation (0-indexed, for logging)
			total_generations: Total generations (for logging)

		Returns:
			Updated population with fitness and accuracy filled in
		"""
		unknown_indices = [i for i, (_, f, _) in enumerate(population) if f is None]

		if not unknown_indices:
			return [(g, f, a) for g, f, a in population]  # All cached

		to_eval = [population[i][0] for i in unknown_indices]

		# Batch evaluate - returns (CE, accuracy) tuples
		if batch_fn is not None:
			results = batch_fn(to_eval)
			new_fitness = [ce for ce, _ in results]
			new_accuracy = [acc for _, acc in results]
		else:
			# Fallback to single evaluation (no accuracy)
			new_fitness = [single_fn(g) for g in to_eval]
			new_accuracy = [None] * len(to_eval)

		# Note: Real-time per-genome logging happens in evaluate_batch (adaptive_cluster.py)
		# with timing info. We don't duplicate logging here.

		result = list(population)
		for idx, fit, acc in zip(unknown_indices, new_fitness, new_accuracy):
			result[idx] = (result[idx][0], fit, acc)

		return result

	def _build_viable_population(
		self,
		target_size: int,
		generator_fn: Callable[[], T],
		batch_fn: Optional[Callable[[list[T]], list[tuple[float, float]]]],
		single_fn: Callable[[T], float],
		min_accuracy: float,
		seed_genomes: Optional[list[T]] = None,
		max_attempts: int = 10,
	) -> list[tuple[T, float, Optional[float]]]:
		"""
		Build a population of viable candidates (accuracy >= min_accuracy).

		Generates candidates, evaluates them, and keeps only viable ones.
		Continues until target_size is reached or max_attempts exceeded.

		Args:
			target_size: Number of viable candidates needed
			generator_fn: Function to generate a new random genome
			batch_fn: Batch evaluation function returning list[(CE, accuracy)]
			single_fn: Single evaluation function (fallback)
			min_accuracy: Minimum accuracy threshold (0.0001 = 0.01%)
			seed_genomes: Optional seed genomes to include (evaluated first)
			max_attempts: Maximum generation attempts before giving up

		Returns:
			List of (genome, fitness, accuracy) tuples for viable candidates
		"""
		viable: list[tuple[T, float, Optional[float]]] = []
		filtered_count = 0

		# First, evaluate seed genomes if provided
		if seed_genomes:
			to_eval = [self.clone_genome(g) for g in seed_genomes[:target_size]]
			if batch_fn is not None:
				results = batch_fn(to_eval, min_accuracy=min_accuracy)
				for genome, (ce, acc) in zip(to_eval, results):
					if acc is None or acc >= min_accuracy:
						viable.append((genome, ce, acc))
					else:
						filtered_count += 1
			else:
				for genome in to_eval:
					ce = single_fn(genome)
					viable.append((genome, ce, None))

		# Generate new candidates until we have enough
		attempt = 0
		while len(viable) < target_size and attempt < max_attempts:
			attempt += 1
			needed = target_size - len(viable)
			# Generate a batch of candidates (extra to account for filtering)
			batch_size = min(needed * 2, needed + 10)  # Generate extra
			candidates = [generator_fn() for _ in range(batch_size)]

			# Evaluate
			if batch_fn is not None:
				results = batch_fn(candidates, min_accuracy=min_accuracy)
				for genome, (ce, acc) in zip(candidates, results):
					if acc is None or acc >= min_accuracy:
						viable.append((genome, ce, acc))
						if len(viable) >= target_size:
							break
					else:
						filtered_count += 1
			else:
				for genome in candidates:
					ce = single_fn(genome)
					viable.append((genome, ce, None))
					if len(viable) >= target_size:
						break

		if filtered_count > 0:
			self._log.trace(f"[{self.name}] Filtered {filtered_count} candidates with accuracy < {min_accuracy:.2%}")

		if len(viable) < target_size:
			self._log.warning(f"[{self.name}] Warning: only {len(viable)}/{target_size} viable candidates after {max_attempts} attempts")

		return viable[:target_size]

	def _tournament_select(self, population: list[tuple[T, float, Optional[float]]], tournament_size: int = 3) -> T:
		"""Tournament selection: pick best from random subset."""
		indices = self._rng.sample(range(len(population)), min(tournament_size, len(population)))
		best_idx = min(indices, key=lambda i: population[i][1])
		return population[best_idx][0]


class GenericTSStrategy(ABC, Generic[T]):
	"""
	Generic Tabu Search strategy.

	Subclasses must implement genome operations:
	- clone_genome: Copy a genome
	- mutate_genome: Generate a neighbor with move info
	- is_tabu_move: Check if a move reverses a tabu move

	The core TS loop (neighbor generation, tabu filtering, selection) is implemented here.
	"""

	def __init__(
		self,
		config: Optional[TSConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		log_level: int = logging.DEBUG,
	):
		self._config = config or TSConfig()
		self._seed = seed
		# Use OptimizationLogger with optional file logging (DEBUG+ goes to file)
		self._log = OptimizationLogger(self.name, level=log_level, file_logger=logger)
		self._rng: Optional[random.Random] = None

	@property
	def config(self) -> TSConfig:
		return self._config

	@property
	def name(self) -> str:
		return "GenericTS"

	def _ensure_rng(self) -> None:
		if self._rng is None:
			self._rng = random.Random(self._seed)

	# =========================================================================
	# Abstract genome operations - subclasses must implement
	# =========================================================================

	@abstractmethod
	def clone_genome(self, genome: T) -> T:
		"""Create a deep copy of the genome."""
		...

	@abstractmethod
	def mutate_genome(self, genome: T, mutation_rate: float) -> tuple[T, Any]:
		"""Create a neighbor. Returns (new_genome, move_info)."""
		...

	@abstractmethod
	def is_tabu_move(self, move: Any, tabu_list: list[Any]) -> bool:
		"""Check if a move is tabu (reverses a recent move)."""
		...

	# =========================================================================
	# Core TS loop
	# =========================================================================

	def optimize(
		self,
		initial_genome: T,
		initial_fitness: float,
		evaluate_fn: Callable[[T], float],
		initial_neighbors: Optional[list[T]] = None,
		batch_evaluate_fn: Optional[Callable[[list[T]], list[tuple[float, float]]]] = None,
		overfitting_callback: Optional[Callable[[T, float], Any]] = None,
	) -> OptimizerResult[T]:
		"""
		Run Tabu Search optimization with dual-path CE/Acc optimization.

		Each iteration:
		- Generate N/2 neighbors from best_ce, pick best by CE
		- Generate N/2 neighbors from best_acc, pick best by Acc
		- Maintain total_neighbors cache: top K/2 by CE + top K/2 by Acc

		Args:
			initial_genome: Starting genome
			initial_fitness: Fitness of initial genome
			evaluate_fn: Function to evaluate a single genome
			initial_neighbors: Optional seed neighbors from previous phase
			batch_evaluate_fn: Optional batch evaluation function returning list[(CE, accuracy)]
			overfitting_callback: Optional callback for overfitting detection.

		Returns:
			OptimizerResult with best genome, statistics, and final_threshold for next phase
		"""
		self._ensure_rng()
		cfg = self._config

		# Threshold continuity: use initial_threshold from config if set (passed from previous phase)
		# Otherwise, fall back to min_accuracy (first phase)
		start_threshold = cfg.initial_threshold if cfg.initial_threshold is not None else cfg.min_accuracy
		end_threshold = start_threshold + cfg.threshold_delta

		# Helper to get current threshold based on progress
		def get_threshold(progress: float = 0.0) -> float:
			if not cfg.progressive_threshold:
				return start_threshold
			progress = max(0.0, min(1.0, progress))
			return start_threshold + progress * cfg.threshold_delta

		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.2%} → {end_threshold:.2%}")

		# Cache size for total_neighbors (top K/2 by CE + top K/2 by Acc)
		cache_size = cfg.total_neighbors_size or cfg.neighbors_per_iter
		cache_size_per_metric = cache_size // 2

		# Dual-path tracking: best by CE and best by Acc
		best_ce_genome = self.clone_genome(initial_genome)
		best_ce_fitness = initial_fitness
		best_ce_accuracy: Optional[float] = None

		best_acc_genome = self.clone_genome(initial_genome)
		best_acc_fitness = initial_fitness
		best_acc_accuracy: Optional[float] = None

		# Global best (by CE, for return value)
		best = self.clone_genome(initial_genome)
		best_fitness = initial_fitness
		best_accuracy: Optional[float] = None
		start_fitness = initial_fitness

		# Tabu lists (separate for CE and Acc paths to allow independent exploration)
		tabu_list_ce: deque = deque(maxlen=cfg.tabu_size)
		tabu_list_acc: deque = deque(maxlen=cfg.tabu_size)

		# Total neighbors cache: all genomes with their CE and Acc
		# Will be split into top K/2 by CE + top K/2 by Acc at the end
		all_neighbors: list[tuple[T, float, Optional[float]]] = [
			(self.clone_genome(initial_genome), initial_fitness, None)
		]

		# Initial threshold
		current_threshold = get_threshold(0.0)

		# Seed with initial neighbors if provided
		if initial_neighbors:
			self._log.info(f"[{self.name}] Seeding from {len(initial_neighbors)} neighbors")
			if batch_evaluate_fn is not None:
				results = batch_evaluate_fn(initial_neighbors, min_accuracy=current_threshold)
				seed_fitness = [ce for ce, _ in results]
				seed_accuracy = [acc for _, acc in results]
			else:
				seed_fitness = [evaluate_fn(g) for g in initial_neighbors]
				seed_accuracy = [None] * len(initial_neighbors)

			# Add all seed neighbors to the cache
			for g, f, a in zip(initial_neighbors, seed_fitness, seed_accuracy):
				all_neighbors.append((self.clone_genome(g), f, a))

			# Find best CE and best Acc from seeds
			best_ce_idx = min(range(len(seed_fitness)), key=lambda i: seed_fitness[i])
			if seed_fitness[best_ce_idx] < best_ce_fitness:
				best_ce_genome = self.clone_genome(initial_neighbors[best_ce_idx])
				best_ce_fitness = seed_fitness[best_ce_idx]
				best_ce_accuracy = seed_accuracy[best_ce_idx]

			# Find best by Acc (only consider those with accuracy)
			acc_with_values = [(i, a) for i, a in enumerate(seed_accuracy) if a is not None]
			if acc_with_values:
				best_acc_idx = max(acc_with_values, key=lambda x: x[1])[0]
				if seed_accuracy[best_acc_idx] is not None and (best_acc_accuracy is None or seed_accuracy[best_acc_idx] > best_acc_accuracy):
					best_acc_genome = self.clone_genome(initial_neighbors[best_acc_idx])
					best_acc_fitness = seed_fitness[best_acc_idx]
					best_acc_accuracy = seed_accuracy[best_acc_idx]

			# Update global best
			if best_ce_fitness < best_fitness:
				best = self.clone_genome(best_ce_genome)
				best_fitness = best_ce_fitness
				best_accuracy = best_ce_accuracy

			# Log seed summary
			viable = [(f, a) for f, a in zip(seed_fitness, seed_accuracy) if a is None or a >= current_threshold]
			seed_best = min(f for f, _ in viable) if viable else min(seed_fitness)
			seed_avg = sum(f for f, _ in viable) / len(viable) if viable else sum(seed_fitness) / len(seed_fitness)
			self._log.info(f"[{self.name}] Seed: best_ce={best_ce_fitness:.4f}, best_acc={best_acc_accuracy:.2%}" if best_acc_accuracy else
						   f"[{self.name}] Seed: best_ce={best_ce_fitness:.4f}, best_acc=N/A")

		history = [(0, best_fitness)]

		# Analysis tracking
		ce_improved = 0
		acc_improved = 0
		improved_iterations = 0

		# Initialize early stopping tracker
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stopper = EarlyStoppingTracker(early_stop_config, self._log, self.name)
		early_stopper.reset(best_fitness)

		# Log config
		neighbors_per_path = cfg.neighbors_per_iter // 2
		self._log.info(f"[{self.name}] Config: neighbors={cfg.neighbors_per_iter} ({neighbors_per_path} CE + {neighbors_per_path} Acc), "
					   f"iters={cfg.iterations}, cache={cache_size} ({cache_size_per_metric} CE + {cache_size_per_metric} Acc)")

		# Track threshold changes
		prev_threshold: Optional[float] = None

		iteration = 0
		for iteration in range(cfg.iterations):
			# Progressive threshold
			current_threshold = get_threshold(iteration / cfg.iterations)
			# Only log if formatted values differ (avoid noise from tiny internal differences)
			if prev_threshold is not None and f"{prev_threshold:.4%}" != f"{current_threshold:.4%}":
				self._log.debug(f"[{self.name}] Threshold changed: {prev_threshold:.4%} → {current_threshold:.4%}")
			prev_threshold = current_threshold

			# === Path A: Generate neighbors from best_ce ===
			ce_candidates: list[tuple[T, Any]] = []
			for _ in range(neighbors_per_path):
				neighbor, move = self.mutate_genome(self.clone_genome(best_ce_genome), cfg.mutation_rate)
				if not self.is_tabu_move(move, list(tabu_list_ce)):
					ce_candidates.append((neighbor, move))

			# === Path B: Generate neighbors from best_acc ===
			acc_candidates: list[tuple[T, Any]] = []
			for _ in range(neighbors_per_path):
				neighbor, move = self.mutate_genome(self.clone_genome(best_acc_genome), cfg.mutation_rate)
				if not self.is_tabu_move(move, list(tabu_list_acc)):
					acc_candidates.append((neighbor, move))

			# Combine and evaluate all candidates
			all_candidates = ce_candidates + acc_candidates
			if not all_candidates:
				continue

			if batch_evaluate_fn is not None:
				to_eval = [n for n, _ in all_candidates]
				results = batch_evaluate_fn(to_eval, min_accuracy=current_threshold)
				fitness_values = [ce for ce, _ in results]
				accuracy_values = [acc for _, acc in results]
			else:
				fitness_values = [evaluate_fn(n) for n, _ in all_candidates]
				accuracy_values = [None] * len(all_candidates)

			# Build evaluated neighbors list
			evaluated = [(n, m, f, a) for (n, m), f, a in zip(all_candidates, fitness_values, accuracy_values)]

			# Filter by threshold
			viable = [(n, m, f, a) for n, m, f, a in evaluated if a is None or a >= current_threshold]
			if not viable:
				self._log.warning(f"[{self.name}] All neighbors filtered, keeping best by CE")
				viable = sorted(evaluated, key=lambda x: x[2])[:1]

			# Add all viable to the total neighbors cache
			for n, _, f, a in viable:
				all_neighbors.append((self.clone_genome(n), f, a))

			# === Update best_ce: find best by CE among CE-path neighbors ===
			ce_evaluated = evaluated[:len(ce_candidates)]
			if ce_evaluated:
				ce_best_idx = min(range(len(ce_evaluated)), key=lambda i: ce_evaluated[i][2])
				n, m, f, a = ce_evaluated[ce_best_idx]
				if f < best_ce_fitness:
					best_ce_genome = self.clone_genome(n)
					best_ce_fitness = f
					best_ce_accuracy = a
					ce_improved += 1
				if m is not None:
					tabu_list_ce.append(m)

			# === Update best_acc: find best by Acc among Acc-path neighbors ===
			acc_evaluated = evaluated[len(ce_candidates):]
			if acc_evaluated:
				acc_with_values = [(i, x[3]) for i, x in enumerate(acc_evaluated) if x[3] is not None]
				if acc_with_values:
					acc_best_idx = max(acc_with_values, key=lambda x: x[1])[0]
					n, m, f, a = acc_evaluated[acc_best_idx]
					if best_acc_accuracy is None or (a is not None and a > best_acc_accuracy):
						best_acc_genome = self.clone_genome(n)
						best_acc_fitness = f
						best_acc_accuracy = a
						acc_improved += 1
					if m is not None:
						tabu_list_acc.append(m)

			# Update global best (by CE)
			prev_best = best_fitness
			if best_ce_fitness < best_fitness:
				best = self.clone_genome(best_ce_genome)
				best_fitness = best_ce_fitness
				best_accuracy = best_ce_accuracy
				improved_iterations += 1

			history.append((iteration + 1, best_fitness))

			# Log progress
			iter_width = len(str(cfg.iterations))
			acc_str = f"{best_acc_accuracy:.2%}" if best_acc_accuracy is not None else "N/A"
			self._log.info(f"[{self.name}] Iter {iteration + 1:0{iter_width}d}/{cfg.iterations}: "
						   f"best_ce={best_fitness:.4f}, best_acc={acc_str}")

			# Early stopping check
			if early_stopper.check(iteration, best_fitness):
				break

			# Overfitting callback
			if overfitting_callback is not None and (iteration + 1) % cfg.check_interval == 0:
				control = overfitting_callback(best, best_fitness)
				if hasattr(control, 'early_stop') and control.early_stop:
					self._log.warning(f"[{self.name}] Overfitting early stop at iter {iteration + 1}")
					break

		# === Build final_population: top K/2 by CE + top K/2 by Acc (mutually exclusive) ===
		# Sort by CE, take top K/2
		by_ce = sorted(all_neighbors, key=lambda x: x[1])
		top_by_ce = by_ce[:cache_size_per_metric]
		top_ce_set = set(id(g) for g, _, _ in top_by_ce)

		# Sort by Acc (descending), take top K/2 excluding those already in CE set
		by_acc = sorted(
			[(g, f, a) for g, f, a in all_neighbors if a is not None and id(g) not in top_ce_set],
			key=lambda x: -x[2]  # type: ignore
		)
		top_by_acc = by_acc[:cache_size_per_metric]

		# Combine: top K/2 by CE + top K/2 by Acc
		final_population_with_metrics = top_by_ce + top_by_acc
		final_population = [self.clone_genome(g) for g, _, _ in final_population_with_metrics]

		# Final threshold (for next phase)
		final_threshold = get_threshold(iteration / cfg.iterations) if cfg.iterations > 0 else start_threshold

		# Log analysis summary
		total_iters = iteration + 1
		self._log.info(f"\n[{self.name}] Analysis Summary:")
		self._log.info(f"  CE improvement: {start_fitness:.4f} → {best_fitness:.4f} ({(1 - best_fitness/start_fitness)*100:+.2f}%)")
		self._log.info(f"  CE path improvements: {ce_improved}/{total_iters}")
		self._log.info(f"  Acc path improvements: {acc_improved}/{total_iters}")
		self._log.info(f"  Final population: {len(top_by_ce)} by CE + {len(top_by_acc)} by Acc = {len(final_population)}")
		self._log.info(f"  Final threshold: {final_threshold:.2%} (for next phase)")

		improvement_pct = (start_fitness - best_fitness) / start_fitness * 100 if start_fitness > 0 else 0

		return OptimizerResult(
			initial_genome=initial_genome,
			best_genome=best,
			initial_fitness=start_fitness,
			final_fitness=best_fitness,
			improvement_percent=improvement_pct,
			iterations_run=iteration + 1,
			method_name=self.name,
			history=history,
			early_stopped=early_stopper.patience_exhausted,
			stop_reason=StopReason.CONVERGENCE if early_stopper.patience_exhausted else None,
			final_population=final_population,
			initial_accuracy=None,
			final_accuracy=best_accuracy,
			final_threshold=final_threshold,
		)
