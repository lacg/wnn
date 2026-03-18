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
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, Generic, Optional, TypeVar, Any

from wnn.ram.fitness import FitnessCalculatorType, FitnessCalculatorFactory

# Optional tracker integration
try:
	from wnn.ram.experiments.tracker import ExperimentTracker, TierConfig, GenomeConfig, GenomeRole
	HAS_TRACKER = True
except ImportError:
	HAS_TRACKER = False
	ExperimentTracker = None
	TierConfig = None
	GenomeConfig = None
	GenomeRole = None

# Generic genome type
T = TypeVar('T')

# Custom TRACE level (below DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


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

	def _flush(self) -> None:
		"""Flush all handlers to ensure output is visible immediately."""
		for handler in self._logger.handlers:
			handler.flush()

	def debug(self, msg: str) -> None:
		"""Log at DEBUG level (individual genome info)."""
		if self._logger.isEnabledFor(logging.DEBUG):
			if self._file_logger:
				self._file_logger(msg)  # file_logger handles stdout + file
			else:
				self._logger.debug(msg)
				self._flush()

	def info(self, msg: str) -> None:
		"""Log at INFO level (progress summaries)."""
		if self._logger.isEnabledFor(logging.INFO):
			if self._file_logger:
				self._file_logger(msg)
			else:
				self._logger.info(msg)
				self._flush()

	def warning(self, msg: str) -> None:
		"""Log at WARNING level."""
		if self._logger.isEnabledFor(logging.WARNING):
			if self._file_logger:
				self._file_logger(msg)
			else:
				self._logger.warning(msg)
				self._flush()

	def error(self, msg: str) -> None:
		"""Log at ERROR level."""
		if self._logger.isEnabledFor(logging.ERROR):
			if self._file_logger:
				self._file_logger(msg)
			else:
				self._logger.error(msg)
				self._flush()

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
	SHUTDOWN = auto()  # External shutdown request (e.g., flow cancelled)


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
	# Per-genome (CE, accuracy, f1?, fpr?) matching final_population order
	population_metrics: Optional[list[tuple]] = None
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

	def reset_baseline(self, initial_fitness: list[float]) -> None:
		"""
		Reset tracker for baseline-based overfitting and stagnation detection.

		Args:
			initial_fitness: Fitness values of top-K elites on FULL validation at init
		"""
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveLevel
		self._patience_counter = 0
		self._overfit_detector = OverfitDetector(initial_fitness)
		self._last_level = AdaptiveLevel.NEUTRAL
		# Initialize prev_health_mean for stagnation detection in check_health()
		self._prev_health_mean = self._overfit_detector.baseline_mean

	def check_overfit(self, iteration: int, current_fitness: list[float]) -> bool:
		"""
		Check overfitting by comparing current elites on FULL data vs baseline.

		Uses OverfitDetector to compute delta:
		Delta = (current_mean - baseline_mean) / baseline_mean × 100
		- Positive delta = overfitting (worse on full data than baseline)
		- Negative delta = generalizing (better on full data than baseline)

		Args:
			iteration: Current iteration (0-indexed)
			current_fitness: Fitness values of top-K elites on FULL validation NOW

		Returns:
			True if should stop, False otherwise
		"""
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveLevel
		from wnn.core.thresholds import OverfitThreshold
		cfg = self._config

		# Only check at specified intervals (1-indexed iteration)
		if (iteration + 1) % cfg.check_interval != 0:
			return False

		# Get detector
		detector = getattr(self, '_overfit_detector', None)
		if detector is None:
			return False

		# Compute delta using OverfitDetector
		delta_pct, current_mean = detector.tick_with_mean(current_fitness)

		# Check if within acceptable range (we want delta to stay low/negative)
		if delta_pct <= cfg.min_improvement_pct:
			# Within acceptable range, recover patience
			self._patience_counter = max(0, self._patience_counter - 1)
		else:
			# Overfitting detected
			self._patience_counter += 1

		# Determine level using OverfitThreshold
		if delta_pct < OverfitThreshold.HEALTHY:  # < -1% (improving a lot)
			level = AdaptiveLevel.HEALTHY
		elif delta_pct < OverfitThreshold.WARNING:  # -1% to 0% (stable/slight improve)
			level = AdaptiveLevel.NEUTRAL
		elif delta_pct < OverfitThreshold.CRITICAL:  # 0% to 3% (mild overfitting)
			level = AdaptiveLevel.WARNING
		else:  # >= 3% (severe overfitting)
			level = AdaptiveLevel.CRITICAL

		self._last_level = level

		# Log progress with delta vs baseline
		remaining = cfg.patience - self._patience_counter
		display = self._LEVEL_DISPLAY[level.name]
		top_k_count = detector.k
		baseline = detector.baseline_mean
		self._log(
			f"[{self._method_name}] Overfit check (top-{top_k_count} vs baseline): "
			f"mean={current_mean:.4f}, baseline={baseline:.4f}, Δ={delta_pct:+.2f}%, "
			f"patience={remaining}/{cfg.patience} {display}"
		)

		# Check if patience exhausted
		if self._patience_counter >= cfg.patience:
			total_iters = self._patience_counter * cfg.check_interval
			self._log(
				f"[{self._method_name}] Early stop: overfitting delta > {cfg.min_improvement_pct}% "
				f"for {total_iters} iterations"
			)
			return True

		return False

	def check_health(self, iteration: int, current_fitness: list[float]) -> bool:
		"""
		Unified health check combining overfitting detection AND stagnation detection.

		This method checks TWO conditions:
		1. Overfitting: delta vs baseline (is the model getting worse on full data?)
		2. Stagnation: improvement vs previous check (is the model still improving?)

		Both issues consume from the SAME patience counter. The status is determined
		by the WORST of the two conditions.

		Delta = (current_mean - baseline_mean) / baseline_mean × 100
		- Positive delta = overfitting (worse on full data than baseline)
		- Negative delta = generalizing (better on full data than baseline)

		Improvement = (prev_mean - current_mean) / prev_mean × 100
		- Positive improvement = getting better
		- Negative improvement = getting worse (stagnating/regressing)

		Args:
			iteration: Current iteration (0-indexed)
			current_fitness: Fitness values of top-K elites on FULL validation NOW

		Returns:
			True if should stop, False otherwise
		"""
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveLevel
		from wnn.core.thresholds import OverfitThreshold
		cfg = self._config

		# Only check at specified intervals (1-indexed iteration)
		if (iteration + 1) % cfg.check_interval != 0:
			return False

		# Get detector (for baseline delta)
		detector = getattr(self, '_overfit_detector', None)
		if detector is None:
			return False

		# Compute current mean
		if not current_fitness:
			return False
		current_mean = sum(current_fitness) / len(current_fitness)

		# === 1. Overfitting check: delta vs baseline ===
		delta_pct, _ = detector.tick_with_mean(current_fitness)

		# === 2. Stagnation check: delta vs previous check ===
		# On first check, use baseline as the reference (not 0%)
		prev_mean = getattr(self, '_prev_health_mean', None)
		if prev_mean is None:
			prev_mean = detector.baseline_mean  # First check compares to baseline
		if prev_mean is not None and prev_mean > 0:
			# delta_prev = (current - prev) / prev * 100
			# Negative = improving (current < prev), Positive = getting worse
			delta_prev = (current_mean - prev_mean) / prev_mean * 100
		else:
			delta_prev = 0.0

		# Update prev_mean for next check
		self._prev_health_mean = current_mean

		# === Determine if there's a problem (for display only) ===
		# Simple logic: negative delta = good (improving), positive delta = bad
		# Problem 1: Overfitting (current worse than baseline on validation)
		overfit_problem = delta_pct > 0

		# Problem 2: Stagnation (not improving from previous check)
		# delta_prev >= 0 means current >= previous (not improving)
		stagnation_problem = delta_prev >= 0

		# === Determine level FIRST (used for both display AND patience) ===
		# Use the worst (highest) delta to determine level
		# Both delta_pct and delta_prev use same convention: negative=good, positive=bad
		worst_delta = max(delta_pct, delta_prev)

		if worst_delta < OverfitThreshold.HEALTHY:  # < -1% (improving a lot)
			level = AdaptiveLevel.HEALTHY
		elif worst_delta <= OverfitThreshold.WARNING:  # -1% to 0% inclusive (stable)
			level = AdaptiveLevel.NEUTRAL
		elif worst_delta < OverfitThreshold.CRITICAL:  # >0% to 3% (mild issues)
			level = AdaptiveLevel.WARNING
		else:  # >= 3% (severe issues)
			level = AdaptiveLevel.CRITICAL

		self._last_level = level

		# === Update patience based on level ===
		# HEALTHY: Significant improvement, recover patience
		# NEUTRAL: Stable, no change to patience
		# WARNING/CRITICAL: Issues detected, decrease patience
		if level == AdaptiveLevel.HEALTHY:
			self._patience_counter = max(0, self._patience_counter - 1)
		elif level == AdaptiveLevel.NEUTRAL:
			pass  # No change to patience
		else:  # WARNING or CRITICAL
			self._patience_counter += 1

		# === Log progress with BOTH metrics transparently ===
		remaining = cfg.patience - self._patience_counter
		display = self._LEVEL_DISPLAY[level.name]
		top_k_count = detector.k
		baseline = detector.baseline_mean

		# Build problem indicators
		problems = []
		if overfit_problem:
			problems.append("OVERFIT")
		if stagnation_problem:
			problems.append("STAGNATE")
		problem_str = f" [{'+'.join(problems)}]" if problems else ""

		self._log(
			f"[{self._method_name}] Health check (top-{top_k_count}): "
			f"mean={current_mean:.4f}, baseline={baseline:.4f}, "
			f"Δbase={delta_pct:+.4f}%, Δprev={delta_prev:+.4f}%, "
			f"patience={remaining}/{cfg.patience} {display}{problem_str}"
		)

		# Check if patience exhausted
		if self._patience_counter >= cfg.patience:
			total_iters = self._patience_counter * cfg.check_interval
			stop_reasons = []
			if overfit_problem:
				stop_reasons.append(f"overfitting (Δbase={delta_pct:+.4f}%)")
			if stagnation_problem:
				stop_reasons.append(f"stagnation (Δprev={delta_prev:+.4f}%)")
			reason_str = " and ".join(stop_reasons) if stop_reasons else "exhausted patience"
			self._log(
				f"[{self._method_name}] Early stop: {reason_str} "
				f"for {total_iters} iterations"
			)
			return True

		return False

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
		# Filter out None values before computing mean
		valid_fitness = [f for f in top_k_fitness if f is not None] if top_k_fitness else []
		self._prev_trend_mean = sum(valid_fitness) / len(valid_fitness) if valid_fitness else 0.0
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

		# Calculate current mean of top-K% (filter out None values)
		valid_fitness = [f for f in top_k_fitness if f is not None] if top_k_fitness else []
		if not valid_fitness:
			return False
		current_mean = sum(valid_fitness) / len(valid_fitness)

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
class OptimizationConfig:
	"""Shared configuration for all optimization strategies (GA, TS, etc.).

	Single source of truth for fitness ranking, threshold progression,
	early stopping, and percentile filtering.
	"""
	mutation_rate: float = 0.1
	# Threshold continuity: start threshold passed from previous phase
	initial_threshold: Optional[float] = None
	min_accuracy: float = 0.0
	threshold_delta: float = 0.01
	threshold_reference: int = 1000
	progressive_threshold: bool = True
	# Fitness percentile filter (None = disabled)
	fitness_percentile: Optional[float] = None
	# Fitness calculator: unified ranking for all selection/sorting
	# HARMONIC_RANK = harmonic mean of CE+Acc ranks (default)
	# CE = pure CE ranking
	# NORMALIZED = normalized [0,1] weighted sum
	# NORMALIZED_HARMONIC = normalized values with harmonic mean
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	fitness_weight_f1: float = 0.0
	fitness_weight_fpr: float = 0.0
	min_accuracy_floor: float = 0.0
	# Early stopping
	patience: int = 5
	check_interval: int = 10
	min_improvement_pct: float = 0.1

	@property
	def fitness_weights(self) -> 'FitnessWeights':
		from wnn.ram.metrics import FitnessWeights
		return FitnessWeights(ce=self.fitness_weight_ce, acc=self.fitness_weight_acc,
							  f1=self.fitness_weight_f1, fpr=self.fitness_weight_fpr)

	def create_fitness_calculator(self) -> 'FitnessCalculator':
		"""Create a FitnessCalculator from this config."""
		return FitnessCalculatorFactory.create(
			self.fitness_calculator_type,
			weights=self.fitness_weights,
			min_accuracy_floor=self.min_accuracy_floor if self.min_accuracy_floor > 0 else None,
		)


@dataclass
class GAConfig(OptimizationConfig):
	"""Configuration for Genetic Algorithm."""
	population_size: int = 50
	generations: int = 50
	crossover_rate: float = 0.7
	tournament_size: int = 3
	# Elitism: keep top N% by fitness score (unified ranking)
	# With elitism_pct=0.1 and the 2x multiplier in optimize(), keeps ~20% of population
	elitism_pct: float = 0.1
	# GA-specific early stopping threshold (lower than TS because GA needs diversity)
	min_improvement_pct: float = 0.05
	# Fresh population: ignore initial_population and generate random genomes
	fresh_population: bool = False
	# Seed only: use seed genomes as-is without generating mutations to fill population
	seed_only: bool = False


@dataclass
class TSConfig(OptimizationConfig):
	"""Configuration for Tabu Search optimization."""
	iterations: int = 100
	neighbors_per_iter: int = 50
	tabu_size: int = 10
	# Total neighbors cache for seeding next phase (top K by fitness)
	total_neighbors_size: int = 50
	# TS-specific early stopping threshold (higher than GA because TS is more focused)
	min_improvement_pct: float = 0.5
	# Cooperative multi-start: fraction of top genomes used as neighbor sources.
	# 0.0 = single best (classic TS), 0.2 = top 20% of cache as reference set.
	# Based on Crainic, Toulouse & Gendreau (1997) cooperative TS taxonomy.
	diversity_sources_pct: float = 0.2


# Late import to avoid circular dependency (OptimizationTemplate imports from this file)
from wnn.ram.strategies.connectivity.optimization_template import OptimizationTemplate


class GenericGAStrategy(OptimizationTemplate[T]):
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
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		super().__init__(config or GAConfig(), seed=seed, logger=logger, log_level=log_level, shutdown_check=shutdown_check)

	@property
	def config(self) -> GAConfig:
		return self._config

	@property
	def name(self) -> str:
		return "GenericGA"

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
	# Hooks for subclass customization
	# =========================================================================

	def _generate_offspring(
		self,
		population: list[tuple[T, Optional[float], Optional[float]]],
		n_needed: int,
		threshold: float,
		generation: int,
	) -> list[tuple[T, float, Optional[float]]]:
		"""Generate and evaluate offspring for one generation.

		Override in subclasses for Rust-accelerated offspring generation.
		Default: Python tournament selection + crossover/mutation via _build_viable_population.
		"""
		cfg = self._config

		def offspring_generator() -> T:
			p1 = self._tournament_select(population)
			p2 = self._tournament_select(population)
			if self._rng.random() < cfg.crossover_rate:
				child = self.crossover_genomes(p1, p2)
			else:
				child = self.clone_genome(p1)
			return self.mutate_genome(child, cfg.mutation_rate)

		return self._build_viable_population(
			target_size=n_needed,
			generator_fn=offspring_generator,
			batch_fn=self._batch_evaluate_fn,
			single_fn=self._evaluate_fn,
			min_accuracy=threshold,
			generation=generation,
			total_generations=cfg.generations,
		)

	def _on_generation_start(self, generation: int, **ctx) -> None:
		"""Hook called at start of each generation.

		Override for Metal cleanup, checkpoint save, shutdown check, etc.
		Raise StopIteration to stop the optimization loop gracefully.

		ctx keys: population, best_genome, best_fitness, best_accuracy, threshold, early_stopper
		"""
		pass

	# =========================================================================
	# Core GA loop (Template Method: called by OptimizationTemplate.optimize())
	# =========================================================================

	def _run_optimization_loop(
		self,
		population: list[tuple[T, Optional[float], Optional[float]]],
		fitness_calculator: Any,
		early_stopper: EarlyStoppingTracker,
		**kwargs,
	) -> tuple[
		T, list[tuple[int, float]], list[T], list[tuple[float, float]],
		int, bool, Optional[StopReason], Optional[float], Optional[float],
	]:
		"""
		Run the GA-specific optimization loop.

		Receives a seeded population from the base class (with cached evals
		from previous phase if available), completes it to target size, then
		runs the standard GA loop: elitism, tournament selection, crossover,
		mutation.
		"""
		cfg = self._config

		# Extract strategy-specific kwargs
		initial_genome = kwargs.get('initial_genome')
		overfitting_callback = kwargs.get('overfitting_callback')
		batch_evaluate_fn = self._batch_evaluate_fn
		evaluate_fn = self._evaluate_fn

		# Threshold setup (uses base infrastructure)
		start_threshold, end_threshold = self._threshold_range(cfg.generations)
		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.2%} → {end_threshold:.2%} (rate: {cfg.threshold_delta/cfg.threshold_reference:.4%}/gen)")
		initial_threshold = self._compute_threshold(0.0)

		# Complete population from base's seed_population output
		if population:
			# Evaluate any without cached metrics (seeded genomes may need evaluation)
			population = self._evaluate_population(population, batch_evaluate_fn, evaluate_fn)
			remaining = cfg.population_size - len(population)
			if remaining > 0:
				self._log.info(f"[{self.name}] Filling {remaining} remaining slots from seeded population")
				best_seed = population[0][0]
				def seed_fill_generator() -> T:
					return self.mutate_genome(self.clone_genome(best_seed), cfg.mutation_rate * 3)
				new_pop = self._build_viable_population(
					target_size=remaining,
					generator_fn=seed_fill_generator,
					batch_fn=batch_evaluate_fn,
					single_fn=evaluate_fn,
					min_accuracy=initial_threshold,
				)
				population = list(population) + new_pop
		elif initial_genome is not None:
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

		# Extract Metrics from population tuples
		# Population format: (genome, Metrics) — Metrics has ce, acc, f1, fpr
		from wnn.ram.metrics import Metrics

		def _pop_metrics(pop) -> list[Metrics]:
			"""Extract Metrics list from population."""
			return [t[1] for t in pop]

		# Find initial best using fitness calculator
		metrics_list = _pop_metrics(population)
		init_scores = fitness_calculator.fitness(metrics_list)
		best_idx = min(range(len(init_scores)), key=lambda i: init_scores[i])
		best = self.clone_genome(population[best_idx][0])
		best_fitness = init_scores[best_idx]
		initial_fitness = init_scores[0] if initial_genome else best_fitness
		best_accuracy_val = metrics_list[best_idx].acc
		# Running global best F1/FPR (for dashboard tracking)
		init_f1s = [m.f1 for m in metrics_list if m.f1 is not None]
		init_fprs = [m.fpr for m in metrics_list if m.fpr is not None]
		best_f1_global = max(init_f1s) if init_f1s else None
		best_fpr_global = min(init_fprs) if init_fprs else None

		history = [(0, best_fitness)]

		# Initialize early stopping tracker (uses base infrastructure)
		early_stopper = self._setup_early_stopping(best_fitness)

		# Initialize adaptive scaler for dynamic parameter adjustment
		adaptive_scaler = AdaptiveScaler(
			base_population=cfg.population_size,
			base_mutation=cfg.mutation_rate,
			name=self.name,
		)

		# Track initial diversity (CE spread)
		initial_ce_spread = max(m.ce for m in metrics_list) - min(m.ce for m in metrics_list) if metrics_list else 0.0

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

		# Track previous best for delta computation
		prev_best_fitness = best_fitness

		def _fmt_duration(s):
			if s < 60:
				return f"{s:.0f}s"
			elif s < 3600:
				return f"{s/60:.1f}m"
			else:
				return f"{s/3600:.1f}h"

		shutdown_requested = False
		generation = 0
		loop_start_time = time.time()
		cumulative_offspring_secs = 0.0
		for generation in range(cfg.generations):
			gen_start_time = time.time()
			# Progressive threshold: gets stricter as generations progress
			current_threshold = self._compute_threshold(generation / cfg.threshold_reference)
			# Only log if formatted values differ (avoid noise from tiny internal differences)
			if prev_threshold is not None and f"{prev_threshold:.4%}" != f"{current_threshold:.4%}":
				self._log.debug(f"[{self.name}] Threshold changed: {prev_threshold:.4%} → {current_threshold:.4%}")
			prev_threshold = current_threshold

			# Hook for subclass (Metal cleanup, checkpoint, shutdown check)
			try:
				self._on_generation_start(
					generation,
					population=population,
					best_genome=best,
					best_fitness=best_fitness,
					best_accuracy=best_accuracy_val,
					threshold=current_threshold,
					early_stopper=early_stopper,
					total_generations=cfg.generations,
				)
			except StopIteration:
				shutdown_requested = True
				break

			# Selection and reproduction
			new_population: list[tuple[T, Optional[float], Optional[float]]] = []

			# Unified elitism: use fitness calculator to rank, keep top 20%
			n_elites = max(1, int(cfg.population_size * cfg.elitism_pct * 2))

			# Compute fitness scores from population metrics
			pop_metrics = _pop_metrics(population)
			combined_scores = fitness_calculator.fitness(pop_metrics)
			# Debug: detect broken fitness (all identical)
			if len(combined_scores) > 1 and all(s == combined_scores[0] for s in combined_scores):
				sample = pop_metrics[:3]
				self._log.warning(
					f"[{self.name}] WARNING: All fitness scores identical ({combined_scores[0]:.4f})! "
					f"pop_size={len(population)}, "
					f"sample_metrics={[str(m) for m in sample]}, "
					f"calculator={fitness_calculator.name}"
				)
			elite_sorted = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i])

			# Deduplicate elites by fingerprint — same architecture = same eval,
			# so keeping multiple copies wastes elite slots and reduces diversity.
			seen_fingerprints: set = set()
			all_elite_indices = []
			for idx in elite_sorted:
				if len(all_elite_indices) >= n_elites:
					break
				genome = population[idx][0]
				fp = genome.fingerprint() if hasattr(genome, 'fingerprint') else id(genome)
				if fp not in seen_fingerprints:
					seen_fingerprints.add(fp)
					all_elite_indices.append(idx)
			total_elites = len(all_elite_indices)

			# Track initial elites (first generation only) for survival analysis
			if generation == 0:
				initial_elite_genomes = [
					(self.clone_genome(population[idx][0]), combined_scores[idx])
					for idx in all_elite_indices
				]
				self._log.info(f"[{self.name}] Elitism: {total_elites} by {fitness_calculator.name}")

			# Add elites to new population as (genome, Metrics)
			elite_width = len(str(total_elites))
			for i, elite_idx in enumerate(all_elite_indices):
				elite_genome = self.clone_genome(population[elite_idx][0])
				elite_m = pop_metrics[elite_idx]
				new_population.append((elite_genome, elite_m))

				self._log.debug(f"[Elite {i + 1:0{elite_width}d}/{total_elites}] {elite_m} (score={combined_scores[elite_idx]:.4f})")

			# Store fitness scores for tournament selection (so offspring parents
			# are selected by the same metric as elites, not just raw CE)
			self._current_fitness_scores = combined_scores

			# Generate offspring via hook (overridable for Rust acceleration)
			# μ+λ: generate pop_size offspring (not pop_size - elites), then
			# pool with elites and truncation-select top pop_size.
			needed_offspring = cfg.population_size
			if self._tracker and self._tracker_experiment_id:
				best_ce_str = f"{best_fitness:.4f}" if best_fitness < 999 else "N/A"
				self._tracker.update_experiment_progress(
					self._tracker_experiment_id,
					status_message=f"Gen {generation + 1}/{cfg.generations}: evaluating {needed_offspring} offspring (best CE={best_ce_str})",
				)
			offspring_start = time.time()
			offspring = self._generate_offspring(population, needed_offspring, current_threshold, generation)
			offspring_secs = time.time() - offspring_start
			cumulative_offspring_secs += offspring_secs

			# μ+λ selection: pool elites + offspring, keep top pop_size
			pool = new_population + offspring
			pool_metrics = _pop_metrics(pool)
			pool_scores = fitness_calculator.fitness(pool_metrics)

			# Truncation select: keep top pop_size by fitness score
			ranked_indices = sorted(range(len(pool)), key=lambda i: pool_scores[i])
			keep_indices = ranked_indices[:cfg.population_size]
			population = [pool[i] for i in keep_indices]
			combined_scores = [pool_scores[i] for i in keep_indices]

			# Update best (by fitness calculator score)
			gen_best_idx = 0  # After sorting, index 0 is the best
			if combined_scores[gen_best_idx] < best_fitness:
				best = self.clone_genome(population[gen_best_idx][0])
				best_fitness = combined_scores[gen_best_idx]
				best_accuracy_val = population[gen_best_idx][1].acc

			history.append((generation + 1, best_fitness))

			# Track elite survival: how many old elites survived into new population?
			# Elite indices in the pool are [0, total_elites), offspring are [total_elites, ...)
			surviving_elites = sum(1 for i in keep_indices if i < total_elites)
			new_candidate_count = sum(1 for i in keep_indices if i >= total_elites)

			# Track elite wins: did any old elite rank higher than best offspring?
			best_elite_rank = min(
				(ranked_indices.index(i) for i in range(total_elites) if i in set(keep_indices)),
				default=len(pool),
			)
			best_offspring_rank = min(
				(ranked_indices.index(i) for i in range(total_elites, len(pool)) if i in set(keep_indices)),
				default=len(pool),
			)
			if best_elite_rank <= best_offspring_rank:
				elite_wins += 1

			# Track improvement
			prev_best = history[-2][1] if len(history) >= 2 else history[-1][1]
			if best_fitness < prev_best:
				improved_iterations += 1

			# Log progress with timing
			gen_elapsed = time.time() - gen_start_time
			total_elapsed = time.time() - loop_start_time
			cur_metrics = _pop_metrics(population)
			gen_avg_ce = sum(m.ce for m in cur_metrics) / len(cur_metrics)
			gen_width = len(str(cfg.generations))
			rate = len(offspring) / offspring_secs if offspring_secs > 0 else 0
			gens_done = generation + 1
			gens_remaining = cfg.generations - gens_done
			avg_gen_secs = total_elapsed / gens_done
			eta_secs = gens_remaining * avg_gen_secs
			delta = best_fitness - prev_best_fitness
			delta_str = f"{delta:+.4f}" if delta != 0 else "="
			acc_str = f", acc={best_accuracy_val:.2%}" if best_accuracy_val is not None else ""
			self._log.info(
				f"[{self.name}] Gen {generation + 1:0{gen_width}d}/{cfg.generations}: "
				f"best={best_fitness:.4f} ({delta_str}), avg={gen_avg_ce:.4f}{acc_str} "
				f"[elites survived: {surviving_elites}/{total_elites}] "
				f"| {gen_elapsed:.1f}s (offspring: {offspring_secs:.1f}s, {rate:.1f} gen/s) "
				f"[elapsed: {_fmt_duration(total_elapsed)}, ETA: {_fmt_duration(eta_secs)}]"
			)

			# Record iteration to tracker (if set)
			if self._tracker and self._tracker_experiment_id:
				try:
					# Bests from population using fitness calculator
					genomes_list = [t[0] for t in population]
					iter_bests = fitness_calculator.bests(genomes_list, cur_metrics)
					avg_acc = sum(m.acc for m in cur_metrics) / len(cur_metrics)

					# Patience and baseline info
					baseline_ce = early_stopper._best_fitness if hasattr(early_stopper, '_best_fitness') else None
					delta_baseline = (best_fitness - baseline_ce) if baseline_ce is not None else None
					delta_previous = best_fitness - prev_best_fitness
					patience_counter = early_stopper._patience_counter if hasattr(early_stopper, '_patience_counter') else 0
					candidates_total = len(offspring)

					# Update running global best F1/FPR
					gen_f1s = [m.f1 for m in cur_metrics if m.f1 is not None]
					gen_fprs = [m.fpr for m in cur_metrics if m.fpr is not None]
					if gen_f1s:
						gen_best_f1 = max(gen_f1s)
						if best_f1_global is None or gen_best_f1 > best_f1_global:
							best_f1_global = gen_best_f1
					if gen_fprs:
						gen_best_fpr = min(gen_fprs)
						if best_fpr_global is None or gen_best_fpr < best_fpr_global:
							best_fpr_global = gen_best_fpr

					iteration_id = self._tracker.record_iteration(
						experiment_id=self._tracker_experiment_id,
						iteration_num=generation + 1,
						best_ce=iter_bests.best_ce.metrics.ce,
						best_accuracy=iter_bests.best_acc.metrics.acc,
						avg_ce=gen_avg_ce,
						avg_accuracy=avg_acc,
						elite_count=total_elites,
						offspring_count=len(offspring),
						offspring_viable=len(offspring),  # All offspring are viable at this point
						fitness_threshold=current_threshold,
						elapsed_secs=time.time() - gen_start_time,
						baseline_ce=baseline_ce,
						delta_baseline=delta_baseline,
						delta_previous=delta_previous,
						patience_counter=patience_counter,
						patience_max=cfg.patience,
						candidates_total=candidates_total,
						best_f1=best_f1_global,
						best_fpr=best_fpr_global,
					)

					# Record genome evaluations (if genome_to_config is implemented)
					if iteration_id and self._tracker_experiment_id and HAS_TRACKER and GenomeRole is not None:
						evaluations = []
						for pos, (genome, m) in enumerate(population):
							config = self.genome_to_config(genome)
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								role = GenomeRole.ELITE if pos < total_elites else GenomeRole.OFFSPRING
								fs = combined_scores[pos] if pos < len(combined_scores) else None
								evaluations.append({
									"iteration_id": iteration_id,
									"genome_id": genome_id,
									"position": pos,
									"role": role,
									"ce": m.ce,
									"accuracy": m.acc,
									"elite_rank": pos if pos < total_elites else None,
									"fitness_score": fs,
									"f1_macro": m.f1,
									"fpr": m.fpr,
								})
						if evaluations:
							self._tracker.record_genome_evaluations_batch(evaluations)
				except Exception as e:
					self._log.warning(f"Tracker error: {e}")
					import traceback
					traceback.print_exc()

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
				elif cfg.population_size < old_pop_size:
					# Shrink population — keep best by fitness score
					shrink_metrics = _pop_metrics(population)
					shrink_scores = fitness_calculator.fitness(shrink_metrics)
					shrink_order = sorted(range(len(shrink_scores)), key=lambda i: shrink_scores[i])
					keep_indices = shrink_order[:cfg.population_size]
					population = [population[i] for i in keep_indices]

			# Overfitting callback check (same interval as early stopping)
			if overfitting_callback is not None and (generation + 1) % cfg.check_interval == 0:
				control = overfitting_callback(best, best_fitness)
				if hasattr(control, 'early_stop') and control.early_stop:
					self._log.warning(f"[{self.name}] Overfitting early stop at gen {generation + 1}")
					early_metrics = _pop_metrics(population)
					early_scores = fitness_calculator.fitness(early_metrics)
					early_order = sorted(range(len(early_scores)), key=lambda i: early_scores[i])
					sorted_pop = [population[i] for i in early_order]
					final_population = [self.clone_genome(t[0]) for t in sorted_pop]
					pop_metrics_out = [t[1] for t in sorted_pop]
					current_final_threshold = self._compute_threshold(generation / cfg.threshold_reference)
					genomes_list = [t[0] for t in population]
					early_bests = fitness_calculator.bests(genomes_list, early_metrics)
					return (
						best, history, final_population, pop_metrics_out,
						generation + 1, True, StopReason.OVERFITTING,
						early_bests.best_acc.metrics.acc, current_final_threshold,
					)

			# Update previous best for next iteration's delta computation
			prev_best_fitness = best_fitness

		# Get final bests using fitness calculator
		final_metrics = _pop_metrics(population)
		genomes_list = [t[0] for t in population]
		final_bests = fitness_calculator.bests(genomes_list, final_metrics)
		final_accuracy = final_bests.best_acc.metrics.acc

		# Sort final population by fitness score for seeding next phase
		final_scores = fitness_calculator.fitness(final_metrics)
		scored_pop = list(zip(population, final_scores))
		scored_pop.sort(key=lambda x: x[1])
		final_population = [self.clone_genome(t[0]) for t, _ in scored_pop]
		population_metrics = [t[1] for t, _ in scored_pop]  # list[Metrics]

		# Compute final diversity
		final_ce_spread = max(m.ce for m in final_metrics) - min(m.ce for m in final_metrics) if final_metrics else 0.0

		# Count elite survivals
		elite_survivals = 0
		if initial_elite_genomes:
			final_scores_set = set(combined_scores)
			for _, elite_score in initial_elite_genomes:
				if elite_score in final_scores_set:
					elite_survivals += 1

		# Log analysis summary
		total_gens = generation + 1
		elite_win_rate = elite_wins / total_gens * 100 if total_gens > 0 else 0
		improvement_rate = improved_iterations / total_gens * 100 if total_gens > 0 else 0
		diversity_change = final_ce_spread - initial_ce_spread

		# Compute final threshold for next phase continuity
		final_threshold = self._compute_threshold(generation / cfg.threshold_reference) if cfg.generations > 0 else self._compute_threshold(0.0)

		total_wall_time = time.time() - loop_start_time
		offspring_pct = cumulative_offspring_secs / total_wall_time * 100 if total_wall_time > 0 else 0
		self._log.info(f"[{self.name}] Analysis Summary:")
		self._log.info(f"  CE improvement: {initial_fitness:.4f} → {best_fitness:.4f} ({(1 - best_fitness/initial_fitness)*100:+.2f}%)")
		self._log.info(f"  CE spread: {initial_ce_spread:.4f} → {final_ce_spread:.4f} ({diversity_change:+.4f})")
		self._log.info(f"  Elite survivals: {elite_survivals}/{len(initial_elite_genomes) if initial_elite_genomes else 0}")
		self._log.info(f"  Elite win rate: {elite_wins}/{total_gens} ({elite_win_rate:.1f}%)")
		self._log.info(f"  Improvement rate: {improved_iterations}/{total_gens} ({improvement_rate:.1f}%)")
		self._log.info(f"  Wall time: {_fmt_duration(total_wall_time)} total, {_fmt_duration(cumulative_offspring_secs)} offspring ({offspring_pct:.0f}%)")
		self._log.info(f"  Avg gen: {total_wall_time / total_gens:.1f}s" if total_gens > 0 else "")
		self._log.info(f"  Final threshold: {final_threshold:.2%} (for next phase)")

		# Determine stop reason (uses base infrastructure)
		stop_reason = self._determine_stop_reason(shutdown_requested, early_stopper)

		return (
			best, history, final_population, population_metrics,
			generation + 1,
			early_stopper.patience_exhausted or shutdown_requested,
			stop_reason, final_accuracy, final_threshold,
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
		from wnn.ram.metrics import Metrics

		# Re-evaluate items without Metrics
		unknown_indices = [i for i, t in enumerate(population) if not isinstance(t[1], Metrics)]

		if not unknown_indices:
			return list(population)  # All have Metrics

		to_eval = [population[i][0] for i in unknown_indices]

		# Batch evaluate — returns list[Metrics]
		if batch_fn is not None:
			results = batch_fn(to_eval)
			new_metrics = [r if isinstance(r, Metrics) else Metrics(ce=r.ce, acc=r.acc, f1=r.f1, fpr=r.fpr) for r in results]
		else:
			new_metrics = [Metrics(ce=single_fn(g), acc=0.0) for g in to_eval]

		result = list(population)
		for idx, m in zip(unknown_indices, new_metrics):
			result[idx] = (result[idx][0], m)

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
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
	):
		"""
		Build a population of viable candidates (accuracy >= min_accuracy).

		Returns list of (genome, Metrics) tuples.
		"""
		from wnn.ram.metrics import Metrics

		viable: list[tuple] = []
		filtered_count = 0

		batch_kwargs: dict = {"min_accuracy": min_accuracy}
		if generation is not None:
			batch_kwargs["generation"] = generation
		if total_generations is not None:
			batch_kwargs["total_generations"] = total_generations

		import time as _time
		known_fps: set = set()

		def _to_metrics(r) -> Metrics:
			"""Convert evaluator result to Metrics."""
			if isinstance(r, Metrics):
				return r
			return Metrics(ce=r.ce, acc=r.acc, f1=r.f1, fpr=r.fpr,
						   threshold=getattr(r, 'threshold', None),
						   bit_accuracy=getattr(r, 'bit_accuracy', None))

		# Evaluate seed genomes (always accepted)
		if seed_genomes:
			unique_seeds = []
			for g in seed_genomes[:target_size]:
				fp = g.fingerprint() if hasattr(g, 'fingerprint') else id(g)
				if fp not in known_fps:
					known_fps.add(fp)
					unique_seeds.append(self.clone_genome(g))
			to_eval = unique_seeds
			self._log.info(f"[{self.name}] Evaluating {len(to_eval)} seed genomes...")
			t0 = _time.time()
			if batch_fn is not None:
				results = batch_fn(to_eval, **batch_kwargs)
				elapsed = _time.time() - t0
				metrics = [_to_metrics(r) for r in results]
				best_ce = min(m.ce for m in metrics) if metrics else 0.0
				best_acc = max(m.acc for m in metrics) if metrics else 0.0
				self._log.info(f"[{self.name}] Seed eval: {len(to_eval)} genomes in {elapsed:.1f}s (best CE={best_ce:.4f}, Acc={best_acc:.2%})")
				for genome, m in zip(to_eval, metrics):
					viable.append((genome, m))
			else:
				for genome in to_eval:
					ce = single_fn(genome)
					viable.append((genome, Metrics(ce=ce, acc=0.0)))
			self._log.info(f"[{self.name}] {len(viable)}/{target_size} viable after seed eval")

		# Generate new candidates until we have enough
		attempt = 0
		while len(viable) < target_size and attempt < max_attempts:
			attempt += 1
			needed = target_size - len(viable)
			batch_size = min(needed * 2, needed + 10)
			candidates = []
			for _ in range(batch_size):
				g = generator_fn()
				fp = g.fingerprint() if hasattr(g, 'fingerprint') else id(g)
				if fp not in known_fps:
					known_fps.add(fp)
					candidates.append(g)
			if not candidates:
				continue

			self._log.info(f"[{self.name}] Building population: attempt {attempt}, evaluating {len(candidates)} candidates ({len(viable)}/{target_size} viable)")
			t0 = _time.time()
			if batch_fn is not None:
				results = batch_fn(candidates, **batch_kwargs)
				elapsed = _time.time() - t0
				self._log.info(f"[{self.name}] Batch eval: {len(candidates)} candidates in {elapsed:.1f}s")
				for genome, r in zip(candidates, results):
					m = _to_metrics(r)
					if m.acc >= min_accuracy:
						viable.append((genome, m))
						if len(viable) >= target_size:
							break
					else:
						filtered_count += 1
			else:
				for genome in candidates:
					ce = single_fn(genome)
					viable.append((genome, Metrics(ce=ce, acc=0.0)))
					if len(viable) >= target_size:
						break

		if filtered_count > 0:
			self._log.trace(f"[{self.name}] Filtered {filtered_count} candidates with accuracy < {min_accuracy:.2%}")

		if len(viable) < target_size:
			self._log.warning(f"[{self.name}] Warning: only {len(viable)}/{target_size} viable candidates after {max_attempts} attempts")

		return viable[:target_size]

	def _tournament_select(self, population: list[tuple[T, float, Optional[float]]], tournament_size: int = 3) -> T:
		"""Tournament selection using fitness scores (not raw CE).

		Uses pre-computed fitness scores from _current_fitness_scores if available,
		otherwise falls back to CE (population[i][1]).  Fitness scores align tournament
		selection with elite selection — both respect the fitness calculator.
		"""
		indices = self._rng.sample(range(len(population)), min(tournament_size, len(population)))
		scores = getattr(self, '_current_fitness_scores', None)
		if scores is not None and len(scores) == len(population):
			best_idx = min(indices, key=lambda i: scores[i])
		else:
			best_idx = min(indices, key=lambda i: population[i][1])
		return population[best_idx][0]


class GenericTSStrategy(OptimizationTemplate[T]):
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
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		super().__init__(config or TSConfig(), seed=seed, logger=logger, log_level=log_level, shutdown_check=shutdown_check)

	@property
	def config(self) -> TSConfig:
		return self._config

	@property
	def name(self) -> str:
		return "GenericTS"

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
	# Hooks for subclass customization
	# =========================================================================

	def _generate_neighbors(
		self,
		best_genome: T,
		n_neighbors: int,
		threshold: float,
		iteration: int,
		tabu_list: list,
	) -> list[tuple[T, float, Optional[float]]]:
		"""Generate and evaluate neighbors for one iteration.

		Override in subclasses for Rust-accelerated neighbor generation.
		Default: Python path with retry loop — generates batches of candidates,
		evaluates them, and retries until target_count viable neighbors found
		or max_attempts reached. Falls back to best-N if threshold not met.

		Args:
			best_genome: Best genome by fitness ranking (used as mutation source)
			n_neighbors: Number of neighbors to generate
			threshold: Minimum accuracy threshold
			iteration: Current iteration number
			tabu_list: Tabu list (deque) for move tracking

		Returns list of (genome, ce, accuracy) tuples for viable neighbors.
		"""
		cfg = self._config
		max_attempts = n_neighbors * 5
		total_evaluated = 0

		viable: list[tuple[T, Any, float, Optional[float]]] = []  # (genome, move, ce, acc)
		all_below_threshold: list[tuple[T, Any, float, Optional[float]]] = []

		while len(viable) < n_neighbors and total_evaluated < max_attempts:
			remaining = n_neighbors - len(viable)
			batch_n = min(remaining + 5, n_neighbors, max_attempts - total_evaluated)
			if batch_n <= 0:
				break

			# Generate batch of candidates
			candidates: list[tuple[T, Any]] = []
			for _ in range(batch_n):
				neighbor, move = self.mutate_genome(self.clone_genome(best_genome), cfg.mutation_rate)
				if not self.is_tabu_move(move, tabu_list):
					candidates.append((neighbor, move))

			if not candidates:
				total_evaluated += batch_n
				continue

			# Evaluate batch
			from wnn.ram.metrics import Metrics as _Metrics
			if self._batch_evaluate_fn is not None:
				to_eval = [n for n, _ in candidates]
				results = self._batch_evaluate_fn(
					to_eval, min_accuracy=threshold,
					generation=iteration, total_generations=self._config.iterations,
				)
				eval_metrics = [r if isinstance(r, _Metrics) else _Metrics(ce=r.ce, acc=r.acc, f1=r.f1, fpr=r.fpr) for r in results]
			else:
				eval_metrics = [_Metrics(ce=self._evaluate_fn(n), acc=0.0) for n, _ in candidates]

			total_evaluated += len(candidates)

			# Sort into viable vs below-threshold
			for (n, m_phase), em in zip(candidates, eval_metrics):
				if em.acc >= threshold:
					viable.append((n, m_phase, em))
				else:
					all_below_threshold.append((n, m_phase, em))

		# If not enough viable, fall back to best by accuracy then CE
		if not viable:
			if all_below_threshold:
				all_below_threshold.sort(key=lambda x: (-(x[2].acc), x[2].ce))
				viable = all_below_threshold[:1]
		elif len(viable) < n_neighbors:
			all_below_threshold.sort(key=lambda x: (-(x[2].acc), x[2].ce))
			need = n_neighbors - len(viable)
			viable.extend(all_below_threshold[:need])

		# Update tabu list with best neighbor's move
		if viable:
			viable_metrics = [em for _, _, em in viable]
			viable_genomes = [n for n, _, _ in viable]
			ranked = self._fitness_calculator.rank(viable_genomes, viable_metrics)
			best_genome = ranked[0][0]
			best_idx = next(i for i, (n, _, _) in enumerate(viable) if n is best_genome)
			_, m_phase, _ = viable[best_idx]
			if m_phase is not None:
				tabu_list.append(m_phase)

		# Return viable neighbors as (genome, Metrics) tuples
		return [(n, em) for n, _, em in viable]

	def _on_iteration_start(self, iteration: int, **ctx) -> None:
		"""Hook called at start of each iteration.

		Override for Metal cleanup, shutdown check, etc.
		Raise StopIteration to stop the optimization loop gracefully.

		ctx keys: best_genome, best_fitness, best_accuracy, threshold
		"""
		pass

	def _select_top_n(
		self,
		candidates: list[tuple[T, float, Optional[float]]],
		n: int,
		fitness_calculator: Any,
	) -> list[tuple[T, float, Optional[float]]]:
		"""Select top N unique candidates by fitness ranking."""
		# Deduplicate by fingerprint before ranking — same architecture = same eval
		seen_fps: set = set()
		unique_candidates = []
		for t in candidates:
			fp = t[0].fingerprint() if hasattr(t[0], 'fingerprint') else id(t[0])
			if fp not in seen_fps:
				seen_fps.add(fp)
				unique_candidates.append(t)
		candidates = unique_candidates

		if len(candidates) <= n:
			return candidates
		# Rank by fitness calculator
		genomes = [t[0] for t in candidates]
		metrics = [t[1] for t in candidates]
		scores = fitness_calculator.fitness(metrics)
		ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i])
		return [candidates[i] for i in ranked_idx[:n]]

	def _find_best_ranked(
		self,
		pop: list[tuple],
		fitness_calculator: Any,
	) -> tuple:
		"""Find best genome in population by fitness ranking."""
		genomes = [t[0] for t in pop]
		metrics = [t[1] for t in pop]
		scores = fitness_calculator.fitness(metrics)
		best_idx = min(range(len(scores)), key=lambda i: scores[i])
		best_item = pop[best_idx]
		return (self.clone_genome(best_item[0]),) + best_item[1:]

	# =========================================================================
	# Core TS loop (Template Method: called by OptimizationTemplate.optimize())
	# =========================================================================

	def _run_optimization_loop(
		self,
		population: list[tuple[T, Optional[float], Optional[float]]],
		fitness_calculator: Any,
		early_stopper: EarlyStoppingTracker,
		**kwargs,
	) -> tuple[
		T, list[tuple[int, float]], list[T], list[tuple[float, float]],
		int, bool, Optional[StopReason], Optional[float], Optional[float],
	]:
		"""
		Run the population-based TS optimization loop (μ + λ strategy).

		Each iteration:
		1. Select top 20% of population as elite sources
		2. Each source generates equal offspring (total = neighbors_per_iter)
		3. Merge population + offspring, rank, keep top population_size

		Receives kwargs with initial_genome, initial_fitness, initial_neighbors,
		etc. The population parameter from base's seed_population is typically
		empty (TS uses initial_neighbors instead of initial_population).
		"""
		cfg = self._config

		# Extract strategy-specific kwargs
		initial_genome = kwargs.get('initial_genome')
		initial_fitness = kwargs.get('initial_fitness', 0.0)
		initial_neighbors = kwargs.get('initial_neighbors')
		overfitting_callback = kwargs.get('overfitting_callback')
		batch_evaluate_fn = self._batch_evaluate_fn
		evaluate_fn = self._evaluate_fn

		# Threshold setup (uses base infrastructure)
		start_threshold, end_threshold = self._threshold_range(cfg.iterations)
		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.2%} → {end_threshold:.2%} (rate: {cfg.threshold_delta/cfg.threshold_reference:.4%}/iter)")

		# Population size (fixed throughout the run)
		pop_size = cfg.total_neighbors_size or cfg.neighbors_per_iter

		# Single tabu list
		tabu_list: deque = deque(maxlen=cfg.tabu_size)

		# Re-evaluate initial genome on current phase's train subset
		# (cached evals from previous phase used a different subset)
		initial_accuracy: Optional[float] = None
		# initial_f1/initial_fpr removed — now part of Metrics
		initial_evals = kwargs.get('initial_evals')
		if initial_evals and batch_evaluate_fn is not None:
			self._log.info(f"[{self.name}] Re-evaluating initial genome (phase transition — different train subset)")
			try:
				init_results = batch_evaluate_fn([initial_genome])
				if init_results:
					r = init_results[0]
					initial_fitness = r.ce if hasattr(r, 'ce') else r[0]
					initial_accuracy = r.acc if hasattr(r, 'acc') else r[1]
					self._log.info(f"[{self.name}] Re-evaluated initial genome: CE={initial_fitness:.4f}, Acc={initial_accuracy:.2%}")
			except Exception as e:
				self._log.warning(f"[{self.name}] Failed to re-evaluate initial genome: {e}, falling back to cached")
				e0 = initial_evals[0]
				initial_fitness = e0.ce if hasattr(e0, 'ce') else e0[0]
				initial_accuracy = e0.acc if hasattr(e0, 'acc') else e0[1]
		elif initial_evals:
			# No batch_evaluate_fn — fall back to cached evals
			e0 = initial_evals[0]
			initial_fitness = e0.ce if hasattr(e0, 'ce') else e0[0]
			initial_accuracy = e0.acc if hasattr(e0, 'acc') else e0[1]
			self._log.info(f"[{self.name}] Using cached eval for initial genome (no evaluator): CE={initial_fitness:.4f}, Acc={initial_accuracy:.2%}")
		elif batch_evaluate_fn is not None:
			try:
				init_results = batch_evaluate_fn([initial_genome])
				if init_results:
					r = init_results[0]
					initial_fitness = r.ce if hasattr(r, 'ce') else r[0]
					initial_accuracy = r.acc if hasattr(r, 'acc') else r[1]
			except Exception as e:
				self._log.warning(f"[{self.name}] Failed to re-evaluate initial genome: {e}")

		# === Build initial population as (genome, Metrics) ===
		from wnn.ram.metrics import Metrics as _M
		init_metrics = _M(ce=initial_fitness, acc=initial_accuracy or 0.0)
		pop: list = [
			(self.clone_genome(initial_genome), init_metrics)
		]

		# Initial threshold
		current_threshold = self._compute_threshold(0.0)

		# Seed with initial neighbors if provided (e.g., from previous GA phase)
		# Deduplicate seeds against initial genome and each other
		if initial_neighbors:
			# Build fingerprint set from initial genome
			seen_fps: set = set()
			ig = pop[0][0]  # initial_genome clone
			if hasattr(ig, 'fingerprint'):
				seen_fps.add(ig.fingerprint())

			# Filter out duplicate neighbors
			unique_neighbors = []
			for g in initial_neighbors:
				fp = g.fingerprint() if hasattr(g, 'fingerprint') else id(g)
				if fp not in seen_fps:
					seen_fps.add(fp)
					unique_neighbors.append(g)
			if len(unique_neighbors) < len(initial_neighbors):
				self._log.info(f"[{self.name}] Dedup: {len(initial_neighbors)} → {len(unique_neighbors)} unique seeded neighbors")
			initial_neighbors = unique_neighbors

			# Always re-evaluate at phase transitions (cached evals used different train subset)
			if batch_evaluate_fn is not None:
				reason = "phase transition" if initial_evals is not None else "no cached evals"
				self._log.info(f"[{self.name}] Re-evaluating {len(initial_neighbors)} seeded neighbors ({reason})")
				results = batch_evaluate_fn(initial_neighbors, min_accuracy=current_threshold)
				seed_metrics = [r if isinstance(r, _M) else _M(ce=r.ce, acc=r.acc, f1=r.f1, fpr=r.fpr) for r in results]
			else:
				seed_metrics = [_M(ce=evaluate_fn(g), acc=0.0) for g in initial_neighbors]

			for g, m in zip(initial_neighbors, seed_metrics):
				pop.append((self.clone_genome(g), m))

		# Trim initial population to pop_size by fitness ranking
		if len(pop) > pop_size:
			pop = self._select_top_n(pop, pop_size, fitness_calculator)

		from wnn.ram.metrics import Metrics

		def _ts_pop_metrics(p) -> list[Metrics]:
			return [t[1] for t in p]

		# Global best F1/FPR tracking
		ts_init_metrics = _ts_pop_metrics(pop)
		init_f1s = [m.f1 for m in ts_init_metrics if m.f1 is not None]
		init_fprs = [m.fpr for m in ts_init_metrics if m.fpr is not None]
		best_f1_global: Optional[float] = max(init_f1s) if init_f1s else None
		best_fpr_global: Optional[float] = min(init_fprs) if init_fprs else None

		# Find best from initial population using fitness calculator
		init_scores = fitness_calculator.fitness(ts_init_metrics)
		best_idx = min(range(len(init_scores)), key=lambda i: init_scores[i])
		best = self.clone_genome(pop[best_idx][0])
		best_fitness = init_scores[best_idx]
		best_accuracy: Optional[float] = ts_init_metrics[best_idx].acc
		start_fitness = best_fitness if not initial_fitness else initial_fitness

		# Best ranked genome
		best_ranked = self._find_best_ranked(pop, fitness_calculator)
		best_ranked_genome = best_ranked[0]
		best_ranked_m = best_ranked[1]
		best_ranked_ce = best_ranked_m.ce if hasattr(best_ranked_m, 'ce') else best_ranked_m
		best_ranked_accuracy = best_ranked_m.acc if hasattr(best_ranked_m, 'acc') else None

		# Log seed summary
		best_acc_val = max((m.acc for m in ts_init_metrics), default=None)
		self._log.info(f"[{self.name}] Seed: best_ce={best_fitness:.4f}, best_acc={best_acc_val:.2%}, pop={len(pop)}" if best_acc_val else
					   f"[{self.name}] Seed: best_ce={best_fitness:.4f}, best_acc=N/A, pop={len(pop)}")

		history = [(0, best_fitness)]

		# Analysis tracking
		improved_iterations = 0

		# Initialize early stopping tracker (uses base infrastructure)
		early_stopper = self._setup_early_stopping(best_fitness)

		# Log config
		elite_pct = cfg.diversity_sources_pct if cfg.diversity_sources_pct > 0 else 1.0
		n_elite_est = max(1, int(pop_size * elite_pct))
		offspring_per_elite = max(1, cfg.neighbors_per_iter // n_elite_est)
		self._log.info(f"[{self.name}] Config: pop={pop_size}, elite={elite_pct:.0%} ({n_elite_est} sources × {offspring_per_elite} offspring), "
					   f"iters={cfg.iterations}, fitness={fitness_calculator.name}")

		# Track threshold changes
		prev_threshold: Optional[float] = None

		# Track previous best for delta computation
		prev_best_fitness = best_fitness

		def _fmt_duration_ts(s):
			if s < 60:
				return f"{s:.0f}s"
			elif s < 3600:
				return f"{s/60:.1f}m"
			else:
				return f"{s/3600:.1f}h"

		shutdown_requested = False
		iteration = 0
		loop_start_time = time.time()
		cumulative_offspring_secs = 0.0
		for iteration in range(cfg.iterations):
			iter_start_time = time.time()
			# Progressive threshold
			current_threshold = self._compute_threshold(iteration / cfg.threshold_reference)
			if prev_threshold is not None and f"{prev_threshold:.4%}" != f"{current_threshold:.4%}":
				self._log.debug(f"[{self.name}] Threshold changed: {prev_threshold:.4%} → {current_threshold:.4%}")
			prev_threshold = current_threshold

			# Hook for subclass (Metal cleanup, shutdown check)
			try:
				self._on_iteration_start(
					iteration,
					best_genome=best_ranked_genome,
					best_fitness=best_fitness,
					best_accuracy=best_accuracy,
					threshold=current_threshold,
					total_generations=cfg.iterations,
				)
			except StopIteration:
				shutdown_requested = True
				break

			# === (μ + λ) selection: generate offspring from elite sources ===
			offspring_start = time.time()
			if cfg.diversity_sources_pct > 0 and len(pop) > 1:
				# Population-based TS: top 20% of population as sources
				n_sources = max(1, int(len(pop) * cfg.diversity_sources_pct))
				ts_rank_genomes = [t[0] for t in pop]
				ts_rank_metrics = [t[1] for t in pop]
				if ts_rank_metrics:
					ranked = fitness_calculator.rank(ts_rank_genomes, ts_rank_metrics)
					sources = [self.clone_genome(g) for g, _ in ranked[:n_sources]]
				else:
					sources = [self.clone_genome(pop[0][0])]
					n_sources = 1

				# Equal share per source
				total_offspring = cfg.neighbors_per_iter
				per_source = max(1, total_offspring // n_sources)
				remainder = total_offspring - (per_source * n_sources)
				counts = [per_source + (1 if si < remainder else 0) for si in range(n_sources)]

				# Try batch evaluation (single Rust call for all sources)
				offspring: list[tuple[T, float, Optional[float]]] = []
				batch_result = None
				if hasattr(self, '_generate_neighbors_batch'):
					if self._tracker and self._tracker_experiment_id:
						best_ce_str = f"{best_fitness:.4f}" if best_fitness < 999 else "N/A"
						self._tracker.update_experiment_progress(
							self._tracker_experiment_id,
							status_message=f"Iter {iteration + 1}/{cfg.iterations}: batch evaluating {total_offspring} offspring from {n_sources} sources (best CE={best_ce_str})",
						)
					batch_result = self._generate_neighbors_batch(
						sources, counts, current_threshold, iteration, tabu_list,
					)

				if batch_result is not None:
					for source_offspring in batch_result:
						offspring.extend(source_offspring)
				else:
					# Fallback: per-source evaluation
					for si, source in enumerate(sources):
						if self._tracker and self._tracker_experiment_id:
							best_ce_str = f"{best_fitness:.4f}" if best_fitness < 999 else "N/A"
							self._tracker.update_experiment_progress(
								self._tracker_experiment_id,
								status_message=f"Iter {iteration + 1}/{cfg.iterations}: source {si + 1}/{n_sources}, {len(offspring)}/{total_offspring} offspring (best CE={best_ce_str})",
							)
						batch = self._generate_neighbors(
							best_genome=source, n_neighbors=counts[si],
							threshold=current_threshold, iteration=iteration,
							tabu_list=tabu_list,
						)
						offspring.extend(batch)
			else:
				# Classic TS: all offspring from single best-ranked genome
				if self._tracker and self._tracker_experiment_id:
					best_ce_str = f"{best_fitness:.4f}" if best_fitness < 999 else "N/A"
					self._tracker.update_experiment_progress(
						self._tracker_experiment_id,
						status_message=f"Iter {iteration + 1}/{cfg.iterations}: evaluating {cfg.neighbors_per_iter} neighbors (best CE={best_ce_str})",
					)
				offspring = self._generate_neighbors(
					best_genome=best_ranked_genome,
					n_neighbors=cfg.neighbors_per_iter,
					threshold=current_threshold,
					iteration=iteration,
					tabu_list=tabu_list,
				)

			offspring_secs = time.time() - offspring_start
			cumulative_offspring_secs += offspring_secs

			if offspring:
				# === (μ + λ) replacement: merge population + offspring, keep top pop_size ===
				combined = []
				for t in pop:
					combined.append((self.clone_genome(t[0]),) + t[1:])
				for t in offspring:
					combined.append((self.clone_genome(t[0]),) + t[1:])

				pop = self._select_top_n(combined, pop_size, fitness_calculator)

				# Update best_ranked from new population
				best_ranked = self._find_best_ranked(pop, fitness_calculator)
				best_ranked_genome = best_ranked[0]
		best_ranked_m = best_ranked[1]
		best_ranked_ce = best_ranked_m.ce if hasattr(best_ranked_m, 'ce') else best_ranked_m
		best_ranked_accuracy = best_ranked_m.acc if hasattr(best_ranked_m, 'acc') else None

				# Update global best (by fitness calculator score)
				iter_metrics = _ts_pop_metrics(pop)
				iter_scores = fitness_calculator.fitness(iter_metrics)
				iter_best_idx = min(range(len(iter_scores)), key=lambda i: iter_scores[i])
				if iter_scores[iter_best_idx] < best_fitness:
					best = self.clone_genome(pop[iter_best_idx][0])
					best_fitness = iter_scores[iter_best_idx]
					best_accuracy = iter_metrics[iter_best_idx].acc
					improved_iterations += 1

			history.append((iteration + 1, best_fitness))

			# Log progress with timing
			iter_elapsed = time.time() - iter_start_time
			total_elapsed = time.time() - loop_start_time
			iter_width = len(str(cfg.iterations))
			rate = len(offspring) / offspring_secs if offspring_secs > 0 else 0
			iters_done = iteration + 1
			iters_remaining = cfg.iterations - iters_done
			avg_iter_secs = total_elapsed / iters_done
			eta_secs = iters_remaining * avg_iter_secs
			# Delta from previous iteration
			delta = best_fitness - prev_best_fitness
			delta_str = f"{delta:+.4f}" if delta != 0 else "="
			ranked_acc_str = f"{best_ranked_accuracy:.2%}" if best_ranked_accuracy is not None else "N/A"
			self._log.info(
				f"[{self.name}] Iter {iteration + 1:0{iter_width}d}/{cfg.iterations}: "
				f"best_ranked=(CE={best_ranked_ce:.4f}, Acc={ranked_acc_str}), "
				f"best_ce={best_fitness:.4f} ({delta_str}), pop={len(pop)} "
				f"| {iter_elapsed:.1f}s (offspring: {offspring_secs:.1f}s, {rate:.1f} gen/s) "
				f"[elapsed: {_fmt_duration_ts(total_elapsed)}, ETA: {_fmt_duration_ts(eta_secs)}]"
			)

			# Record iteration to tracker (if set)
			if self._tracker and self._tracker_experiment_id:
				try:
					# Compute population stats
					ts_cur_metrics = _ts_pop_metrics(pop)
					pop_avg_ce = sum(m.ce for m in ts_cur_metrics) / len(ts_cur_metrics) if ts_cur_metrics else None
					pop_avg_acc = sum(m.acc for m in ts_cur_metrics) / len(ts_cur_metrics) if ts_cur_metrics else None

					# Baseline and patience info
					baseline_ce = early_stopper._best_fitness if hasattr(early_stopper, '_best_fitness') else None
					delta_baseline = (best_fitness - baseline_ce) if baseline_ce is not None else None
					delta_previous = best_fitness - prev_best_fitness
					patience_counter = early_stopper._patience_counter if hasattr(early_stopper, '_patience_counter') else 0

					# Bests from population
					ts_genomes = [t[0] for t in pop]
					iter_bests = fitness_calculator.bests(ts_genomes, ts_cur_metrics)

					# Update running global best F1/FPR
					pop_f1s = [m.f1 for m in ts_cur_metrics if m.f1 is not None]
					pop_fprs = [m.fpr for m in ts_cur_metrics if m.fpr is not None]
					if pop_f1s:
						iter_best_f1 = max(pop_f1s)
						if best_f1_global is None or iter_best_f1 > best_f1_global:
							best_f1_global = iter_best_f1
					if pop_fprs:
						iter_best_fpr = min(pop_fprs)
						if best_fpr_global is None or iter_best_fpr < best_fpr_global:
							best_fpr_global = iter_best_fpr

					iteration_id = self._tracker.record_iteration(
						experiment_id=self._tracker_experiment_id,
						iteration_num=iteration + 1,
						best_ce=iter_bests.best_ce.metrics.ce,
						best_accuracy=iter_bests.best_acc.metrics.acc,
						avg_ce=pop_avg_ce,
						avg_accuracy=pop_avg_acc,
						elite_count=len(pop),
						offspring_count=len(offspring),
						offspring_viable=len(offspring),
						fitness_threshold=current_threshold,
						elapsed_secs=time.time() - iter_start_time,
						baseline_ce=baseline_ce,
						delta_baseline=delta_baseline,
						delta_previous=delta_previous,
						patience_counter=patience_counter,
						patience_max=cfg.patience,
						candidates_total=len(pop) + len(offspring),
						best_f1=best_f1_global,
						best_fpr=best_fpr_global,
					)

					# Record genome evaluations (if genome_to_config is implemented)
					if iteration_id and self._tracker_experiment_id and HAS_TRACKER and GenomeRole is not None:
						evaluations = []

						# Compute fitness scores for combined pop + offspring
						all_items = list(pop) + list(offspring)
						all_metrics = [t[1] for t in all_items]
						all_scores = fitness_calculator.fitness(all_metrics) if fitness_calculator else [None] * len(all_items)

						# Record population members as TOP_K
						for pos, item in enumerate(pop):
							config = self.genome_to_config(item[0])
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								evaluations.append({
									"iteration_id": iteration_id,
									"genome_id": genome_id,
									"position": pos,
									"role": GenomeRole.TOP_K,
									"ce": item[1],
									"accuracy": item[2] if item[2] is not None else 0.0,
									"elite_rank": pos,
									"fitness_score": all_scores[pos],
									"f1_macro": item[3] if len(item) > 3 else None,
									"fpr": item[4] if len(item) > 4 else None,
								})

						# Record offspring as NEIGHBOR
						for pos, item in enumerate(offspring):
							config = self.genome_to_config(item[0])
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								evaluations.append({
									"iteration_id": iteration_id,
									"genome_id": genome_id,
									"position": len(pop) + pos,
									"role": GenomeRole.NEIGHBOR,
									"ce": item[1],
									"accuracy": item[2] if item[2] is not None else 0.0,
									"fitness_score": all_scores[len(pop) + pos],
									"f1_macro": item[3] if len(item) > 3 else None,
									"fpr": item[4] if len(item) > 4 else None,
								})

						if evaluations:
							self._tracker.record_genome_evaluations_batch(evaluations)
				except Exception as e:
					self._log.warning(f"Tracker error: {e}")
					import traceback
					traceback.print_exc()

			# Early stopping check
			if early_stopper.check(iteration, best_fitness):
				break

			# Overfitting callback
			if overfitting_callback is not None and (iteration + 1) % cfg.check_interval == 0:
				control = overfitting_callback(best, best_fitness)
				if hasattr(control, 'early_stop') and control.early_stop:
					self._log.warning(f"[{self.name}] Overfitting early stop at iter {iteration + 1}")
					break

			# Update previous best for next iteration's delta computation
			prev_best_fitness = best_fitness

		# === Build final_population from current population ===
		final_population = [self.clone_genome(t[0]) for t in pop]
		population_metrics = [t[1] for t in pop]  # list[Metrics]

		# Final threshold (for next phase)
		final_threshold = self._compute_threshold(iteration / cfg.threshold_reference) if cfg.iterations > 0 else self._compute_threshold(0.0)

		# Log analysis summary
		total_iters = iteration + 1
		total_wall_time = time.time() - loop_start_time
		offspring_pct = cumulative_offspring_secs / total_wall_time * 100 if total_wall_time > 0 else 0
		self._log.info(f"[{self.name}] Analysis Summary:")
		self._log.info(f"  CE improvement: {start_fitness:.4f} → {best_fitness:.4f} ({(1 - best_fitness/start_fitness)*100:+.2f}%)")
		self._log.info(f"  Improved iterations: {improved_iterations}/{total_iters}")
		self._log.info(f"  Wall time: {_fmt_duration_ts(total_wall_time)} total, {_fmt_duration_ts(cumulative_offspring_secs)} offspring ({offspring_pct:.0f}%)")
		self._log.info(f"  Avg iter: {total_wall_time / total_iters:.1f}s" if total_iters > 0 else "")
		self._log.info(f"  Final population: {len(final_population)} by {fitness_calculator.name}")
		self._log.info(f"  Final threshold: {final_threshold:.2%} (for next phase)")

		# Determine stop reason (uses base infrastructure)
		stop_reason = self._determine_stop_reason(shutdown_requested, early_stopper)

		# Three independent bests from final population
		ts_bests = fitness_calculator.bests(pop)

		return (
			best, history, final_population, population_metrics,
			iteration + 1,
			early_stopper.patience_exhausted or shutdown_requested,
			stop_reason, ts_bests.best_acc.metrics.acc, final_threshold,
		)
