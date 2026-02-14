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
	# Per-genome (CE, accuracy) matching final_population order
	population_metrics: Optional[list[tuple[float, float]]] = None
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
	# Accuracy floor: genomes below this get fitness = infinity (0.0 = disabled)
	min_accuracy_floor: float = 0.0
	# Early stopping
	patience: int = 5
	check_interval: int = 10
	min_improvement_pct: float = 0.1

	def create_fitness_calculator(self) -> 'FitnessCalculator':
		"""Create a FitnessCalculator from this config."""
		return FitnessCalculatorFactory.create(
			self.fitness_calculator_type,
			weight_ce=self.fitness_weight_ce,
			weight_acc=self.fitness_weight_acc,
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
	neighbors_per_iter: int = 20
	tabu_size: int = 10
	# Total neighbors cache for seeding next phase (top K by fitness)
	total_neighbors_size: int = 50
	# TS-specific early stopping threshold (higher than GA because TS is more focused)
	min_improvement_pct: float = 0.5
	# Cooperative multi-start: fraction of top genomes used as neighbor sources.
	# 0.0 = single best (classic TS), 0.2 = top 20% of cache as reference set.
	# Based on Crainic, Toulouse & Gendreau (1997) cooperative TS taxonomy.
	diversity_sources_pct: float = 0.0


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
		# Tracker for iteration recording (set via set_tracker)
		self._tracker: Optional["ExperimentTracker"] = None
		self._tracker_experiment_id: Optional[int] = None

	def set_tracker(self, tracker: "ExperimentTracker", experiment_id: int, _unused: Optional[int] = None) -> None:
		"""Set the experiment tracker for iteration recording."""
		self._tracker = tracker
		self._tracker_experiment_id = experiment_id

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
	# Optional genome tracking (subclasses can override)
	# =========================================================================

	def genome_to_config(self, genome: T) -> Optional["GenomeConfig"]:
		"""
		Convert a genome to a GenomeConfig for tracking.

		Override in subclasses to enable genome-level tracking.
		Returns None by default (genome tracking disabled).
		"""
		return None

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

		# Store for use by _generate_offspring() hook
		self._batch_evaluate_fn = batch_evaluate_fn
		self._evaluate_fn = evaluate_fn

		# Create fitness calculator for unified ranking (from OptimizationConfig)
		fitness_calculator = cfg.create_fitness_calculator()
		self._fitness_calculator = fitness_calculator
		self._log.info(f"[{self.name}] Fitness calculator: {fitness_calculator.name}")

		# Threshold continuity: use initial_threshold from config if set (passed from previous phase)
		# Otherwise, fall back to min_accuracy (first phase)
		start_threshold = cfg.initial_threshold if cfg.initial_threshold is not None else cfg.min_accuracy
		# End threshold depends on actual generations vs reference (constant rate)
		actual_progress = min(1.0, cfg.generations / cfg.threshold_reference)
		end_threshold = start_threshold + actual_progress * cfg.threshold_delta

		# Helper to get current threshold based on progress
		def get_threshold(progress: float = 0.0) -> float:
			if not cfg.progressive_threshold:
				return start_threshold
			progress = max(0.0, min(1.0, progress))
			return start_threshold + progress * cfg.threshold_delta

		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.2%} → {end_threshold:.2%} (rate: {cfg.threshold_delta/cfg.threshold_reference:.4%}/gen)")

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

		# Find initial best using fitness calculator (unified ranking)
		init_tuples = [
			(i, fitness_values[i], accuracy_values[i] or 0.0)
			for i in range(len(population))
		]
		init_scores = fitness_calculator.fitness(init_tuples)
		best_idx = min(range(len(init_scores)), key=lambda i: init_scores[i])
		best = self.clone_genome(population[best_idx][0])
		best_fitness = fitness_values[best_idx]  # Report CE as fitness for compatibility
		initial_fitness = fitness_values[0] if initial_genome else best_fitness
		initial_accuracy = accuracy_values[best_idx]
		best_accuracy_val = initial_accuracy

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

		# Track previous best for delta computation
		prev_best_fitness = best_fitness

		shutdown_requested = False
		generation = 0
		for generation in range(cfg.generations):
			# Progressive threshold: gets stricter as generations progress
			current_threshold = get_threshold(generation / cfg.threshold_reference)
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
				)
			except StopIteration:
				shutdown_requested = True
				break

			# Selection and reproduction
			new_population: list[tuple[T, Optional[float], Optional[float]]] = []

			# Unified elitism: use fitness calculator to rank, keep top 20%
			n_elites = max(1, int(cfg.population_size * cfg.elitism_pct * 2))

			# Build (genome, ce, acc) tuples for fitness ranking
			pop_tuples = [
				(i, fitness_values[i], accuracy_values[i] or 0.0)
				for i in range(len(population))
			]
			combined_scores = fitness_calculator.fitness(pop_tuples)
			elite_sorted = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i])
			all_elite_indices = elite_sorted[:n_elites]
			total_elites = len(all_elite_indices)

			# Track initial elites (first generation only) for survival analysis
			if generation == 0:
				initial_elite_genomes = [
					(self.clone_genome(population[idx][0]), fitness_values[idx])
					for idx in all_elite_indices
				]
				self._log.info(f"[{self.name}] Elitism: {total_elites} by {fitness_calculator.name}")

			# Add elites to new population
			elite_width = len(str(total_elites))
			for i, elite_idx in enumerate(all_elite_indices):
				elite_genome = self.clone_genome(population[elite_idx][0])
				elite_fitness = fitness_values[elite_idx]
				elite_accuracy = accuracy_values[elite_idx]
				new_population.append((elite_genome, elite_fitness, elite_accuracy))

				acc_str = f", Acc={elite_accuracy:.2%}" if elite_accuracy is not None else ""
				self._log.debug(f"[Elite {i + 1:0{elite_width}d}/{total_elites}] CE={elite_fitness:.4f}{acc_str} (score={combined_scores[elite_idx]:.4f})")

			# Generate offspring via hook (overridable for Rust acceleration)
			needed_offspring = cfg.population_size - len(new_population)
			offspring = self._generate_offspring(population, needed_offspring, current_threshold, generation)

			# Combine elites with viable offspring
			population = new_population + offspring
			fitness_values = [f for _, f, _ in population]
			accuracy_values = [a for _, _, a in population]

			# Update best using fitness calculator (unified ranking)
			gen_tuples = [
				(i, fitness_values[i], accuracy_values[i] or 0.0)
				for i in range(len(population))
			]
			gen_scores = fitness_calculator.fitness(gen_tuples)
			gen_best_idx = min(range(len(gen_scores)), key=lambda i: gen_scores[i])
			if fitness_values[gen_best_idx] < best_fitness:
				best = self.clone_genome(population[gen_best_idx][0])
				best_fitness = fitness_values[gen_best_idx]
				best_accuracy_val = accuracy_values[gen_best_idx]

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

			# Record iteration to tracker (if set)
			if self._tracker and self._tracker_phase_id:
				try:
					best_acc = accuracy_values[gen_best_idx] if accuracy_values else None
					# Compute average accuracy of the population
					valid_accs = [a for a in accuracy_values if a is not None]
					avg_acc = sum(valid_accs) / len(valid_accs) if valid_accs else None

					# Get baseline and patience info for dashboard
					baseline_ce = early_stopper._best_fitness if hasattr(early_stopper, '_best_fitness') else None
					delta_baseline = (best_fitness - baseline_ce) if baseline_ce is not None else None
					delta_previous = best_fitness - prev_best_fitness
					patience_counter = early_stopper._patience_counter if hasattr(early_stopper, '_patience_counter') else 0
					candidates_total = len(offspring)  # In generic GA, all offspring are viable

					iteration_id = self._tracker.record_iteration(
						experiment_id=self._tracker_experiment_id,
						iteration_num=generation + 1,
						best_ce=best_fitness,
						best_accuracy=best_acc,
						avg_ce=gen_avg,
						avg_accuracy=avg_acc,
						elite_count=total_elites,
						offspring_count=len(offspring),
						offspring_viable=len(offspring),  # All offspring are viable at this point
						fitness_threshold=current_threshold,
						baseline_ce=baseline_ce,
						delta_baseline=delta_baseline,
						delta_previous=delta_previous,
						patience_counter=patience_counter,
						patience_max=cfg.patience,
						candidates_total=candidates_total,
					)

					# Record genome evaluations (if genome_to_config is implemented)
					if iteration_id and self._tracker_experiment_id and HAS_TRACKER and GenomeRole is not None:
						evaluations = []
						for pos, (genome, ce, acc) in enumerate(population):
							config = self.genome_to_config(genome)
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								# Role: first total_elites are elites, rest are offspring
								role = GenomeRole.ELITE if pos < total_elites else GenomeRole.OFFSPRING
								evaluations.append({
									"iteration_id": iteration_id,
									"genome_id": genome_id,
									"position": pos,
									"role": role,
									"ce": ce,
									"accuracy": acc if acc is not None else 0.0,
									"elite_rank": pos if pos < total_elites else None,
								})
						if evaluations:
							self._tracker.record_genome_evaluations_batch(evaluations)
				except Exception as e:
					self._log.debug(f"Tracker error: {e}")

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
					# Shrink population - keep best by fitness score
					shrink_tuples = [
						(i, fitness_values[i], accuracy_values[i] or 0.0)
						for i in range(len(population))
					]
					shrink_scores = fitness_calculator.fitness(shrink_tuples)
					shrink_order = sorted(range(len(shrink_scores)), key=lambda i: shrink_scores[i])
					keep_indices = shrink_order[:cfg.population_size]
					population = [population[i] for i in keep_indices]
					fitness_values = [fitness_values[i] for i in keep_indices]
					accuracy_values = [accuracy_values[i] for i in keep_indices]

			# Overfitting callback check (same interval as early stopping)
			if overfitting_callback is not None and (generation + 1) % cfg.check_interval == 0:
				control = overfitting_callback(best, best_fitness)
				if hasattr(control, 'early_stop') and control.early_stop:
					self._log.warning(f"[{self.name}] Overfitting early stop at gen {generation + 1}")
					# Return early with overfitting stop reason
					sorted_pop = sorted(population, key=lambda x: x[1])
					final_population = [self.clone_genome(g) for g, _, _ in sorted_pop]
					early_pop_metrics = [(ce, acc) for _, ce, acc in sorted_pop]
					improvement_pct = (initial_fitness - best_fitness) / initial_fitness * 100 if initial_fitness > 0 else 0
					# Compute final_threshold at current progress for next phase continuity
					current_final_threshold = get_threshold(generation / cfg.threshold_reference)
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
						population_metrics=early_pop_metrics,
						initial_accuracy=initial_accuracy,
						final_accuracy=accuracy_values[gen_best_idx] if accuracy_values else None,
						final_threshold=current_final_threshold,
					)

			# Update previous best for next iteration's delta computation
			prev_best_fitness = best_fitness

		# Get final best using fitness calculator (unified ranking)
		final_tuples = [
			(i, fitness_values[i], accuracy_values[i] or 0.0)
			for i in range(len(population))
		]
		final_scores = fitness_calculator.fitness(final_tuples)
		best_idx_final = min(range(len(final_scores)), key=lambda i: final_scores[i])
		final_accuracy = accuracy_values[best_idx_final] if accuracy_values else None

		# Extract final population for seeding next phase (sorted by fitness score)
		scored_pop = list(zip(population, final_scores))
		scored_pop.sort(key=lambda x: x[1])
		final_population = [self.clone_genome(g) for (g, _, _), _ in scored_pop]
		population_metrics = [(ce, acc) for (_, ce, acc), _ in scored_pop]

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
		final_threshold = get_threshold(generation / cfg.threshold_reference) if cfg.generations > 0 else start_threshold

		self._log.info(f"[{self.name}] Analysis Summary:")
		self._log.info(f"  CE improvement: {initial_fitness:.4f} → {best_fitness:.4f} ({(1 - best_fitness/initial_fitness)*100:+.2f}%)")
		self._log.info(f"  CE spread: {initial_ce_spread:.4f} → {final_ce_spread:.4f} ({diversity_change:+.4f})")
		self._log.info(f"  Elite survivals: {elite_survivals}/{len(initial_elite_genomes) if initial_elite_genomes else 0}")
		self._log.info(f"  Elite win rate: {elite_wins}/{total_gens} ({elite_win_rate:.1f}%)")
		self._log.info(f"  Improvement rate: {improved_iterations}/{total_gens} ({improvement_rate:.1f}%)")
		self._log.info(f"  Final threshold: {final_threshold:.2%} (for next phase)")

		improvement_pct = (initial_fitness - best_fitness) / initial_fitness * 100 if initial_fitness > 0 else 0

		# Determine stop reason
		if shutdown_requested:
			stop_reason = StopReason.SHUTDOWN
		elif early_stopper.patience_exhausted:
			stop_reason = StopReason.CONVERGENCE
		else:
			stop_reason = None

		return OptimizerResult(
			initial_genome=initial_genome if initial_genome else population[0][0],
			best_genome=best,
			initial_fitness=initial_fitness,
			final_fitness=best_fitness,
			improvement_percent=improvement_pct,
			iterations_run=generation + 1,
			method_name=self.name,
			history=history,
			early_stopped=early_stopper.patience_exhausted or shutdown_requested,
			stop_reason=stop_reason,
			final_population=final_population,
			population_metrics=population_metrics,
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

		# Batch evaluate - returns (CE, accuracy[, bit_acc]) tuples
		if batch_fn is not None:
			results = batch_fn(to_eval)
			new_fitness = [r[0] for r in results]
			new_accuracy = [r[1] for r in results]
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
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
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
			generation: Current generation/iteration number (passed to batch_fn for logging)
			total_generations: Total generations/iterations (passed to batch_fn for logging)

		Returns:
			List of (genome, fitness, accuracy) tuples for viable candidates
		"""
		viable: list[tuple[T, float, Optional[float]]] = []
		filtered_count = 0

		# Extra kwargs for batch evaluation (generation/total for per-batch logging)
		batch_kwargs: dict = {"min_accuracy": min_accuracy}
		if generation is not None:
			batch_kwargs["generation"] = generation
		if total_generations is not None:
			batch_kwargs["total_generations"] = total_generations

		import time as _time

		# First, evaluate seed genomes if provided (always accepted — they're explicit seeds)
		if seed_genomes:
			to_eval = [self.clone_genome(g) for g in seed_genomes[:target_size]]
			self._log.info(f"[{self.name}] Evaluating {len(to_eval)} seed genomes...")
			t0 = _time.time()
			if batch_fn is not None:
				results = batch_fn(to_eval, **batch_kwargs)
				elapsed = _time.time() - t0
				best_ce = min(r[0] for r in results) if results else 0.0
				best_acc = max(r[1] for r in results if r[1] is not None) if results else 0.0
				self._log.info(f"[{self.name}] Seed eval: {len(to_eval)} genomes in {elapsed:.1f}s (best CE={best_ce:.4f}, Acc={best_acc:.2%})")
				for genome, r in zip(to_eval, results):
					viable.append((genome, r[0], r[1]))
			else:
				for genome in to_eval:
					ce = single_fn(genome)
					viable.append((genome, ce, None))
			self._log.info(f"[{self.name}] {len(viable)}/{target_size} viable after seed eval")

		# Generate new candidates until we have enough
		attempt = 0
		while len(viable) < target_size and attempt < max_attempts:
			attempt += 1
			needed = target_size - len(viable)
			# Generate a batch of candidates (extra to account for filtering)
			batch_size = min(needed * 2, needed + 10)  # Generate extra
			candidates = [generator_fn() for _ in range(batch_size)]

			self._log.info(f"[{self.name}] Building population: attempt {attempt}, evaluating {batch_size} candidates ({len(viable)}/{target_size} viable)")
			t0 = _time.time()
			# Evaluate
			if batch_fn is not None:
				results = batch_fn(candidates, **batch_kwargs)
				elapsed = _time.time() - t0
				self._log.info(f"[{self.name}] Batch eval: {len(candidates)} candidates in {elapsed:.1f}s")
				for genome, r in zip(candidates, results):
					acc = r[1]
					ce = r[0]
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
		# Tracker for iteration recording (set via set_tracker)
		self._tracker: Optional["ExperimentTracker"] = None
		self._tracker_experiment_id: Optional[int] = None

	def set_tracker(self, tracker: "ExperimentTracker", experiment_id: int, _unused: Optional[int] = None) -> None:
		"""Set the experiment tracker for iteration recording."""
		self._tracker = tracker
		self._tracker_experiment_id = experiment_id

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
	# Optional genome tracking (subclasses can override)
	# =========================================================================

	def genome_to_config(self, genome: T) -> Optional["GenomeConfig"]:
		"""
		Convert a genome to a GenomeConfig for tracking.

		Override in subclasses to enable genome-level tracking.
		Returns None by default (genome tracking disabled).
		"""
		return None

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
		Default: Python single-path — all N neighbors from best-ranked genome.

		Args:
			best_genome: Best genome by fitness ranking (used as mutation source)
			n_neighbors: Number of neighbors to generate
			threshold: Minimum accuracy threshold
			iteration: Current iteration number
			tabu_list: Tabu list (deque) for move tracking

		Returns list of (genome, ce, accuracy) tuples for viable neighbors.
		"""
		cfg = self._config

		# Generate N candidates from best genome
		candidates: list[tuple[T, Any]] = []
		for _ in range(n_neighbors):
			neighbor, move = self.mutate_genome(self.clone_genome(best_genome), cfg.mutation_rate)
			if not self.is_tabu_move(move, tabu_list):
				candidates.append((neighbor, move))

		if not candidates:
			return []

		# Evaluate all candidates (pass iteration info for per-batch logging)
		if self._batch_evaluate_fn is not None:
			to_eval = [n for n, _ in candidates]
			results = self._batch_evaluate_fn(
				to_eval, min_accuracy=threshold,
				generation=iteration, total_generations=self._config.iterations,
			)
			fitness_values = [r[0] for r in results]
			accuracy_values = [r[1] for r in results]
		else:
			fitness_values = [self._evaluate_fn(n) for n, _ in candidates]
			accuracy_values = [None] * len(candidates)

		# Build evaluated list: (genome, move, ce, acc)
		evaluated = [(n, m, f, a) for (n, m), f, a in zip(candidates, fitness_values, accuracy_values)]

		# Filter by threshold
		viable = [(n, m, f, a) for n, m, f, a in evaluated if a is None or a >= threshold]
		if not viable:
			viable = sorted(evaluated, key=lambda x: x[2])[:1]

		# Update tabu list with best neighbor's move
		if viable:
			# Use fitness calculator to find best neighbor for tabu tracking
			viable_3t = [(n, f, a or 0.0) for n, m, f, a in viable]
			ranked = self._fitness_calculator.rank(viable_3t)
			best_idx = next(
				i for i, (n, m, f, a) in enumerate(viable)
				if n is ranked[0][0]
			)
			_, m, _, _ = viable[best_idx]
			if m is not None:
				tabu_list.append(m)

		# Return viable neighbors as 3-tuples (genome, ce, accuracy)
		return [(n, f, a) for n, _, f, a in viable]

	def _on_iteration_start(self, iteration: int, **ctx) -> None:
		"""Hook called at start of each iteration.

		Override for Metal cleanup, shutdown check, etc.
		Raise StopIteration to stop the optimization loop gracefully.

		ctx keys: best_genome, best_fitness, best_accuracy, threshold
		"""
		pass

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
		Run Tabu Search optimization with fitness-ranked single-path search.

		Each iteration generates N neighbors from the best-ranked genome
		(ranked by the configured fitness calculator: CE, HarmonicRank, etc.).

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

		# Store for use by _generate_neighbors() hook
		self._batch_evaluate_fn = batch_evaluate_fn
		self._evaluate_fn = evaluate_fn

		# Create fitness calculator for unified ranking (from OptimizationConfig)
		fitness_calculator = cfg.create_fitness_calculator()
		self._fitness_calculator = fitness_calculator
		self._log.info(f"[{self.name}] Fitness calculator: {fitness_calculator.name}")

		# Threshold continuity: use initial_threshold from config if set (passed from previous phase)
		# Otherwise, fall back to min_accuracy (first phase)
		start_threshold = cfg.initial_threshold if cfg.initial_threshold is not None else cfg.min_accuracy
		# End threshold depends on actual iterations vs reference (constant rate)
		actual_progress = min(1.0, cfg.iterations / cfg.threshold_reference)
		end_threshold = start_threshold + actual_progress * cfg.threshold_delta

		# Helper to get current threshold based on progress
		def get_threshold(progress: float = 0.0) -> float:
			if not cfg.progressive_threshold:
				return start_threshold
			progress = max(0.0, min(1.0, progress))
			return start_threshold + progress * cfg.threshold_delta

		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.2%} → {end_threshold:.2%} (rate: {cfg.threshold_delta/cfg.threshold_reference:.4%}/iter)")

		# Cache size for total_neighbors
		cache_size = cfg.total_neighbors_size or cfg.neighbors_per_iter

		# Best genome tracking (single path by fitness ranking)
		best_ranked_genome = self.clone_genome(initial_genome)
		best_ranked_ce = initial_fitness
		best_ranked_accuracy: Optional[float] = None

		# Global best (by CE for return value / early stopping)
		best = self.clone_genome(initial_genome)
		best_fitness = initial_fitness
		best_accuracy: Optional[float] = None
		start_fitness = initial_fitness

		# Single tabu list
		tabu_list: deque = deque(maxlen=cfg.tabu_size)

		# All neighbors cache: genomes with their CE and Acc
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
				seed_fitness = [r[0] for r in results]
				seed_accuracy = [r[1] for r in results]
			else:
				seed_fitness = [evaluate_fn(g) for g in initial_neighbors]
				seed_accuracy = [None] * len(initial_neighbors)

			# Add all seed neighbors to the cache
			for g, f, a in zip(initial_neighbors, seed_fitness, seed_accuracy):
				all_neighbors.append((self.clone_genome(g), f, a))

			# Find best by fitness ranking
			valid_for_rank = [(g, ce, acc or 0.0) for g, ce, acc in all_neighbors if acc is not None]
			if valid_for_rank:
				ranked = fitness_calculator.rank(valid_for_rank)
				best_ranked_obj = ranked[0][0]
				best_ranked_genome = self.clone_genome(best_ranked_obj)
				# Look up CE/acc from the original tuples
				for g, ce, acc in valid_for_rank:
					if g is best_ranked_obj:
						best_ranked_ce = ce
						best_ranked_accuracy = acc
						break
			else:
				# Fallback: best by CE
				best_ce_idx = min(range(len(seed_fitness)), key=lambda i: seed_fitness[i])
				best_ranked_genome = self.clone_genome(initial_neighbors[best_ce_idx])
				best_ranked_ce = seed_fitness[best_ce_idx]
				best_ranked_accuracy = seed_accuracy[best_ce_idx]

			# Update global best (by CE)
			best_ce_idx = min(range(len(all_neighbors)), key=lambda i: all_neighbors[i][1])
			if all_neighbors[best_ce_idx][1] < best_fitness:
				best = self.clone_genome(all_neighbors[best_ce_idx][0])
				best_fitness = all_neighbors[best_ce_idx][1]
				best_accuracy = all_neighbors[best_ce_idx][2]

			# Log seed summary
			best_acc_val = max((a for _, _, a in all_neighbors if a is not None), default=None)
			self._log.info(f"[{self.name}] Seed: best_ce={best_fitness:.4f}, best_acc={best_acc_val:.2%}" if best_acc_val else
						   f"[{self.name}] Seed: best_ce={best_fitness:.4f}, best_acc=N/A")

		history = [(0, best_fitness)]

		# Analysis tracking
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
		diversity_str = f", diversity={cfg.diversity_sources_pct:.0%}" if cfg.diversity_sources_pct > 0 else ""
		self._log.info(f"[{self.name}] Config: neighbors={cfg.neighbors_per_iter} (single path by {fitness_calculator.name}), "
					   f"iters={cfg.iterations}, cache={cache_size}{diversity_str}")

		# Track threshold changes
		prev_threshold: Optional[float] = None

		# Track previous best for delta computation
		prev_best_fitness = best_fitness

		shutdown_requested = False
		iteration = 0
		for iteration in range(cfg.iterations):
			# Progressive threshold
			current_threshold = get_threshold(iteration / cfg.threshold_reference)
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
				)
			except StopIteration:
				shutdown_requested = True
				break

			# Generate neighbors — cooperative multi-start or single-source
			if cfg.diversity_sources_pct > 0 and len(all_neighbors) > 1:
				# Cooperative multi-start: top N% of cache as equal reference set
				n_sources = max(1, int(len(all_neighbors) * cfg.diversity_sources_pct))
				# Rank by fitness to pick the reference set
				valid_for_div = [(g, ce, acc or 0.0) for g, ce, acc in all_neighbors if acc is not None]
				if valid_for_div:
					ranked_div = fitness_calculator.rank(valid_for_div)
					sources = [self.clone_genome(g) for g, _ in ranked_div[:n_sources]]
				else:
					sources = [self.clone_genome(all_neighbors[0][0])]
					n_sources = 1

				# Equal share per source, total capped at neighbors_per_iter
				total_nbrs = cfg.neighbors_per_iter
				per_source = max(1, total_nbrs // n_sources)
				remainder = total_nbrs - (per_source * n_sources)

				all_generated: list[tuple] = []
				for si, source in enumerate(sources):
					n = per_source + (1 if si < remainder else 0)
					batch = self._generate_neighbors(
						best_genome=source, n_neighbors=n,
						threshold=current_threshold, iteration=iteration,
						tabu_list=tabu_list,
					)
					all_generated.extend(batch)
				viable = all_generated
			else:
				# Classic TS: all neighbors from single best-ranked genome
				viable = self._generate_neighbors(
					best_genome=best_ranked_genome,
					n_neighbors=cfg.neighbors_per_iter,
					threshold=current_threshold,
					iteration=iteration,
					tabu_list=tabu_list,
				)

			if not viable:
				continue

			# Add all viable to the total neighbors cache
			for n, f, a in viable:
				all_neighbors.append((self.clone_genome(n), f, a))

			# Cap cache by fitness ranking
			if len(all_neighbors) > cache_size * 2:
				valid_for_cap = [(g, ce, acc or 0.0) for g, ce, acc in all_neighbors if acc is not None]
				if valid_for_cap:
					ranked_for_cap = fitness_calculator.rank(valid_for_cap)
					# Build a lookup: genome identity -> (ce, acc) from original tuples
					genome_data = {id(g): (ce, acc) for g, ce, acc in valid_for_cap}
					all_neighbors = [
						(g, genome_data[id(g)][0], genome_data[id(g)][1])
						for g, _ in ranked_for_cap[:cache_size]
					]
				else:
					all_neighbors = sorted(all_neighbors, key=lambda x: x[1])[:cache_size]

			# Update best_ranked by fitness ranking (includes all neighbors)
			valid_for_rank = [(g, ce, acc or 0.0) for g, ce, acc in all_neighbors if acc is not None]
			if valid_for_rank:
				ranked = fitness_calculator.rank(valid_for_rank)
				best_ranked_obj = ranked[0][0]
				best_ranked_genome = self.clone_genome(best_ranked_obj)
				# Look up CE/acc from original tuples
				for g, ce, acc in valid_for_rank:
					if g is best_ranked_obj:
						best_ranked_ce = ce
						best_ranked_accuracy = acc
						break

			# Update global best (by CE)
			for n, f, a in viable:
				if f < best_fitness:
					best = self.clone_genome(n)
					best_fitness = f
					best_accuracy = a
					improved_iterations += 1

			history.append((iteration + 1, best_fitness))

			# Log progress
			iter_width = len(str(cfg.iterations))
			ranked_acc_str = f"{best_ranked_accuracy:.2%}" if best_ranked_accuracy is not None else "N/A"
			self._log.info(f"[{self.name}] Iter {iteration + 1:0{iter_width}d}/{cfg.iterations}: "
						   f"best_ranked=(CE={best_ranked_ce:.4f}, Acc={ranked_acc_str}), "
						   f"best_ce={best_fitness:.4f}")

			# Record iteration to tracker (if set)
			if self._tracker and self._tracker_phase_id:
				try:
					# Compute top-k stats from all_neighbors by fitness ranking
					valid_neighbors = [(g, ce, acc) for g, ce, acc in all_neighbors if acc is not None]
					if valid_neighbors:
						ranked_for_stats = fitness_calculator.rank(valid_neighbors)
						top_k = ranked_for_stats[:cache_size]
						top_k_count = len(top_k)
						top_k_avg_ce = sum(ce for (_, ce, _) in valid_neighbors[:top_k_count]) / top_k_count
						valid_accs = [acc for _, _, acc in valid_neighbors[:top_k_count] if acc is not None]
						top_k_avg_acc = sum(valid_accs) / len(valid_accs) if valid_accs else None
					else:
						top_k_count = len(all_neighbors)
						top_k_avg_ce = sum(f for _, f, _ in all_neighbors) / len(all_neighbors) if all_neighbors else None
						top_k_avg_acc = None

					# Get baseline and patience info for dashboard
					baseline_ce = early_stopper._best_fitness if hasattr(early_stopper, '_best_fitness') else None
					delta_baseline = (best_fitness - baseline_ce) if baseline_ce is not None else None
					delta_previous = best_fitness - prev_best_fitness
					patience_counter = early_stopper._patience_counter if hasattr(early_stopper, '_patience_counter') else 0

					iteration_id = self._tracker.record_iteration(
						experiment_id=self._tracker_experiment_id,
						iteration_num=iteration + 1,
						best_ce=best_fitness,
						best_accuracy=best_accuracy,
						avg_ce=top_k_avg_ce,
						avg_accuracy=top_k_avg_acc,
						elite_count=top_k_count,
						offspring_count=len(viable),
						offspring_viable=len(viable),
						fitness_threshold=current_threshold,
						baseline_ce=baseline_ce,
						delta_baseline=delta_baseline,
						delta_previous=delta_previous,
						patience_counter=patience_counter,
						patience_max=cfg.patience,
						candidates_total=len(all_neighbors),
					)

					# Record genome evaluations (if genome_to_config is implemented)
					if iteration_id and self._tracker_experiment_id and HAS_TRACKER and GenomeRole is not None:
						evaluations = []
						top_k_ids = set()

						# Record current best genome
						best_config = self.genome_to_config(best)
						if best_config is not None:
							best_genome_id = self._tracker.get_or_create_genome(
								self._tracker_experiment_id, best_config
							)
							evaluations.append({
								"iteration_id": iteration_id,
								"genome_id": best_genome_id,
								"position": 0,
								"role": GenomeRole.CURRENT,
								"ce": best_fitness,
								"accuracy": best_accuracy if best_accuracy is not None else 0.0,
								"elite_rank": 0,
							})
							top_k_ids.add(best_genome_id)

						# Record top-k neighbors
						for pos, (genome, ce, acc) in enumerate(all_neighbors[:cache_size]):
							config = self.genome_to_config(genome)
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								if genome_id not in top_k_ids:
									evaluations.append({
										"iteration_id": iteration_id,
										"genome_id": genome_id,
										"position": pos + 1,
										"role": GenomeRole.TOP_K,
										"ce": ce,
										"accuracy": acc if acc is not None else 0.0,
										"elite_rank": pos + 1,
									})
									top_k_ids.add(genome_id)

						# Record new viable neighbors
						for pos, (genome, ce, acc) in enumerate(viable):
							config = self.genome_to_config(genome)
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								if genome_id not in top_k_ids:
									evaluations.append({
										"iteration_id": iteration_id,
										"genome_id": genome_id,
										"position": cache_size + pos + 1,
										"role": GenomeRole.NEIGHBOR,
										"ce": ce,
										"accuracy": acc if acc is not None else 0.0,
									})

						if evaluations:
							self._tracker.record_genome_evaluations_batch(evaluations)
				except Exception as e:
					self._log.debug(f"Tracker error: {e}")

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

		# === Build final_population: top K by fitness (unified ranking) ===
		neighbor_tuples = [
			(i, f, a or 0.0) for i, (g, f, a) in enumerate(all_neighbors)
		]
		neighbor_scores = fitness_calculator.fitness(neighbor_tuples)
		scored = sorted(range(len(neighbor_scores)), key=lambda i: neighbor_scores[i])
		top_indices = scored[:cache_size]
		final_population = [self.clone_genome(all_neighbors[i][0]) for i in top_indices]
		population_metrics = [(all_neighbors[i][1], all_neighbors[i][2] or 0.0) for i in top_indices]

		# Final threshold (for next phase)
		final_threshold = get_threshold(iteration / cfg.threshold_reference) if cfg.iterations > 0 else start_threshold

		# Log analysis summary
		total_iters = iteration + 1
		self._log.info(f"[{self.name}] Analysis Summary:")
		self._log.info(f"  CE improvement: {start_fitness:.4f} → {best_fitness:.4f} ({(1 - best_fitness/start_fitness)*100:+.2f}%)")
		self._log.info(f"  Improved iterations: {improved_iterations}/{total_iters}")
		self._log.info(f"  Final population: {len(final_population)} by {fitness_calculator.name}")
		self._log.info(f"  Final threshold: {final_threshold:.2%} (for next phase)")

		improvement_pct = (start_fitness - best_fitness) / start_fitness * 100 if start_fitness > 0 else 0

		# Determine stop reason
		if shutdown_requested:
			stop_reason = StopReason.SHUTDOWN
		elif early_stopper.patience_exhausted:
			stop_reason = StopReason.CONVERGENCE
		else:
			stop_reason = None

		return OptimizerResult(
			initial_genome=initial_genome,
			best_genome=best,
			initial_fitness=start_fitness,
			final_fitness=best_fitness,
			improvement_percent=improvement_pct,
			iterations_run=iteration + 1,
			method_name=self.name,
			history=history,
			early_stopped=early_stopper.patience_exhausted or shutdown_requested,
			stop_reason=stop_reason,
			final_population=final_population,
			population_metrics=population_metrics,
			initial_accuracy=None,
			final_accuracy=best_accuracy,
			final_threshold=final_threshold,
		)
