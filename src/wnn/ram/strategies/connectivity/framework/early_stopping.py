"""Early stopping: patience tracking with improvement-threshold checks."""

from dataclasses import dataclass
from typing import Callable, Optional

from wnn.ram.strategies.connectivity.framework.adaptive_scaling import AdaptiveLevel


@dataclass
class EarlyStoppingConfig:
	"""Configuration for early stopping with patience."""
	patience: int = 5              # Number of checks without improvement before stopping
	check_interval: int = 5        # Check every N iterations/generations
	min_improvement_pct: float = 0.02  # Minimum % improvement required to reset patience
	# Magnitude-aware patience (controller redesign (a), 16/06/2026). See
	# OptimizationConfig.magnitude_aware_patience for the rationale. These mirror
	# the knobs there and only take effect via check_magnitude().
	magnitude_aware: bool = False
	mag_eps_err: float = 0.5            # ε_err floor (deg)
	mag_stable_offset: float = 0.05     # s0 additive offset
	mag_delta: float = 0.05             # δ noise gate
	mag_rho_cap: float = 0.0            # 0 ⇒ use `patience` as the cap


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
		# Magnitude-aware watermarks (best err° seen / best stable% seen). Seeded
		# lazily on the first check_magnitude() call (we don't have err°/stable%
		# here — only fitness). None ⇒ "first check, nothing to compare yet".
		self._best_err_deg: Optional[float] = None
		self._best_stable: Optional[float] = None

	def restore(self, patience_counter: int) -> None:
		"""Restore checkpointed patience after reset() — the explicit resume
		counterpart (callers used to poke _patience_counter directly)."""
		self._patience_counter = max(0, int(patience_counter))

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

	def check_magnitude(self, iteration: int, best_err_deg: float, best_stable: float) -> bool:
		"""Magnitude-aware early-stop check (controller redesign (a)).

		Unlike check() — which watches the rank-based WHM and recovers exactly 1
		patience per improving check — this watches the controller's PHYSICAL
		metrics (err° down, stable% up) and recovers patience PROPORTIONAL to the
		size of the real gain, so a genuine jump (e.g. stable 20%→70%) keeps the
		search alive instead of being lost in the rank nudge.

		ρ_err = best_err_prev / max(err_cur, ε_err)        (>1 when err drops; halving → 2)
		ρ_stb = (stb_cur + s0) / (best_stb_prev + s0)       (additive s0 tames stb=0; 20→70% → ~3.5)
		ρ     = min(max(ρ_err, ρ_stb), ρ_cap)               (biggest real gain drives recovery)
		ρ ≥ 1+δ → counter -= ρ (floored at 0); else counter += 1 (drain by the floor).

		Watermarks ratchet independently (best_err = min, best_stable = max), so a
		single-metric regression can't poison the other metric's reference.
		"""
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveLevel
		cfg = self._config

		# Only check at specified intervals (1-indexed iteration).
		if (iteration + 1) % cfg.check_interval != 0:
			return False

		# First check: seed the watermarks, nothing to compare against yet.
		if self._best_err_deg is None or self._best_stable is None:
			self._best_err_deg = best_err_deg
			self._best_stable = best_stable
			self._last_level = AdaptiveLevel.NEUTRAL
			return False

		eps_err = cfg.mag_eps_err
		s0 = cfg.mag_stable_offset
		rho_cap = cfg.mag_rho_cap if cfg.mag_rho_cap > 0 else float(cfg.patience)

		rho_err = self._best_err_deg / max(best_err_deg, eps_err)
		rho_stb = (best_stable + s0) / (self._best_stable + s0)
		rho = min(max(rho_err, rho_stb), rho_cap)

		if rho >= 1.0 + cfg.mag_delta:
			# Genuine improvement → recover patience by the ratio (kept as float).
			self._patience_counter = max(0.0, self._patience_counter - rho)
		else:
			self._patience_counter += 1

		# Ratchet watermarks independently (monotone min / max).
		self._best_err_deg = min(self._best_err_deg, best_err_deg)
		self._best_stable = max(self._best_stable, best_stable)

		# Map the recovery ratio to an AdaptiveLevel for the adaptive scaler.
		if rho >= 1.5:
			level = AdaptiveLevel.HEALTHY
		elif rho >= 1.0 + cfg.mag_delta:
			level = AdaptiveLevel.NEUTRAL
		elif rho >= 1.0:
			level = AdaptiveLevel.WARNING
		else:
			level = AdaptiveLevel.CRITICAL  # got worse on both metrics
		self._last_level = level

		remaining = cfg.patience - self._patience_counter
		display = self._LEVEL_DISPLAY[level.name]
		self._log(
			f"[{self._method_name}] Early stop check (magnitude): "
			f"ρ={rho:.2f} (err={best_err_deg:.2f}°→best {self._best_err_deg:.2f}°, "
			f"stable={best_stable:.1%}→best {self._best_stable:.1%}), "
			f"patience={remaining:.1f}/{cfg.patience} {display}"
		)

		if self._patience_counter >= cfg.patience:
			self._log(
				f"[{self._method_name}] Early stop: no magnitude improvement "
				f"(ρ<{1.0 + cfg.mag_delta:.2f}) — patience exhausted"
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
