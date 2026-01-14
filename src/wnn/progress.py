"""
Progress tracking utilities for optimization algorithms.

Provides reusable classes to track, calculate, and log optimization
progress metrics in a standardized way.
"""

from typing import List, Optional, Callable, Protocol, TypeVar, Generic
from dataclasses import dataclass, field


class HasFitness(Protocol):
	"""Protocol for objects that have a fitness value."""
	@property
	def fitness(self) -> float: ...


T = TypeVar('T')


@dataclass
class ProgressStats:
	"""Statistics for a single tick/generation."""
	generation: int
	best_global: float
	best_current: float
	avg_current: float
	worst_current: float
	improved: bool = False


class ProgressTracker(Generic[T]):
	"""
	Tracks optimization progress and logs standardized metrics.

	Designed to be reusable across different optimization algorithms
	(GA, simulated annealing, etc.) with consistent logging format.

	Usage:
		# With Logger instance
		tracker = ProgressTracker(
			logger=my_logger,
			minimize=True,  # Lower is better (e.g., cross-entropy)
		)

		# During optimization loop
		for gen in range(generations):
			fitness_values = [evaluate(g) for g in population]
			tracker.tick(fitness_values, generation=gen)

		# Get summary
		summary = tracker.summary()

	The tracker will log lines like:
		[Gen 1/50] best=10.49, current=10.52, avg=10.55, improved=True
	"""

	def __init__(
		self,
		logger: Optional[Callable[[str], None]] = None,
		minimize: bool = True,
		prefix: str = "",
		total_generations: Optional[int] = None,
	):
		"""
		Initialize progress tracker.

		Args:
			logger: Callable that logs messages (e.g., Logger instance, print)
			minimize: If True, lower fitness is better (default for loss/CE)
			prefix: Prefix for log messages (e.g., "[GA]")
			total_generations: Total expected generations (for progress display)
		"""
		self._log = logger or print
		self._minimize = minimize
		self._prefix = prefix + " " if prefix else ""
		self._total = total_generations

		# State
		self._best_global: Optional[float] = None
		self._best_generation: int = 0
		self._history: List[ProgressStats] = []

	def tick(
		self,
		fitness_values: List[float],
		generation: Optional[int] = None,
		log: bool = True,
	) -> ProgressStats:
		"""
		Record a tick (generation) of fitness values.

		Args:
			fitness_values: List of fitness values for current population
			generation: Current generation number (auto-incremented if None)
			log: Whether to log progress

		Returns:
			ProgressStats for this tick
		"""
		if not fitness_values:
			raise ValueError("fitness_values cannot be empty")

		gen = generation if generation is not None else len(self._history)

		# Calculate current stats
		if self._minimize:
			best_current = min(fitness_values)
			worst_current = max(fitness_values)
		else:
			best_current = max(fitness_values)
			worst_current = min(fitness_values)

		avg_current = sum(fitness_values) / len(fitness_values)

		# Update global best
		improved = False
		if self._best_global is None:
			self._best_global = best_current
			self._best_generation = gen
			improved = True
		elif self._minimize and best_current < self._best_global:
			self._best_global = best_current
			self._best_generation = gen
			improved = True
		elif not self._minimize and best_current > self._best_global:
			self._best_global = best_current
			self._best_generation = gen
			improved = True

		# Create stats
		stats = ProgressStats(
			generation=gen,
			best_global=self._best_global,
			best_current=best_current,
			avg_current=avg_current,
			worst_current=worst_current,
			improved=improved,
		)
		self._history.append(stats)

		# Log if requested
		if log:
			self._log_tick(stats)

		return stats

	def _log_tick(self, stats: ProgressStats) -> None:
		"""Log a single tick."""
		gen_str = f"Gen {stats.generation + 1}"
		if self._total:
			gen_str = f"Gen {stats.generation + 1}/{self._total}"

		improved_str = " *" if stats.improved else ""

		self._log(
			f"{self._prefix}[{gen_str}] "
			f"best={stats.best_global:.4f}, "
			f"current={stats.best_current:.4f}, "
			f"avg={stats.avg_current:.4f}{improved_str}"
		)

	def tick_individual(
		self,
		fitness: float,
		index: int,
		total: int,
		phase: str = "Evaluating",
	) -> None:
		"""
		Log progress for individual evaluation (during population eval).

		Args:
			fitness: Fitness value for this individual
			index: Current index (0-based)
			total: Total number of individuals
			phase: Description of current phase
		"""
		self._log(f"  [{index + 1}/{total}] CE={fitness:.4f}")

	@property
	def best_global(self) -> Optional[float]:
		"""Best fitness value seen so far."""
		return self._best_global

	@property
	def best_generation(self) -> int:
		"""Generation where best fitness was found."""
		return self._best_generation

	@property
	def history(self) -> List[ProgressStats]:
		"""Full history of progress stats."""
		return self._history.copy()

	@property
	def generations_run(self) -> int:
		"""Number of generations completed."""
		return len(self._history)

	def summary(self) -> dict:
		"""Get summary statistics."""
		if not self._history:
			return {"generations": 0}

		first = self._history[0]
		last = self._history[-1]

		if self._minimize:
			improvement = (first.best_current - last.best_global) / first.best_current * 100
		else:
			improvement = (last.best_global - first.best_current) / first.best_current * 100

		return {
			"generations": len(self._history),
			"initial_fitness": first.best_current,
			"final_fitness": last.best_global,
			"improvement_pct": improvement,
			"best_generation": self._best_generation,
			"improvements": sum(1 for s in self._history if s.improved),
		}

	def log_summary(self) -> None:
		"""Log a summary of the optimization run."""
		s = self.summary()
		if s["generations"] == 0:
			self._log(f"{self._prefix}No generations completed")
			return

		self._log(f"{self._prefix}Summary:")
		self._log(f"  Generations: {s['generations']}")
		self._log(f"  Initial: {s['initial_fitness']:.4f}")
		self._log(f"  Final: {s['final_fitness']:.4f}")
		self._log(f"  Improvement: {s['improvement_pct']:.2f}%")
		self._log(f"  Best at generation: {s['best_generation'] + 1}")
		self._log(f"  Total improvements: {s['improvements']}")


class PopulationTracker(ProgressTracker):
	"""
	Extended tracker for population-based algorithms.

	Adds support for tracking population diversity and individual evaluations.
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._current_eval_count = 0
		self._current_eval_total = 0

	def start_population_eval(self, population_size: int, phase: str = "Evaluating") -> None:
		"""Start tracking individual evaluations for a population."""
		self._current_eval_count = 0
		self._current_eval_total = population_size
		self._log(f"{self._prefix}{phase} {population_size} individuals...")

	def record_individual(self, fitness: float) -> None:
		"""Record an individual evaluation."""
		self._current_eval_count += 1
		self._log(
			f"  [{self._current_eval_count}/{self._current_eval_total}] "
			f"fitness={fitness:.4f}"
		)
