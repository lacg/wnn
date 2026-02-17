"""
Template Method base class for all optimization strategies.

Implements the GoF Template Method pattern: the algorithm skeleton lives in
the concrete optimize() method; subclasses override hook methods for
strategy-specific behavior (GA loop, TS loop, grid eval, adaptation).

Shared infrastructure extracted from GenericGAStrategy, GenericTSStrategy,
GridSearchStrategy, and AdaptationStrategy:
- Constructor boilerplate (config, seed, logger, rng, tracker)
- Fitness calculator setup
- Early stopping setup
- Population seeding with cached eval reuse between phases
- Iteration recording to tracker
- Result building (OptimizerResult)
- Progressive threshold computation
- Stop reason determination
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar, Any

from wnn.ram.fitness import FitnessCalculatorType, FitnessCalculatorFactory

# Optional tracker integration
try:
	from wnn.ram.experiments.tracker import ExperimentTracker, GenomeConfig, GenomeRole
	HAS_TRACKER = True
except ImportError:
	HAS_TRACKER = False
	ExperimentTracker = None
	GenomeConfig = None
	GenomeRole = None

from wnn.ram.strategies.connectivity.generic_strategies import (
	OptimizationConfig,
	OptimizationLogger,
	OptimizerResult,
	StopReason,
	EarlyStoppingConfig,
	EarlyStoppingTracker,
	AdaptiveScaler,
)

# Generic genome type
T = TypeVar('T')


class OptimizationTemplate(ABC, Generic[T]):
	"""
	Template Method base for all optimization strategies.

	The algorithm skeleton is a concrete optimize() method that orchestrates:
	1. Pre-optimize hook (Rust setup, evaluator wrapping)
	2. Population seeding with cached eval reuse
	3. Fitness calculator + early stopping setup
	4. Strategy-specific optimization loop (abstract)
	5. Result building
	6. Post-optimize hook (validation summary, cleanup)

	Subclasses MUST implement:
	- _run_optimization_loop(): The strategy-specific optimization logic
	- clone_genome(): Deep copy a genome

	Subclasses MAY override:
	- _pre_optimize(): Setup before the loop (Rust state, evaluator wrapping)
	- _post_optimize(): Cleanup/validation after the loop
	- genome_to_config(): Convert genome to GenomeConfig for tracker
	- _on_iteration_start(): Hook at each iteration start
	"""

	def __init__(
		self,
		config: OptimizationConfig,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		log_level: int = logging.DEBUG,
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		self._config = config
		self._seed = seed
		self._log = OptimizationLogger(self.name, level=log_level, file_logger=logger)
		self._rng: Optional[random.Random] = None
		self._shutdown_check = shutdown_check
		# Tracker for iteration recording (set via set_tracker)
		self._tracker: Optional["ExperimentTracker"] = None
		self._tracker_experiment_id: Optional[int] = None
		# Fitness calculator (set during optimize)
		self._fitness_calculator = None

	def set_tracker(self, tracker: "ExperimentTracker", experiment_id: int, _unused: Optional[int] = None) -> None:
		"""Set the experiment tracker for iteration recording."""
		self._tracker = tracker
		self._tracker_experiment_id = experiment_id

	@property
	def config(self) -> OptimizationConfig:
		return self._config

	@property
	@abstractmethod
	def name(self) -> str:
		"""Strategy name for logging."""
		...

	def _ensure_rng(self) -> None:
		if self._rng is None:
			self._rng = random.Random(self._seed)

	# =========================================================================
	# Abstract methods — subclasses MUST implement
	# =========================================================================

	@abstractmethod
	def clone_genome(self, genome: T) -> T:
		"""Create a deep copy of the genome."""
		...

	@abstractmethod
	def _run_optimization_loop(
		self,
		population: list[tuple[T, float, Optional[float]]],
		fitness_calculator: Any,
		early_stopper: EarlyStoppingTracker,
		**kwargs,
	) -> tuple[
		T,                                        # best_genome
		list[tuple[int, float]],                  # history: [(iteration, best_fitness)]
		list[T],                                  # final_population (sorted by fitness)
		list[tuple[float, float]],                # population_metrics: [(ce, acc)]
		int,                                      # iterations_run
		bool,                                     # early_stopped
		Optional[StopReason],                     # stop_reason
		Optional[float],                          # final_accuracy
		Optional[float],                          # final_threshold
	]:
		"""
		Run the strategy-specific optimization loop.

		Args:
			population: Initial population as (genome, ce, accuracy) triples.
				May contain None values for ce/accuracy if not yet evaluated.
			fitness_calculator: FitnessCalculator instance for ranking.
			early_stopper: EarlyStoppingTracker for convergence detection.
			**kwargs: Additional strategy-specific arguments passed through from optimize().

		Returns:
			Tuple of (best_genome, history, final_population, population_metrics,
			          iterations_run, early_stopped, stop_reason, final_accuracy,
			          final_threshold)
		"""
		...

	# =========================================================================
	# Optional hooks — subclasses MAY override
	# =========================================================================

	def genome_to_config(self, genome: T) -> Optional["GenomeConfig"]:
		"""Convert a genome to a GenomeConfig for tracking. Default: disabled."""
		return None

	def _pre_optimize(self, **kwargs) -> dict:
		"""
		Setup before the optimization loop.

		Override for Rust state initialization, evaluator wrapping, etc.
		Returns a dict of extra kwargs to pass to _run_optimization_loop().
		"""
		return {}

	def _post_optimize(self, result: OptimizerResult[T]) -> OptimizerResult[T]:
		"""Post-processing after optimization (validation summary, cleanup)."""
		return result

	# =========================================================================
	# Shared infrastructure
	# =========================================================================

	def _setup_fitness_calculator(self) -> Any:
		"""Create and store a FitnessCalculator from config."""
		calculator = self._config.create_fitness_calculator()
		self._fitness_calculator = calculator
		self._log.info(f"[{self.name}] Fitness calculator: {calculator.name}")
		return calculator

	def _setup_early_stopping(self, initial_fitness: float) -> EarlyStoppingTracker:
		"""Create and configure an EarlyStoppingTracker."""
		cfg = self._config
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		tracker = EarlyStoppingTracker(early_stop_config, self._log, self.name)
		tracker.reset(initial_fitness)
		return tracker

	def _compute_threshold(self, progress: float = 0.0) -> float:
		"""Compute progressive accuracy threshold at given progress [0, 1]."""
		cfg = self._config
		start = cfg.initial_threshold if cfg.initial_threshold is not None else cfg.min_accuracy
		if not cfg.progressive_threshold:
			return start
		progress = max(0.0, min(1.0, progress))
		return start + progress * cfg.threshold_delta

	def _threshold_range(self, total_iterations: int) -> tuple[float, float]:
		"""Return (start_threshold, end_threshold) for logging."""
		start = self._compute_threshold(0.0)
		actual_progress = min(1.0, total_iterations / self._config.threshold_reference)
		end = start + actual_progress * self._config.threshold_delta
		return start, end

	def _determine_stop_reason(
		self,
		shutdown_requested: bool,
		early_stopper: EarlyStoppingTracker,
	) -> Optional[StopReason]:
		"""Determine the stop reason based on shutdown flag and early stopper state."""
		if shutdown_requested:
			return StopReason.SHUTDOWN
		elif early_stopper.patience_exhausted:
			return StopReason.CONVERGENCE
		return None

	def _build_result(
		self,
		initial_genome: T,
		best_genome: T,
		initial_fitness: float,
		final_fitness: float,
		iterations_run: int,
		history: list[tuple[int, float]],
		early_stopped: bool,
		stop_reason: Optional[StopReason],
		final_population: Optional[list[T]],
		population_metrics: Optional[list[tuple[float, float]]],
		initial_accuracy: Optional[float] = None,
		final_accuracy: Optional[float] = None,
		final_threshold: Optional[float] = None,
	) -> OptimizerResult[T]:
		"""Build an OptimizerResult with computed improvement percentage."""
		improvement_pct = (
			(initial_fitness - final_fitness) / initial_fitness * 100
			if initial_fitness > 0 else 0.0
		)
		return OptimizerResult(
			initial_genome=initial_genome,
			best_genome=best_genome,
			initial_fitness=initial_fitness,
			final_fitness=final_fitness,
			improvement_percent=improvement_pct,
			iterations_run=iterations_run,
			method_name=self.name,
			history=history,
			early_stopped=early_stopped,
			stop_reason=stop_reason,
			final_population=final_population,
			population_metrics=population_metrics,
			initial_accuracy=initial_accuracy,
			final_accuracy=final_accuracy,
			final_threshold=final_threshold,
		)

	def seed_population(
		self,
		initial_population: Optional[list[T]],
		initial_evals: Optional[list[tuple[float, float]]],
		target_size: int,
	) -> list[tuple[T, Optional[float], Optional[float]]]:
		"""
		Seed population from previous phase with cached (CE, acc) reuse.

		This is the core mechanism for phase-to-phase data flow. When a
		previous phase provides both genomes AND their evaluation metrics,
		we reuse those metrics instead of re-evaluating — saving significant
		compute time.

		Rules for size mismatch:
		- same size → use all, skip re-evaluation
		- prev > target → take top-k by fitness (greedy selection)
		- prev < target → use all (caller fills remaining with strategy-specific logic)

		Args:
			initial_population: Genomes from previous phase
			initial_evals: Cached (CE, accuracy) per genome, matching order
			target_size: Desired population size

		Returns:
			List of (genome, ce, accuracy) triples. May be shorter than
			target_size if prev < target (caller fills the rest).
			Returns empty list if no initial_population.
		"""
		if not initial_population:
			return []

		# Pair genomes with cached metrics
		has_cached = (
			initial_evals is not None
			and len(initial_evals) == len(initial_population)
		)

		if has_cached:
			self._log.info(
				f"[{self.name}] Using {len(initial_population)} cached evals "
				f"from previous phase (no re-evaluation)"
			)
			seeded = [
				(self.clone_genome(g), ce, acc)
				for g, (ce, acc) in zip(initial_population, initial_evals)
			]
		else:
			# No cached metrics — mark for evaluation
			self._log.info(
				f"[{self.name}] Seeding {len(initial_population)} genomes "
				f"(metrics not cached, will need evaluation)"
			)
			seeded = [
				(self.clone_genome(g), None, None)
				for g in initial_population
			]

		# Handle size mismatch
		if len(seeded) > target_size:
			if has_cached and self._fitness_calculator is not None:
				# Top-k by fitness ranking (greedy — best genomes survive)
				fitness_scores = self._fitness_calculator.fitness(seeded)
				ranked_indices = sorted(
					range(len(fitness_scores)), key=lambda i: fitness_scores[i]
				)
				seeded = [seeded[i] for i in ranked_indices[:target_size]]
				self._log.info(
					f"[{self.name}] Trimmed population {len(initial_population)} → "
					f"{target_size} (top-k by fitness)"
				)
			else:
				# No fitness calculator yet — just truncate
				seeded = seeded[:target_size]

		elif len(seeded) < target_size:
			self._log.info(
				f"[{self.name}] Population {len(seeded)}/{target_size} — "
				f"strategy will fill remaining {target_size - len(seeded)}"
			)

		return seeded

	# =========================================================================
	# Template Method: optimize()
	# =========================================================================

	def optimize(
		self,
		evaluate_fn: Optional[Callable] = None,
		initial_genome: Optional[T] = None,
		initial_population: Optional[list[T]] = None,
		initial_fitness: Optional[float] = None,
		initial_evals: Optional[list[tuple[float, float]]] = None,
		batch_evaluate_fn: Optional[Callable] = None,
		overfitting_callback: Optional[Callable] = None,
		# TS-specific (kept for backward compat)
		initial_neighbors: Optional[list[T]] = None,
		**kwargs,
	) -> OptimizerResult[T]:
		"""
		Template Method: orchestrates the full optimization lifecycle.

		1. Pre-optimize hook (Rust setup, evaluator wrapping)
		2. Seed population with cached evals
		3. Setup fitness calculator + early stopping
		4. Delegate to strategy-specific loop
		5. Build result
		6. Post-optimize hook (validation summary)

		All strategies accept the same unified signature. Unused params
		are ignored by each strategy's _run_optimization_loop().
		"""
		self._ensure_rng()

		# Store for use by subclass hooks
		self._batch_evaluate_fn = batch_evaluate_fn
		self._evaluate_fn = evaluate_fn

		# 1. Pre-optimize hook (may modify batch_evaluate_fn, initial_population, etc.)
		extra = self._pre_optimize(
			evaluate_fn=evaluate_fn,
			initial_genome=initial_genome,
			initial_population=initial_population,
			initial_fitness=initial_fitness,
			initial_evals=initial_evals,
			batch_evaluate_fn=batch_evaluate_fn,
			overfitting_callback=overfitting_callback,
			initial_neighbors=initial_neighbors,
			**kwargs,
		)
		# Pre-optimize can override these
		if 'initial_population' in extra:
			initial_population = extra.pop('initial_population')
		if 'batch_evaluate_fn' in extra:
			batch_evaluate_fn = extra.pop('batch_evaluate_fn')
			self._batch_evaluate_fn = batch_evaluate_fn
		if 'initial_genome' in extra:
			initial_genome = extra.pop('initial_genome')

		# 2. Setup fitness calculator
		fitness_calculator = self._setup_fitness_calculator()

		# 3. Seed population with cached evals
		target_size = self._get_target_size()
		seeded = self.seed_population(initial_population, initial_evals, target_size)

		# 4. Delegate to strategy-specific loop
		(
			best_genome, history, final_pop, pop_metrics,
			iterations_run, early_stopped, stop_reason,
			final_accuracy, final_threshold,
		) = self._run_optimization_loop(
			population=seeded,
			fitness_calculator=fitness_calculator,
			early_stopper=None,  # Each loop creates its own (needs initial_fitness first)
			evaluate_fn=evaluate_fn,
			initial_genome=initial_genome,
			initial_fitness=initial_fitness,
			batch_evaluate_fn=batch_evaluate_fn,
			overfitting_callback=overfitting_callback,
			initial_neighbors=initial_neighbors,
			initial_evals=initial_evals,
			**extra,
		)

		# 5. Build result
		result_initial_genome = initial_genome if initial_genome else (
			seeded[0][0] if seeded else best_genome
		)
		initial_fitness_val = initial_fitness if initial_fitness is not None else (
			seeded[0][1] if seeded and seeded[0][1] is not None else history[0][1] if history else 0.0
		)
		result = self._build_result(
			initial_genome=result_initial_genome,
			best_genome=best_genome,
			initial_fitness=initial_fitness_val,
			final_fitness=history[-1][1] if history else initial_fitness_val,
			iterations_run=iterations_run,
			history=history,
			early_stopped=early_stopped,
			stop_reason=stop_reason,
			final_population=final_pop,
			population_metrics=pop_metrics,
			final_accuracy=final_accuracy,
			final_threshold=final_threshold,
		)

		# 6. Post-optimize hook (validation summary)
		return self._post_optimize(result)

	def _get_target_size(self) -> int:
		"""Get the target population size from config. Override if needed."""
		cfg = self._config
		# GAConfig has population_size, TSConfig has total_neighbors_size
		if hasattr(cfg, 'population_size'):
			return cfg.population_size
		elif hasattr(cfg, 'total_neighbors_size'):
			return cfg.total_neighbors_size or getattr(cfg, 'neighbors_per_iter', 20)
		return 50  # Default
