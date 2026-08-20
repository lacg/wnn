"""
AdaptationConfig + AdaptationStrategy — neurogenesis / synaptogenesis / axonogenesis architecture adaptation

Split out of architecture_strategies.py (D3, 11/06/2026); that module
re-exports everything, so existing imports keep working.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.strategies.connectivity.framework import OptimizerResult, StopReason, EarlyStoppingConfig, EarlyStoppingTracker
from wnn.ram.strategies.connectivity.genome_tracking import HAS_GENOME_TRACKING, TierConfig, GenomeConfig, GenomeRole
from wnn.ram.strategies.connectivity.architecture_mixin import ArchitectureStrategyMixin

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		AdaptiveClusterConfig,
	)

@dataclass
class AdaptationConfig:
	"""Configuration for stats-guided adaptation strategy.

	Unlike GA/TS, adaptation strategies deterministically modify architecture
	based on training statistics (neuron/cluster error, fill rates, entropy).
	The "search" is iterative refinement: each round, stats change after
	adaptation, leading to different pruning/growing/rewiring decisions.
	"""
	num_clusters: int
	min_bits: int = 8
	max_bits: int = 25
	min_neurons: int = 3
	max_neurons: int = 33
	total_input_bits: Optional[int] = None
	adaptation_mode: str = "neurogenesis"  # "neurogenesis", "synaptogenesis", "axonogenesis"
	iterations: int = 50
	population_size: int = 50
	patience: int = 5
	check_interval: int = 10
	min_improvement_pct: float = 0.5
	initial_threshold: Optional[float] = None
	threshold_delta: float = 0.01
	threshold_reference: int = 1000
	progressive_threshold: bool = True
	min_accuracy: float = 0.0
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	fitness_weight_f1: float = 0.0
	fitness_weight_fpr: float = 0.0
	# Rank-combine step, mirrored from OptimizationConfig: this class does NOT
	# inherit it, so the field has to exist here too or the builder that passes
	# it raises "unexpected keyword argument".
	fitness_aggregation: str = "harmonic"
	zrank_clamp: float = 3.0
	min_accuracy_floor: Optional[float] = None


class AdaptationStrategy(ArchitectureStrategyMixin):
	"""
	Stats-guided adaptation strategy (neurogenesis, synaptogenesis, axonogenesis).

	Unlike GA/TS which use random mutation/selection, this strategy deterministically
	modifies architecture based on training statistics. Each iteration:
	  1. Evaluate all genomes with adaptation enabled (train → GPU stats → adapt → retrain → eval)
	  2. Sort by fitness, keep top performers
	  3. Report iteration metrics to dashboard
	  4. Check early stopping

	The Rust evaluator (evaluate_genomes_adaptive) handles the full cycle and
	modifies genomes in-place with adapted architectures.
	"""

	def __init__(
		self,
		config: AdaptationConfig,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		cached_evaluator: Optional[Any] = None,  # BitwiseEvaluator
		phase_name: str = "Adaptation",
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		self._config = config
		self._seed = seed or 42
		self._rng = random.Random(self._seed)
		self._log = logger or (lambda x: None)
		self._cached_evaluator = cached_evaluator
		self._phase_name = phase_name
		self._shutdown_check = shutdown_check
		self._tracker = None
		self._tracker_experiment_id = None

	@property
	def name(self) -> str:
		mode = self._config.adaptation_mode.capitalize()
		return f"Adaptation({mode})"

	def set_tracker(self, tracker, experiment_id: int, _flow_experiment_id: int = None):
		"""Set V2 tracker for iteration/genome recording."""
		self._tracker = tracker
		self._tracker_experiment_id = experiment_id

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def optimize(
		self,
		evaluate_fn: Optional[Callable] = None,
		initial_population: Optional[list['ClusterGenome']] = None,
		initial_genome: Optional['ClusterGenome'] = None,
		**kwargs,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run stats-guided adaptation.

		Takes a population from the previous phase, iteratively adapts all genomes,
		sorts by fitness, and returns the best.
		"""
		import time as _time
		from wnn.ram.fitness import FitnessCalculatorFactory
		from wnn.ram.metrics import Metrics as _M

		cfg = self._config
		evaluator = self._cached_evaluator
		if evaluator is None:
			raise ValueError(f"{self.name} requires a BitwiseEvaluator (cached_evaluator)")

		# Build population from initial data
		population: list['ClusterGenome'] = []
		if initial_population:
			population = [g.clone() for g in initial_population[:cfg.population_size]]
		elif initial_genome:
			population = [initial_genome.clone()]

		if not population:
			raise ValueError(f"{self.name} requires initial_population or initial_genome")

		# Pad population if needed (clone best genome with fresh connections)
		while len(population) < cfg.population_size and len(population) > 0:
			base = population[len(population) % len(population)]
			clone = base.clone()
			if hasattr(clone, 'initialize_connections') and cfg.total_input_bits:
				clone.initialize_connections(cfg.total_input_bits)
			population.append(clone)

		from wnn.ram.metrics import FitnessWeights
		calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weights=FitnessWeights(ce=cfg.fitness_weight_ce, acc=cfg.fitness_weight_acc,
								   f1=cfg.fitness_weight_f1, fpr=cfg.fitness_weight_fpr),
			aggregation=cfg.fitness_aggregation,
			zrank_clamp=cfg.zrank_clamp,
		)

		# Configure evaluator's adaptation mode for this phase
		original_adapt_config = evaluator._adapt_config
		self._configure_evaluator_adaptation(evaluator)

		# Fix train subset for the entire phase so all iterations are comparable
		# (auto-advancing would evaluate each iteration on a different subset,
		# confusing early stopping and fitness tracking)
		phase_train_idx = kwargs.get('train_subset_idx', None)
		if phase_train_idx is None:
			phase_train_idx = evaluator.random_train_idx(self._rng)

		self._log(f"\n{'='*70}")
		self._log(f"  {self._phase_name}: {self.name}")
		self._log(f"{'='*70}")
		self._log(f"  Mode: {cfg.adaptation_mode}")
		self._log(f"  Population: {len(population)}")
		self._log(f"  Iterations: {cfg.iterations}")
		self._log(f"  Patience: {cfg.patience} (check every {cfg.check_interval})")
		self._log(f"  Fitness: {calculator.name}")
		self._log(f"  Train subset: {phase_train_idx}")
		start_threshold = self._compute_threshold(0.0)
		end_threshold = self._compute_threshold(min(1.0, cfg.iterations / cfg.threshold_reference))
		self._log(f"  Threshold: {start_threshold:.4%} → {end_threshold:.4%}")
		self._log("")

		# Early stopping
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stopper = EarlyStoppingTracker(early_stop_config, self._log)

		best_genome = population[0].clone()
		best_ce = float('inf')
		best_acc = 0.0
		best_f1_global: Optional[float] = None
		best_fpr_global: Optional[float] = None
		evals = [(float('inf'), 0.0)] * len(population)
		fitness_scores = [float('inf')] * len(population)
		history = []
		shutdown_requested = [False]
		start_time = _time.time()

		try:
			for iteration in range(cfg.iterations):
				iter_start = _time.time()

				# Progressive threshold (shared with GA/TS via mixin)
				current_threshold = self._compute_threshold(iteration / cfg.threshold_reference)

				# Check shutdown
				if self._check_and_set_shutdown(shutdown_requested):
					self._log(f"  Shutdown requested at iteration {iteration}")
					break

				# Set generation for cosine annealing
				evaluator.set_generation(iteration, total_generations=cfg.iterations)

				# Evaluate all genomes (adaptation happens in-place via Rust)
				evals = evaluator.evaluate_batch(
					population,
					train_subset_idx=phase_train_idx,
					generation=iteration,
					total_generations=cfg.iterations,
				)

				# Convert evals to Metrics for fitness ranking
				eval_metrics = [e if isinstance(e, _M) else _M(ce=e.ce, acc=e.acc, f1=e.f1, fpr=e.fpr) for e in evals]

				# Compute fitness and sort
				fitness_scores = calculator.fitness(eval_metrics)
				sorted_indices = sorted(
					range(len(population)),
					key=lambda i: fitness_scores[i],
				)

				# Reorder population by fitness
				population = [population[i] for i in sorted_indices]
				evals = [evals[i] for i in sorted_indices]
				fitness_scores = [fitness_scores[i] for i in sorted_indices]

				# Update bests independently (CE and accuracy tracked separately, like GA/TS)
				pop_bests = calculator.bests(population, eval_metrics)
				if pop_bests.best_ce.metrics.ce < best_ce:
					best_ce = pop_bests.best_ce.metrics.ce
					best_genome = pop_bests.best_fitness.genome.clone()
				if pop_bests.best_acc.metrics.acc > best_acc:
					best_acc = pop_bests.best_acc.metrics.acc

				history.append((iteration + 1, best_ce))
				iter_elapsed = _time.time() - iter_start

				# Log progress
				avg_ce = sum(e.ce for e in evals) / len(evals)
				avg_acc = sum(e.acc for e in evals) / len(evals)
				self._log(
					f"  [{iteration+1:3d}/{cfg.iterations}] "
					f"best CE={best_ce:.4f} acc={best_acc:.2%} | "
					f"avg CE={avg_ce:.4f} acc={avg_acc:.2%} | "
					f"{iter_elapsed:.1f}s"
				)

				# Record iteration in tracker (with patience/threshold for dashboard)
				prev_best = early_stopper._prev_best if early_stopper._prev_best is not None else best_ce
				delta_previous = best_ce - prev_best
				if self._tracker and self._tracker_experiment_id:
					try:
						# Update running global best F1/FPR (like best_ce/best_acc)
						ev_f1s = [e.f1 for e in evals]
						ev_fprs = [e.fpr for e in evals]
						valid_f1s = [v for v in ev_f1s if v is not None]
						valid_fprs = [v for v in ev_fprs if v is not None]
						if valid_f1s:
							iter_best_f1 = max(valid_f1s)
							if best_f1_global is None or iter_best_f1 > best_f1_global:
								best_f1_global = iter_best_f1
						if valid_fprs:
							iter_best_fpr = min(valid_fprs)
							if best_fpr_global is None or iter_best_fpr < best_fpr_global:
								best_fpr_global = iter_best_fpr
						iter_id = self._tracker.record_iteration(
							experiment_id=self._tracker_experiment_id,
							iteration_num=iteration + 1,
							best_ce=best_ce,
							best_accuracy=best_acc,
							avg_ce=avg_ce,
							avg_accuracy=avg_acc,
							elapsed_secs=iter_elapsed,
							candidates_total=len(population),
							fitness_threshold=current_threshold,
							delta_previous=delta_previous,
							patience_counter=early_stopper._patience_counter,
							patience_max=cfg.patience,
							best_f1=best_f1_global,
							best_fpr=best_fpr_global,
						)
						# Record genome evaluations
						if HAS_GENOME_TRACKING:
							for pos, (genome, ev_m, fit) in enumerate(
								zip(population, evals, fitness_scores)
							):
								ce, acc = ev_m.ce, ev_m.acc
								ev_f1 = ev_m.f1
								ev_fpr = ev_m.fpr
								genome_config = self.genome_to_config(genome)
								if genome_config:
									genome_id = self._tracker.get_or_create_genome(
										self._tracker_experiment_id, genome_config
									)
									self._tracker.record_genome_evaluation(
										iteration_id=iter_id,
										genome_id=genome_id,
										position=pos,
										role=GenomeRole.SURVIVOR if pos == 0 else GenomeRole.OFFSPRING,
										ce=ce,
										accuracy=acc,
										fitness_score=fit,
										f1_macro=ev_f1,
										fpr=ev_fpr,
									)
						self._tracker.update_experiment_progress(
							self._tracker_experiment_id,
							current_iteration=iteration + 1,
							best_ce=best_ce,
							best_accuracy=best_acc,
						)
					except Exception as e:
						self._log(f"  Warning: tracker error: {e}")

				# Early stopping check
				if early_stopper.check(iteration, best_ce):
					self._log(f"  Early stopping at iteration {iteration + 1}")
					break

				# Metal cleanup every 10 iterations
				if iteration % 10 == 9:
					self._cleanup_metal(iteration)

		finally:
			# Restore original adaptation config
			evaluator._adapt_config = original_adapt_config

		elapsed = _time.time() - start_time

		# Determine stop reason
		stop_reason = self._determine_stop_reason(shutdown_requested[0], early_stopper)
		if stop_reason is None:
			stop_reason = StopReason.MAX_ITERATIONS

		# Build population metrics for downstream phases
		population_metrics = [e if isinstance(e, _M) else _M(ce=e.ce, acc=e.acc, f1=e.f1, fpr=e.fpr) for e in evals]

		initial_ce = history[0][1] if history else best_ce
		improvement = ((initial_ce - best_ce) / initial_ce * 100) if initial_ce > 0 else 0.0

		self._log(f"\n{'─'*70}")
		self._log(f"  {self.name} Complete")
		self._log(f"  Best CE: {best_ce:.4f}  Acc: {best_acc:.2%}")
		self._log(f"  Improvement: {improvement:.2f}%")
		self._log(f"  Duration: {elapsed:.1f}s")
		self._log(f"{'─'*70}\n")

		return OptimizerResult(
			initial_genome=best_genome,
			best_genome=best_genome,
			initial_fitness=initial_ce,
			final_fitness=best_ce,
			improvement_percent=improvement,
			iterations_run=len(history),
			method_name=self.name,
			history=history,
			early_stopped=stop_reason in (StopReason.CONVERGENCE, StopReason.SHUTDOWN),
			stop_reason=stop_reason,
			final_population=population,
			population_metrics=population_metrics,
			initial_accuracy=0.0,
			final_accuracy=best_acc,
			final_threshold=self._compute_threshold(min(1.0, (len(history) - 1) / cfg.threshold_reference) if history else 0.0),
		)

	def _configure_evaluator_adaptation(self, evaluator):
		"""Configure the evaluator's adaptation settings for this strategy's mode.

		Creates or modifies the evaluator's _adapt_config to enable only
		the adaptation mode needed by this strategy.
		"""
		from wnn.ram.architecture.bitwise_evaluator import AdaptationConfig as EvalAdaptConfig

		mode = self._config.adaptation_mode
		cfg = self._config

		# Create adaptation config with only the relevant mode enabled
		adapt = EvalAdaptConfig(
			synaptogenesis_enabled=(mode == "synaptogenesis"),
			neurogenesis_enabled=(mode == "neurogenesis"),
			axonogenesis_enabled=(mode == "axonogenesis"),
			min_bits=cfg.min_bits,
			max_bits=cfg.max_bits,
			min_neurons=cfg.min_neurons,
			# Use sensible defaults for adaptation parameters
			warmup_generations=0,  # No warmup — dedicated phase, always active
			cooldown_iterations=0,
			total_generations=cfg.iterations,
			passes_per_eval=1,
			stats_sample_size=10_000,
		)
		evaluator._adapt_config = adapt
