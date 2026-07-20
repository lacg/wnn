"""
GenericTSStrategy — genome-agnostic tabu search on OptimizationTemplate.

Split out of generic_strategies.py (D3, 11/06/2026); that module re-exports
everything, so existing imports keep working.
"""

import logging
import math
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

# Framework primitives moved to the framework/ package (D6a, 11/06/2026).
# Re-exported here so existing imports keep working.
from wnn.ram.strategies.connectivity.framework import (
	OptimizationLogger, TRACE, OverfitDetector,
	OptimizerResult, StopReason,
	EarlyStoppingConfig, EarlyStoppingTracker,
	AdaptiveLevel, AdaptiveScalerConfig, AdaptiveScaler,
	ProgressiveThresholdConfig, ProgressiveThreshold,
	OptimizationConfig, GAConfig, TSConfig, SAConfig,
)

# Cycle broken (D6a): the template now imports framework primitives from the
# framework/ package, so this import is no longer circular.
from wnn.ram.strategies.connectivity.optimization_template import OptimizationTemplate



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
		# Population-seeded callers (every phased_ga stage: the grid top-K / carried
		# pool is passed as initial_population, with NO initial_genome) previously
		# crashed here on clone_genome(None). TS needs ONE incumbent to start the
		# local search from — take the best seeded genome, which is exactly the
		# population's rank-0 (seed_population returns it ranked). 19/07/2026.
		if initial_genome is None and population:
			first = population[0]
			initial_genome = first[0] if isinstance(first, tuple) else first
			if isinstance(first, tuple) and len(first) > 1 and first[1] is not None:
				ce = getattr(first[1], "ce", None)
				if ce is not None:          # seeded-but-unevaluated genomes carry ce=None
					initial_fitness = ce
			self._log.info(
				f"[{self.name}] No initial_genome supplied — starting local search from the "
				f"best of {len(population)} seeded genomes (CE={initial_fitness if initial_fitness is not None else 'pending'})")
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
						best_fit_str = f"{best_fitness:.4f}" if best_fitness < 999 else "N/A"
						self._tracker.update_experiment_progress(
							self._tracker_experiment_id,
							status_message=f"Iter {iteration + 1}/{cfg.iterations}: batch evaluating {total_offspring} offspring from {n_sources} sources (best fitness={best_fit_str})",
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
							best_fit_str = f"{best_fitness:.4f}" if best_fitness < 999 else "N/A"
							self._tracker.update_experiment_progress(
								self._tracker_experiment_id,
								status_message=f"Iter {iteration + 1}/{cfg.iterations}: source {si + 1}/{n_sources}, {len(offspring)}/{total_offspring} offspring (best fitness={best_fit_str})",
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
					best_fit_str = f"{best_fitness:.4f}" if best_fitness < 999 else "N/A"
					self._tracker.update_experiment_progress(
						self._tracker_experiment_id,
						status_message=f"Iter {iteration + 1}/{cfg.iterations}: evaluating {cfg.neighbors_per_iter} neighbors (best fitness={best_fit_str})",
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
					baseline_ce = early_stopper._initial_fitness
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
						mean_attitude_error_deg=getattr(iter_bests.best_ce.metrics, "mean_attitude_error_deg", None),
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
								m = item[1]  # Metrics object
								evaluations.append({
									"iteration_id": iteration_id,
									"genome_id": genome_id,
									"position": pos,
									"role": GenomeRole.TOP_K,
									"ce": m.ce,
									"accuracy": m.acc if m.acc is not None else 0.0,
									"elite_rank": pos,
									"fitness_score": all_scores[pos],
									"f1_macro": m.f1,
									"fpr": m.fpr,
								})

						# Record offspring as NEIGHBOR
						for pos, item in enumerate(offspring):
							config = self.genome_to_config(item[0])
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								m = item[1]  # Metrics object
								evaluations.append({
									"iteration_id": iteration_id,
									"genome_id": genome_id,
									"position": len(pop) + pos,
									"role": GenomeRole.NEIGHBOR,
									"ce": m.ce,
									"accuracy": m.acc if m.acc is not None else 0.0,
									"fitness_score": all_scores[len(pop) + pos],
									"f1_macro": m.f1,
									"fpr": m.fpr,
									"eval_time_ms": m.eval_time_ms,
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
		ts_final_genomes = [t[0] for t in pop]
		ts_final_metrics = [t[1] for t in pop]
		ts_bests = fitness_calculator.bests(ts_final_genomes, ts_final_metrics)

		return (
			best, history, final_population, population_metrics,
			iteration + 1,
			early_stopper.patience_exhausted or shutdown_requested,
			stop_reason, ts_bests.best_acc.metrics.acc, final_threshold,
		)
