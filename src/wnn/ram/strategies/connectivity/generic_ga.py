"""
GenericGAStrategy — genome-agnostic genetic algorithm on OptimizationTemplate.

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
			# Random immigrants (E1, diversity preservation): a fresh random genome
			# enters the evaluated offspring pool with prob immigrant_fraction, so it
			# competes on REAL fitness in the mu+lambda truncation (an unevaluated
			# insert would never win a tournament and the lineage pressure would
			# remain). Counters premature convergence / lineage fixation.
			if cfg.immigrant_fraction > 0.0 and self._rng.random() < cfg.immigrant_fraction:
				return self.create_random_genome()
			p1 = self._tournament_select(population)
			p2 = self._tournament_select(population)
			if self._rng.random() < cfg.crossover_rate:
				child = self.crossover_genomes(p1, p2)
			else:
				child = self.clone_genome(p1)
			return self.mutate_genome(child, cfg.mutation_rate)

		# Random-search baseline (Review C): zero selection pressure — every
		# slot is a fresh random genome; evaluation, viability filtering and
		# μ+λ best-of-pool tracking stay identical to the GA.
		generator_fn = self.create_random_genome if getattr(cfg, 'random_search', False) \
			else offspring_generator

		return self._build_viable_population(
			target_size=n_needed,
			generator_fn=generator_fn,
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

	# ---- shared cooperative-cancel + adaptive crash-save --------------------
	# The ONE implementation of the per-generation checkpoint/shutdown logic both
	# strands used to duplicate (IDS inline in ArchitectureGAStrategy; the
	# controller via a monkey-patch). A subclass's _on_generation_start calls
	# _checkpoint_and_maybe_stop(...) and supplies _build_checkpoint(...).

	def _build_checkpoint(self, generation: int, genomes: list, ctx: dict,
	                      complete: bool):
		"""Build the PhaseCheckpoint to persist this generation. Subclasses that
		checkpoint MUST override; the default returns None (no save). `genomes` is
		the live population as BARE genomes — already unpacked from the
		(genome, metrics) tuples the GA loop carries (the shape the codecs
		serialize; the controller's old monkey-patch stored raw tuples → a latent
		serialization bug this shared path fixes)."""
		return None

	def _checkpoint_and_maybe_stop(self, generation: int, ctx: dict) -> None:
		"""Adaptive crash-save the live population via the shared
		PhasedCheckpointManager, then — if the injected shutdown_check fires —
		save once more and raise StopIteration to unwind the GA cleanly. Reads
		`self._checkpoint_mgr` / `self._shutdown_check` (both set by subclasses;
		absent → no-op). `save()` joins any in-flight async write first."""
		mgr = getattr(self, "_checkpoint_mgr", None)
		shutdown = getattr(self, "_shutdown_check", None)
		if mgr is None and shutdown is None:
			return
		genomes = [t[0] for t in ctx.get("population", [])]
		if generation > 0 and mgr is not None:
			ckpt = self._build_checkpoint(generation, genomes, ctx, complete=False)
			if ckpt is not None:
				mgr.maybe_save(generation, ckpt)
		if shutdown and shutdown():
			if mgr is not None:
				ckpt = self._build_checkpoint(generation, genomes, ctx, complete=False)
				if ckpt is not None:
					mgr.save(ckpt)
			self._log_shutdown(generation)
			raise StopIteration("Shutdown requested")

	def _log_shutdown(self, generation: int) -> None:
		"""Log the cooperative-stop line, tolerating either a logger or a plain
		callable/`print` in `self._log`."""
		msg = f"[{self.name}] Shutdown requested at generation {generation}, stopping..."
		log = getattr(self, "_log", None)
		if log is not None and hasattr(log, "info"):
			log.info(msg)
		else:
			print(msg, flush=True)

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
		best_err_deg = getattr(metrics_list[best_idx], "mean_attitude_error_deg", None)  # controller-only; always bound
		# Fitness-best genome's F1/FPR — feed the IDS magnitude-aware patience
		# (None for LM/controller; the patience branch then falls back).
		best_f1_val = metrics_list[best_idx].f1
		best_fpr_val = metrics_list[best_idx].fpr
		# Running global best F1/FPR (for dashboard tracking)
		init_f1s = [m.f1 for m in metrics_list if m.f1 is not None]
		init_fprs = [m.fpr for m in metrics_list if m.fpr is not None]
		best_f1_global = max(init_f1s) if init_f1s else None
		best_fpr_global = min(init_fprs) if init_fprs else None

		# Resume support: ArchitectureGAStrategy.optimize() sets _resume_start_gen
		# (next generation to run) + _resume_patience when loading a checkpoint, so
		# the loop CONTINUES instead of restarting at gen 0 with patience reset.
		resume_start_gen = self._resume_start_gen  # formal field on OptimizationTemplate

		history = [(resume_start_gen, best_fitness)]

		# Initialize early stopping tracker (uses base infrastructure)
		early_stopper = self._setup_early_stopping(best_fitness)
		if resume_start_gen > 0:
			# Carry the checkpointed patience; baseline against the restored
			# population's best so further improvement is measured correctly.
			early_stopper.restore(self._resume_patience)
			early_stopper._best_fitness = best_fitness
			self._log.info(
				f"[{self.name}] Resume: continuing at generation {resume_start_gen} "
				f"with patience {early_stopper._patience_counter}/{cfg.patience}")

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
		generation = resume_start_gen
		loop_start_time = time.time()
		cumulative_offspring_secs = 0.0
		for generation in range(resume_start_gen, cfg.generations):
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

			# Unified elitism: keep the top elitism_pct of the population (e.g. 0.2 = 20%).
			# (Was `* elitism_pct * 2` with elitism_pct=0.1 — the ×2 made the config value
			# mean double itself, which was confusing; now elitism_pct IS the fraction.)
			n_elites = max(1, int(cfg.population_size * cfg.elitism_pct))

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
				best_fit_str = f"{best_fitness:.4f}" if best_fitness < 999 else "N/A"
				self._tracker.update_experiment_progress(
					self._tracker_experiment_id,
					status_message=f"Gen {generation + 1}/{cfg.generations}: evaluating {needed_offspring} offspring (best fitness={best_fit_str})",
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
				# Controller flows carry closed-loop mean attitude error (degrees);
				# None for IDS/LM (the log suffix below is then omitted).
				best_err_deg = getattr(population[gen_best_idx][1], "mean_attitude_error_deg", None)
				best_f1_val = population[gen_best_idx][1].f1
				best_fpr_val = population[gen_best_idx][1].fpr

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
			# Controller repurposes acc as the stability rate — label it 'stable' when
			# the controller-only err is present, else 'acc' for IDS/LM.
			_acc_label = "stable" if best_err_deg is not None else "acc"
			acc_str = f", {_acc_label}={best_accuracy_val:.2%}" if best_accuracy_val is not None else ""
			# Controller-only: mean attitude error in degrees.
			err_str = f", err={best_err_deg:.2f}°" if best_err_deg is not None else ""
			# Controller-only: population shape + cell-count spread — diagnoses the
			# variable-shape GPU explosion AND the memory bloat (cells replicate on
			# bit-grow). Guarded so non-controller genomes never trip it.
			shape_str = ""
			try:
				gpop = [t[0] for t in population]
				if gpop and hasattr(gpop[0], "state_neurons") and hasattr(gpop[0], "state_bits_per_neuron"):
					sn = [g.state_neurons for g in gpop]
					sb = [g.state_bits_per_neuron for g in gpop]
					nshapes = len({(g.state_neurons, g.output_neurons,
					                g.state_bits_per_neuron, g.output_bits_per_neuron) for g in gpop})
					cells = [len(g.cells.state_universe) + len(g.cells.output_universe)
					         for g in gpop if getattr(g, "cells", None) is not None]
					cstr = f" cells[{min(cells)}-{max(cells)}]" if cells else ""
					shape_str = f" | shapes={nshapes} n[{min(sn)}-{max(sn)}] b[{min(sb)}-{max(sb)}]{cstr}"
			except Exception:
				shape_str = ""
			# Patience countdown. The early-stop counter ticks once per `check_interval`
			# gens of no-improvement and fires at `patience`, so a stage runs up to
			# patience*check_interval idle gens. Show GENS-TO-STOP (intuitive when
			# scanning) + the raw counter and cadence, not a bare count-up that misreads
			# as "5/5 = stop now".
			_pc = getattr(early_stopper, "_patience_counter", 0)
			_pmax = getattr(cfg, "patience", 0)
			_ci = max(1, getattr(cfg, "check_interval", 1))
			_gens_left = max(0, (_pmax - _pc)) * _ci
			patience_str = (
				f" | early-stop in ~{_gens_left:.0f}g "
				f"(patience {_pc:g}/{_pmax}, check every {_ci}g)"
			)
			self._log.info(
				f"[{self.name}] Gen {generation + 1:0{gen_width}d}/{cfg.generations}: "
				f"best={best_fitness:.4f} ({delta_str}), avg={gen_avg_ce:.4f}{acc_str}{err_str} "
				f"[elites survived: {surviving_elites}/{total_elites}]{shape_str} "
				f"| {gen_elapsed:.1f}s (offspring: {offspring_secs:.1f}s, {rate:.1f} gen/s) "
				f"[elapsed: {_fmt_duration(total_elapsed)}, ETA: {_fmt_duration(eta_secs)}]"
				f"{patience_str}"
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
						mean_attitude_error_deg=getattr(iter_bests.best_ce.metrics, "mean_attitude_error_deg", None),
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
									# Per-genome wall-clock from IDS hybrid evaluator
									# (None on paths that don't measure).
									"eval_time_ms": m.eval_time_ms,
								})
						if evaluations:
							self._tracker.record_genome_evaluations_batch(evaluations)
				except Exception as e:
					self._log.warning(f"Tracker error: {e}")
					import traceback
					traceback.print_exc()

			# Early stopping check (checks at configured intervals).
			# Magnitude-aware path: when enabled, watch the domain's PHYSICAL
			# metrics instead of the magnitude-blind rank-WHM — err°/stable%
			# for controllers, F1/FPR for IDS — both delegate to the shared
			# check_magnitude_metrics core. Falls back to the WHM check for
			# LM (no physical metrics) or when the flag is off.
			_mag_on = getattr(early_stopper._config, "magnitude_aware", False)
			if _mag_on and best_err_deg is not None:
				_stop = early_stopper.check_magnitude(generation, best_err_deg, best_accuracy_val)
			elif _mag_on and best_f1_val is not None and best_fpr_val is not None:
				_stop = early_stopper.check_magnitude_ids(generation, best_f1_val, best_fpr_val)
			else:
				_stop = early_stopper.check(generation, best_fitness)
			if _stop:
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

		# Batch evaluate — returns list[Metrics].
		#
		# Dual-path dispatch (15/05/2026): group by (neurons_per_cluster,
		# bits_per_neuron) shape and pick GPU-batched vs CPU-per-genome per
		# group based on size.
		#
		# Why: GPU batched_train_offspring has ~1-2s fixed kernel-launch
		# overhead. Below a threshold (~8 genomes), per-genome dispatch
		# (which lets B11 affinity route single-genome calls to CPU) is
		# faster — same lesson as Phase 5.5 (commit ca9ce609). Above the
		# threshold, batching amortizes the kernel launch and GPU wins.
		#
		# Threshold tunable via WNN_SMALL_GROUP_THRESHOLD (default 8).
		# Each per-group log line includes "via {GPU-batched,CPU-per-genome}"
		# so the dashboard / human reader can see which path fired.
		if batch_fn is not None:
			if len(to_eval) >= 10:
				# Shape-grouped streaming path
				from collections import defaultdict as _defaultdict
				import time as _time
				import os as _os
				_log = getattr(self, "_log", None)
				_name = getattr(self, "name", "GA")

				# Auto-scale small-group threshold against dataset size.
				#
				# Rationale: GPU batched_train_offspring has ~constant per-call
				# overhead (kernel launch + setup). Per-call work scales with
				# n_examples × n_neurons. As n_examples grows, the fixed
				# overhead becomes a smaller fraction of wall time AND the
				# Metal queue serializes parallel single-genome calls anyway
				# (4 ThreadPool workers all share one Metal queue) — so
				# batching wins at smaller and smaller group sizes.
				#
				# Reference point: 1.1M neto-subsample → threshold 8 (empirical).
				# Scale inversely; clamp to [2, 16] to avoid pathological values.
				#
				# Env override: WNN_SMALL_GROUP_THRESHOLD wins over auto-scale.
				_evaluator = getattr(self, "_batch_evaluator", None)
				_n_train = getattr(_evaluator, "_kfold_n_train", None) if _evaluator else None
				if _n_train and _n_train > 0:
					_auto_threshold = max(2, min(16, int(8 * (1_100_000 / _n_train))))
				else:
					_auto_threshold = 8
				small_threshold = int(_os.environ.get("WNN_SMALL_GROUP_THRESHOLD", str(_auto_threshold)))

				shape_to_locals: dict = _defaultdict(list)
				for i, g in enumerate(to_eval):
					_n = tuple(getattr(g, "neurons_per_cluster", []) or [])
					_b = tuple(getattr(g, "bits_per_neuron", []) or [])
					shape_to_locals[(_n, _b)].append((i, g))

				if _log is not None:
					_log.info(
						f"[{_name}] Re-eval streaming: {len(to_eval)} genomes "
						f"→ {len(shape_to_locals)} shape groups "
						f"(threshold={small_threshold})"
					)

				new_metrics_indexed: list = [None] * len(to_eval)
				completed = 0
				best_ce_so_far: Optional[float] = None
				best_acc_so_far: Optional[float] = None
				# Track tracker access — same pattern as the main GA loop above.
				_has_tracker = bool(getattr(self, "_tracker", None) and getattr(self, "_tracker_experiment_id", None))

				# Small-group parallelism: same pattern as Phase 5.5
				# (architecture_strategies.py). Default 4 workers; B11 routes
				# single-genome calls to CPU so 4 threads parallelize cleanly
				# across CPU cores. Without this, a 5-genome small group runs
				# ~5x slower than necessary.
				_ga_parallel = int(_os.environ.get("WNN_GA_SMALL_GROUP_PARALLEL", "4"))
				from concurrent.futures import ThreadPoolExecutor as _TPE
				for shape, items in shape_to_locals.items():
					t0 = _time.time()
					if len(items) >= small_threshold:
						# Large group → GPU-batched (amortizes kernel launch)
						group_genomes = [g for _, g in items]
						group_results = batch_fn(group_genomes)
						path = "GPU-batched"
					else:
						# Small group → CPU-per-genome via ThreadPool (parallelism=4).
						# B11 routes single-genome calls to CPU; ThreadPool wraps
						# the iteration so 4 genomes run concurrently across cores.
						group_genomes = [g for _, g in items]
						if _ga_parallel > 1 and len(group_genomes) > 1:
							with _TPE(max_workers=min(_ga_parallel, len(group_genomes))) as _pool:
								group_results = list(_pool.map(lambda g: batch_fn([g])[0], group_genomes))
						else:
							group_results = [batch_fn([g])[0] for g in group_genomes]
						path = f"CPU-per-genome (p={min(_ga_parallel, len(group_genomes))})"
					elapsed = _time.time() - t0
					for (idx, _), r in zip(items, group_results):
						m = r if isinstance(r, Metrics) else Metrics(
							ce=r.ce, acc=r.acc, f1=r.f1, fpr=r.fpr
						)
						new_metrics_indexed[idx] = m
						completed += 1
						if best_ce_so_far is None or m.ce < best_ce_so_far:
							best_ce_so_far = m.ce
						if best_acc_so_far is None or m.acc > best_acc_so_far:
							best_acc_so_far = m.acc
					if _log is not None:
						_n_tuple, _b_tuple = shape
						_n = _n_tuple[0] if _n_tuple else 0
						_b = _b_tuple[0] if _b_tuple else 0
						_log.info(
							f"[{_name}] Re-eval [{completed}/{len(to_eval)}] "
							f"n={_n} b={_b} group_size={len(items)} via {path} ({elapsed:.1f}s)"
						)
					# Push status to dashboard — current_iteration stays at 0
					# (pre-generation 1), but best_ce / best_accuracy / status_message
					# tick visibly so the dashboard isn't dark during seed re-eval.
					if _has_tracker:
						try:
							self._tracker.update_experiment_progress(
								self._tracker_experiment_id,
								current_iteration=0,
								best_ce=best_ce_so_far,
								best_accuracy=best_acc_so_far,
								status_message=f"Seed re-eval [{completed}/{len(to_eval)}] via {path}",
							)
						except Exception as e:
							if _log is not None:
								_log.warning(f"[{_name}] Re-eval tracker error: {e}")
				new_metrics = new_metrics_indexed
			else:
				# Small population — keep the single-call path (no streaming overhead)
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
