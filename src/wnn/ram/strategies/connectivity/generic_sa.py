"""
GenericSAStrategy — genome-agnostic simulated annealing (Metropolis acceptance,
geometric cooling) on OptimizationTemplate.

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



class GenericSAStrategy(OptimizationTemplate[T]):
	"""
	Generic Simulated Annealing strategy (Metropolis acceptance, geometric cooling).

	Port of the Garcia (2003) connectivity SA onto the OptimizationTemplate
	framework — same acceptance rule (accept worse with P = exp(-delta/T))
	and temperature schedule (T(i+1) = cooling_rate * T(i)), modernized:

	- Runs `chains` INDEPENDENT annealing chains in lockstep, so each
	  iteration's proposals are evaluated in a single batch_evaluate_fn
	  call (Rust/Metal batch evaluation) instead of one-at-a-time.
	- Chains seed from the previous phase's population (full population
	  carry); the chain states are the final population for the next phase.
	- Inherits early stopping, progressive threshold, fitness calculators,
	  checkpoint/tracker reporting from the shared framework.

	Subclasses must implement:
	- clone_genome(genome)
	- mutate_genome(genome, mutation_rate) -> (neighbor, move_info)
	  move_info is IGNORED by SA — the signature matches GenericTSStrategy
	  so architecture subclasses can share mutation implementations.

	Energy: the Metropolis criterion needs an ABSOLUTE per-genome scalar
	(rank-based fitness is population-relative, so it cannot serve as an
	energy). Default energy is Metrics.ce; override genome_energy() to
	change. The fitness_calculator is still used to rank the final
	population and select reported bests, keeping cross-phase semantics
	identical to GA/TS.
	"""

	def __init__(
		self,
		config: Optional[SAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		log_level: int = logging.DEBUG,
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		super().__init__(config or SAConfig(), seed=seed, logger=logger, log_level=log_level, shutdown_check=shutdown_check)

	@property
	def config(self) -> SAConfig:
		return self._config

	@property
	def name(self) -> str:
		return "GenericSA"

	# =========================================================================
	# Abstract genome operations - subclasses must implement
	# =========================================================================

	@abstractmethod
	def clone_genome(self, genome: T) -> T:
		"""Create a deep copy of the genome."""
		...

	@abstractmethod
	def mutate_genome(self, genome: T, mutation_rate: float) -> tuple[T, Any]:
		"""Create a neighbor. Returns (new_genome, move_info); move_info unused."""
		...

	# =========================================================================
	# Hooks
	# =========================================================================

	def genome_energy(self, metrics: Any) -> float:
		"""Absolute scalar energy for Metropolis (lower = better). Default: CE."""
		return metrics.ce

	def _get_target_size(self) -> int:
		return self._config.chains

	# =========================================================================
	# SA loop
	# =========================================================================

	def _run_optimization_loop(
		self,
		population: list,
		fitness_calculator: Any,
		early_stopper: Optional[EarlyStoppingTracker],
		**kwargs,
	) -> tuple:
		from wnn.ram.metrics import Metrics as _M
		cfg = self._config

		initial_genome = kwargs.get('initial_genome')
		batch_evaluate_fn = self._batch_evaluate_fn
		evaluate_fn = self._evaluate_fn
		overfitting_callback = kwargs.get('overfitting_callback')

		def _to_metrics(r) -> '_M':
			if hasattr(r, 'ce'):
				return r
			return _M(ce=r[0], acc=r[1] if len(r) > 1 else 0.0)

		def _eval_batch(genomes: list) -> list:
			if batch_evaluate_fn is not None:
				return [_to_metrics(r) for r in batch_evaluate_fn(genomes)]
			if evaluate_fn is not None:
				return [_M(ce=float(evaluate_fn(g)), acc=0.0) for g in genomes]
			raise ValueError(f"[{self.name}] requires batch_evaluate_fn or evaluate_fn")

		# === Seed chains from population (full population carry) ===
		chain_genomes: list = []
		chain_metrics: list = []
		for item in population:
			g = item[0]
			m = item[1] if len(item) > 1 and hasattr(item[1], 'ce') else (
				_M(ce=item[1], acc=item[2] if len(item) > 2 and item[2] is not None else 0.0)
				if len(item) > 1 and item[1] is not None else None
			)
			chain_genomes.append(self.clone_genome(g))
			chain_metrics.append(m)
		# Fill up to `chains` from initial_genome (or first chain)
		fill_source = initial_genome if initial_genome is not None else (chain_genomes[0] if chain_genomes else None)
		if fill_source is None:
			raise ValueError(f"[{self.name}] needs initial_genome or initial_population")
		while len(chain_genomes) < cfg.chains:
			chain_genomes.append(self.clone_genome(fill_source))
			chain_metrics.append(None)
		# Trim if oversupplied (seed_population already top-k'd, keep order)
		chain_genomes = chain_genomes[:cfg.chains]
		chain_metrics = chain_metrics[:cfg.chains]

		# Evaluate chains lacking metrics (phase transition / fills)
		missing = [i for i, m in enumerate(chain_metrics) if m is None]
		if missing:
			self._log.info(f"[{self.name}] Evaluating {len(missing)} unseeded chains")
			results = _eval_batch([chain_genomes[i] for i in missing])
			for i, m in zip(missing, results):
				chain_metrics[i] = m

		chain_energy = [self.genome_energy(m) for m in chain_metrics]

		# Global best (by energy, like the 2003 SA tracked best error)
		best_idx = min(range(len(chain_energy)), key=lambda i: chain_energy[i])
		best = self.clone_genome(chain_genomes[best_idx])
		best_metrics = chain_metrics[best_idx]
		best_energy = chain_energy[best_idx]
		start_energy = best_energy

		self._log.info(
			f"[{self.name}] Seed: best_ce={best_energy:.4f}, chains={len(chain_genomes)}, "
			f"T0={cfg.initial_temp}, cooling={cfg.cooling_rate}, iters={cfg.iterations}, "
			f"fitness={fitness_calculator.name}"
		)

		history = [(0, best_energy)]
		early_stopper = self._setup_early_stopping(best_energy)
		improved_iterations = 0
		accepted_total = 0
		prev_best = best_energy
		temperature = cfg.initial_temp
		shutdown_requested = False
		iteration = 0
		loop_start_time = time.time()

		for iteration in range(cfg.iterations):
			iter_start_time = time.time()
			current_threshold = self._compute_threshold(iteration / cfg.threshold_reference)

			try:
				self._on_iteration_start(
					iteration,
					best_genome=best,
					best_fitness=best_energy,
					best_accuracy=best_metrics.acc if best_metrics is not None else None,
					threshold=current_threshold,
					total_generations=cfg.iterations,
				)
			except StopIteration:
				shutdown_requested = True
				break

			# === Propose one neighbor per chain, evaluate as ONE batch ===
			if self._tracker and self._tracker_experiment_id:
				self._tracker.update_experiment_progress(
					self._tracker_experiment_id,
					status_message=(
						f"Iter {iteration + 1}/{cfg.iterations}: evaluating {len(chain_genomes)} "
						f"chain proposals (T={temperature:.4f}, best_ce={best_energy:.4f})"
					),
				)
			proposals = [self.mutate_genome(g, cfg.mutation_rate)[0] for g in chain_genomes]
			proposal_metrics = _eval_batch(proposals)

			# === Metropolis acceptance per chain ===
			accepted = 0
			for ci in range(len(chain_genomes)):
				p_energy = self.genome_energy(proposal_metrics[ci])
				delta = p_energy - chain_energy[ci]
				if delta < 0:
					accept = True
				elif temperature > 0:
					accept = self._rng.random() < math.exp(-delta / temperature)
				else:
					accept = False
				if accept:
					chain_genomes[ci] = proposals[ci]
					chain_metrics[ci] = proposal_metrics[ci]
					chain_energy[ci] = p_energy
					accepted += 1
					if p_energy < best_energy:
						best = self.clone_genome(proposals[ci])
						best_metrics = proposal_metrics[ci]
						best_energy = p_energy
			accepted_total += accepted
			if best_energy < prev_best:
				improved_iterations += 1

			# Cool down (geometric schedule, Garcia 2003)
			temperature *= cfg.cooling_rate

			history.append((iteration + 1, best_energy))

			iter_elapsed = time.time() - iter_start_time
			total_elapsed = time.time() - loop_start_time
			iters_done = iteration + 1
			eta_secs = (cfg.iterations - iters_done) * (total_elapsed / iters_done)
			delta_str = f"{best_energy - prev_best:+.4f}" if best_energy != prev_best else "="
			iter_width = len(str(cfg.iterations))
			self._log.info(
				f"[{self.name}] Iter {iteration + 1:0{iter_width}d}/{cfg.iterations}: "
				f"best_ce={best_energy:.4f} ({delta_str}), accepted={accepted}/{len(chain_genomes)}, "
				f"T={temperature:.5f} | {iter_elapsed:.1f}s "
				f"[elapsed: {total_elapsed:.0f}s, ETA: {eta_secs:.0f}s]"
			)

			# Record iteration to tracker (if set)
			if self._tracker and self._tracker_experiment_id:
				try:
					pop_avg_ce = sum(m.ce for m in chain_metrics) / len(chain_metrics)
					accs = [m.acc for m in chain_metrics if m.acc is not None]
					pop_avg_acc = sum(accs) / len(accs) if accs else None
					baseline_ce = early_stopper._initial_fitness
					sa_bests = fitness_calculator.bests(chain_genomes, chain_metrics)
					iteration_id = self._tracker.record_iteration(
						experiment_id=self._tracker_experiment_id,
						iteration_num=iteration + 1,
						best_ce=sa_bests.best_ce.metrics.ce,
						best_accuracy=sa_bests.best_acc.metrics.acc,
						avg_ce=pop_avg_ce,
						avg_accuracy=pop_avg_acc,
						elite_count=len(chain_genomes),
						offspring_count=len(proposals),
						offspring_viable=accepted,
						fitness_threshold=current_threshold,
						elapsed_secs=iter_elapsed,
						baseline_ce=baseline_ce,
						delta_baseline=(best_energy - baseline_ce) if baseline_ce is not None else None,
						delta_previous=best_energy - prev_best,
						patience_counter=getattr(early_stopper, '_patience_counter', 0),
						patience_max=cfg.patience,
						candidates_total=len(chain_genomes) + len(proposals),
						best_f1=max((m.f1 for m in chain_metrics if m.f1 is not None), default=None),
						best_fpr=min((m.fpr for m in chain_metrics if m.fpr is not None), default=None),
					)
					# Record genome evaluations (chains TOP_K, proposals NEIGHBOR)
					if iteration_id and HAS_TRACKER and GenomeRole is not None:
						evaluations = []
						all_metrics = chain_metrics + proposal_metrics
						all_scores = fitness_calculator.fitness(all_metrics) if fitness_calculator else [None] * len(all_metrics)
						for pos, (g, m) in enumerate(zip(chain_genomes, chain_metrics)):
							config = self.genome_to_config(g)
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(self._tracker_experiment_id, config)
								evaluations.append({
									"iteration_id": iteration_id, "genome_id": genome_id,
									"position": pos, "role": GenomeRole.TOP_K,
									"ce": m.ce, "accuracy": m.acc if m.acc is not None else 0.0,
									"elite_rank": pos, "fitness_score": all_scores[pos],
									"f1_macro": m.f1, "fpr": m.fpr,
								})
						for pos, (g, m) in enumerate(zip(proposals, proposal_metrics)):
							config = self.genome_to_config(g)
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(self._tracker_experiment_id, config)
								evaluations.append({
									"iteration_id": iteration_id, "genome_id": genome_id,
									"position": len(chain_genomes) + pos, "role": GenomeRole.NEIGHBOR,
									"ce": m.ce, "accuracy": m.acc if m.acc is not None else 0.0,
									"fitness_score": all_scores[len(chain_genomes) + pos],
									"f1_macro": m.f1, "fpr": m.fpr,
									"eval_time_ms": m.eval_time_ms,
								})
						if evaluations:
							self._tracker.record_genome_evaluations_batch(evaluations)
				except Exception as e:
					self._log.warning(f"Tracker error: {e}")

			# Early stopping check
			if early_stopper.check(iteration, best_energy):
				break

			# Overfitting callback
			if overfitting_callback is not None and (iteration + 1) % cfg.check_interval == 0:
				control = overfitting_callback(best, best_energy)
				if hasattr(control, 'early_stop') and control.early_stop:
					self._log.warning(f"[{self.name}] Overfitting early stop at iter {iteration + 1}")
					break

			prev_best = best_energy

		# === Final population: chain states ranked by fitness calculator ===
		ranked = fitness_calculator.rank(chain_genomes, chain_metrics)
		final_population = [self.clone_genome(g) for g, _ in ranked]
		ranked_metrics = [m for _, m in ranked]

		final_threshold = self._compute_threshold(iteration / cfg.threshold_reference) if cfg.iterations > 0 else self._compute_threshold(0.0)

		total_iters = iteration + 1
		total_wall = time.time() - loop_start_time
		self._log.info(f"[{self.name}] Analysis Summary:")
		self._log.info(f"  energy improvement: {start_energy:.4f} → {best_energy:.4f} ({(1 - best_energy/start_energy)*100:+.2f}%)" if start_energy else "")
		self._log.info(f"  Improved iterations: {improved_iterations}/{total_iters}, acceptance rate: {accepted_total / (total_iters * len(chain_genomes)):.1%}")
		self._log.info(f"  Final temperature: {temperature:.6f}, wall time: {total_wall:.0f}s")

		stop_reason = self._determine_stop_reason(shutdown_requested, early_stopper)
		sa_bests = fitness_calculator.bests(chain_genomes, chain_metrics)

		return (
			best, history, final_population, ranked_metrics,
			total_iters,
			early_stopper.patience_exhausted or shutdown_requested,
			stop_reason, sa_bests.best_acc.metrics.acc, final_threshold,
		)
