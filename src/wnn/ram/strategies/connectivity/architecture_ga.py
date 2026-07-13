"""
ArchitectureGAStrategy — GA over ClusterGenome on the generic GA core

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

from wnn.ram.strategies.connectivity.framework import GAConfig, OptimizerResult, StopReason
from wnn.ram.strategies.connectivity.generic_ga import GenericGAStrategy
from wnn.ram.strategies.connectivity.adaptive_cluster import PhaseType
from wnn.ram.strategies.connectivity.genome_tracking import HAS_GENOME_TRACKING, TierConfig, GenomeConfig, GenomeRole
from wnn.ram.strategies.connectivity.architecture_mixin import ArchitectureStrategyMixin
from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig
from wnn.ram.strategies.connectivity.checkpoint_manager import CheckpointConfig
from wnn.ram.strategies.phased import (
	PhasedCheckpointManager, PhaseCheckpoint, SaveCadence, ClusterGenomeCodec,
)

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		AdaptiveClusterConfig,
	)


def _ids_to_checkpoint(phase_name, iteration, population, best_genome,
                       best_fitness, threshold, extra_state) -> PhaseCheckpoint:
	"""IDS GA loop state → unified PhaseCheckpoint. `population` is a list of
	(genome, ce); only the genomes are persisted (each carries its own metrics via
	serialize, and resume re-evaluates anyway). patience_counter rides in the
	dedicated `patience` field; everything else (complete flag, config) in extra."""
	extra = dict(extra_state or {})
	patience = int(extra.pop("patience_counter", 0) or 0)
	bf = best_fitness if isinstance(best_fitness, (tuple, list)) else (best_fitness, None)
	return PhaseCheckpoint(
		phase_key=str(phase_name), phase_name=str(phase_name), strategy_type="GA",
		best_genome=best_genome,
		final_population=[g for g, _ce in population] if population else None,
		iterations_run=int(iteration), patience=patience,
		final_fitness=bf[0] if len(bf) > 0 else None,
		final_accuracy=bf[1] if len(bf) > 1 else None,
		final_threshold=threshold,
		extra=extra,
	)


def _ids_resume_from_checkpoint(ckpt: PhaseCheckpoint) -> dict:
	"""Inverse: PhaseCheckpoint → the resume_state dict the GA loop expects. ce is
	a placeholder (resume discards it and re-evaluates); patience_counter is read
	back from the dedicated field; complete/config come from extra."""
	pop = [(g, (g.metrics.ce if getattr(g, "metrics", None) else 0.0))
	       for g in (ckpt.final_population or [])]
	extra = dict(ckpt.extra or {})
	extra.setdefault("patience_counter", ckpt.patience)
	return {
		"current_iteration": ckpt.iterations_run,
		"population": pop,
		"best_genome": ckpt.best_genome,
		"best_fitness": (ckpt.final_fitness, ckpt.final_accuracy),
		"current_threshold": ckpt.final_threshold,
		"config": extra.get("config", {}),
		"extra_state": extra,
	}


class ArchitectureGAStrategy(ArchitectureStrategyMixin, GenericGAStrategy['ClusterGenome']):
	"""
	Genetic Algorithm for architecture (bits, neurons per cluster) optimization.

	Inherits core GA loop from GenericGAStrategy, implements ClusterGenome operations.
	Uses ArchitectureStrategyMixin for shared functionality (Metal cleanup, shutdown, etc.)

	Features:
	- Rust/Metal batch evaluation (default when available)
	- Rust-based offspring search with threshold (when cached_evaluator provided)
	- Population seeding from previous phases
	- Checkpoint/resume support for long runs
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ga_config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional[Any] = None,
		cached_evaluator: Optional[Any] = None,  # BaseEvaluator for Rust search_offspring
		checkpoint_config: Optional[CheckpointConfig] = None,  # Checkpoint configuration
		phase_name: str = "GA Optimization",  # Phase name for checkpoints
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
	):
		super().__init__(config=ga_config, seed=seed, logger=logger)
		self._arch_config = arch_config
		self._batch_evaluator = batch_evaluator
		# Use cached_evaluator if provided, or check if batch_evaluator has search_offspring
		if cached_evaluator is not None:
			self._cached_evaluator = cached_evaluator
		elif batch_evaluator is not None and hasattr(batch_evaluator, 'search_offspring'):
			self._cached_evaluator = batch_evaluator
		else:
			self._cached_evaluator = None
		self._checkpoint_config = checkpoint_config
		self._phase_name = phase_name
		self._shutdown_check = shutdown_check
		self._phase_type = self._derive_phase_type()

	@property
	def name(self) -> str:
		return "ArchitectureGA"

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> 'ClusterGenome':
		"""Phase-aware mutation dispatching to ClusterGenome.mutate()."""
		from wnn.ram.strategies.connectivity.adaptive_cluster import AdaptiveClusterConfig
		self._ensure_rng()
		cfg = self._arch_config
		mutation_config = AdaptiveClusterConfig(
			min_bits=cfg.min_bits, max_bits=cfg.max_bits,
			min_neurons=cfg.min_neurons, max_neurons=cfg.max_neurons,
		)
		tib = cfg.total_input_bits or 64
		return genome.mutate(self._phase_type, mutation_rate, mutation_config, tib, self._rng)

	def crossover_genomes(self, parent1: 'ClusterGenome', parent2: 'ClusterGenome') -> 'ClusterGenome':
		"""Phase-aware crossover dispatching to ClusterGenome.crossover()."""
		self._ensure_rng()
		return parent1.crossover(parent2, self._phase_type, self._rng)

	def create_random_genome(self) -> 'ClusterGenome':
		"""
		Create a random genome based on optimize_* flags.

		- If optimize_bits=True: random bits per neuron in [min_bits, max_bits]
		- If optimize_bits=False: use default_bits for all neurons
		- Same logic for neurons

		Bits are generated per-neuron (flat list), not per-cluster.
		When optimizing connections only, both bits and neurons use defaults.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		self._ensure_rng()
		cfg = self._arch_config

		if cfg.token_frequencies is not None:
			return self._create_frequency_scaled_genome()

		# Initialize neurons: random if optimizing, default otherwise
		if cfg.optimize_neurons:
			neurons = [self._rng.randint(cfg.min_neurons, cfg.max_neurons) for _ in range(cfg.num_clusters)]
		else:
			neurons = [cfg.default_neurons] * cfg.num_clusters

		# Initialize per-neuron bits: random if optimizing, default otherwise
		total_neurons = sum(neurons)
		if cfg.optimize_bits:
			bits_per_neuron = [self._rng.randint(cfg.min_bits, cfg.max_bits) for _ in range(total_neurons)]
		else:
			bits_per_neuron = [cfg.default_bits] * total_neurons

		# Initialize connections if total_input_bits available
		connections = None
		if cfg.total_input_bits is not None:
			from wnn.ram.strategies.connectivity.adaptive_cluster import generate_connections
			connections = generate_connections(bits_per_neuron, cfg.total_input_bits, self._rng.randint(0, 2**63))

		return ClusterGenome(bits_per_neuron=bits_per_neuron, neurons_per_cluster=neurons, connections=connections)

	def _create_frequency_scaled_genome(self) -> 'ClusterGenome':
		"""
		Create genome with bits/neurons scaled by token frequency.

		- If optimize_bits=True: scale bits by frequency (per-neuron)
		- If optimize_bits=False: use default_bits
		- Same logic for neurons

		Bits are expanded to per-neuron (flat list) after computing per-cluster values.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		cfg = self._arch_config
		freqs = cfg.token_frequencies

		# Normalize frequencies to [0, 1]
		max_freq = max(freqs) if freqs else 1
		norm_freqs = [f / max_freq if max_freq > 0 else 0 for f in freqs]

		cluster_bits = []
		neurons = []
		for nf in norm_freqs:
			# Bits: scaled if optimizing, default otherwise
			if cfg.optimize_bits:
				b = int(cfg.min_bits + nf * (cfg.max_bits - cfg.min_bits))
			else:
				b = cfg.default_bits

			# Neurons: scaled if optimizing, default otherwise
			if cfg.optimize_neurons:
				n = int(cfg.min_neurons + nf * (cfg.max_neurons - cfg.min_neurons))
			else:
				n = cfg.default_neurons

			cluster_bits.append(max(cfg.min_bits, min(cfg.max_bits, b)))
			neurons.append(max(cfg.min_neurons, min(cfg.max_neurons, n)))

		# Expand per-cluster bits to per-neuron (flat list)
		bits_per_neuron = []
		for i in range(cfg.num_clusters):
			bits_per_neuron.extend([cluster_bits[i]] * neurons[i])

		# Initialize connections if total_input_bits available
		connections = None
		if cfg.total_input_bits is not None:
			from wnn.ram.strategies.connectivity.adaptive_cluster import generate_connections
			connections = generate_connections(bits_per_neuron, cfg.total_input_bits, self._rng.randint(0, 2**63))

		return ClusterGenome(bits_per_neuron=bits_per_neuron, neurons_per_cluster=neurons, connections=connections)

	# =========================================================================
	# Hooks: Rust-accelerated offspring generation + lifecycle
	# =========================================================================

	def _generate_offspring(self, population, n_needed, threshold, generation):
		"""Generate offspring via Rust search_offspring or Python fallback."""
		# Random-search baseline: bypass the Rust breeding path (it tournaments/
		# crosses/mutates) — the base implementation samples fresh random genomes
		# and still evaluates through the Rust batch evaluator.
		if getattr(self._config, 'random_search', False):
			return super()._generate_offspring(population, n_needed, threshold, generation)
		if self._cached_evaluator is not None:
			cfg = self._config
			arch_cfg = self._arch_config
			evaluator = self._cached_evaluator

			# Phase-aware mutation rates: each phase uses cfg.mutation_rate
			# for its own dimension, 0.0 for others
			if self._phase_type == PhaseType.NEURONS:
				bits_mutation_rate = 0.0
				neurons_mutation_rate = cfg.mutation_rate
			elif self._phase_type == PhaseType.BITS:
				bits_mutation_rate = cfg.mutation_rate
				neurons_mutation_rate = 0.0
			else:  # CONNECTIONS
				bits_mutation_rate = cfg.mutation_rate
				neurons_mutation_rate = 0.0

			# fitness_percentile controls selectivity: generate a larger pool,
			# rank by fitness, keep only the top fraction → return exactly n_needed.
			# e.g. percentile=0.75 → generate ceil(24/0.75)=32, rank, keep best 24.
			import math
			pct = cfg.fitness_percentile if cfg.fitness_percentile and 0 < cfg.fitness_percentile < 1.0 else None
			generate_count = math.ceil(n_needed / pct) if pct else n_needed

			# Convert (genome, Metrics) to (genome, ce_float) for Rust evaluator
			rust_population = [(t[0], t[1].ce) for t in population]

			# Pre-compute fitness scores so tournament selection uses the
			# same metric as elite selection (e.g. HarmonicRank), not raw CE
			fitness_scores = None
			if self._fitness_calculator is not None:
				pop_metrics_list = [t[1] for t in population]
				fitness_scores = self._fitness_calculator.fitness(pop_metrics_list)

			search_result = evaluator.search_offspring(
				population=rust_population,
				target_count=generate_count,
				max_attempts=generate_count * 5,
				accuracy_threshold=threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				bits_mutation_rate=bits_mutation_rate,
				neurons_mutation_rate=neurons_mutation_rate,
				crossover_rate=cfg.crossover_rate,
				tournament_size=cfg.tournament_size,
				train_subset_idx=self._phase_train_idx,
				eval_subset_idx=0,
				seed=self._seed_offset + generation,
				logger=self._log,
				generation=generation,
				total_generations=cfg.generations,
				return_best_n=True,
				mutable_clusters=arch_cfg.mutable_clusters,
				phase_type=int(self._phase_type),
				fitness_scores=fitness_scores,
				cluster_crossover_ratio=arch_cfg.cluster_crossover_ratio,
				pool_shuffle_ratio=arch_cfg.pool_shuffle_ratio,
				assortative_mating_ratio=arch_cfg.assortative_mating_ratio,
			)

			# Convert to (genome, Metrics) tuples, rank by fitness, return best n_needed
			offspring = [
				(g, g.metrics)
				for g in search_result.genomes
				if hasattr(g, 'metrics') and g.metrics is not None
			]

			if pct and len(offspring) > n_needed:
				scores = self._fitness_calculator.fitness([t[1] for t in offspring])
				ranked = sorted(zip(offspring, scores), key=lambda x: x[1])
				offspring = [item for item, _ in ranked[:n_needed]]

			return offspring

		# Fallback to Python generation
		return super()._generate_offspring(population, n_needed, threshold, generation)

	def _on_generation_start(self, generation, **ctx):
		"""IDS-specific per-gen work (Baldwin generation tracking + Metal cleanup),
		then the SHARED cooperative-cancel + adaptive crash-save
		(_checkpoint_and_maybe_stop, base) using _build_checkpoint below."""
		# Update evaluator generation for adaptive evaluation (Baldwin effect)
		evaluator = self._cached_evaluator or self._batch_evaluator
		if evaluator is not None and hasattr(evaluator, 'set_generation'):
			evaluator.set_generation(generation, total_generations=ctx.get('total_generations'))

		# Metal cleanup (every generation except first)
		if generation > 0 and self._cached_evaluator is not None:
			self._cleanup_metal(generation, log_interval=10)

		# Shared: adaptive crash-save (cadence-throttled) + cooperative shutdown.
		self._checkpoint_and_maybe_stop(generation, ctx)

	def _build_checkpoint(self, generation, genomes, ctx, complete):
		"""IDS GA loop state → PhaseCheckpoint. `genomes` are bare (already
		unpacked by the shared path); _ids_to_checkpoint expects (genome, ce)
		pairs but only reads the genome, so pair with a placeholder ce."""
		return _ids_to_checkpoint(
			self._phase_name, generation,
			[(g, None) for g in genomes],
			ctx.get('best_genome'),
			(ctx.get('best_fitness'), ctx.get('best_accuracy')),
			ctx.get('threshold', 0.0),
			{'patience_counter': getattr(ctx.get('early_stopper'), '_patience_counter', 0),
			 'complete': complete},
		)

	# =========================================================================
	# Simplified optimize: setup + super() + validation
	# =========================================================================

	def optimize(
		self,
		evaluate_fn: Callable[['ClusterGenome'], float] = None,
		initial_genome: Optional['ClusterGenome'] = None,
		initial_population: Optional[list['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[list['ClusterGenome']], list[tuple[float, float]]]] = None,
		**kwargs,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run GA with optional Rust acceleration.

		Sets up Rust state, delegates to unified loop in base class (which uses
		our _generate_offspring override for Rust-accelerated offspring), then
		runs validation summary on full data.
		"""
		import time

		# Checkpoint manager setup. Resume state (consumed by _run_optimization_loop):
		# default to a fresh run; overwritten below if a checkpoint is loaded.
		self.restore_resume_state(0, 0)
		self._checkpoint_mgr: Optional[PhasedCheckpointManager] = None
		cfg_ck = self._checkpoint_config
		if cfg_ck and cfg_ck.enabled and cfg_ck.checkpoint_dir:
			# Unified store (same as the controller): codec-based yaml.gz + adaptive
			# cadence. Path stem matches the legacy "{prefix}_ga.json" so a worker
			# restart resumes pre-migration .json checkpoints (find_checkpoint_file).
			self._checkpoint_mgr = PhasedCheckpointManager(
				cfg_ck.checkpoint_dir / f"{cfg_ck.filename_prefix}_ga",
				ClusterGenomeCodec(),
				SaveCadence(cfg_ck.target_loss_seconds, cfg_ck.max_interval),
				logger=self._log.info,
			)
			# Checkpoint resume
			if self._checkpoint_mgr.has_checkpoint():
				resume_state = _ids_resume_from_checkpoint(self._checkpoint_mgr.load())
				_extra = resume_state.get('extra_state') or {}
				_resume_pop = [g for g, _ in resume_state['population']]
				# GUARD: a completion checkpoint is written with population=[] +
				# complete=True purely as a "done" marker (see the final save below).
				# Resuming from it would seed an EMPTY population and offset the
				# generation counter — silently poisoning the run. Never resume from
				# a complete OR empty checkpoint; fall back to a fresh start (the
				# provided grid-seeded initial_population) and say so loudly.
				if _extra.get('complete') or not _resume_pop:
					reason = "marked complete" if _extra.get('complete') else "has an empty population"
					self._log.warning(
						f"[{self.name}] Checkpoint {reason} — NOT resuming from it; "
						f"starting fresh from the provided seed population.")
				else:
					self._log.info(f"[{self.name}] Resuming from checkpoint at generation {resume_state['current_iteration']}")
					# Restore population as initial_population (will be re-evaluated).
					initial_population = _resume_pop
					# Restore GA control state so resume CONTINUES from the exact stop.
					# The checkpoint is saved in _on_generation_start with
					# iterations_run=generation (the IN-PROGRESS gen, whose population +
					# patience are the post-(gen-1) state), so we resume AT that gen — NOT
					# +1, which would SKIP the in-progress gen and drop its patience tick.
					# (threshold is a pure function of generation, so it follows start_gen;
					#  best_fitness is recomputed from the restored population.)
					self.restore_resume_state(
						int(resume_state['current_iteration']),
						int(_extra.get('patience_counter', 0)),
					)

		# Set up phase state for Rust acceleration
		if self._cached_evaluator is not None:
			# Use explicit train subset if provided (phased_search cycles through subsets),
			# otherwise pick randomly. Ensures different phases use different data.
			if 'train_subset_idx' in kwargs:
				self._phase_train_idx = kwargs.pop('train_subset_idx')
			else:
				self._ensure_rng()
				self._phase_train_idx = self._cached_evaluator.random_train_idx(self._rng)
			self._log.info(f"[{self.name}] Using train subset {self._phase_train_idx}")
			self._seed_offset = int(time.time() * 1000) % (2**16)
			cfg = self._config

			# Ensure all seed genomes have connections
			if initial_population:
				for g in initial_population:
					if not g.has_connections():
						g.initialize_connections(self._cached_evaluator.total_input_bits)

				# Expand population with mutations if needed (unless seed_only)
				seed_count = len(initial_population)
				need_count = cfg.population_size - seed_count
				if need_count > 0 and not cfg.seed_only and not cfg.fresh_population:
					from wnn.ram.strategies.connectivity.adaptive_cluster import AdaptiveClusterConfig
					arch_cfg = self._arch_config
					mutation_config = AdaptiveClusterConfig(
						min_bits=arch_cfg.min_bits,
						max_bits=arch_cfg.max_bits,
						min_neurons=arch_cfg.min_neurons,
						max_neurons=arch_cfg.max_neurons,
					)
					mutation_rate = 0.3
					expanded = list(initial_population)
					# Dedup: track known fingerprints to avoid duplicate mutants
					known_fps = set()
					for g in initial_population:
						if hasattr(g, 'fingerprint'):
							known_fps.add(g.fingerprint())
					for i in range(need_count):
						seed = initial_population[i % seed_count]
						mutated = seed.mutate(
							self._phase_type, mutation_rate,
							mutation_config,
							self._cached_evaluator.total_input_bits,
							self._rng,
						)
						# Re-mutate if duplicate (up to 3 retries)
						if hasattr(mutated, 'fingerprint'):
							for _ in range(3):
								fp = mutated.fingerprint()
								if fp not in known_fps:
									break
								mutated = seed.mutate(
									self._phase_type, mutation_rate,
									mutation_config,
									self._cached_evaluator.total_input_bits,
									self._rng,
								)
							known_fps.add(mutated.fingerprint())
						expanded.append(mutated)
					initial_population = expanded

			if cfg.fresh_population:
				initial_population = None

			# Wrap cached evaluator as batch_evaluate_fn
			evaluator = self._cached_evaluator
			phase_train_idx = self._phase_train_idx
			batch_evaluate_fn = lambda genomes, min_accuracy=None: evaluator.evaluate_batch(
				genomes,
				train_subset_idx=phase_train_idx,
				eval_subset_idx=0,
				logger=self._log,
				min_accuracy=min_accuracy,
			)

		elif self._batch_evaluator is not None and batch_evaluate_fn is None:
			batch_evaluate_fn = lambda genomes, min_accuracy=None: self._batch_evaluator.evaluate_batch(
				genomes, logger=self._log, min_accuracy=min_accuracy,
			)

		# Start live progress observer — covers genesis, re-evaluation, and all generations
		observer = self._start_live_observer()
		try:
			# Delegate to unified loop (uses our _generate_offspring override)
			result = super().optimize(
				evaluate_fn=evaluate_fn,
				initial_genome=initial_genome,
				initial_population=initial_population,
				batch_evaluate_fn=batch_evaluate_fn,
				**kwargs,
			)
		finally:
			self._stop_live_observer(observer)

		# Validation summary (Rust path only: full-data evaluation)
		if self._cached_evaluator is not None:
			result = self._run_validation_summary(result)

		# Flip the per-gen checkpoint to complete=True. Resume logic uses this
		# flag to decide whether to re-enter the GA loop (False) or skip the
		# experiment entirely (True; already done).
		# NEVER flip on SHUTDOWN: _on_generation_start just wrote the resume
		# checkpoint (population + gen + patience) for exactly this case, and
		# flipping would clobber it with the empty "done" marker — the next
		# admission would then see complete=True and restart from gen 1
		# (flow 4326 lost 83 gens to this on 09/07/2026; graceful stops broke
		# resume while hard kills — which skip this code — preserved it).
		if self._checkpoint_mgr is not None and result is not None \
				and getattr(result, 'stop_reason', None) == StopReason.SHUTDOWN:
			self._log.info(
				f"[{self.name}] Stopped by shutdown — keeping the resume checkpoint "
				f"(complete flag NOT set; next admission resumes from the saved gen)")
		elif self._checkpoint_mgr is not None and result is not None:
			try:
				final_iter = getattr(result, 'iterations_run', None) or self._config.generations
				best_genome = getattr(result, 'best_genome', None)
				best_fitness = getattr(result, 'best_fitness', None)
				if best_genome is not None:
					self._checkpoint_mgr.save(_ids_to_checkpoint(
						self._phase_name, int(final_iter) - 1,
						[],  # no need to persist final pop; experiment-level checkpoint covers it
						best_genome,
						best_fitness if best_fitness is not None else (0.0, 0.0),
						0.0,
						{'complete': True},
					))
			except Exception as exc:
				self._log.warning(f"[{self.name}] Could not flip checkpoint complete=True: {exc}")

		return result
