"""
ArchitectureTSStrategy — tabu search over ClusterGenome on the generic TS core

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

from wnn.ram.strategies.connectivity.framework import TSConfig, OptimizerResult
from wnn.ram.strategies.connectivity.generic_ts import GenericTSStrategy
from wnn.ram.strategies.connectivity.adaptive_cluster import PhaseType
from wnn.ram.strategies.connectivity.genome_tracking import HAS_GENOME_TRACKING, TierConfig, GenomeConfig, GenomeRole
from wnn.ram.strategies.connectivity.architecture_mixin import ArchitectureStrategyMixin
from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig
from wnn.ram.strategies.connectivity.checkpoint_manager import CheckpointConfig

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		AdaptiveClusterConfig,
	)

class ArchitectureTSStrategy(ArchitectureStrategyMixin, GenericTSStrategy['ClusterGenome']):
	"""
	Tabu Search for architecture (bits, neurons per cluster) optimization.

	Inherits core TS loop from GenericTSStrategy, implements ClusterGenome operations.
	Uses ArchitectureStrategyMixin for shared functionality (Metal cleanup, shutdown, etc.)

	Features:
	- Rust/Metal batch evaluation (default when available)
	- Rust-based neighbor search with threshold (when cached_evaluator provided)
	- Population seeding from previous phases
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ts_config: Optional[TSConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional[Any] = None,
		cached_evaluator: Optional[Any] = None,  # BaseEvaluator for Rust search_neighbors
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
	):
		super().__init__(config=ts_config, seed=seed, logger=logger)
		self._arch_config = arch_config
		self._batch_evaluator = batch_evaluator
		# Use cached_evaluator if provided, or check if batch_evaluator has search_neighbors
		if cached_evaluator is not None:
			self._cached_evaluator = cached_evaluator
		elif batch_evaluator is not None and hasattr(batch_evaluator, 'search_neighbors'):
			self._cached_evaluator = batch_evaluator
		else:
			self._cached_evaluator = None
		self._shutdown_check = shutdown_check
		self._phase_type = self._derive_phase_type()

	@property
	def name(self) -> str:
		return "ArchitectureTS"

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> tuple['ClusterGenome', Any]:
		"""Phase-aware mutation dispatching to ClusterGenome.mutate().

		Returns (new_genome, move_info) where move_info is a hash of the mutated
		architecture (for tabu tracking).
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import AdaptiveClusterConfig
		self._ensure_rng()
		cfg = self._arch_config
		mutation_config = AdaptiveClusterConfig(
			min_bits=cfg.min_bits, max_bits=cfg.max_bits,
			min_neurons=cfg.min_neurons, max_neurons=cfg.max_neurons,
		)
		tib = cfg.total_input_bits or 64
		mutant = genome.mutate(self._phase_type, mutation_rate, mutation_config, tib, self._rng)

		# Compute move info for tabu tracking: tuple of changed cluster indices
		if self._phase_type == PhaseType.NEURONS:
			changed = tuple(c for c in range(len(genome.neurons_per_cluster))
						   if genome.neurons_per_cluster[c] != mutant.neurons_per_cluster[c])
		elif self._phase_type == PhaseType.BITS:
			changed = tuple(c for c in range(len(genome.bits_per_neuron))
						   if genome.bits_per_neuron[c] != mutant.bits_per_neuron[c])
		else:
			# Connections phase: track which clusters had any connection change
			changed_clusters = []
			if genome.connections is not None and mutant.connections is not None:
				g_off = genome.cluster_neuron_offsets
				g_conn_off = genome.connection_offsets
				m_conn_off = mutant.connection_offsets
				for c in range(len(genome.neurons_per_cluster)):
					c_start = g_conn_off[g_off[c]]
					c_end = g_conn_off[g_off[c + 1]]
					m_start = m_conn_off[g_off[c]]
					m_end = m_conn_off[g_off[c + 1]]
					if (c_end - c_start != m_end - m_start or
						genome.connections[c_start:c_end] != mutant.connections[m_start:m_end]):
						changed_clusters.append(c)
			changed = tuple(changed_clusters)
		move = changed if changed else None
		return mutant, move

	def is_tabu_move(self, move: Any, tabu_list: list[Any]) -> bool:
		"""
		Check if move overlaps significantly with recent tabu moves.

		Move is now a tuple of mutated cluster indices. A move is tabu if
		it shares more than 50% of clusters with a recent tabu move.
		"""
		if move is None or not move:
			return False

		move_set = set(move)
		for tabu_move in tabu_list:
			if tabu_move is None:
				continue
			tabu_set = set(tabu_move)
			overlap = len(move_set & tabu_set)
			# Tabu if >50% overlap with any recent move
			if overlap > len(move_set) * 0.5:
				return True

		return False

	# =========================================================================
	# Hooks: Rust-accelerated neighbor generation + lifecycle
	# =========================================================================

	def _compute_move_info(self, source: 'ClusterGenome', neighbor: 'ClusterGenome') -> Any:
		"""Compute tabu move info by comparing source and neighbor genomes."""
		if self._phase_type == PhaseType.NEURONS:
			changed = tuple(c for c in range(len(source.neurons_per_cluster))
						   if source.neurons_per_cluster[c] != neighbor.neurons_per_cluster[c])
		elif self._phase_type == PhaseType.BITS:
			changed = tuple(c for c in range(len(source.bits_per_neuron))
						   if source.bits_per_neuron[c] != neighbor.bits_per_neuron[c])
		else:
			# Connections phase: track which NEURONS had connection changes
			# (not clusters — single-cluster genomes only have cluster 0,
			# so cluster-level tracking makes every move tabu after the first)
			changed_neurons = []
			if source.connections is not None and neighbor.connections is not None:
				s_off = source.connection_offsets
				n_off = neighbor.connection_offsets
				num_neurons = len(source.bits_per_neuron)
				for n in range(num_neurons):
					s_conns = source.connections[s_off[n]:s_off[n + 1]]
					n_conns = neighbor.connections[n_off[n]:n_off[n + 1]]
					if s_conns != n_conns:
						changed_neurons.append(n)
			changed = tuple(changed_neurons)
		return changed if changed else None

	def _generate_neighbors(self, best_genome, n_neighbors, threshold, iteration, tabu_list):
		"""Generate neighbors via Rust search_neighbors or Python fallback."""
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

			# fitness_percentile: generate larger pool, rank, keep best n_neighbors
			# Also over-generate to compensate for tabu filtering
			import math
			pct = cfg.fitness_percentile if cfg.fitness_percentile and 0 < cfg.fitness_percentile < 1.0 else None
			generate_count = math.ceil(n_neighbors / pct) if pct else n_neighbors
			# Over-generate by 50% to compensate for tabu filtering
			if tabu_list:
				generate_count = math.ceil(generate_count * 1.5)

			self._log.debug(f"[{self.name}] Searching {generate_count} neighbors from best ranked (keeping best {n_neighbors})...")
			neighbors_raw = evaluator.search_neighbors(
				genome=best_genome,
				target_count=generate_count,
				max_attempts=generate_count * 5,
				accuracy_threshold=threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				bits_mutation_rate=bits_mutation_rate,
				neurons_mutation_rate=neurons_mutation_rate,
				train_subset_idx=self._phase_train_idx,
				eval_subset_idx=0,
				seed=self._seed_offset + iteration * 1000,
				logger=self._log,
				generation=iteration,
				total_generations=cfg.iterations,
				return_best_n=True,
				mutable_clusters=arch_cfg.mutable_clusters,
				phase_type=int(self._phase_type),
			)

			# Convert to (genome, Metrics) tuples, rank by fitness, return best n_neighbors
			neighbors = [
				(g, g.metrics)
				for g in neighbors_raw
				if hasattr(g, 'metrics') and g.metrics is not None
			]

			# Post-filter: remove tabu neighbors
			if tabu_list:
				non_tabu = []
				for t in neighbors:
					move = self._compute_move_info(best_genome, t[0])
					if not self.is_tabu_move(move, tabu_list):
						non_tabu.append(t)
				filtered_count = len(neighbors) - len(non_tabu)
				if filtered_count > 0:
					self._log.debug(f"[{self.name}] Tabu filtered {filtered_count}/{len(neighbors)} neighbors")
				neighbors = non_tabu

			if pct and len(neighbors) > n_neighbors:
				scores = self._fitness_calculator.fitness([t[1] for t in neighbors])
				ranked = sorted(zip(neighbors, scores), key=lambda x: x[1])
				neighbors = [item for item, _ in ranked[:n_neighbors]]

			# Add best neighbor's move to tabu list
			if neighbors:
				best_neighbor = neighbors[0]  # Already ranked or first viable
				if len(neighbors) > 1:
					# Find best by fitness ranking
					best_ranked_neighbors = self._fitness_calculator.rank(
						[t[0] for t in neighbors], [t[1] for t in neighbors]
					)
					best_neighbor = next(
						t for t in neighbors if t[0] is best_ranked_neighbors[0][0]
					)
				move = self._compute_move_info(best_genome, best_neighbor[0])
				if move is not None:
					tabu_list.append(move)

			return neighbors

		# Fallback to Python single-path generation
		return super()._generate_neighbors(best_genome, n_neighbors, threshold, iteration, tabu_list)

	def _generate_neighbors_batch(self, sources, counts, threshold, iteration, tabu_list):
		"""Generate neighbors for multiple sources in a single Rust evaluation call.

		Returns list of offspring lists, one per source. Falls back to per-source
		_generate_neighbors if cached evaluator doesn't support batch search.
		"""
		evaluator = self._cached_evaluator
		if evaluator is None or not hasattr(evaluator, 'search_neighbors_batch'):
			return None  # Signal caller to fall back to per-source loop

		cfg = self._config
		arch_cfg = self._arch_config

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

		import math
		pct = cfg.fitness_percentile if cfg.fitness_percentile and 0 < cfg.fitness_percentile < 1.0 else None

		# Build source list with inflated counts for fitness percentile + tabu filtering
		batch_sources = []
		for source, count in zip(sources, counts):
			gen_count = math.ceil(count / pct) if pct else count
			# Over-generate by 50% to compensate for tabu filtering
			if tabu_list:
				gen_count = math.ceil(gen_count * 1.5)
			batch_sources.append((source, gen_count))

		total_candidates = sum(gc for _, gc in batch_sources)
		self._log.info(
			f"[{self.name}] Batch searching {total_candidates} neighbors "
			f"from {len(sources)} sources"
		)

		def on_progress(batch_num, total_batches, done, total):
			self._log.info(
				f"[{self.name}] Evaluating {done}/{total} candidates "
				f"(sub-batch {batch_num}/{total_batches})"
			)

		if hasattr(evaluator, 'set_progress_callback'):
			evaluator.set_progress_callback(on_progress)
		try:
			results_by_source = evaluator.search_neighbors_batch(
				sources=batch_sources,
				max_attempts_multiplier=1,
				accuracy_threshold=threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				bits_mutation_rate=bits_mutation_rate,
				neurons_mutation_rate=neurons_mutation_rate,
				train_subset_idx=self._phase_train_idx,
				eval_subset_idx=0,
				seed=self._seed_offset + iteration * 1000,
				return_best_n=True,
				mutable_clusters=arch_cfg.mutable_clusters,
				phase_type=int(self._phase_type),
				logger=self._log,
				generation=iteration,
				total_generations=cfg.iterations,
			)
		finally:
			if hasattr(evaluator, 'set_progress_callback'):
				evaluator.set_progress_callback(None)

		# Convert to tuples, tabu-filter, and apply fitness percentile filtering per source
		all_offspring = []
		total_tabu_filtered = 0
		for (source, _), source_neighbors, target_count in zip(batch_sources, results_by_source, counts):
			neighbors = [
				(g, g.metrics)
				for g in source_neighbors
				if hasattr(g, 'metrics') and g.metrics is not None
			]

			# Post-filter: remove tabu neighbors
			if tabu_list:
				non_tabu = []
				for t in neighbors:
					move = self._compute_move_info(source, t[0])
					if not self.is_tabu_move(move, tabu_list):
						non_tabu.append(t)
				total_tabu_filtered += len(neighbors) - len(non_tabu)
				neighbors = non_tabu

			if pct and len(neighbors) > target_count:
				scores = self._fitness_calculator.fitness([t[1] for t in neighbors])
				ranked = sorted(zip(neighbors, scores), key=lambda x: x[1])
				neighbors = [item for item, _ in ranked[:target_count]]

			# Add best neighbor's move to tabu list
			if neighbors:
				best_neighbor = neighbors[0]
				if len(neighbors) > 1:
					best_ranked_neighbors = self._fitness_calculator.rank(
						[t[0] for t in neighbors], [t[1] for t in neighbors]
					)
					best_neighbor = next(
						t for t in neighbors if t[0] is best_ranked_neighbors[0][0]
					)
				move = self._compute_move_info(source, best_neighbor[0])
				if move is not None:
					tabu_list.append(move)

			all_offspring.append(neighbors)

		if total_tabu_filtered > 0:
			self._log.debug(f"[{self.name}] Tabu filtered {total_tabu_filtered} neighbors across all sources")

		return all_offspring

	def _on_iteration_start(self, iteration, **ctx):
		"""Metal cleanup, shutdown check, generation tracking."""
		# Update evaluator generation for adaptive evaluation (Baldwin effect)
		evaluator = self._cached_evaluator or self._batch_evaluator
		if evaluator is not None and hasattr(evaluator, 'set_generation'):
			evaluator.set_generation(iteration, total_generations=ctx.get('total_generations'))

		# Metal cleanup (every iteration except first)
		if iteration > 0 and self._cached_evaluator is not None:
			self._cleanup_metal(iteration, log_interval=10)

		# Shutdown check
		if self._shutdown_check and self._shutdown_check():
			self._log.info(f"[{self.name}] Shutdown requested at iteration {iteration}, stopping...")
			raise StopIteration("Shutdown requested")

	# =========================================================================
	# Simplified optimize: setup + super() + validation
	# =========================================================================

	def optimize(
		self,
		initial_genome: 'ClusterGenome' = None,
		initial_fitness: Optional[float] = None,
		evaluate_fn: Callable[['ClusterGenome'], float] = None,
		initial_neighbors: Optional[list['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[list['ClusterGenome']], list[tuple[float, float]]]] = None,
		**kwargs,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run TS with optional Rust acceleration.

		Sets up Rust state, delegates to unified loop in base class (which uses
		our _generate_neighbors override for Rust-accelerated search), then
		runs validation summary on full data.

		IMPORTANT: initial_fitness is REQUIRED.
		"""
		import time

		# initial_fitness is REQUIRED - fail fast if missing
		if initial_fitness is None:
			raise ValueError(
				f"[{self.name}] initial_fitness is REQUIRED but was None. "
				"This indicates the previous phase's final_fitness was not properly passed. "
				"Check that: (1) GA saved a checkpoint with final_fitness, "
				"(2) Flow checkpoint loading works correctly, "
				"(3) Flow is not creating new experiments instead of resuming."
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

			# Ensure initial genome has connections
			if not initial_genome.has_connections():
				initial_genome.initialize_connections(self._cached_evaluator.total_input_bits)
			if initial_neighbors:
				for g in initial_neighbors:
					if not g.has_connections():
						g.initialize_connections(self._cached_evaluator.total_input_bits)

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

		# Start live progress observer — covers initial eval, seeded neighbors, and all iterations
		observer = self._start_live_observer()
		try:
			# Delegate to unified loop (uses our _generate_neighbors override)
			result = super().optimize(
				initial_genome=initial_genome,
				initial_fitness=initial_fitness,
				evaluate_fn=evaluate_fn,
				initial_neighbors=initial_neighbors,
				batch_evaluate_fn=batch_evaluate_fn,
				**kwargs,
			)
		finally:
			self._stop_live_observer(observer)

		# Validation summary (Rust path only: full-data evaluation)
		if self._cached_evaluator is not None:
			result = self._run_validation_summary(result)

		return result


# =============================================================================
# Grid Search Strategy
# =============================================================================
