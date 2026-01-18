"""
Architecture optimization strategies using generic GA/TS base classes.

These implement ClusterGenome-specific operations while reusing the core
GA/TS algorithms from generic_strategies.py.

Features:
- Rust/Metal batch evaluation support for parallel genome evaluation
- ProgressTracker integration for consistent logging
- Population seeding between phases (GA → TS → GA → ...)
"""

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

from wnn.progress import ProgressTracker
from wnn.ram.strategies.connectivity.generic_strategies import (
	GenericGAStrategy,
	GenericTSStrategy,
	GAConfig,
	TSConfig,
	OptimizerResult,
)
from wnn.ram.architecture.genome_log import (
	GenomeLogType,
	format_genome_log,
	format_gen_prefix,
)

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		RustParallelEvaluator,
		AdaptiveClusterConfig,
	)


@dataclass
class ArchitectureConfig:
	"""
	Configuration for architecture optimization.

	Controls both the search space bounds and what gets optimized.
	The optimizer is phase-agnostic - callers control what to optimize
	by setting the optimize_* flags.

	Example usage:
		# Phase 1: Optimize neurons only (bits fixed at default_bits)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=False,
			optimize_neurons=True,
			default_bits=8,  # All genomes start with 8 bits
		)

		# Phase 2: Optimize bits only (pass seed genome from Phase 1)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=True,
			optimize_neurons=False,
		)

		# Phase 3: Optimize connections only (pass seed genome from Phase 2)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=False,
			optimize_neurons=False,
			optimize_connections=True,
		)
	"""
	num_clusters: int
	min_bits: int = 4
	max_bits: int = 20
	min_neurons: int = 1
	max_neurons: int = 15
	# Explicit control over what gets optimized (no magic phase numbers)
	optimize_bits: bool = True
	optimize_neurons: bool = True
	optimize_connections: bool = False
	# Default values for dimensions not being optimized (used in random genome init)
	default_bits: int = 8
	default_neurons: int = 5
	# Token frequencies for frequency-scaled initialization
	token_frequencies: Optional[list[int]] = None
	# Total input bits for connection initialization/mutation
	total_input_bits: Optional[int] = None


class ArchitectureGAStrategy(GenericGAStrategy['ClusterGenome']):
	"""
	Genetic Algorithm for architecture (bits, neurons per cluster) optimization.

	Inherits core GA loop from GenericGAStrategy, implements ClusterGenome operations.

	Features:
	- Rust/Metal batch evaluation (default when available)
	- Rust-based offspring search with threshold (when cached_evaluator provided)
	- ProgressTracker for consistent logging with accuracy
	- Population seeding from previous phases
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ga_config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
		cached_evaluator: Optional[Any] = None,  # CachedEvaluator for Rust search_offspring
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
		self._tracker: Optional[ProgressTracker] = None

	@property
	def name(self) -> str:
		return "ArchitectureGA"

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> 'ClusterGenome':
		"""
		Mutate genome based on optimize_* flags in config.

		The optimizer is phase-agnostic. Callers control what gets optimized
		by setting optimize_bits, optimize_neurons, optimize_connections.

		Mutation delta ranges are 10% of (min + max):
		- Neurons: 10% × (min_neurons + max_neurons)
		- Bits: 10% × (min_bits + max_bits)
		- Connections: 10% × bits_per_token (stays close to token boundaries)

		Also adjusts connections when architecture changes.
		"""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		# Calculate delta ranges: 10% of (min + max), minimum 1
		bits_delta_max = max(1, round(0.1 * (cfg.min_bits + cfg.max_bits)))
		neurons_delta_max = max(1, round(0.1 * (cfg.min_neurons + cfg.max_neurons)))

		# Track old architecture for connection adjustment
		old_bits = genome.bits_per_cluster.copy()
		old_neurons = genome.neurons_per_cluster.copy()

		# If only optimizing connections, skip architecture mutation
		if cfg.optimize_connections and not cfg.optimize_bits and not cfg.optimize_neurons:
			if genome.connections is not None and cfg.total_input_bits is not None:
				mutant.connections = self._mutate_connections_only(
					genome.connections.copy(), cfg.total_input_bits, mutation_rate
				)
			return mutant

		# Mutate architecture (bits and/or neurons)
		for i in range(cfg.num_clusters):
			if self._rng.random() < mutation_rate:
				if cfg.optimize_bits:
					# Random delta in [-bits_delta_max, +bits_delta_max]
					delta = self._rng.randint(-bits_delta_max, bits_delta_max)
					new_bits = mutant.bits_per_cluster[i] + delta
					mutant.bits_per_cluster[i] = max(cfg.min_bits, min(cfg.max_bits, new_bits))

				if cfg.optimize_neurons:
					# Random delta in [-neurons_delta_max, +neurons_delta_max]
					delta = self._rng.randint(-neurons_delta_max, neurons_delta_max)
					new_neurons = mutant.neurons_per_cluster[i] + delta
					mutant.neurons_per_cluster[i] = max(cfg.min_neurons, min(cfg.max_neurons, new_neurons))

		# Adjust connections if they exist and architecture changed
		if genome.connections is not None and cfg.total_input_bits is not None:
			mutant.connections = self._adjust_connections_for_mutation(
				genome, mutant, old_bits, old_neurons, cfg.total_input_bits
			)

		return mutant

	def _mutate_connections_only(
		self,
		connections: list[int],
		total_input_bits: int,
		mutation_rate: float,
	) -> list[int]:
		"""
		Mutate connections without changing architecture.

		Delta range is 10% of bits_per_token (stays close to token boundaries).
		Assumes context × bits_per_token = total_input_bits.
		"""
		# Estimate bits_per_token: typically 16 for GPT-2 vocab
		# Use sqrt heuristic: bits_per_token ≈ log2(vocab) ≈ 16
		bits_per_token = 16  # Could be passed in, but 16 is reasonable default
		conn_delta_max = max(1, round(0.1 * bits_per_token))  # 10% of 16 = 2

		result = connections.copy()
		for i in range(len(result)):
			if self._rng.random() < mutation_rate:
				delta = self._rng.randint(-conn_delta_max, conn_delta_max)
				result[i] = max(0, min(total_input_bits - 1, result[i] + delta))
		return result

	def _adjust_connections_for_mutation(
		self,
		old_genome: 'ClusterGenome',
		new_genome: 'ClusterGenome',
		old_bits: list[int],
		old_neurons: list[int],
		total_input_bits: int,
	) -> list[int]:
		"""Adjust connections when architecture changes during mutation."""
		result = []
		old_idx = 0

		for cluster_idx in range(len(new_genome.bits_per_cluster)):
			o_neurons = old_neurons[cluster_idx]
			o_bits = old_bits[cluster_idx]
			n_neurons = new_genome.neurons_per_cluster[cluster_idx]
			n_bits = new_genome.bits_per_cluster[cluster_idx]

			for neuron_idx in range(n_neurons):
				if neuron_idx < o_neurons:
					# Existing neuron - copy and adjust connections
					for bit_idx in range(n_bits):
						if bit_idx < o_bits:
							# Copy existing connection, with small random mutation
							conn_idx = old_idx + neuron_idx * o_bits + bit_idx
							old_conn = old_genome.connections[conn_idx]
							# 10% chance of small perturbation
							if self._rng.random() < 0.1:
								delta = self._rng.choice([-2, -1, 1, 2])
								new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
							else:
								new_conn = old_conn
							result.append(new_conn)
						else:
							# New bit position - add random connection
							result.append(self._rng.randint(0, total_input_bits - 1))
				else:
					# New neuron - copy connections from random existing neuron with mutations
					if o_neurons > 0:
						template_neuron = self._rng.randint(0, o_neurons - 1)
						for bit_idx in range(n_bits):
							if bit_idx < o_bits:
								# Copy from template with mutation
								conn_idx = old_idx + template_neuron * o_bits + bit_idx
								old_conn = old_genome.connections[conn_idx]
								delta = self._rng.choice([-2, -1, 1, 2])
								new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
								result.append(new_conn)
							else:
								result.append(self._rng.randint(0, total_input_bits - 1))
					else:
						# No existing neurons to copy from - fully random
						for _ in range(n_bits):
							result.append(self._rng.randint(0, total_input_bits - 1))

			# Update old_idx for next cluster
			old_idx += o_neurons * o_bits

		return result

	def crossover_genomes(self, parent1: 'ClusterGenome', parent2: 'ClusterGenome') -> 'ClusterGenome':
		"""
		Single-point crossover at cluster boundary.

		Connections are inherited from the parent whose cluster config is taken.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		self._ensure_rng()
		n = len(parent1.bits_per_cluster)
		crossover_point = self._rng.randint(1, n - 1)

		# Build child architecture
		child_bits = parent1.bits_per_cluster[:crossover_point] + parent2.bits_per_cluster[crossover_point:]
		child_neurons = parent1.neurons_per_cluster[:crossover_point] + parent2.neurons_per_cluster[crossover_point:]

		# Build child connections if parents have them
		child_connections = None
		if parent1.connections is not None and parent2.connections is not None:
			child_connections = []
			p1_idx = 0
			p2_idx = 0
			for i in range(n):
				p1_conn_size = parent1.neurons_per_cluster[i] * parent1.bits_per_cluster[i]
				p2_conn_size = parent2.neurons_per_cluster[i] * parent2.bits_per_cluster[i]

				if i < crossover_point:
					# Take from parent1
					child_connections.extend(parent1.connections[p1_idx:p1_idx + p1_conn_size])
				else:
					# Take from parent2
					child_connections.extend(parent2.connections[p2_idx:p2_idx + p2_conn_size])

				p1_idx += p1_conn_size
				p2_idx += p2_conn_size

		return ClusterGenome(
			bits_per_cluster=child_bits,
			neurons_per_cluster=child_neurons,
			connections=child_connections,
		)

	def create_random_genome(self) -> 'ClusterGenome':
		"""
		Create a random genome based on optimize_* flags.

		- If optimize_bits=True: random bits in [min_bits, max_bits]
		- If optimize_bits=False: use default_bits for all clusters
		- Same logic for neurons

		When optimizing connections only, both bits and neurons use defaults.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		self._ensure_rng()
		cfg = self._arch_config

		if cfg.token_frequencies is not None:
			return self._create_frequency_scaled_genome()

		# Initialize bits: random if optimizing, default otherwise
		if cfg.optimize_bits:
			bits = [self._rng.randint(cfg.min_bits, cfg.max_bits) for _ in range(cfg.num_clusters)]
		else:
			bits = [cfg.default_bits] * cfg.num_clusters

		# Initialize neurons: random if optimizing, default otherwise
		if cfg.optimize_neurons:
			neurons = [self._rng.randint(cfg.min_neurons, cfg.max_neurons) for _ in range(cfg.num_clusters)]
		else:
			neurons = [cfg.default_neurons] * cfg.num_clusters

		# Initialize connections if total_input_bits available
		connections = None
		if cfg.total_input_bits is not None:
			connections = []
			for i in range(cfg.num_clusters):
				for _ in range(neurons[i]):
					for _ in range(bits[i]):
						connections.append(self._rng.randint(0, cfg.total_input_bits - 1))

		return ClusterGenome(bits_per_cluster=bits, neurons_per_cluster=neurons, connections=connections)

	def _create_frequency_scaled_genome(self) -> 'ClusterGenome':
		"""
		Create genome with bits/neurons scaled by token frequency.

		- If optimize_bits=True: scale bits by frequency
		- If optimize_bits=False: use default_bits
		- Same logic for neurons
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		cfg = self._arch_config
		freqs = cfg.token_frequencies

		# Normalize frequencies to [0, 1]
		max_freq = max(freqs) if freqs else 1
		norm_freqs = [f / max_freq if max_freq > 0 else 0 for f in freqs]

		bits = []
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

			bits.append(max(cfg.min_bits, min(cfg.max_bits, b)))
			neurons.append(max(cfg.min_neurons, min(cfg.max_neurons, n)))

		# Initialize connections if total_input_bits available
		connections = None
		if cfg.total_input_bits is not None:
			connections = []
			for i in range(cfg.num_clusters):
				for _ in range(neurons[i]):
					for _ in range(bits[i]):
						connections.append(self._rng.randint(0, cfg.total_input_bits - 1))

		return ClusterGenome(bits_per_cluster=bits, neurons_per_cluster=neurons, connections=connections)

	def optimize(
		self,
		evaluate_fn: Callable[['ClusterGenome'], float],
		initial_genome: Optional['ClusterGenome'] = None,
		initial_population: Optional[list['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[list['ClusterGenome']], list[tuple[float, float]]]] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run GA with Rust batch evaluation and ProgressTracker logging.

		If cached_evaluator was provided at init, uses Rust search_offspring for
		offspring generation (eliminates Python↔Rust round trips).

		If batch_evaluator was provided at init, uses it for parallel evaluation.
		batch_evaluate_fn should return list[(CE, accuracy)] tuples.
		"""
		# If we have a cached_evaluator, use Rust-based offspring search
		if self._cached_evaluator is not None:
			return self._optimize_with_rust_search(
				initial_population=initial_population,
			)

		# Fall back to original behavior
		if self._batch_evaluator is not None and batch_evaluate_fn is None:
			batch_evaluate_fn = lambda genomes, min_accuracy=None: self._batch_evaluator.evaluate_batch(
				genomes, logger=self._log, min_accuracy=min_accuracy,
			)

		# Create progress tracker
		self._tracker = ProgressTracker(
			logger=self._log,
			minimize=True,
			prefix=f"[{self.name}]",
			total_generations=self._config.generations,
		)

		return super().optimize(
			evaluate_fn=evaluate_fn,
			initial_genome=initial_genome,
			initial_population=initial_population,
			batch_evaluate_fn=batch_evaluate_fn,
		)

	def _optimize_with_rust_search(
		self,
		initial_population: Optional[list['ClusterGenome']] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run GA using Rust search_offspring for offspring generation.

		This eliminates Python↔Rust round trips by doing tournament selection,
		crossover, mutation, evaluation, and filtering entirely in Rust.
		"""
		import time
		from wnn.ram.strategies.connectivity.generic_strategies import (
			OptimizerResult, EarlyStoppingConfig, EarlyStoppingTracker
		)

		cfg = self._config
		arch_cfg = self._arch_config
		evaluator = self._cached_evaluator

		# Threshold continuity
		start_threshold = cfg.initial_threshold if cfg.initial_threshold is not None else cfg.min_accuracy
		end_threshold = start_threshold + cfg.threshold_delta

		def get_threshold(progress: float = 0.0) -> float:
			if not cfg.progressive_threshold:
				return start_threshold
			progress = max(0.0, min(1.0, progress))
			return start_threshold + progress * cfg.threshold_delta

		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.4%} → {end_threshold:.4%}")
		self._log.info(f"[{self.name}] Using Rust search_offspring (single-call offspring search)")

		# Initialize population (show ALL genomes - no threshold filtering for init)
		init_size = len(initial_population) if initial_population else cfg.population_size
		self._log.info(f"[{self.name}] Initializing population with {init_size} random genomes (no threshold filter)")

		if initial_population and len(initial_population) > 0:
			# Evaluate initial population
			results = evaluator.evaluate_batch(
				initial_population,
				train_subset_idx=evaluator.next_train_idx(),
				eval_subset_idx=0,
				logger=self._log,
				generation=0,
				total_generations=cfg.generations,
				min_accuracy=None,  # Show all for initial population
			)
			# Store cached fitness on each genome for elite logging
			population = []
			for g, (ce, acc) in zip(initial_population, results):
				g._cached_fitness = (ce, acc)
				population.append((g, ce))
		else:
			# Generate random initial population
			random_genomes = [self.create_random_genome() for _ in range(cfg.population_size)]
			results = evaluator.evaluate_batch(
				random_genomes,
				train_subset_idx=evaluator.next_train_idx(),
				eval_subset_idx=0,
				logger=self._log,
				generation=0,
				total_generations=cfg.generations,
				min_accuracy=None,  # Show all for initial population
			)
			# Store cached fitness on each genome for elite logging
			population = []
			for g, (ce, acc) in zip(random_genomes, results):
				g._cached_fitness = (ce, acc)
				population.append((g, ce))

		# Sort by fitness (lower CE is better)
		population.sort(key=lambda x: x[1])

		# Track best
		best_genome, best_fitness = population[0]
		initial_fitness = best_fitness

		# Early stopping
		from wnn.ram.strategies.connectivity.generic_strategies import EarlyStoppingConfig
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stop = EarlyStoppingTracker(early_stop_config, self._log.debug, self.name)
		early_stop.reset(best_fitness)

		# Track threshold for logging
		prev_threshold: Optional[float] = None
		seed_offset = int(time.time() * 1000) % (2**16)

		for generation in range(cfg.generations):
			gen_start = time.time()
			current_threshold = get_threshold(generation / cfg.generations)
			# Only log if formatted values differ
			if prev_threshold is not None and f"{prev_threshold:.4%}" != f"{current_threshold:.4%}":
				self._log.debug(f"[{self.name}] Threshold changed: {prev_threshold:.4%} → {current_threshold:.4%}")
			prev_threshold = current_threshold

			# Get train subset for this generation
			train_idx = evaluator.next_train_idx()

			# Dual elitism: 5 best by CE + 5 best by Acc (all unique, NO overlaps)
			elite_per_metric = max(1, int(cfg.elitism_pct * len(population)))  # 5 per metric

			# Population is already sorted by CE, so top 5 by CE are first 5
			elites_by_ce = population[:elite_per_metric]

			# For Acc elites, we need to re-evaluate with accuracy (cached in _cached_fitness)
			# Extract accuracy from each genome and sort by it (descending)
			pop_with_acc = []
			for g, ce in population:
				if hasattr(g, '_cached_fitness') and g._cached_fitness is not None:
					_, acc = g._cached_fitness
					pop_with_acc.append((g, ce, acc))
				else:
					# If no cached accuracy, assume 0
					pop_with_acc.append((g, ce, 0.0))

			# Sort by accuracy descending to get top by accuracy
			pop_by_acc = sorted(pop_with_acc, key=lambda x: -x[2])

			# Take top 5 by accuracy that are NOT already in CE elites
			ce_elite_genomes = set(id(g) for g, _ in elites_by_ce)
			elites_by_acc = []
			for g, ce, acc in pop_by_acc:
				if id(g) not in ce_elite_genomes:
					elites_by_acc.append((g, ce))
					if len(elites_by_acc) >= elite_per_metric:
						break

			# Combine: 5 CE + 5 Acc = 10 elites (all unique, no overlaps)
			elites = elites_by_ce + elites_by_acc
			elite_count = len(elites)

			# Log elites using shared formatter
			gen_prefix = format_gen_prefix(generation + 1, cfg.generations)
			self._log.info(f"{gen_prefix} Elites: {elite_per_metric} CE + {elite_per_metric} Acc = {elite_count} total")
			for i, (g, ce) in enumerate(elites_by_ce):
				acc = g._cached_fitness[1] if hasattr(g, '_cached_fitness') and g._cached_fitness else 0.0
				self._log.info(format_genome_log(
					generation + 1, cfg.generations, GenomeLogType.ELITE_CE,
					i + 1, elite_count, ce, acc
				))
			for i, (g, ce) in enumerate(elites_by_acc):
				acc = g._cached_fitness[1] if hasattr(g, '_cached_fitness') and g._cached_fitness else 0.0
				self._log.info(format_genome_log(
					generation + 1, cfg.generations, GenomeLogType.ELITE_ACC,
					elite_per_metric + i + 1, elite_count, ce, acc
				))

			# Generate offspring using Rust
			needed_offspring = cfg.population_size - elite_count
			offspring = evaluator.search_offspring(
				population=population,
				target_count=needed_offspring,
				max_attempts=needed_offspring * 5,  # 5x cap
				accuracy_threshold=current_threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				mutation_rate=cfg.mutation_rate,
				crossover_rate=cfg.crossover_rate,
				tournament_size=cfg.tournament_size,
				train_subset_idx=train_idx,
				eval_subset_idx=0,
				seed=seed_offset + generation,
				generation=generation,
				total_generations=cfg.generations,
			)

			# Build new population: elites + offspring
			new_population = list(elites)
			for g in offspring:
				if hasattr(g, '_cached_fitness'):
					ce, acc = g._cached_fitness
					new_population.append((g, ce))

			# Sort and truncate
			new_population.sort(key=lambda x: x[1])
			population = new_population[:cfg.population_size]

			# Update best
			if population[0][1] < best_fitness:
				best_genome, best_fitness = population[0]

			# Log generation summary with duration
			gen_elapsed = time.time() - gen_start
			avg_fitness = sum(ce for _, ce in population) / len(population)
			self._log.info(f"[{self.name}] Gen {generation+1:03d}/{cfg.generations}: "
						   f"best={best_fitness:.4f}, avg={avg_fitness:.4f} ({gen_elapsed:.1f}s)")

			# Early stopping check
			if early_stop.check(generation, best_fitness):
				self._log.info(f"[{self.name}] Early stopping at generation {generation + 1}")
				break

		# Final evaluation for accuracy
		final_ce, final_acc = evaluator.evaluate_batch(
			[best_genome],
			train_subset_idx=evaluator.next_train_idx(),
			eval_subset_idx=0,
			logger=self._log,
		)[0]

		return OptimizerResult(
			best_genome=best_genome,
			final_fitness=final_ce,
			initial_fitness=initial_fitness,
			iterations_run=generation + 1,
			final_population=[g for g, _ in population],
			final_threshold=current_threshold,
			final_accuracy=final_acc,
			initial_accuracy=None,
		)


class ArchitectureTSStrategy(GenericTSStrategy['ClusterGenome']):
	"""
	Tabu Search for architecture (bits, neurons per cluster) optimization.

	Inherits core TS loop from GenericTSStrategy, implements ClusterGenome operations.

	Features:
	- Rust/Metal batch evaluation (default when available)
	- Rust-based neighbor search with threshold (when cached_evaluator provided)
	- ProgressTracker for consistent logging with accuracy
	- Population seeding from previous phases
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ts_config: Optional[TSConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
		cached_evaluator: Optional[Any] = None,  # CachedEvaluator for Rust search_neighbors
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
		self._tracker: Optional[ProgressTracker] = None

	@property
	def name(self) -> str:
		return "ArchitectureTS"

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> tuple['ClusterGenome', Any]:
		"""
		Generate a neighbor by mutating multiple clusters based on mutation_rate.

		The optimizer is phase-agnostic. Callers control what gets optimized
		by setting optimize_bits, optimize_neurons, optimize_connections.

		Mutation delta ranges are 10% of (min + max):
		- Neurons: 10% × (min_neurons + max_neurons)
		- Bits: 10% × (min_bits + max_bits)
		- Connections: 10% × bits_per_token

		Returns (new_genome, move_info) where move_info is a tuple of mutated cluster indices
		for tabu tracking.
		"""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		# Calculate delta ranges: 10% of (min + max), minimum 1
		bits_delta_max = max(1, round(0.1 * (cfg.min_bits + cfg.max_bits)))
		neurons_delta_max = max(1, round(0.1 * (cfg.min_neurons + cfg.max_neurons)))
		bits_per_token = 16  # Reasonable default for GPT-2 vocab
		conn_delta_max = max(1, round(0.1 * bits_per_token))

		# If only optimizing connections, mutate connections based on mutation_rate
		if cfg.optimize_connections and not cfg.optimize_bits and not cfg.optimize_neurons:
			if genome.connections is not None and cfg.total_input_bits is not None:
				mutated_indices = []
				for i in range(len(genome.connections)):
					if self._rng.random() < mutation_rate:
						old_val = mutant.connections[i]
						delta = self._rng.randint(-conn_delta_max, conn_delta_max)
						new_val = max(0, min(cfg.total_input_bits - 1, old_val + delta))
						mutant.connections[i] = new_val
						mutated_indices.append(i)
				move = tuple(mutated_indices) if mutated_indices else None
			else:
				move = None
			return mutant, move

		# Track old architecture for connection adjustment
		old_bits = genome.bits_per_cluster.copy()
		old_neurons = genome.neurons_per_cluster.copy()

		# Mutate multiple clusters based on mutation_rate
		mutated_clusters = []
		for i in range(cfg.num_clusters):
			if self._rng.random() < mutation_rate:
				mutated_clusters.append(i)

				if cfg.optimize_bits:
					delta = self._rng.randint(-bits_delta_max, bits_delta_max)
					new_bits = mutant.bits_per_cluster[i] + delta
					mutant.bits_per_cluster[i] = max(cfg.min_bits, min(cfg.max_bits, new_bits))

				if cfg.optimize_neurons:
					delta = self._rng.randint(-neurons_delta_max, neurons_delta_max)
					new_neurons = mutant.neurons_per_cluster[i] + delta
					mutant.neurons_per_cluster[i] = max(cfg.min_neurons, min(cfg.max_neurons, new_neurons))

		# Move is tuple of mutated cluster indices (for tabu tracking)
		move = tuple(mutated_clusters) if mutated_clusters else None

		# Adjust connections if they exist and architecture changed
		if genome.connections is not None and cfg.total_input_bits is not None:
			mutant.connections = self._adjust_connections_for_mutation_ts(
				genome, mutant, old_bits, old_neurons, cfg.total_input_bits
			)

		return mutant, move

	def _adjust_connections_for_mutation_ts(
		self,
		old_genome: 'ClusterGenome',
		new_genome: 'ClusterGenome',
		old_bits: list[int],
		old_neurons: list[int],
		total_input_bits: int,
	) -> list[int]:
		"""Adjust connections when architecture changes during TS mutation."""
		result = []
		old_idx = 0

		for cluster_idx in range(len(new_genome.bits_per_cluster)):
			o_neurons = old_neurons[cluster_idx]
			o_bits = old_bits[cluster_idx]
			n_neurons = new_genome.neurons_per_cluster[cluster_idx]
			n_bits = new_genome.bits_per_cluster[cluster_idx]

			for neuron_idx in range(n_neurons):
				if neuron_idx < o_neurons:
					# Existing neuron - copy connections
					for bit_idx in range(n_bits):
						if bit_idx < o_bits:
							conn_idx = old_idx + neuron_idx * o_bits + bit_idx
							result.append(old_genome.connections[conn_idx])
						else:
							# New bit position - add random connection
							result.append(self._rng.randint(0, total_input_bits - 1))
				else:
					# New neuron - copy from random existing with slight mutation
					if o_neurons > 0:
						template_neuron = self._rng.randint(0, o_neurons - 1)
						for bit_idx in range(n_bits):
							if bit_idx < o_bits:
								conn_idx = old_idx + template_neuron * o_bits + bit_idx
								old_conn = old_genome.connections[conn_idx]
								delta = self._rng.choice([-2, -1, 1, 2])
								new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
								result.append(new_conn)
							else:
								result.append(self._rng.randint(0, total_input_bits - 1))
					else:
						for _ in range(n_bits):
							result.append(self._rng.randint(0, total_input_bits - 1))

			old_idx += o_neurons * o_bits

		return result

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

	def optimize(
		self,
		initial_genome: 'ClusterGenome',
		initial_fitness: float,
		evaluate_fn: Callable[['ClusterGenome'], float],
		initial_neighbors: Optional[list['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[list['ClusterGenome']], list[tuple[float, float]]]] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run TS with Rust batch evaluation and ProgressTracker logging.

		If cached_evaluator was provided at init, uses Rust search_neighbors for
		neighbor generation (eliminates Python↔Rust round trips).

		If batch_evaluator was provided at init, uses it for parallel evaluation.
		batch_evaluate_fn should return list[(CE, accuracy)] tuples.
		"""
		# If we have a cached_evaluator, use Rust-based neighbor search
		if self._cached_evaluator is not None:
			return self._optimize_with_rust_search(
				initial_genome=initial_genome,
				initial_fitness=initial_fitness,
				initial_neighbors=initial_neighbors,
			)

		# Fall back to original behavior
		if self._batch_evaluator is not None and batch_evaluate_fn is None:
			batch_evaluate_fn = lambda genomes, min_accuracy=None: self._batch_evaluator.evaluate_batch(
				genomes, logger=self._log, min_accuracy=min_accuracy,
			)

		# Create progress tracker
		self._tracker = ProgressTracker(
			logger=self._log,
			minimize=True,
			prefix=f"[{self.name}]",
			total_generations=self._config.iterations,
		)

		return super().optimize(
			initial_genome=initial_genome,
			initial_fitness=initial_fitness,
			evaluate_fn=evaluate_fn,
			initial_neighbors=initial_neighbors,
			batch_evaluate_fn=batch_evaluate_fn,
		)

	def _optimize_with_rust_search(
		self,
		initial_genome: 'ClusterGenome',
		initial_fitness: float,
		initial_neighbors: Optional[list['ClusterGenome']] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run TS using Rust search_neighbors for neighbor generation.

		This eliminates Python↔Rust round trips by doing mutation, evaluation,
		and filtering entirely in Rust. Much faster than the traditional approach.
		"""
		import time
		from collections import deque
		from wnn.ram.strategies.connectivity.generic_strategies import (
			OptimizerResult, EarlyStoppingConfig, EarlyStoppingTracker
		)

		cfg = self._config
		arch_cfg = self._arch_config
		evaluator = self._cached_evaluator

		# Threshold continuity
		start_threshold = cfg.initial_threshold if cfg.initial_threshold is not None else cfg.min_accuracy
		end_threshold = start_threshold + cfg.threshold_delta

		def get_threshold(progress: float = 0.0) -> float:
			if not cfg.progressive_threshold:
				return start_threshold
			progress = max(0.0, min(1.0, progress))
			return start_threshold + progress * cfg.threshold_delta

		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.2%} → {end_threshold:.2%}")
		self._log.info(f"[{self.name}] Using Rust search_neighbors (single-call neighbor search)")

		# Dual-path tracking
		best_ce_genome = initial_genome.clone()
		best_ce_fitness = initial_fitness
		best_ce_accuracy: Optional[float] = None

		best_acc_genome = initial_genome.clone()
		best_acc_fitness = initial_fitness
		best_acc_accuracy: Optional[float] = None

		# Global best
		best = initial_genome.clone()
		best_fitness = initial_fitness
		best_accuracy: Optional[float] = None
		start_fitness = initial_fitness

		# All neighbors cache
		all_neighbors: list[tuple['ClusterGenome', float, Optional[float]]] = [
			(initial_genome.clone(), initial_fitness, None)
		]

		current_threshold = get_threshold(0.0)

		# Seed with initial neighbors
		if initial_neighbors:
			self._log.info(f"[{self.name}] Seeding from {len(initial_neighbors)} neighbors")
			results = evaluator.evaluate_batch(initial_neighbors)
			for g, (ce, acc) in zip(initial_neighbors, results):
				all_neighbors.append((g.clone(), ce, acc))
				if ce < best_ce_fitness:
					best_ce_genome = g.clone()
					best_ce_fitness = ce
					best_ce_accuracy = acc
				if acc is not None and (best_acc_accuracy is None or acc > best_acc_accuracy):
					best_acc_genome = g.clone()
					best_acc_fitness = ce
					best_acc_accuracy = acc

			if best_ce_fitness < best_fitness:
				best = best_ce_genome.clone()
				best_fitness = best_ce_fitness
				best_accuracy = best_ce_accuracy

		history = [(0, best_fitness)]

		# Early stopping
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stopper = EarlyStoppingTracker(early_stop_config, self._log, self.name)
		early_stopper.reset(best_fitness)

		neighbors_per_path = cfg.neighbors_per_iter // 2
		self._log.info(f"[{self.name}] Config: neighbors={cfg.neighbors_per_iter} ({neighbors_per_path} CE + {neighbors_per_path} Acc), "
					   f"iters={cfg.iterations}")

		prev_threshold: Optional[float] = None
		iteration = 0
		seed_offset = int(time.time() * 1000) % (2**16)

		for iteration in range(cfg.iterations):
			current_threshold = get_threshold(iteration / cfg.iterations)
			# Only log if formatted values differ (avoid noise from tiny internal differences)
			if prev_threshold is not None and f"{prev_threshold:.4%}" != f"{current_threshold:.4%}":
				self._log.debug(f"[{self.name}] Threshold changed: {prev_threshold:.4%} → {current_threshold:.4%}")
			prev_threshold = current_threshold

			# Get next train subset for this iteration
			train_idx = evaluator.next_train_idx()

			# === Path A: Search neighbors from best_ce ===
			# max_attempts = 4x target gives room for threshold filtering
			ce_neighbors = evaluator.search_neighbors(
				genome=best_ce_genome,
				target_count=neighbors_per_path,
				max_attempts=neighbors_per_path * 5,
				accuracy_threshold=current_threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				bits_mutation_rate=cfg.mutation_rate if arch_cfg.optimize_bits else 0.0,
				neurons_mutation_rate=cfg.mutation_rate if arch_cfg.optimize_neurons else 0.0,
				train_subset_idx=train_idx,
				eval_subset_idx=0,
				seed=seed_offset + iteration * 1000,
				generation=iteration,
				total_generations=cfg.iterations,
			)

			# === Path B: Search neighbors from best_acc ===
			acc_neighbors = evaluator.search_neighbors(
				genome=best_acc_genome,
				target_count=neighbors_per_path,
				max_attempts=neighbors_per_path * 5,
				accuracy_threshold=current_threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				bits_mutation_rate=cfg.mutation_rate if arch_cfg.optimize_bits else 0.0,
				neurons_mutation_rate=cfg.mutation_rate if arch_cfg.optimize_neurons else 0.0,
				train_subset_idx=train_idx,
				eval_subset_idx=0,
				seed=seed_offset + iteration * 1000 + 500,
				generation=iteration,
				total_generations=cfg.iterations,
			)

			# Process CE path neighbors
			for g in ce_neighbors:
				ce, acc = g._cached_fitness
				all_neighbors.append((g.clone(), ce, acc))
				if ce < best_ce_fitness:
					best_ce_genome = g.clone()
					best_ce_fitness = ce
					best_ce_accuracy = acc
				if acc is not None and (best_acc_accuracy is None or acc > best_acc_accuracy):
					best_acc_genome = g.clone()
					best_acc_fitness = ce
					best_acc_accuracy = acc

			# Process Acc path neighbors
			for g in acc_neighbors:
				ce, acc = g._cached_fitness
				all_neighbors.append((g.clone(), ce, acc))
				if ce < best_ce_fitness:
					best_ce_genome = g.clone()
					best_ce_fitness = ce
					best_ce_accuracy = acc
				if acc is not None and (best_acc_accuracy is None or acc > best_acc_accuracy):
					best_acc_genome = g.clone()
					best_acc_fitness = ce
					best_acc_accuracy = acc

			# Update global best
			if best_ce_fitness < best_fitness:
				best = best_ce_genome.clone()
				best_fitness = best_ce_fitness
				best_accuracy = best_ce_accuracy

			history.append((iteration + 1, best_fitness))

			# Log iteration summary
			self._log.info(f"[{self.name}] Iter {iteration+1:03d}/{cfg.iterations}: "
						   f"best_ce={best_ce_fitness:.4f}, best_acc={best_acc_accuracy:.2%}" if best_acc_accuracy else
						   f"[{self.name}] Iter {iteration+1:03d}/{cfg.iterations}: best_ce={best_ce_fitness:.4f}")

			# Early stopping check
			if early_stopper.check(best_fitness):
				self._log.info(f"[{self.name}] Early stopping at iteration {iteration + 1}")
				break

		# Build final population (top by CE + top by Acc)
		cache_size = cfg.total_neighbors_size or cfg.neighbors_per_iter
		cache_size_per_metric = cache_size // 2

		by_ce = sorted([n for n in all_neighbors if n[2] is not None], key=lambda x: x[1])[:cache_size_per_metric]
		by_acc = sorted([n for n in all_neighbors if n[2] is not None], key=lambda x: -x[2])[:cache_size_per_metric]

		seen = set()
		final_population = []
		for g, _, _ in by_ce + by_acc:
			key = (tuple(g.bits_per_cluster[:10]), tuple(g.neurons_per_cluster[:10]))
			if key not in seen:
				seen.add(key)
				final_population.append(g)

		return OptimizerResult(
			best_genome=best,
			initial_fitness=start_fitness,
			final_fitness=best_fitness,
			iterations_run=iteration + 1,
			early_stopped=iteration + 1 < cfg.iterations,
			history=history,
			final_population=final_population,
			final_threshold=current_threshold,
			initial_accuracy=None,
			final_accuracy=best_accuracy,
		)
