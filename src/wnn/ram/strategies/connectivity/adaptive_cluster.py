"""
Adaptive Cluster Architecture Search

Instead of hand-designed tiers, each cluster discovers its optimal architecture
through evolutionary optimization. The GA mutates both connectivity AND structure
(bits per neuron, number of neurons per cluster).

Phase 1: Bits-per-cluster only (neurons fixed at 1)
Phase 2: Add neuron count per cluster

Key insight: Frequent tokens need different architectures than rare tokens.
Let the data decide rather than hand-tuning tier boundaries.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
from torch import Tensor

from wnn.progress import ProgressTracker

if TYPE_CHECKING:
	pass


class GenomeInitStrategy(IntEnum):
	"""
	Initialization strategy for adaptive cluster genomes.

	Determines how bits-per-cluster are initialized before GA optimization.
	Each strategy represents a different prior belief about optimal architecture.
	"""

	UNIFORM_MINIMAL = 0
	"""All clusters start at minimum (1 bit = 2 addresses).

	Pros: Maximum exploration, no prior bias
	Cons: Slow convergence, everything starts tiny
	Use when: You want pure data-driven discovery
	"""

	UNIFORM_MEDIUM = 1
	"""All clusters start at medium (8 bits = 256 addresses).

	Pros: Balanced start, can grow or shrink equally
	Cons: May waste iterations if optimal is far from 8
	Use when: No strong prior about frequency distribution
	"""

	UNIFORM_MAXIMUM = 2
	"""All clusters start at maximum (e.g., 20 bits).

	Pros: Start with full capacity, shrink to fit
	Cons: Memory intensive, may be slow to shrink
	Use when: Memory is not a constraint
	"""

	FREQUENCY_SCALED = 3
	"""Scale initial bits by token frequency (recommended).

	Uses the tiered insight as prior: frequent tokens get more bits,
	rare tokens get fewer. GA refines from this informed starting point.

	Pros: Leverages known insight, faster convergence
	Cons: May anchor too strongly to prior
	Use when: You have token frequency data (most LM cases)
	"""

	RANDOM_UNIFORM = 4
	"""Random bits in valid range for each cluster.

	Pros: Diverse initial population
	Cons: No informed prior
	Use when: Exploring with population-based methods
	"""


@dataclass
class AdaptiveClusterConfig:
	"""Configuration for adaptive cluster optimization."""

	min_bits: int = 1
	"""Minimum bits per neuron (2^1 = 2 addresses)"""

	max_bits: int = 30
	"""Maximum bits per neuron (2^30 = 1B addresses)"""

	min_neurons: int = 1
	"""Minimum neurons per cluster (Phase 2)"""

	max_neurons: int = 50
	"""Maximum neurons per cluster (Phase 2)"""

	total_memory_budget: int = 1_000_000_000
	"""Total memory cells allowed across all clusters"""

	init_strategy: GenomeInitStrategy = GenomeInitStrategy.FREQUENCY_SCALED
	"""How to initialize the genome"""

	# Mutation rates (Phase 1: bits)
	bits_mutation_rate: float = 0.1
	"""Probability of mutating bits for a cluster"""

	bits_mutation_step: int = 1
	"""How much to change bits per mutation (+/- this value)"""

	# Mutation rates (Phase 2: neurons)
	neurons_mutation_rate: float = 0.05
	"""Probability of mutating neuron count for a cluster"""

	neurons_mutation_step: int = 1
	"""How much to change neuron count per mutation (+/- this value)"""

	# Phase control
	phase: int = 2
	"""Optimization phase: 1 = bits only, 2 = bits + neurons"""


# =============================================================================
# ClusterGenome - The DNA of an adaptive architecture
# =============================================================================

class ClusterGenome:
	"""
	Genome representing adaptive architecture for all clusters.

	Phase 1: bits_per_cluster (bits per neuron for each cluster)
	Phase 2: neurons_per_cluster (number of neurons for each cluster)

	Each cluster can have different (neurons, bits) configuration,
	allowing the GA to discover optimal architectures per token.

	Example:
		# Initialize from token frequencies
		genome = ClusterGenome.initialize(
			num_clusters=50257,
			strategy=GenomeInitStrategy.FREQUENCY_SCALED,
			config=config,
			token_frequencies=frequencies,
		)

		# Mutate
		child = genome.mutate(config)

		# Crossover
		child = genome.crossover(other_genome)

		# Statistics
		stats = genome.stats()
	"""

	def __init__(
		self,
		bits_per_cluster: List[int],
		neurons_per_cluster: Optional[List[int]] = None,
	):
		"""
		Create a genome with specified architecture.

		Args:
			bits_per_cluster: Bits per neuron for each cluster [num_clusters]
			neurons_per_cluster: Neurons per cluster [num_clusters] (default: all 1s)
		"""
		self.bits_per_cluster = bits_per_cluster
		self.neurons_per_cluster = (
			neurons_per_cluster if neurons_per_cluster is not None
			else [1] * len(bits_per_cluster)
		)

	# =========================================================================
	# Factory Methods
	# =========================================================================

	@classmethod
	def initialize(
		cls,
		num_clusters: int,
		strategy: GenomeInitStrategy,
		config: AdaptiveClusterConfig,
		token_frequencies: Optional[List[int]] = None,
		rng: Optional[int] = None,
	) -> ClusterGenome:
		"""
		Initialize a cluster genome using the specified strategy.

		Args:
			num_clusters: Total number of clusters (e.g., 50257 for GPT-2)
			strategy: Initialization strategy to use
			config: Configuration with min/max bounds
			token_frequencies: Token occurrence counts (required for FREQUENCY_SCALED)
			rng: Random seed for reproducibility

		Returns:
			Initialized ClusterGenome with bits and neurons per cluster

		Example:
			genome = ClusterGenome.initialize(
				num_clusters=50257,
				strategy=GenomeInitStrategy.FREQUENCY_SCALED,
				config=config,
				token_frequencies=token_counts,
			)
		"""
		if rng is not None:
			random.seed(rng)

		# Initialize bits based on strategy
		if strategy == GenomeInitStrategy.UNIFORM_MINIMAL:
			bits = [config.min_bits] * num_clusters

		elif strategy == GenomeInitStrategy.UNIFORM_MEDIUM:
			medium = (config.min_bits + config.max_bits) // 2
			bits = [medium] * num_clusters

		elif strategy == GenomeInitStrategy.UNIFORM_MAXIMUM:
			bits = [config.max_bits] * num_clusters

		elif strategy == GenomeInitStrategy.FREQUENCY_SCALED:
			if token_frequencies is None:
				raise ValueError("FREQUENCY_SCALED requires token_frequencies")
			bits = _frequency_scaled_init(
				num_clusters, token_frequencies, config, for_bits=True
			)

		elif strategy == GenomeInitStrategy.RANDOM_UNIFORM:
			# Random with frequency-aware caps to avoid slow sparse memory for rare tokens
			# Rare tokens (low frequency) get capped bits since they can't fill large address spaces
			bits = []
			for i in range(num_clusters):
				freq = token_frequencies[i] if token_frequencies else 1000
				# Cap max bits based on frequency (sparse threshold is 12)
				if freq < 10:
					max_b = min(config.max_bits, 8)   # Very rare: max 8 bits
				elif freq < 100:
					max_b = min(config.max_bits, 10)  # Rare: max 10 bits
				elif freq < 1000:
					max_b = min(config.max_bits, 12)  # Medium: max 12 bits (dense)
				else:
					max_b = config.max_bits           # Frequent: full range
				bits.append(random.randint(config.min_bits, max_b))
		else:
			raise ValueError(f"Unknown strategy: {strategy}")

		# Initialize neurons based on strategy (Phase 2)
		if config.phase >= 2:
			if strategy == GenomeInitStrategy.FREQUENCY_SCALED and token_frequencies is not None:
				neurons = _frequency_scaled_init(
					num_clusters, token_frequencies, config, for_bits=False
				)
			elif strategy in (GenomeInitStrategy.UNIFORM_MINIMAL, GenomeInitStrategy.UNIFORM_MEDIUM):
				neurons = [config.min_neurons] * num_clusters
			elif strategy == GenomeInitStrategy.UNIFORM_MAXIMUM:
				neurons = [config.max_neurons] * num_clusters
			elif strategy == GenomeInitStrategy.RANDOM_UNIFORM:
				neurons = [
					random.randint(config.min_neurons, config.max_neurons)
					for _ in range(num_clusters)
				]
			else:
				neurons = [1] * num_clusters
		else:
			neurons = [1] * num_clusters  # Phase 1: fixed at 1

		return cls(bits_per_cluster=bits, neurons_per_cluster=neurons)

	@classmethod
	def from_tensor(cls, t: Tensor) -> ClusterGenome:
		"""Create genome from tensor [num_clusters, 2] with (bits, neurons)."""
		return cls(
			bits_per_cluster=t[:, 0].tolist(),
			neurons_per_cluster=t[:, 1].tolist(),
		)

	# =========================================================================
	# Genetic Operations
	# =========================================================================

	def mutate(
		self,
		config: AdaptiveClusterConfig,
		rng: Optional[int] = None,
	) -> ClusterGenome:
		"""
		Create a mutated copy of this genome.

		Randomly adjusts bits and neurons per cluster based on config.
		Phase 1: Only mutates bits
		Phase 2: Mutates both bits and neurons

		Args:
			config: Configuration with mutation rates and bounds
			rng: Random seed

		Returns:
			New mutated genome (self unchanged)
		"""
		if rng is not None:
			random.seed(rng)

		new_bits = self.bits_per_cluster.copy()
		new_neurons = self.neurons_per_cluster.copy()

		for i in range(len(new_bits)):
			# Mutate bits
			if random.random() < config.bits_mutation_rate:
				delta = random.choice([-config.bits_mutation_step, config.bits_mutation_step])
				new_bits[i] = max(
					config.min_bits,
					min(config.max_bits, new_bits[i] + delta)
				)

			# Mutate neurons (Phase 2 only)
			if config.phase >= 2 and random.random() < config.neurons_mutation_rate:
				delta = random.choice([-config.neurons_mutation_step, config.neurons_mutation_step])
				new_neurons[i] = max(
					config.min_neurons,
					min(config.max_neurons, new_neurons[i] + delta)
				)

		return ClusterGenome(bits_per_cluster=new_bits, neurons_per_cluster=new_neurons)

	def crossover(
		self,
		other: ClusterGenome,
		crossover_rate: float = 0.5,
		rng: Optional[int] = None,
	) -> ClusterGenome:
		"""
		Create child genome by crossing over with another parent.

		Uses uniform crossover: each cluster's (bits, neurons) come from either parent.
		The entire cluster config is inherited together (not bits from one, neurons from other).

		Args:
			other: Second parent genome
			crossover_rate: Probability of taking from other (default 0.5)
			rng: Random seed

		Returns:
			New child genome
		"""
		if rng is not None:
			random.seed(rng)

		child_bits = []
		child_neurons = []

		for i in range(len(self.bits_per_cluster)):
			if random.random() < crossover_rate:
				child_bits.append(other.bits_per_cluster[i])
				child_neurons.append(other.neurons_per_cluster[i])
			else:
				child_bits.append(self.bits_per_cluster[i])
				child_neurons.append(self.neurons_per_cluster[i])

		return ClusterGenome(bits_per_cluster=child_bits, neurons_per_cluster=child_neurons)

	# =========================================================================
	# Properties and Utilities
	# =========================================================================

	@property
	def num_clusters(self) -> int:
		"""Number of clusters in this genome."""
		return len(self.bits_per_cluster)

	def total_memory_cells(self) -> int:
		"""Calculate total memory cells needed for this genome."""
		return sum(
			n * (2 ** b)
			for n, b in zip(self.neurons_per_cluster, self.bits_per_cluster)
		)

	def total_neurons(self) -> int:
		"""Total neurons across all clusters."""
		return sum(self.neurons_per_cluster)

	def clone(self) -> ClusterGenome:
		"""Create a deep copy of this genome."""
		return ClusterGenome(
			bits_per_cluster=self.bits_per_cluster.copy(),
			neurons_per_cluster=self.neurons_per_cluster.copy(),
		)

	def to_tensor(self) -> Tensor:
		"""Convert to tensor [num_clusters, 2] with (bits, neurons) per cluster."""
		data = list(zip(self.bits_per_cluster, self.neurons_per_cluster))
		return torch.tensor(data, dtype=torch.int32)

	def get_cluster_config(self, cluster_id: int) -> tuple:
		"""Get (neurons, bits) for a specific cluster."""
		return (self.neurons_per_cluster[cluster_id], self.bits_per_cluster[cluster_id])

	def stats(self) -> Dict:
		"""Get statistics about this genome."""
		bits = self.bits_per_cluster
		neurons = self.neurons_per_cluster

		return {
			"num_clusters": len(bits),
			# Bits stats
			"min_bits": min(bits),
			"max_bits": max(bits),
			"mean_bits": sum(bits) / len(bits),
			# Neurons stats (Phase 2)
			"min_neurons": min(neurons),
			"max_neurons": max(neurons),
			"mean_neurons": sum(neurons) / len(neurons),
			"total_neurons": sum(neurons),
			# Memory stats
			"total_memory_cells": self.total_memory_cells(),
			# Distributions
			"bits_distribution": {
				b: bits.count(b) for b in sorted(set(bits))
			},
			"neurons_distribution": {
				n: neurons.count(n) for n in sorted(set(neurons))
			},
		}

	def enforce_budget(self, config: AdaptiveClusterConfig) -> ClusterGenome:
		"""
		Ensure genome stays within memory budget by shrinking if needed.

		Args:
			config: Configuration with budget constraint

		Returns:
			New genome within budget (may be self if already within)
		"""
		if self.total_memory_cells() <= config.total_memory_budget:
			return self

		# Shrink largest clusters until under budget
		bits = self.bits_per_cluster.copy()
		while sum(2 ** b for b in bits) > config.total_memory_budget:
			# Find cluster with most bits
			max_idx = max(range(len(bits)), key=lambda i: bits[i])
			if bits[max_idx] > config.min_bits:
				bits[max_idx] -= 1
			else:
				break  # Can't shrink further

		return ClusterGenome(
			bits_per_cluster=bits,
			neurons_per_cluster=self.neurons_per_cluster.copy(),
		)

	def __repr__(self) -> str:
		stats = self.stats()
		return (
			f"ClusterGenome(clusters={stats['num_clusters']}, "
			f"bits=[{stats['min_bits']}-{stats['max_bits']}], "
			f"neurons=[{stats['min_neurons']}-{stats['max_neurons']}], "
			f"memory={stats['total_memory_cells']:,})"
		)


# =============================================================================
# Helper Functions (module-level, internal)
# =============================================================================

def _frequency_scaled_init(
	num_clusters: int,
	token_frequencies: List[int],
	config: AdaptiveClusterConfig,
	for_bits: bool = True,
) -> List[int]:
	"""
	Initialize bits or neurons scaled by token frequency.

	More frequent tokens get more bits/neurons (larger capacity).
	Rare tokens get fewer bits/neurons (small capacity sufficient).

	Uses log-scale mapping from frequency to value.
	"""
	if len(token_frequencies) != num_clusters:
		raise ValueError(
			f"token_frequencies length ({len(token_frequencies)}) != "
			f"num_clusters ({num_clusters})"
		)

	# Find frequency range (avoid log(0))
	freqs = [max(1, f) for f in token_frequencies]
	max_freq = max(freqs)
	min_freq = min(freqs)

	# Log-scale mapping: high freq -> high value, low freq -> low value
	log_max = math.log(max_freq)
	log_min = math.log(min_freq)
	log_range = log_max - log_min if log_max > log_min else 1.0

	if for_bits:
		min_val, max_val = config.min_bits, config.max_bits
	else:
		min_val, max_val = config.min_neurons, config.max_neurons

	val_range = max_val - min_val

	values = []
	for freq in freqs:
		# Normalize log frequency to [0, 1]
		log_freq = math.log(freq)
		normalized = (log_freq - log_min) / log_range

		# Map to value range
		val = min_val + int(normalized * val_range)
		val = max(min_val, min(max_val, val))
		values.append(val)

	return values


# =============================================================================
# Adaptive Cluster Optimizer (GA for architecture search)
# =============================================================================

@dataclass
class AdaptiveOptConfig:
	"""Configuration for adaptive cluster optimization."""

	# GA parameters
	population_size: int = 20
	generations: int = 50
	mutation_rate: float = 0.1
	crossover_rate: float = 0.7
	elitism: int = 2
	tournament_size: int = 3

	# Early stopping
	patience: int = 5
	min_improvement_pct: float = 0.1

	# Architecture constraints
	cluster_config: AdaptiveClusterConfig = None

	def __post_init__(self):
		if self.cluster_config is None:
			self.cluster_config = AdaptiveClusterConfig()


@dataclass
class AdaptiveOptResult:
	"""Result from adaptive cluster optimization."""

	best_genome: ClusterGenome
	initial_fitness: float
	final_fitness: float
	generations_run: int
	history: List[tuple]  # [(gen, best_fitness, avg_fitness)]
	early_stopped: bool


class RustParallelEvaluator:
	"""
	Rust-accelerated parallel genome evaluation using rayon.

	Evaluates multiple genomes concurrently in Rust threads.
	Much faster than Python multiprocessing - no process spawn or pickle overhead.

	Usage:
		evaluator = RustParallelEvaluator(config)
		fitness_list = evaluator.evaluate_batch(genomes)
	"""

	def __init__(self, config: 'EvaluatorConfig'):
		"""
		Initialize Rust parallel evaluator.

		Args:
			config: Evaluation configuration with pre-computed training data
		"""
		self.config = config
		self._prepared = False
		self._train_data = None
		self._eval_data = None

	def _prepare_data(self):
		"""Pre-encode training and evaluation data for Rust (vectorized)."""
		if self._prepared:
			return

		import numpy as np
		from collections import Counter
		from wnn.ram.core import bits_needed

		cfg = self.config
		bits_per_token = bits_needed(cfg.vocab_size)
		total_input_bits = cfg.context_size * bits_per_token

		# Build cluster map once (if needed)
		cluster_map = None
		if cfg.cluster_order is not None:
			cluster_map = np.zeros(cfg.vocab_size, dtype=np.int64)
			for idx, tid in enumerate(cfg.cluster_order):
				if tid < cfg.vocab_size:
					cluster_map[tid] = idx

		# Convert tokens to numpy array for vectorized ops
		train_tokens = np.array(cfg.train_tokens, dtype=np.int64)
		eval_tokens = np.array(cfg.eval_tokens, dtype=np.int64)

		# === TRAINING DATA (vectorized) ===
		n_train = len(train_tokens) - cfg.context_size

		# Build context windows: [n_train, context_size]
		train_contexts = np.lib.stride_tricks.sliding_window_view(
			train_tokens[:n_train + cfg.context_size - 1], cfg.context_size
		)[:n_train]

		# Encode contexts to bits using vectorized operations
		# Shape: [n_train, context_size, bits_per_token]
		bit_shifts = np.arange(bits_per_token - 1, -1, -1, dtype=np.int64)
		train_bits_3d = ((train_contexts[:, :, np.newaxis] >> bit_shifts) & 1).astype(np.uint8)
		train_input_bits = train_bits_3d.reshape(-1)  # Flatten to 1D

		# Targets
		train_targets_raw = train_tokens[cfg.context_size:cfg.context_size + n_train]
		if cluster_map is not None:
			train_targets = cluster_map[train_targets_raw]
		else:
			train_targets = train_targets_raw

		# === NEGATIVE SAMPLES (vectorized) ===
		counts = Counter(cfg.train_tokens)
		top_k_tokens = np.array([t for t, _ in counts.most_common(cfg.global_top_k)], dtype=np.int64)
		num_negatives = min(5, cfg.global_top_k)

		rng = np.random.RandomState(42)
		neg_indices = rng.randint(0, cfg.global_top_k, (n_train, num_negatives))
		neg_tokens = top_k_tokens[neg_indices]  # [n_train, num_negatives]

		if cluster_map is not None:
			train_negatives = cluster_map[neg_tokens].reshape(-1)
		else:
			train_negatives = neg_tokens.reshape(-1)

		# === EVALUATION DATA (vectorized) ===
		n_eval = len(eval_tokens) - cfg.context_size

		eval_contexts = np.lib.stride_tricks.sliding_window_view(
			eval_tokens[:n_eval + cfg.context_size - 1], cfg.context_size
		)[:n_eval]

		eval_bits_3d = ((eval_contexts[:, :, np.newaxis] >> bit_shifts) & 1).astype(np.uint8)
		eval_input_bits = eval_bits_3d.reshape(-1)

		eval_targets_raw = eval_tokens[cfg.context_size:cfg.context_size + n_eval]
		if cluster_map is not None:
			eval_targets = cluster_map[eval_targets_raw]
		else:
			eval_targets = eval_targets_raw

		self._train_data = {
			'input_bits': train_input_bits,
			'targets': train_targets.astype(np.int64),
			'negatives': train_negatives.astype(np.int64),
			'num_examples': n_train,
			'num_negatives': num_negatives,
		}
		self._eval_data = {
			'input_bits': eval_input_bits,
			'targets': eval_targets.astype(np.int64),
			'num_examples': n_eval,
		}
		self._total_input_bits = total_input_bits
		self._prepared = True

	def evaluate_batch(
		self,
		genomes: List[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
		batch_size: int = 1,  # Sequential genomes with inner parallelism
		generation: Optional[int] = None,  # Current generation for logging
		total_generations: Optional[int] = None,  # Total generations for logging
	) -> List[float]:
		"""
		Evaluate multiple genomes in parallel using Rust/rayon.

		Args:
			genomes: List of genomes to evaluate
			logger: Optional logging function
			batch_size: Number of genomes to evaluate in parallel (all 30 - training data shared)
			generation: Current generation number for logging (None = initial population)
			total_generations: Total number of generations for logging context

		Returns:
			List of cross-entropy values for each genome
		"""
		import ram_accelerator
		import time

		log = logger or (lambda x: None)

		# Prepare data on first call
		self._prepare_data()

		all_fitness = []
		total_genomes = len(genomes)
		start_time = time.time()

		# Generation prefix for logs
		if generation is not None:
			if total_generations is not None:
				gen_prefix = f"[Gen {generation + 1}/{total_generations}]"
			else:
				gen_prefix = f"[Gen {generation + 1}]"
		else:
			gen_prefix = "[Init]"

		# Process in batches to limit memory usage
		for batch_start in range(0, total_genomes, batch_size):
			batch_end = min(batch_start + batch_size, total_genomes)
			batch_genomes = genomes[batch_start:batch_end]

			# Flatten genome configurations for this batch
			genomes_bits_flat = []
			genomes_neurons_flat = []
			for g in batch_genomes:
				genomes_bits_flat.extend(g.bits_per_cluster)
				genomes_neurons_flat.extend(g.neurons_per_cluster)

			# Call Rust parallel evaluator for this batch
			batch_fitness = ram_accelerator.evaluate_genomes_parallel(
				genomes_bits_flat,
				genomes_neurons_flat,
				len(batch_genomes),
				self.config.vocab_size,
				self._train_data['input_bits'],
				self._train_data['targets'],
				self._train_data['negatives'],
				self._train_data['num_examples'],
				self._train_data['num_negatives'],
				self._eval_data['input_bits'],
				self._eval_data['targets'],
				self._eval_data['num_examples'],
				self._total_input_bits,
				self.config.empty_value,
			)

			all_fitness.extend(batch_fitness)

			elapsed = time.time() - start_time
			if batch_size == 1:
				# Per-genome timing
				log(f"{gen_prefix} Genome {batch_end}/{total_genomes}: CE={batch_fitness[0]:.4f} in {elapsed:.1f}s")
			else:
				log(f"{gen_prefix} Batch {batch_start//batch_size + 1}/{(total_genomes + batch_size - 1)//batch_size}: "
					f"{batch_end}/{total_genomes} genomes in {elapsed:.1f}s")

		return all_fitness

	def evaluate_single(self, genome: ClusterGenome) -> float:
		"""Evaluate a single genome."""
		return self.evaluate_batch([genome])[0]


class AdaptiveClusterOptimizer:
	"""
	Evolves cluster architectures (bits-per-cluster, neurons-per-cluster) using GA.

	Fitness is evaluated by training a model with the genome's architecture
	and measuring cross-entropy on held-out data.

	Example:
		optimizer = AdaptiveClusterOptimizer(
			config=opt_config,
			evaluate_fn=lambda genome: train_and_eval(genome),
			num_clusters=50257,
			token_frequencies=frequencies,
		)
		result = optimizer.optimize()
	"""

	def __init__(
		self,
		config: AdaptiveOptConfig,
		evaluate_fn: Callable[[ClusterGenome], float],
		num_clusters: int,
		token_frequencies: Optional[List[int]] = None,
		seed: int = 42,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
	):
		"""
		Initialize the optimizer.

		Args:
			config: Optimization configuration
			evaluate_fn: Function that takes a ClusterGenome and returns fitness
			            (lower is better, e.g., cross-entropy)
			num_clusters: Number of clusters (vocabulary size)
			token_frequencies: Token occurrence counts for FREQUENCY_SCALED init
			seed: Random seed
			logger: Optional logging function
			batch_evaluator: Optional Rust parallel evaluator for batch evaluation.
			                 If provided, evaluates all genomes in parallel using Rust/rayon.
			                 Falls back to evaluate_fn if not provided.
		"""
		self.config = config
		self.evaluate_fn = evaluate_fn
		self.num_clusters = num_clusters
		self.token_frequencies = token_frequencies
		self.seed = seed
		self._log = logger or print
		self._rng = None
		self._batch_evaluator = batch_evaluator

	def _ensure_rng(self):
		if self._rng is None:
			self._rng = random.Random(self.seed)

	def _evaluate_population(
		self,
		population: List[ClusterGenome],
		tracker: Optional[ProgressTracker] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
	) -> List[float]:
		"""
		Evaluate fitness of all genomes in the population.

		Uses Rust parallel evaluator if available, otherwise falls back to sequential.

		Args:
			population: List of genomes to evaluate
			tracker: Optional progress tracker for logging
			generation: Current generation number (None = initial population)
			total_generations: Total number of generations for logging

		Returns:
			List of fitness values (cross-entropy, lower is better)
		"""
		if self._batch_evaluator is not None:
			# Rust parallel evaluation (all genomes at once)
			fitness = self._batch_evaluator.evaluate_batch(
				population, logger=self._log, generation=generation,
				total_generations=total_generations
			)
		else:
			# Sequential Python evaluation with progress tracking
			fitness = []
			for i, genome in enumerate(population):
				fit = self.evaluate_fn(genome)
				fitness.append(fit)
				if tracker:
					tracker.tick_individual(fit, i, len(population))
		return fitness

	def optimize(
		self,
		init_strategy: GenomeInitStrategy = GenomeInitStrategy.FREQUENCY_SCALED,
	) -> AdaptiveOptResult:
		"""
		Run the adaptive architecture optimization.

		Args:
			init_strategy: How to initialize the population

		Returns:
			AdaptiveOptResult with best genome and statistics
		"""
		self._ensure_rng()
		cfg = self.config
		cc = cfg.cluster_config

		self._log(f"[AdaptiveGA] Starting optimization")
		self._log(f"  Clusters: {self.num_clusters}")
		self._log(f"  Population: {cfg.population_size}")
		self._log(f"  Generations: {cfg.generations}")
		self._log(f"  Init strategy: {init_strategy.name}")
		self._log(f"  Bits range: [{cc.min_bits}, {cc.max_bits}]")

		# Initialize population
		population = self._init_population(init_strategy)

		# Create progress tracker
		tracker = ProgressTracker(
			logger=self._log,
			minimize=True,  # Lower CE is better
			prefix="[AdaptiveGA]",
			total_generations=cfg.generations,
		)

		# Evaluate initial population with progress logging
		self._log(f"[AdaptiveGA] Evaluating initial population ({cfg.population_size} genomes)...")
		import time as _time
		_start = _time.time()
		fitness = self._evaluate_population(
			population, tracker if self._batch_evaluator is None else None, generation=None
		)
		_elapsed = _time.time() - _start
		self._log(f"[AdaptiveGA] Initial population complete in {_elapsed:.1f}s")

		# Record initial population stats
		tracker.tick(fitness, generation=-1, log=False)  # Gen -1 = initial
		best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
		best_genome = population[best_idx].clone()
		best_fitness = fitness[best_idx]
		initial_fitness = best_fitness

		history = [(0, best_fitness, sum(fitness) / len(fitness))]
		self._log(f"[AdaptiveGA] Initial population complete: best={best_fitness:.4f}, avg={sum(fitness)/len(fitness):.4f}")

		# Early stopping tracking
		patience_counter = 0
		prev_best = best_fitness

		for gen in range(cfg.generations):
			# Selection + Crossover + Mutation
			new_population = []

			# Elitism: keep best individuals
			sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
			for i in range(cfg.elitism):
				new_population.append(population[sorted_indices[i]].clone())

			# Fill rest with offspring
			while len(new_population) < cfg.population_size:
				# Tournament selection
				parent1 = self._tournament_select(population, fitness)
				parent2 = self._tournament_select(population, fitness)

				# Crossover
				if self._rng.random() < cfg.crossover_rate:
					child = parent1.crossover(parent2, rng=self._rng.randint(0, 2**31))
				else:
					child = parent1.clone()

				# Mutation
				child = child.mutate(cc, rng=self._rng.randint(0, 2**31))

				# Enforce memory budget
				child = child.enforce_budget(cc)

				new_population.append(child)

			population = new_population

			# Evaluate new population
			self._log(f"[AdaptiveGA] Generation {gen + 1}/{cfg.generations}: Evaluating {cfg.population_size} genomes...")
			fitness = self._evaluate_population(population, generation=gen)
			gen_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])

			if fitness[gen_best_idx] < best_fitness:
				best_genome = population[gen_best_idx].clone()
				best_fitness = fitness[gen_best_idx]

			avg_fitness = sum(fitness) / len(fitness)
			history.append((gen + 1, best_fitness, avg_fitness))

			# Log progress using tracker
			progress = tracker.tick(fitness, generation=gen)
			genome_stats = best_genome.stats()
			self._log(
				f"  bits=[{genome_stats['min_bits']}-{genome_stats['max_bits']}], "
				f"neurons=[{genome_stats['min_neurons']}-{genome_stats['max_neurons']}]"
			)

			# Early stopping check
			improvement_pct = (prev_best - best_fitness) / prev_best * 100 if prev_best > 0 else 0
			if improvement_pct >= cfg.min_improvement_pct:
				patience_counter = 0
				prev_best = best_fitness
			else:
				patience_counter += 1

			if patience_counter >= cfg.patience:
				self._log(f"[AdaptiveGA] Early stop at gen {gen + 1}: no improvement for {cfg.patience} gens")
				return AdaptiveOptResult(
					best_genome=best_genome,
					initial_fitness=initial_fitness,
					final_fitness=best_fitness,
					generations_run=gen + 1,
					history=history,
					early_stopped=True,
				)

		self._log(f"[AdaptiveGA] Completed {cfg.generations} generations")
		tracker.log_summary()

		return AdaptiveOptResult(
			best_genome=best_genome,
			initial_fitness=initial_fitness,
			final_fitness=best_fitness,
			generations_run=cfg.generations,
			history=history,
			early_stopped=False,
		)

	def _init_population(self, strategy: GenomeInitStrategy) -> List[ClusterGenome]:
		"""Initialize the population with some diversity."""
		population = []
		cc = self.config.cluster_config

		# First individual uses the specified strategy
		population.append(ClusterGenome.initialize(
			self.num_clusters, strategy, cc,
			token_frequencies=self.token_frequencies,
			rng=self.seed,
		))

		# Rest are mutations of the first, or frequency-aware random
		for i in range(1, self.config.population_size):
			if i < self.config.population_size // 2:
				# Mutate from first
				genome = population[0].mutate(cc, rng=self.seed + i)
			else:
				# Frequency-aware random (caps bits for rare tokens to avoid sparse memory slowdown)
				genome = ClusterGenome.initialize(
					self.num_clusters,
					GenomeInitStrategy.RANDOM_UNIFORM,
					cc,
					token_frequencies=self.token_frequencies,  # Pass frequencies for smart capping
					rng=self.seed + i,
				)
			population.append(genome)

		return population

	def _tournament_select(
		self,
		population: List[ClusterGenome],
		fitness: List[float],
	) -> ClusterGenome:
		"""Select individual via tournament selection."""
		candidates = self._rng.sample(range(len(population)), self.config.tournament_size)
		winner = min(candidates, key=lambda i: fitness[i])
		return population[winner]


# =============================================================================
# Genome Evaluation Wrapper
# =============================================================================

@dataclass
class EvaluatorConfig:
	"""Configuration for genome evaluation."""

	# Data
	train_tokens: List[int] = None
	eval_tokens: List[int] = None
	vocab_size: int = 50257
	context_size: int = 4

	# Training
	batch_size: int = 500
	global_top_k: int = 100
	empty_value: float = 0.0

	# Evaluation
	eval_batch_size: int = 1000

	# Token ordering (for cluster assignment)
	cluster_order: Optional[List[int]] = None  # sorted by frequency

	# Random seed for connectivity (None = truly random, no seeding)
	# NOTE: Architecture optimization should NOT depend on specific connectivity
	# patterns, so we default to None for unbiased evaluation.
	rng: Optional[int] = None


class AdaptiveRAMLMWrapper:
	"""
	Lightweight RAMLM-like wrapper using AdaptiveClusteredRAM.

	This class provides train/evaluate functionality similar to RAMLM
	but uses AdaptiveClusteredRAM for per-cluster architecture.

	Used for genome fitness evaluation during architecture search.
	"""

	def __init__(
		self,
		genome: ClusterGenome,
		config: EvaluatorConfig,
	):
		"""
		Initialize wrapper from genome.

		Args:
			genome: ClusterGenome defining per-cluster architecture
			config: Evaluation configuration
		"""
		from wnn.ram.core import AdaptiveClusteredRAM, bits_needed
		from torch import arange, long

		self.genome = genome
		self.config = config
		self.vocab_size = config.vocab_size
		self.context_size = config.context_size
		self.bits_per_token = bits_needed(config.vocab_size)
		self.total_input_bits = config.context_size * self.bits_per_token

		# Create the adaptive layer
		self.layer = AdaptiveClusteredRAM.from_genome(
			genome=genome,
			total_input_bits=self.total_input_bits,
			empty_value=config.empty_value,
			rng=config.rng,
		)

		# Cluster order for training (maps token IDs to cluster IDs)
		self._cluster_order = config.cluster_order
		if self._cluster_order is not None:
			# Build reverse mapping: token_id -> logical_cluster_idx
			self._token_to_cluster = {tid: idx for idx, tid in enumerate(self._cluster_order)}
		else:
			self._token_to_cluster = None

		# Bit positions for encoding
		self._bit_positions = arange(self.bits_per_token - 1, -1, -1, dtype=long)

	def _encode_tokens(self, tokens: List[int]) -> Tensor:
		"""Encode tokens to binary input bits."""
		from torch import tensor, zeros, bool as torch_bool

		n = len(tokens)
		bits = zeros(n, self.bits_per_token, dtype=torch_bool)
		tokens_t = tensor(tokens, dtype=self._bit_positions.dtype)

		for i in range(self.bits_per_token):
			bits[:, i] = ((tokens_t >> self._bit_positions[i]) & 1).bool()

		return bits

	def _encode_context(self, context_tokens: List[int]) -> Tensor:
		"""Encode context tokens to flat input bits."""
		bits = self._encode_tokens(context_tokens)
		return bits.flatten()

	def train_epoch(
		self,
		tokens: List[int],
		global_top_k: int = 100,
		batch_size: int = 500,
		verbose: bool = False,
	) -> Dict:
		"""
		Train on token sequence.

		Args:
			tokens: Training token sequence
			global_top_k: Number of top tokens to use as negatives
			batch_size: Batch size for training
			verbose: Print progress

		Returns:
			Dict with training statistics
		"""
		from collections import Counter
		import time

		from torch import long, randint, stack, tensor, zeros

		start = time.time()

		# Compute global top-k tokens
		counts = Counter(tokens)
		top_k_tokens = [t for t, _ in counts.most_common(global_top_k)]

		# Prepare examples
		n_examples = len(tokens) - self.context_size
		contexts = []
		targets = []

		for i in range(n_examples):
			context = tokens[i:i + self.context_size]
			target = tokens[i + self.context_size]
			contexts.append(context)
			targets.append(target)

		# Encode all contexts
		all_input_bits = []
		for ctx in contexts:
			all_input_bits.append(self._encode_context(ctx))
		input_bits = stack(all_input_bits)  # [n_examples, total_input_bits]

		# Convert targets to cluster indices
		if self._token_to_cluster is not None:
			true_clusters = tensor([self._token_to_cluster.get(t, t) for t in targets], dtype=long)
		else:
			true_clusters = tensor(targets, dtype=long)

		# Generate negative samples (global top-k)
		num_negatives = min(5, global_top_k)
		false_clusters = zeros(n_examples, num_negatives, dtype=long)
		top_k_tensor = tensor(top_k_tokens, dtype=long)

		for i in range(n_examples):
			# Sample from top-k, excluding true target
			neg_indices = randint(0, global_top_k, (num_negatives,))
			neg_tokens = top_k_tensor[neg_indices]
			if self._token_to_cluster is not None:
				neg_clusters = tensor([self._token_to_cluster.get(int(t), int(t)) for t in neg_tokens], dtype=long)
			else:
				neg_clusters = neg_tokens
			false_clusters[i] = neg_clusters

		# Train in batches
		modified = 0
		for start_idx in range(0, n_examples, batch_size):
			end_idx = min(start_idx + batch_size, n_examples)
			batch_input = input_bits[start_idx:end_idx]
			batch_true = true_clusters[start_idx:end_idx]
			batch_false = false_clusters[start_idx:end_idx]

			modified += self.layer.train_batch(
				batch_input, batch_true, batch_false, allow_override=False
			)

			if verbose and (start_idx // batch_size) % 10 == 0:
				pct = start_idx / n_examples * 100
				print(f"  Training: {pct:.1f}%")

		elapsed = time.time() - start

		return {
			'modified': modified,
			'examples': n_examples,
			'time': elapsed,
		}

	def evaluate(
		self,
		tokens: List[int],
		batch_size: int = 1000,
		verbose: bool = False,
	) -> Dict:
		"""
		Evaluate on token sequence.

		Args:
			tokens: Evaluation token sequence
			batch_size: Batch size for evaluation
			verbose: Print progress

		Returns:
			Dict with cross_entropy, perplexity, accuracy
		"""
		import time

		from torch import clamp, log, long, stack, tensor
		from torch.nn.functional import softmax

		start = time.time()

		# Prepare examples
		n_examples = len(tokens) - self.context_size
		contexts = []
		targets = []

		for i in range(n_examples):
			context = tokens[i:i + self.context_size]
			target = tokens[i + self.context_size]
			contexts.append(context)
			targets.append(target)

		# Encode all contexts
		all_input_bits = []
		for ctx in contexts:
			all_input_bits.append(self._encode_context(ctx))
		input_bits = stack(all_input_bits)

		# Convert targets to cluster indices for accuracy
		if self._token_to_cluster is not None:
			target_clusters = tensor([self._token_to_cluster.get(t, t) for t in targets], dtype=long)
		else:
			target_clusters = tensor(targets, dtype=long)

		# Evaluate in batches
		total_ce = 0.0
		total_correct = 0

		for start_idx in range(0, n_examples, batch_size):
			end_idx = min(start_idx + batch_size, n_examples)
			batch_input = input_bits[start_idx:end_idx]
			batch_targets = target_clusters[start_idx:end_idx]

			# Forward pass
			probs = self.layer.forward(batch_input)  # [batch, vocab_size]

			# Softmax over vocabulary
			probs_softmax = softmax(probs, dim=-1)

			# Cross-entropy: -log(p[target])
			target_probs = probs_softmax.gather(1, batch_targets.unsqueeze(1)).squeeze(1)
			target_probs = clamp(target_probs, min=1e-10)
			batch_ce = -log(target_probs).sum().item()
			total_ce += batch_ce

			# Accuracy
			predictions = probs_softmax.argmax(dim=-1)
			total_correct += (predictions == batch_targets).sum().item()

			if verbose and (start_idx // batch_size) % 10 == 0:
				pct = start_idx / n_examples * 100
				print(f"  Evaluating: {pct:.1f}%")

		elapsed = time.time() - start

		avg_ce = total_ce / n_examples
		perplexity = 2 ** (avg_ce / 0.693147)  # Convert to base-2 perplexity
		accuracy = total_correct / n_examples

		return {
			'cross_entropy': avg_ce,
			'perplexity': perplexity,
			'accuracy': accuracy,
			'examples': n_examples,
			'time': elapsed,
		}


def create_genome_evaluator(
	config: EvaluatorConfig,
	verbose: bool = False,
) -> Callable[[ClusterGenome], float]:
	"""
	Create an evaluation function for genome fitness.

	The returned function:
	1. Takes a ClusterGenome
	2. Builds an AdaptiveRAMLMWrapper
	3. Trains on config.train_tokens
	4. Evaluates on config.eval_tokens
	5. Returns cross-entropy (lower is better)

	Args:
		config: Evaluation configuration with tokens and parameters
		verbose: Print progress during train/eval

	Returns:
		Function mapping ClusterGenome -> float (cross-entropy)

	Example:
		config = EvaluatorConfig(
			train_tokens=train_tokens[:100000],
			eval_tokens=val_tokens[:10000],
			vocab_size=50257,
			context_size=4,
		)
		evaluate_fn = create_genome_evaluator(config)

		# Use with optimizer
		optimizer = AdaptiveClusterOptimizer(
			config=opt_config,
			evaluate_fn=evaluate_fn,
			num_clusters=50257,
		)
		result = optimizer.optimize()
	"""
	def evaluate_genome(genome: ClusterGenome) -> float:
		"""Evaluate a genome and return fitness (cross-entropy)."""
		# Build wrapper from genome
		wrapper = AdaptiveRAMLMWrapper(genome, config)

		# Train
		wrapper.train_epoch(
			config.train_tokens,
			global_top_k=config.global_top_k,
			batch_size=config.batch_size,
			verbose=verbose,
		)

		# Evaluate
		stats = wrapper.evaluate(
			config.eval_tokens,
			batch_size=config.eval_batch_size,
			verbose=verbose,
		)

		return stats['cross_entropy']

	return evaluate_genome


# =============================================================================
# High-Level API
# =============================================================================

# =============================================================================
# Architecture Tabu Search (Phase 1b)
# =============================================================================

@dataclass
class ArchitectureTSConfig:
	"""Configuration for architecture Tabu Search."""

	iterations: int = 100
	"""Number of TS iterations"""

	neighbors_per_iter: int = 20
	"""Number of neighbors to generate per iteration"""

	tabu_size: int = 10
	"""Size of tabu list (moves to avoid)"""

	# Neighborhood generation
	bits_change_prob: float = 0.7
	"""Probability of changing bits (vs neurons) per cluster mutation"""

	clusters_per_neighbor: int = 50
	"""Number of clusters to mutate per neighbor (sparse mutations)"""

	step_size: int = 1
	"""How much to change bits/neurons (+/- this value)"""

	# Early stopping
	patience: int = 10
	"""Iterations without improvement before stopping"""

	min_improvement_pct: float = 0.01
	"""Minimum improvement % to reset patience"""


@dataclass
class ArchitectureTSResult:
	"""Result from architecture Tabu Search."""

	best_genome: ClusterGenome
	initial_fitness: float
	final_fitness: float
	iterations_run: int
	history: List[tuple]  # [(iter, best_fitness)]
	early_stopped: bool


class ArchitectureTabuSearch:
	"""
	Tabu Search optimizer for architecture refinement (bits + neurons).

	Unlike GA which explores globally, TS performs local search from a starting
	genome. Good for refining a GA-discovered solution.

	Key features:
	- Sparse mutations: only changes a few clusters per neighbor
	- Tabu list: avoids cycling back to recent solutions
	- Always moves to best neighbor (TS characteristic)
	"""

	def __init__(
		self,
		config: ArchitectureTSConfig,
		evaluate_fn: Callable[[ClusterGenome], float],
		cluster_config: AdaptiveClusterConfig,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
		seed: int = 42,
		logger: Optional[Callable[[str], None]] = None,
	):
		self.config = config
		self.evaluate_fn = evaluate_fn
		self.cluster_config = cluster_config
		self._batch_evaluator = batch_evaluator
		self.seed = seed
		self._log = logger or print
		self._rng = random.Random(seed)

	def _generate_neighbor(
		self,
		genome: ClusterGenome,
	) -> tuple:
		"""
		Generate a neighbor by mutating a few clusters.

		Returns:
			(neighbor_genome, move_descriptor)
		"""
		cc = self.cluster_config
		cfg = self.config

		new_bits = genome.bits_per_cluster.copy()
		new_neurons = genome.neurons_per_cluster.copy()

		# Track which clusters were changed (for tabu)
		changes = []

		# Select random clusters to mutate
		clusters_to_mutate = self._rng.sample(
			range(genome.num_clusters),
			min(cfg.clusters_per_neighbor, genome.num_clusters)
		)

		for cluster_idx in clusters_to_mutate:
			# Decide what to change
			if cc.phase >= 2 and self._rng.random() > cfg.bits_change_prob:
				# Change neurons
				old_val = new_neurons[cluster_idx]
				delta = self._rng.choice([-cfg.step_size, cfg.step_size])
				new_val = max(cc.min_neurons, min(cc.max_neurons, old_val + delta))
				if new_val != old_val:
					new_neurons[cluster_idx] = new_val
					changes.append(('N', cluster_idx, old_val, new_val))
			else:
				# Change bits
				old_val = new_bits[cluster_idx]
				delta = self._rng.choice([-cfg.step_size, cfg.step_size])
				new_val = max(cc.min_bits, min(cc.max_bits, old_val + delta))
				if new_val != old_val:
					new_bits[cluster_idx] = new_val
					changes.append(('B', cluster_idx, old_val, new_val))

		neighbor = ClusterGenome(
			bits_per_cluster=new_bits,
			neurons_per_cluster=new_neurons,
		)

		# Move descriptor is a frozenset of changes for tabu matching
		move_desc = frozenset(changes)
		return neighbor, move_desc

	def _is_tabu(self, move_desc: frozenset, tabu_list: list) -> bool:
		"""Check if a move would reverse a recent move."""
		for tabu_move in tabu_list:
			# Check if any change would reverse a tabu change
			for change in move_desc:
				change_type, cluster_idx, old_val, new_val = change
				# Look for reversal: same cluster, same type, going back
				for tabu_change in tabu_move:
					if (tabu_change[0] == change_type and
						tabu_change[1] == cluster_idx and
						tabu_change[3] == old_val):  # Going back to where we were
						return True
		return False

	def optimize(
		self,
		initial_genome: ClusterGenome,
		initial_fitness: Optional[float] = None,
	) -> ArchitectureTSResult:
		"""
		Run Tabu Search from an initial genome.

		Args:
			initial_genome: Starting genome (e.g., from GA)
			initial_fitness: Optional pre-computed fitness (avoids re-evaluation)

		Returns:
			ArchitectureTSResult with refined genome
		"""
		cfg = self.config

		self._log(f"[ArchTS] Starting Tabu Search")
		self._log(f"  Iterations: {cfg.iterations}")
		self._log(f"  Neighbors/iter: {cfg.neighbors_per_iter}")
		self._log(f"  Tabu size: {cfg.tabu_size}")
		self._log(f"  Clusters/neighbor: {cfg.clusters_per_neighbor}")

		# Initialize
		current = initial_genome.clone()
		if initial_fitness is not None:
			current_fitness = initial_fitness
		else:
			current_fitness = self._evaluate(current)

		best = current.clone()
		best_fitness = current_fitness
		start_fitness = current_fitness

		tabu_list = []  # List of recent move descriptors
		history = [(0, best_fitness)]

		# Progress tracker for consistent logging
		tracker = ProgressTracker(
			logger=self._log,
			minimize=True,
			prefix="[ArchTS]",
			total_generations=cfg.iterations,
		)

		# Early stopping
		patience_counter = 0
		prev_best = best_fitness

		self._log(f"[ArchTS] Initial fitness: {current_fitness:.4f}")

		for iteration in range(cfg.iterations):
			# Generate neighbors
			neighbors = []
			for _ in range(cfg.neighbors_per_iter):
				neighbor, move_desc = self._generate_neighbor(current)
				if not self._is_tabu(move_desc, tabu_list) and len(move_desc) > 0:
					neighbors.append((neighbor, move_desc))

			if not neighbors:
				self._log(f"[ArchTS] No valid neighbors at iter {iteration + 1}")
				continue

			# Evaluate neighbors
			if self._batch_evaluator is not None:
				genomes = [n for n, _ in neighbors]
				fitness_list = self._batch_evaluator.evaluate_batch(
					genomes, logger=self._log
				)
				evaluated = [(n, f, m) for (n, m), f in zip(neighbors, fitness_list)]
			else:
				evaluated = [(n, self.evaluate_fn(n), m) for n, m in neighbors]

			# Log progress using tracker (all neighbor fitness values)
			fitness_values = [f for _, f, _ in evaluated]
			tracker.tick(fitness_values, generation=iteration)

			# Select best neighbor
			evaluated.sort(key=lambda x: x[1])
			best_neighbor, best_neighbor_fitness, best_move = evaluated[0]

			# Move to best neighbor (always, TS characteristic)
			current = best_neighbor
			current_fitness = best_neighbor_fitness

			# Add move to tabu list
			if len(tabu_list) >= cfg.tabu_size:
				tabu_list.pop(0)
			tabu_list.append(best_move)

			# Update global best
			if current_fitness < best_fitness:
				best = current.clone()
				best_fitness = current_fitness

			history.append((iteration + 1, best_fitness))

			# Early stopping check
			improvement_pct = (prev_best - best_fitness) / prev_best * 100 if prev_best > 0 else 0
			if improvement_pct >= cfg.min_improvement_pct:
				patience_counter = 0
				prev_best = best_fitness
			else:
				patience_counter += 1

			if patience_counter >= cfg.patience:
				self._log(f"[ArchTS] Early stop at iter {iteration + 1}: "
						  f"no improvement for {cfg.patience} iters")
				return ArchitectureTSResult(
					best_genome=best,
					initial_fitness=start_fitness,
					final_fitness=best_fitness,
					iterations_run=iteration + 1,
					history=history,
					early_stopped=True,
				)

		self._log(f"[ArchTS] Completed {cfg.iterations} iterations")
		improvement = (1 - best_fitness / start_fitness) * 100 if start_fitness > 0 else 0
		self._log(f"[ArchTS] Final: {best_fitness:.4f} ({improvement:.2f}% improvement)")

		return ArchitectureTSResult(
			best_genome=best,
			initial_fitness=start_fitness,
			final_fitness=best_fitness,
			iterations_run=cfg.iterations,
			history=history,
			early_stopped=False,
		)

	def _evaluate(self, genome: ClusterGenome) -> float:
		"""Evaluate a single genome."""
		if self._batch_evaluator is not None:
			return self._batch_evaluator.evaluate_batch([genome])[0]
		return self.evaluate_fn(genome)


def run_architecture_tabu_search(
	initial_genome: ClusterGenome,
	initial_fitness: float,
	train_tokens: List[int],
	eval_tokens: List[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	cluster_order: Optional[List[int]] = None,
	# TS parameters
	iterations: int = 100,
	neighbors_per_iter: int = 20,
	patience: int = 10,
	# Architecture bounds
	min_bits: int = 4,
	max_bits: int = 20,
	min_neurons: int = 1,
	max_neurons: int = 15,
	phase: int = 2,
	# Other
	empty_value: float = 0.0,
	seed: int = 42,
	logger: Optional[Callable[[str], None]] = None,
) -> ArchitectureTSResult:
	"""
	Run Tabu Search to refine architecture from a GA solution.

	Phase 1b: Takes the best genome from GA and applies local search
	to potentially find better nearby solutions.

	Args:
		initial_genome: Best genome from Phase 1a (GA)
		initial_fitness: Fitness of initial genome
		train_tokens: Training data
		eval_tokens: Evaluation data
		vocab_size: Vocabulary size
		context_size: Context window
		cluster_order: Token ordering by frequency
		iterations: Number of TS iterations
		neighbors_per_iter: Neighbors to evaluate per iteration
		patience: Early stop patience
		min_bits, max_bits: Bits bounds
		min_neurons, max_neurons: Neurons bounds
		phase: Optimization phase
		empty_value: EMPTY cell value
		seed: Random seed
		logger: Logging function

	Returns:
		ArchitectureTSResult with refined genome
	"""
	log = logger or print

	log()
	log("=" * 60)
	log("  Phase 1b: Architecture Tabu Search (Refinement)")
	log("=" * 60)
	log(f"  Initial fitness: {initial_fitness:.4f}")
	log(f"  Iterations: {iterations}")
	log(f"  Neighbors/iter: {neighbors_per_iter}")
	log()

	# Create evaluator config (no rng = truly random connectivity)
	eval_config = EvaluatorConfig(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		batch_size=500,
		global_top_k=100,
		empty_value=empty_value,
		eval_batch_size=1000,
		cluster_order=cluster_order,
		# rng=None by default: architecture search should not depend on specific connectivity
	)

	# Create evaluation function
	evaluate_fn = create_genome_evaluator(eval_config, verbose=False)

	# Create Rust parallel evaluator
	batch_evaluator = None
	try:
		batch_evaluator = RustParallelEvaluator(eval_config)
		log("[ArchTS] Using Rust parallel evaluator")
	except Exception as e:
		log(f"[ArchTS] Using Python evaluator ({e})")

	# Create configs
	cluster_config = AdaptiveClusterConfig(
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
	)

	ts_config = ArchitectureTSConfig(
		iterations=iterations,
		neighbors_per_iter=neighbors_per_iter,
		patience=patience,
	)

	# Create optimizer
	optimizer = ArchitectureTabuSearch(
		config=ts_config,
		evaluate_fn=evaluate_fn,
		cluster_config=cluster_config,
		batch_evaluator=batch_evaluator,
		seed=seed,
		logger=log,
	)

	# Run optimization
	result = optimizer.optimize(initial_genome, initial_fitness)

	# Log results
	log()
	log("=" * 60)
	log("  Phase 1b Complete")
	log("=" * 60)
	stats = result.best_genome.stats()
	log(f"  Initial CE: {result.initial_fitness:.4f}")
	log(f"  Final CE: {result.final_fitness:.4f}")
	improvement = (1 - result.final_fitness / result.initial_fitness) * 100
	log(f"  Improvement: {improvement:.2f}%")
	log(f"  Iterations: {result.iterations_run}")
	log()
	log("  Refined genome:")
	log(f"    Bits: [{stats['min_bits']}, {stats['max_bits']}], mean: {stats['mean_bits']:.1f}")
	log(f"    Neurons: [{stats['min_neurons']}, {stats['max_neurons']}], mean: {stats['mean_neurons']:.1f}")

	return result


def run_architecture_search(
	train_tokens: List[int],
	eval_tokens: List[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	token_frequencies: Optional[List[int]] = None,
	cluster_order: Optional[List[int]] = None,
	# GA parameters
	population_size: int = 10,
	generations: int = 20,
	patience: int = 5,
	# Architecture bounds
	min_bits: int = 4,
	max_bits: int = 20,
	min_neurons: int = 1,
	max_neurons: int = 15,
	phase: int = 2,
	# Other
	init_strategy: GenomeInitStrategy = GenomeInitStrategy.FREQUENCY_SCALED,
	empty_value: float = 0.0,
	seed: int = 42,
	logger: Optional[Callable[[str], None]] = None,
) -> AdaptiveOptResult:
	"""
	Run complete architecture search for adaptive cluster configuration.

	This is the main entry point for discovering optimal per-cluster
	architectures using genetic algorithm optimization.

	Args:
		train_tokens: Training token sequence
		eval_tokens: Evaluation token sequence
		vocab_size: Vocabulary size
		context_size: Context window size
		token_frequencies: Token occurrence counts (for FREQUENCY_SCALED init)
		cluster_order: Token IDs sorted by frequency (for tier assignment)

		population_size: GA population size
		generations: Maximum generations
		patience: Early stop patience

		min_bits, max_bits: Bits per neuron bounds
		min_neurons, max_neurons: Neurons per cluster bounds
		phase: 1 = bits only, 2 = bits + neurons

		init_strategy: How to initialize genomes
		empty_value: Value for EMPTY cells (0.0 recommended)
		seed: Random seed
		logger: Logging function

	Returns:
		AdaptiveOptResult with best genome and optimization history

	Example:
		from collections import Counter

		# Compute token frequencies
		counts = Counter(train_tokens)
		token_frequencies = [counts.get(i, 0) for i in range(vocab_size)]
		cluster_order = sorted(range(vocab_size), key=lambda t: -counts.get(t, 0))

		result = run_architecture_search(
			train_tokens=train_tokens[:500000],
			eval_tokens=val_tokens[:50000],
			vocab_size=50257,
			token_frequencies=token_frequencies,
			cluster_order=cluster_order,
			population_size=10,
			generations=50,
		)

		print(f"Best cross-entropy: {result.final_fitness:.4f}")
		print(f"Improvement: {(1 - result.final_fitness/result.initial_fitness)*100:.1f}%")
	"""
	log = logger or print

	log("=" * 60)
	log("  Adaptive Architecture Search")
	log("=" * 60)
	log(f"  Train tokens: {len(train_tokens):,}")
	log(f"  Eval tokens: {len(eval_tokens):,}")
	log(f"  Vocab size: {vocab_size:,}")
	log(f"  Context size: {context_size}")
	log(f"  Population: {population_size}")
	log(f"  Generations: {generations}")
	log(f"  Phase: {phase} ({'bits only' if phase == 1 else 'bits + neurons'})")
	log(f"  Init strategy: {init_strategy.name}")
	log()

	# Create evaluator config (no rng = truly random connectivity)
	eval_config = EvaluatorConfig(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		batch_size=500,
		global_top_k=100,
		empty_value=empty_value,
		eval_batch_size=1000,
		cluster_order=cluster_order,
		# rng=None by default: architecture search should not depend on specific connectivity
	)

	# Create evaluation function (fallback for single genome evaluation)
	evaluate_fn = create_genome_evaluator(eval_config, verbose=False)

	# Create Rust parallel evaluator for batch evaluation (hybrid dense/sparse memory)
	batch_evaluator = None
	try:
		batch_evaluator = RustParallelEvaluator(eval_config)
		log("[AdaptiveGA] Using Rust parallel evaluator (hybrid dense/sparse)")
	except ImportError:
		log("[AdaptiveGA] Rust accelerator not available, using Python sequential")
	except Exception as e:
		log(f"[AdaptiveGA] Warning: Rust evaluator init failed ({e}), using Python")

	# Create optimizer config
	cluster_config = AdaptiveClusterConfig(
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
	)

	opt_config = AdaptiveOptConfig(
		population_size=population_size,
		generations=generations,
		patience=patience,
		cluster_config=cluster_config,
	)

	# Create optimizer
	optimizer = AdaptiveClusterOptimizer(
		config=opt_config,
		evaluate_fn=evaluate_fn,
		num_clusters=vocab_size,
		token_frequencies=token_frequencies,
		seed=seed,
		logger=log,
		batch_evaluator=batch_evaluator,
	)

	# Run optimization
	log()
	result = optimizer.optimize(init_strategy=init_strategy)

	# Log final results
	log()
	log("=" * 60)
	log("  Architecture Search Complete")
	log("=" * 60)
	stats = result.best_genome.stats()
	log(f"  Initial CE: {result.initial_fitness:.4f}")
	log(f"  Final CE: {result.final_fitness:.4f}")
	improvement = (1 - result.final_fitness / result.initial_fitness) * 100
	log(f"  Improvement: {improvement:.1f}%")
	log(f"  Generations: {result.generations_run}")
	log(f"  Early stopped: {result.early_stopped}")
	log()
	log("  Best genome:")
	log(f"    Bits: [{stats['min_bits']}, {stats['max_bits']}], mean: {stats['mean_bits']:.1f}")
	log(f"    Neurons: [{stats['min_neurons']}, {stats['max_neurons']}], mean: {stats['mean_neurons']:.1f}")
	log(f"    Total memory: {stats['total_memory_cells']:,} cells")

	return result


# =============================================================================
# Connectivity Optimization (Phase 2)
# =============================================================================

@dataclass
class ConnectivityOptResult:
	"""Result from connectivity optimization."""

	initial_fitness: float  # Phase 1 baseline (different connectivity seed)
	phase2_baseline: float  # Trial 0 fitness (fair comparison baseline)
	final_fitness: float
	ga_improvement_pct: float  # Improvement vs phase2_baseline (fair)
	ts_improvement_pct: float
	total_improvement_pct: float  # Improvement vs Phase 1 baseline
	ga_iterations: int
	ts_iterations: int
	early_stopped: bool


def run_connectivity_optimization(
	genome: ClusterGenome,
	genome_fitness: float,
	train_tokens: List[int],
	eval_tokens: List[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	cluster_order: Optional[List[int]] = None,
	# GA parameters for connectivity
	ga_population: int = 20,
	ga_generations: int = 30,
	ga_patience: int = 5,
	# TS parameters for connectivity
	ts_iterations: int = 50,
	ts_neighbors: int = 30,
	ts_patience: int = 5,
	# Other
	empty_value: float = 0.0,
	seed: int = 42,
	logger: Optional[Callable[[str], None]] = None,
) -> ConnectivityOptResult:
	"""
	Run Phase 2: Optimize connectivity patterns for a fixed architecture.

	After architecture (bits, neurons per cluster) is locked from Phase 1a/1b,
	this phase optimizes which input bits each neuron observes.

	The pipeline:
	1. Initialize model with genome's architecture
	2. Run GA to explore connectivity space
	3. Run TS to refine best GA solution

	Args:
		genome: Fixed architecture from Phase 1
		genome_fitness: Baseline fitness before connectivity optimization
		train_tokens: Training data
		eval_tokens: Evaluation data
		vocab_size: Vocabulary size
		context_size: Context window
		cluster_order: Token ordering by frequency
		ga_population: GA population size
		ga_generations: GA generations
		ga_patience: GA early stop patience
		ts_iterations: TS iterations
		ts_neighbors: TS neighbors per iteration
		ts_patience: TS early stop patience
		empty_value: EMPTY cell value
		seed: Random seed
		logger: Logging function

	Returns:
		ConnectivityOptResult with optimization statistics
	"""
	from wnn.ram.core import AdaptiveClusteredRAM, bits_needed

	log = logger or print

	log()
	log("=" * 60)
	log("  Phase 2: Connectivity Optimization")
	log("=" * 60)
	log(f"  Architecture fitness: {genome_fitness:.4f}")
	log(f"  GA: pop={ga_population}, gens={ga_generations}")
	log(f"  TS: iters={ts_iterations}, neighbors={ts_neighbors}")
	log()

	bits_per_token = bits_needed(vocab_size)
	total_input_bits = context_size * bits_per_token

	# Create evaluator config (no rng = truly random connectivity each time)
	eval_config = EvaluatorConfig(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		batch_size=500,
		global_top_k=100,
		empty_value=empty_value,
		eval_batch_size=1000,
		cluster_order=cluster_order,
		# rng=None: each trial gets truly random connectivity
	)

	log("[Phase2] Connectivity variance analysis...")
	log(f"[Phase2] Evaluating architecture {ga_population} times with random connectivity")
	log(f"[Phase2] Phase 1 baseline: {genome_fitness:.4f}")

	# Track fitness across random connectivity trials
	all_fitness = []
	best_fitness = float('inf')

	# Progress tracker for consistent logging
	tracker = ProgressTracker(
		logger=log,
		minimize=True,  # Lower CE is better
		prefix="[Phase2]",
		total_generations=ga_population,
	)

	# Evaluate the same architecture with different random connectivity
	# This measures how robust the architecture is to connectivity variance
	for trial in range(ga_population):
		# Create layer with truly random connections (no seeding)
		layer_trial = AdaptiveClusteredRAM.from_genome(
			genome=genome,
			total_input_bits=total_input_bits,
			empty_value=empty_value,
			rng=None,  # Truly random connectivity
		)

		# Evaluate with trial connections
		wrapper = AdaptiveRAMLMWrapper.__new__(AdaptiveRAMLMWrapper)
		wrapper.genome = genome
		wrapper.config = eval_config
		wrapper.vocab_size = vocab_size
		wrapper.context_size = context_size
		wrapper.bits_per_token = bits_per_token
		wrapper.total_input_bits = total_input_bits
		wrapper.layer = layer_trial
		wrapper._cluster_order = cluster_order
		if cluster_order is not None:
			wrapper._token_to_cluster = {tid: idx for idx, tid in enumerate(cluster_order)}
		else:
			wrapper._token_to_cluster = None

		from torch import arange, long
		wrapper._bit_positions = arange(bits_per_token - 1, -1, -1, dtype=long)

		wrapper.train_epoch(
			train_tokens,
			global_top_k=eval_config.global_top_k,
			batch_size=eval_config.batch_size,
			verbose=False,
		)

		stats = wrapper.evaluate(
			eval_tokens,
			batch_size=eval_config.eval_batch_size,
			verbose=False,
		)

		trial_fitness = stats['cross_entropy']
		all_fitness.append(trial_fitness)

		# Track best
		if trial_fitness < best_fitness:
			best_fitness = trial_fitness

		# Use tracker for consistent logging (global_best, trial_best, trial_avg)
		tracker.tick([trial_fitness], generation=trial)

	# Calculate statistics across random connectivity trials
	import statistics
	mean_fitness = statistics.mean(all_fitness)
	std_fitness = statistics.stdev(all_fitness) if len(all_fitness) > 1 else 0.0
	min_fitness = min(all_fitness)
	max_fitness = max(all_fitness)

	log()
	log("=" * 60)
	log("  Phase 2 Complete: Connectivity Variance Analysis")
	log("=" * 60)
	log(f"  Phase 1 baseline CE: {genome_fitness:.4f}")
	log(f"  Trials: {ga_population} random connectivity patterns")
	log()
	log(f"  Statistics across {ga_population} trials:")
	log(f"    Mean CE: {mean_fitness:.4f}")
	log(f"    Std CE:  {std_fitness:.4f}")
	log(f"    Min CE:  {min_fitness:.4f}")
	log(f"    Max CE:  {max_fitness:.4f}")
	log(f"    Range:   {max_fitness - min_fitness:.4f}")

	return ConnectivityOptResult(
		initial_fitness=genome_fitness,
		phase2_baseline=mean_fitness,  # Mean is the "fair" baseline
		final_fitness=min_fitness,  # Best across trials
		ga_improvement_pct=(mean_fitness - min_fitness) / mean_fitness * 100 if mean_fitness > 0 else 0,
		ts_improvement_pct=0.0,
		total_improvement_pct=(genome_fitness - min_fitness) / genome_fitness * 100 if genome_fitness > 0 else 0,
		ga_iterations=ga_population,
		ts_iterations=0,
		early_stopped=False,
	)
