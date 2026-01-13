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
			bits = [
				random.randint(config.min_bits, config.max_bits)
				for _ in range(num_clusters)
			]
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
		"""
		self.config = config
		self.evaluate_fn = evaluate_fn
		self.num_clusters = num_clusters
		self.token_frequencies = token_frequencies
		self.seed = seed
		self._log = logger or print
		self._rng = None

	def _ensure_rng(self):
		if self._rng is None:
			self._rng = random.Random(self.seed)

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

		# Evaluate initial population with progress logging
		self._log(f"[AdaptiveGA] Evaluating initial population ({cfg.population_size} genomes)...")
		fitness = []
		for i, genome in enumerate(population):
			fit = self.evaluate_fn(genome)
			fitness.append(fit)
			self._log(f"  [{i+1}/{cfg.population_size}] CE={fit:.4f}")

		best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
		best_genome = population[best_idx].clone()
		best_fitness = fitness[best_idx]
		initial_fitness = best_fitness

		history = [(0, best_fitness, sum(fitness) / len(fitness))]
		self._log(f"[AdaptiveGA] Initial best fitness: {best_fitness:.4f}")

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
			self._log(f"[AdaptiveGA] Gen {gen + 1}: evaluating {cfg.population_size} genomes...")
			fitness = []
			for i, genome in enumerate(population):
				fit = self.evaluate_fn(genome)
				fitness.append(fit)
			gen_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])

			if fitness[gen_best_idx] < best_fitness:
				best_genome = population[gen_best_idx].clone()
				best_fitness = fitness[gen_best_idx]

			avg_fitness = sum(fitness) / len(fitness)
			history.append((gen + 1, best_fitness, avg_fitness))

			# Log progress every generation
			stats = best_genome.stats()
			self._log(
				f"[AdaptiveGA] Gen {gen + 1}/{cfg.generations}: "
				f"best={best_fitness:.4f}, avg={avg_fitness:.4f}, "
				f"bits=[{stats['min_bits']}-{stats['max_bits']}], "
				f"neurons=[{stats['min_neurons']}-{stats['max_neurons']}]"
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
		self._log(f"  Initial fitness: {initial_fitness:.4f}")
		self._log(f"  Final fitness: {best_fitness:.4f}")
		improvement = (initial_fitness - best_fitness) / initial_fitness * 100
		self._log(f"  Improvement: {improvement:.2f}%")

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

		# Rest are mutations of the first, or random
		for i in range(1, self.config.population_size):
			if i < self.config.population_size // 2:
				# Mutate from first
				genome = population[0].mutate(cc, rng=self.seed + i)
			else:
				# Random initialization for diversity
				genome = ClusterGenome.initialize(
					self.num_clusters,
					GenomeInitStrategy.RANDOM_UNIFORM,
					cc,
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

	# Random seed
	rng: int = 42


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

	# Create evaluator config
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
		rng=seed,
	)

	# Create evaluation function
	evaluate_fn = create_genome_evaluator(eval_config, verbose=False)

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
