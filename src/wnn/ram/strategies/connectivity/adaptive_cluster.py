"""
Adaptive Cluster Architecture Search

Instead of hand-designed tiers, each cluster discovers its optimal architecture
through evolutionary optimization. The GA mutates both connectivity AND structure
(bits per neuron, number of neurons per cluster).

Phase 1: Bits-per-cluster only (neurons fixed at 1)
Phase 2: Add neuron count per cluster (future)

Key insight: Frequent tokens need different architectures than rare tokens.
Let the data decide rather than hand-tuning tier boundaries.
"""

from enum import IntEnum
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import torch
from torch import Tensor


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


@dataclass
class ClusterGenome:
	"""
	Genome representing adaptive architecture for all clusters.

	Phase 1: bits_per_cluster (bits per neuron for each cluster)
	Phase 2: neurons_per_cluster (number of neurons for each cluster)

	Each cluster can have different (neurons, bits) configuration,
	allowing the GA to discover optimal architectures per token.
	"""

	bits_per_cluster: List[int]
	"""Bits per neuron for each cluster [num_clusters]"""

	neurons_per_cluster: Optional[List[int]] = None
	"""Number of neurons for each cluster [num_clusters] (Phase 2)"""

	def __post_init__(self):
		"""Initialize neurons_per_cluster to 1 if not provided (Phase 1 compat)."""
		if self.neurons_per_cluster is None:
			self.neurons_per_cluster = [1] * len(self.bits_per_cluster)

	def total_memory_cells(self) -> int:
		"""Calculate total memory cells needed for this genome."""
		# Each cluster: neurons × 2^bits addresses
		return sum(
			n * (2 ** b)
			for n, b in zip(self.neurons_per_cluster, self.bits_per_cluster)
		)

	def total_neurons(self) -> int:
		"""Total neurons across all clusters."""
		return sum(self.neurons_per_cluster)

	def clone(self) -> 'ClusterGenome':
		"""Create a deep copy of this genome."""
		return ClusterGenome(
			bits_per_cluster=self.bits_per_cluster.copy(),
			neurons_per_cluster=self.neurons_per_cluster.copy(),
		)

	def to_tensor(self) -> Tensor:
		"""Convert to tensor [num_clusters, 2] with (bits, neurons) per cluster."""
		data = list(zip(self.bits_per_cluster, self.neurons_per_cluster))
		return torch.tensor(data, dtype=torch.int32)

	@staticmethod
	def from_tensor(t: Tensor) -> 'ClusterGenome':
		"""Create genome from tensor [num_clusters, 2]."""
		return ClusterGenome(
			bits_per_cluster=t[:, 0].tolist(),
			neurons_per_cluster=t[:, 1].tolist(),
		)

	def get_cluster_config(self, cluster_id: int) -> tuple:
		"""Get (neurons, bits) for a specific cluster."""
		return (self.neurons_per_cluster[cluster_id], self.bits_per_cluster[cluster_id])


def initialize_genome(
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
	"""
	import random
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

	return ClusterGenome(bits_per_cluster=bits, neurons_per_cluster=neurons)


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

	Args:
		num_clusters: Number of clusters
		token_frequencies: Occurrence count per token
		config: Configuration with min/max bounds
		for_bits: If True, scale bits; if False, scale neurons

	Returns:
		List of bits or neurons per cluster
	"""
	import math

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


def mutate_genome(
	genome: ClusterGenome,
	config: AdaptiveClusterConfig,
	rng: Optional[int] = None,
) -> ClusterGenome:
	"""
	Mutate a genome by randomly adjusting bits and neurons per cluster.

	Phase 1: Only mutates bits
	Phase 2: Mutates both bits and neurons

	Args:
		genome: Genome to mutate
		config: Configuration with mutation rates and bounds
		rng: Random seed

	Returns:
		New mutated genome (original unchanged)
	"""
	import random
	if rng is not None:
		random.seed(rng)

	new_bits = genome.bits_per_cluster.copy()
	new_neurons = genome.neurons_per_cluster.copy()

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


def crossover_genomes(
	parent1: ClusterGenome,
	parent2: ClusterGenome,
	crossover_rate: float = 0.5,
	rng: Optional[int] = None,
) -> ClusterGenome:
	"""
	Create child genome by crossing over two parents.

	Uses uniform crossover: each cluster's (bits, neurons) come from either parent.
	The entire cluster config is inherited together (not bits from one, neurons from other).

	Args:
		parent1: First parent genome
		parent2: Second parent genome
		crossover_rate: Probability of taking from parent2 (default 0.5)
		rng: Random seed

	Returns:
		New child genome
	"""
	import random
	if rng is not None:
		random.seed(rng)

	child_bits = []
	child_neurons = []

	for i in range(len(parent1.bits_per_cluster)):
		if random.random() < crossover_rate:
			child_bits.append(parent2.bits_per_cluster[i])
			child_neurons.append(parent2.neurons_per_cluster[i])
		else:
			child_bits.append(parent1.bits_per_cluster[i])
			child_neurons.append(parent1.neurons_per_cluster[i])

	return ClusterGenome(bits_per_cluster=child_bits, neurons_per_cluster=child_neurons)


def genome_stats(genome: ClusterGenome) -> Dict:
	"""Get statistics about a genome."""
	bits = genome.bits_per_cluster
	neurons = genome.neurons_per_cluster

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
		"total_memory_cells": genome.total_memory_cells(),
		# Distributions
		"bits_distribution": {
			b: bits.count(b) for b in sorted(set(bits))
		},
		"neurons_distribution": {
			n: neurons.count(n) for n in sorted(set(neurons))
		},
	}


# =============================================================================
# Phase 1: Adaptive Cluster Optimizer (bits-per-cluster only)
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
	Evolves cluster architectures (bits-per-cluster) using GA.

	Phase 1: Each cluster has 1 neuron, GA optimizes bits per cluster.
	Fitness is evaluated by training a model with the genome's architecture
	and measuring cross-entropy on held-out data.
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
		import random
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

		# Evaluate initial population
		fitness = [self.evaluate_fn(g) for g in population]
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
					child = crossover_genomes(parent1, parent2, rng=self._rng.randint(0, 2**31))
				else:
					child = parent1.clone()

				# Mutation
				child = mutate_genome(child, cc, rng=self._rng.randint(0, 2**31))

				# Enforce memory budget
				child = self._enforce_budget(child, cc)

				new_population.append(child)

			population = new_population

			# Evaluate new population
			fitness = [self.evaluate_fn(g) for g in population]
			gen_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])

			if fitness[gen_best_idx] < best_fitness:
				best_genome = population[gen_best_idx].clone()
				best_fitness = fitness[gen_best_idx]

			avg_fitness = sum(fitness) / len(fitness)
			history.append((gen + 1, best_fitness, avg_fitness))

			# Log progress
			if (gen + 1) % 5 == 0 or gen == 0:
				stats = genome_stats(best_genome)
				self._log(
					f"[AdaptiveGA] Gen {gen + 1}/{cfg.generations}: "
					f"best={best_fitness:.4f}, avg={avg_fitness:.4f}, "
					f"bits=[{stats['min_bits']}-{stats['max_bits']}]"
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
		population.append(initialize_genome(
			self.num_clusters, strategy, cc,
			token_frequencies=self.token_frequencies,
			rng=self.seed,
		))

		# Rest are mutations of the first, or random
		for i in range(1, self.config.population_size):
			if i < self.config.population_size // 2:
				# Mutate from first
				genome = mutate_genome(population[0], cc, rng=self.seed + i)
			else:
				# Random initialization for diversity
				genome = initialize_genome(
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

	def _enforce_budget(
		self,
		genome: ClusterGenome,
		cc: AdaptiveClusterConfig,
	) -> ClusterGenome:
		"""Ensure genome stays within memory budget by shrinking if needed."""
		if genome.total_memory_cells() <= cc.total_memory_budget:
			return genome

		# Shrink largest clusters until under budget
		bits = genome.bits_per_cluster.copy()
		while sum(2 ** b for b in bits) > cc.total_memory_budget:
			# Find cluster with most bits
			max_idx = max(range(len(bits)), key=lambda i: bits[i])
			if bits[max_idx] > cc.min_bits:
				bits[max_idx] -= 1
			else:
				break  # Can't shrink further

		return ClusterGenome(bits_per_cluster=bits)
