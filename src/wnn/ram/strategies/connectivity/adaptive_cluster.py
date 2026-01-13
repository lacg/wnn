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

	max_bits: int = 20
	"""Maximum bits per neuron (2^20 = 1M addresses)"""

	min_neurons: int = 1
	"""Minimum neurons per cluster (Phase 2)"""

	max_neurons: int = 50
	"""Maximum neurons per cluster (Phase 2)"""

	total_memory_budget: int = 1_000_000_000
	"""Total memory cells allowed across all clusters"""

	init_strategy: GenomeInitStrategy = GenomeInitStrategy.FREQUENCY_SCALED
	"""How to initialize the genome"""

	# Mutation rates
	bits_mutation_rate: float = 0.1
	"""Probability of mutating bits for a cluster"""

	bits_mutation_step: int = 1
	"""How much to change bits per mutation (+/- this value)"""


@dataclass
class ClusterGenome:
	"""
	Genome representing adaptive architecture for all clusters.

	Phase 1: Only bits_per_cluster (neurons fixed at 1)
	Phase 2: Add neurons_per_cluster
	"""

	bits_per_cluster: List[int]
	"""Bits per neuron for each cluster [num_clusters]"""

	# Phase 2 fields (future)
	# neurons_per_cluster: List[int]
	# connectivity_per_cluster: List[Tensor]

	def total_memory_cells(self) -> int:
		"""Calculate total memory cells needed for this genome."""
		# Each cluster: 1 neuron × 2^bits addresses × 1 cell per address
		return sum(2 ** bits for bits in self.bits_per_cluster)

	def clone(self) -> 'ClusterGenome':
		"""Create a deep copy of this genome."""
		return ClusterGenome(
			bits_per_cluster=self.bits_per_cluster.copy()
		)

	def to_tensor(self) -> Tensor:
		"""Convert to tensor for batch operations."""
		return torch.tensor(self.bits_per_cluster, dtype=torch.int32)

	@staticmethod
	def from_tensor(t: Tensor) -> 'ClusterGenome':
		"""Create genome from tensor."""
		return ClusterGenome(bits_per_cluster=t.tolist())


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
		Initialized ClusterGenome
	"""
	import random
	if rng is not None:
		random.seed(rng)

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
			num_clusters, token_frequencies, config
		)

	elif strategy == GenomeInitStrategy.RANDOM_UNIFORM:
		bits = [
			random.randint(config.min_bits, config.max_bits)
			for _ in range(num_clusters)
		]
	else:
		raise ValueError(f"Unknown strategy: {strategy}")

	return ClusterGenome(bits_per_cluster=bits)


def _frequency_scaled_init(
	num_clusters: int,
	token_frequencies: List[int],
	config: AdaptiveClusterConfig,
) -> List[int]:
	"""
	Initialize bits scaled by token frequency.

	More frequent tokens get more bits (larger address space).
	Rare tokens get fewer bits (small address space sufficient).

	Uses log-scale mapping from frequency to bits.
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

	# Log-scale mapping: high freq -> high bits, low freq -> low bits
	log_max = math.log(max_freq)
	log_min = math.log(min_freq)
	log_range = log_max - log_min if log_max > log_min else 1.0

	bits_range = config.max_bits - config.min_bits

	bits = []
	for freq in freqs:
		# Normalize log frequency to [0, 1]
		log_freq = math.log(freq)
		normalized = (log_freq - log_min) / log_range

		# Map to bits range
		cluster_bits = config.min_bits + int(normalized * bits_range)
		cluster_bits = max(config.min_bits, min(config.max_bits, cluster_bits))
		bits.append(cluster_bits)

	return bits


def mutate_genome(
	genome: ClusterGenome,
	config: AdaptiveClusterConfig,
	rng: Optional[int] = None,
) -> ClusterGenome:
	"""
	Mutate a genome by randomly adjusting bits per cluster.

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

	for i in range(len(new_bits)):
		if random.random() < config.bits_mutation_rate:
			# Randomly grow or shrink
			delta = random.choice([-config.bits_mutation_step, config.bits_mutation_step])
			new_bits[i] = max(
				config.min_bits,
				min(config.max_bits, new_bits[i] + delta)
			)

	return ClusterGenome(bits_per_cluster=new_bits)


def crossover_genomes(
	parent1: ClusterGenome,
	parent2: ClusterGenome,
	crossover_rate: float = 0.5,
	rng: Optional[int] = None,
) -> ClusterGenome:
	"""
	Create child genome by crossing over two parents.

	Uses uniform crossover: each cluster's bits come from either parent.

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
	for b1, b2 in zip(parent1.bits_per_cluster, parent2.bits_per_cluster):
		if random.random() < crossover_rate:
			child_bits.append(b2)
		else:
			child_bits.append(b1)

	return ClusterGenome(bits_per_cluster=child_bits)


def genome_stats(genome: ClusterGenome) -> Dict:
	"""Get statistics about a genome."""
	bits = genome.bits_per_cluster
	return {
		"num_clusters": len(bits),
		"min_bits": min(bits),
		"max_bits": max(bits),
		"mean_bits": sum(bits) / len(bits),
		"total_memory_cells": genome.total_memory_cells(),
		"bits_distribution": {
			b: bits.count(b) for b in sorted(set(bits))
		}
	}
