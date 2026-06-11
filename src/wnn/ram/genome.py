"""
ClusterGenome — the DNA of an adaptive RAM architecture.

Owns: per-cluster neuron counts, per-neuron bit counts, flat connections,
optional memory cells (Lamarckian), mutation/crossover operators, and
(de)serialization. Plus its tightly-coupled config (AdaptiveClusterConfig),
init-strategy enum, and connection helpers — the class-per-file rule's
"coupled helpers" exception.

Moved from strategies/connectivity/adaptive_cluster.py (D6c, 11/06/2026) to a
neutral module so core/ no longer imports from strategies/ (layering fix).
adaptive_cluster re-exports everything for backward compatibility.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor

def generate_connections(bits_per_neuron: list[int], total_input_bits: int, seed: int | None = None) -> list[int]:
	"""Generate random connections using Rust accelerator with numpy fallback."""
	if seed is None:
		seed = random.randint(0, 2**63)
	if _HAS_RUST:
		return _accel.generate_random_connections(bits_per_neuron, total_input_bits, seed)
	np_rng = np.random.default_rng(seed)
	return np_rng.integers(0, total_input_bits, size=sum(bits_per_neuron)).tolist()


def enforce_unique_connections(
	conns: list[int],
	bits_per_neuron: list[int],
	total_input_bits: int,
	rng: random.Random,
) -> None:
	"""Ensure no duplicate input bit indices within any neuron. Modifies conns in-place."""
	offset = 0
	for b in bits_per_neuron:
		neuron_slice = conns[offset:offset + b]
		seen: set[int] = set()
		for i, c in enumerate(neuron_slice):
			if c in seen:
				# Replace with a random index not already in this neuron
				for _ in range(100):
					candidate = rng.randint(0, total_input_bits - 1)
					if candidate not in seen:
						conns[offset + i] = candidate
						seen.add(candidate)
						break
			else:
				seen.add(c)
		offset += b


# PhaseType was promoted to the shared framework as OptimizationDimension (it is
# not IDS-specific). Re-exported here so existing `from ...adaptive_cluster import
# PhaseType` imports keep working. New code should import OptimizationDimension.
from wnn.ram.strategies.optimization_dimension import OptimizationDimension, PhaseType  # noqa: E402,F401


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

	max_bit_delta: int = 0
	"""Max bits change per mutation (0 = auto: ~10% of bit range). Caps overfitting jumps."""

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

	Contains three components:
	- bits_per_neuron: Address bits for each neuron [total_neurons]
	- neurons_per_cluster: Number of neurons per cluster [num_clusters]
	- connections: Input bit indices each neuron observes (flat: sum(bits_per_neuron))

	Each neuron owns its own bit count, enabling per-neuron evolution.
	Connections are stored flat: neuron 0's connections, then neuron 1's, etc.

	CRITICAL: Connections must be preserved across mutations and crossovers.
	Random regeneration breaks evolutionary search because "neighbors" become
	completely different models.
	"""

	def __init__(
		self,
		bits_per_neuron: list[int],
		neurons_per_cluster: list[int],
		connections: Optional[list[int]] = None,
		threshold: float = 0.5,
	):
		"""
		Create a genome with specified architecture and connections.

		Args:
			bits_per_neuron: Bits per neuron [total_neurons]
			neurons_per_cluster: Neurons per cluster [num_clusters]
			connections: Flattened connection indices (default: None = not initialized)
			threshold: Decision threshold for single-cluster classification (default: 0.5)
		"""
		self.bits_per_neuron = bits_per_neuron
		self.neurons_per_cluster = neurons_per_cluster
		self.connections = connections
		self.threshold = threshold

	# =========================================================================
	# Factory Methods
	# =========================================================================

	@classmethod
	def create_uniform(
		cls,
		num_clusters: int,
		bits: int,
		neurons: int,
		total_input_bits: Optional[int] = None,
		rng: Optional[int] = None,
	) -> 'ClusterGenome':
		"""
		Create a genome with uniform bits and neurons across all clusters.

		Args:
			num_clusters: Total number of clusters
			bits: Bits per neuron for all neurons
			neurons: Neurons per cluster for all clusters
			total_input_bits: Total input bits (for random connection init)
			rng: Random seed for connection initialization

		Returns:
			ClusterGenome with uniform architecture
		"""
		import random
		total_neurons = num_clusters * neurons
		bits_per_neuron = [bits] * total_neurons
		neurons_per_cluster = [neurons] * num_clusters

		connections = None
		if total_input_bits is not None:
			connections = generate_connections(bits_per_neuron, total_input_bits, rng)

		return cls(
			bits_per_neuron=bits_per_neuron,
			neurons_per_cluster=neurons_per_cluster,
			connections=connections,
		)

	@classmethod
	def initialize(
		cls,
		num_clusters: int,
		strategy: GenomeInitStrategy,
		config: AdaptiveClusterConfig,
		token_frequencies: Optional[list[int]] = None,
		total_input_bits: Optional[int] = None,
		rng: Optional[int] = None,
	) -> ClusterGenome:
		"""
		Initialize a cluster genome using the specified strategy.

		Args:
			num_clusters: Total number of clusters (e.g., 50257 for GPT-2)
			strategy: Initialization strategy to use
			config: Configuration with min/max bounds
			token_frequencies: Token occurrence counts (required for FREQUENCY_SCALED)
			total_input_bits: Total input bits (required for connection initialization)
			rng: Random seed for reproducibility

		Returns:
			Initialized ClusterGenome with per-neuron bits, neurons, and connections
		"""
		if rng is not None:
			random.seed(rng)

		# Initialize per-cluster bits (will expand to per-neuron below)
		if strategy == GenomeInitStrategy.UNIFORM_MINIMAL:
			cluster_bits = [config.min_bits] * num_clusters
		elif strategy == GenomeInitStrategy.UNIFORM_MEDIUM:
			medium = (config.min_bits + config.max_bits) // 2
			cluster_bits = [medium] * num_clusters
		elif strategy == GenomeInitStrategy.UNIFORM_MAXIMUM:
			cluster_bits = [config.max_bits] * num_clusters
		elif strategy == GenomeInitStrategy.FREQUENCY_SCALED:
			if token_frequencies is None:
				raise ValueError("FREQUENCY_SCALED requires token_frequencies")
			cluster_bits = _frequency_scaled_init(
				num_clusters, token_frequencies, config, for_bits=True
			)
		elif strategy == GenomeInitStrategy.RANDOM_UNIFORM:
			cluster_bits = []
			for i in range(num_clusters):
				freq = token_frequencies[i] if token_frequencies else 1000
				if freq < 10:
					max_b = min(config.max_bits, 8)
				elif freq < 100:
					max_b = min(config.max_bits, 10)
				elif freq < 1000:
					max_b = min(config.max_bits, 12)
				else:
					max_b = config.max_bits
				cluster_bits.append(random.randint(config.min_bits, max_b))
		else:
			raise ValueError(f"Unknown strategy: {strategy}")

		# Initialize neurons per cluster
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
			neurons = [1] * num_clusters

		# Expand cluster_bits to per-neuron bits
		bits_per_neuron = []
		for i in range(num_clusters):
			bits_per_neuron.extend([cluster_bits[i]] * neurons[i])

		# Initialize connections if total_input_bits provided
		connections = None
		if total_input_bits is not None:
			connections = generate_connections(bits_per_neuron, total_input_bits, rng)

		return cls(bits_per_neuron=bits_per_neuron, neurons_per_cluster=neurons, connections=connections)

	@classmethod
	def from_tensor(cls, t: Tensor) -> ClusterGenome:
		"""Create genome from tensor [num_clusters, 2] with (bits, neurons).

		Expands per-cluster bits to per-neuron bits.
		"""
		cluster_bits = t[:, 0].tolist()
		neurons_per_cluster = t[:, 1].tolist()
		bits_per_neuron = []
		for b, n in zip(cluster_bits, neurons_per_cluster):
			bits_per_neuron.extend([int(b)] * int(n))
		return cls(
			bits_per_neuron=bits_per_neuron,
			neurons_per_cluster=[int(n) for n in neurons_per_cluster],
		)

	# =========================================================================
	# Genetic Operations (Phase-Aware)
	# =========================================================================

	def crossover(
		self,
		other: ClusterGenome,
		phase_type: PhaseType,
		rng: random.Random | None = None,
		pool_shuffle_ratio: float = 0.0,
	) -> ClusterGenome:
		"""Phase-aware crossover dispatch. Uses pool-shuffle or uniform based on ratio."""
		if rng is None:
			rng = random.Random()
		if pool_shuffle_ratio > 0.0 and rng.random() < pool_shuffle_ratio:
			return self._crossover_pool_shuffle(other, phase_type, rng)
		child1, _ = self.crossover2(other, phase_type, rng)
		return child1

	# =========================================================================
	# Pool-and-Shuffle Crossover (Old-style 2→1)
	# =========================================================================

	def crossover_pool_shuffle2(self, other: ClusterGenome, phase_type: PhaseType, rng: random.Random) -> tuple[ClusterGenome, ClusterGenome]:
		"""Pool-and-shuffle crossover producing 2 complementary offspring."""
		match phase_type:
			case PhaseType.NEURONS:
				return self._crossover_ps_neurons(other, rng)
			case PhaseType.BITS:
				return self._crossover_ps_bits(other, rng)
			case PhaseType.CONNECTIONS:
				return self._crossover_ps_connections(other, rng)
			case PhaseType.CLUSTER:
				return self._crossover2_cluster(other, rng)

	def _crossover_ps_neurons(self, other: ClusterGenome, rng: random.Random) -> tuple[ClusterGenome, ClusterGenome]:
		"""Pool all neurons, shuffle once, complementary partition: child1 gets first p1_n, child2 gets next p2_n."""
		num_clusters = len(self.neurons_per_cluster)
		self_off = self.cluster_neuron_offsets
		other_off = other.cluster_neuron_offsets
		self_conn_off = self.connection_offsets
		other_conn_off = other.connection_offsets
		has_conns = self.connections is not None and other.connections is not None
		c1_bits, c1_neurons, c1_conns = [], [], [] if has_conns else None
		c2_bits, c2_neurons, c2_conns = [], [], [] if has_conns else None
		for c in range(num_clusters):
			p1_n = self.neurons_per_cluster[c]
			p2_n = other.neurons_per_cluster[c]
			c1_neurons.append(p1_n)
			c2_neurons.append(p2_n)
			pool = []
			for local in range(p1_n):
				g = self_off[c] + local
				bits = self.bits_per_neuron[g]
				conns = self.connections[self_conn_off[g]:self_conn_off[g + 1]] if self.connections else []
				pool.append((bits, conns))
			for local in range(p2_n):
				g = other_off[c] + local
				bits = other.bits_per_neuron[g]
				conns = other.connections[other_conn_off[g]:other_conn_off[g + 1]] if other.connections else []
				pool.append((bits, conns))
			rng.shuffle(pool)
			for i, (bits, conns) in enumerate(pool):
				if i < p1_n:
					c1_bits.append(bits)
					if c1_conns is not None:
						c1_conns.extend(conns)
				elif i < p1_n + p2_n:
					c2_bits.append(bits)
					if c2_conns is not None:
						c2_conns.extend(conns)
		return (
			ClusterGenome(bits_per_neuron=c1_bits, neurons_per_cluster=c1_neurons, connections=c1_conns),
			ClusterGenome(bits_per_neuron=c2_bits, neurons_per_cluster=c2_neurons, connections=c2_conns),
		)

	def _crossover_ps_bits(self, other: ClusterGenome, rng: random.Random) -> tuple[ClusterGenome, ClusterGenome]:
		"""Per-neuron coin flip with complement. Both children preserve their parent's neuron counts."""
		num_clusters = len(self.neurons_per_cluster)
		self_off = self.cluster_neuron_offsets
		other_off = other.cluster_neuron_offsets
		self_conn_off = self.connection_offsets
		other_conn_off = other.connection_offsets
		has_conns = self.connections is not None and other.connections is not None
		c1_neurons = self.neurons_per_cluster.copy()
		c2_neurons = other.neurons_per_cluster.copy()
		c1_bits, c1_conns = [], [] if has_conns else None
		c2_bits, c2_conns = [], [] if has_conns else None
		for c in range(num_clusters):
			p1_n = self.neurons_per_cluster[c]
			p2_n = other.neurons_per_cluster[c]
			shared = min(p1_n, p2_n)
			for local in range(p1_n):
				g_self = self_off[c] + local
				if local < shared and rng.random() < 0.5:
					g_other = other_off[c] + local
					c1_bits.append(other.bits_per_neuron[g_other])
					if c1_conns is not None:
						c1_conns.extend(other.connections[other_conn_off[g_other]:other_conn_off[g_other + 1]])
				else:
					c1_bits.append(self.bits_per_neuron[g_self])
					if c1_conns is not None:
						c1_conns.extend(self.connections[self_conn_off[g_self]:self_conn_off[g_self + 1]])
			for local in range(p2_n):
				g_other = other_off[c] + local
				if local < shared and rng.random() < 0.5:
					g_self = self_off[c] + local
					c2_bits.append(self.bits_per_neuron[g_self])
					if c2_conns is not None:
						c2_conns.extend(self.connections[self_conn_off[g_self]:self_conn_off[g_self + 1]])
				else:
					c2_bits.append(other.bits_per_neuron[g_other])
					if c2_conns is not None:
						c2_conns.extend(other.connections[other_conn_off[g_other]:other_conn_off[g_other + 1]])
		return (
			ClusterGenome(bits_per_neuron=c1_bits, neurons_per_cluster=c1_neurons, connections=c1_conns),
			ClusterGenome(bits_per_neuron=c2_bits, neurons_per_cluster=c2_neurons, connections=c2_conns),
		)

	def _crossover_ps_connections(self, other: ClusterGenome, rng: random.Random) -> tuple[ClusterGenome, ClusterGenome]:
		"""Per-connection coin flip with complement. Falls back to bits if different arch."""
		same_arch = (self.neurons_per_cluster == other.neurons_per_cluster and self.bits_per_neuron == other.bits_per_neuron)
		if same_arch and self.connections is not None and other.connections is not None:
			c1_conns, c2_conns = [], []
			for c1, c2 in zip(self.connections, other.connections):
				if rng.random() < 0.5:
					c1_conns.append(c2)
					c2_conns.append(c1)
				else:
					c1_conns.append(c1)
					c2_conns.append(c2)
			return (
				ClusterGenome(bits_per_neuron=self.bits_per_neuron.copy(), neurons_per_cluster=self.neurons_per_cluster.copy(), connections=c1_conns),
				ClusterGenome(bits_per_neuron=other.bits_per_neuron.copy(), neurons_per_cluster=other.neurons_per_cluster.copy(), connections=c2_conns),
			)
		return self._crossover_ps_bits(other, rng)

	# =========================================================================
	# Two-Offspring Crossover (Classical GA: 2 parents → 2 children)
	# =========================================================================

	def crossover2(
		self,
		other: ClusterGenome,
		phase_type: PhaseType,
		rng: random.Random | None = None,
	) -> tuple[ClusterGenome, ClusterGenome]:
		"""Phase-aware crossover returning 2 complementary offspring."""
		if rng is None:
			rng = random.Random()
		match phase_type:
			case PhaseType.NEURONS:
				return self._crossover2_neurons(other, rng)
			case PhaseType.BITS:
				return self._crossover2_bits(other, rng)
			case PhaseType.CONNECTIONS:
				return self._crossover2_connections(other, rng)
			case PhaseType.CLUSTER:
				return self._crossover2_cluster(other, rng)

	def _crossover2_neurons(self, other: ClusterGenome, rng: random.Random) -> tuple[ClusterGenome, ClusterGenome]:
		"""Per-position uniform crossover: coin flip per neuron position, complementary children."""
		num_clusters = len(self.neurons_per_cluster)
		self_off = self.cluster_neuron_offsets
		other_off = other.cluster_neuron_offsets
		self_conn_off = self.connection_offsets
		other_conn_off = other.connection_offsets

		c1_bits, c1_neurons, c1_conns = [], [], []
		c2_bits, c2_neurons, c2_conns = [], [], []
		has_conns = self.connections is not None and other.connections is not None

		for c in range(num_clusters):
			p1_n = self.neurons_per_cluster[c]
			p2_n = other.neurons_per_cluster[c]
			shared = min(p1_n, p2_n)

			c1_neurons.append(p1_n)
			c2_neurons.append(p2_n)

			# Shared positions: coin flip determines which parent each child draws from
			for local in range(shared):
				g1 = self_off[c] + local
				g2 = other_off[c] + local
				if rng.random() < 0.5:
					# child1 ← parent2, child2 ← parent1
					c1_bits.append(other.bits_per_neuron[g2])
					c2_bits.append(self.bits_per_neuron[g1])
					if has_conns:
						c1_conns.extend(other.connections[other_conn_off[g2]:other_conn_off[g2 + 1]])
						c2_conns.extend(self.connections[self_conn_off[g1]:self_conn_off[g1 + 1]])
				else:
					# child1 ← parent1, child2 ← parent2
					c1_bits.append(self.bits_per_neuron[g1])
					c2_bits.append(other.bits_per_neuron[g2])
					if has_conns:
						c1_conns.extend(self.connections[self_conn_off[g1]:self_conn_off[g1 + 1]])
						c2_conns.extend(other.connections[other_conn_off[g2]:other_conn_off[g2 + 1]])

			# Extra neurons: child1 inherits parent1's extras, child2 inherits parent2's extras
			for local in range(shared, p1_n):
				g1 = self_off[c] + local
				c1_bits.append(self.bits_per_neuron[g1])
				if has_conns:
					c1_conns.extend(self.connections[self_conn_off[g1]:self_conn_off[g1 + 1]])
			for local in range(shared, p2_n):
				g2 = other_off[c] + local
				c2_bits.append(other.bits_per_neuron[g2])
				if has_conns:
					c2_conns.extend(other.connections[other_conn_off[g2]:other_conn_off[g2 + 1]])

		child1 = ClusterGenome(bits_per_neuron=c1_bits, neurons_per_cluster=c1_neurons,
							   connections=c1_conns if has_conns else None)
		child2 = ClusterGenome(bits_per_neuron=c2_bits, neurons_per_cluster=c2_neurons,
							   connections=c2_conns if has_conns else None)
		return child1, child2

	def _crossover2_bits(self, other: ClusterGenome, rng: random.Random) -> tuple[ClusterGenome, ClusterGenome]:
		"""Per-neuron swap: pre-record swap decisions, build both children incrementally."""
		num_clusters = len(self.neurons_per_cluster)
		self_off = self.cluster_neuron_offsets
		other_off = other.cluster_neuron_offsets
		self_conn_off = self.connection_offsets
		other_conn_off = other.connection_offsets

		# Pre-compute swap decisions
		swap_map = []
		for c in range(num_clusters):
			min_n = min(self.neurons_per_cluster[c], other.neurons_per_cluster[c])
			swap_map.append([rng.random() < 0.5 for _ in range(min_n)])

		c1_neurons = self.neurons_per_cluster.copy()
		c2_neurons = other.neurons_per_cluster.copy()
		c1_bits, c2_bits = [], []
		has_conns = self.connections is not None and other.connections is not None
		c1_conns, c2_conns = [], []

		for c in range(num_clusters):
			p1_n = self.neurons_per_cluster[c]
			p2_n = other.neurons_per_cluster[c]

			# Build child1 (p1_n neurons)
			for local in range(p1_n):
				swap = local < len(swap_map[c]) and swap_map[c][local]
				if swap:
					g = other_off[c] + local
					c1_bits.append(other.bits_per_neuron[g])
					if has_conns:
						c1_conns.extend(other.connections[other_conn_off[g]:other_conn_off[g + 1]])
				else:
					g = self_off[c] + local
					c1_bits.append(self.bits_per_neuron[g])
					if has_conns:
						c1_conns.extend(self.connections[self_conn_off[g]:self_conn_off[g + 1]])

			# Build child2 (p2_n neurons) — complementary at shared positions
			for local in range(p2_n):
				swap = local < len(swap_map[c]) and swap_map[c][local]
				if swap:
					g = self_off[c] + local
					c2_bits.append(self.bits_per_neuron[g])
					if has_conns:
						c2_conns.extend(self.connections[self_conn_off[g]:self_conn_off[g + 1]])
				else:
					g = other_off[c] + local
					c2_bits.append(other.bits_per_neuron[g])
					if has_conns:
						c2_conns.extend(other.connections[other_conn_off[g]:other_conn_off[g + 1]])

		child1 = ClusterGenome(bits_per_neuron=c1_bits, neurons_per_cluster=c1_neurons,
							   connections=c1_conns if has_conns else None)
		child2 = ClusterGenome(bits_per_neuron=c2_bits, neurons_per_cluster=c2_neurons,
							   connections=c2_conns if has_conns else None)
		return child1, child2

	def _crossover2_connections(self, other: ClusterGenome, rng: random.Random) -> tuple[ClusterGenome, ClusterGenome]:
		"""Per-neuron connection swap: coin flip picks entire connection set from one parent."""
		num_clusters = len(self.neurons_per_cluster)
		self_off = self.cluster_neuron_offsets
		other_off = other.cluster_neuron_offsets
		self_conn_off = self.connection_offsets
		other_conn_off = other.connection_offsets

		# Children inherit architecture from their respective parent
		c1_neurons = self.neurons_per_cluster.copy()
		c2_neurons = other.neurons_per_cluster.copy()
		c1_bits = self.bits_per_neuron.copy()
		c2_bits = other.bits_per_neuron.copy()
		c1_conns = []
		c2_conns = []

		same_arch = (self.neurons_per_cluster == other.neurons_per_cluster and
					 self.bits_per_neuron == other.bits_per_neuron)

		if same_arch and self.connections is not None and other.connections is not None:
			# Same architecture: per-neuron coin flip, entire connection set swapped
			for n_idx in range(len(self.bits_per_neuron)):
				s1 = self_conn_off[n_idx]
				e1 = self_conn_off[n_idx + 1]
				s2 = other_conn_off[n_idx]
				e2 = other_conn_off[n_idx + 1]
				if rng.random() < 0.5:
					# child1 ← parent2, child2 ← parent1
					c1_conns.extend(other.connections[s2:e2])
					c2_conns.extend(self.connections[s1:e1])
				else:
					# child1 ← parent1, child2 ← parent2
					c1_conns.extend(self.connections[s1:e1])
					c2_conns.extend(other.connections[s2:e2])
		elif self.connections is not None and other.connections is not None:
			# Different architecture: per-neuron coin flip with fill-from-pool
			for c in range(num_clusters):
				p1_n = self.neurons_per_cluster[c]
				p2_n = other.neurons_per_cluster[c]
				shared = min(p1_n, p2_n)

				for local in range(shared):
					g1 = self_off[c] + local
					g2 = other_off[c] + local
					p1_conns = self.connections[self_conn_off[g1]:self_conn_off[g1 + 1]]
					p2_conns = other.connections[other_conn_off[g2]:other_conn_off[g2 + 1]]
					c1_need = c1_bits[self_off[c] + local]  # child1 has parent1's bits
					c2_need = c2_bits[other_off[c] + local]  # child2 has parent2's bits

					if rng.random() < 0.5:
						# child1 draws from parent2, child2 draws from parent1
						c1_conns.extend(self._fill_connections(p2_conns, p1_conns, c1_need))
						c2_conns.extend(self._fill_connections(p1_conns, p2_conns, c2_need))
					else:
						c1_conns.extend(self._fill_connections(p1_conns, p2_conns, c1_need))
						c2_conns.extend(self._fill_connections(p2_conns, p1_conns, c2_need))

				# Extra neurons inherit their own parent's connections
				for local in range(shared, p1_n):
					g1 = self_off[c] + local
					c1_conns.extend(self.connections[self_conn_off[g1]:self_conn_off[g1 + 1]])
				for local in range(shared, p2_n):
					g2 = other_off[c] + local
					c2_conns.extend(other.connections[other_conn_off[g2]:other_conn_off[g2 + 1]])

		child1 = ClusterGenome(bits_per_neuron=c1_bits, neurons_per_cluster=c1_neurons,
							   connections=c1_conns if c1_conns else None)
		child2 = ClusterGenome(bits_per_neuron=c2_bits, neurons_per_cluster=c2_neurons,
							   connections=c2_conns if c2_conns else None)
		return child1, child2

	@staticmethod
	def _fill_connections(source: list[int], supplement: list[int], need: int) -> list[int]:
		"""Fill connection slots from source, supplementing from supplement if needed. No random."""
		if need <= len(source):
			return list(source[:need])
		result = list(source)
		source_set = set(source)
		for c in supplement:
			if c not in source_set:
				result.append(c)
				source_set.add(c)
				if len(result) >= need:
					break
		# If still not enough (rare: both parents combined < need), cycle source
		while len(result) < need:
			result.append(source[len(result) % len(source)])
		return result[:need]

	def _crossover2_cluster(self, other: ClusterGenome, rng: random.Random) -> tuple[ClusterGenome, ClusterGenome]:
		"""Cluster-level crossover: per-cluster coin flip, entire cluster swapped as a unit."""
		num_clusters = len(self.neurons_per_cluster)
		self_off = self.cluster_neuron_offsets
		other_off = other.cluster_neuron_offsets
		self_conn_off = self.connection_offsets
		other_conn_off = other.connection_offsets

		c1_bits, c1_neurons, c1_conns = [], [], []
		c2_bits, c2_neurons, c2_conns = [], [], []
		has_conns = self.connections is not None and other.connections is not None

		for c in range(num_clusters):
			if rng.random() < 0.5:
				# child1 ← parent2's cluster, child2 ← parent1's cluster
				src1, src1_off, src1_conn_off = other, other_off, other_conn_off
				src2, src2_off, src2_conn_off = self, self_off, self_conn_off
			else:
				# child1 ← parent1's cluster, child2 ← parent2's cluster
				src1, src1_off, src1_conn_off = self, self_off, self_conn_off
				src2, src2_off, src2_conn_off = other, other_off, other_conn_off

			# child1
			n1 = src1.neurons_per_cluster[c]
			c1_neurons.append(n1)
			for local in range(n1):
				g = src1_off[c] + local
				c1_bits.append(src1.bits_per_neuron[g])
				if has_conns:
					c1_conns.extend(src1.connections[src1_conn_off[g]:src1_conn_off[g + 1]])

			# child2
			n2 = src2.neurons_per_cluster[c]
			c2_neurons.append(n2)
			for local in range(n2):
				g = src2_off[c] + local
				c2_bits.append(src2.bits_per_neuron[g])
				if has_conns:
					c2_conns.extend(src2.connections[src2_conn_off[g]:src2_conn_off[g + 1]])

		child1 = ClusterGenome(bits_per_neuron=c1_bits, neurons_per_cluster=c1_neurons,
							   connections=c1_conns if has_conns else None)
		child2 = ClusterGenome(bits_per_neuron=c2_bits, neurons_per_cluster=c2_neurons,
							   connections=c2_conns if has_conns else None)
		return child1, child2

	def mutate(
		self,
		phase_type: PhaseType,
		mutation_rate: float,
		config: AdaptiveClusterConfig,
		total_input_bits: int,
		rng: random.Random | None = None,
	) -> ClusterGenome:
		"""Phase-aware mutation dispatch."""
		if rng is None:
			rng = random.Random()
		match phase_type:
			case PhaseType.CLUSTER:
				# All-dimension mutation: neurons, then bits, then connections
				g = self._mutate_neurons(mutation_rate, config, total_input_bits, rng)
				g = g._mutate_bits(mutation_rate, config, total_input_bits, rng)
				return g._mutate_connections(mutation_rate, config, total_input_bits, rng)
			case PhaseType.NEURONS:
				return self._mutate_neurons(mutation_rate, config, total_input_bits, rng)
			case PhaseType.BITS:
				return self._mutate_bits(mutation_rate, config, total_input_bits, rng)
			case PhaseType.CONNECTIONS:
				return self._mutate_connections(mutation_rate, config, total_input_bits, rng)

	def _mutate_neurons(
		self, mutation_rate: float, config: AdaptiveClusterConfig,
		total_input_bits: int, rng: random.Random,
	) -> ClusterGenome:
		"""Neurons phase: add/remove neurons. Existing neurons keep bits + connections."""
		neurons_delta_max = max(1, round(0.1 * (config.min_neurons + config.max_neurons)))
		offsets = self.cluster_neuron_offsets
		conn_off = self.connection_offsets

		new_neurons = self.neurons_per_cluster.copy()
		for c in range(len(new_neurons)):
			if rng.random() < mutation_rate:
				delta = rng.randint(-neurons_delta_max, neurons_delta_max)
				new_neurons[c] = max(config.min_neurons, min(config.max_neurons, new_neurons[c] + delta))

		# Rebuild bits + connections
		new_bits = []
		new_conns = [] if self.connections is not None else None
		for c in range(len(new_neurons)):
			old_n = self.neurons_per_cluster[c]
			new_n = new_neurons[c]
			keep = min(old_n, new_n)

			# Mode bit size for new neurons
			cluster_bits = self.bits_per_neuron[offsets[c]:offsets[c + 1]]
			mode_bits = max(set(cluster_bits), key=cluster_bits.count) if cluster_bits else config.min_bits

			# Copy existing neurons verbatim
			for local in range(keep):
				g = offsets[c] + local
				new_bits.append(self.bits_per_neuron[g])
				if new_conns is not None:
					new_conns.extend(self.connections[conn_off[g]:conn_off[g + 1]])

			# Add new neurons with mode bits + random connections
			for _ in range(keep, new_n):
				new_bits.append(mode_bits)
				if new_conns is not None:
					for _ in range(mode_bits):
						new_conns.append(rng.randint(0, total_input_bits - 1))

		return ClusterGenome(bits_per_neuron=new_bits, neurons_per_cluster=new_neurons, connections=new_conns)

	def _mutate_bits(
		self, mutation_rate: float, config: AdaptiveClusterConfig,
		total_input_bits: int, rng: random.Random,
	) -> ClusterGenome:
		"""Bits phase: change bit counts per neuron. No drift on existing connections."""
		if config.max_bit_delta > 0:
			bits_delta_max = config.max_bit_delta
		else:
			bits_delta_max = max(1, round(0.1 * (config.min_bits + config.max_bits)))
		new_neurons = self.neurons_per_cluster.copy()  # unchanged
		new_bits = self.bits_per_neuron.copy()
		conn_off = self.connection_offsets

		# Mutate bit counts
		for n_idx in range(len(new_bits)):
			if rng.random() < mutation_rate:
				delta = rng.randint(-bits_delta_max, bits_delta_max)
				new_bits[n_idx] = max(config.min_bits, min(config.max_bits, new_bits[n_idx] + delta))

		# Rebuild connections
		new_conns = None
		if self.connections is not None:
			new_conns = []
			for n_idx in range(len(new_bits)):
				old_b = self.bits_per_neuron[n_idx]
				new_b = new_bits[n_idx]
				cs = conn_off[n_idx]

				if new_b == old_b:
					new_conns.extend(self.connections[cs:cs + old_b])
				elif new_b > old_b:
					# Keep all existing, add random for new bits
					new_conns.extend(self.connections[cs:cs + old_b])
					for _ in range(new_b - old_b):
						new_conns.append(rng.randint(0, total_input_bits - 1))
				else:
					# Fewer bits: randomly select which to keep (Fisher-Yates)
					indices = list(range(old_b))
					for i in range(new_b):
						j = rng.randint(i, old_b - 1)
						indices[i], indices[j] = indices[j], indices[i]
					kept = sorted(indices[:new_b])
					for idx in kept:
						new_conns.append(self.connections[cs + idx])

		return ClusterGenome(bits_per_neuron=new_bits, neurons_per_cluster=new_neurons, connections=new_conns)

	def _mutate_connections(
		self, mutation_rate: float, config: AdaptiveClusterConfig,
		total_input_bits: int, rng: random.Random,
	) -> ClusterGenome:
		"""Connections phase: perturb connection targets. Architecture unchanged."""
		new_conns = None
		if self.connections is not None:
			new_conns = self.connections.copy()
			for i in range(len(new_conns)):
				if rng.random() < mutation_rate:
					delta = rng.randint(-2, 2)
					new_conns[i] = max(0, min(total_input_bits - 1, new_conns[i] + delta))

		return ClusterGenome(
			bits_per_neuron=self.bits_per_neuron.copy(),
			neurons_per_cluster=self.neurons_per_cluster.copy(),
			connections=new_conns,
		)

	# =========================================================================
	# Properties and Utilities
	# =========================================================================

	@property
	def num_clusters(self) -> int:
		"""Number of clusters in this genome."""
		return len(self.neurons_per_cluster)

	@property
	def total_neurons(self) -> int:
		"""Total neurons across all clusters."""
		return sum(self.neurons_per_cluster)

	@property
	def cluster_neuron_offsets(self) -> list[int]:
		"""Cumulative neuron offsets per cluster: [0, n0, n0+n1, ...]."""
		offsets = [0]
		for n in self.neurons_per_cluster:
			offsets.append(offsets[-1] + n)
		return offsets

	@property
	def connection_offsets(self) -> list[int]:
		"""Cumulative connection offsets per neuron: [0, b0, b0+b1, ...]."""
		offsets = [0]
		for b in self.bits_per_neuron:
			offsets.append(offsets[-1] + b)
		return offsets

	def bits_for_cluster(self, cluster_idx: int) -> list[int]:
		"""Get per-neuron bits for a specific cluster."""
		offsets = self.cluster_neuron_offsets
		return self.bits_per_neuron[offsets[cluster_idx]:offsets[cluster_idx + 1]]

	def total_memory_cells(self) -> int:
		"""Calculate total memory cells needed for this genome."""
		return sum(2 ** b for b in self.bits_per_neuron)

	def total_connections(self) -> int:
		"""Total connection count across all neurons."""
		return sum(self.bits_per_neuron)

	def has_connections(self) -> bool:
		"""Check if this genome has initialized connections."""
		return self.connections is not None and len(self.connections) > 0

	def initialize_connections(self, total_input_bits: int, rng: Optional[int] = None) -> None:
		"""Initialize random connections for this genome."""
		self.connections = generate_connections(self.bits_per_neuron, total_input_bits, rng)

	def fingerprint(self) -> tuple:
		"""Identity tuple for deduplication — same genome = same eval result.

		Includes connections because they define which input bits each neuron
		observes. Same bits/neurons but different connections = different genome.
		"""
		conn = tuple(self.connections) if self.connections is not None else ()
		return (tuple(self.bits_per_neuron), tuple(self.neurons_per_cluster), conn)

	def clone(self) -> ClusterGenome:
		"""Create a deep copy of this genome including connections and cached fitness."""
		genome = ClusterGenome(
			bits_per_neuron=self.bits_per_neuron.copy(),
			neurons_per_cluster=self.neurons_per_cluster.copy(),
			connections=self.connections.copy() if self.connections is not None else None,
			threshold=self.threshold,
		)
		if hasattr(self, 'metrics') and self.metrics is not None:
			genome.metrics = self.metrics
		return genome

	def to_tensor(self) -> Tensor:
		"""Convert to tensor [num_clusters, 2] with (mean_bits, neurons) per cluster."""
		offsets = self.cluster_neuron_offsets
		data = []
		for i in range(self.num_clusters):
			cluster_bits = self.bits_per_neuron[offsets[i]:offsets[i + 1]]
			avg_bits = sum(cluster_bits) / len(cluster_bits) if cluster_bits else 0
			data.append((avg_bits, self.neurons_per_cluster[i]))
		return torch.tensor(data, dtype=torch.int32)

	def get_cluster_config(self, cluster_id: int) -> tuple:
		"""Get (neurons, bits_list) for a specific cluster."""
		return (self.neurons_per_cluster[cluster_id], self.bits_for_cluster(cluster_id))

	def stats(self) -> dict:
		"""Get statistics about this genome."""
		bits = self.bits_per_neuron
		neurons = self.neurons_per_cluster

		# Per-cluster breakdown
		offsets = self.cluster_neuron_offsets
		cluster_stats = []
		for i in range(len(neurons)):
			n = neurons[i]
			cb = bits[offsets[i]:offsets[i + 1]]
			connections = sum(cb)
			memory_cells = sum(2 ** b for b in cb)
			memory_words = (memory_cells + 30) // 31
			cluster_stats.append({
				"cluster": i,
				"min_bits": min(cb) if cb else 0,
				"max_bits": max(cb) if cb else 0,
				"mean_bits": sum(cb) / len(cb) if cb else 0,
				"neurons": n,
				"connections": connections,
				"memory_words": memory_words,
			})

		return {
			"num_clusters": len(neurons),
			# Per-neuron bits stats
			"min_bits": min(bits) if bits else 0,
			"max_bits": max(bits) if bits else 0,
			"mean_bits": sum(bits) / len(bits) if bits else 0,
			# Neurons stats
			"min_neurons": min(neurons) if neurons else 0,
			"max_neurons": max(neurons) if neurons else 0,
			"mean_neurons": sum(neurons) / len(neurons) if neurons else 0,
			"total_neurons": sum(neurons),
			# Connections stats
			"total_connections": self.total_connections(),
			# Memory stats
			"total_memory_cells": self.total_memory_cells(),
			# Distributions
			"bits_distribution": {
				b: bits.count(b) for b in sorted(set(bits))
			},
			"neurons_distribution": {
				n: neurons.count(n) for n in sorted(set(neurons))
			},
			# Per-cluster breakdown
			"cluster_stats": cluster_stats,
		}

	def compute_tier_stats(self, tier_config: list[tuple]) -> list[dict]:
		"""
		Compute per-tier statistics from genome configuration.

		Args:
			tier_config: List of (cluster_count, neurons, bits) tuples.
			             cluster_count=None means "rest".

		Returns:
			List of dicts with tier stats.
		"""
		if not tier_config:
			return []

		offsets = self.cluster_neuron_offsets
		tier_stats = []
		cluster_idx = 0

		for tier_num, tier in enumerate(tier_config):
			count = tier[0]
			if count is None:
				count = self.num_clusters - cluster_idx

			end_idx = min(cluster_idx + count, self.num_clusters)
			tier_neurons = self.neurons_per_cluster[cluster_idx:end_idx]

			if tier_neurons:
				# Gather all per-neuron bits in this tier
				neuron_start = offsets[cluster_idx]
				neuron_end = offsets[end_idx]
				tier_bits = self.bits_per_neuron[neuron_start:neuron_end]
				tier_connections = sum(tier_bits)
				tier_stats.append({
					"tier_index": tier_num,
					"cluster_count": end_idx - cluster_idx,
					"start_cluster": cluster_idx,
					"end_cluster": end_idx,
					"avg_bits": sum(tier_bits) / len(tier_bits) if tier_bits else 0,
					"avg_neurons": sum(tier_neurons) / len(tier_neurons),
					"min_bits": min(tier_bits) if tier_bits else 0,
					"max_bits": max(tier_bits) if tier_bits else 0,
					"min_neurons": min(tier_neurons),
					"max_neurons": max(tier_neurons),
					"total_neurons": sum(tier_neurons),
					"total_connections": tier_connections,
				})

			cluster_idx = end_idx

		return tier_stats

	def serialize(self) -> dict[str, Any]:
		"""Serialize genome to dictionary."""
		data: dict[str, Any] = {
			"bits_per_neuron": self.bits_per_neuron,
			"neurons_per_cluster": self.neurons_per_cluster,
		}
		if self.connections is not None:
			data["connections"] = self.connections
		if self.threshold != 0.5:
			data["threshold"] = self.threshold
		if hasattr(self, 'metrics') and self.metrics is not None:
			data["cached_metrics"] = self.metrics.to_dict()
		return data

	def to_dict(self) -> dict[str, Any]:
		"""Alias for serialize()."""
		return self.serialize()

	@classmethod
	def deserialize(cls, data: dict[str, Any]) -> 'ClusterGenome':
		"""Deserialize genome from dictionary."""
		genome = cls(
			bits_per_neuron=data["bits_per_neuron"],
			neurons_per_cluster=data["neurons_per_cluster"],
			connections=data.get("connections"),
			threshold=data.get("threshold", 0.5),
		)
		if "cached_metrics" in data:
			from wnn.ram.metrics import Metrics
			genome.metrics = Metrics.from_dict(data["cached_metrics"])
		elif "cached_fitness" in data:
			# Legacy: convert old (ce, acc[, f1, fpr]) tuple to Metrics
			from wnn.ram.metrics import Metrics
			cf = data["cached_fitness"]
			genome.metrics = Metrics(
				ce=cf[0], acc=cf[1],
				f1=cf[2] if len(cf) > 2 else None,
				fpr=cf[3] if len(cf) > 3 else None,
			)
		return genome

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> 'ClusterGenome':
		"""Alias for deserialize()."""
		return cls.deserialize(data)

	def save(
		self,
		filepath: str,
		fitness: Optional[float] = None,
		accuracy: Optional[float] = None,
		**metadata: Any,
	) -> None:
		"""
		Save genome to a compressed JSON file (.json.gz).

		Args:
			filepath: Output file path (auto-adds .gz if not present)
			fitness: Optional fitness (CE) value to include
			accuracy: Optional accuracy value to include
			**metadata: Additional metadata to include
		"""
		data: dict[str, Any] = {
			"genome": self.serialize(),
			"stats": self.stats(),
		}
		if fitness is not None:
			data["fitness"] = fitness
		if accuracy is not None:
			data["accuracy"] = accuracy
		if metadata:
			data["_metadata"] = metadata

		path = Path(filepath)
		# Auto-add .gz extension for compression
		if not path.suffix == '.gz':
			path = path.with_suffix(path.suffix + '.gz')
		path.parent.mkdir(parents=True, exist_ok=True)

		# Write compressed (no indent for better compression)
		with gzip.open(path, 'wt', encoding='utf-8') as f:
			json.dump(data, f, separators=(',', ':'))

	@classmethod
	def load(cls, filepath: str) -> tuple['ClusterGenome', dict[str, Any]]:
		"""
		Load genome from a JSON file (compressed or uncompressed).

		Args:
			filepath: Input file path

		Returns:
			Tuple of (genome, full_data) where full_data includes fitness, accuracy, metadata
		"""
		path = Path(filepath)

		# Try compressed first, then uncompressed
		if path.suffix == '.gz' or path.with_suffix(path.suffix + '.gz').exists():
			gz_path = path if path.suffix == '.gz' else path.with_suffix(path.suffix + '.gz')
			with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
				data = json.load(f)
		else:
			with open(path, 'r') as f:
				data = json.load(f)

		genome = cls.deserialize(data["genome"])
		return genome, data

	def enforce_budget(self, config: AdaptiveClusterConfig) -> ClusterGenome:
		"""Ensure genome stays within memory budget by shrinking per-neuron bits."""
		if self.total_memory_cells() <= config.total_memory_budget:
			return self

		bits = self.bits_per_neuron.copy()
		while sum(2 ** b for b in bits) > config.total_memory_budget:
			max_idx = max(range(len(bits)), key=lambda i: bits[i])
			if bits[max_idx] > config.min_bits:
				bits[max_idx] -= 1
			else:
				break

		return ClusterGenome(
			bits_per_neuron=bits,
			neurons_per_cluster=self.neurons_per_cluster.copy(),
		)

	def to_json_dict(self) -> dict:
		"""Serialize genome to a JSON-friendly dict (full per-neuron bits + connections).

		Stored in the genomes.tiers_json DB column so the genome can be reconstructed
		later without depending on the gzipped checkpoint files. Used for:
		- Per-class analysis on best_fpr / best_ce genomes that may not be in the
		  final-generation checkpoint.
		- Dashboard display of full genome details.
		- Reproducibility of any historical run.
		"""
		return {
			"bits_per_neuron": list(self.bits_per_neuron),
			"neurons_per_cluster": list(self.neurons_per_cluster),
			"threshold": float(self.threshold),
		}

	def __repr__(self) -> str:
		stats = self.stats()
		return (
			f"ClusterGenome(clusters={stats['num_clusters']}, "
			f"neurons={stats['total_neurons']}, "
			f"bits=[{stats['min_bits']}-{stats['max_bits']}], "
			f"memory={stats['total_memory_cells']:,})"
		)
