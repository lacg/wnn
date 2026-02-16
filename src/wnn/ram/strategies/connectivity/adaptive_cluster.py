"""
Adaptive Cluster Architecture Search — Per-Neuron Bits

Each neuron owns its synapse count (bit count) and evolves independently.
The GA mutates both connectivity AND structure (bits per neuron, neurons per cluster).

Key data structures:
- bits_per_neuron: [total_neurons] — each neuron's synapse count
- neurons_per_cluster: [num_clusters] — structural grouping
- connections: flat list, sum(bits_per_neuron) entries

Key insight: Frequent tokens need different architectures than rare tokens.
Let the data decide rather than hand-tuning tier boundaries.
"""

from __future__ import annotations

import gzip
import json
import math
import random

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch import Tensor

from wnn.progress import ProgressTracker
from wnn.ram.strategies.connectivity.generic_strategies import OptimizationLogger
from wnn.ram.strategies.factory import (
	OptimizerStrategyFactory,
	OptimizerStrategyType,
)
from wnn.ram.strategies.connectivity.generic_strategies import (
	GAConfig,
	TSConfig,
	OptimizerResult,
)

if TYPE_CHECKING:
	pass

# Try to import Rust accelerator for fast connection generation
try:
	import ram_accelerator as _accel
	_HAS_RUST = True
except ImportError:
	_HAS_RUST = False


def generate_connections(bits_per_neuron: list[int], total_input_bits: int, seed: int | None = None) -> list[int]:
	"""Generate random connections using Rust accelerator with numpy fallback."""
	if seed is None:
		seed = random.randint(0, 2**63)
	if _HAS_RUST:
		return _accel.generate_random_connections(bits_per_neuron, total_input_bits, seed)
	np_rng = np.random.default_rng(seed)
	return np_rng.integers(0, total_input_bits, size=sum(bits_per_neuron)).tolist()


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
	):
		"""
		Create a genome with specified architecture and connections.

		Args:
			bits_per_neuron: Bits per neuron [total_neurons]
			neurons_per_cluster: Neurons per cluster [num_clusters]
			connections: Flattened connection indices (default: None = not initialized)
		"""
		self.bits_per_neuron = bits_per_neuron
		self.neurons_per_cluster = neurons_per_cluster
		self.connections = connections

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
	# Genetic Operations
	# =========================================================================

	def mutate(
		self,
		config: AdaptiveClusterConfig,
		total_input_bits: Optional[int] = None,
		rng: Optional[int] = None,
	) -> ClusterGenome:
		"""
		Create a mutated copy of this genome.

		Per-neuron bit mutation + per-cluster neuron mutation.
		Connections adjusted when architecture changes.

		Args:
			config: Configuration with mutation rates and bounds
			total_input_bits: Total input bits (required if connections need adjustment)
			rng: Random seed

		Returns:
			New mutated genome (self unchanged)
		"""
		if rng is not None:
			random.seed(rng)

		new_bits = self.bits_per_neuron.copy()
		new_neurons = self.neurons_per_cluster.copy()
		old_neurons = self.neurons_per_cluster.copy()

		# Phase 1: Mutate bits per-neuron
		offsets = self.cluster_neuron_offsets
		for c in range(len(new_neurons)):
			for n_idx in range(offsets[c], offsets[c + 1]):
				if random.random() < config.bits_mutation_rate:
					delta = random.choice([-config.bits_mutation_step, config.bits_mutation_step])
					new_bits[n_idx] = max(config.min_bits, min(config.max_bits, new_bits[n_idx] + delta))

		# Phase 2: Mutate neuron count per-cluster
		if config.phase >= 2:
			for c in range(len(new_neurons)):
				if random.random() < config.neurons_mutation_rate:
					delta = random.choice([-config.neurons_mutation_step, config.neurons_mutation_step])
					old_n = new_neurons[c]
					new_n = max(config.min_neurons, min(config.max_neurons, old_n + delta))
					if new_n > old_n:
						# Add neurons: clone random existing neuron's bits
						for _ in range(new_n - old_n):
							template = random.randint(offsets[c], offsets[c + 1] - 1)
							# Insert at end of this cluster's neurons
							insert_pos = offsets[c + 1] + (new_neurons[c] - old_neurons[c])
							new_bits.insert(insert_pos, new_bits[template])
					elif new_n < old_n:
						# Remove neurons from end of cluster
						for _ in range(old_n - new_n):
							remove_pos = offsets[c + 1] - 1 + (new_neurons[c] - old_neurons[c])
							if 0 <= remove_pos < len(new_bits):
								new_bits.pop(remove_pos)
					new_neurons[c] = new_n

		# Adjust connections
		new_connections = None
		tib = total_input_bits or 64
		if self.connections is not None:
			new_connections = self._adjust_connections(
				new_bits, new_neurons, old_neurons, tib
			)
		elif total_input_bits is not None:
			new_connections = generate_connections(new_bits, tib)

		return ClusterGenome(
			bits_per_neuron=new_bits,
			neurons_per_cluster=new_neurons,
			connections=new_connections,
		)

	def _adjust_connections(
		self,
		new_bits: list[int],
		new_neurons: list[int],
		old_neurons: list[int],
		total_input_bits: int,
	) -> list[int]:
		"""Adjust connections when architecture changes (per-neuron)."""
		result = []
		old_conn_offsets = self.connection_offsets
		old_neuron_offsets = self.cluster_neuron_offsets

		new_neuron_idx = 0
		for c in range(len(new_neurons)):
			o_n = old_neurons[c]
			n_n = new_neurons[c]
			old_cluster_start = old_neuron_offsets[c]

			for local_n in range(n_n):
				n_bits = new_bits[new_neuron_idx]

				if local_n < o_n:
					# Existing neuron
					old_global_idx = old_cluster_start + local_n
					old_b = self.bits_per_neuron[old_global_idx]
					old_start = old_conn_offsets[old_global_idx]

					for bit_idx in range(n_bits):
						if bit_idx < old_b:
							old_conn = self.connections[old_start + bit_idx]
							if random.random() < 0.1:
								delta = random.choice([-2, -1, 1, 2])
								old_conn = max(0, min(total_input_bits - 1, old_conn + delta))
							result.append(old_conn)
						else:
							result.append(random.randint(0, total_input_bits - 1))
				else:
					# New neuron: clone from random existing in this cluster
					if o_n > 0:
						template_local = random.randint(0, o_n - 1)
						template_global = old_cluster_start + template_local
						old_b = self.bits_per_neuron[template_global]
						old_start = old_conn_offsets[template_global]
						for bit_idx in range(n_bits):
							if bit_idx < old_b:
								old_conn = self.connections[old_start + bit_idx]
								delta = random.choice([-2, -1, 1, 2])
								result.append(max(0, min(total_input_bits - 1, old_conn + delta)))
							else:
								result.append(random.randint(0, total_input_bits - 1))
					else:
						for _ in range(n_bits):
							result.append(random.randint(0, total_input_bits - 1))

				new_neuron_idx += 1

		return result

	def crossover(
		self,
		other: ClusterGenome,
		crossover_rate: float = 0.5,
		rng: Optional[int] = None,
	) -> ClusterGenome:
		"""
		Create child genome by crossing over with another parent.

		Crossover at cluster boundary: each cluster's neurons (with their
		per-neuron bits + connections) come from either parent.
		"""
		if rng is not None:
			random.seed(rng)

		num_clusters = len(self.neurons_per_cluster)
		child_bits = []
		child_neurons = []
		child_connections = [] if (self.connections is not None and other.connections is not None) else None

		self_offsets = self.cluster_neuron_offsets
		other_offsets = other.cluster_neuron_offsets
		self_conn_offsets = self.connection_offsets
		other_conn_offsets = other.connection_offsets

		for i in range(num_clusters):
			if random.random() < crossover_rate:
				# Take from other
				start = other_offsets[i]
				end = other_offsets[i + 1]
				child_bits.extend(other.bits_per_neuron[start:end])
				child_neurons.append(other.neurons_per_cluster[i])
				if child_connections is not None:
					conn_start = other_conn_offsets[start]
					conn_end = other_conn_offsets[end]
					child_connections.extend(other.connections[conn_start:conn_end])
			else:
				# Take from self
				start = self_offsets[i]
				end = self_offsets[i + 1]
				child_bits.extend(self.bits_per_neuron[start:end])
				child_neurons.append(self.neurons_per_cluster[i])
				if child_connections is not None:
					conn_start = self_conn_offsets[start]
					conn_end = self_conn_offsets[end]
					child_connections.extend(self.connections[conn_start:conn_end])

		return ClusterGenome(
			bits_per_neuron=child_bits,
			neurons_per_cluster=child_neurons,
			connections=child_connections,
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

	def clone(self) -> ClusterGenome:
		"""Create a deep copy of this genome including connections and cached fitness."""
		genome = ClusterGenome(
			bits_per_neuron=self.bits_per_neuron.copy(),
			neurons_per_cluster=self.neurons_per_cluster.copy(),
			connections=self.connections.copy() if self.connections is not None else None,
		)
		if hasattr(self, '_cached_fitness') and self._cached_fitness is not None:
			genome._cached_fitness = self._cached_fitness
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
		if hasattr(self, '_cached_fitness') and self._cached_fitness is not None:
			data["cached_fitness"] = self._cached_fitness
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
		)
		if "cached_fitness" in data:
			genome._cached_fitness = tuple(data["cached_fitness"])
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

	def __repr__(self) -> str:
		stats = self.stats()
		return (
			f"ClusterGenome(clusters={stats['num_clusters']}, "
			f"neurons={stats['total_neurons']}, "
			f"bits=[{stats['min_bits']}-{stats['max_bits']}], "
			f"memory={stats['total_memory_cells']:,})"
		)


# =============================================================================
# Helper Functions (module-level, internal)
# =============================================================================

def _frequency_scaled_init(
	num_clusters: int,
	token_frequencies: list[int],
	config: AdaptiveClusterConfig,
	for_bits: bool = True,
) -> list[int]:
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
# Rust Parallel Evaluator
# =============================================================================

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
		# Cap to actual unique tokens in case vocab is smaller than global_top_k
		actual_top_k = min(cfg.global_top_k, len(counts))
		top_k_tokens = np.array([t for t, _ in counts.most_common(actual_top_k)], dtype=np.int64)
		num_negatives = min(5, actual_top_k)

		rng = np.random.RandomState(42)
		neg_indices = rng.randint(0, actual_top_k, (n_train, num_negatives))
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
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
		batch_size: int = 1,  # Sequential: each genome gets full thread pool for token parallelism
		generation: Optional[int] = None,  # Current generation for logging
		total_generations: Optional[int] = None,  # Total generations for logging
		min_accuracy: Optional[float] = None,  # Threshold for log level selection
	) -> list[tuple[float, float]]:
		"""
		Evaluate multiple genomes using Rust/rayon.

		Rust evaluates genomes SEQUENTIALLY (memory-safe) while each genome's
		training/eval uses full CPU parallelism. This avoids the memory explosion
		that occurred with parallel genome evaluation.

		Args:
			genomes: List of genomes to evaluate
			logger: Optional logging function
			batch_size: Genomes per Rust call (1 = per-genome logging, >1 = batch logging)
			generation: Current generation number for logging (None = initial population)
			total_generations: Total number of generations for logging context
			min_accuracy: If provided, genomes below this threshold log at TRACE level

		Returns:
			List of (cross-entropy, accuracy) tuples for each genome
		"""
		import ram_accelerator
		import time

		# Use OptimizationLogger for leveled logging, or fallback to callable
		if isinstance(logger, OptimizationLogger):
			log_debug = logger.debug
			log_trace = logger.trace
		elif logger is not None:
			log_debug = logger
			log_trace = logger  # Fallback: no level distinction
		else:
			log_debug = lambda x: None
			log_trace = lambda x: None

		# Prepare data on first call
		self._prepare_data()

		all_fitness = []
		total_genomes = len(genomes)
		genome_width = len(str(total_genomes))  # For zero-padded logging
		start_time = time.time()

		# Generation prefix for logs
		if generation is not None:
			if total_generations is not None:
				gen_width = len(str(total_generations))
				gen_prefix = f"[Gen {generation + 1:0{gen_width}d}/{total_generations}]"
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
			genomes_connections_flat = []
			for g in batch_genomes:
				genomes_bits_flat.extend(g.bits_per_neuron)
				genomes_neurons_flat.extend(g.neurons_per_cluster)
				# Include connections if available (for connection-preserving search)
				if g.connections is not None:
					genomes_connections_flat.extend(g.connections)

			# Call Rust parallel evaluator for this batch
			# Returns list of (CE, accuracy) tuples
			batch_results = ram_accelerator.evaluate_genomes_parallel(
				genomes_bits_flat,
				genomes_neurons_flat,
				genomes_connections_flat,  # Pass connections (empty = random fallback)
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

			# batch_results is already list[(CE, accuracy)]
			all_fitness.extend(batch_results)

			elapsed = time.time() - start_time
			# Log each genome in the batch (with batch timing for efficiency)
			for i, (ce, acc) in enumerate(batch_results):
				genome_idx = batch_start + i + 1
				if batch_size == 1:
					msg = f"{gen_prefix} Genome {genome_idx:0{genome_width}d}/{total_genomes}: CE={ce:.4f}, Acc={acc:.2%} in {elapsed:.1f}s"
				else:
					# Parallel batch: show genome results without individual timing
					msg = f"{gen_prefix} Genome {genome_idx:0{genome_width}d}/{total_genomes}: CE={ce:.4f}, Acc={acc:.2%}"
				# Use TRACE for filtered (below threshold), DEBUG for passed
				if min_accuracy is not None and acc < min_accuracy:
					log_trace(msg)  # Filtered candidate
				else:
					log_debug(msg)  # Passed candidate
			# Log batch timing summary for parallel batches
			if batch_size > 1:
				log_debug(f"{gen_prefix} Batch {batch_start//batch_size + 1}: {len(batch_results)} genomes in {elapsed:.1f}s")

		return all_fitness

	def evaluate_single(self, genome: ClusterGenome) -> float:
		"""Evaluate a single genome, returning CE only."""
		ce, _ = self.evaluate_batch([genome])[0]
		return ce

	def evaluate_single_with_accuracy(self, genome: ClusterGenome) -> tuple[float, float]:
		"""Evaluate a single genome, returning (CE, accuracy)."""
		return self.evaluate_batch([genome])[0]


# =============================================================================
# Genome Evaluation Wrapper
# =============================================================================

@dataclass
class EvaluatorConfig:
	"""Configuration for genome evaluation."""

	# Data
	train_tokens: list[int] = None
	eval_tokens: list[int] = None
	vocab_size: int = 50257
	context_size: int = 4

	# Training
	batch_size: int = 500
	global_top_k: int = 100
	empty_value: float = 0.0

	# Evaluation
	eval_batch_size: int = 1000

	# Token ordering (for cluster assignment)
	cluster_order: Optional[list[int]] = None  # sorted by frequency

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

	def _encode_tokens(self, tokens: list[int]) -> Tensor:
		"""Encode tokens to binary input bits."""
		from torch import tensor, zeros, bool as torch_bool

		n = len(tokens)
		bits = zeros(n, self.bits_per_token, dtype=torch_bool)
		tokens_t = tensor(tokens, dtype=self._bit_positions.dtype)

		for i in range(self.bits_per_token):
			bits[:, i] = ((tokens_t >> self._bit_positions[i]) & 1).bool()

		return bits

	def _encode_context(self, context_tokens: list[int]) -> Tensor:
		"""Encode context tokens to flat input bits."""
		bits = self._encode_tokens(context_tokens)
		return bits.flatten()

	def train_epoch(
		self,
		tokens: list[int],
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
		tokens: list[int],
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


def evaluate_genome_with_accuracy(
	genome: ClusterGenome,
	train_tokens: list[int],
	eval_tokens: list[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	cluster_order: Optional[list[int]] = None,
	global_top_k: int = 1000,
	logger: Optional[Callable[[str], None]] = None,
) -> tuple[float, float]:
	"""
	Evaluate a genome and return (cross_entropy, accuracy).

	This is for checkpoint evaluation where we need accuracy in addition to CE.
	Slower than Rust evaluation but provides full metrics.

	Args:
		genome: ClusterGenome to evaluate
		train_tokens: Training token sequence
		eval_tokens: Evaluation token sequence
		vocab_size: Vocabulary size
		context_size: Context window size
		cluster_order: Token-to-cluster mapping order
		global_top_k: Top-k tokens for clustering
		logger: Optional logging function

	Returns:
		Tuple of (cross_entropy, accuracy)
	"""
	# Build config (EvaluatorConfig is defined in this file)
	config = EvaluatorConfig(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		cluster_order=cluster_order,
		cluster_config=genome.cluster_config if genome.cluster_config else AdaptiveClusterConfig(),
		global_top_k=global_top_k,
	)

	# Create wrapper and train
	wrapper = AdaptiveRAMLMWrapper(genome, config)
	wrapper.train_epoch(train_tokens, global_top_k=global_top_k)

	# Evaluate with full metrics
	stats = wrapper.evaluate(eval_tokens)

	if logger:
		# Use DEBUG level if OptimizationLogger, otherwise call directly
		if isinstance(logger, OptimizationLogger):
			logger.debug(f"  Checkpoint eval: CE={stats['cross_entropy']:.4f}, Acc={stats['accuracy']:.2%}")
		else:
			logger(f"  Checkpoint eval: CE={stats['cross_entropy']:.4f}, Acc={stats['accuracy']:.2%}")

	return stats['cross_entropy'], stats['accuracy']


# =============================================================================
# High-Level API Functions
# =============================================================================

def run_architecture_tabu_search(
	initial_genome: ClusterGenome,
	initial_fitness: float,
	train_tokens: list[int],
	eval_tokens: list[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	cluster_order: Optional[list[int]] = None,
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
	seed: Optional[int] = None,  # None = time-based
	logger: Optional[Callable[[str], None]] = None,
	# Population seeding from previous phase
	initial_neighbors: Optional[list[ClusterGenome]] = None,
) -> OptimizerResult['ClusterGenome']:
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
		initial_neighbors: Optional seed neighbors from Phase 1a population

	Returns:
		OptimizerResult with refined genome
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
		log("[ArchitectureTS] Using Rust parallel evaluator")
	except Exception as e:
		log(f"[ArchitectureTS] Using Python evaluator ({e})")

	# Compute total input bits for connection preservation
	from wnn.ram.core import bits_needed
	bits_per_token = bits_needed(vocab_size)
	total_input_bits = context_size * bits_per_token
	log(f"[ArchitectureTS] Connection-preserving search enabled ({total_input_bits} input bits)")

	# Create TS strategy using factory
	strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_TS,
		num_clusters=vocab_size,
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
		total_input_bits=total_input_bits,  # Enable connection preservation
		# TS parameters
		iterations=iterations,
		neighbors_per_iter=neighbors_per_iter,
		patience=patience,
		seed=seed,
		logger=log,
		batch_evaluator=batch_evaluator,
	)

	# Create batch evaluation function using Rust evaluator (returns list[(CE, accuracy)])
	batch_evaluate_fn = None
	if batch_evaluator is not None:
		batch_evaluate_fn = lambda genomes, min_accuracy=None: batch_evaluator.evaluate_batch(genomes, logger=log, min_accuracy=min_accuracy)

	# Run optimization
	result = strategy.optimize(
		initial_genome=initial_genome,
		initial_fitness=initial_fitness,
		evaluate_fn=evaluate_fn,
		initial_neighbors=initial_neighbors,
		batch_evaluate_fn=batch_evaluate_fn,
	)

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
	train_tokens: list[int],
	eval_tokens: list[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	token_frequencies: Optional[list[int]] = None,
	cluster_order: Optional[list[int]] = None,
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
	seed: Optional[int] = None,  # None = time-based
	logger: Optional[Callable[[str], None]] = None,
	# Population seeding from previous phase
	initial_population: Optional[list[ClusterGenome]] = None,
) -> OptimizerResult['ClusterGenome']:
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
		OptimizerResult with best genome and optimization history

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
		log("[ArchitectureGA] Using Rust parallel evaluator (hybrid dense/sparse)")
	except ImportError:
		log("[ArchitectureGA] Rust accelerator not available, using Python sequential")
	except Exception as e:
		log(f"[ArchitectureGA] Warning: Rust evaluator init failed ({e}), using Python")

	# Compute total input bits for connection initialization
	from wnn.ram.core import bits_needed
	bits_per_token = bits_needed(vocab_size)
	total_input_bits = context_size * bits_per_token
	log(f"[ArchitectureGA] Connection-preserving search enabled ({total_input_bits} input bits)")

	# Create GA strategy using factory
	strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_GA,
		num_clusters=vocab_size,
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,  # Enable connection preservation
		# GA parameters
		population_size=population_size,
		generations=generations,
		patience=patience,
		seed=seed,
		logger=log,
		batch_evaluator=batch_evaluator,
	)

	# Create batch evaluation function using Rust evaluator (returns list[(CE, accuracy)])
	batch_evaluate_fn = None
	if batch_evaluator is not None:
		batch_evaluate_fn = lambda genomes, min_accuracy=None: batch_evaluator.evaluate_batch(genomes, logger=log, min_accuracy=min_accuracy)

	# Run optimization
	log()
	result = strategy.optimize(
		evaluate_fn=evaluate_fn,
		initial_population=initial_population,
		batch_evaluate_fn=batch_evaluate_fn,
	)

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
	log(f"  Generations: {result.iterations_run}")
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
	"""Result from connectivity optimization (Phase 2 GA→TS)."""

	initial_fitness: float  # Phase 1 baseline
	phase2_baseline: float  # Fitness after Phase 2a GA
	final_fitness: float    # Fitness after Phase 2b TS
	ga_improvement_pct: float  # Improvement from Phase 2a GA
	ts_improvement_pct: float  # Improvement from Phase 2b TS
	total_improvement_pct: float  # Total improvement vs Phase 1 baseline
	ga_iterations: int
	ts_iterations: int
	early_stopped: bool
	initial_accuracy: Optional[float] = None   # Accuracy at Phase 2 start
	ga_final_accuracy: Optional[float] = None  # Accuracy after Phase 2a GA
	final_accuracy: Optional[float] = None     # Accuracy after Phase 2b TS
	# Population seeding for potential future phases
	final_population: Optional[list[ClusterGenome]] = None  # From Phase 2 TS


def run_connectivity_optimization(
	genome: ClusterGenome,
	genome_fitness: float,
	train_tokens: list[int],
	eval_tokens: list[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	cluster_order: Optional[list[int]] = None,
	token_frequencies: Optional[list[int]] = None,
	# GA parameters
	ga_population: int = 20,
	ga_generations: int = 30,
	ga_patience: int = 5,
	# TS parameters
	ts_iterations: int = 50,
	ts_neighbors: int = 30,
	ts_patience: int = 5,
	# Architecture bounds (same as Phase 1)
	min_bits: int = 8,
	max_bits: int = 25,
	min_neurons: int = 3,
	max_neurons: int = 33,
	phase: int = 2,
	# Other
	empty_value: float = 0.0,
	seed: Optional[int] = None,  # None = time-based
	logger: Optional[Callable[[str], None]] = None,
	# Population seeding from Phase 1b
	initial_population: Optional[list[ClusterGenome]] = None,
) -> ConnectivityOptResult:
	"""
	Run Phase 2: Continue architecture optimization with GA→TS.

	Continues optimizing architecture from Phase 1b using GA followed by TS.
	Each evaluation uses random connectivity, so this effectively finds
	architectures robust to connectivity variations.

	The pipeline:
	1. Phase 2a (GA): Evolve architecture with initial_population from Phase 1b
	2. Phase 2b (TS): Refine best GA solution with GA's population as neighbors

	Args:
		genome: Best architecture from Phase 1b (used if no initial_population)
		genome_fitness: Baseline fitness from Phase 1b
		train_tokens: Training data
		eval_tokens: Evaluation data
		vocab_size: Vocabulary size
		context_size: Context window
		cluster_order: Token ordering by frequency
		token_frequencies: Token occurrence counts (for genome generation)
		ga_population: GA population size
		ga_generations: GA generations
		ga_patience: GA early stop patience
		ts_iterations: TS iterations
		ts_neighbors: TS neighbors per iteration
		ts_patience: TS early stop patience
		min_bits, max_bits: Bits bounds
		min_neurons, max_neurons: Neurons bounds
		phase: Optimization phase (1=bits only, 2=bits+neurons)
		empty_value: EMPTY cell value
		seed: Random seed
		logger: Logging function
		initial_population: Seed population from Phase 1b's final_neighbors

	Returns:
		ConnectivityOptResult with optimization statistics and final_population
	"""
	log = logger or print

	log()
	log("=" * 60)
	log("  Phase 2: Architecture Refinement (GA→TS)")
	log("=" * 60)
	log(f"  Phase 1b fitness: {genome_fitness:.4f}")
	log(f"  Phase 2a (GA): pop={ga_population}, gens={ga_generations}")
	log(f"  Phase 2b (TS): iters={ts_iterations}, neighbors={ts_neighbors}")
	if initial_population:
		log(f"  Seed population: {len(initial_population)} genomes from Phase 1b")
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
	)

	# Create evaluation function
	evaluate_fn = create_genome_evaluator(eval_config, verbose=False)

	# Create Rust parallel evaluator
	batch_evaluator = None
	try:
		batch_evaluator = RustParallelEvaluator(eval_config)
		log("[Phase2] Using Rust parallel evaluator")
	except Exception as e:
		log(f"[Phase2] Using Python evaluator ({e})")

	# Create batch evaluation function using Rust evaluator (returns list[(CE, accuracy)])
	batch_evaluate_fn = None
	if batch_evaluator is not None:
		batch_evaluate_fn = lambda genomes, min_accuracy=None: batch_evaluator.evaluate_batch(genomes, logger=log, min_accuracy=min_accuracy)

	# Compute total input bits for connection preservation
	from wnn.ram.core import bits_needed
	bits_per_token = bits_needed(vocab_size)
	total_input_bits = context_size * bits_per_token
	log(f"[Phase2] Connection-preserving search enabled ({total_input_bits} input bits)")

	# =========================================================================
	# Phase 2a: GA
	# =========================================================================
	log()
	log("-" * 40)
	log("  Phase 2a: GA Architecture Refinement")
	log("-" * 40)

	# Create GA strategy using factory
	ga_strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_GA,
		num_clusters=vocab_size,
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,  # Enable connection preservation
		# GA parameters
		population_size=ga_population,
		generations=ga_generations,
		patience=ga_patience,
		seed=seed,
		logger=log,
		batch_evaluator=batch_evaluator,
	)

	# Run GA with seeded population from Phase 1b
	ga_result = ga_strategy.optimize(
		evaluate_fn=evaluate_fn,
		initial_population=initial_population,
		batch_evaluate_fn=batch_evaluate_fn,
	)

	log()
	log(f"[Phase2a] GA complete: {ga_result.final_fitness:.4f} "
		f"({(1 - ga_result.final_fitness / genome_fitness) * 100:.2f}% vs Phase 1b)")

	# =========================================================================
	# Phase 2b: TS
	# =========================================================================
	log()
	log("-" * 40)
	log("  Phase 2b: TS Architecture Refinement")
	log("-" * 40)

	# Create TS strategy using factory
	ts_strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_TS,
		num_clusters=vocab_size,
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
		total_input_bits=total_input_bits,  # Enable connection preservation
		# TS parameters
		iterations=ts_iterations,
		neighbors_per_iter=ts_neighbors,
		patience=ts_patience,
		seed=(seed + 500) if seed is not None else None,
		logger=log,
		batch_evaluator=batch_evaluator,
	)

	# Run TS with GA's population as initial neighbors
	ts_result = ts_strategy.optimize(
		initial_genome=ga_result.best_genome,
		initial_fitness=ga_result.final_fitness,
		evaluate_fn=evaluate_fn,
		initial_neighbors=ga_result.final_population,
		batch_evaluate_fn=batch_evaluate_fn,
	)

	log()
	log(f"[Phase2b] TS complete: {ts_result.final_fitness:.4f} "
		f"({(1 - ts_result.final_fitness / ga_result.final_fitness) * 100:.2f}% vs Phase 2a)")

	# =========================================================================
	# Summary
	# =========================================================================
	ga_improvement = (genome_fitness - ga_result.final_fitness) / genome_fitness * 100 if genome_fitness > 0 else 0
	ts_improvement = (ga_result.final_fitness - ts_result.final_fitness) / ga_result.final_fitness * 100 if ga_result.final_fitness > 0 else 0
	total_improvement = (genome_fitness - ts_result.final_fitness) / genome_fitness * 100 if genome_fitness > 0 else 0

	log()
	log("=" * 60)
	log("  Phase 2 Complete")
	log("=" * 60)
	log(f"  Phase 1b baseline: {genome_fitness:.4f}")
	log(f"  After Phase 2a (GA): {ga_result.final_fitness:.4f} ({ga_improvement:.2f}% improvement)")
	log(f"  After Phase 2b (TS): {ts_result.final_fitness:.4f} ({ts_improvement:.2f}% improvement)")
	log(f"  Total Phase 2 improvement: {total_improvement:.2f}%")
	log()
	stats = ts_result.best_genome.stats()
	log("  Best genome:")
	log(f"    Bits: [{stats['min_bits']}, {stats['max_bits']}], mean: {stats['mean_bits']:.1f}")
	log(f"    Neurons: [{stats['min_neurons']}, {stats['max_neurons']}], mean: {stats['mean_neurons']:.1f}")

	return ConnectivityOptResult(
		initial_fitness=genome_fitness,
		phase2_baseline=ga_result.final_fitness,
		final_fitness=ts_result.final_fitness,
		ga_improvement_pct=ga_improvement,
		ts_improvement_pct=ts_improvement,
		total_improvement_pct=total_improvement,
		ga_iterations=ga_result.iterations_run,
		ts_iterations=ts_result.iterations_run,
		early_stopped=ga_result.early_stopped or ts_result.early_stopped,
		initial_accuracy=ga_result.initial_accuracy,
		ga_final_accuracy=ga_result.final_accuracy,  # After Phase 2a GA
		final_accuracy=ts_result.final_accuracy,     # After Phase 2b TS
		final_population=ts_result.final_population,  # For potential Phase 3
	)
