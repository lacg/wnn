"""
AdaptiveClusteredRAM - RAM Layer with per-cluster architecture.

Unlike TieredRAMClusterLayer where all clusters in a tier share the same (neurons, bits),
this layer allows EACH cluster to have its own architecture. The architecture is defined
by a ClusterGenome from adaptive_cluster.py.

Key optimization: Clusters are grouped by their (neurons, bits) configuration to share
Memory objects, making it memory-efficient even with 50K clusters.

Example:
    genome = ClusterGenome(
        bits_per_cluster=[20, 20, 18, 18, ..., 8, 8],      # 50K values
        neurons_per_cluster=[15, 15, 13, 13, ..., 5, 5],   # 50K values
    )
    layer = AdaptiveClusteredRAM.from_genome(genome, total_input_bits=64)
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from torch import zeros, ones, arange, long, Tensor, bool as torch_bool

from wnn.ram.core.Memory import Memory
from wnn.ram.core.base import RAMComponent
from wnn.ram.core import MemoryVal
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


@dataclass
class ConfigGroup:
	"""A group of clusters sharing the same (neurons, bits) configuration."""

	neurons: int
	bits: int
	cluster_ids: List[int]  # Physical cluster IDs in this group
	memory: Optional[Memory] = None

	@property
	def cluster_count(self) -> int:
		return len(self.cluster_ids)

	@property
	def total_neurons(self) -> int:
		return self.cluster_count * self.neurons

	@property
	def memory_size(self) -> int:
		return 2 ** self.bits


class AdaptiveClusteredRAM(RAMComponent):
	"""
	RAM Layer with per-cluster (neurons, bits) architecture.

	Each cluster can have a unique (neurons, bits) configuration, as specified
	by a ClusterGenome. Internally, clusters are grouped by configuration for
	memory efficiency.

	Attributes:
		num_clusters: Total number of output clusters (vocabulary size)
		genome: The ClusterGenome defining per-cluster architecture
		config_groups: List of ConfigGroup objects (clusters with same config)
		total_neurons: Total neurons across all clusters
	"""

	def __init__(
		self,
		total_input_bits: int,
		genome: ClusterGenome,
		empty_value: float = 0.0,
		rng: Optional[int] = None,
	):
		"""
		Initialize AdaptiveClusteredRAM.

		Args:
			total_input_bits: Number of input bits (e.g., 64 for 4 tokens × 16 bits)
			genome: ClusterGenome with bits_per_cluster and neurons_per_cluster
			empty_value: Value for EMPTY cells (0.0, 0.5, etc.)
			rng: Random seed for reproducible connectivity initialization
		"""
		super().__init__()

		self.num_clusters = len(genome.bits_per_cluster)
		self.total_input_bits = total_input_bits
		self.genome = genome
		self.empty_value = empty_value

		# Group clusters by their (neurons, bits) configuration
		# config_key -> list of cluster_ids
		config_to_clusters: Dict[Tuple[int, int], List[int]] = {}

		for cluster_id in range(self.num_clusters):
			neurons = genome.neurons_per_cluster[cluster_id]
			bits = genome.bits_per_cluster[cluster_id]
			key = (neurons, bits)

			if key not in config_to_clusters:
				config_to_clusters[key] = []
			config_to_clusters[key].append(cluster_id)

		# Create ConfigGroups and Memory objects
		self.config_groups: List[ConfigGroup] = []
		self._cluster_to_group: List[Tuple[int, int]] = [None] * self.num_clusters

		group_idx = 0
		for (neurons, bits), cluster_ids in sorted(config_to_clusters.items()):
			# Create Memory for this group
			memory = Memory(
				total_input_bits=total_input_bits,
				num_neurons=len(cluster_ids) * neurons,
				n_bits_per_neuron=bits,
				rng=rng + group_idx if rng is not None else None,
			)

			group = ConfigGroup(
				neurons=neurons,
				bits=bits,
				cluster_ids=cluster_ids,
				memory=memory,
			)
			self.config_groups.append(group)

			# Build reverse mapping: cluster_id -> (group_idx, local_idx)
			for local_idx, cluster_id in enumerate(cluster_ids):
				self._cluster_to_group[cluster_id] = (group_idx, local_idx)

			group_idx += 1

		# Compute totals
		self.total_neurons = genome.total_neurons()
		self.total_memory_cells = genome.total_memory_cells()

		# Check if any group needs sparse backend (>10 bits)
		self._use_sparse = any(g.bits > 10 for g in self.config_groups)
		self._sparse_memories: Dict[int, object] = {}  # group_idx -> TieredSparseMemory

		if self._use_sparse:
			self._init_sparse_backends()

	def _init_sparse_backends(self) -> None:
		"""Initialize Rust sparse memory for groups with >10 bits."""
		try:
			import ram_accelerator
			for group_idx, group in enumerate(self.config_groups):
				if group.bits > 10:
					# Create sparse memory for this group
					# Each group acts like a single-tier model
					tier_configs = [(group.cluster_count, group.neurons, group.bits)]
					self._sparse_memories[group_idx] = ram_accelerator.TieredSparseMemory(
						tier_configs, group.cluster_count
					)
		except (ImportError, AttributeError):
			self._use_sparse = False
			self._sparse_memories = {}

	@classmethod
	def from_genome(
		cls,
		genome: ClusterGenome,
		total_input_bits: int,
		empty_value: float = 0.0,
		rng: Optional[int] = None,
	) -> "AdaptiveClusteredRAM":
		"""
		Factory method to create AdaptiveClusteredRAM from a genome.

		Args:
			genome: ClusterGenome with per-cluster architecture
			total_input_bits: Number of input bits
			empty_value: Value for EMPTY cells
			rng: Random seed

		Returns:
			Configured AdaptiveClusteredRAM instance
		"""
		return cls(
			total_input_bits=total_input_bits,
			genome=genome,
			empty_value=empty_value,
			rng=rng,
		)

	def __repr__(self) -> str:
		stats = self.genome.stats()
		return (
			f"AdaptiveClusteredRAM("
			f"clusters={self.num_clusters}, "
			f"groups={len(self.config_groups)}, "
			f"total_neurons={self.total_neurons}, "
			f"bits=[{stats['min_bits']}-{stats['max_bits']}], "
			f"neurons=[{stats['min_neurons']}-{stats['max_neurons']}])"
		)

	def __str__(self) -> str:
		stats = self.genome.stats()
		lines = [
			"=== AdaptiveClusteredRAM ===",
			f"  Total clusters: {self.num_clusters:,}",
			f"  Config groups: {len(self.config_groups)}",
			f"  Total neurons: {self.total_neurons:,}",
			f"  Total memory cells: {self.total_memory_cells:,}",
			f"  Input bits: {self.total_input_bits}",
			f"  Empty value: {self.empty_value}",
			"",
			"  Architecture range:",
			f"    Bits: [{stats['min_bits']}, {stats['max_bits']}] (mean: {stats['mean_bits']:.1f})",
			f"    Neurons: [{stats['min_neurons']}, {stats['max_neurons']}] (mean: {stats['mean_neurons']:.1f})",
			"",
			f"  Top config groups (by cluster count):",
		]

		# Show top 5 config groups
		sorted_groups = sorted(self.config_groups, key=lambda g: -g.cluster_count)
		for i, group in enumerate(sorted_groups[:5]):
			lines.append(
				f"    {i+1}. {group.cluster_count:,} clusters × "
				f"{group.neurons} neurons × {group.bits} bits"
			)

		return "\n".join(lines)

	@property
	def use_sparse_backend(self) -> bool:
		"""Check if using sparse memory backend for any group."""
		return self._use_sparse and len(self._sparse_memories) > 0

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass returning probabilities per cluster.

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			[batch, num_clusters] float tensor of probabilities in [0, 1]
		"""
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]
		device = input_bits.device

		# Output tensor for all clusters
		probs = zeros(batch_size, self.num_clusters, device=device)

		# Process each config group
		for group_idx, group in enumerate(self.config_groups):
			# Check if using sparse backend for this group
			if group_idx in self._sparse_memories:
				group_probs = self._forward_sparse_group(input_bits, group_idx)
			else:
				group_probs = self._forward_dense_group(input_bits, group_idx)

			# Scatter to output positions
			for local_idx, cluster_id in enumerate(group.cluster_ids):
				probs[:, cluster_id] = group_probs[:, local_idx]

		return probs

	def _forward_dense_group(self, input_bits: Tensor, group_idx: int) -> Tensor:
		"""Dense forward for a single config group."""
		group = self.config_groups[group_idx]
		memory = group.memory
		batch_size = input_bits.shape[0]
		device = input_bits.device

		# Get raw memory values: [batch, group_neurons]
		raw = memory.get_memories_for_bits(input_bits)

		# Reshape to clusters: [batch, group_clusters, neurons_per_cluster]
		clustered = raw.view(batch_size, group.cluster_count, group.neurons)

		# Compute probabilities with configurable EMPTY value
		count_true = (clustered == MemoryVal.TRUE).sum(dim=-1).float()
		count_empty = (clustered == MemoryVal.EMPTY).sum(dim=-1).float()

		# prob = (count_true + empty_value * count_empty) / neurons
		group_probs = (count_true + self.empty_value * count_empty) / group.neurons

		return group_probs

	def _forward_sparse_group(self, input_bits: Tensor, group_idx: int) -> Tensor:
		"""Sparse forward for a single config group (using Rust backend)."""
		import ram_accelerator
		import numpy as np
		from torch import from_numpy

		group = self.config_groups[group_idx]
		sparse_memory = self._sparse_memories[group_idx]
		batch_size = input_bits.shape[0]

		# Flatten inputs to numpy
		input_bits_np = input_bits.flatten().bool().numpy().astype(np.uint8)

		# Get connections for this group
		connections_np = group.memory.connections.flatten().numpy().astype(np.int64)

		# Call Rust sparse forward
		probs_np = ram_accelerator.sparse_forward_batch_tiered_numpy(
			sparse_memory,
			input_bits_np,
			connections_np,
			batch_size,
			self.total_input_bits,
		)

		return from_numpy(probs_np).view(batch_size, group.cluster_count)

	def train_batch(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Optional[Tensor] = None,
		allow_override: bool = False,
	) -> int:
		"""
		Batch training: train TRUE for target clusters, FALSE for negative clusters.

		Routes each cluster to its appropriate config group for training.

		Args:
			input_bits: [batch, total_input_bits] input patterns
			true_clusters: [batch] physical cluster indices to train as TRUE
			false_clusters: Optional [batch, k] physical cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		batch_size = input_bits.shape[0]
		device = input_bits.device
		modified = 0

		# Group training data by config group
		# true_by_group[group_idx] = [(batch_idx, local_cluster), ...]
		true_by_group: List[List[Tuple[int, int]]] = [[] for _ in self.config_groups]
		false_by_group: List[List[Tuple[int, int]]] = [[] for _ in self.config_groups]

		# Route true clusters
		for b in range(batch_size):
			cluster_id = int(true_clusters[b].item())
			group_idx, local_idx = self._cluster_to_group[cluster_id]
			true_by_group[group_idx].append((b, local_idx))

		# Route false clusters
		if false_clusters is not None:
			num_false = false_clusters.shape[1] if false_clusters.ndim == 2 else 1
			if false_clusters.ndim == 1:
				false_clusters = false_clusters.unsqueeze(1)

			for b in range(batch_size):
				true_cluster = int(true_clusters[b].item())
				for k in range(num_false):
					cluster_id = int(false_clusters[b, k].item())
					if cluster_id == true_cluster:
						continue
					group_idx, local_idx = self._cluster_to_group[cluster_id]
					false_by_group[group_idx].append((b, local_idx))

		# Train each config group
		for group_idx, group in enumerate(self.config_groups):
			npc = group.neurons
			memory = group.memory

			# Get addresses for this group
			addresses = memory.get_addresses(input_bits)  # [batch, group_neurons]

			# Train TRUE clusters in this group
			for batch_idx, local_cluster in true_by_group[group_idx]:
				start_neuron = local_cluster * npc
				end_neuron = start_neuron + npc

				neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
				cluster_addresses = addresses[batch_idx, start_neuron:end_neuron]
				target_bits = ones(npc, dtype=torch_bool, device=device)

				if memory.explore_batch(neuron_indices, cluster_addresses, target_bits, allow_override):
					modified += npc

			# Train FALSE clusters in this group
			for batch_idx, local_cluster in false_by_group[group_idx]:
				start_neuron = local_cluster * npc
				end_neuron = start_neuron + npc

				neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
				cluster_addresses = addresses[batch_idx, start_neuron:end_neuron]
				target_bits = zeros(npc, dtype=torch_bool, device=device)

				if memory.explore_batch(neuron_indices, cluster_addresses, target_bits, allow_override):
					modified += npc

		return modified

	def train_rust(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Rust-accelerated training for adaptive models.

		Currently falls back to PyTorch train_batch. Future optimization:
		implement a Rust function that handles adaptive architecture.

		Args:
			input_bits: [num_examples, total_input_bits] input patterns
			true_clusters: [num_examples] cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] indices to train as FALSE
			allow_override: Whether to override existing cells

		Returns:
			Number of memory cells modified
		"""
		# TODO: Implement optimized Rust training for adaptive architecture
		# For now, fall back to PyTorch implementation
		return self.train_batch(input_bits, true_clusters, false_clusters, allow_override)

	def reset_memory(self) -> None:
		"""Reset all memory cells to EMPTY, preserving connectivity."""
		for group in self.config_groups:
			group.memory.reset()
		# Also reset sparse memories
		for sparse_mem in self._sparse_memories.values():
			sparse_mem.reset()

	def get_cluster_config(self, cluster_id: int) -> Tuple[int, int]:
		"""
		Get (neurons, bits) configuration for a specific cluster.

		Args:
			cluster_id: Physical cluster ID

		Returns:
			Tuple of (neurons_per_cluster, bits_per_neuron)
		"""
		return (
			self.genome.neurons_per_cluster[cluster_id],
			self.genome.bits_per_cluster[cluster_id],
		)

	def get_config(self) -> dict:
		"""Get configuration dict for model recreation."""
		return {
			'total_input_bits': self.total_input_bits,
			'genome': {
				'bits_per_cluster': self.genome.bits_per_cluster,
				'neurons_per_cluster': self.genome.neurons_per_cluster,
			},
			'empty_value': self.empty_value,
		}

	@classmethod
	def from_config(cls, config: dict, rng: Optional[int] = None) -> "AdaptiveClusteredRAM":
		"""Create an AdaptiveClusteredRAM from a configuration dict."""
		genome = ClusterGenome(
			bits_per_cluster=config['genome']['bits_per_cluster'],
			neurons_per_cluster=config['genome']['neurons_per_cluster'],
		)
		return cls(
			total_input_bits=config['total_input_bits'],
			genome=genome,
			empty_value=config.get('empty_value', 0.0),
			rng=rng,
		)
