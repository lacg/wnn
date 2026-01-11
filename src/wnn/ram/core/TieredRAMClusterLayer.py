"""
TieredRAMClusterLayer - RAM Layer with variable neurons/bits per token frequency tier.

This layer organizes clusters into tiers based on token frequency. Each tier can
have different numbers of neurons and bits per neuron, allowing the model to
allocate more capacity to frequent tokens and less to rare ones.

Key insight: Frequent tokens benefit from more neurons (finer probability granularity)
while rare tokens can share simpler representations (fewer neurons, same generalization).

Tier definition format:
    tiers = [
        (count, neurons, bits),  # first 'count' clusters get 'neurons' neurons with 'bits' bits each
        (count, neurons, bits),  # next 'count' clusters...
        (None, neurons, bits),   # remaining clusters (None means "all rest")
    ]

Example:
    tiers = [
        (100, 11, 8),   # top 100 tokens: 11 neurons × 8 bits = 256 addresses
        (400, 7, 8),    # next 400 tokens: 7 neurons × 8 bits = 256 addresses
        (None, 5, 8),   # remaining ~49,757 tokens: 5 neurons × 8 bits = 256 addresses
    ]

Memory savings:
    - Original: 50,257 × 8 neurons × 1024 addresses = 412M cells
    - Tiered: (100×11 + 400×7 + 49,757×5) × 256 addresses = 65M cells (~84% reduction)
"""

from typing import Optional

from torch import arange, long, zeros, ones, Tensor, bool as torch_bool

from wnn.ram.core.Memory import Memory
from wnn.ram.core.base import RAMComponent
from wnn.ram.core import MemoryVal


class TierConfig:
	"""Configuration for a single tier."""

	def __init__(
		self,
		cluster_count: int,  # Number of clusters in this tier
		neurons_per_cluster: int,
		bits_per_neuron: int,
		cluster_start: int,  # Global cluster index where this tier starts
	):
		self.cluster_count = cluster_count
		self.neurons_per_cluster = neurons_per_cluster
		self.bits_per_neuron = bits_per_neuron
		self.cluster_start = cluster_start
		self.cluster_end = cluster_start + cluster_count
		self.total_neurons = cluster_count * neurons_per_cluster
		self.memory_size = 2 ** bits_per_neuron  # Addresses per neuron

	def __repr__(self) -> str:
		return (
			f"TierConfig(clusters={self.cluster_count}, neurons={self.neurons_per_cluster}, "
			f"bits={self.bits_per_neuron}, range=[{self.cluster_start}, {self.cluster_end}))"
		)


class TieredRAMClusterLayer(RAMComponent):
	"""
	RAM Layer with variable neurons/bits per token frequency tier.

	Each tier has its own Memory object, allowing different bits_per_neuron
	and neurons_per_cluster configurations.

	Attributes:
		num_clusters: Total number of output clusters (vocabulary size)
		tier_configs: List of TierConfig objects
		tier_memories: List of Memory objects, one per tier
		total_neurons: Total neurons across all tiers
	"""

	def __init__(
		self,
		total_input_bits: int,
		num_clusters: int,
		tiers: list[tuple[Optional[int], int, int]],
		cluster_order: Optional[list[int]] = None,
		rng: Optional[int] = None,
	):
		"""
		Initialize TieredRAMClusterLayer.

		Args:
			total_input_bits: Number of input bits (e.g., 64 for 4 tokens × 16 bits)
			num_clusters: Total number of output clusters (vocabulary size)
			tiers: List of (count, neurons_per_cluster, bits_per_neuron) tuples.
			       Use count=None for the final "rest" tier.
			cluster_order: Optional list mapping logical cluster index to physical cluster.
			               If None, clusters are assigned to tiers in order [0, 1, 2, ...].
			               To use frequency-based tiers, pass sorted token IDs by frequency.
			rng: Random seed for reproducible connectivity initialization

		Example:
			# Top-100 tokens get 11 neurons, next 400 get 7, rest get 5
			layer = TieredRAMClusterLayer(
				total_input_bits=64,
				num_clusters=50257,
				tiers=[(100, 11, 8), (400, 7, 8), (None, 5, 8)],
				cluster_order=sorted_token_ids_by_frequency,
			)
		"""
		super().__init__()

		self.num_clusters = num_clusters
		self.total_input_bits = total_input_bits

		# Build cluster order mapping (logical → physical)
		# cluster_order[logical_idx] = physical_cluster_idx
		# This allows "cluster 0" in tier assignment to mean "most frequent token"
		if cluster_order is not None:
			assert len(cluster_order) == num_clusters
			self._cluster_order = list(cluster_order)
			# Build reverse mapping (physical → logical)
			self._physical_to_logical = {phys: log for log, phys in enumerate(cluster_order)}
		else:
			self._cluster_order = list(range(num_clusters))
			self._physical_to_logical = {i: i for i in range(num_clusters)}

		# Parse tier configuration
		self.tier_configs: list[TierConfig] = []
		self.tier_memories: list[Memory] = []

		cluster_offset = 0
		for i, (count, neurons, bits) in enumerate(tiers):
			# Handle "rest" tier (count=None)
			if count is None:
				count = num_clusters - cluster_offset
				if count <= 0:
					raise ValueError(f"No clusters remaining for 'rest' tier (tier {i})")

			if cluster_offset + count > num_clusters:
				raise ValueError(
					f"Tier {i} requests {count} clusters but only {num_clusters - cluster_offset} remain"
				)

			config = TierConfig(
				cluster_count=count,
				neurons_per_cluster=neurons,
				bits_per_neuron=bits,
				cluster_start=cluster_offset,
			)
			self.tier_configs.append(config)

			# Create Memory for this tier
			memory = Memory(
				total_input_bits=total_input_bits,
				num_neurons=config.total_neurons,
				n_bits_per_neuron=bits,
				rng=rng + i if rng is not None else None,
			)
			self.tier_memories.append(memory)

			cluster_offset += count

		# Verify all clusters are assigned
		if cluster_offset != num_clusters:
			raise ValueError(
				f"Tiers only account for {cluster_offset} clusters, but num_clusters={num_clusters}"
			)

		# Compute totals
		self.total_neurons = sum(tc.total_neurons for tc in self.tier_configs)
		self.total_memory_cells = sum(
			tc.total_neurons * tc.memory_size for tc in self.tier_configs
		)

		# Check if any tier needs sparse backend (>10 bits)
		self._use_sparse = any(tc.bits_per_neuron > 10 for tc in self.tier_configs)
		self._sparse_memory = None

		if self._use_sparse:
			# Initialize Rust sparse tiered memory
			try:
				import ram_accelerator
				# Build tier configs for Rust: (end_cluster, neurons_per_cluster, bits_per_neuron)
				rust_tier_configs = [
					(tc.cluster_end, tc.neurons_per_cluster, tc.bits_per_neuron)
					for tc in self.tier_configs
				]
				self._sparse_memory = ram_accelerator.TieredSparseMemory(
					rust_tier_configs, num_clusters
				)
			except (ImportError, AttributeError):
				# Fall back to dense if Rust sparse not available
				self._use_sparse = False

		# Build lookup tables for fast cluster → tier mapping
		# For each physical cluster, store (tier_idx, local_cluster_idx)
		self._cluster_to_tier: list[tuple[int, int]] = []
		for phys_cluster in range(num_clusters):
			logical_cluster = self._physical_to_logical[phys_cluster]
			for tier_idx, tc in enumerate(self.tier_configs):
				if tc.cluster_start <= logical_cluster < tc.cluster_end:
					local_cluster = logical_cluster - tc.cluster_start
					self._cluster_to_tier.append((tier_idx, local_cluster))
					break

		# Precompute scatter indices for fast forward pass
		self._scatter_indices: list[Tensor] = []
		for tc in self.tier_configs:
			indices = []
			for local_cluster in range(tc.cluster_count):
				logical_cluster = tc.cluster_start + local_cluster
				physical_cluster = self._cluster_order[logical_cluster]
				indices.append(physical_cluster)
			self._scatter_indices.append(Tensor(indices).long())

	def __repr__(self) -> str:
		tier_strs = ", ".join(
			f"({tc.cluster_count}, {tc.neurons_per_cluster}, {tc.bits_per_neuron})"
			for tc in self.tier_configs
		)
		return (
			f"TieredRAMClusterLayer("
			f"clusters={self.num_clusters}, "
			f"tiers=[{tier_strs}], "
			f"total_neurons={self.total_neurons})"
		)

	def __str__(self) -> str:
		lines = [
			"=== TieredRAMClusterLayer ===",
			f"  Total clusters: {self.num_clusters:,}",
			f"  Total neurons: {self.total_neurons:,}",
			f"  Total memory cells: {self.total_memory_cells:,}",
			f"  Input bits: {self.total_input_bits}",
			"",
			"  Tiers:",
		]
		for i, tc in enumerate(self.tier_configs):
			lines.append(
				f"    Tier {i}: {tc.cluster_count:,} clusters × "
				f"{tc.neurons_per_cluster} neurons × {tc.bits_per_neuron} bits "
				f"= {tc.total_neurons:,} neurons, {tc.memory_size} addresses each"
			)
		return "\n".join(lines)

	@property
	def use_sparse_backend(self) -> bool:
		"""Check if using sparse memory backend."""
		return self._use_sparse and self._sparse_memory is not None

	@property
	def connections(self) -> list[Tensor]:
		"""Get connectivity matrices for all tiers."""
		return [mem.connections for mem in self.tier_memories]

	def get_tier_connections(self, tier_idx: int) -> Tensor:
		"""Get connectivity matrix for a specific tier."""
		return self.tier_memories[tier_idx].connections

	def set_tier_connections(self, tier_idx: int, connections: Tensor) -> None:
		"""Set connectivity matrix for a specific tier."""
		tc = self.tier_configs[tier_idx]
		expected_shape = (tc.total_neurons, tc.bits_per_neuron)
		assert connections.shape == expected_shape, f"Expected {expected_shape}, got {connections.shape}"
		self.tier_memories[tier_idx].connections = connections

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass returning probabilities per cluster.

		For architectures with >10 bits per neuron, uses sparse memory backend.

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			[batch, num_clusters] float tensor of probabilities in [0, 1]
		"""
		# Use sparse forward if available (for >10 bits per neuron)
		if self._use_sparse and self._sparse_memory is not None:
			return self.forward_sparse(input_bits)

		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]
		device = input_bits.device

		# Output tensor for all clusters
		probs = zeros(batch_size, self.num_clusters, device=device)

		# Process each tier and scatter results
		for tier_idx, (tc, memory) in enumerate(zip(self.tier_configs, self.tier_memories)):
			# Get raw memory values for this tier: [batch, tier_neurons]
			raw = memory.get_memories_for_bits(input_bits)

			# Reshape to clusters: [batch, tier_clusters, neurons_per_cluster]
			clustered = raw.view(batch_size, tc.cluster_count, tc.neurons_per_cluster)

			# Compute probabilities: (TRUE + 0.5 × EMPTY) / neurons_per_cluster
			count_true = (clustered == MemoryVal.TRUE).sum(dim=-1)
			count_empty = (clustered == MemoryVal.EMPTY).sum(dim=-1)
			tier_probs = (count_true.float() + 0.5 * count_empty.float()) / tc.neurons_per_cluster

			# Scatter to output using precomputed indices
			scatter_idx = self._scatter_indices[tier_idx].to(device)
			probs.scatter_(1, scatter_idx.unsqueeze(0).expand(batch_size, -1), tier_probs)

		return probs

	def forward_auto(self, input_bits: Tensor) -> Tensor:
		"""
		Auto-optimized forward pass.

		Uses sparse backend for >10 bits per neuron, otherwise standard forward.

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			[batch, num_clusters] float tensor of probabilities
		"""
		# forward() already handles sparse dispatch
		return self.forward(input_bits)

	def train_batch(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Optional[Tensor] = None,
		allow_override: bool = False,
	) -> int:
		"""
		Batch training: train TRUE for target clusters, FALSE for negative clusters.

		Routes each cluster to its appropriate tier for training.

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

		# Group clusters by tier for efficient training
		# true_by_tier[tier_idx] = [(batch_idx, local_cluster), ...]
		true_by_tier: list[list[tuple[int, int]]] = [[] for _ in self.tier_configs]
		false_by_tier: list[list[tuple[int, int]]] = [[] for _ in self.tier_configs]

		# Route true clusters
		for b in range(batch_size):
			phys_cluster = int(true_clusters[b].item())
			tier_idx, local_cluster = self._cluster_to_tier[phys_cluster]
			true_by_tier[tier_idx].append((b, local_cluster))

		# Route false clusters
		if false_clusters is not None:
			num_false = false_clusters.shape[1] if false_clusters.ndim == 2 else 1
			if false_clusters.ndim == 1:
				false_clusters = false_clusters.unsqueeze(1)

			for b in range(batch_size):
				true_phys = int(true_clusters[b].item())
				for k in range(num_false):
					phys_cluster = int(false_clusters[b, k].item())
					if phys_cluster == true_phys:
						continue
					tier_idx, local_cluster = self._cluster_to_tier[phys_cluster]
					false_by_tier[tier_idx].append((b, local_cluster))

		# Train each tier
		for tier_idx, (tc, memory) in enumerate(zip(self.tier_configs, self.tier_memories)):
			npc = tc.neurons_per_cluster
			offsets = arange(npc, device=device)

			# Get addresses for this tier
			addresses = memory.get_addresses(input_bits)  # [batch, tier_neurons]

			# Train TRUE clusters in this tier
			for batch_idx, local_cluster in true_by_tier[tier_idx]:
				start_neuron = local_cluster * npc
				end_neuron = start_neuron + npc

				neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
				cluster_addresses = addresses[batch_idx, start_neuron:end_neuron]
				target_bits = ones(npc, dtype=torch_bool, device=device)

				if memory.explore_batch(neuron_indices, cluster_addresses, target_bits, allow_override):
					modified += npc

			# Train FALSE clusters in this tier
			for batch_idx, local_cluster in false_by_tier[tier_idx]:
				start_neuron = local_cluster * npc
				end_neuron = start_neuron + npc

				neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
				cluster_addresses = addresses[batch_idx, start_neuron:end_neuron]
				target_bits = zeros(npc, dtype=torch_bool, device=device)

				if memory.explore_batch(neuron_indices, cluster_addresses, target_bits, allow_override):
					modified += npc

		return modified

	def train_multi_examples(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Vectorized training for multiple examples at once.

		Groups examples by tier and uses batch operations within each tier.

		Args:
			input_bits: [num_examples, total_input_bits] input patterns
			true_clusters: [num_examples] physical cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] physical cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		from torch import stack, tensor as torch_tensor, int64, unique, cat

		num_examples = input_bits.shape[0]
		num_negatives = false_clusters.shape[1]
		device = input_bits.device
		modified = 0

		# Pre-compute tier assignments for all clusters we'll train
		# Build tier_idx and local_cluster for each physical cluster
		true_tier_indices = torch_tensor(
			[self._cluster_to_tier[int(c)][0] for c in true_clusters], dtype=int64, device=device
		)
		true_local_clusters = torch_tensor(
			[self._cluster_to_tier[int(c)][1] for c in true_clusters], dtype=int64, device=device
		)

		# For false clusters: [num_examples, num_negatives]
		false_tier_indices = torch_tensor(
			[[self._cluster_to_tier[int(c)][0] for c in row] for row in false_clusters],
			dtype=int64, device=device
		)
		false_local_clusters = torch_tensor(
			[[self._cluster_to_tier[int(c)][1] for c in row] for row in false_clusters],
			dtype=int64, device=device
		)

		# Process each tier
		for tier_idx, (tc, memory) in enumerate(zip(self.tier_configs, self.tier_memories)):
			npc = tc.neurons_per_cluster

			# Find examples where TRUE cluster is in this tier
			true_mask = (true_tier_indices == tier_idx)
			true_example_indices = true_mask.nonzero(as_tuple=True)[0]

			if len(true_example_indices) == 0:
				continue

			# Get inputs and local clusters for this tier's TRUE training
			tier_inputs = input_bits[true_example_indices]  # [n, total_input_bits]
			tier_true_local = true_local_clusters[true_example_indices]  # [n]

			# Build false clusters for these examples (only those in THIS tier)
			# For each example, filter false_clusters to those in this tier
			tier_false_local_list = []
			for i, ex_idx in enumerate(true_example_indices):
				ex_false_tiers = false_tier_indices[ex_idx]  # [num_negatives]
				ex_false_local = false_local_clusters[ex_idx]  # [num_negatives]
				# Filter to this tier
				mask = (ex_false_tiers == tier_idx)
				tier_false = ex_false_local[mask]
				# Pad to same length (use -1 as padding, will be filtered out)
				padded = ones(num_negatives, dtype=int64, device=device) * -1
				padded[:len(tier_false)] = tier_false
				tier_false_local_list.append(padded)

			tier_false_local = stack(tier_false_local_list)  # [n, num_negatives]

			# Train this tier using its memory's batch training
			# Get addresses for all neurons in this tier
			addresses = memory.get_addresses(tier_inputs)  # [n, tier_neurons]

			# Train TRUE - vectorized
			for i in range(len(true_example_indices)):
				local_cluster = int(tier_true_local[i].item())
				start_neuron = local_cluster * npc
				end_neuron = start_neuron + npc

				neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
				cluster_addresses = addresses[i, start_neuron:end_neuron]
				target_bits = ones(npc, dtype=torch_bool, device=device)

				if memory.explore_batch(neuron_indices, cluster_addresses, target_bits, allow_override):
					modified += npc

			# Train FALSE - vectorized
			for i in range(len(true_example_indices)):
				true_local = int(tier_true_local[i].item())
				for k in range(num_negatives):
					local_cluster = int(tier_false_local[i, k].item())
					if local_cluster == -1:  # padding
						continue
					if local_cluster == true_local:  # skip if same as true
						continue

					start_neuron = local_cluster * npc
					end_neuron = start_neuron + npc

					neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
					cluster_addresses = addresses[i, start_neuron:end_neuron]
					target_bits = zeros(npc, dtype=torch_bool, device=device)

					if memory.explore_batch(neuron_indices, cluster_addresses, target_bits, allow_override):
						modified += npc

		return modified

	def train_multi_examples_rust_numpy(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Rust-accelerated training for tiered models.

		Uses optimized single-call Rust function that handles ALL tiers internally.
		This eliminates Python loop overhead and provides full parallelization.

		For architectures with >10 bits per neuron, automatically uses sparse
		memory backend to avoid massive data transfer overhead.

		Args:
			input_bits: [num_examples, total_input_bits] input patterns
			true_clusters: [num_examples] physical cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] physical cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		# Use sparse training if available (for >10 bits per neuron)
		if self._use_sparse and self._sparse_memory is not None:
			return self.train_sparse(input_bits, true_clusters, false_clusters, allow_override)

		try:
			import ram_accelerator
			# Check if optimized tiered function is available
			if not hasattr(ram_accelerator, 'ramlm_train_batch_tiered_numpy'):
				return self._train_multi_examples_rust_numpy_legacy(
					input_bits, true_clusters, false_clusters, allow_override
				)
		except ImportError:
			# Fall back to PyTorch if Rust not available
			return self.train_multi_examples(input_bits, true_clusters, false_clusters, allow_override)

		import numpy as np
		from torch import tensor as torch_tensor, int64

		num_examples = input_bits.shape[0]
		num_negatives = false_clusters.shape[1]

		# Flatten all inputs to numpy
		input_bits_np = input_bits.flatten().bool().numpy().astype(np.uint8)
		true_clusters_np = true_clusters.cpu().numpy().astype(np.int64)
		false_clusters_np = false_clusters.flatten().cpu().numpy().astype(np.int64)

		# Build tier configs with offsets
		# Format: (cluster_start, cluster_end, neurons_per_cluster, bits_per_neuron,
		#          words_per_neuron, memory_offset, conn_offset)
		tier_configs = []
		memory_offset = 0
		conn_offset = 0
		cluster_start = 0

		for tc, memory in zip(self.tier_configs, self.tier_memories):
			tier_configs.append((
				cluster_start,                    # cluster_start (global)
				cluster_start + tc.cluster_count, # cluster_end (global)
				tc.neurons_per_cluster,
				tc.bits_per_neuron,
				memory.words_per_neuron,
				memory_offset,
				conn_offset,
			))
			cluster_start += tc.cluster_count
			memory_offset += tc.total_neurons * memory.words_per_neuron
			conn_offset += tc.total_neurons * tc.bits_per_neuron

		# Concatenate all tier connections and memory
		connections_list = [m.connections.flatten().numpy().astype(np.int64) for m in self.tier_memories]
		memory_list = [m.memory_words.flatten().numpy().astype(np.int64) for m in self.tier_memories]

		connections_flat = np.concatenate(connections_list)
		memory_flat = np.concatenate(memory_list)

		# Call optimized Rust function (all tiers in one call)
		modified, new_memory = ram_accelerator.ramlm_train_batch_tiered_numpy(
			input_bits_np,
			true_clusters_np,
			false_clusters_np,
			connections_flat,
			memory_flat,
			num_examples,
			self.total_input_bits,
			num_negatives,
			tier_configs,
			allow_override,
		)

		# Split returned memory back to individual tiers
		offset = 0
		for tc, memory in zip(self.tier_configs, self.tier_memories):
			tier_size = tc.total_neurons * memory.words_per_neuron
			tier_memory = new_memory[offset:offset + tier_size]
			memory.memory_words[:] = torch_tensor(tier_memory, dtype=int64).view_as(memory.memory_words)
			offset += tier_size

		return modified

	def _train_multi_examples_rust_numpy_legacy(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Legacy Rust-accelerated training (calls Rust per-tier).

		Used as fallback if optimized tiered function is not available.
		"""
		import ram_accelerator
		import numpy as np
		from torch import tensor as torch_tensor, int64

		num_examples = input_bits.shape[0]
		num_negatives = false_clusters.shape[1]
		modified = 0

		# Pre-compute tier assignments using numpy for speed
		true_clusters_np = true_clusters.cpu().numpy()
		false_clusters_np = false_clusters.cpu().numpy()

		# Build tier and local cluster mappings
		true_tier_idx = np.array([self._cluster_to_tier[c][0] for c in true_clusters_np])
		true_local_cluster = np.array([self._cluster_to_tier[c][1] for c in true_clusters_np])

		# Process each tier with Rust acceleration
		for tier_idx, (tc, memory) in enumerate(zip(self.tier_configs, self.tier_memories)):
			# Find examples where TRUE cluster is in this tier
			tier_mask = (true_tier_idx == tier_idx)
			tier_example_indices = np.where(tier_mask)[0]

			if len(tier_example_indices) == 0:
				continue

			# Get inputs for this tier's examples
			tier_inputs = input_bits[tier_example_indices]  # [n, total_input_bits]
			tier_true_local = true_local_cluster[tier_example_indices]  # [n]

			# Build false clusters: only include those in THIS tier
			tier_false_local = []
			for ex_idx in tier_example_indices:
				ex_false = false_clusters_np[ex_idx]  # [num_negatives]
				local_false = []
				for fc in ex_false:
					fc_tier, fc_local = self._cluster_to_tier[fc]
					if fc_tier == tier_idx:
						local_false.append(fc_local)
				# Pad with first valid negative or 0 if none (will be skipped if same as true)
				while len(local_false) < num_negatives:
					local_false.append(local_false[0] if local_false else 0)
				tier_false_local.append(local_false[:num_negatives])

			tier_false_local = np.array(tier_false_local, dtype=np.int64)  # [n, num_negatives]

			# Convert to numpy arrays for Rust
			tier_inputs_np = tier_inputs.flatten().bool().numpy().astype(np.uint8)
			tier_true_np = tier_true_local.astype(np.int64)
			tier_false_np = tier_false_local.flatten().astype(np.int64)
			connections_np = memory.connections.flatten().numpy().astype(np.int64)
			memory_words_np = memory.memory_words.flatten().numpy().astype(np.int64)

			# Call Rust training for this tier
			tier_modified, new_memory = ram_accelerator.ramlm_train_batch_numpy(
				tier_inputs_np,
				tier_true_np,
				tier_false_np,
				connections_np,
				memory_words_np,
				len(tier_example_indices),
				self.total_input_bits,
				tc.total_neurons,
				tc.bits_per_neuron,
				tc.neurons_per_cluster,
				num_negatives,
				memory.words_per_neuron,
				allow_override,
			)

			# Update memory from Rust result
			memory.memory_words[:] = torch_tensor(new_memory, dtype=int64).view_as(memory.memory_words)
			modified += tier_modified

		return modified

	def train_sparse(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Training using Rust sparse tiered memory backend.

		Memory stays in Rust - only returns count of modified cells.
		This avoids the massive data transfer overhead of dense training.

		Args:
			input_bits: [num_examples, total_input_bits] input patterns
			true_clusters: [num_examples] physical cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] physical cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells (ignored for sparse)

		Returns:
			Number of memory cells modified
		"""
		if not self._use_sparse or self._sparse_memory is None:
			raise RuntimeError("train_sparse called but sparse memory not initialized")

		import ram_accelerator
		import numpy as np

		num_examples = input_bits.shape[0]
		num_negatives = false_clusters.shape[1]

		# Flatten inputs for Rust
		input_bits_flat = input_bits.flatten().bool().tolist()
		true_clusters_list = true_clusters.tolist()
		false_clusters_flat = false_clusters.flatten().tolist()

		# Build flattened connections for ALL neurons across tiers
		connections_list = []
		for memory in self.tier_memories:
			connections_list.append(memory.connections.flatten().numpy().astype(np.int64))
		connections_flat = np.concatenate(connections_list).tolist()

		# Call Rust sparse tiered training
		modified = ram_accelerator.sparse_train_batch_tiered(
			self._sparse_memory,
			input_bits_flat,
			true_clusters_list,
			false_clusters_flat,
			connections_flat,
			num_examples,
			self.total_input_bits,
			num_negatives,
		)

		return modified

	def forward_sparse(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass using Rust sparse tiered memory backend.

		Args:
			input_bits: [batch_size, total_input_bits] boolean tensor

		Returns:
			[batch_size, num_clusters] float tensor of probabilities
		"""
		if not self._use_sparse or self._sparse_memory is None:
			raise RuntimeError("forward_sparse called but sparse memory not initialized")

		import ram_accelerator
		import numpy as np
		from torch import tensor as torch_tensor, float32

		# Handle single example
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]

		# Flatten inputs for Rust
		input_bits_flat = input_bits.flatten().bool().tolist()

		# Build flattened connections for ALL neurons across tiers
		connections_list = []
		for memory in self.tier_memories:
			connections_list.append(memory.connections.flatten().numpy().astype(np.int64))
		connections_flat = np.concatenate(connections_list).tolist()

		# Call Rust sparse forward
		probs_flat = ram_accelerator.sparse_forward_batch_tiered(
			self._sparse_memory,
			input_bits_flat,
			connections_flat,
			batch_size,
			self.total_input_bits,
		)

		return torch_tensor(probs_flat, dtype=float32).view(batch_size, self.num_clusters)

	def reset_memory(self) -> None:
		"""Reset all memory cells to EMPTY, preserving connectivity."""
		for memory in self.tier_memories:
			memory.reset()
		# Also reset sparse memory if using it
		if self._sparse_memory is not None:
			self._sparse_memory.reset()

	def get_config(self) -> dict:
		"""Get configuration dict for model recreation."""
		tiers = [
			(tc.cluster_count, tc.neurons_per_cluster, tc.bits_per_neuron)
			for tc in self.tier_configs
		]
		return {
			'total_input_bits': self.total_input_bits,
			'num_clusters': self.num_clusters,
			'tiers': tiers,
		}

	@classmethod
	def from_config(cls, config: dict) -> "TieredRAMClusterLayer":
		"""Create a TieredRAMClusterLayer from a configuration dict."""
		return cls(**config)
