"""
RAMClusterLayer - RAM Layer with clustered neurons for probabilistic output.

This layer organizes neurons into clusters, where each cluster represents
one output class. The probability for each class is computed from the
responses of its cluster's neurons.

Key insight: This enables RAM WNNs to produce probability distributions
over large vocabularies (e.g., 50K tokens) using the native RAM architecture
with partial connectivity for generalization.

Probability formula per cluster:
    P(class) = (count_TRUE + 0.5 * count_EMPTY) / neurons_per_cluster

Where:
    - TRUE (1): Strong evidence for this class
    - FALSE (0): Evidence against this class
    - EMPTY (2): No evidence (treated as 0.5 uncertainty)
"""

from wnn.ram.core.Memory import Memory
from wnn.ram.core.base import RAMComponent
from wnn.ram.core import MemoryVal

from typing import Optional

from torch import arange
from torch import long
from torch import Tensor


def bits_needed(n: int) -> int:
	"""
	Number of bits needed to represent n distinct values (0 to n-1).

	Uses bitwise operations for efficiency (no floats, no log).

	Examples:
		bits_needed(256) → 8
		bits_needed(50257) → 16
		bits_needed(1) → 1
	"""
	if n <= 1:
		return 1
	return (n - 1).bit_length()


class RAMClusterLayer(RAMComponent):
	"""
	RAM Layer organized into clusters for probabilistic output.

	Each cluster of neurons represents one output class (e.g., one token in vocabulary).
	The layer outputs a probability for each cluster based on its neurons' responses.

	Architecture:
		- num_clusters: Number of output classes (e.g., vocab_size = 50,257)
		- neurons_per_cluster: Neurons per class (default 5, odd for majority)
		- Total neurons = num_clusters × neurons_per_cluster

	Memory footprint example (GPT-2 vocabulary):
		- 50,257 clusters × 7 neurons = 351,799 neurons
		- 10 bits per neuron = 1,024 memory cells each
		- Total: ~360M cells, packed into ~11.6M int64 = ~93MB

	Usage:
		layer = RAMClusterLayer(
			total_input_bits=96,      # 6 tokens × 16 bits
			num_clusters=50257,        # vocabulary size
			neurons_per_cluster=7,     # odd for majority voting
			bits_per_neuron=10,        # partial connectivity
		)

		probs = layer(input_bits)  # [batch, num_clusters] probabilities
	"""

	def __init__(
		self,
		total_input_bits: int,
		num_clusters: int,
		neurons_per_cluster: int = 5,
		bits_per_neuron: int = 10,
		connections: Optional[Tensor] = None,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: Optional[int] = None,
	):
		"""
		Initialize RAMClusterLayer.

		Args:
			total_input_bits: Number of input bits (e.g., 96 for 6 tokens × 16 bits)
			num_clusters: Number of output classes (e.g., vocab_size)
			neurons_per_cluster: Neurons per cluster (default 5, odd recommended)
			bits_per_neuron: Bits each neuron observes (partial connectivity)
			connections: Optional pre-defined connectivity [total_neurons, bits_per_neuron]
			use_hashing: Whether to hash addresses (for very large connectivity)
			hash_size: Hash table size if use_hashing=True
			rng: Random seed for reproducible connectivity initialization
		"""
		super().__init__()

		self.num_clusters = num_clusters
		self.neurons_per_cluster = neurons_per_cluster
		self.total_neurons = num_clusters * neurons_per_cluster

		# Create underlying Memory with total neurons
		self.memory = Memory(
			total_input_bits=total_input_bits,
			num_neurons=self.total_neurons,
			n_bits_per_neuron=bits_per_neuron,
			connections=connections,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=rng,
		)

	def __repr__(self) -> str:
		return (
			f"RAMClusterLayer("
			f"clusters={self.num_clusters}, "
			f"neurons_per_cluster={self.neurons_per_cluster}, "
			f"total_neurons={self.total_neurons}, "
			f"bits_per_neuron={self.memory.n_bits_per_neuron}, "
			f"input_bits={self.memory.total_input_bits})"
		)

	def __str__(self) -> str:
		lines = [
			"=== RAMClusterLayer ===",
			f"  Clusters: {self.num_clusters}",
			f"  Neurons per cluster: {self.neurons_per_cluster}",
			f"  Total neurons: {self.total_neurons}",
			f"  Input bits: {self.memory.total_input_bits}",
			f"  Bits per neuron: {self.memory.n_bits_per_neuron}",
			f"  Memory cells per neuron: {self.memory.memory_size}",
			f"  Total memory cells: {self.total_neurons * self.memory.memory_size:,}",
		]
		return "\n".join(lines)

	@property
	def total_input_bits(self) -> int:
		return self.memory.total_input_bits

	@property
	def bits_per_neuron(self) -> int:
		return self.memory.n_bits_per_neuron

	@property
	def connections(self) -> Tensor:
		"""Get connectivity matrix [total_neurons, bits_per_neuron]."""
		return self.memory.connections

	@connections.setter
	def connections(self, value: Tensor) -> None:
		"""Set connectivity matrix (for optimization)."""
		assert value.shape == (self.total_neurons, self.memory.n_bits_per_neuron)
		self.memory.connections = value

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass returning probabilities per cluster (PyTorch implementation).

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			[batch, num_clusters] float tensor of probabilities in [0, 1]

		Probability formula:
			P(cluster) = (count_TRUE + 0.5 * count_EMPTY) / neurons_per_cluster
		"""
		# Get raw memory values: [batch, total_neurons]
		# Values are: FALSE=0, TRUE=1, EMPTY=2
		raw = self.memory.get_memories_for_bits(input_bits)  # [batch, total_neurons]

		# Reshape to clusters: [batch, num_clusters, neurons_per_cluster]
		batch_size = raw.shape[0]
		clustered = raw.view(batch_size, self.num_clusters, self.neurons_per_cluster)

		# Count TRUE (1) and EMPTY (2) per cluster
		count_true = (clustered == MemoryVal.TRUE).sum(dim=-1)   # [batch, num_clusters]
		count_empty = (clustered == MemoryVal.EMPTY).sum(dim=-1) # [batch, num_clusters]

		# Probability: (TRUE + 0.5 × EMPTY) / neurons_per_cluster
		probs = (count_true.float() + 0.5 * count_empty.float()) / self.neurons_per_cluster

		return probs  # [batch, num_clusters]

	def forward_auto(self, input_bits: Tensor) -> Tensor:
		"""
		Auto-optimized forward pass that picks the best backend.

		Transparently selects between PyTorch and Metal GPU based on batch size:
		- Batch < 1000: PyTorch (optimized for small batches, ~1700 ex/s)
		- Batch >= 1000: Metal GPU (optimized for large batches, ~2400 ex/s)

		This provides the best performance without manual backend selection.

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			[batch, num_clusters] float tensor of probabilities
		"""
		# Ensure 2D input
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]

		# Threshold determined by benchmarking on M4 Max:
		# - PyTorch peaks at batch ~200 with 1700 ex/s
		# - Metal becomes faster at batch 1000+ and scales to 2400+ ex/s
		METAL_THRESHOLD = 1000

		if batch_size < METAL_THRESHOLD:
			# Use PyTorch for small batches (lower per-call overhead)
			return self.forward(input_bits)
		else:
			# Use Metal GPU for large batches (better parallelism)
			return self._forward_metal_cached(input_bits)

	def _forward_metal_cached(self, input_bits: Tensor) -> Tensor:
		"""
		Metal GPU forward with cached evaluator (internal use).

		Uses a globally cached Metal evaluator to avoid shader recompilation.
		"""
		try:
			import ram_accelerator
		except ImportError:
			# Fall back to PyTorch if Rust not available
			return self.forward(input_bits)

		if not ram_accelerator.ramlm_metal_available():
			# Fall back to PyTorch if Metal not available
			return self.forward(input_bits)

		from torch import from_numpy
		import numpy as np

		# Ensure 2D input
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]

		# Use numpy arrays for fast data transfer
		input_bits_np = input_bits.flatten().to(dtype=long).numpy().astype(np.uint8)
		connections_np = self.memory.connections.flatten().numpy()
		memory_words_np = self.memory.memory_words.flatten().numpy()

		# Call Metal GPU with cached evaluator
		probs_flat = ram_accelerator.ramlm_forward_batch_metal_cached(
			input_bits_np,
			connections_np,
			memory_words_np,
			batch_size,
			self.memory.total_input_bits,
			self.total_neurons,
			self.memory.n_bits_per_neuron,
			self.neurons_per_cluster,
			self.num_clusters,
			self.memory.words_per_neuron,
		)

		# Reshape back to [batch, num_clusters]
		probs = from_numpy(np.array(probs_flat, dtype=np.float32)).view(batch_size, self.num_clusters)
		return probs

	def forward_raw(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass returning raw memory values (for debugging/analysis).

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			[batch, num_clusters, neurons_per_cluster] int64 tensor
			with values in {0=FALSE, 1=TRUE, 2=EMPTY}
		"""
		raw = self.memory.get_memories_for_bits(input_bits)
		batch_size = raw.shape[0]
		return raw.view(batch_size, self.num_clusters, self.neurons_per_cluster)

	def predict(self, input_bits: Tensor) -> tuple[Tensor, Tensor]:
		"""
		Predict the most likely cluster and its probability.

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			Tuple of:
				- predicted_clusters: [batch] int64 cluster indices
				- confidences: [batch] float probabilities
		"""
		probs = self.forward(input_bits)  # [batch, num_clusters]
		confidences, predicted = probs.max(dim=-1)
		return predicted, confidences

	def train_pattern(
		self,
		input_bits: Tensor,
		target_cluster: int,
		value: bool = True,
		allow_override: bool = False,
	) -> bool:
		"""
		Train all neurons in target cluster to output TRUE or FALSE for this input.

		Args:
			input_bits: [total_input_bits] or [1, total_input_bits] input pattern
			target_cluster: Index of the target cluster (0 to num_clusters-1)
			value: TRUE or FALSE to train (default TRUE)
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			True if any memory was modified
		"""
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		# Get neuron indices for target cluster
		start_neuron = target_cluster * self.neurons_per_cluster
		end_neuron = start_neuron + self.neurons_per_cluster

		# Get addresses for all neurons
		addresses = self.memory.get_addresses(input_bits)[0]  # [total_neurons]

		# Train only neurons in target cluster
		neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=input_bits.device)
		cluster_addresses = addresses[start_neuron:end_neuron]

		# All neurons in cluster should output the specified value
		from torch import full, bool as torch_bool
		target_bits = full(
			(self.neurons_per_cluster,),
			value,
			dtype=torch_bool,
			device=input_bits.device,
		)

		return self.memory.explore_batch(
			neuron_indices,
			cluster_addresses,
			target_bits,
			allow_override=allow_override,
		)

	def train_batch(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Optional[Tensor] = None,
		allow_override: bool = False,
	) -> int:
		"""
		Batch training: train TRUE for target clusters, FALSE for negative clusters.

		This is the efficient method for training with the hybrid top-k strategy:
		- TRUE for target cluster (the correct next token)
		- FALSE for negative clusters (global top-k + context-specific alternatives)

		Args:
			input_bits: [batch, total_input_bits] input patterns
			true_clusters: [batch] cluster indices to train as TRUE
			false_clusters: Optional [batch, k] cluster indices to train as FALSE
			                If None, only TRUE clusters are trained.
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified

		Example:
			# Train: context → target=42, negatives=[0, 1, 5, 100]
			layer.train_batch(
				input_bits=encoded_context,  # [1, 96]
				true_clusters=torch.tensor([42]),
				false_clusters=torch.tensor([[0, 1, 5, 100]]),
			)
		"""
		from torch import cat, zeros, ones, bool as torch_bool

		batch_size = input_bits.shape[0]
		device = input_bits.device
		modified = 0

		# Get addresses for all neurons for all batch items
		# addresses: [batch, total_neurons]
		addresses = self.memory.get_addresses(input_bits)

		# ===== Train TRUE clusters =====
		for b in range(batch_size):
			cluster_idx = int(true_clusters[b].item())
			start_neuron = cluster_idx * self.neurons_per_cluster
			end_neuron = start_neuron + self.neurons_per_cluster

			neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
			cluster_addresses = addresses[b, start_neuron:end_neuron]
			target_bits = ones(self.neurons_per_cluster, dtype=torch_bool, device=device)

			if self.memory.explore_batch(neuron_indices, cluster_addresses, target_bits, allow_override):
				modified += self.neurons_per_cluster

		# ===== Train FALSE clusters =====
		if false_clusters is not None:
			num_false = false_clusters.shape[1] if false_clusters.ndim == 2 else 1
			if false_clusters.ndim == 1:
				false_clusters = false_clusters.unsqueeze(1)

			for b in range(batch_size):
				for k in range(num_false):
					cluster_idx = int(false_clusters[b, k].item())

					# Skip if this is the same as true cluster (shouldn't happen, but safety)
					if cluster_idx == int(true_clusters[b].item()):
						continue

					start_neuron = cluster_idx * self.neurons_per_cluster
					end_neuron = start_neuron + self.neurons_per_cluster

					neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
					cluster_addresses = addresses[b, start_neuron:end_neuron]
					target_bits = zeros(self.neurons_per_cluster, dtype=torch_bool, device=device)

					if self.memory.explore_batch(neuron_indices, cluster_addresses, target_bits, allow_override):
						modified += self.neurons_per_cluster

		return modified

	def train_multi_examples(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Fully vectorized training for multiple examples at once.

		This is 50-100x faster than calling train_batch in a loop because it:
		1. Computes addresses ONLY for needed clusters (not all 250K neurons)
		2. Collects ALL writes into single batched operations

		Args:
			input_bits: [num_examples, total_input_bits] input patterns
			true_clusters: [num_examples] cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		from torch import cat, zeros, ones, bool as torch_bool, unique, repeat_interleave, searchsorted

		num_examples = input_bits.shape[0]
		num_negatives = false_clusters.shape[1]
		npc = self.neurons_per_cluster
		device = input_bits.device
		offsets = arange(npc, device=device)

		# ===== Compute addresses ONLY for clusters we'll train =====
		# Get unique cluster indices and build sorted mapping for fast lookup
		all_clusters = cat([true_clusters, false_clusters.flatten()])
		unique_clusters, inverse_indices = unique(all_clusters, sorted=True, return_inverse=True)
		num_unique = unique_clusters.shape[0]

		# Compute neuron indices for unique clusters: [num_unique * npc]
		unique_start_neurons = unique_clusters * npc
		unique_neuron_indices = (unique_start_neurons.unsqueeze(1) + offsets).flatten()

		# Compute addresses only for unique neurons: [num_examples, num_unique * npc]
		addresses = self.memory.get_addresses_for_neurons(input_bits, unique_neuron_indices)

		# ===== Collect TRUE writes (fully vectorized) =====
		# Map true_clusters to positions in unique_clusters using inverse_indices
		# inverse_indices[0:num_examples] corresponds to true_clusters
		true_positions = inverse_indices[:num_examples]  # [num_examples]

		# Compute global neuron indices
		true_start_neurons = true_clusters * npc
		true_neuron_indices = (true_start_neurons.unsqueeze(1) + offsets).flatten()

		# Compute local indices into addresses tensor: [num_examples, npc]
		true_local_base = true_positions * npc  # [num_examples]
		true_local_indices = (true_local_base.unsqueeze(1) + offsets).flatten()

		# Get addresses
		example_indices = arange(num_examples, device=device).unsqueeze(1).expand(-1, npc).flatten()
		true_addresses = addresses[example_indices, true_local_indices]

		true_bits = ones(num_examples * npc, dtype=torch_bool, device=device)

		# ===== Collect FALSE writes (fully vectorized) =====
		# inverse_indices[num_examples:] corresponds to false_clusters.flatten()
		false_positions = inverse_indices[num_examples:].view(num_examples, num_negatives)

		# Compute global neuron indices
		false_start_neurons = false_clusters * npc
		false_neuron_indices = (false_start_neurons.unsqueeze(-1) + offsets).flatten()

		# Compute local indices: [num_examples, num_negatives, npc]
		false_local_base = false_positions * npc
		false_local_indices = (false_local_base.unsqueeze(-1) + offsets).flatten()

		# Get addresses
		example_indices_false = repeat_interleave(arange(num_examples, device=device), num_negatives * npc)
		false_addresses = addresses[example_indices_false, false_local_indices]

		false_bits = zeros(num_examples * num_negatives * npc, dtype=torch_bool, device=device)

		# ===== Write TRUE first, then FALSE =====
		modified = 0

		if self.memory.explore_batch(true_neuron_indices, true_addresses, true_bits, allow_override):
			modified += len(true_neuron_indices)

		if self.memory.explore_batch(false_neuron_indices, false_addresses, false_bits, allow_override):
			modified += len(false_neuron_indices)

		return modified

	def get_cluster_connectivity(self, cluster_idx: int) -> Tensor:
		"""
		Get connectivity for a specific cluster.

		Args:
			cluster_idx: Cluster index (0 to num_clusters-1)

		Returns:
			[neurons_per_cluster, bits_per_neuron] connectivity tensor
		"""
		start = cluster_idx * self.neurons_per_cluster
		end = start + self.neurons_per_cluster
		return self.memory.connections[start:end]

	def set_cluster_connectivity(self, cluster_idx: int, connections: Tensor) -> None:
		"""
		Set connectivity for a specific cluster.

		Args:
			cluster_idx: Cluster index (0 to num_clusters-1)
			connections: [neurons_per_cluster, bits_per_neuron] connectivity tensor
		"""
		assert connections.shape == (self.neurons_per_cluster, self.memory.n_bits_per_neuron)
		start = cluster_idx * self.neurons_per_cluster
		end = start + self.neurons_per_cluster
		self.memory.connections[start:end] = connections

	def reset_memory(self) -> None:
		"""
		Reset all memory cells to EMPTY, preserving connectivity.

		Clears all learned mappings while keeping the connectivity pattern.
		Useful for retraining after connectivity optimization.
		"""
		self.memory.reset()

	# =========================================================================
	# Serialization
	# =========================================================================

	def get_config(self) -> dict:
		"""Get configuration dict for model recreation."""
		return {
			'total_input_bits': self.memory.total_input_bits,
			'num_clusters': self.num_clusters,
			'neurons_per_cluster': self.neurons_per_cluster,
			'bits_per_neuron': self.memory.n_bits_per_neuron,
			'use_hashing': self.memory.use_hashing,
			'hash_size': self.memory.memory_size if self.memory.use_hashing else 1024,
		}

	@classmethod
	def from_config(cls, config: dict) -> "RAMClusterLayer":
		"""Create a RAMClusterLayer from a configuration dict."""
		return cls(**config)

	def save(self, path: str) -> None:
		"""Save model to file."""
		from wnn.ram.core.serialization import save_model
		save_model(self, path)

	@classmethod
	def load(cls, path: str, device: str = 'cpu') -> "RAMClusterLayer":
		"""Load model from file."""
		from wnn.ram.core.serialization import load_model
		return load_model(path, model_class=cls, device=device)

	# =========================================================================
	# Rust Acceleration
	# =========================================================================

	def forward_rust(self, input_bits: Tensor) -> Tensor:
		"""
		Rust-accelerated forward pass (16-core parallel via rayon).

		Uses numpy arrays for fast data transfer and rayon for parallel processing.

		Args:
			input_bits: [batch, total_input_bits] boolean tensor

		Returns:
			[batch, num_clusters] float tensor of probabilities
		"""
		try:
			import ram_accelerator
		except ImportError:
			raise RuntimeError("Rust accelerator not available. Build with: cd src/wnn/ram/strategies/accelerator && maturin develop --release")

		from torch import from_numpy
		import numpy as np

		# Ensure 2D input
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]

		# Use numpy arrays for fast data transfer
		input_bits_np = input_bits.flatten().to(dtype=long).numpy().astype(np.uint8)
		connections_np = self.memory.connections.flatten().numpy()
		memory_words_np = self.memory.memory_words.flatten().numpy()

		# Call Rust with numpy arrays
		probs_flat = ram_accelerator.ramlm_forward_batch_numpy(
			input_bits_np,
			connections_np,
			memory_words_np,
			batch_size,
			self.memory.total_input_bits,
			self.total_neurons,
			self.memory.n_bits_per_neuron,
			self.neurons_per_cluster,
			self.num_clusters,
			self.memory.words_per_neuron,
		)

		# Reshape back to [batch, num_clusters]
		probs = from_numpy(np.array(probs_flat, dtype=np.float32)).view(batch_size, self.num_clusters)
		return probs

	def forward_metal(self, input_bits: Tensor) -> Tensor:
		"""
		Metal GPU-accelerated forward pass (40 cores on M4 Max).

		Uses Metal compute shaders for massive parallelism. Each (example, cluster)
		pair is computed by a separate GPU thread.

		Args:
			input_bits: [batch, total_input_bits] boolean tensor

		Returns:
			[batch, num_clusters] float tensor of probabilities
		"""
		try:
			import ram_accelerator
		except ImportError:
			raise RuntimeError("Rust accelerator not available. Build with: cd src/wnn/ram/strategies/accelerator && maturin develop --release")

		if not ram_accelerator.ramlm_metal_available():
			raise RuntimeError("Metal GPU not available on this system")

		from torch import from_numpy
		import numpy as np

		# Ensure 2D input
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]

		# Use numpy arrays for fast data transfer (zero-copy)
		input_bits_np = input_bits.flatten().to(dtype=long).numpy().astype(np.uint8)
		connections_np = self.memory.connections.flatten().numpy()
		memory_words_np = self.memory.memory_words.flatten().numpy()

		# Call Metal GPU with numpy arrays
		probs_flat = ram_accelerator.ramlm_forward_batch_metal_numpy(
			input_bits_np,
			connections_np,
			memory_words_np,
			batch_size,
			self.memory.total_input_bits,
			self.total_neurons,
			self.memory.n_bits_per_neuron,
			self.neurons_per_cluster,
			self.num_clusters,
			self.memory.words_per_neuron,
		)

		# Reshape back to [batch, num_clusters]
		probs = from_numpy(np.array(probs_flat, dtype=np.float32)).view(batch_size, self.num_clusters)
		return probs

	def forward_hybrid(self, input_bits: Tensor) -> Tensor:
		"""
		Hybrid CPU+GPU forward pass (56 cores: 16 CPU + 40 GPU on M4 Max).

		Splits work between CPU (rayon) and GPU (Metal) for maximum throughput.

		Args:
			input_bits: [batch, total_input_bits] boolean tensor

		Returns:
			[batch, num_clusters] float tensor of probabilities
		"""
		try:
			import ram_accelerator
		except ImportError:
			raise RuntimeError("Rust accelerator not available.")

		from torch import from_numpy
		import numpy as np

		# Ensure 2D input
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]

		# Use numpy arrays for fast data transfer
		input_bits_np = input_bits.flatten().to(dtype=long).numpy().astype(np.uint8)
		connections_np = self.memory.connections.flatten().numpy()
		memory_words_np = self.memory.memory_words.flatten().numpy()

		# Call hybrid CPU+GPU
		probs_flat = ram_accelerator.ramlm_forward_batch_hybrid_numpy(
			input_bits_np,
			connections_np,
			memory_words_np,
			batch_size,
			self.memory.total_input_bits,
			self.total_neurons,
			self.memory.n_bits_per_neuron,
			self.neurons_per_cluster,
			self.num_clusters,
			self.memory.words_per_neuron,
		)

		# Reshape back to [batch, num_clusters]
		probs = from_numpy(np.array(probs_flat, dtype=np.float32)).view(batch_size, self.num_clusters)
		return probs

	def train_multi_examples_rust(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Rust-accelerated training for multiple examples at once.

		Uses rayon for parallel processing and atomic memory writes.
		Typically 10-50x faster than PyTorch.

		Args:
			input_bits: [num_examples, total_input_bits] input patterns
			true_clusters: [num_examples] cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		try:
			import ram_accelerator
		except ImportError:
			raise RuntimeError("Rust accelerator not available. Build with: cd src/wnn/ram/strategies/accelerator && maturin develop --release")

		from torch import tensor as torch_tensor, int64

		num_examples = input_bits.shape[0]
		num_negatives = false_clusters.shape[1]

		# Flatten tensors for Rust
		input_bits_flat = input_bits.flatten().bool().tolist()
		true_clusters_list = true_clusters.tolist()
		false_clusters_flat = false_clusters.flatten().tolist()
		connections_flat = self.memory.connections.flatten().tolist()
		memory_words_flat = self.memory.memory_words.flatten().tolist()

		# Call Rust
		modified, new_memory = ram_accelerator.ramlm_train_batch(
			input_bits_flat,
			true_clusters_list,
			false_clusters_flat,
			connections_flat,
			memory_words_flat,
			num_examples,
			self.memory.total_input_bits,
			self.total_neurons,
			self.memory.n_bits_per_neuron,
			self.neurons_per_cluster,
			num_negatives,
			self.memory.words_per_neuron,
			allow_override,
		)

		# Update memory from Rust result
		self.memory.memory_words[:] = torch_tensor(new_memory, dtype=int64).view_as(self.memory.memory_words)

		return modified

	def train_multi_examples_rust_numpy(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Numpy-accelerated Rust training (FAST - avoids .tolist() overhead).

		Uses numpy arrays for near-zero-copy data transfer to Rust.
		Typically 5-10x faster than train_multi_examples_rust for large batches.

		Args:
			input_bits: [num_examples, total_input_bits] input patterns
			true_clusters: [num_examples] cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		try:
			import ram_accelerator
		except ImportError:
			raise RuntimeError("Rust accelerator not available. Build with: cd src/wnn/ram/strategies/accelerator && maturin develop --release")

		import numpy as np
		from torch import tensor as torch_tensor, int64

		num_examples = input_bits.shape[0]

		# Convert to numpy arrays (fast path for Rust)
		input_bits_np = input_bits.flatten().bool().numpy().astype(np.uint8)
		true_clusters_np = true_clusters.numpy().astype(np.int64)
		false_clusters_np = false_clusters.flatten().numpy().astype(np.int64)
		connections_np = self.memory.connections.flatten().numpy().astype(np.int64)
		memory_words_np = self.memory.memory_words.flatten().numpy().astype(np.int64)

		# Call Rust with numpy arrays
		modified, new_memory = ram_accelerator.ramlm_train_batch_numpy(
			input_bits_np,
			true_clusters_np,
			false_clusters_np,
			connections_np,
			memory_words_np,
			num_examples,
			self.memory.total_input_bits,
			self.total_neurons,
			self.memory.n_bits_per_neuron,
			self.neurons_per_cluster,
			false_clusters.shape[1],  # num_negatives
			self.memory.words_per_neuron,
			allow_override,
		)

		# Update memory from Rust result
		self.memory.memory_words[:] = torch_tensor(new_memory, dtype=int64).view_as(self.memory.memory_words)

		return modified

	def train_auto(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Auto-optimized training that picks the best backend.

		Automatically selects between:
		- PyTorch (train_batch): Best for small batches (< 100 examples)
		- Rust numpy (train_multi_examples_rust_numpy): Best for large batches

		Args:
			input_bits: [num_examples, total_input_bits] or [total_input_bits] input patterns
			true_clusters: [num_examples] or scalar cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		# Handle single example case
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)
		if true_clusters.ndim == 0:
			true_clusters = true_clusters.unsqueeze(0)
		if false_clusters.ndim == 1:
			false_clusters = false_clusters.unsqueeze(0)

		batch_size = input_bits.shape[0]

		# Threshold for switching to Rust
		RUST_THRESHOLD = 100

		if batch_size < RUST_THRESHOLD:
			# Small batch: use PyTorch
			return self.train_batch(input_bits, true_clusters, false_clusters, allow_override)
		else:
			# Large batch: try Rust numpy, fall back to PyTorch
			try:
				return self.train_multi_examples_rust_numpy(
					input_bits, true_clusters, false_clusters, allow_override
				)
			except (ImportError, RuntimeError):
				# Rust not available, fall back to PyTorch
				return self.train_batch(input_bits, true_clusters, false_clusters, allow_override)
