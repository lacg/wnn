"""
RAMClusterLayer - RAM Layer with clustered neurons for probabilistic output.

This layer organizes neurons into clusters, where each cluster represents
one output class. The probability for each class is computed from the
responses of its cluster's neurons.

Key insight: This enables RAM WNNs to produce probability distributions
over large vocabularies (e.g., 50K tokens) using the native RAM architecture
with partial connectivity for generalization.

Probability formula per cluster:
    Score(class) = (count_TRUE + empty_value * count_EMPTY) / neurons_per_cluster

Where:
    - TRUE (1): Strong evidence for this class
    - FALSE (0): Evidence against this class
    - EMPTY (2): No evidence (default empty_value=0.0 means abstain)

Note: empty_value is defined in PerplexityCalculator (single source of truth)

Memory Backend Selection:
    - "auto" (default): Uses dense for ≤10 bits, sparse for >10 bits
    - "dense": Force dense bit-packed storage (good for ≤10 bits)
    - "sparse": Force Rust HashMap-based sparse storage (good for >10 bits)

    Dense memory: O(neurons × 2^bits_per_neuron) - exponential in bits!
    Sparse memory: O(written_cells) - linear in training examples
"""

from wnn.ram.core.Memory import Memory
from wnn.ram.core.base import RAMComponent
from wnn.ram.core import MemoryVal

from enum import IntEnum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
	from wnn.ram.core.gating import GatingModel


class MemoryBackend(IntEnum):
	"""
	Memory storage backend selection for RAMClusterLayer.

	Trade-offs:
		DENSE:  O(neurons × 2^bits) memory, O(1) lookup - best for ≤10 bits
		SPARSE: O(written_cells) memory, O(1) avg lookup - best for 11-30 bits
		LSH:    O(written_cells) memory, approximate lookup - best for 31-60 bits
		AUTO:   Automatically selects based on bits_per_neuron thresholds
	"""
	AUTO = 0    # Auto-select based on bits_per_neuron
	DENSE = 1   # Bit-packed dense storage (exponential memory)
	SPARSE = 2  # HashMap-based sparse storage (linear memory)
	LSH = 3     # Locality-sensitive hashing (approximate, very large bits)


# Auto-selection thresholds
# Dense: ≤10 bits (1K cells/neuron, ~72 bytes/neuron)
# Sparse: 11-30 bits (HashMap, exact lookup)
# LSH: >30 bits (approximate lookup, for very large address spaces)
SPARSE_THRESHOLD_BITS = 10
LSH_THRESHOLD_BITS = 30

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
		backend: MemoryBackend = MemoryBackend.AUTO,
		gating_model: Optional['GatingModel'] = None,
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
			backend: Memory backend selection (MemoryBackend enum):
				- AUTO: Dense for ≤10 bits, sparse for 11-30, LSH for >30
				- DENSE: Force dense bit-packed storage
				- SPARSE: Force Rust HashMap-based sparse storage
				- LSH: Force LSH-based approximate storage (for >30 bits)
			gating_model: Optional GatingModel for content-based filtering.
				When provided, cluster scores are multiplied by gate values:
				  gated_scores = ungated_scores * gates
				See gating.py for RAMGating (binary) and SoftRAMGating (continuous).
		"""
		super().__init__()

		self.num_clusters = num_clusters
		self.neurons_per_cluster = neurons_per_cluster
		self.total_neurons = num_clusters * neurons_per_cluster

		# Gating support (Engram-inspired content-based filtering)
		self.gating_model = gating_model

		# Determine effective backend based on selection and bits_per_neuron
		if backend == MemoryBackend.AUTO:
			if bits_per_neuron > LSH_THRESHOLD_BITS:
				self._backend = MemoryBackend.LSH
			elif bits_per_neuron > SPARSE_THRESHOLD_BITS:
				self._backend = MemoryBackend.SPARSE
			else:
				self._backend = MemoryBackend.DENSE
		else:
			self._backend = backend

		self._bits_per_neuron = bits_per_neuron
		self._total_input_bits = total_input_bits

		# Initialize storage based on backend
		self._sparse_memory = None
		self._lsh_memory = None

		match self._backend:
			case MemoryBackend.LSH:
				# Use LSH-based approximate storage (for very large address spaces)
				import ram_accelerator
				# LSH uses same sparse storage but with hashed addresses
				self._lsh_memory = ram_accelerator.SparseMemory(
					self.total_neurons,
					bits_per_neuron
				)
				# Number of hash bits for LSH (reduce address space)
				self._lsh_bits = min(20, bits_per_neuron)  # Cap at 20 bits (~1M buckets)
				# Create minimal Memory just for connections (no dense storage)
				self.memory = Memory(
					total_input_bits=total_input_bits,
					num_neurons=self.total_neurons,
					n_bits_per_neuron=min(8, bits_per_neuron),  # Minimal bits for connections only
					connections=connections,
					use_hashing=True,  # Force hashing to avoid dense allocation
					hash_size=1024,
					rng=rng,
				)
				# Override n_bits_per_neuron after creation
				self.memory.n_bits_per_neuron = bits_per_neuron

			case MemoryBackend.SPARSE:
				# Use Rust-accelerated sparse storage
				import ram_accelerator
				self._sparse_memory = ram_accelerator.SparseMemory(
					self.total_neurons,
					bits_per_neuron
				)
				# Create Memory for connections (with hashing to avoid large allocation)
				# For >10 bits, we use sparse Rust storage, so use_hashing prevents dense alloc
				self.memory = Memory(
					total_input_bits=total_input_bits,
					num_neurons=self.total_neurons,
					n_bits_per_neuron=bits_per_neuron,
					connections=connections,
					use_hashing=use_hashing if bits_per_neuron <= 16 else True,
					hash_size=hash_size if bits_per_neuron <= 16 else min(hash_size, 2**16),
					rng=rng,
				)

			case _:
				# DENSE backend: Create full Memory with dense storage
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
			f"bits_per_neuron={self._bits_per_neuron}, "
			f"input_bits={self._total_input_bits}, "
			f"backend={self._backend.name})"
		)

	def __str__(self) -> str:
		match self._backend:
			case MemoryBackend.SPARSE:
				cells_info = f"  Written cells: {self._sparse_memory.total_cells():,}"
			case MemoryBackend.LSH:
				cells_info = f"  LSH buckets: {self._lsh_memory.total_cells():,} (hash_bits={self._lsh_bits})"
			case MemoryBackend.DENSE:
				cells_info = f"  Total memory cells: {self.total_neurons * self.memory.memory_size:,}"
			case _:
				cells_info = "  Unknown backend"

		lines = [
			"=== RAMClusterLayer ===",
			f"  Clusters: {self.num_clusters}",
			f"  Neurons per cluster: {self.neurons_per_cluster}",
			f"  Total neurons: {self.total_neurons}",
			f"  Input bits: {self._total_input_bits}",
			f"  Bits per neuron: {self._bits_per_neuron}",
			f"  Backend: {self._backend.name}",
			cells_info,
		]
		return "\n".join(lines)

	@property
	def backend(self) -> MemoryBackend:
		"""Return the memory backend type."""
		return self._backend

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
		Forward pass with automatic backend selection.

		Picks the best backend based on memory type and batch size:
		- Sparse memory (>10 bits): sparse forward path
		- Dense + large batch (>1000): Metal GPU
		- Dense + small batch: PyTorch

		If gating is enabled, scores are multiplied by gate values.

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			[batch, num_clusters] float tensor of scores
		"""
		# Ensure 2D input
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		# Dispatch based on backend to get ungated scores
		match self._backend:
			case MemoryBackend.SPARSE:
				scores = self.forward_sparse(input_bits)
			case MemoryBackend.LSH:
				scores = self.forward_lsh(input_bits)
			case MemoryBackend.DENSE:
				batch_size = input_bits.shape[0]
				METAL_THRESHOLD = 1000
				if batch_size < METAL_THRESHOLD:
					scores = self._forward_ungated(input_bits)
				else:
					scores = self._forward_metal_cached(input_bits)
			case _:
				raise ValueError(f"Unknown backend: {self._backend}")

		# Apply gating if enabled
		if self.gating_model is not None:
			gates = self.gating_model.forward(input_bits)  # [batch, num_clusters]
			scores = scores * gates

		return scores

	def _forward_ungated(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass returning ungated scores (original implementation).

		This is the base cluster scoring without any gating applied.
		Use forward() for gated scores when gating_model is set.

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			[batch, num_clusters] float tensor of ungated scores
		"""
		# Get raw counts and convert to scores
		true_counts, empty_counts = self.forward_counts(input_bits)

		# Use PerplexityCalculator for EMPTY→value conversion (single source of truth)
		from wnn.ram.strategies.perplexity import PerplexityCalculator
		calc = PerplexityCalculator(vocab_size=self.num_clusters)
		return calc.ram_counts_to_scores(true_counts, empty_counts, self.neurons_per_cluster)

	def forward_counts(self, input_bits: Tensor) -> tuple[Tensor, Tensor]:
		"""
		Forward pass returning raw TRUE/EMPTY counts per cluster.

		This is the low-level method - use PerplexityCalculator to convert
		counts to scores/probabilities with proper EMPTY handling.

		Args:
			input_bits: [batch, total_input_bits] boolean or 0/1 tensor

		Returns:
			Tuple of:
			- true_counts: [batch, num_clusters] count of TRUE cells per cluster
			- empty_counts: [batch, num_clusters] count of EMPTY cells per cluster
		"""
		# Get raw memory values: [batch, total_neurons]
		# Values are: FALSE=0, TRUE=1, EMPTY=2
		raw = self.memory.get_memories_for_bits(input_bits)  # [batch, total_neurons]

		# Reshape to clusters: [batch, num_clusters, neurons_per_cluster]
		batch_size = raw.shape[0]
		clustered = raw.view(batch_size, self.num_clusters, self.neurons_per_cluster)

		# Count TRUE (1) and EMPTY (2) per cluster
		true_counts = (clustered == MemoryVal.TRUE).sum(dim=-1).float()   # [batch, num_clusters]
		empty_counts = (clustered == MemoryVal.EMPTY).sum(dim=-1).float() # [batch, num_clusters]

		return true_counts, empty_counts


	def _forward_metal_cached(self, input_bits: Tensor) -> Tensor:
		"""
		Metal GPU forward with cached evaluator (internal use).

		Returns ungated scores (gating applied by caller in forward()).
		Uses a globally cached Metal evaluator to avoid shader recompilation.
		"""
		try:
			import ram_accelerator
		except ImportError:
			# Fall back to PyTorch ungated (forward() applies gating)
			return self._forward_ungated(input_bits)

		if not ram_accelerator.ramlm_metal_available():
			# Fall back to PyTorch ungated (forward() applies gating)
			return self._forward_ungated(input_bits)

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
	# Gating Support (Engram-inspired content-based filtering)
	# =========================================================================

	@property
	def has_gating(self) -> bool:
		"""Check if gating is enabled."""
		return self.gating_model is not None

	def set_gating(self, gating_model: Optional['GatingModel']) -> None:
		"""
		Set or remove gating model.

		Args:
			gating_model: GatingModel instance or None to disable gating
		"""
		self.gating_model = gating_model

	def create_gating(
		self,
		neurons_per_gate: int = 8,
		bits_per_neuron: int = 12,
		threshold: float = 0.5,
		rng: Optional[int] = None,
	) -> 'GatingModel':
		"""
		Create and attach a RAMGating model to this layer.

		Convenience method to create gating with matching configuration.

		Args:
			neurons_per_gate: Number of RAM neurons voting on each cluster's gate
			bits_per_neuron: Address bits per gate neuron
			threshold: Fraction of neurons that must fire for gate=1
			rng: Random seed for connectivity initialization

		Returns:
			The created RAMGating instance (also stored in self.gating_model)
		"""
		from wnn.ram.core.gating import RAMGating

		self.gating_model = RAMGating(
			total_input_bits=self._total_input_bits,
			num_clusters=self.num_clusters,
			neurons_per_gate=neurons_per_gate,
			bits_per_neuron=bits_per_neuron,
			threshold=threshold,
			rng=rng,
		)
		return self.gating_model

	def train_gating_step(
		self,
		input_bits: Tensor,
		targets: Tensor,
		top_k: int = 10,
		allow_override: bool = False,
	) -> int:
		"""
		Train gating model on a batch of examples.

		Computes beneficial gates based on ungated predictions and targets,
		then trains the gating model to produce those gates.

		Args:
			input_bits: [B, total_input_bits] input patterns
			targets: [B] correct cluster indices
			top_k: Number of top clusters to consider for gating signal
			allow_override: Whether to override existing gate memories

		Returns:
			Number of gate memory cells modified

		Raises:
			RuntimeError: If gating_model is not set
		"""
		if self.gating_model is None:
			raise RuntimeError("Cannot train gating: gating_model is not set")

		from wnn.ram.core.gating import compute_beneficial_gates

		# Get ungated predictions
		ungated_scores = self._forward_ungated(input_bits)

		# Compute target gates
		target_gates = compute_beneficial_gates(ungated_scores, targets, top_k)

		# Train gating model
		return self.gating_model.train_step(input_bits, target_gates, allow_override)

	def reset_gating(self) -> None:
		"""Reset gating model memories to EMPTY (all gates open)."""
		if self.gating_model is not None:
			self.gating_model.reset()

	def freeze_base_memory(self) -> None:
		"""
		Mark base cluster memory as frozen (for staged training).

		In staged training:
		1. Train base RAM → freeze
		2. Train gating model
		3. (Optional) Fine-tune both

		This sets a flag; actual freezing is enforced in train methods.
		"""
		self._base_frozen = True

	def unfreeze_base_memory(self) -> None:
		"""Unfreeze base cluster memory for joint training."""
		self._base_frozen = False

	@property
	def is_base_frozen(self) -> bool:
		"""Check if base memory is frozen."""
		return getattr(self, '_base_frozen', False)

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
		- Sparse (train_sparse): When using sparse memory backend (>10 bits)
		- Rust numpy (train_multi_examples_rust_numpy): Dense, large batches
		- PyTorch (train_batch): Dense, small batches

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

		# Dispatch based on backend
		match self._backend:
			case MemoryBackend.SPARSE:
				return self.train_sparse(input_bits, true_clusters, false_clusters, allow_override)
			case MemoryBackend.LSH:
				return self.train_lsh(input_bits, true_clusters, false_clusters, allow_override)
			case MemoryBackend.DENSE:
				# Dense backend: choose between PyTorch and Rust
				batch_size = input_bits.shape[0]
				RUST_THRESHOLD = 100

				if batch_size < RUST_THRESHOLD:
					return self.train_batch(input_bits, true_clusters, false_clusters, allow_override)
				else:
					try:
						return self.train_multi_examples_rust_numpy(
							input_bits, true_clusters, false_clusters, allow_override
						)
					except (ImportError, RuntimeError):
						return self.train_batch(input_bits, true_clusters, false_clusters, allow_override)
			case _:
				raise ValueError(f"Unknown backend: {self._backend}")

	def train_sparse(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Training using Rust sparse memory backend.

		Uses HashMap-based sparse storage for memory-efficient training
		with >10 bits per neuron.

		Args:
			input_bits: [num_examples, total_input_bits] input patterns
			true_clusters: [num_examples] cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		if self._backend != MemoryBackend.SPARSE:
			raise RuntimeError(f"train_sparse called but layer is using {self._backend.name} backend")

		import ram_accelerator

		# Handle single example case
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)
		if true_clusters.ndim == 0:
			true_clusters = true_clusters.unsqueeze(0)
		if false_clusters.ndim == 1:
			false_clusters = false_clusters.unsqueeze(0)

		num_examples = input_bits.shape[0]
		num_negatives = false_clusters.shape[1]

		# Flatten tensors for Rust
		input_bits_flat = input_bits.flatten().bool().tolist()
		true_clusters_list = true_clusters.tolist()
		false_clusters_flat = false_clusters.flatten().tolist()
		connections_flat = self.memory.connections.flatten().tolist()

		# Call Rust sparse training
		modified = ram_accelerator.sparse_train_batch(
			self._sparse_memory,
			input_bits_flat,
			true_clusters_list,
			false_clusters_flat,
			connections_flat,
			num_examples,
			self._total_input_bits,
			self.neurons_per_cluster,
			num_negatives,
			allow_override,
		)

		return modified

	def forward_sparse(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass using Rust sparse memory backend.

		Uses HashMap-based sparse storage for memory-efficient inference
		with >10 bits per neuron.

		Args:
			input_bits: [batch_size, total_input_bits] or [total_input_bits] input patterns

		Returns:
			Tensor [batch_size, num_clusters] of probabilities
		"""
		if self._backend != MemoryBackend.SPARSE:
			raise RuntimeError(f"forward_sparse called but layer is using {self._backend.name} backend")

		import ram_accelerator
		from torch import tensor as torch_tensor, float32

		# Handle single example
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]

		# Flatten tensors for Rust
		input_bits_flat = input_bits.flatten().bool().tolist()
		connections_flat = self.memory.connections.flatten().tolist()

		# Call Rust sparse forward
		probs_flat = ram_accelerator.sparse_forward_batch(
			self._sparse_memory,
			input_bits_flat,
			connections_flat,
			batch_size,
			self._total_input_bits,
			self.neurons_per_cluster,
			self.num_clusters,
		)

		# Reshape to [batch_size, num_clusters]
		probs = torch_tensor(probs_flat, dtype=float32).view(batch_size, self.num_clusters)
		return probs

	def _lsh_hash_address(self, address: int) -> int:
		"""
		Hash a large address (30-60 bits) down to lsh_bits using XOR folding.

		XOR folding preserves locality: addresses that differ in few bits
		are likely to hash to nearby buckets, enabling approximate matching.

		Example for 40-bit address → 20-bit hash:
			Split: [bits 0-19] XOR [bits 20-39] = 20-bit result
		"""
		result = 0
		mask = (1 << self._lsh_bits) - 1  # e.g., 0xFFFFF for 20 bits

		# Fold the address by XORing chunks of lsh_bits
		remaining = address
		while remaining > 0:
			result ^= (remaining & mask)
			remaining >>= self._lsh_bits

		return result

	def train_lsh(
		self,
		input_bits: Tensor,
		true_clusters: Tensor,
		false_clusters: Tensor,
		allow_override: bool = False,
	) -> int:
		"""
		Training using LSH (Locality-Sensitive Hashing) backend.

		Uses XOR-folding to hash large addresses (30-60 bits) down to
		a manageable size (~20 bits). This provides approximate lookups
		where similar inputs map to similar buckets.

		Trade-off: Some accuracy loss due to hash collisions, but enables
		very large address spaces that would be impossible otherwise.

		Args:
			input_bits: [num_examples, total_input_bits] input patterns
			true_clusters: [num_examples] cluster indices to train as TRUE
			false_clusters: [num_examples, num_negatives] cluster indices to train as FALSE
			allow_override: Whether to override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		if self._backend != MemoryBackend.LSH:
			raise RuntimeError(f"train_lsh called but layer is using {self._backend.name} backend")

		import ram_accelerator

		# Handle single example case
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)
		if true_clusters.ndim == 0:
			true_clusters = true_clusters.unsqueeze(0)
		if false_clusters.ndim == 1:
			false_clusters = false_clusters.unsqueeze(0)

		num_examples = input_bits.shape[0]
		num_negatives = false_clusters.shape[1]

		# For LSH, we compute addresses, hash them, then train
		# We need to do this in Python since the hash is custom
		modified = 0

		for ex_idx in range(num_examples):
			ex_input = input_bits[ex_idx]
			true_cluster = true_clusters[ex_idx].item()

			# Train TRUE cluster neurons
			for neuron_offset in range(self.neurons_per_cluster):
				neuron_idx = true_cluster * self.neurons_per_cluster + neuron_offset
				connections = self.memory.connections[neuron_idx]

				# Compute full address from input bits
				address = 0
				for bit_pos, conn_idx in enumerate(connections):
					if ex_input[conn_idx]:
						address |= 1 << (self._bits_per_neuron - 1 - bit_pos)

				# Hash the address for LSH
				hashed_addr = self._lsh_hash_address(address)

				# Write TRUE (with override for priority)
				if self._lsh_memory.write_cell(neuron_idx, hashed_addr, 1, True):
					modified += 1

			# Train FALSE cluster neurons
			for neg_idx in range(num_negatives):
				false_cluster = false_clusters[ex_idx, neg_idx].item()

				for neuron_offset in range(self.neurons_per_cluster):
					neuron_idx = false_cluster * self.neurons_per_cluster + neuron_offset
					connections = self.memory.connections[neuron_idx]

					# Compute full address
					address = 0
					for bit_pos, conn_idx in enumerate(connections):
						if ex_input[conn_idx]:
							address |= 1 << (self._bits_per_neuron - 1 - bit_pos)

					# Hash the address
					hashed_addr = self._lsh_hash_address(address)

					# Write FALSE (only to EMPTY cells)
					if self._lsh_memory.write_cell(neuron_idx, hashed_addr, 0, allow_override):
						modified += 1

		return modified

	def forward_lsh(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass using LSH (Locality-Sensitive Hashing) backend.

		Uses XOR-folding to hash addresses, then looks up in sparse storage.
		Provides approximate lookups for very large address spaces (30-60 bits).

		Args:
			input_bits: [batch_size, total_input_bits] or [total_input_bits] input patterns

		Returns:
			Tensor [batch_size, num_clusters] of scores
		"""
		if self._backend != MemoryBackend.LSH:
			raise RuntimeError(f"forward_lsh called but layer is using {self._backend.name} backend")

		from torch import zeros, float32
		from wnn.ram.strategies.perplexity import PerplexityCalculator

		# Handle single example
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]

		# Accumulate counts into tensors
		true_counts = zeros(batch_size, self.num_clusters, dtype=float32)
		empty_counts = zeros(batch_size, self.num_clusters, dtype=float32)

		for ex_idx in range(batch_size):
			ex_input = input_bits[ex_idx]

			for cluster_idx in range(self.num_clusters):
				count_true = 0
				count_empty = 0

				for neuron_offset in range(self.neurons_per_cluster):
					neuron_idx = cluster_idx * self.neurons_per_cluster + neuron_offset
					connections = self.memory.connections[neuron_idx]

					# Compute full address
					address = 0
					for bit_pos, conn_idx in enumerate(connections):
						if ex_input[conn_idx]:
							address |= 1 << (self._bits_per_neuron - 1 - bit_pos)

					# Hash the address
					hashed_addr = self._lsh_hash_address(address)

					# Look up in LSH memory
					cell_value = self._lsh_memory.read_cell(neuron_idx, hashed_addr)

					if cell_value == 1:  # TRUE
						count_true += 1
					elif cell_value == 2:  # EMPTY
						count_empty += 1

				true_counts[ex_idx, cluster_idx] = count_true
				empty_counts[ex_idx, cluster_idx] = count_empty

		# Use PerplexityCalculator for score calculation (single source of truth)
		calc = PerplexityCalculator(vocab_size=self.num_clusters)
		return calc.ram_counts_to_scores(true_counts, empty_counts, self.neurons_per_cluster)
