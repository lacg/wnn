"""
BitwiseRAMClusterLayer - RAMClusterLayer with multi-label training.

Subclass of RAMClusterLayer that adds a train() method for bitwise
(multi-label) training. Instead of training one TRUE cluster + some FALSE
clusters per example, ALL clusters are trained per example based on a
binary label matrix.

Forward pass: inherited from RAMClusterLayer (unchanged).
Training: train() writes TRUE/FALSE to ALL clusters per example.

Hierarchy:
	RAMClusterBase
	 ├── RAMClusterLayer          (existing — unchanged)
	 │    └── BitwiseRAMClusterLayer  (THIS — adds multi-label train)
	 └── TieredRAMClusterLayer    (existing — unchanged)
"""

from typing import Optional

from torch import Tensor, long

from wnn.ram.core.RAMClusterLayer import RAMClusterLayer, MemoryBackend


class BitwiseRAMClusterLayer(RAMClusterLayer):
	"""RAMClusterLayer with multi-label training for bitwise output models.

	Each cluster represents one output bit. Training labels ALL clusters per
	example (TRUE where bit=1, FALSE where bit=0) instead of the standard
	one-hot-true + sampled-false approach.

	Forward pass is inherited unchanged from RAMClusterLayer.
	"""

	def train(
		self,
		input_bits: Tensor,
		target_bits: Tensor,
		allow_override: bool = False,
	) -> int:
		"""Multi-label training: label ALL clusters per example.

		Args:
			input_bits: [N, total_input_bits] input patterns
			target_bits: [N, num_clusters] binary labels (0/1 per cluster)
			allow_override: Whether FALSE can override existing non-EMPTY cells

		Returns:
			Number of memory cells modified
		"""
		# Ensure 2D
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)
		if target_bits.ndim == 1:
			target_bits = target_bits.unsqueeze(0)

		# Dispatch based on backend
		match self._backend:
			case MemoryBackend.SPARSE:
				return self._train_bitwise_sparse(input_bits, target_bits, allow_override)
			case _:
				return self._train_bitwise_dense(input_bits, target_bits, allow_override)

	def _train_bitwise_dense(
		self,
		input_bits: Tensor,
		target_bits: Tensor,
		allow_override: bool,
	) -> int:
		"""Bitwise training using dense (bit-packed) memory backend."""
		import ram_accelerator
		import numpy as np
		from torch import tensor as torch_tensor, int64

		num_examples = input_bits.shape[0]

		input_bits_np = input_bits.flatten().bool().numpy().astype(np.uint8)
		target_bits_np = target_bits.flatten().to(dtype=long).numpy().astype(np.uint8)
		connections_np = self.memory.connections.flatten().numpy().astype(np.int64)
		memory_words_np = self.memory.memory_words.flatten().numpy().astype(np.int64)

		modified, new_memory = ram_accelerator.ramlm_bitwise_train_batch_numpy(
			input_bits_np,
			target_bits_np,
			connections_np,
			memory_words_np,
			num_examples,
			self._total_input_bits,
			self.total_neurons,
			self._bits_per_neuron,
			self.neurons_per_cluster,
			self.num_clusters,
			self.memory.words_per_neuron,
			allow_override,
		)

		# Update memory from Rust result
		self.memory.memory_words[:] = torch_tensor(new_memory, dtype=int64).view_as(self.memory.memory_words)

		return modified

	def _train_bitwise_sparse(
		self,
		input_bits: Tensor,
		target_bits: Tensor,
		allow_override: bool,
	) -> int:
		"""Bitwise training using sparse (DashMap) memory backend."""
		import ram_accelerator

		num_examples = input_bits.shape[0]

		input_bits_flat = input_bits.flatten().bool().tolist()
		target_bits_flat = target_bits.flatten().to(dtype=long).tolist()
		connections_flat = self.memory.connections.flatten().tolist()

		modified = ram_accelerator.sparse_bitwise_train_batch(
			self._sparse_memory,
			input_bits_flat,
			target_bits_flat,
			connections_flat,
			num_examples,
			self._total_input_bits,
			self.neurons_per_cluster,
			self.num_clusters,
			allow_override,
		)

		return modified
