"""
Genome abstraction for optimizer strategies.

Provides a unified Protocol for genome types used by GA/TS optimizers.
Any object with a clone() method can be used as a genome.

Usage:
	from wnn.ram.strategies.connectivity.genome import Genome, TensorGenome

	# Create a TensorGenome wrapper
	genome = TensorGenome(tensor)
	cloned = genome.clone()

	# Or use ClusterGenome directly (already has clone())
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	cluster = ClusterGenome(bits_per_neuron=[16]*10 + [12]*8, neurons_per_cluster=[10, 8])
"""

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from torch import Tensor


@runtime_checkable
class Genome(Protocol):
	"""
	Protocol for genome types used by optimizer strategies.

	Any object with a clone() method can be used as a genome.
	The clone() method must return a deep copy of the genome.
	"""

	def clone(self) -> 'Genome':
		"""Create a deep copy of the genome."""
		...


# TypeVar for generic genome operations
G = TypeVar('G', bound=Genome)


@dataclass
class TensorGenome:
	"""
	Genome wrapper for Tensor-based connectivity patterns.

	Wraps a PyTorch Tensor and provides the Genome protocol.
	Used for connectivity optimization (which input bits each neuron observes).

	The tensor can be:
	- 2D: shape (num_neurons, bits_per_neuron) for uniform architectures
	- 1D: flattened for tiered architectures with variable bits per neuron

	Attributes:
		connections: The connectivity tensor (which input bits each neuron observes)
		metadata: Optional metadata dict for extra context (e.g., neuron_offsets)
	"""
	connections: Tensor
	metadata: dict[str, Any] | None = None

	def clone(self) -> 'TensorGenome':
		"""Create a deep copy of the genome."""
		return TensorGenome(
			connections=self.connections.clone(),
			metadata=dict(self.metadata) if self.metadata else None,
		)

	@property
	def tensor(self) -> Tensor:
		"""Alias for connections (backward compatibility)."""
		return self.connections

	def __hash__(self) -> int:
		"""Hash based on tensor data (for tabu list)."""
		return hash(tuple(self.connections.flatten().tolist()))

	def __eq__(self, other: Any) -> bool:
		"""Equality based on tensor values."""
		if not isinstance(other, TensorGenome):
			return False
		return self.connections.equal(other.connections)

	def __repr__(self) -> str:
		shape = tuple(self.connections.shape)
		return f"TensorGenome(shape={shape})"
