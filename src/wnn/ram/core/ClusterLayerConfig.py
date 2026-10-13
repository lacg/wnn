"""ClusterLayerConfig: the typed construction spec of a uniform cluster layer.

Consumed by ClusterLayerFactory, which picks RAMClusterLayer or
MultiplicityRAMClusterLayer (docs/ids_multiplicity_layer_design.md §4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from torch import Tensor

from wnn.ram.core.RAMClusterLayer import MemoryBackend

if TYPE_CHECKING:
	from wnn.ram.core.gating import GatingModel


@dataclass
class ClusterLayerConfig:
	total_input_bits: int
	num_clusters: int
	neurons_per_cluster: int = 5
	bits_per_neuron: int = 10
	connections: Optional[Tensor] = None
	"""LOGICAL connectivity [num_clusters * neurons_per_cluster, bits_per_neuron];
	None = random (rng), which never carries duplicates."""
	backend: MemoryBackend = MemoryBackend.AUTO
	neuron_sample_rate: float = 1.0
	"""Python cluster layers train every neuron on every example (rate 1.0). The
	field keeps the selection rule identical to Rust's should training ever
	subsample: below 1.0, copies may train differently, so the plain layer."""
	rng: Optional[int] = None
	use_hashing: bool = False
	hash_size: int = 1024
	gating_model: Optional["GatingModel"] = None
