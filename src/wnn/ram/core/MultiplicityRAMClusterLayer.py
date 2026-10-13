"""MultiplicityRAMClusterLayer: a RAMClusterLayer that stores each DISTINCT neuron once.

A layer's logical neurons (copies included) can repeat: same cluster, bits and
connections. At neuron_sample_rate >= 1.0 such copies see the same writes and
hold the same memory, so this layer keeps one storage row per distinct neuron
(a unit) and resolves every logical neuron to its unit:

- reads: each logical neuron reads its unit's row (index_select on the unit axis);
- writes: only a unit's representative writes (copies' identical writes drop);
- Rust paths: logical rows are materialised for the call (dense), or the
  grouping is passed to the Rust sparse seams.

Every score, path and API is the plain layer's, bit for bit; only storage (and
training work) shrinks. Built by ClusterLayerFactory when it applies; the
grouping comes from ram_core via UnitGroupingView (never re-derived here).
docs/ids_multiplicity_layer_design.md §4.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from torch import Tensor

from wnn.ram.core.RAMClusterLayer import MemoryBackend, RAMClusterLayer
from wnn.ram.core.UnitGroupingView import UnitGroupingView

if TYPE_CHECKING:
	from wnn.ram.core.gating import GatingModel


class MultiplicityRAMClusterLayer(RAMClusterLayer):

	def __init__(
		self,
		total_input_bits: int,
		num_clusters: int,
		neurons_per_cluster: int,
		bits_per_neuron: int,
		connections: Tensor,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: Optional[int] = None,
		backend: MemoryBackend = MemoryBackend.AUTO,
		gating_model: Optional['GatingModel'] = None,
		grouping: Optional[UnitGroupingView] = None,
	):
		"""Same signature as RAMClusterLayer, but `connections` (LOGICAL,
		[num_clusters * neurons_per_cluster, bits_per_neuron]) is required.
		`grouping` = its precomputed UnitGroupingView (the factory passes it)."""
		# Set before the base constructor: the storage seams size from them.
		self._logical_conns = connections.clone().long()
		self._grouping = grouping or UnitGroupingView.of_layer(self._logical_conns, num_clusters, neurons_per_cluster)
		super().__init__(
			total_input_bits=total_input_bits, num_clusters=num_clusters, neurons_per_cluster=neurons_per_cluster,
			bits_per_neuron=bits_per_neuron, connections=self._logical_conns, use_hashing=use_hashing,
			hash_size=hash_size, rng=rng, backend=backend, gating_model=gating_model,
		)

	# ---- the five storage seams

	def _storage_neurons(self) -> int:
		return self._grouping.num_groups

	def _storage_connections(self, connections: Optional[Tensor]) -> Optional[Tensor]:
		return self._grouping.gather_rows(connections)

	def _logical_columns(self, per_storage: Tensor) -> Tensor:
		return self._grouping.expand_columns(per_storage)

	def _storage_writes(self, neuron_indices: Tensor) -> tuple[Tensor, Optional[Tensor]]:
		idx = neuron_indices.to('cpu', dtype=self._grouping.index_of.dtype)
		return self._grouping.index_of[idx].to(neuron_indices.device), self._grouping.is_representative[idx]

	def _unit_grouping(self) -> Optional[UnitGroupingView]:
		return self._grouping

	# ---- connectivity is the LOGICAL layout; changing it regroups

	@property
	def connections(self) -> Tensor:
		"""LOGICAL connectivity [total_neurons, bits_per_neuron] (stable tensor)."""
		return self._logical_conns

	@connections.setter
	def connections(self, value: Tensor) -> None:
		"""New connectivity: regroup and RESET storage (a connectivity change
		invalidates trained cells; the plain layer keeps meaningless stale ones)."""
		assert value.shape == (self.total_neurons, self._bits_per_neuron)
		self._logical_conns = value.clone().long()
		self._grouping = UnitGroupingView.of_layer(self._logical_conns, self.num_clusters, self.neurons_per_cluster)
		self._init_storage(self._logical_conns, None)

	def set_cluster_connectivity(self, cluster_idx: int, connections: Tensor) -> None:
		assert connections.shape == (self.neurons_per_cluster, self._bits_per_neuron)
		start = cluster_idx * self.neurons_per_cluster
		updated = self._logical_conns.clone()
		updated[start:start + self.neurons_per_cluster] = connections
		self.connections = updated

	# ---- serialization: the config carries the LOGICAL connections; the
	# grouping is recomputed on load (never serialized twice)

	def get_config(self) -> dict:
		config = super().get_config()
		config['connections'] = self._logical_conns
		config['backend'] = int(self._backend)
		return config

	@classmethod
	def from_config(cls, config: dict) -> RAMClusterLayer:
		from wnn.ram.core.ClusterLayerConfig import ClusterLayerConfig
		from wnn.ram.core.ClusterLayerFactory import ClusterLayerFactory
		return ClusterLayerFactory.create(ClusterLayerConfig(
			total_input_bits=config['total_input_bits'], num_clusters=config['num_clusters'],
			neurons_per_cluster=config['neurons_per_cluster'], bits_per_neuron=config['bits_per_neuron'],
			connections=config['connections'], backend=MemoryBackend(config['backend']),
			use_hashing=config['use_hashing'], hash_size=config['hash_size'],
		))

	def __repr__(self) -> str:
		return super().__repr__().replace("RAMClusterLayer(", f"MultiplicityRAMClusterLayer(units={self._grouping.num_groups}, ", 1)
