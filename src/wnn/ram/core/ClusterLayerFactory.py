"""ClusterLayerFactory: the one place a uniform cluster layer is built.

Selects the multiplicity layer exactly when it is exact AND pays: neurons train
every example (neuron_sample_rate >= 1.0, so identical neurons hold identical
memory) and the connectivity actually carries duplicate neurons. Otherwise the
plain RAMClusterLayer (docs/ids_multiplicity_layer_design.md §4.2).
"""

from __future__ import annotations

from wnn.ram.core.ClusterLayerConfig import ClusterLayerConfig
from wnn.ram.core.RAMClusterLayer import RAMClusterLayer
from wnn.ram.core.UnitGroupingView import UnitGroupingView


class ClusterLayerFactory:

	@staticmethod
	def create(config: ClusterLayerConfig) -> RAMClusterLayer:
		grouping = ClusterLayerFactory._grouping(config)
		if grouping is None or grouping.is_identity:
			return ClusterLayerFactory._plain(config)
		from wnn.ram.core.MultiplicityRAMClusterLayer import MultiplicityRAMClusterLayer
		return MultiplicityRAMClusterLayer(
			total_input_bits=config.total_input_bits,
			num_clusters=config.num_clusters,
			neurons_per_cluster=config.neurons_per_cluster,
			bits_per_neuron=config.bits_per_neuron,
			connections=config.connections,
			use_hashing=config.use_hashing,
			hash_size=config.hash_size,
			rng=config.rng,
			backend=config.backend,
			gating_model=config.gating_model,
			grouping=grouping,
		)

	@staticmethod
	def _grouping(config: ClusterLayerConfig) -> UnitGroupingView | None:
		"""The layer's grouping when the multiplicity layer may apply, else None."""
		if config.neuron_sample_rate < 1.0 or config.connections is None:
			return None
		return UnitGroupingView.of_layer(config.connections, config.num_clusters, config.neurons_per_cluster)

	@staticmethod
	def _plain(config: ClusterLayerConfig) -> RAMClusterLayer:
		return RAMClusterLayer(
			total_input_bits=config.total_input_bits,
			num_clusters=config.num_clusters,
			neurons_per_cluster=config.neurons_per_cluster,
			bits_per_neuron=config.bits_per_neuron,
			connections=config.connections,
			use_hashing=config.use_hashing,
			hash_size=config.hash_size,
			rng=config.rng,
			backend=config.backend,
			gating_model=config.gating_model,
		)
