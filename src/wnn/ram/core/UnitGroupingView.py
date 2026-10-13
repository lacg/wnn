"""UnitGroupingView: Python holder of a ram_core `UnitGrouping` (IDS-37 stage 6).

A layer's LOGICAL neurons (what the genome says, copies included) map onto
PHYSICAL units (one per distinct neuron: same cluster, bits and connections).
The grouping is derived in Rust only (`ram_accelerator.group_neurons`, i.e.
ram_core `NeuronKey` + `UnitGrouping::by_key`); this class never re-derives it,
it only holds the returned arrays and applies them to tensors.

Shared by MultiplicityRAMClusterLayer, the FPGA exporter and footprint reporting
(docs/ids_multiplicity_layer_design.md §2.1, §4.2).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class UnitGroupingView:
	"""Immutable logical -> unit map. Unit u's representative is the first
	logical neuron with its key; units are numbered in first-occurrence order."""

	def __init__(self, index_of: list[int], representatives: list[int], member_offsets: list[int], members: list[int]):
		self._index_of_list = list(index_of)
		self.index_of = torch.tensor(index_of, dtype=torch.long)
		self.representatives = torch.tensor(representatives, dtype=torch.long)
		self.member_offsets = torch.tensor(member_offsets, dtype=torch.long)
		self.members = torch.tensor(members, dtype=torch.long)
		self.is_representative = torch.zeros(len(index_of), dtype=torch.bool)
		self.is_representative[self.representatives] = True

	@classmethod
	def identity(cls, num_units: int) -> UnitGroupingView:
		iota = list(range(num_units))
		return cls(iota, iota, list(range(num_units + 1)), iota)

	@classmethod
	def of_genome(cls, bits_per_neuron: list[int], neurons_per_cluster: list[int], connections_flat: list[int]) -> UnitGroupingView:
		"""Group a flat genome (unpadded connections) in Rust."""
		from wnn.accel import require_accel
		return cls(*require_accel().group_neurons(list(bits_per_neuron), list(neurons_per_cluster), list(connections_flat)))

	@classmethod
	def of_layer(cls, connections: Tensor, num_clusters: int, neurons_per_cluster: int) -> UnitGroupingView:
		"""Group a uniform cluster layer: connections [num_clusters * npc, bits]."""
		bits = int(connections.shape[1])
		total = num_clusters * neurons_per_cluster
		return cls.of_genome([bits] * total, [neurons_per_cluster] * num_clusters, connections.flatten().tolist())

	@property
	def num_units(self) -> int:
		return int(self.index_of.shape[0])

	@property
	def num_groups(self) -> int:
		return int(self.representatives.shape[0])

	@property
	def is_identity(self) -> bool:
		return self.num_groups == self.num_units

	def counts(self) -> Tensor:
		"""Logical neurons per unit."""
		return self.member_offsets[1:] - self.member_offsets[:-1]

	def unit_members(self, unit: int) -> Tensor:
		"""The logical positions of one unit, ascending."""
		return self.members[self.member_offsets[unit]:self.member_offsets[unit + 1]]

	def index_of_list(self) -> list[int]:
		"""The logical -> unit map as a list (for Rust calls)."""
		return list(self._index_of_list)

	def expand_rows(self, per_unit: Tensor) -> Tensor:
		"""Unit-major rows -> logical-major rows (each copy gets its unit's row)."""
		return per_unit if self.is_identity else per_unit.index_select(0, self.index_of.to(per_unit.device))

	def gather_rows(self, per_logical: Tensor) -> Tensor:
		"""Logical-major rows -> unit-major rows (each unit keeps its representative's)."""
		return per_logical if self.is_identity else per_logical.index_select(0, self.representatives.to(per_logical.device))

	def expand_columns(self, per_unit: Tensor) -> Tensor:
		"""[batch, units] -> [batch, logical]."""
		return per_unit if self.is_identity else per_unit.index_select(1, self.index_of.to(per_unit.device))

	def write_slot(self, neuron: int) -> Optional[int]:
		"""The unit a logical neuron writes, or None for a copy (its unit's
		representative writes for it — copies receive identical writes)."""
		return self._index_of_list[neuron] if bool(self.is_representative[neuron]) else None

	def read_slot(self, neuron: int) -> int:
		return self._index_of_list[neuron]
