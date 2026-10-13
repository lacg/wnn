"""TrainedLayerExport: a genome trained exactly as final validation trains it, as PHYSICAL units.

Python form of the Rust `TrainedLayer` (IDSCacheWrapper.export_trained, IDS-37
stage 7). The FPGA exporter's only input: units come out of the very export the
scorer scored, so a deployed table is the validated model; `eval_scores` are
the production scorer's, for the fidelity gate against the DB row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from wnn.ram.core.UnitGroupingView import UnitGroupingView
from wnn.ram.footprint import Footprint


@dataclass(frozen=True)
class TrainedUnit:
	"""One distinct neuron: its connections and every (address, cell) it reads.
	Dense units list all 2^bits addresses; an absent sparse key reads `miss_cell`."""
	cluster: int
	bits: int
	connections: list[int]
	keys: np.ndarray
	cells: np.ndarray


@dataclass(frozen=True)
class TrainedLayerExport:
	grouping: UnitGroupingView
	units: list[TrainedUnit]
	miss_cell: int
	cell_weight_quarters: Optional[list[int]]
	"""Adder weight per cell code in quarters; None = stochastic mode (QSR/PLN: no RTL)."""
	footprint: Footprint
	eval_scores: list[float]
	val_scores: Optional[list[float]]

	@classmethod
	def from_tuple(cls, raw: tuple) -> TrainedLayerExport:
		"""IDSCacheWrapper.export_trained's flat tuple (ABI 17)."""
		index_of, reps, offsets, members, units, miss_cell, weights, footprint, eval_scores, val_scores = raw
		return cls(
			grouping=UnitGroupingView(index_of, reps, offsets, members),
			units=[cls._unit(u) for u in units],
			miss_cell=int(miss_cell),
			cell_weight_quarters=None if weights is None else [int(w) for w in weights],
			footprint=Footprint.from_tuple(footprint),
			eval_scores=list(eval_scores),
			val_scores=None if val_scores is None else list(val_scores),
		)

	@staticmethod
	def _unit(raw: tuple) -> TrainedUnit:
		cluster, bits, connections, keys, cells = raw
		return TrainedUnit(
			cluster=int(cluster), bits=int(bits), connections=[int(c) for c in connections],
			keys=np.asarray(keys, dtype=np.uint64), cells=np.frombuffer(bytes(cells), dtype=np.uint8),
		)

	@property
	def counts(self) -> list[int]:
		"""Logical neurons per unit (the RTL adder weight of each unit)."""
		return [int(c) for c in self.grouping.counts()]
