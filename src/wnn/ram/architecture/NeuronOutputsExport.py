"""NeuronOutputsExport: per-row, per-neuron outputs of a genome trained exactly as final test evaluation trains it.

Python form of the Rust `NeuronOutputs` (IDSCacheWrapper.export_neuron_outputs,
IDS-57 arm E, worker ABI 29). READ-ONLY: the cells are read out of the very
export the production scorer scored, so a learned / weighted readout fitted on
them is fitted on exactly the outputs behind the reported score.

Rust computes everything; this class only holds the arrays and decodes the
2-bit packing (a storage format, not model logic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class NeuronOutputsExport:
	cells: np.ndarray
	"""uint8 [rows, N] effective cell per logical neuron (genome order, cluster-major);
	when `packed`, [rows, ceil(N/4)] with 2 bits per neuron, LSB-first."""
	scores: np.ndarray
	"""float64 [rows, K] production-scorer score of every cluster (column 0 is the
	score evaluate_at_thresholds / export_trained_layer report)."""
	labels: np.ndarray
	"""int64 [rows] target class."""
	neurons_per_cluster: list[int]
	cell_weights: Optional[list[float]]
	"""Weight of each cell code 0..3; None for QSR / PLN (per-example coin)."""
	packed: bool

	@classmethod
	def from_tuple(cls, raw: tuple) -> NeuronOutputsExport:
		"""IDSCacheWrapper.export_neuron_outputs's flat tuple."""
		cells, scores, labels, npc, weights, packed = raw
		return cls(
			cells=cells, scores=scores, labels=labels, neurons_per_cluster=[int(n) for n in npc],
			cell_weights=None if weights is None else [float(w) for w in weights], packed=bool(packed),
		)

	@property
	def total_neurons(self) -> int:
		return sum(self.neurons_per_cluster)

	def unpacked_cells(self) -> np.ndarray:
		"""uint8 [rows, N] cells (decodes the 2-bit packing; a view when unpacked)."""
		if not self.packed:
			return self.cells
		shifts = np.arange(4, dtype=np.uint8) * 2
		wide = (self.cells[:, :, None] >> shifts) & 0x3
		return wide.reshape(self.cells.shape[0], -1)[:, :self.total_neurons]

	def cluster_slices(self) -> list[slice]:
		"""Column slice of each cluster's neurons in `unpacked_cells()`."""
		starts = np.concatenate([[0], np.cumsum(self.neurons_per_cluster)])
		return [slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:])]
