"""Footprint: what a trained genome costs, LOGICAL and PHYSICAL — always both.

Mirrors ram_core::footprint::Footprint (IDS-37 stage 3,
docs/ids_multiplicity_layer_design.md §5). Logical = what the genome says
(copies included); physical = the distinct neurons actually trained and held.
Cells: a sparse neuron counts its trained keys, a dense neuron its 2^bits array.
Computed in Rust only (GenomeExport::footprint) — Python just carries it.

Usage:
	from wnn.ram.footprint import Footprint
"""

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Footprint:
	logical_neurons: int
	physical_neurons: int
	logical_cells: int
	physical_cells: int

	@classmethod
	def from_tuple(cls, raw: Optional[Sequence[int]]) -> Optional["Footprint"]:
		"""The accelerator's flat tuple (ABI 17): (logical_neurons,
		physical_neurons, logical_cells, physical_cells). None passes through."""
		if raw is None:
			return None
		logical_neurons, physical_neurons, logical_cells, physical_cells = (int(v) for v in raw)
		return cls(logical_neurons, physical_neurons, logical_cells, physical_cells)

	@classmethod
	def from_payload(cls, payload: Optional[dict]) -> Optional["Footprint"]:
		"""Inverse of to_payload (dashboard JSON); None or incomplete -> None."""
		if not payload or any(payload.get(k) is None for k in cls.__dataclass_fields__):
			return None
		return cls(**{k: int(payload[k]) for k in cls.__dataclass_fields__})

	def to_payload(self) -> dict:
		return {
			"logical_neurons": self.logical_neurons,
			"physical_neurons": self.physical_neurons,
			"logical_cells": self.logical_cells,
			"physical_cells": self.physical_cells,
		}

	def render(self, bits_per_neuron: Sequence[int]) -> str:
		"""`500n × 34b · 4.41 Mcells (logical) | 137n · 1.21 Mcells (physical)`."""
		return (
			f"{self.logical_neurons}n × {_bits_label(bits_per_neuron)} · {_cells_label(self.logical_cells)} (logical) | "
			f"{self.physical_neurons}n · {_cells_label(self.physical_cells)} (physical)"
		)


def _bits_label(bits_per_neuron: Sequence[int]) -> str:
	if not bits_per_neuron:
		return "?b"
	lo, hi = min(bits_per_neuron), max(bits_per_neuron)
	return f"{hi}b" if lo == hi else f"{lo}-{hi}b"


def _cells_label(cells: int) -> str:
	for scale, unit in ((1e9, "Gcells"), (1e6, "Mcells"), (1e3, "Kcells")):
		if cells >= scale:
			return f"{cells / scale:.2f} {unit}"
	return f"{cells} cells"
