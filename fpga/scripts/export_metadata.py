"""metadata.json for a unit-emitted export (shared by the sparse and dense exporters).

Carries BOTH footprints (logical = the genome, physical = the distinct units
emitted), the unit grouping (index_of / counts), the RTL adder contract
(cell_weight_quarters, miss_cell, acc denominator = 4 x logical neurons), and
the run parameters the tables were trained with.
"""

from __future__ import annotations


def build_metadata(trained, mode: str) -> dict:
	layer, source, settings = trained.layer, trained.source, trained.settings
	fp = layer.footprint
	return {
		"flow_id": source.flow_id,
		"genome_hash": source.genome_hash,
		"phase": source.phase_name,
		"mode": mode,
		"input_bits": input_bits(trained.dataset.X_test),
		"n_bits_thermometer": settings.n_bits,
		"dataset": settings.dataset_name,
		"split": settings.split,
		"neuron_sample_rate": settings.neuron_sample_rate,
		"memory_mode": settings.memory_mode_name,
		"order_independent_train": settings.order_independent_train,
		"coverage_aware": settings.coverage_aware,
		"logical_neurons": fp.logical_neurons,
		"physical_neurons": fp.physical_neurons,
		"logical_cells": fp.logical_cells,
		"physical_cells": fp.physical_cells,
		"index_of": layer.grouping.index_of_list(),
		"counts": layer.counts,
		"unit_bits": [u.bits for u in layer.units],
		"unit_entries": [int(len(u.keys)) for u in layer.units],
		"miss_cell": layer.miss_cell,
		"cell_weight_quarters": layer.cell_weight_quarters,
		"score_denominator_quarters": 4 * fp.logical_neurons,
		"fidelity_modes_verified": trained.modes_verified,
		"threshold_quarters": threshold_quarters(source.threshold_metadata, fp.logical_neurons),
	}


def input_bits(matrix) -> int:
	"""BIT width of an encoded matrix (a packed matrix's shape[1] is bytes)."""
	return int(getattr(matrix, "total_bits", None) or matrix.shape[1])


def threshold_quarters(threshold_metadata: dict, logical_neurons: int) -> dict:
	"""Per recorded mode: the integer adder threshold T with class = (sum >= T),
	from the score rule `score >= t` and score = sum / (4 N). Boundary cases are
	verified per test vector in simulation (expected class from the Rust score)."""
	import math
	out = {}
	for mode, recorded in threshold_metadata.items():
		t = 0.5 if mode == "fixed_05" else (recorded.get("threshold") if isinstance(recorded, dict) else None)
		if t is not None:
			out[mode] = {"threshold": t, "sum_ge": max(0, math.ceil(t * 4 * logical_neurons - 1e-9))}
	return out


def footprint_line(meta: dict) -> str:
	"""`500n x 34b (logical) | 137n · 1.21 Mcells (physical)` — replaces the old BRAM-fit line
	(the deployed designs are combinational LUT logic, BRAM 0; no LUT saving is
	claimed until re-synthesis)."""
	bits = meta["unit_bits"]
	b = f"{min(bits)}b" if min(bits) == max(bits) else f"{min(bits)}-{max(bits)}b"
	return (f"{meta['logical_neurons']}n x {b} · {meta['logical_cells']:,} cells (logical) | "
		f"{meta['physical_neurons']}n · {meta['physical_cells']:,} cells (physical)")
