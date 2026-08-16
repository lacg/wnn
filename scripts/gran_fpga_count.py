#!/usr/bin/env python
"""FPGA deployment-size counter for a saved controller winner (schema-2 yaml.gz).

Sparse sizing per feedback_sparse_fpga_size: size = POPULATED cells only (value !=
EMPTY_U8=2 — uniform across modes, see recurrent_genome.py:210), NEVER dense
n*2^bits. bits/cell: BINARY antagonist E/I = 1 (same cell layout as QUAD, the E/I
split reinterprets the SAME `levels` output cells — storage genuinely halves);
TERNARY/QUAD_WEIGHTED/QSR/PLN = 2 (QSR is QUAD's lattice with a stochastic read;
PLN shares TERNARY's cells — cell_mode.rs).

Prints a summary line + writes <winner>.fpga.json with counts, histogram and
total memory bits. Emits the raw value histogram so the EMPTY convention is
auditable from the artifact itself.

Usage: gran_fpga_count.py <winner.yaml.gz>
"""
import argparse
import json
import sys
from collections import Counter

from wnn.control.phased_ga import _ctl_load

EMPTY_U8 = 2  # uniform EMPTY sentinel (recurrent_genome.py:210; neuron_memory.rs:28)


def _bits_per_cell(mode: str) -> int:
	return 1 if str(mode).upper() == "BINARY" else 2


def _cells_of(genome):
	"""(state_universe, state_values, output_universe, output_values) from either a
	RecurrentArchGenome (.cells MemoryPayload) or a bare MemoryGenome."""
	src = getattr(genome, "cells", None) or genome
	for attr in ("state_universe", "state_values", "output_universe", "output_values"):
		if getattr(src, attr, None) is None:
			return None
	return (src.state_universe, src.state_values, src.output_universe, src.output_values)


def _layer_stats(universe, values):
	hist = Counter(int(v) for v in values)
	populated = sum(n for v, n in hist.items() if v != EMPTY_U8)
	return {
		"universe": len(universe),
		"populated": populated,
		"histogram": {str(k): v for k, v in sorted(hist.items())},
	}


def main():
	p = argparse.ArgumentParser(description="Sparse FPGA size of a saved controller winner")
	p.add_argument("path")
	w = p.parse_args()

	blob = _ctl_load(w.path)
	spec = blob["spec"]
	genome = blob.get("best_genome")
	mode = str(getattr(spec, "memory_mode", "?"))
	sn = getattr(spec, "state_neurons", "?")
	sb = getattr(spec, "state_bits_per_neuron", "?")
	ob = getattr(spec, "output_bits_per_neuron", "?")
	on = getattr(spec, "output_neurons", "?")

	cells = _cells_of(genome) if genome is not None else None
	if cells is None:
		print(f"[fpga-count] {w.path}: best_genome has NO cells payload — "
		      f"was this a grid-only save (cells discarded)? Nothing to count.")
		return 1

	su, sv, ou, ov = cells
	bpc = _bits_per_cell(mode)
	st = _layer_stats(su, sv)
	ot = _layer_stats(ou, ov)
	populated = st["populated"] + ot["populated"]
	total_bits = populated * bpc

	out = {
		"path": w.path,
		"mode": mode,
		"spec": {"state_neurons": sn, "state_bits_per_neuron": sb, "output_bits_per_neuron": ob,
		         "output_neurons": on},
		"bits_per_cell": bpc,
		"state": st,
		"output": ot,
		"populated_cells": populated,
		"total_memory_bits": total_bits,
		"total_memory_bytes": (total_bits + 7) // 8,
	}
	jpath = w.path + ".fpga.json"
	with open(jpath, "w") as f:
		json.dump(out, f, indent=1)

	print(f"[FPGA] mode={mode} sn={sn} sb={sb} ob={ob} on={on}  "
	      f"populated={populated} (state {st['populated']}/{st['universe']}, "
	      f"output {ot['populated']}/{ot['universe']})  "
	      f"x {bpc}b/cell = {total_bits} bits ({out['total_memory_bytes']} bytes)  -> {jpath}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
