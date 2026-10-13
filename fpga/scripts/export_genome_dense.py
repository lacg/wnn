#!/usr/bin/env python3
"""Export a trained WNN genome as per-UNIT dense lookup tables (small address widths).

Same training + fidelity gate as export_genome.py (FlowLayerTrainer); each
physical unit becomes one full 2^bits ROM of cell codes (an address the unit
never stored reads the export's miss cell). metadata.json carries the unit
counts the RTL adder weights by.

Usage:
    python export_genome_dense.py --flow-id 1022
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_genome import parse_args, train_for_export  # noqa: E402
from export_metadata import build_metadata, footprint_line  # noqa: E402
from export_test_vectors import write_test_vectors  # noqa: E402

MAX_DENSE_BITS = 16


def dense_table(unit, miss_cell: int) -> np.ndarray:
	"""Full 2^bits table of cell codes; absent keys read the miss cell."""
	table = np.full(1 << unit.bits, miss_cell, dtype=np.uint8)
	table[unit.keys.astype(np.int64)] = unit.cells
	return table


def write_dense_units(output_dir: Path, trained) -> dict:
	units = trained.layer.units
	too_wide = [u.bits for u in units if u.bits > MAX_DENSE_BITS]
	if too_wide:
		raise ValueError(f"dense export needs <= {MAX_DENSE_BITS}-bit units (got {max(too_wide)}); use export_genome.py")
	mem_dir = output_dir / "mem"
	mem_dir.mkdir(parents=True, exist_ok=True)
	for i, u in enumerate(units):
		table = dense_table(u, trained.layer.miss_cell)
		(mem_dir / f"unit_{i:03d}.mem").write_text("".join(f"{v:02x}\n" for v in table.tolist()))
	meta = build_metadata(trained, "dense")
	meta["dense_bytes"] = sum(1 << u.bits for u in units)
	(output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
	(output_dir / "connections.json").write_text(json.dumps([u.connections for u in units]))
	return meta


def main():
	args, db, output_dir = parse_args("Export a WNN genome as per-unit dense FPGA tables", "_dense")
	trained = train_for_export(args, db)
	meta = write_dense_units(output_dir, trained)
	if args.num_vectors > 0:
		write_test_vectors(output_dir / "testbench", trained, meta, args.num_vectors)
	print(f"\nDense FPGA export ({output_dir}):\n  {footprint_line(meta)}")
	print(f"  ROM: {meta['dense_bytes']:,} entries over {meta['physical_neurons']} units (O(1) lookup)")


if __name__ == "__main__":
	main()
