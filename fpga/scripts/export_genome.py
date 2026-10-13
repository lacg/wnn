#!/usr/bin/env python3
"""Export a trained WNN genome as FPGA-ready per-UNIT sparse tables.

The genome is trained EXACTLY as the worker's final validation trained it
(FlowLayerTrainer: the worker's IdsEvaluatorFactory + IDSCacheWrapper.export_trained,
the flow's real neuron_sample_rate / memory_mode / OI / class weights) and must
reproduce the recorded validation row bit for bit (FidelityGate) before any
file is written. Identical neurons (same connections; at rate >= 1.0 they hold
identical memory) are emitted ONCE: one table per physical unit, and
metadata.json carries each unit's logical count (the RTL adder weight).

Usage:
    python export_genome.py --flow-id 973 --genome-type best_fitness
"""

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_metadata import build_metadata, footprint_line  # noqa: E402
from export_test_vectors import write_test_vectors  # noqa: E402
from flow_genome_source import FlowGenomeSource  # noqa: E402
from flow_layer_trainer import FlowLayerTrainer  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def write_sparse_units(output_dir: Path, trained) -> dict:
	"""metadata.json, connections.json (per unit), sparse_memory.bin/.txt (per unit)."""
	output_dir.mkdir(parents=True, exist_ok=True)
	meta = build_metadata(trained, "sparse")
	units = trained.layer.units
	(output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
	(output_dir / "connections.json").write_text(json.dumps([u.connections for u in units]))
	_write_sparse_bin(output_dir / "sparse_memory.bin", units)
	_write_sparse_txt(output_dir / "sparse_memory.txt", units, meta["counts"])
	return meta


def _write_sparse_bin(path: Path, units) -> None:
	"""Header: num_units (u32), per-unit entry count (u32); then (u64 key, u8 cell) pairs."""
	with open(path, "wb") as f:
		f.write(struct.pack("<I", len(units)))
		for u in units:
			f.write(struct.pack("<I", len(u.keys)))
		for u in units:
			for k, c in zip(u.keys.tolist(), u.cells.tolist()):
				f.write(struct.pack("<QB", k, c))


def _write_sparse_txt(path: Path, units, counts) -> None:
	with open(path, "w") as f:
		for i, u in enumerate(units):
			f.write(f"# Unit {i}: {len(u.keys)} entries, x{counts[i]} logical neurons, connections {u.connections}\n")
			f.writelines(f"{k} {c}\n" for k, c in zip(u.keys.tolist(), u.cells.tolist()))
			f.write("\n")


def parse_args(description: str, default_suffix: str):
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("--flow-id", type=int, required=True)
	parser.add_argument("--genome-type", default="best_fitness",
		help="best_fitness, best_f1, best_acc, best_ce, best_fpr")
	parser.add_argument("--point", default="final", help="validation_point of the recorded row to reproduce")
	parser.add_argument("--db", default="db/wnn.db", help="Dashboard DB (opened read-only)")
	parser.add_argument("--output", default=None)
	parser.add_argument("--phase", default=None, choices=["ga_neurons", "grid_search"],
		help="Force a phase. Default tries GA Neurons then Grid Search.")
	parser.add_argument("--num-vectors", type=int, default=1000, help="HW-vs-Rust test vectors (0 = none)")
	args = parser.parse_args()
	db = Path(args.db) if Path(args.db).is_absolute() else REPO / args.db
	out = Path(args.output) if args.output else REPO / "fpga" / "export" / f"flow_{args.flow_id}{default_suffix}"
	return args, str(db), out


def train_for_export(args, db: str):
	print(f"Loading genome from flow {args.flow_id} ({args.genome_type}, {args.point})...")
	source = FlowGenomeSource.read(db, args.flow_id, args.genome_type, args.phase, args.point)
	print(f"  Genome {source.genome_hash} ({source.phase_name}): {len(source.bits_per_neuron)} neurons")
	return FlowLayerTrainer().train(source)


def main():
	args, db, output_dir = parse_args("Export a WNN genome as per-unit sparse FPGA tables", "")
	trained = train_for_export(args, db)
	meta = write_sparse_units(output_dir, trained)
	if args.num_vectors > 0:
		write_test_vectors(output_dir / "testbench", trained, meta, args.num_vectors)
	print(f"\nFPGA export ({output_dir}):\n  {footprint_line(meta)}")
	print(f"  Fidelity: {meta['fidelity_modes_verified']} recorded threshold modes reproduced exactly")


if __name__ == "__main__":
	main()
