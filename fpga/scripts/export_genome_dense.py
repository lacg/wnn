#!/usr/bin/env python3
"""Export a trained WNN genome as dense lookup tables for FPGA.

For small address widths (8-12 bits), stores the FULL 2^b table per neuron.
No binary search needed — direct O(1) ROM lookup.

Usage:
    python export_genome_dense.py --flow-id 1022
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "wnn"))
sys.path.insert(0, str(Path(__file__).parent))
from export_genome import get_genome_connections, load_dataset, train_and_extract_sparse


def sparse_to_dense(sparse_neurons, addr_bits, default_value=1):
	"""Convert sparse (keys, values) to dense tables."""
	num_entries = 2 ** addr_bits
	dense_neurons = []
	for keys, values in sparse_neurons:
		table = np.full(num_entries, default_value, dtype=np.uint8)
		for k, v in zip(keys, values):
			if k < num_entries:
				table[int(k)] = v
		dense_neurons.append(table)
	return dense_neurons


def write_dense_files(output_dir, connections, dense_neurons, genome_info, params, addr_bits):
	"""Write dense .mem files and metadata."""
	output_dir = Path(output_dir)
	mem_dir = output_dir / "mem"
	mem_dir.mkdir(parents=True, exist_ok=True)

	n_neurons = len(connections)
	num_entries = 2 ** addr_bits

	# Dense memory files (hex, one value per line)
	for n_idx, table in enumerate(dense_neurons):
		with open(mem_dir / f"neuron_{n_idx:03d}.mem", "w") as f:
			for v in table:
				f.write(f"{v:02x}\n")

	# Metadata
	total_bytes = n_neurons * num_entries  # 1 byte per entry
	meta = {
		"flow_id": params.get("_flow_id"),
		"genome_hash": genome_info["hash"],
		"phase": genome_info["phase"],
		"total_neurons": n_neurons,
		"addr_bits": addr_bits,
		"entries_per_neuron": num_entries,
		"input_bits": params["_input_bits"],
		"n_bits_thermometer": params["ids_n_bits"],
		"dense_bytes": total_bytes,
		"dataset": params.get("ids_dataset", "unsw-nb15"),
		"split": params.get("ids_split", "random"),
		"mode": "dense",
	}
	with open(output_dir / "metadata.json", "w") as f:
		json.dump(meta, f, indent=2)

	# Connections
	with open(output_dir / "connections.json", "w") as f:
		json.dump(connections, f)

	print(f"\n{'='*50}")
	print(f"Dense FPGA Export Summary")
	print(f"{'='*50}")
	print(f"  Neurons: {n_neurons}, Address bits: {addr_bits}")
	print(f"  Entries per neuron: {num_entries:,}")
	print(f"  Dense size: {total_bytes/1024:.1f} KB ({total_bytes/1024/1024:.3f} MB)")
	print(f"  Mode: O(1) direct lookup (no binary search)")
	print(f"  Files: {mem_dir}/neuron_*.mem")


def main():
	parser = argparse.ArgumentParser(description="Export dense WNN genome for FPGA")
	parser.add_argument("--flow-id", type=int, required=True)
	parser.add_argument("--genome-type", default="best_fitness")
	parser.add_argument("--db", default="db/wnn.db")
	parser.add_argument("--output", default=None)
	args = parser.parse_args()

	db_path = str(Path(__file__).resolve().parents[2] / args.db)

	print(f"Loading genome from flow {args.flow_id}...")
	params, genome, connections = get_genome_connections(db_path, args.flow_id, args.genome_type)

	n_bits = params["ids_n_bits"]
	dataset_name = params.get("ids_dataset", "unsw-nb15")
	split = params.get("ids_split", "random")
	feature_sel = params.get("ids_feature_selection", "top20")
	addr_bits = len(connections[0])

	print(f"  {genome['total_neurons']}n × {addr_bits}b, {n_bits}-bit thermometer")
	print(f"  Dense table: {2**addr_bits:,} entries per neuron")

	print(f"\nLoading {dataset_name}...")
	dataset = load_dataset(dataset_name, split, n_bits, feature_sel)
	X_train = dataset.X_train
	y_train = dataset.y_train_binary
	params["_input_bits"] = X_train.shape[1]
	params["_flow_id"] = args.flow_id

	print(f"\nTraining {genome['total_neurons']} neurons...")
	sparse_neurons = train_and_extract_sparse(X_train, y_train, connections, addr_bits)

	print(f"\nConverting to dense tables ({2**addr_bits} entries each)...")
	dense_neurons = sparse_to_dense(sparse_neurons, addr_bits)

	output_dir = Path(args.output) if args.output else Path(__file__).resolve().parents[1] / "export" / f"flow_{args.flow_id}_dense"
	write_dense_files(output_dir, connections, dense_neurons, genome, params, addr_bits)


if __name__ == "__main__":
	main()
