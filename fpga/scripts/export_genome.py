#!/usr/bin/env python3
"""Export a trained WNN genome to FPGA-ready sparse memory files.

Trains the genome on the dataset, extracts sparse (address, value) pairs
per neuron, and writes them as sorted arrays for BRAM initialization.

Usage:
    python export_genome.py --flow-id 973 --genome-type best_fitness
"""

import argparse
import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "wnn"))


def load_dataset(ids_dataset: str, ids_split: str, n_bits: int, feature_selection: str):
	"""Load and thermometer-encode an IDS dataset."""
	if ids_dataset == "cicids2017":
		from wnn.ids.cicids2017 import load_cicids2017
		return load_cicids2017(split=ids_split, n_bits=n_bits, feature_selection=feature_selection)
	elif ids_dataset == "unsw-nb15":
		from wnn.ids.dataset import load_unsw_nb15
		return load_unsw_nb15(split=ids_split, n_bits=n_bits, feature_selection=feature_selection)
	elif ids_dataset == "ciciot2023":
		from wnn.ids.ciciot2023 import load_ciciot2023
		return load_ciciot2023(split=ids_split, n_bits=n_bits, feature_selection=feature_selection)
	elif ids_dataset == "ciciot2023_full":
		# Full 46.7M-record CIC-IoT-2023 — same loader, different HF repo via dataset_size="full"
		from wnn.ids.ciciot2023 import load_ciciot2023
		return load_ciciot2023(split=ids_split, n_bits=n_bits, feature_selection=feature_selection, dataset_size="full")
	else:
		raise ValueError(f"Unknown dataset: {ids_dataset}")


def get_genome_connections(db_path: str, flow_id: int, genome_type: str = "best_fitness", phase: str | None = None):
	"""Get genome connections and config from the database.

	Matches experiments by `phase_type` column (grid_search / ga_neurons / ...)
	rather than by `name`, so it works for both PUB50 flows
	("Grid Search (neurons x bits)") and single-genome flows
	("Grid Search (1 point)"). If phase is None, prefers ga_neurons then
	falls back to grid_search.
	"""
	db = sqlite3.connect(db_path)

	# Get flow config
	row = db.execute("SELECT config_json FROM flows WHERE id = ?", (flow_id,)).fetchone()
	if not row:
		raise ValueError(f"Flow {flow_id} not found")
	config = json.loads(row[0])
	params = config["params"]

	# Build phase_type preference list
	valid_phase_types = {"ga_neurons", "grid_search"}
	if phase is not None:
		if phase not in valid_phase_types:
			raise ValueError(f"Unknown phase: {phase}. Use 'ga_neurons' or 'grid_search'.")
		phase_order = [phase]
	else:
		phase_order = ["ga_neurons", "grid_search"]

	# Get the genome from validation_summaries -> genomes by phase_type
	genome = None
	for pt in phase_order:
		row = db.execute("""
			SELECT vs.genome_hash, g.total_neurons, g.connections_json, g.tiers_json, e.name
			FROM validation_summaries vs
			JOIN experiments e ON vs.experiment_id = e.id AND e.phase_type = ?
			JOIN genomes g ON g.experiment_id = e.id AND g.config_hash = vs.genome_hash
			WHERE e.flow_id = ? AND vs.validation_point = 'final' AND vs.genome_type = ?
		""", (pt, flow_id, genome_type)).fetchone()
		if row:
			genome = {
				"hash": row[0],
				"total_neurons": row[1],
				"connections_csv": row[2],
				"tiers_json": row[3],
				"phase": row[4],
			}
			break

	if not genome:
		phase_label = phase if phase else "any phase"
		raise ValueError(f"No {genome_type} genome found for flow {flow_id} in {phase_label}")

	# Parse connections
	all_conns = list(map(int, genome["connections_csv"].split(",")))
	bits_per_neuron = len(all_conns) // genome["total_neurons"]
	connections = []
	for n in range(genome["total_neurons"]):
		start = n * bits_per_neuron
		connections.append(all_conns[start:start + bits_per_neuron])

	db.close()
	return params, genome, connections


def train_and_extract_sparse(X_train, y_train, connections, n_bits_addr):
	"""Train neurons and extract sparse (address, value) pairs.

	Uses QUAD_WEIGHTED training: for each neuron, gather addressed bits,
	form address, and nudge the cell toward the class label.

	Returns list of (sorted_keys, values) per neuron.
	"""
	n_neurons = len(connections)
	# QUAD_WEIGHTED cell values
	QUAD_FALSE = 0
	QUAD_WEAK_FALSE = 1
	QUAD_WEAK_TRUE = 2
	QUAD_TRUE = 3
	EMPTY = QUAD_WEAK_FALSE  # default for untrained cells (0.25 weight)

	sparse_neurons = []

	for n_idx in range(n_neurons):
		conn = connections[n_idx]
		# Gather the addressed bits for all training examples
		selected = X_train[:, conn]  # shape: (n_samples, n_bits_addr)

		# Convert bit patterns to integer addresses
		# Each row is a binary vector -> integer
		powers = 1 << np.arange(len(conn) - 1, -1, -1, dtype=np.uint64)
		addresses = selected.astype(np.uint64) @ powers

		# Build sparse memory via QUAD_WEIGHTED nudging
		memory = {}  # address -> cell_value (0-3)

		for i in range(len(X_train)):
			addr = int(addresses[i])
			label = int(y_train[i])  # 0 = normal, 1 = attack

			current = memory.get(addr, EMPTY)

			if label == 1:  # attack -> nudge toward TRUE
				if current < QUAD_TRUE:
					memory[addr] = current + 1
			else:  # normal -> nudge toward FALSE
				if current > QUAD_FALSE:
					memory[addr] = current - 1

		# Sort by key for binary search
		sorted_items = sorted(memory.items())
		keys = np.array([k for k, v in sorted_items], dtype=np.uint64)
		values = np.array([v for k, v in sorted_items], dtype=np.uint8)

		sparse_neurons.append((keys, values))

		if n_idx < 3 or n_idx == n_neurons - 1:
			features_spanned = len(set(c // 16 for c in conn))
			print(f"  Neuron {n_idx:>3}: {len(keys):>6,} entries, "
				  f"{features_spanned} features spanned")
		elif n_idx == 3:
			print(f"  ... ({n_neurons - 4} more neurons)")

	return sparse_neurons


def write_fpga_files(output_dir: Path, connections, sparse_neurons, genome_info, params):
	"""Write FPGA-ready files: memory init, connection map, and metadata."""
	output_dir.mkdir(parents=True, exist_ok=True)
	n_neurons = len(connections)

	# 1. Metadata JSON
	total_entries = sum(len(k) for k, v in sparse_neurons)
	meta = {
		"flow_id": params.get("_flow_id"),
		"genome_hash": genome_info["hash"],
		"phase": genome_info["phase"],
		"total_neurons": n_neurons,
		"bits_per_neuron": len(connections[0]),
		"input_bits": params["_input_bits"],
		"n_bits_thermometer": params["ids_n_bits"],
		"n_features": 20,  # top20
		"total_sparse_entries": total_entries,
		"per_neuron_entries": [len(k) for k, v in sparse_neurons],
		"sparse_bytes_u64_u8": total_entries * 9,
		"sparse_bytes_packed": total_entries * 6,
		"dataset": params.get("ids_dataset", "unsw-nb15"),
		"split": params.get("ids_split", "temporal"),
	}
	with open(output_dir / "metadata.json", "w") as f:
		json.dump(meta, f, indent=2)
	print(f"\nMetadata: {output_dir / 'metadata.json'}")

	# 2. Connection map (per neuron, list of input bit indices)
	with open(output_dir / "connections.json", "w") as f:
		json.dump(connections, f)
	print(f"Connections: {output_dir / 'connections.json'}")

	# 3. Sparse memory binary files (for BRAM initialization)
	# Format: per neuron, sorted (u64 key, u8 value) pairs
	with open(output_dir / "sparse_memory.bin", "wb") as f:
		# Header: num_neurons (u32), then per-neuron: count (u32)
		f.write(struct.pack("<I", n_neurons))
		for keys, values in sparse_neurons:
			f.write(struct.pack("<I", len(keys)))
		# Then all key-value data concatenated
		for keys, values in sparse_neurons:
			for k, v in zip(keys, values):
				f.write(struct.pack("<QB", int(k), int(v)))
	print(f"Sparse memory: {output_dir / 'sparse_memory.bin'}")

	# 4. Human-readable sparse memory (for verification)
	with open(output_dir / "sparse_memory.txt", "w") as f:
		for n_idx, (keys, values) in enumerate(sparse_neurons):
			f.write(f"# Neuron {n_idx}: {len(keys)} entries\n")
			f.write(f"# Connections: {connections[n_idx]}\n")
			for k, v in zip(keys, values):
				f.write(f"{k} {v}\n")
			f.write("\n")
	print(f"Readable: {output_dir / 'sparse_memory.txt'}")

	# Print summary
	total_bytes = total_entries * 9
	print(f"\n{'='*50}")
	print(f"FPGA Export Summary")
	print(f"{'='*50}")
	print(f"  Genome: {genome_info['hash']} ({genome_info['phase']})")
	print(f"  Neurons: {n_neurons}, Bits/neuron: {len(connections[0])}")
	print(f"  Input bits: {meta['input_bits']}")
	print(f"  Total sparse entries: {total_entries:,}")
	if total_bytes < 1024:
		size_str = f"{total_bytes} B"
	elif total_bytes < 1024*1024:
		size_str = f"{total_bytes/1024:.1f} KB"
	else:
		size_str = f"{total_bytes/1024/1024:.1f} MB"
	print(f"  Model size (sparse): {size_str}")
	if total_bytes < 612*1024:
		print(f"  Z-7020 BRAM (612 KB): FITS in on-chip BRAM")
	else:
		print(f"  Z-7020: requires external DDR ({size_str} > 612 KB BRAM)")


def main():
	parser = argparse.ArgumentParser(description="Export WNN genome for FPGA")
	parser.add_argument("--flow-id", type=int, required=True, help="Flow ID")
	parser.add_argument("--genome-type", default="best_fitness",
						help="Genome type: best_fitness, best_f1, best_acc, best_ce, best_fpr")
	parser.add_argument("--db", default="db/wnn.db", help="Database path")
	parser.add_argument("--output", default=None, help="Output directory")
	parser.add_argument("--phase", default=None, choices=["ga_neurons", "grid_search"],
		help="Force a specific phase. Default tries GA Neurons then Grid Search.")
	args = parser.parse_args()

	db_path = str(Path(__file__).resolve().parents[2] / args.db)

	# 1. Get genome connections and flow config
	print(f"Loading genome from flow {args.flow_id}...")
	params, genome, connections = get_genome_connections(db_path, args.flow_id, args.genome_type, args.phase)

	n_bits = params["ids_n_bits"]
	dataset_name = params.get("ids_dataset", "unsw-nb15")
	split = params.get("ids_split", "temporal")
	feature_sel = params.get("ids_feature_selection", "top20")

	print(f"  Dataset: {dataset_name}, Split: {split}, Thermometer: {n_bits}-bit")
	print(f"  Genome: {genome['total_neurons']}n × {len(connections[0])}b ({genome['phase']})")
	print(f"  Hash: {genome['hash']}")

	# 2. Load dataset
	print(f"\nLoading {dataset_name} ({n_bits}-bit thermometer)...")
	dataset = load_dataset(dataset_name, split, n_bits, feature_sel)
	X_train = dataset.X_train
	y_train = dataset.y_train_binary
	params["_input_bits"] = X_train.shape[1]
	params["_flow_id"] = args.flow_id

	print(f"  Train: {X_train.shape}")

	# 3. Train and extract sparse memory
	print(f"\nTraining {genome['total_neurons']} neurons...")
	sparse_neurons = train_and_extract_sparse(X_train, y_train, connections, len(connections[0]))

	# 4. Write FPGA files
	output_dir = Path(args.output) if args.output else Path(__file__).resolve().parents[1] / "export" / f"flow_{args.flow_id}"
	write_fpga_files(output_dir, connections, sparse_neurons, genome, params)


if __name__ == "__main__":
	main()
