#!/usr/bin/env python3
"""Export test vectors for FPGA functional verification.

Generates test input vectors (thermometer-encoded) and expected outputs
from the Python WNN model, for use in a SystemVerilog testbench.

Usage:
    python export_test_vectors.py --flow-id 973 --num-vectors 1000
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "wnn"))


def load_dataset(ids_dataset, ids_split, n_bits, feature_selection):
	if ids_dataset == "cicids2017":
		from wnn.ids.cicids2017 import load_cicids2017
		return load_cicids2017(split=ids_split, n_bits=n_bits, feature_selection=feature_selection)
	elif ids_dataset == "unsw-nb15":
		from wnn.ids.dataset import load_unsw_nb15
		return load_unsw_nb15(split=ids_split, n_bits=n_bits, feature_selection=feature_selection)
	else:
		raise ValueError(f"Unknown dataset: {ids_dataset}")


def python_classify(X, y_true, connections, sparse_neurons, threshold=0.5):
	"""Classify using Python WNN model (ground truth for RTL verification).

	Returns per-sample: neuron scores, total score, predicted class, true class.
	"""
	n_samples = len(X)
	n_neurons = len(connections)

	# QUAD_WEIGHTED values: 0=FALSE(0.0), 1=WEAK_FALSE(0.25), 2=WEAK_TRUE(0.75), 3=TRUE(1.0)
	EMPTY = 1  # QUAD_WEAK_FALSE = default for untrained

	results = []
	for i in range(n_samples):
		neuron_scores = []
		for n_idx in range(n_neurons):
			conn = connections[n_idx]
			keys, values = sparse_neurons[n_idx]

			# Form address from selected input bits
			addr_bits = X[i, conn]
			addr = 0
			for bit in addr_bits:
				addr = (addr << 1) | int(bit)

			# Binary search lookup
			idx = np.searchsorted(keys, addr)
			if idx < len(keys) and keys[idx] == addr:
				value = int(values[idx])
			else:
				value = EMPTY

			neuron_scores.append(value)

		total_score = sum(neuron_scores)
		# Threshold: score > threshold * n_neurons * 2 (midpoint of QUAD range)
		# Simple: compare raw sum against threshold
		predicted = 1 if total_score > int(threshold * n_neurons * 2) else 0

		results.append({
			"index": i,
			"neuron_scores": neuron_scores,
			"total_score": total_score,
			"predicted": predicted,
			"true_label": int(y_true[i]),
		})

	return results


def write_test_vectors(output_dir, X_test, y_test, results, connections, meta, num_vectors):
	"""Write test vectors in formats suitable for SystemVerilog $readmemh."""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	input_bits = meta["input_bits"]
	n_neurons = meta["total_neurons"]

	# 1. Input vectors file (hex, one per line, MSB-first)
	with open(output_dir / "test_inputs.mem", "w") as f:
		for i in range(num_vectors):
			# Convert boolean array to hex string
			bits = X_test[i]
			# Pad to multiple of 4 for hex
			padded_len = ((input_bits + 3) // 4) * 4
			padded = np.zeros(padded_len, dtype=np.uint8)
			padded[:input_bits] = bits
			# Convert to hex (MSB first)
			hex_str = ""
			for j in range(0, padded_len, 4):
				nibble = (padded[j] << 3) | (padded[j+1] << 2) | (padded[j+2] << 1) | padded[j+3]
				hex_str += f"{nibble:x}"
			f.write(hex_str + "\n")

	# 2. Expected outputs file (per-neuron scores and total)
	with open(output_dir / "test_expected.mem", "w") as f:
		for r in results[:num_vectors]:
			# Format: total_score (hex)
			f.write(f"{r['total_score']:02x}\n")

	# 3. Expected class labels
	with open(output_dir / "test_labels.mem", "w") as f:
		for r in results[:num_vectors]:
			f.write(f"{r['predicted']}\n")

	# 4. Human-readable verification file
	with open(output_dir / "test_verification.txt", "w") as f:
		f.write(f"# Test vectors for flow {meta.get('flow_id', '?')}\n")
		f.write(f"# {num_vectors} vectors, {input_bits} input bits, {n_neurons} neurons\n")
		f.write(f"# Format: index | true_label | predicted | total_score | neuron_scores\n\n")

		correct = 0
		for r in results[:num_vectors]:
			match = "OK" if r["predicted"] == r["true_label"] else "MISS"
			if r["predicted"] == r["true_label"]:
				correct += 1
			scores_str = ",".join(str(s) for s in r["neuron_scores"])
			f.write(f"{r['index']:>5} | true={r['true_label']} pred={r['predicted']} "
					f"| score={r['total_score']:>3} | [{scores_str}] | {match}\n")

		accuracy = correct / num_vectors * 100
		f.write(f"\n# Accuracy: {correct}/{num_vectors} = {accuracy:.2f}%\n")

	# 5. Metadata for testbench parameters
	tb_meta = {
		"num_vectors": num_vectors,
		"input_bits": input_bits,
		"num_neurons": n_neurons,
		"addr_bits": len(connections[0]),
		"accuracy_pct": correct / num_vectors * 100,
		"flow_id": meta.get("flow_id"),
	}
	with open(output_dir / "tb_metadata.json", "w") as f:
		json.dump(tb_meta, f, indent=2)

	print(f"  Test vectors: {output_dir / 'test_inputs.mem'}")
	print(f"  Expected scores: {output_dir / 'test_expected.mem'}")
	print(f"  Labels: {output_dir / 'test_labels.mem'}")
	print(f"  Verification: {output_dir / 'test_verification.txt'}")
	print(f"  Python accuracy: {correct}/{num_vectors} = {accuracy:.2f}%")


def main():
	parser = argparse.ArgumentParser(description="Export test vectors for FPGA verification")
	parser.add_argument("--flow-id", type=int, required=True)
	parser.add_argument("--genome-type", default="best_fitness")
	parser.add_argument("--num-vectors", type=int, default=1000)
	parser.add_argument("--db", default="db/wnn.db")
	parser.add_argument("--output", default=None)
	parser.add_argument("--threshold", type=float, default=0.5,
						help="Classification threshold (fraction of max possible score)")
	args = parser.parse_args()

	# Import the export_genome module for shared functions
	sys.path.insert(0, str(Path(__file__).parent))
	from export_genome import get_genome_connections, load_dataset as load_ds

	db_path = str(Path(__file__).resolve().parents[2] / args.db)

	print(f"Loading genome from flow {args.flow_id}...")
	params, genome, connections = get_genome_connections(db_path, args.flow_id, args.genome_type)

	n_bits = params["ids_n_bits"]
	dataset_name = params.get("ids_dataset", "unsw-nb15")
	split = params.get("ids_split", "temporal")
	feature_sel = params.get("ids_feature_selection", "top20")

	print(f"  {genome['total_neurons']}n × {len(connections[0])}b, {n_bits}-bit thermometer")

	print(f"\nLoading {dataset_name}...")
	dataset = load_ds(dataset_name, split, n_bits, feature_sel)
	X_test = dataset.X_test
	y_test = dataset.y_test_binary

	print(f"  Test set: {X_test.shape}")

	# Load sparse memory from exported files
	export_dir = Path(__file__).resolve().parents[1] / "export" / f"flow_{args.flow_id}"
	print(f"\nLoading sparse memory from {export_dir}...")
	sparse_neurons = []
	with open(export_dir / "sparse_memory.bin", "rb") as f:
		num_neurons = struct.unpack("<I", f.read(4))[0]
		counts = [struct.unpack("<I", f.read(4))[0] for _ in range(num_neurons)]
		for count in counts:
			keys = []
			values = []
			for _ in range(count):
				k, v = struct.unpack("<QB", f.read(9))
				keys.append(k)
				values.append(v)
			sparse_neurons.append((np.array(keys, dtype=np.uint64), np.array(values, dtype=np.uint8)))

	# Select test vectors (mix of normal and attack)
	num_vectors = min(args.num_vectors, len(X_test))
	# Stratified sample: half normal, half attack
	normal_idx = np.where(y_test == 0)[0]
	attack_idx = np.where(y_test == 1)[0]
	n_each = num_vectors // 2
	rng = np.random.RandomState(42)
	selected = np.concatenate([
		rng.choice(normal_idx, min(n_each, len(normal_idx)), replace=False),
		rng.choice(attack_idx, min(n_each, len(attack_idx)), replace=False),
	])
	rng.shuffle(selected)
	selected = selected[:num_vectors]

	X_sel = X_test[selected]
	y_sel = y_test[selected]

	print(f"\nClassifying {num_vectors} test vectors in Python...")
	results = python_classify(X_sel, y_sel, connections, sparse_neurons, threshold=args.threshold)

	output_dir = Path(args.output) if args.output else export_dir / "testbench"
	meta = {
		"input_bits": X_test.shape[1],
		"total_neurons": len(connections),
		"flow_id": args.flow_id,
	}
	print(f"\nWriting test vectors...")
	write_test_vectors(output_dir, X_sel, y_sel, results, connections, meta, num_vectors)


if __name__ == "__main__":
	main()
