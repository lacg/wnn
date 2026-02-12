#!/usr/bin/env python3
"""Debug script to trace memory state between batches."""

import torch
import sys
sys.path.insert(0, 'src/wnn')

from wnn.ram.core import RAMClusterLayer, MemoryVal


def trace_two_batches():
	"""Trace memory state after each of 2 batches."""
	print("=" * 60)
	print("Two-Batch Memory Trace")
	print("=" * 60)

	vocab_size = 100
	neurons_per_cluster = 3
	bits_per_neuron = 6  # 64 addresses per neuron
	total_input_bits = 24
	num_negatives = 10

	print(f"\nConfig:")
	print(f"  vocab_size: {vocab_size}")
	print(f"  neurons_per_cluster: {neurons_per_cluster}")
	print(f"  bits_per_neuron: {bits_per_neuron}")
	print(f"  num_negatives: {num_negatives}")

	# Create two layers with same connectivity
	layer_pt = RAMClusterLayer(
		total_input_bits=total_input_bits,
		num_clusters=vocab_size,
		neurons_per_cluster=neurons_per_cluster,
		bits_per_neuron=bits_per_neuron,
	)
	layer_rs = RAMClusterLayer(
		total_input_bits=total_input_bits,
		num_clusters=vocab_size,
		neurons_per_cluster=neurons_per_cluster,
		bits_per_neuron=bits_per_neuron,
	)
	layer_rs.memory.connections = layer_pt.memory.connections.clone()

	# Create 2 batches of 50 examples each
	torch.manual_seed(42)
	batch1_bits = torch.randint(0, 2, (50, total_input_bits), dtype=torch.bool)
	batch1_targets = torch.randint(0, vocab_size, (50,))
	batch1_negatives = torch.randint(0, vocab_size, (50, num_negatives))

	batch2_bits = torch.randint(0, 2, (50, total_input_bits), dtype=torch.bool)
	batch2_targets = torch.randint(0, vocab_size, (50,))
	batch2_negatives = torch.randint(0, vocab_size, (50, num_negatives))

	# ===== BATCH 1 =====
	print("\n--- BATCH 1 ---")

	# PyTorch
	mod_pt1 = layer_pt.train_multi_examples(batch1_bits, batch1_targets, batch1_negatives)
	pt_mem_after_b1 = layer_pt.memory.memory_words.clone()
	print(f"PyTorch modified: {mod_pt1}")

	# Rust
	try:
		mod_rs1 = layer_rs.train_multi_examples_rust_numpy(batch1_bits, batch1_targets, batch1_negatives)
		rs_mem_after_b1 = layer_rs.memory.memory_words.clone()
		print(f"Rust modified: {mod_rs1}")

		# Compare after batch 1
		diff1 = (pt_mem_after_b1 != rs_mem_after_b1).sum().item()
		print(f"Different words after batch 1: {diff1}")

		if diff1 > 0:
			analyze_differences(pt_mem_after_b1, rs_mem_after_b1, "After Batch 1")

	except Exception as e:
		print(f"Rust failed: {e}")
		return

	# ===== BATCH 2 =====
	print("\n--- BATCH 2 ---")

	# PyTorch
	mod_pt2 = layer_pt.train_multi_examples(batch2_bits, batch2_targets, batch2_negatives)
	pt_mem_after_b2 = layer_pt.memory.memory_words.clone()
	print(f"PyTorch modified: {mod_pt2}")

	# Rust
	mod_rs2 = layer_rs.train_multi_examples_rust_numpy(batch2_bits, batch2_targets, batch2_negatives)
	rs_mem_after_b2 = layer_rs.memory.memory_words.clone()
	print(f"Rust modified: {mod_rs2}")

	# Compare after batch 2
	diff2 = (pt_mem_after_b2 != rs_mem_after_b2).sum().item()
	print(f"Different words after batch 2: {diff2}")

	if diff2 > 0:
		analyze_differences(pt_mem_after_b2, rs_mem_after_b2, "After Batch 2")

	# Check if batch 2 introduced new differences
	print(f"\nDifferences introduced by batch 2: {diff2 - diff1}")

	print("\n" + "=" * 60)


def analyze_differences(pt_mem, rs_mem, label):
	"""Analyze cell-level differences between two memory states."""
	pt_flat = pt_mem.flatten()
	rs_flat = rs_mem.flatten()
	diff_mask = pt_flat != rs_flat

	counts = {
		"PT=EMPTY, RS=FALSE": 0,
		"PT=EMPTY, RS=TRUE": 0,
		"PT=FALSE, RS=EMPTY": 0,
		"PT=TRUE, RS=EMPTY": 0,
		"PT=TRUE, RS=FALSE": 0,
		"PT=FALSE, RS=TRUE": 0,
	}

	diff_indices = diff_mask.nonzero().squeeze()
	if diff_indices.dim() == 0:
		diff_indices = [diff_indices.item()]
	else:
		diff_indices = diff_indices[:1000].tolist()

	for word_idx in diff_indices:
		pt_word = pt_flat[word_idx].item()
		rs_word = rs_flat[word_idx].item()

		for cell_idx in range(31):
			shift = cell_idx * 2
			pt_cell = (pt_word >> shift) & 0b11
			rs_cell = (rs_word >> shift) & 0b11

			if pt_cell != rs_cell:
				key = f"PT={'EMPTY' if pt_cell == 2 else 'TRUE' if pt_cell == 1 else 'FALSE'}, RS={'EMPTY' if rs_cell == 2 else 'TRUE' if rs_cell == 1 else 'FALSE'}"
				if key in counts:
					counts[key] += 1

	print(f"\n  Cell differences ({label}):")
	for key, count in counts.items():
		if count > 0:
			print(f"    {key}: {count}")


if __name__ == "__main__":
	trace_two_batches()
