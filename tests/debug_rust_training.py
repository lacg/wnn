#!/usr/bin/env python3
"""Debug script to trace Rust training on a single example."""

import torch
import sys
sys.path.insert(0, 'src/wnn')

from wnn.ram.core import RAMClusterLayer, MemoryVal


def trace_single_example():
	"""Trace what happens when we train a single example."""
	print("=" * 60)
	print("Single Example Training Debug")
	print("=" * 60)

	# Small config for easy debugging
	vocab_size = 10
	neurons_per_cluster = 3
	bits_per_neuron = 4  # Only 16 addresses per neuron
	total_input_bits = 16

	print(f"\nConfig:")
	print(f"  vocab_size (clusters): {vocab_size}")
	print(f"  neurons_per_cluster: {neurons_per_cluster}")
	print(f"  bits_per_neuron: {bits_per_neuron}")
	print(f"  total_input_bits: {total_input_bits}")

	# Create layer
	layer = RAMClusterLayer(
		total_input_bits=total_input_bits,
		num_clusters=vocab_size,
		neurons_per_cluster=neurons_per_cluster,
		bits_per_neuron=bits_per_neuron,
	)

	# Verify initial state
	print(f"\n--- Initial Memory State ---")
	expected_empty = 0x2AAAAAAAAAAAAAAA
	mem = layer.memory.memory_words.flatten()
	all_empty = (mem == expected_empty).all().item()
	print(f"All EMPTY: {all_empty}")

	# Create a single example
	torch.manual_seed(42)
	input_bits = torch.randint(0, 2, (1, total_input_bits), dtype=torch.bool)
	true_cluster = torch.tensor([3])  # Target cluster 3
	false_clusters = torch.tensor([[0, 1, 5]])  # Negatives: 0, 1, 5

	print(f"\n--- Training Example ---")
	print(f"Input bits: {input_bits[0].tolist()}")
	print(f"True cluster: {true_cluster.item()}")
	print(f"False clusters: {false_clusters[0].tolist()}")

	# Get addresses that will be written
	addresses = layer.memory.get_addresses(input_bits)  # [1, num_neurons]
	print(f"\nAddresses for true cluster 3 (neurons 9-11):")
	for n in range(9, 12):
		print(f"  Neuron {n}: address {addresses[0, n].item()}")

	print(f"\nAddresses for false clusters:")
	for cluster in [0, 1, 5]:
		start_neuron = cluster * neurons_per_cluster
		for offset in range(neurons_per_cluster):
			n = start_neuron + offset
			print(f"  Cluster {cluster}, Neuron {n}: address {addresses[0, n].item()}")

	# Train with PyTorch
	print(f"\n--- PyTorch Training ---")
	modified_pt = layer.train_multi_examples(
		input_bits, true_cluster, false_clusters, allow_override=False
	)
	print(f"Modified cells (reported): {modified_pt}")

	# Check specific cells after PyTorch training
	print(f"\nCell values after PyTorch training:")
	for cluster in [3, 0, 1, 5]:
		start_neuron = cluster * neurons_per_cluster
		expected = "TRUE" if cluster == 3 else "FALSE"
		for offset in range(neurons_per_cluster):
			n = start_neuron + offset
			addr = addresses[0, n].item()
			val = layer.memory.get_memory(n, addr)
			val_name = {MemoryVal.FALSE: "FALSE", MemoryVal.TRUE: "TRUE", MemoryVal.EMPTY: "EMPTY"}.get(val, f"UNK({val})")
			status = "✓" if val_name == expected else "✗"
			print(f"  Cluster {cluster}, Neuron {n}, Addr {addr}: {val_name} (expected {expected}) {status}")

	# Save PyTorch memory state
	pt_memory = layer.memory.memory_words.clone()

	# Reset memory and train with Rust
	print(f"\n--- Rust Training ---")
	layer.reset_memory()

	# Verify reset
	mem = layer.memory.memory_words.flatten()
	all_empty = (mem == expected_empty).all().item()
	print(f"Memory reset to ALL EMPTY: {all_empty}")

	try:
		modified_rs = layer.train_multi_examples_rust_numpy(
			input_bits, true_cluster, false_clusters, allow_override=False
		)
		print(f"Modified cells (reported): {modified_rs}")

		# Check specific cells after Rust training
		print(f"\nCell values after Rust training:")
		for cluster in [3, 0, 1, 5]:
			start_neuron = cluster * neurons_per_cluster
			expected = "TRUE" if cluster == 3 else "FALSE"
			for offset in range(neurons_per_cluster):
				n = start_neuron + offset
				addr = addresses[0, n].item()
				val = layer.memory.get_memory(n, addr)
				val_name = {MemoryVal.FALSE: "FALSE", MemoryVal.TRUE: "TRUE", MemoryVal.EMPTY: "EMPTY"}.get(val, f"UNK({val})")
				status = "✓" if val_name == expected else "✗"
				print(f"  Cluster {cluster}, Neuron {n}, Addr {addr}: {val_name} (expected {expected}) {status}")

		# Compare
		rs_memory = layer.memory.memory_words.clone()
		diff_mask = pt_memory != rs_memory
		num_diff = diff_mask.sum().item()
		print(f"\n--- Comparison ---")
		print(f"Different words: {num_diff}")

		if num_diff > 0:
			print("MISMATCH DETECTED!")

	except Exception as e:
		print(f"Rust training failed: {e}")

	print("\n" + "=" * 60)


def trace_conflict_examples():
	"""Trace what happens when examples have conflicting writes."""
	print("=" * 60)
	print("Conflict Example Training Debug")
	print("=" * 60)

	# Small config for easy debugging
	vocab_size = 10
	neurons_per_cluster = 3
	bits_per_neuron = 4  # Only 16 addresses per neuron
	total_input_bits = 16

	print(f"\nConfig:")
	print(f"  vocab_size (clusters): {vocab_size}")
	print(f"  neurons_per_cluster: {neurons_per_cluster}")
	print(f"  bits_per_neuron: {bits_per_neuron}")
	print(f"  total_input_bits: {total_input_bits}")

	# Create layer
	layer = RAMClusterLayer(
		total_input_bits=total_input_bits,
		num_clusters=vocab_size,
		neurons_per_cluster=neurons_per_cluster,
		bits_per_neuron=bits_per_neuron,
	)

	# Create examples where:
	# - Example A: target=3, negatives=[0]
	# - Example B: target=0, negatives=[3]
	# Both examples use SAME input bits, so they compute SAME addresses!
	# This creates a conflict: cluster 3 should be TRUE (from A) but B wants FALSE
	torch.manual_seed(42)
	same_input = torch.randint(0, 2, (1, total_input_bits), dtype=torch.bool)
	input_bits = same_input.expand(2, -1).contiguous()  # [2, 16] same bits

	true_clusters = torch.tensor([3, 0])  # A targets 3, B targets 0
	false_clusters = torch.tensor([[0], [3]])  # A negatives 0, B negatives 3

	print(f"\n--- Training Examples (CONFLICT SCENARIO) ---")
	print(f"Input bits (same for both): {same_input[0].tolist()}")
	print(f"Example A: target=3, negative=0")
	print(f"Example B: target=0, negative=3")
	print(f"\nExpected behavior:")
	print(f"  Cluster 3: TRUE (from A's target), NOT FALSE (B's negative should lose)")
	print(f"  Cluster 0: TRUE (from B's target), NOT FALSE (A's negative should lose)")

	# Get addresses
	addresses = layer.memory.get_addresses(same_input)

	# Train with PyTorch
	print(f"\n--- PyTorch Training ---")
	modified_pt = layer.train_multi_examples(
		input_bits, true_clusters, false_clusters, allow_override=False
	)
	print(f"Modified cells: {modified_pt}")

	print(f"\nCell values after PyTorch:")
	for cluster in [3, 0]:
		start = cluster * neurons_per_cluster
		for offset in range(neurons_per_cluster):
			n = start + offset
			addr = addresses[0, n].item()
			val = layer.memory.get_memory(n, addr)
			val_name = {MemoryVal.FALSE: "FALSE", MemoryVal.TRUE: "TRUE", MemoryVal.EMPTY: "EMPTY"}.get(val, f"UNK({val})")
			print(f"  Cluster {cluster}, Neuron {n}: {val_name} (expected TRUE)")

	pt_memory = layer.memory.memory_words.clone()

	# Reset and train with Rust
	print(f"\n--- Rust Training ---")
	layer.reset_memory()

	try:
		modified_rs = layer.train_multi_examples_rust_numpy(
			input_bits, true_clusters, false_clusters, allow_override=False
		)
		print(f"Modified cells: {modified_rs}")

		print(f"\nCell values after Rust:")
		for cluster in [3, 0]:
			start = cluster * neurons_per_cluster
			for offset in range(neurons_per_cluster):
				n = start + offset
				addr = addresses[0, n].item()
				val = layer.memory.get_memory(n, addr)
				val_name = {MemoryVal.FALSE: "FALSE", MemoryVal.TRUE: "TRUE", MemoryVal.EMPTY: "EMPTY"}.get(val, f"UNK({val})")
				print(f"  Cluster {cluster}, Neuron {n}: {val_name} (expected TRUE)")

		# Compare
		rs_memory = layer.memory.memory_words.clone()
		diff_mask = pt_memory != rs_memory
		num_diff = diff_mask.sum().item()
		print(f"\n--- Comparison ---")
		print(f"Different words: {num_diff}")

	except Exception as e:
		print(f"Rust training failed: {e}")
		import traceback
		traceback.print_exc()

	print("\n" + "=" * 60)


if __name__ == "__main__":
	trace_single_example()
	print("\n\n")
	trace_conflict_examples()
