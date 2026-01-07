#!/usr/bin/env python3
"""Benchmark comparing PyTorch vs Rust training for RAMLM."""

import time
import torch
import sys
sys.path.insert(0, 'src/wnn')

from wnn.ram.core.models.ramlm import RAMLM


def benchmark_training():
	"""Compare PyTorch and Rust training backends."""
	print("=" * 60)
	print("RAMLM Training Backend Benchmark")
	print("=" * 60)

	# Create test data
	vocab_size = 1000
	context_size = 4
	num_tokens = 5000  # Small dataset for quick comparison

	print(f"\nConfig:")
	print(f"  vocab_size: {vocab_size}")
	print(f"  context_size: {context_size}")
	print(f"  num_tokens: {num_tokens}")

	# Generate random token sequence
	torch.manual_seed(42)
	token_ids = torch.randint(0, vocab_size, (num_tokens,)).tolist()

	# ========== PyTorch Training ==========
	print("\n" + "-" * 40)
	print("PyTorch Training (train_epoch_fast)")
	print("-" * 40)

	model_pt = RAMLM(
		vocab_size=vocab_size,
		context_size=context_size,
		neurons_per_cluster=3,
		bits_per_neuron=8,
	)

	start = time.perf_counter()
	stats_pt = model_pt.train_epoch_fast(
		token_ids,
		global_top_k=100,
		batch_size=500,
		verbose=False,
	)
	pt_time = time.perf_counter() - start
	print(f"  Time: {pt_time:.3f}s")
	print(f"  Modified: {stats_pt['modified']:,} cells")

	# Evaluate PyTorch model
	eval_pt = model_pt.evaluate_fast(token_ids[:1000], batch_size=500, verbose=False)
	print(f"  PPL: {eval_pt['perplexity']:.2f}")
	print(f"  Acc: {eval_pt['accuracy']:.2%}")

	# Save PyTorch memory state
	pt_memory = model_pt.layer.memory.memory_words.clone()

	# ========== Rust Training ==========
	print("\n" + "-" * 40)
	print("Rust Training (train_epoch_fast_rust)")
	print("-" * 40)

	try:
		import ram_accelerator
		rust_available = True
	except ImportError:
		rust_available = False
		print("  Rust accelerator not available!")

	if rust_available:
		model_rs = RAMLM(
			vocab_size=vocab_size,
			context_size=context_size,
			neurons_per_cluster=3,
			bits_per_neuron=8,
		)
		# CRITICAL: Copy same connectivity and global top-k for fair comparison
		model_rs.layer.memory.connections = model_pt.layer.memory.connections.clone()
		model_rs._global_top_k = model_pt._global_top_k

		start = time.perf_counter()
		stats_rs = model_rs.train_epoch_fast_rust(
			token_ids,
			global_top_k=100,
			batch_size=500,
			verbose=False,
		)
		rs_time = time.perf_counter() - start
		print(f"  Time: {rs_time:.3f}s")
		print(f"  Modified: {stats_rs['modified']:,} cells")

		# Evaluate Rust model
		eval_rs = model_rs.evaluate_fast(token_ids[:1000], batch_size=500, verbose=False)
		print(f"  PPL: {eval_rs['perplexity']:.2f}")
		print(f"  Acc: {eval_rs['accuracy']:.2%}")

		# Save Rust memory state
		rs_memory = model_rs.layer.memory.memory_words.clone()

		# ========== Compare ==========
		print("\n" + "-" * 40)
		print("Comparison")
		print("-" * 40)

		speedup = pt_time / rs_time if rs_time > 0 else float('inf')
		print(f"  Speedup: {speedup:.2f}x ({'Rust faster' if speedup > 1 else 'PyTorch faster'})")

		# Compare memory states
		diff_mask = pt_memory != rs_memory
		num_diff_words = diff_mask.sum().item()
		print(f"  Different memory words: {num_diff_words}")

		if num_diff_words > 0:
			# Analyze differences
			EMPTY = 2
			FALSE = 0
			TRUE = 1

			pt_flat = pt_memory.flatten()
			rs_flat = rs_memory.flatten()

			# Sample some differences
			diff_indices = diff_mask.flatten().nonzero().squeeze()[:10]
			print(f"\n  Sample differences (first 10 words):")
			for idx in diff_indices[:10]:
				pt_word = pt_flat[idx.item()].item()
				rs_word = rs_flat[idx.item()].item()
				print(f"    Word {idx.item()}: PT={pt_word:016x}, RS={rs_word:016x}")

			# Count cell-level differences
			total_cells_diff = 0
			pt_empty_rs_false = 0
			pt_empty_rs_true = 0
			pt_false_rs_empty = 0
			pt_true_rs_empty = 0
			pt_true_rs_false = 0
			pt_false_rs_true = 0

			for word_idx in diff_mask.flatten().nonzero().squeeze().tolist():
				if isinstance(word_idx, int):
					word_indices = [word_idx]
				else:
					word_indices = word_idx if isinstance(word_idx, list) else [word_idx]
					break

			# Analyze up to 1000 different words
			diff_word_indices = diff_mask.flatten().nonzero().squeeze()
			if diff_word_indices.dim() == 0:
				diff_word_indices = [diff_word_indices.item()]
			else:
				diff_word_indices = diff_word_indices[:1000].tolist()

			for word_idx in diff_word_indices:
				pt_word = pt_flat[word_idx].item()
				rs_word = rs_flat[word_idx].item()

				# Extract 31 cells from each word
				for cell_idx in range(31):
					shift = cell_idx * 2
					pt_cell = (pt_word >> shift) & 0b11
					rs_cell = (rs_word >> shift) & 0b11

					if pt_cell != rs_cell:
						total_cells_diff += 1
						if pt_cell == EMPTY and rs_cell == FALSE:
							pt_empty_rs_false += 1
						elif pt_cell == EMPTY and rs_cell == TRUE:
							pt_empty_rs_true += 1
						elif pt_cell == FALSE and rs_cell == EMPTY:
							pt_false_rs_empty += 1
						elif pt_cell == TRUE and rs_cell == EMPTY:
							pt_true_rs_empty += 1
						elif pt_cell == TRUE and rs_cell == FALSE:
							pt_true_rs_false += 1
						elif pt_cell == FALSE and rs_cell == TRUE:
							pt_false_rs_true += 1

			print(f"\n  Cell-level differences (from {len(diff_word_indices)} words):")
			print(f"    Total: {total_cells_diff}")
			print(f"    PT=EMPTY, RS=FALSE: {pt_empty_rs_false}")
			print(f"    PT=EMPTY, RS=TRUE: {pt_empty_rs_true}")
			print(f"    PT=FALSE, RS=EMPTY: {pt_false_rs_empty}")
			print(f"    PT=TRUE, RS=EMPTY: {pt_true_rs_empty}")
			print(f"    PT=TRUE, RS=FALSE: {pt_true_rs_false}")
			print(f"    PT=FALSE, RS=TRUE: {pt_false_rs_true}")
		else:
			print("  Memory states MATCH! ✓")

	print("\n" + "=" * 60)
	print("Benchmark complete")
	print("=" * 60)


if __name__ == "__main__":
	benchmark_training()
