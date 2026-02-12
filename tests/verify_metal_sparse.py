#!/usr/bin/env python3
"""
Verify Metal GPU sparse forward matches CPU sparse forward.

Trains a 12_10_8 tiered model and compares Metal vs CPU outputs.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "wnn"))

from wnn.ram.core.models import RAMLM


def load_wikitext2_tokens(split: str) -> list[int]:
	from datasets import load_dataset
	from transformers import GPT2TokenizerFast
	ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
	text = "\n\n".join(ds["text"])
	tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
	return tokenizer.encode(text)


def build_cluster_order(train_tokens, vocab_size=50257):
	counts = Counter(train_tokens)
	seen = [t for t, _ in counts.most_common()]
	unseen = [t for t in range(vocab_size) if t not in seen]
	return seen + unseen


def force_sparse_mode(model):
	import ram_accelerator
	layer = model.layer
	if layer._use_sparse and layer._sparse_memory is not None:
		return
	rust_tier_configs = [
		(tc.cluster_end, tc.neurons_per_cluster, tc.bits_per_neuron)
		for tc in layer.tier_configs
	]
	layer._sparse_memory = ram_accelerator.TieredSparseMemory(
		rust_tier_configs, layer.num_clusters
	)
	layer._use_sparse = True


def main():
	print("Loading data...")
	train_tokens = load_wikitext2_tokens("train")
	val_tokens = load_wikitext2_tokens("validation")
	cluster_order = build_cluster_order(train_tokens)

	# 12_10_8 config
	tiers = [(100, 15, 12), (400, 10, 10), (None, 5, 8)]
	print(f"Config: {tiers}")

	model = RAMLM(
		vocab_size=50257,
		context_size=4,
		tiers=tiers,
		cluster_order=cluster_order,
		rng=42,
	)
	force_sparse_mode(model)

	print("Training...")
	model.train_epoch_fast_auto(
		train_tokens, global_top_k=1000, batch_size=5000, verbose=True,
	)

	# Encode a batch of validation examples
	layer = model.layer
	num_test = 2000
	all_bits = model.encode_sequence(val_tokens[:num_test + model.context_size])
	print(f"Test batch: {all_bits.shape}")

	# --- CPU sparse forward ---
	print("\nCPU sparse forward...")
	t0 = time.time()
	cpu_out = layer.forward_sparse(all_bits)
	cpu_time = time.time() - t0
	print(f"  Time: {cpu_time:.3f}s")
	print(f"  Shape: {cpu_out.shape}")
	print(f"  Range: [{cpu_out.min():.4f}, {cpu_out.max():.4f}]")

	# --- Metal GPU forward ---
	print("\nMetal GPU forward...")
	t0 = time.time()
	metal_out = layer.forward_metal(all_bits)
	metal_time = time.time() - t0
	print(f"  Time: {metal_time:.3f}s (includes GPU cache export)")

	# Second call (cached)
	t0 = time.time()
	metal_out2 = layer.forward_metal(all_bits)
	metal_time2 = time.time() - t0
	print(f"  Cached: {metal_time2:.3f}s")
	print(f"  Shape: {metal_out.shape}")
	print(f"  Range: [{metal_out.min():.4f}, {metal_out.max():.4f}]")

	# --- Compare ---
	diff = torch.abs(cpu_out - metal_out)
	max_diff = diff.max().item()
	mean_diff = diff.mean().item()
	print(f"\nMax absolute difference: {max_diff:.2e}")
	print(f"Mean absolute difference: {mean_diff:.2e}")

	# Check where differences occur
	if max_diff > 1e-5:
		# Find largest differences
		flat_diff = diff.flatten()
		top_k = min(10, flat_diff.numel())
		top_vals, top_idx = flat_diff.topk(top_k)
		print(f"\nTop {top_k} differences:")
		for i in range(top_k):
			idx = top_idx[i].item()
			batch_i = idx // cpu_out.shape[1]
			cluster_i = idx % cpu_out.shape[1]
			print(f"  [{batch_i}, {cluster_i}]: CPU={cpu_out.flatten()[idx]:.6f}, "
				  f"Metal={metal_out.flatten()[idx]:.6f}, diff={top_vals[i]:.2e}")

	# --- Full evaluation via AccelerationMode dispatch ---
	print("\n" + "="*60)
	print("Full evaluation via RAMLM.evaluate_fast (all backends):")
	print("="*60)

	from wnn.ram.core import AccelerationMode

	eval_tokens = val_tokens[:5000]
	results = {}

	for mode in [AccelerationMode.CPU, AccelerationMode.METAL, AccelerationMode.HYBRID]:
		name = mode.name
		print(f"\n{name} evaluation...")
		t0 = time.time()
		stats = model.evaluate_fast(
			eval_tokens, batch_size=2000,
			backend=mode, verbose=False,
		)
		elapsed = time.time() - t0
		results[name] = (stats, elapsed)
		print(f"  CE={stats['cross_entropy']:.4f}, Acc={stats['accuracy']:.4%}, Time={elapsed:.3f}s")

	# Compare all against CPU baseline
	cpu_ce = results['CPU'][0]['cross_entropy']
	cpu_acc = results['CPU'][0]['accuracy']
	cpu_time = results['CPU'][1]

	print(f"\n{'Backend':<12} {'CE':>10} {'Accuracy':>10} {'Time':>10} {'Speedup':>10} {'CE Match':>10}")
	print("-" * 66)
	for name, (stats, elapsed) in results.items():
		ce_match = abs(stats['cross_entropy'] - cpu_ce) < 0.01
		speedup = cpu_time / elapsed if elapsed > 0 else float('inf')
		print(f"{name:<12} {stats['cross_entropy']:>10.4f} {stats['accuracy']:>10.4%} {elapsed:>10.3f}s {speedup:>9.1f}x {'YES' if ce_match else 'NO':>10}")

	# Verdict
	all_ce_match = all(
		abs(stats['cross_entropy'] - cpu_ce) < 0.01
		for name, (stats, _) in results.items()
	)
	print("\n" + "="*60)
	if max_diff < 1e-5 and all_ce_match:
		print("PASS: All backends match (tensor diff < 1e-5, CE within 0.01)")
	elif max_diff < 0.01 and all_ce_match:
		print(f"PASS (with tolerance): max_diff={max_diff:.2e}, all CE match")
	else:
		print(f"FAIL: max_diff={max_diff:.2e}, all_ce_match={all_ce_match}")
	print("="*60)


if __name__ == "__main__":
	main()
