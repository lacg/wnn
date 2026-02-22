#!/usr/bin/env python3
"""
Quick grid sweep for multi-stage architecture.

Tests neurons × bits combinations for Stage 0 and Stage 1 independently
on WikiText-2 to find reasonable ranges for the multi_stage_flow() defaults.

Usage:
	python tests/sweep_multistage_grid.py
	python tests/sweep_multistage_grid.py --small  # Use small subset for speed
"""

import argparse
import sys
import time

sys.path.insert(0, "src/wnn")

from wnn.ram.architecture.multistage_evaluator import MultiStageEvaluator, compute_default_k
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


def create_uniform_genome(num_clusters, total_input_bits, neurons, bits):
	"""Create a uniform genome with random connections."""
	import random
	bits_per_neuron = [bits] * (num_clusters * neurons)
	neurons_per_cluster = [neurons] * num_clusters
	connections = [random.randint(0, total_input_bits - 1) for _ in range(num_clusters * neurons * bits)]
	return ClusterGenome(
		bits_per_neuron=bits_per_neuron,
		neurons_per_cluster=neurons_per_cluster,
		connections=connections,
	)


def sweep_stage(evaluator, stage, neurons_grid, bits_grid, seed=42):
	"""Run grid sweep for a single stage."""
	import random
	random.seed(seed)

	evaluator.target_stage = stage
	num_clusters = evaluator.num_clusters
	total_input_bits = evaluator.total_input_bits

	print(f"\n{'='*70}")
	print(f"  Stage {stage} Grid Sweep")
	print(f"  num_clusters={num_clusters}, total_input_bits={total_input_bits}")
	print(f"  neurons: {neurons_grid}")
	print(f"  bits: {bits_grid}")
	print(f"{'='*70}")

	results = []
	total = len(neurons_grid) * len(bits_grid)
	idx = 0

	for neurons in neurons_grid:
		for bits in bits_grid:
			idx += 1
			# Skip impossible configs (bits > total_input_bits)
			if bits > total_input_bits:
				print(f"  [{idx:2d}/{total}] n={neurons:3d}, b={bits:2d}: SKIP (bits > input_bits={total_input_bits})")
				continue

			genome = create_uniform_genome(num_clusters, total_input_bits, neurons, bits)

			t0 = time.time()
			evals = evaluator.evaluate_batch([genome])
			elapsed = time.time() - t0

			ce = evals[0].ce
			acc = evals[0].accuracy
			bit_acc = evals[0].bit_accuracy

			print(f"  [{idx:2d}/{total}] n={neurons:3d}, b={bits:2d}: "
				  f"CE={ce:.4f}  Acc={acc:.2%}  BitAcc={bit_acc:.2%}  ({elapsed:.1f}s)")

			results.append({
				"neurons": neurons,
				"bits": bits,
				"ce": ce,
				"accuracy": acc,
				"bit_accuracy": bit_acc,
				"elapsed_s": round(elapsed, 1),
			})

	# Sort by CE
	results.sort(key=lambda r: r["ce"])

	print(f"\n  Top 10 by CE:")
	print(f"  {'Rank':>4s}  {'Neurons':>7s}  {'Bits':>4s}  {'CE':>8s}  {'Acc':>8s}  {'BitAcc':>8s}")
	print(f"  {'─'*4}  {'─'*7}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}")
	for i, r in enumerate(results[:10]):
		marker = " ★" if i < 3 else ""
		print(f"  {i+1:4d}  {r['neurons']:7d}  {r['bits']:4d}  {r['ce']:8.4f}  {r['accuracy']:7.2%}  {r['bit_accuracy']:7.2%}{marker}")

	return results


def main():
	parser = argparse.ArgumentParser(description="Multi-stage grid sweep")
	parser.add_argument("--small", action="store_true",
						help="Use small data subset for faster sweep")
	parser.add_argument("--context", type=int, default=4,
						help="Context size (default: 4)")
	parser.add_argument("--stage-k", type=str, default="auto",
						help="K per stage: 'auto' or comma-separated")
	parser.add_argument("--stage-type", type=str, default="bitwise,bitwise",
						help="Stage cluster types (default: bitwise,bitwise)")
	args = parser.parse_args()

	# Parse stage config
	stage_cluster_type = [s.strip() for s in args.stage_type.split(",")]
	num_stages = len(stage_cluster_type)

	if args.stage_k == "auto":
		stage_k = compute_default_k(num_stages, stage_cluster_type)
	else:
		stage_k = [int(k) for k in args.stage_k.split(",")]

	print(f"Multi-Stage Grid Sweep")
	print(f"  stages={num_stages}, k={stage_k}, types={stage_cluster_type}")
	print(f"  context={args.context}")

	# Load data
	if args.small:
		# Small synthetic data for speed
		import random
		random.seed(42)
		vocab_size = 50257
		train_tokens = [random.randint(0, vocab_size - 1) for _ in range(2000)]
		eval_tokens = [random.randint(0, vocab_size - 1) for _ in range(500)]
		print(f"  Using SYNTHETIC data: {len(train_tokens)} train, {len(eval_tokens)} eval")
	else:
		# Real WikiText-2
		from datasets import load_dataset
		from transformers import AutoTokenizer

		print("  Loading WikiText-2...")
		tokenizer = AutoTokenizer.from_pretrained("gpt2")
		tokenizer.model_max_length = int(1e12)
		dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

		train_tokens = tokenizer.encode("\n".join(dataset["train"]["text"]))
		eval_tokens = tokenizer.encode("\n".join(dataset["test"]["text"]))
		vocab_size = tokenizer.vocab_size
		print(f"  WikiText-2: {len(train_tokens):,} train, {len(eval_tokens):,} eval, vocab={vocab_size}")

	# Create evaluator
	print("  Creating MultiStageEvaluator...")
	evaluator = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size if not args.small else 50257,
		context_size=args.context,
		stage_k=stage_k,
		stage_cluster_type=stage_cluster_type,
		target_stage=0,
		num_parts=2 if args.small else 4,
		num_eval_parts=1,
		seed=42,
		memory_mode=2,  # QUAD_WEIGHTED
		neuron_sample_rate=0.25,
	)

	# Grid values — wider than bitwise since each stage has smaller output space
	neurons_grid = [5, 10, 25, 50, 100, 200]
	bits_grid = [4, 6, 8, 10, 12, 16, 20]

	# Sweep each stage
	all_results = {}
	for stage in range(num_stages):
		results = sweep_stage(evaluator, stage, neurons_grid, bits_grid)
		all_results[f"stage_{stage}"] = results

	# Summary
	print(f"\n{'='*70}")
	print(f"  SUMMARY — Recommended Grid Ranges")
	print(f"{'='*70}")
	for stage in range(num_stages):
		results = all_results[f"stage_{stage}"]
		if not results:
			print(f"  Stage {stage}: No valid results")
			continue
		top5 = results[:5]
		min_n = min(r["neurons"] for r in top5)
		max_n = max(r["neurons"] for r in top5)
		min_b = min(r["bits"] for r in top5)
		max_b = max(r["bits"] for r in top5)
		best = results[0]
		print(f"  Stage {stage}: neurons=[{min_n}..{max_n}], bits=[{min_b}..{max_b}]")
		print(f"           best: n={best['neurons']}, b={best['bits']}, CE={best['ce']:.4f}")


if __name__ == "__main__":
	main()
