#!/usr/bin/env python3
"""
Quick test: compare normal SELECTOR CE vs invalid_mode=True CE.

Creates a MultiStageEvaluator with SELECTOR mode, builds random genomes,
and compares compute_combined_metrics with and without invalid_mode.
"""

import sys
import time
import random

sys.path.insert(0, "src/wnn")

from wnn.ram.architecture.multistage_evaluator import MultiStageEvaluator
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.experiments.experiment import StageMode


def create_random_genome(num_clusters: int, total_input_bits: int, neurons: int = 50, bits: int = 10) -> ClusterGenome:
	bits_per_neuron = [bits] * (num_clusters * neurons)
	neurons_per_cluster = [neurons] * num_clusters
	connections = [random.randint(0, total_input_bits - 1) for _ in range(num_clusters * neurons * bits)]
	return ClusterGenome(
		bits_per_neuron=bits_per_neuron,
		neurons_per_cluster=neurons_per_cluster,
		connections=connections,
	)


def main():
	# Use real WikiText-2 data for meaningful CE comparison
	print("Loading GPT-2 tokenizer + WikiText-2...")
	import tiktoken
	from datasets import load_dataset

	tokenizer = tiktoken.get_encoding("gpt2")
	dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
	train_text = "\n\n".join([t for t in dataset["train"]["text"] if t.strip()])
	test_text = "\n\n".join([t for t in dataset["test"]["text"] if t.strip()])
	train_tokens = tokenizer.encode(train_text)
	test_tokens = tokenizer.encode(test_text)
	vocab_size = tokenizer.n_vocab
	print(f"  Train: {len(train_tokens)} tokens, Test: {len(test_tokens)} tokens")

	# Create evaluator with SELECTOR mode
	context_size = 4
	k = 225  # Typical for GPT-2 vocab

	print(f"\nCreating MultiStageEvaluator (SELECTOR mode, k={k})...")
	evaluator = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=test_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		num_stages=2,
		stage_k=[k, None],
		stage_cluster_type=["bitwise", "bitwise"],
		stage_mode=[StageMode.SELECTOR],
		target_stage=0,
		num_parts=1,
		num_eval_parts=1,
		seed=42,
		memory_mode=0,  # TERNARY
		neuron_sample_rate=1.0,
	)

	# Create genomes
	evaluator.target_stage = 0
	s0_genome = create_random_genome(
		num_clusters=evaluator.num_clusters,
		total_input_bits=evaluator.total_input_bits,
		neurons=20, bits=8,
	)
	print(f"  S0 genome: {len(s0_genome.neurons_per_cluster)} clusters, "
		  f"{sum(s0_genome.neurons_per_cluster)} neurons")

	evaluator.target_stage = 1
	s1_genome = create_random_genome(
		num_clusters=evaluator.num_clusters,
		total_input_bits=evaluator.total_input_bits,
		neurons=20, bits=8,
	)
	print(f"  S1 genome: {len(s1_genome.neurons_per_cluster)} clusters, "
		  f"{sum(s1_genome.neurons_per_cluster)} neurons")

	# Check max_cluster_size for the power-of-2 edge case
	sizes = evaluator.cluster_sizes
	max_cs = max(sizes)
	s1_bits = evaluator._cache.bitwise_output_bits(1)
	print(f"\n  max_cluster_size={max_cs}, s1_output_bits={s1_bits}, 2^bits={1 << s1_bits}")
	if max_cs >= (1 << s1_bits):
		print(f"  WARNING: max_cluster_size is a power of 2, invalid mode may have issues")

	# ── Normal mode ──
	print("\n" + "=" * 60)
	print("  Normal SELECTOR mode (no invalid)")
	print("=" * 60)
	start = time.time()
	normal = evaluator.compute_combined_metrics([s0_genome, s1_genome])
	t_normal = time.time() - start
	print(f"  CE:       {normal.ce:.4f}")
	print(f"  Accuracy: {normal.accuracy:.4%}")
	print(f"  S0 CE:    {normal.cluster_ce:.4f}")
	print(f"  S1 CE:    {normal.within_ce:.4f}")
	print(f"  Time:     {t_normal:.2f}s")

	# ── Invalid mode, top_m=5 ──
	print("\n" + "=" * 60)
	print("  Invalid token mode (top_m=5)")
	print("=" * 60)
	start = time.time()
	invalid5 = evaluator.compute_combined_metrics(
		[s0_genome, s1_genome], invalid_mode=True, top_m=5,
	)
	t_inv5 = time.time() - start
	print(f"  CE:       {invalid5.ce:.4f}")
	print(f"  Accuracy: {invalid5.accuracy:.4%}")
	print(f"  S0 CE:    {invalid5.cluster_ce:.4f}")
	print(f"  S1 CE:    {invalid5.within_ce:.4f}")
	print(f"  Time:     {t_inv5:.2f}s")

	# ── Invalid mode, top_m=10 ──
	print("\n" + "=" * 60)
	print("  Invalid token mode (top_m=10)")
	print("=" * 60)
	start = time.time()
	invalid10 = evaluator.compute_combined_metrics(
		[s0_genome, s1_genome], invalid_mode=True, top_m=10,
	)
	t_inv10 = time.time() - start
	print(f"  CE:       {invalid10.ce:.4f}")
	print(f"  Accuracy: {invalid10.accuracy:.4%}")
	print(f"  S0 CE:    {invalid10.cluster_ce:.4f}")
	print(f"  S1 CE:    {invalid10.within_ce:.4f}")
	print(f"  Time:     {t_inv10:.2f}s")

	# ── Summary ──
	print("\n" + "=" * 60)
	print("  Summary")
	print("=" * 60)
	print(f"  {'Mode':<25} {'CE':>10} {'Acc':>10} {'S0 CE':>10} {'S1 CE':>10} {'Time':>8}")
	print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
	for name, r, t in [
		("Normal", normal, t_normal),
		("Invalid (top_m=5)", invalid5, t_inv5),
		("Invalid (top_m=10)", invalid10, t_inv10),
	]:
		print(f"  {name:<25} {r.ce:10.4f} {r.accuracy:9.4%} {r.cluster_ce:10.4f} {r.within_ce:10.4f} {t:7.2f}s")

	ce_delta_5 = invalid5.ce - normal.ce
	ce_delta_10 = invalid10.ce - normal.ce
	print(f"\n  CE delta (invalid5 - normal):  {ce_delta_5:+.4f} {'(better)' if ce_delta_5 < 0 else '(worse)'}")
	print(f"  CE delta (invalid10 - normal): {ce_delta_10:+.4f} {'(better)' if ce_delta_10 < 0 else '(worse)'}")


if __name__ == "__main__":
	main()
