"""
End-to-end test for TwoStageEvaluator.

Tests:
1. Construct TwoStageEvaluator with small token data
2. Create random genomes for Stage 1 and Stage 2
3. Evaluate Stage 1 genomes (subset + full)
4. Evaluate Stage 2 genomes (subset + full)
5. Compute combined CE and verify CE_combined ≈ CE_s1 + CE_s2
6. Run a mini GA-style evaluation (search_neighbors)
"""

import sys
import time
import random

# Setup path
sys.path.insert(0, "src/wnn")

from wnn.ram.architecture.twostage_evaluator import TwoStageEvaluator
from wnn.ram.architecture.base_evaluator import EvalResult
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


def create_random_genome(num_clusters: int, total_input_bits: int, neurons: int = 50, bits: int = 10) -> ClusterGenome:
	"""Create a random uniform genome."""
	bits_per_neuron = [bits] * (num_clusters * neurons)
	neurons_per_cluster = [neurons] * num_clusters
	connections = [random.randint(0, total_input_bits - 1) for _ in range(num_clusters * neurons * bits)]
	return ClusterGenome(
		bits_per_neuron=bits_per_neuron,
		neurons_per_cluster=neurons_per_cluster,
		connections=connections,
	)


def test_basic_evaluation():
	"""Test basic Stage 1 and Stage 2 evaluation."""
	print("=" * 60)
	print("  Test: Basic TwoStageEvaluator Evaluation")
	print("=" * 60)

	# Small synthetic token data (vocab 1000, ~2K train, ~500 eval)
	vocab_size = 1000
	context_size = 4
	k = 32  # 32 token groups
	random.seed(42)

	train_tokens = [random.randint(0, vocab_size - 1) for _ in range(2000)]
	eval_tokens = [random.randint(0, vocab_size - 1) for _ in range(500)]

	# Create evaluator (target_stage=1)
	print("\n  Creating TwoStageEvaluator (target_stage=1)...")
	eval_s1 = TwoStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		k=k,
		target_stage=1,
		num_parts=2,
		num_eval_parts=1,
		seed=42,
		pad_token_id=0,
	)
	print(f"  {eval_s1}")
	print(f"  Stage 1: num_clusters={eval_s1.num_clusters}, total_input_bits={eval_s1.total_input_bits}")

	# Stage 1 genome
	s1_genome = create_random_genome(
		num_clusters=eval_s1.num_clusters,
		total_input_bits=eval_s1.total_input_bits,
		neurons=20,
		bits=8,
	)
	print(f"  Stage 1 genome: {len(s1_genome.neurons_per_cluster)} clusters, "
		  f"{sum(s1_genome.neurons_per_cluster)} neurons")

	# Evaluate Stage 1
	start = time.time()
	results_s1 = eval_s1.evaluate_batch([s1_genome])
	elapsed = time.time() - start
	print(f"\n  Stage 1 subset eval: CE={results_s1[0].ce:.4f}, "
		  f"Acc={results_s1[0].accuracy:.4%}, "
		  f"BitAcc={results_s1[0].bit_accuracy:.4%} ({elapsed:.2f}s)")

	# Full eval
	results_s1_full = eval_s1.evaluate_batch_full([s1_genome])
	print(f"  Stage 1 full eval:   CE={results_s1_full[0].ce:.4f}, "
		  f"Acc={results_s1_full[0].accuracy:.4%}, "
		  f"BitAcc={results_s1_full[0].bit_accuracy:.4%}")

	# Switch to Stage 2
	eval_s1.target_stage = 2
	print(f"\n  Switched to Stage 2: num_clusters={eval_s1.num_clusters}, "
		  f"total_input_bits={eval_s1.total_input_bits}")

	s2_genome = create_random_genome(
		num_clusters=eval_s1.num_clusters,
		total_input_bits=eval_s1.total_input_bits,
		neurons=20,
		bits=8,
	)
	print(f"  Stage 2 genome: {len(s2_genome.neurons_per_cluster)} clusters, "
		  f"{sum(s2_genome.neurons_per_cluster)} neurons")

	# Evaluate Stage 2
	start = time.time()
	results_s2 = eval_s1.evaluate_batch([s2_genome])
	elapsed = time.time() - start
	print(f"  Stage 2 subset eval: CE={results_s2[0].ce:.4f}, "
		  f"Acc={results_s2[0].accuracy:.4%}, "
		  f"BitAcc={results_s2[0].bit_accuracy:.4%} ({elapsed:.2f}s)")

	results_s2_full = eval_s1.evaluate_batch_full([s2_genome])
	print(f"  Stage 2 full eval:   CE={results_s2_full[0].ce:.4f}, "
		  f"Acc={results_s2_full[0].accuracy:.4%}, "
		  f"BitAcc={results_s2_full[0].bit_accuracy:.4%}")

	print("\n  PASS: Basic evaluation works for both stages")
	return eval_s1, s1_genome, s2_genome


def test_combined_ce(evaluator, s1_genome, s2_genome):
	"""Test combined CE computation and verify CE_combined ≈ CE_s1 + CE_s2."""
	print("\n" + "=" * 60)
	print("  Test: Combined CE Computation")
	print("=" * 60)

	# Compute combined CE
	start = time.time()
	combined = evaluator.compute_combined_metrics(s1_genome, s2_genome)
	elapsed = time.time() - start

	print(f"\n  Combined CE:    {combined.ce:.4f}")
	print(f"  Combined Acc:   {combined.accuracy:.4%}")
	print(f"  Stage 1 CE:     {combined.cluster_ce:.4f}")
	print(f"  Stage 2 CE:     {combined.within_ce:.4f}")
	print(f"  S1 + S2:        {combined.cluster_ce + combined.within_ce:.4f}")
	print(f"  Elapsed:        {elapsed:.2f}s")

	# Verify CE_combined ≈ CE_s1 + CE_s2 (within 1% relative tolerance)
	sum_ce = combined.cluster_ce + combined.within_ce
	rel_error = abs(combined.ce - sum_ce) / max(combined.ce, 1e-10)
	print(f"\n  Relative error: {rel_error:.6f}")
	assert rel_error < 0.01, f"Combined CE mismatch: {combined.ce} vs {sum_ce} (rel={rel_error:.4f})"
	print("  PASS: CE_combined ≈ CE_s1 + CE_s2 (< 1% relative error)")


def test_search_neighbors(evaluator, s1_genome):
	"""Test search_neighbors for Stage 1."""
	print("\n" + "=" * 60)
	print("  Test: search_neighbors (Stage 1)")
	print("=" * 60)

	evaluator.target_stage = 1

	start = time.time()
	neighbors = evaluator.search_neighbors(
		genome=s1_genome,
		target_count=5,
		max_attempts=20,
		accuracy_threshold=0.0,  # Accept all (random data, accuracy will be low)
		min_bits=4,
		max_bits=12,
		min_neurons=5,
		max_neurons=30,
		bits_mutation_rate=0.3,
		neurons_mutation_rate=0.2,
		seed=42,
	)
	elapsed = time.time() - start

	print(f"\n  Found {len(neighbors)} neighbors in {elapsed:.2f}s")
	for i, g in enumerate(neighbors):
		ce, acc = g._cached_fitness
		print(f"    [{i}] CE={ce:.4f}, Acc={acc:.4%}, "
			  f"clusters={len(g.neurons_per_cluster)}, "
			  f"neurons={sum(g.neurons_per_cluster)}")

	assert len(neighbors) == 5, f"Expected 5 neighbors, got {len(neighbors)}"
	print("  PASS: search_neighbors returns correct count")


def test_batch_multi_genome(evaluator):
	"""Test evaluating multiple genomes in a single batch."""
	print("\n" + "=" * 60)
	print("  Test: Multi-Genome Batch Evaluation")
	print("=" * 60)

	evaluator.target_stage = 1
	genomes = [
		create_random_genome(evaluator.num_clusters, evaluator.total_input_bits, neurons=15, bits=6)
		for _ in range(5)
	]

	start = time.time()
	results = evaluator.evaluate_batch(genomes)
	elapsed = time.time() - start

	print(f"\n  Evaluated {len(genomes)} genomes in {elapsed:.2f}s")
	for i, r in enumerate(results):
		print(f"    [{i}] CE={r.ce:.4f}, Acc={r.accuracy:.4%}, BitAcc={r.bit_accuracy:.4%}")

	assert len(results) == 5
	assert all(isinstance(r, EvalResult) for r in results)
	assert all(r.ce > 0 for r in results)
	print("  PASS: Multi-genome batch evaluation works")


if __name__ == "__main__":
	print("\n" + "=" * 60)
	print("  TwoStageEvaluator End-to-End Test")
	print("=" * 60 + "\n")

	evaluator, s1_genome, s2_genome = test_basic_evaluation()
	test_combined_ce(evaluator, s1_genome, s2_genome)
	test_search_neighbors(evaluator, s1_genome)
	test_batch_multi_genome(evaluator)

	print("\n" + "=" * 60)
	print("  ALL TESTS PASSED")
	print("=" * 60 + "\n")
