"""
End-to-end test for MultiStageEvaluator.

Tests:
1. Construct MultiStageEvaluator with small token data
2. Create random genomes for Stage 0 and Stage 1
3. Evaluate Stage 0 genomes (subset + full)
4. Evaluate Stage 1 genomes (subset + full)
5. Compute combined CE and verify CE_combined ≈ CE_s0 + CE_s1
6. Run a mini GA-style evaluation (search_neighbors)
"""

import sys
import time
import random

# Setup path
sys.path.insert(0, "src/wnn")

from wnn.ram.architecture.multistage_evaluator import MultiStageEvaluator, compute_default_k
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


def test_compute_default_k():
	"""Test compute_default_k utility function."""
	print("=" * 60)
	print("  Test: compute_default_k")
	print("=" * 60)

	k_bb = compute_default_k(2, ["bitwise", "bitwise"])
	print(f"  bitwise+bitwise: {k_bb}")
	assert k_bb == [256, 256], f"Expected [256,256], got {k_bb}"

	k_tt = compute_default_k(2, ["tiered", "tiered"])
	print(f"  tiered+tiered:   {k_tt}")
	assert k_tt[0] * k_tt[1] >= 50257, f"Product {k_tt[0]*k_tt[1]} < 50257"

	k_mixed = compute_default_k(2, ["tiered", "bitwise"])
	print(f"  tiered+bitwise:  {k_mixed}")
	assert k_mixed[1] == 256
	assert k_mixed[0] * k_mixed[1] >= 50257

	print("  PASS: compute_default_k returns correct values")


def test_basic_evaluation():
	"""Test basic Stage 0 and Stage 1 evaluation."""
	print("\n" + "=" * 60)
	print("  Test: Basic MultiStageEvaluator Evaluation")
	print("=" * 60)

	# Small synthetic token data (vocab 1000, ~2K train, ~500 eval)
	vocab_size = 1000
	context_size = 4
	k = 32  # 32 token groups
	random.seed(42)

	train_tokens = [random.randint(0, vocab_size - 1) for _ in range(2000)]
	eval_tokens = [random.randint(0, vocab_size - 1) for _ in range(500)]

	# Create evaluator (target_stage=0, 0-indexed)
	print("\n  Creating MultiStageEvaluator (target_stage=0)...")
	evaluator = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		stage_k=[k, vocab_size // k + 1],
		target_stage=0,  # 0-indexed
		num_parts=2,
		num_eval_parts=1,
		seed=42,
		pad_token_id=0,
	)
	print(f"  Stage 0: num_clusters={evaluator.num_clusters}, total_input_bits={evaluator.total_input_bits}")

	# Stage 0 genome
	s0_genome = create_random_genome(
		num_clusters=evaluator.num_clusters,
		total_input_bits=evaluator.total_input_bits,
		neurons=20,
		bits=8,
	)
	print(f"  Stage 0 genome: {len(s0_genome.neurons_per_cluster)} clusters, "
		  f"{sum(s0_genome.neurons_per_cluster)} neurons")

	# Evaluate Stage 0
	start = time.time()
	results_s0 = evaluator.evaluate_batch([s0_genome])
	elapsed = time.time() - start
	print(f"\n  Stage 0 subset eval: CE={results_s0[0].ce:.4f}, "
		  f"Acc={results_s0[0].accuracy:.4%}, "
		  f"BitAcc={results_s0[0].bit_accuracy:.4%} ({elapsed:.2f}s)")

	# Full eval
	results_s0_full = evaluator.evaluate_batch_full([s0_genome])
	print(f"  Stage 0 full eval:   CE={results_s0_full[0].ce:.4f}, "
		  f"Acc={results_s0_full[0].accuracy:.4%}, "
		  f"BitAcc={results_s0_full[0].bit_accuracy:.4%}")

	# Switch to Stage 1 (0-indexed)
	evaluator.target_stage = 1
	print(f"\n  Switched to Stage 1: num_clusters={evaluator.num_clusters}, "
		  f"total_input_bits={evaluator.total_input_bits}")

	s1_genome = create_random_genome(
		num_clusters=evaluator.num_clusters,
		total_input_bits=evaluator.total_input_bits,
		neurons=20,
		bits=8,
	)
	print(f"  Stage 1 genome: {len(s1_genome.neurons_per_cluster)} clusters, "
		  f"{sum(s1_genome.neurons_per_cluster)} neurons")

	# Evaluate Stage 1
	start = time.time()
	results_s1 = evaluator.evaluate_batch([s1_genome])
	elapsed = time.time() - start
	print(f"  Stage 1 subset eval: CE={results_s1[0].ce:.4f}, "
		  f"Acc={results_s1[0].accuracy:.4%}, "
		  f"BitAcc={results_s1[0].bit_accuracy:.4%} ({elapsed:.2f}s)")

	results_s1_full = evaluator.evaluate_batch_full([s1_genome])
	print(f"  Stage 1 full eval:   CE={results_s1_full[0].ce:.4f}, "
		  f"Acc={results_s1_full[0].accuracy:.4%}, "
		  f"BitAcc={results_s1_full[0].bit_accuracy:.4%}")

	print("\n  PASS: Basic evaluation works for both stages")
	return evaluator, s0_genome, s1_genome


def test_combined_ce(evaluator, s0_genome, s1_genome):
	"""Test combined CE computation and verify CE_combined ≈ CE_s0 + CE_s1."""
	print("\n" + "=" * 60)
	print("  Test: Combined CE Computation")
	print("=" * 60)

	# Compute combined CE — takes a list of stage genomes (0-indexed)
	start = time.time()
	combined = evaluator.compute_combined_metrics([s0_genome, s1_genome])
	elapsed = time.time() - start

	print(f"\n  Combined CE:    {combined.ce:.4f}")
	print(f"  Combined Acc:   {combined.accuracy:.4%}")
	print(f"  Stage 0 CE:     {combined.cluster_ce:.4f}")
	print(f"  Stage 1 CE:     {combined.within_ce:.4f}")
	print(f"  S0 + S1:        {combined.cluster_ce + combined.within_ce:.4f}")
	print(f"  Elapsed:        {elapsed:.2f}s")

	# Verify CE_combined ≈ CE_s0 + CE_s1 (within 1% relative tolerance)
	sum_ce = combined.cluster_ce + combined.within_ce
	rel_error = abs(combined.ce - sum_ce) / max(combined.ce, 1e-10)
	print(f"\n  Relative error: {rel_error:.6f}")
	assert rel_error < 0.01, f"Combined CE mismatch: {combined.ce} vs {sum_ce} (rel={rel_error:.4f})"
	print("  PASS: CE_combined ≈ CE_s0 + CE_s1 (< 1% relative error)")


def test_search_neighbors(evaluator, s0_genome):
	"""Test search_neighbors for Stage 0."""
	print("\n" + "=" * 60)
	print("  Test: search_neighbors (Stage 0)")
	print("=" * 60)

	evaluator.target_stage = 0  # 0-indexed

	start = time.time()
	neighbors = evaluator.search_neighbors(
		genome=s0_genome,
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

	evaluator.target_stage = 0  # 0-indexed
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


def test_tiered_evaluation():
	"""Test tiered (direct K-class) evaluation for both stages."""
	print("\n" + "=" * 60)
	print("  Test: Tiered Evaluation (tiered+tiered)")
	print("=" * 60)

	vocab_size = 200
	context_size = 4
	k = 15
	random.seed(42)

	train_tokens = [random.randint(0, vocab_size - 1) for _ in range(2000)]
	eval_tokens = [random.randint(0, vocab_size - 1) for _ in range(500)]

	evaluator = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		stage_k=[k, vocab_size // k + 1],
		stage_cluster_type=["tiered", "tiered"],
		target_stage=0,
		num_parts=2,
		num_eval_parts=1,
		seed=42,
		pad_token_id=0,
	)

	print(f"  Stage 0: num_clusters={evaluator.num_clusters}, input_bits={evaluator.total_input_bits}")
	assert evaluator.num_clusters == k, f"Expected {k} clusters, got {evaluator.num_clusters}"

	# Stage 0 tiered genome
	s0_genome = create_random_genome(
		num_clusters=evaluator.num_clusters,
		total_input_bits=evaluator.total_input_bits,
		neurons=10,
		bits=6,
	)

	start = time.time()
	results = evaluator.evaluate_batch([s0_genome])
	elapsed = time.time() - start
	print(f"  Stage 0 tiered: CE={results[0].ce:.4f}, Acc={results[0].accuracy:.4%} ({elapsed:.2f}s)")
	assert results[0].ce > 0, f"CE should be positive, got {results[0].ce}"
	assert results[0].bit_accuracy == 0.0, f"Tiered has no bit_accuracy, got {results[0].bit_accuracy}"

	# Full eval
	results_full = evaluator.evaluate_batch_full([s0_genome])
	print(f"  Stage 0 full:   CE={results_full[0].ce:.4f}, Acc={results_full[0].accuracy:.4%}")

	# Switch to Stage 1
	evaluator.target_stage = 1
	print(f"\n  Stage 1: num_clusters={evaluator.num_clusters}, input_bits={evaluator.total_input_bits}")

	s1_genome = create_random_genome(
		num_clusters=evaluator.num_clusters,
		total_input_bits=evaluator.total_input_bits,
		neurons=10,
		bits=6,
	)
	results_s1 = evaluator.evaluate_batch([s1_genome])
	print(f"  Stage 1 tiered: CE={results_s1[0].ce:.4f}, Acc={results_s1[0].accuracy:.4%}")
	assert results_s1[0].ce > 0

	print("  PASS: Tiered evaluation works for both stages")
	return evaluator, s0_genome, s1_genome


def test_mixed_evaluation():
	"""Test mixed stage types (tiered+bitwise and bitwise+tiered)."""
	print("\n" + "=" * 60)
	print("  Test: Mixed Evaluation (tiered+bitwise)")
	print("=" * 60)

	vocab_size = 200
	context_size = 4
	k = 15
	random.seed(42)

	train_tokens = [random.randint(0, vocab_size - 1) for _ in range(2000)]
	eval_tokens = [random.randint(0, vocab_size - 1) for _ in range(500)]

	# tiered+bitwise
	evaluator_tb = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		stage_k=[k, vocab_size // k + 1],
		stage_cluster_type=["tiered", "bitwise"],
		target_stage=0,
		num_parts=2,
		num_eval_parts=1,
		seed=42,
		pad_token_id=0,
	)

	# Stage 0 is tiered → num_clusters = K
	assert evaluator_tb.num_clusters == k
	s0_genome = create_random_genome(evaluator_tb.num_clusters, evaluator_tb.total_input_bits, neurons=10, bits=6)
	results = evaluator_tb.evaluate_batch([s0_genome])
	print(f"  Stage 0 (tiered): CE={results[0].ce:.4f}, Acc={results[0].accuracy:.4%}")

	# Stage 1 is bitwise → num_clusters = bits_per_within_index
	evaluator_tb.target_stage = 1
	s1_genome = create_random_genome(evaluator_tb.num_clusters, evaluator_tb.total_input_bits, neurons=10, bits=6)
	results_s1 = evaluator_tb.evaluate_batch([s1_genome])
	print(f"  Stage 1 (bitwise): CE={results_s1[0].ce:.4f}, Acc={results_s1[0].accuracy:.4%}")

	# bitwise+tiered
	print("\n  Testing bitwise+tiered...")
	evaluator_bt = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		stage_k=[k, vocab_size // k + 1],
		stage_cluster_type=["bitwise", "tiered"],
		target_stage=0,
		num_parts=2,
		num_eval_parts=1,
		seed=42,
		pad_token_id=0,
	)

	s0_genome_bt = create_random_genome(evaluator_bt.num_clusters, evaluator_bt.total_input_bits, neurons=10, bits=6)
	results_bt = evaluator_bt.evaluate_batch([s0_genome_bt])
	print(f"  Stage 0 (bitwise): CE={results_bt[0].ce:.4f}, Acc={results_bt[0].accuracy:.4%}")

	evaluator_bt.target_stage = 1
	s1_genome_bt = create_random_genome(evaluator_bt.num_clusters, evaluator_bt.total_input_bits, neurons=10, bits=6)
	results_bt_s1 = evaluator_bt.evaluate_batch([s1_genome_bt])
	print(f"  Stage 1 (tiered): CE={results_bt_s1[0].ce:.4f}, Acc={results_bt_s1[0].accuracy:.4%}")

	print("  PASS: Mixed evaluation works for both combinations")


def test_frequency_mapping():
	"""Test that frequency-interleaved mapping produces balanced clusters."""
	print("\n" + "=" * 60)
	print("  Test: Frequency-Interleaved Cluster Mapping")
	print("=" * 60)

	vocab_size = 200
	context_size = 4
	k = 10
	random.seed(42)

	# Create tokens with skewed frequency (Zipfian-like)
	train_tokens = []
	for _ in range(5000):
		# Token 0 appears most, token 199 least
		token = min(int(random.expovariate(0.02)), vocab_size - 1)
		train_tokens.append(token)
	eval_tokens = [random.randint(0, vocab_size - 1) for _ in range(500)]

	evaluator = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		stage_k=[k, vocab_size // k + 1],
		stage_cluster_type=["tiered", "tiered"],
		target_stage=0,
		num_parts=2,
		num_eval_parts=1,
		seed=42,
		pad_token_id=0,
	)

	# Check cluster sizes are balanced
	sizes = evaluator.cluster_sizes
	min_size = min(sizes)
	max_size = max(sizes)
	print(f"  Cluster sizes: min={min_size}, max={max_size}, k={k}")
	print(f"  Size ratio: {max_size / max(min_size, 1):.2f}")

	# With frequency-interleaved mapping, sizes should be nearly equal
	# (vocab_size / k = 20, so sizes should all be ~20)
	assert max_size - min_size <= 1, f"Clusters unbalanced: {min_size} to {max_size}"
	print("  PASS: Frequency mapping produces balanced clusters")


def test_combined_ce_tiered(evaluator, s0_genome, s1_genome):
	"""Test combined CE for tiered stages."""
	print("\n" + "=" * 60)
	print("  Test: Combined CE (Tiered)")
	print("=" * 60)

	start = time.time()
	combined = evaluator.compute_combined_metrics([s0_genome, s1_genome])
	elapsed = time.time() - start

	print(f"\n  Combined CE:    {combined.ce:.4f}")
	print(f"  Combined Acc:   {combined.accuracy:.4%}")
	print(f"  Stage 0 CE:     {combined.cluster_ce:.4f}")
	print(f"  Stage 1 CE:     {combined.within_ce:.4f}")
	print(f"  S0 + S1:        {combined.cluster_ce + combined.within_ce:.4f}")
	print(f"  Elapsed:        {elapsed:.2f}s")

	sum_ce = combined.cluster_ce + combined.within_ce
	rel_error = abs(combined.ce - sum_ce) / max(combined.ce, 1e-10)
	print(f"\n  Relative error: {rel_error:.6f}")
	assert rel_error < 0.01, f"Combined CE mismatch: {combined.ce} vs {sum_ce} (rel={rel_error:.4f})"
	print("  PASS: CE_combined ≈ CE_s0 + CE_s1 (tiered, < 1% relative error)")


if __name__ == "__main__":
	print("\n" + "=" * 60)
	print("  MultiStageEvaluator End-to-End Test")
	print("=" * 60 + "\n")

	# Bitwise tests
	test_compute_default_k()
	evaluator, s0_genome, s1_genome = test_basic_evaluation()
	test_combined_ce(evaluator, s0_genome, s1_genome)
	test_search_neighbors(evaluator, s0_genome)
	test_batch_multi_genome(evaluator)

	# Tiered tests
	tiered_evaluator, tiered_s0, tiered_s1 = test_tiered_evaluation()
	test_mixed_evaluation()
	test_frequency_mapping()
	test_combined_ce_tiered(tiered_evaluator, tiered_s0, tiered_s1)

	print("\n" + "=" * 60)
	print("  ALL TESTS PASSED")
	print("=" * 60 + "\n")
