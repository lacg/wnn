"""
Contrastive Learning Test

Tests for contrastive learning capabilities:
- Triplet generation with hard negative mining
- Training on triplets to distinguish similar patterns
- Similarity metrics for bit vectors

Key insight: RAM networks learn to distinguish patterns by explicitly
training on pairs that should produce different outputs, with emphasis
on hard cases (similar inputs, different outputs).
"""

from datetime import datetime

from torch import tensor, uint8, zeros

from wnn.ram.core import (
	ContrastiveTrainer,
	Triplet,
	RAMTrainer,
	hamming_distance,
	jaccard_similarity,
	normalized_hamming_similarity,
)
from wnn.ram.core.models import RAMSeq2Seq
from wnn.ram.enums import MapperStrategy


# =============================================================================
# UTILITIES
# =============================================================================

def int_to_bits(n: int, n_bits: int) -> tensor:
	"""Convert integer to bit tensor (MSB first)."""
	return tensor([(n >> i) & 1 for i in range(n_bits - 1, -1, -1)], dtype=uint8)


def create_classification_dataset(n_bits: int = 4, n_classes: int = 4):
	"""
	Create a classification dataset with multiple examples per class.

	Each class has multiple input patterns that map to the same output.
	This creates opportunities for triplet learning.
	"""
	dataset = []
	examples_per_class = 4

	for class_id in range(n_classes):
		class_output = int_to_bits(class_id, n_bits)
		for variant in range(examples_per_class):
			# Create input: class_id in upper bits, variant in lower bits
			input_val = (class_id << 2) | variant
			input_bits = int_to_bits(input_val, n_bits)
			dataset.append(([input_bits], [class_output]))

	return dataset


def create_binary_classification_dataset(n_bits: int = 4, n_examples: int = 16):
	"""
	Create a binary classification dataset.

	Class 0: Numbers with even number of 1-bits
	Class 1: Numbers with odd number of 1-bits
	"""
	dataset = []

	for i in range(n_examples):
		input_bits = int_to_bits(i, n_bits)
		# Class based on parity of 1-bits
		parity = sum(int(b) for b in input_bits.tolist()) % 2
		output_bits = int_to_bits(parity, 1)
		dataset.append(([input_bits], [output_bits]))

	return dataset


# =============================================================================
# SIMILARITY METRICS TEST
# =============================================================================

def test_similarity_metrics():
	"""Test the similarity metrics for bit vectors."""
	print(f"\n{'='*60}")
	print("Testing Similarity Metrics")
	print(f"{'='*60}")

	# Test vectors
	a = tensor([1, 0, 1, 0], dtype=uint8)
	b = tensor([1, 0, 1, 0], dtype=uint8)  # Identical
	c = tensor([0, 1, 0, 1], dtype=uint8)  # Opposite
	d = tensor([1, 0, 0, 0], dtype=uint8)  # Partial match

	# Hamming distance
	print("\nHamming distance:")
	print(f"  a vs b (identical): {hamming_distance(a, b)}")
	print(f"  a vs c (opposite):  {hamming_distance(a, c)}")
	print(f"  a vs d (partial):   {hamming_distance(a, d)}")

	# Jaccard similarity
	print("\nJaccard similarity:")
	print(f"  a vs b (identical): {jaccard_similarity(a, b):.3f}")
	print(f"  a vs c (opposite):  {jaccard_similarity(a, c):.3f}")
	print(f"  a vs d (partial):   {jaccard_similarity(a, d):.3f}")

	# Normalized Hamming similarity
	print("\nNormalized Hamming similarity:")
	print(f"  a vs b (identical): {normalized_hamming_similarity(a, b):.3f}")
	print(f"  a vs c (opposite):  {normalized_hamming_similarity(a, c):.3f}")
	print(f"  a vs d (partial):   {normalized_hamming_similarity(a, d):.3f}")

	# Verify correctness
	assert hamming_distance(a, b) == 0, "Identical vectors should have distance 0"
	assert hamming_distance(a, c) == 4, "Opposite vectors should have max distance"
	assert normalized_hamming_similarity(a, b) == 1.0, "Identical should be 1.0"
	assert normalized_hamming_similarity(a, c) == 0.0, "Opposite should be 0.0"

	print("\n[OK] Similarity metrics working correctly")
	return True


# =============================================================================
# TRIPLET GENERATION TEST
# =============================================================================

def test_triplet_generation():
	"""Test triplet generation with hard negative mining."""
	print(f"\n{'='*60}")
	print("Testing Triplet Generation")
	print(f"{'='*60}")

	# Create dataset
	dataset = create_classification_dataset(n_bits=4, n_classes=4)
	print(f"Dataset: {len(dataset)} examples, 4 classes")

	# Create model and trainer
	model = RAMSeq2Seq(
		input_bits=4,
		hidden_bits=8,
		output_bits=4,
		num_layers=1,
		num_heads=2,
		use_residual=True,
		rng=42,
	)

	trainer = RAMTrainer(model, verbose=False)
	contrastive = ContrastiveTrainer(trainer, hard_negative_ratio=0.5)

	# Generate triplets
	triplets = contrastive.generate_triplets(dataset, max_triplets=20)
	print(f"\nGenerated {len(triplets)} triplets")

	if triplets:
		# Analyze hardness distribution
		hardnesses = [t.hardness for t in triplets]
		print(f"Hardness range: {min(hardnesses):.3f} - {max(hardnesses):.3f}")
		print(f"Average hardness: {sum(hardnesses)/len(hardnesses):.3f}")

		# Show a few example triplets
		print("\nExample triplets:")
		for i, t in enumerate(triplets[:3]):
			anchor_in = t.anchor[0][0].tolist()
			pos_in = t.positive[0][0].tolist()
			neg_in = t.negative[0][0].tolist()
			print(f"  {i+1}. anchor={anchor_in}, pos={pos_in}, neg={neg_in}, hardness={t.hardness:.3f}")

	return len(triplets) > 0


# =============================================================================
# CONTRASTIVE TRAINING TEST
# =============================================================================

def test_contrastive_training():
	"""Test contrastive training on a classification task."""
	print(f"\n{'='*60}")
	print("Testing Contrastive Training")
	print(f"{'='*60}")

	# Create dataset with clear class structure
	dataset = create_classification_dataset(n_bits=4, n_classes=4)
	print(f"Dataset: {len(dataset)} examples, 4 classes")

	# Create model with BIT_LEVEL generalization for better learning
	model = RAMSeq2Seq(
		input_bits=4,
		hidden_bits=16,  # Larger hidden layer
		output_bits=4,
		num_layers=2,    # More layers
		num_heads=4,     # More heads
		use_residual=True,
		generalization=MapperStrategy.BIT_LEVEL,  # Better generalization
		rng=42,
	)

	# Create trainers
	trainer = RAMTrainer(model, verbose=False)
	contrastive = ContrastiveTrainer(trainer, hard_negative_ratio=0.5)

	# Train with contrastive learning - more epochs
	print("\nTraining with contrastive learning:")
	history = contrastive.train(
		dataset,
		epochs=20,       # More epochs
		triplets_per_epoch=50,  # More triplets
		verbose=True,
	)

	# Evaluate
	if history:
		final_acc = history[-1]['accuracy']
		final_dist = history[-1]['distinction_rate']
		print(f"\nFinal accuracy: {final_acc:.1f}%")
		print(f"Final distinction rate: {final_dist:.1f}%")
		return final_acc, final_dist

	return 0, 0


# =============================================================================
# HARD NEGATIVE MINING TEST
# =============================================================================

def test_hard_negative_mining():
	"""Test hard negative mining to find confusing pairs."""
	print(f"\n{'='*60}")
	print("Testing Hard Negative Mining")
	print(f"{'='*60}")

	# Create binary classification dataset (parity)
	dataset = create_binary_classification_dataset(n_bits=4, n_examples=16)
	print(f"Dataset: {len(dataset)} examples (parity classification)")

	# Create model and trainer
	model = RAMSeq2Seq(
		input_bits=4,
		hidden_bits=8,
		output_bits=1,
		num_layers=1,
		num_heads=2,
		use_residual=True,
		rng=42,
	)

	trainer = RAMTrainer(model, verbose=False)
	contrastive = ContrastiveTrainer(trainer, margin=0.3)

	# Mine hard negatives
	hard_pairs = contrastive.mine_hard_negatives(dataset, top_k=5)
	print(f"\nFound {len(hard_pairs)} hard negative pairs:")

	for idx_a, idx_b, sim in hard_pairs:
		input_a = dataset[idx_a][0][0].tolist()
		input_b = dataset[idx_b][0][0].tolist()
		output_a = dataset[idx_a][1][0].tolist()
		output_b = dataset[idx_b][1][0].tolist()
		print(f"  [{idx_a}] {input_a}→{output_a} vs [{idx_b}] {input_b}→{output_b} (sim={sim:.3f})")

	return len(hard_pairs) > 0


# =============================================================================
# COMPARISON: STANDARD VS CONTRASTIVE
# =============================================================================

def create_copy_dataset(n_bits: int = 4, seq_len: int = 3, n_examples: int = 16):
	"""
	Create a copy task dataset: input sequence → same output sequence.

	This is a task RAM networks excel at and creates natural triplets:
	- Positive: Similar sequences that should produce similar outputs
	- Negative: Different sequences that should produce different outputs
	"""
	dataset = []
	for i in range(n_examples):
		seq = [int_to_bits((i + j) % (2**n_bits), n_bits) for j in range(seq_len)]
		dataset.append((seq, seq))  # Copy: output = input
	return dataset


def test_standard_vs_contrastive():
	"""Compare standard training vs contrastive refinement on copy task."""
	print(f"\n{'='*60}")
	print("Comparing Standard vs Contrastive on Copy Task")
	print(f"{'='*60}")

	# Create copy task dataset (RAM networks excel at this)
	dataset = create_copy_dataset(n_bits=4, seq_len=3, n_examples=20)
	print(f"Dataset: {len(dataset)} sequences of length 3")

	results = {}

	# Standard training only
	print("\n--- Standard Training Only ---")
	model_std = RAMSeq2Seq(
		input_bits=4,
		hidden_bits=8,
		output_bits=4,
		num_layers=1,
		num_heads=2,
		use_residual=True,
		rng=42,
	)
	trainer_std = RAMTrainer(model_std, verbose=False)
	history_std = trainer_std.train(dataset, epochs=30, early_stop=True)

	if history_std:
		results['standard_only'] = history_std[-1].accuracy
		print(f"Final accuracy: {history_std[-1].accuracy:.1f}%")

	# Hybrid: Standard training + Contrastive refinement
	print("\n--- Hybrid: Standard + Contrastive Refinement ---")
	model_hyb = RAMSeq2Seq(
		input_bits=4,
		hidden_bits=8,
		output_bits=4,
		num_layers=1,
		num_heads=2,
		use_residual=True,
		rng=42,
	)
	trainer_hyb = RAMTrainer(model_hyb, verbose=False)

	# Phase 1: Standard training to learn basic mappings
	print("  Phase 1: Standard training...")
	history_phase1 = trainer_hyb.train(dataset, epochs=20, early_stop=True)
	phase1_acc = history_phase1[-1].accuracy if history_phase1 else 0
	print(f"  After standard: {phase1_acc:.1f}%")

	# Phase 2: Contrastive refinement to improve distinctions
	print("  Phase 2: Contrastive refinement...")
	contrastive = ContrastiveTrainer(trainer_hyb, hard_negative_ratio=0.7)
	history_phase2 = contrastive.train(dataset, epochs=15, triplets_per_epoch=30, verbose=False)

	if history_phase2:
		results['hybrid'] = history_phase2[-1]['accuracy']
		results['hybrid_distinction'] = history_phase2[-1]['distinction_rate']
		print(f"  After contrastive: {history_phase2[-1]['accuracy']:.1f}%")
		print(f"  Distinction rate: {history_phase2[-1]['distinction_rate']:.1f}%")

	# Evaluate on full dataset after hybrid training
	print("\n  Evaluating hybrid model on full dataset...")
	correct = 0
	total = 0
	for inputs, targets in dataset:
		outputs = model_hyb.forward(inputs)
		for o, t in zip(outputs, targets):
			if (o.squeeze() == t.squeeze()).all():
				correct += 1
			total += 1
	final_acc = 100 * correct / total
	results['hybrid_final'] = final_acc
	print(f"  Hybrid final accuracy: {final_acc:.1f}%")

	return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
	print(f"\n{'='*60}")
	print("Contrastive Learning Test")
	print(f"Started at: {datetime.now()}")
	print(f"{'='*60}")

	# Test similarity metrics
	test_similarity_metrics()

	# Test triplet generation
	test_triplet_generation()

	# Test contrastive training
	train_acc, dist_rate = test_contrastive_training()

	# Test hard negative mining
	test_hard_negative_mining()

	# Compare standard vs contrastive
	comparison = test_standard_vs_contrastive()

	# Summary
	print(f"\n{'='*60}")
	print("SUMMARY")
	print(f"{'='*60}")

	print("\nSimilarity Metrics:")
	print("  - hamming_distance: Count differing bits")
	print("  - jaccard_similarity: Intersection/Union of 1-bits")
	print("  - normalized_hamming_similarity: 1 - (distance/max)")

	print("\nContrastive Training:")
	print(f"  Final accuracy: {train_acc:.1f}%")
	print(f"  Distinction rate: {dist_rate:.1f}%")

	print("\nStandard vs Hybrid (Standard + Contrastive):")
	for method, acc in comparison.items():
		print(f"  {method}: {acc:.1f}%")

	print("""
Key insights:
- Triplet training: (anchor, positive, negative) teaches discrimination
- Hard negative mining: Focus on similar inputs with different outputs
- Distinction rate: How well the model separates different classes
- Best approach: Standard training first, then contrastive refinement
- Contrastive alone doesn't teach all mappings; use it as refinement
""")
	print(f"Finished at: {datetime.now()}")
	print(f"{'='*60}")
