"""
Test script demonstrating advanced transformer training features:
1. Scheduled Sampling - bridges train/test gap for autoregressive models
2. Soft Attention via Voting - multiple heads vote on attention weights
3. Hard Example Mining - focuses training on difficult examples

Run: python tests/transformer_features.py
"""

import torch
from torch import tensor, zeros, ones


def test_scheduled_sampling():
	"""
	Demonstrates Scheduled Sampling for autoregressive training.

	Problem: During training, models see ground truth at every step (teacher forcing).
	During inference, they use their own predictions. This mismatch causes
	error accumulation.

	Solution: Gradually replace ground truth with model predictions during training.
	"""
	print("\n" + "=" * 70)
	print("1. SCHEDULED SAMPLING")
	print("=" * 70)

	from wnn.ram.core.models.seq2seq import RAMSeq2Seq

	# Create a small model for demonstration
	model = RAMSeq2Seq(
		input_bits=4,
		hidden_bits=8,
		num_layers=1,
		num_heads=2,
		max_seq_len=8,
		use_residual=True,
		rng=42,
	)

	# Create a simple copy dataset (input = output)
	def make_copy_dataset(n_samples: int, seq_len: int):
		dataset = []
		for i in range(n_samples):
			# Random sequence
			seq = [torch.randint(0, 2, (4,), dtype=torch.uint8) for _ in range(seq_len)]
			# For copy task: input = output
			dataset.append((seq, seq))
		return dataset

	dataset = make_copy_dataset(20, 4)

	print("\n--- Training with Teacher Forcing (baseline) ---")
	model_tf = RAMSeq2Seq(input_bits=4, hidden_bits=8, num_layers=1, num_heads=2, rng=42)
	history_tf = model_tf.train(dataset, epochs=5, verbose=True)

	print("\n--- Training with Scheduled Sampling (linear schedule) ---")
	model_ss = RAMSeq2Seq(input_bits=4, hidden_bits=8, num_layers=1, num_heads=2, rng=42)
	history_ss = model_ss.train_scheduled(
		dataset,
		epochs=5,
		schedule="linear",      # Linear decay from 1.0 to 0.0
		start_prob=1.0,         # Start with full teacher forcing
		end_prob=0.0,           # End with model predictions only
		verbose=True,
		rng_seed=42,
	)

	print("\n--- Training with Inverse Sigmoid Schedule ---")
	model_is = RAMSeq2Seq(input_bits=4, hidden_bits=8, num_layers=1, num_heads=2, rng=42)
	history_is = model_is.train_scheduled(
		dataset,
		epochs=5,
		schedule="inverse_sigmoid",  # Slower decay at start
		start_prob=1.0,
		end_prob=0.1,                # Don't go all the way to 0
		verbose=True,
		rng_seed=42,
	)

	print("\nSchedule comparison:")
	print(f"  Teacher Forcing final accuracy: {history_tf[-1]['accuracy']:.1f}%")
	print(f"  Linear SS final accuracy: {history_ss[-1]['accuracy']:.1f}%")
	print(f"  Inverse Sigmoid SS final accuracy: {history_is[-1]['accuracy']:.1f}%")

	# Show the schedule progression
	print("\nLinear schedule progression:")
	for i, h in enumerate(history_ss):
		print(f"  Epoch {i+1}: p(GT)={h['sampling_prob']:.2f}, actual GT ratio={h['gt_ratio']:.2f}")

	return True


def test_soft_attention_voting():
	"""
	Demonstrates Soft Attention via Voting.

	Problem: Standard RAM attention is "hard" - each head picks exactly one
	position to attend to. This is discrete and can be brittle.

	Solution: Multiple heads "vote" on attention weights, approximating
	continuous attention through aggregation.
	"""
	print("\n" + "=" * 70)
	print("2. SOFT ATTENTION VIA VOTING")
	print("=" * 70)

	from wnn.ram.core.models.soft_ram_attention import (
		SoftRAMAttention,
		AggregationStrategy,
	)

	# Create soft attention layer with voting
	soft_attn = SoftRAMAttention(
		input_bits=8,
		num_heads=4,            # 4 heads vote per position
		aggregation=AggregationStrategy.MAJORITY,  # Majority vote
		max_seq_len=8,
		rng=42,
	)

	# Create test tokens
	tokens = [
		tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8),  # Token 0
		tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.uint8),  # Token 1
		tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.uint8),  # Token 2
		tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.uint8),  # Token 3
	]

	print(f"\nSoftRAMAttention configuration:")
	print(f"  Input bits: {soft_attn.input_bits}")
	print(f"  Num heads: {soft_attn.num_heads}")
	print(f"  Aggregation: {soft_attn.aggregation.name}")

	# Forward pass
	outputs = soft_attn.forward(tokens)

	print(f"\nInput tokens: {len(tokens)}")
	print(f"Output tokens: {len(outputs)}")

	for i, (inp, out) in enumerate(zip(tokens, outputs)):
		print(f"  Position {i}: {inp.tolist()} → {out.tolist()}")

	# Compare different aggregation strategies
	print("\n--- Comparing Aggregation Strategies ---")

	for strategy in AggregationStrategy:
		attn = SoftRAMAttention(
			input_bits=8,
			num_heads=4,
			aggregation=strategy,
			max_seq_len=8,
			rng=42,
		)
		outputs = attn.forward(tokens)
		print(f"\n{strategy.name}:")
		for i, out in enumerate(outputs):
			print(f"  Position {i}: {out.tolist()}")

	# Train the soft attention
	print("\n--- Training Soft Attention ---")

	# Create target outputs (just copy for simplicity)
	targets = tokens.copy()

	# Train (using the parent class train_step if available)
	try:
		stats = soft_attn.train_step(tokens, targets)
		print(f"Training stats: {stats}")
	except AttributeError:
		print("Note: train_step not directly available on SoftRAMAttention")
		print("Use via RAMSeq2Seq with soft attention layer instead")

	return True


def test_hard_example_mining():
	"""
	Demonstrates Hard Example Mining.

	Problem: Random sampling treats all examples equally, but some are
	harder than others. Easy examples waste training time.

	Solution: Track example difficulty and focus training on hard examples.
	"""
	print("\n" + "=" * 70)
	print("3. HARD EXAMPLE MINING")
	print("=" * 70)

	from wnn.ram.core.trainer import HardExampleMiner

	# Create hard example miner
	num_examples = 15
	miner = HardExampleMiner(
		num_examples=num_examples,
		ema_alpha=0.3,  # Weight for new observations
	)

	print(f"\nHardExampleMiner configuration:")
	print(f"  Num examples: {num_examples}")
	print(f"  EMA alpha: {miner.ema_alpha}")

	# Simulate training with varying difficulty
	# Easy examples (0-4): rarely fail
	# Medium examples (5-9): sometimes fail
	# Hard examples (10-14): often fail
	print("\n--- Simulating Training Epochs ---")

	for epoch in range(5):
		print(f"\nEpoch {epoch + 1}:")

		for idx in range(num_examples):
			# Simulate different failure rates
			if idx < 5:
				# Easy examples: 10% failure rate
				had_error = torch.rand(1).item() < 0.1
			elif idx < 10:
				# Medium examples: 40% failure rate
				had_error = torch.rand(1).item() < 0.4
			else:
				# Hard examples: 80% failure rate
				had_error = torch.rand(1).item() < 0.8

			# Update miner
			miner.update(idx, had_error=had_error)

		# Get hard examples
		hard_indices = miner.get_hard_examples(count=5, min_difficulty=0.1)

		# Calculate average difficulty by category
		easy_diff = sum(miner.difficulties[i].difficulty_score for i in range(5)) / 5
		medium_diff = sum(miner.difficulties[i].difficulty_score for i in range(5, 10)) / 5
		hard_diff = sum(miner.difficulties[i].difficulty_score for i in range(10, 15)) / 5

		print(f"  Hard examples (top 5): {hard_indices}")
		print(f"  Avg difficulty - Easy: {easy_diff:.2f}, Medium: {medium_diff:.2f}, Hard: {hard_diff:.2f}")

	# Show individual example stats
	print("\n--- Example Difficulty Details ---")
	print(f"{'Idx':<4} {'Type':<8} {'Errors':<8} {'Successes':<10} {'EMA':<6} {'Score':<6}")
	print("-" * 50)
	for idx in range(num_examples):
		d = miner.difficulties[idx]
		category = "Easy" if idx < 5 else ("Medium" if idx < 10 else "Hard")
		print(f"{idx:<4} {category:<8} {d.error_count:<8} {d.success_count:<10} {d.ema_error:.2f}   {d.difficulty_score:.2f}")

	# Demonstrate focused training with RAMSeq2Seq
	print("\n--- Focused Training Integration ---")

	from wnn.ram.core.models.seq2seq import RAMSeq2Seq

	# Create dataset with varying difficulty
	def create_varied_dataset():
		dataset = []
		for i in range(15):
			if i < 5:
				# Easy: identity
				seq = [torch.randint(0, 2, (4,), dtype=torch.uint8) for _ in range(3)]
				dataset.append((seq, seq))
			elif i < 10:
				# Medium: bit flip last token
				seq = [torch.randint(0, 2, (4,), dtype=torch.uint8) for _ in range(3)]
				target = seq.copy()
				target[-1] = 1 - target[-1]
				dataset.append((seq, target))
			else:
				# Hard: reverse sequence
				seq = [torch.randint(0, 2, (4,), dtype=torch.uint8) for _ in range(3)]
				target = seq[::-1]
				dataset.append((seq, target))
		return dataset

	dataset = create_varied_dataset()
	model = RAMSeq2Seq(input_bits=4, hidden_bits=8, num_layers=1, num_heads=2, rng=42)

	print("\nTraining for 3 epochs with hard example tracking...")
	miner2 = HardExampleMiner(num_examples=len(dataset), ema_alpha=0.3)

	for epoch in range(3):
		epoch_errors = 0
		for idx, (inputs, targets) in enumerate(dataset):
			# Train
			stats = model.train_step(inputs, targets)
			had_error = stats["output_errors"] > 0
			epoch_errors += stats["output_errors"]

			# Track difficulty
			miner2.update(idx, had_error=had_error)

		hard_indices = miner2.get_hard_examples(count=5, min_difficulty=0.1)
		print(f"  Epoch {epoch+1}: {epoch_errors} errors, hard examples: {hard_indices}")

	return True


def main():
	"""Run all feature demonstrations."""
	print("\n" + "=" * 70)
	print("ADVANCED TRANSFORMER TRAINING FEATURES")
	print("=" * 70)
	print("\nThis script demonstrates three key features for improving")
	print("RAM transformer training:")
	print("  1. Scheduled Sampling - reduce exposure bias")
	print("  2. Soft Attention via Voting - smoother attention")
	print("  3. Hard Example Mining - focus on difficult cases")

	results = {}

	# Test 1: Scheduled Sampling
	try:
		results["scheduled_sampling"] = test_scheduled_sampling()
		print("\n✓ Scheduled Sampling: PASSED")
	except Exception as e:
		print(f"\n✗ Scheduled Sampling: FAILED - {e}")
		import traceback
		traceback.print_exc()
		results["scheduled_sampling"] = False

	# Test 2: Soft Attention via Voting
	try:
		results["soft_attention"] = test_soft_attention_voting()
		print("\n✓ Soft Attention via Voting: PASSED")
	except Exception as e:
		print(f"\n✗ Soft Attention via Voting: FAILED - {e}")
		import traceback
		traceback.print_exc()
		results["soft_attention"] = False

	# Test 3: Hard Example Mining
	try:
		results["hard_example_mining"] = test_hard_example_mining()
		print("\n✓ Hard Example Mining: PASSED")
	except Exception as e:
		print(f"\n✗ Hard Example Mining: FAILED - {e}")
		import traceback
		traceback.print_exc()
		results["hard_example_mining"] = False

	# Summary
	print("\n" + "=" * 70)
	print("SUMMARY")
	print("=" * 70)
	passed = sum(1 for v in results.values() if v)
	total = len(results)
	print(f"\nPassed: {passed}/{total}")

	for name, passed in results.items():
		status = "✓" if passed else "✗"
		print(f"  {status} {name.replace('_', ' ').title()}")

	return all(results.values())


if __name__ == "__main__":
	success = main()
	exit(0 if success else 1)
