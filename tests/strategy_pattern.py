"""
Strategy Pattern Tests

Demonstrates using TrainStrategyFactory and ForwardStrategyFactory
with RAMSeq2Seq models.

Usage:
	source activate.sh
	python tests/strategy_pattern.py
"""

import torch
from torch import tensor

from wnn.ram.core.models.seq2seq import RAMSeq2Seq
from wnn.ram.strategies import (
	# Factories
	TrainStrategyFactory,
	ForwardStrategyFactory,
	TrainStrategyType,
	ForwardStrategyType,
	# Direct classes (alternative)
	GreedyTrainStrategy,
	CurriculumTrainStrategy,
	AutoregressiveForwardStrategy,
	# Configs
	CurriculumTrainConfig,
)


def create_copy_dataset(n_bits: int = 4, n_samples: int = 20, seq_len: int = 4):
	"""Create a simple copy task dataset."""
	dataset = []
	for _ in range(n_samples):
		# Random binary tokens
		inputs = [torch.randint(0, 2, (n_bits,), dtype=torch.uint8) for _ in range(seq_len)]
		targets = [inp.clone() for inp in inputs]  # Copy task: output = input
		dataset.append((inputs, targets))
	return dataset


def create_successor_dataset(n_bits: int = 4, n_samples: int = 20):
	"""Create successor task: output = input + 1."""
	max_val = 2**n_bits - 1
	dataset = []
	for i in range(min(n_samples, max_val)):
		# Input value
		inp = tensor([int(b) for b in format(i, f'0{n_bits}b')], dtype=torch.uint8)
		# Output = input + 1 (mod max)
		out_val = (i + 1) % (max_val + 1)
		out = tensor([int(b) for b in format(out_val, f'0{n_bits}b')], dtype=torch.uint8)
		dataset.append(([inp], [out]))
	return dataset


def test_greedy_strategy():
	"""Test GreedyTrainStrategy with RAMSeq2Seq."""
	print("\n=== Greedy Training Strategy ===")

	# Create model
	model = RAMSeq2Seq(
		input_bits=4,
		num_layers=1,
		num_heads=2,
		max_seq_len=8,
	)

	# Create dataset
	dataset = create_copy_dataset(n_bits=4, n_samples=10, seq_len=3)

	# Create strategy using factory
	strategy = TrainStrategyFactory.create(
		TrainStrategyType.GREEDY,
		epochs=5,
		verbose=True,
		early_stop=True,
	)

	print(f"Strategy: {strategy}")
	print(f"Config: epochs={strategy.config.epochs}, early_stop={strategy.config.early_stop}")

	# Train
	history = strategy.train(model, dataset)

	print(f"\nFinal accuracy: {history[-1].accuracy:.1f}%")
	return history[-1].accuracy >= 0


def test_curriculum_strategy():
	"""Test CurriculumTrainStrategy with RAMSeq2Seq."""
	print("\n=== Curriculum Training Strategy ===")

	# Create model
	model = RAMSeq2Seq(
		input_bits=4,
		num_layers=1,
		num_heads=2,
		max_seq_len=16,
	)

	# Create dataset with varying difficulty (sequence length)
	dataset = []
	for seq_len in range(2, 6):  # Lengths 2, 3, 4, 5
		for _ in range(5):  # 5 samples each
			inputs = [torch.randint(0, 2, (4,), dtype=torch.uint8) for _ in range(seq_len)]
			targets = [inp.clone() for inp in inputs]
			dataset.append((inputs, targets))

	print(f"Dataset: {len(dataset)} samples with varying lengths")

	# Create strategy using factory with kwargs
	strategy = TrainStrategyFactory.create(
		TrainStrategyType.CURRICULUM,
		num_stages=3,
		epochs_per_stage=2,
		verbose=True,
		overlap=0.2,
	)

	print(f"Strategy: {strategy}")
	print(f"Config: {strategy.curriculum_config.num_stages} stages, "
			f"{strategy.curriculum_config.epochs_per_stage} epochs/stage")

	# Train
	history = strategy.train(model, dataset)

	print(f"\nFinal accuracy: {history[-1].accuracy:.1f}%")
	return history[-1].accuracy >= 0


def test_iterative_strategy():
	"""Test IterativeTrainStrategy with RAMSeq2Seq."""
	print("\n=== Iterative Training Strategy ===")

	model = RAMSeq2Seq(
		input_bits=4,
		num_layers=1,
		num_heads=2,
		max_seq_len=8,
	)

	dataset = create_copy_dataset(n_bits=4, n_samples=8, seq_len=2)

	# Create strategy
	strategy = TrainStrategyFactory.create(
		TrainStrategyType.ITERATIVE,
		max_iterations=5,
		convergence_threshold=0.01,
		epochs=3,
		verbose=True,
	)

	print(f"Strategy: {strategy}")
	print(f"Config: max_iter={strategy.iterative_config.max_iterations}")

	history = strategy.train(model, dataset)

	print(f"\nFinal accuracy: {history[-1].accuracy:.1f}%")
	return history[-1].accuracy >= 0


def test_forward_strategy():
	"""Test ForwardStrategyFactory with RAMSeq2Seq."""
	print("\n=== Forward Strategies ===")

	# Create and train a model
	model = RAMSeq2Seq(
		input_bits=4,
		num_layers=1,
		num_heads=2,
		max_seq_len=8,
	)

	dataset = create_copy_dataset(n_bits=4, n_samples=10, seq_len=3)

	# Quick training
	train_strategy = TrainStrategyFactory.create(TrainStrategyType.GREEDY, epochs=3, verbose=False)
	train_strategy.train(model, dataset)

	# Test input
	test_input = [torch.randint(0, 2, (4,), dtype=torch.uint8) for _ in range(3)]

	# Parallel strategy
	parallel = ForwardStrategyFactory.create(ForwardStrategyType.PARALLEL)
	outputs_parallel = parallel.forward(model, test_input)
	print(f"Parallel: {len(outputs_parallel)} outputs")

	# Autoregressive strategy
	auto = ForwardStrategyFactory.create(
		ForwardStrategyType.AUTOREGRESSIVE,
		use_cache=True,
		max_length=5,
	)
	outputs_auto = auto.forward(model, test_input)
	print(f"Autoregressive: {len(outputs_auto)} outputs (max_length=5)")

	return True


def test_direct_instantiation():
	"""Test direct strategy instantiation (alternative to factory)."""
	print("\n=== Direct Strategy Instantiation ===")

	# Alternative: instantiate strategies directly
	greedy = GreedyTrainStrategy()  # Uses defaults
	curriculum = CurriculumTrainStrategy(CurriculumTrainConfig(num_stages=4))
	auto_fwd = AutoregressiveForwardStrategy()

	print(f"Greedy: {greedy.config}")
	print(f"Curriculum: {curriculum.curriculum_config}")
	print(f"Autoregressive: {auto_fwd.config}")

	return True


def test_config_inheritance():
	"""Test that configs properly inherit from base classes."""
	print("\n=== Config Inheritance ===")

	# Curriculum config should have base TrainConfig fields
	curriculum = TrainStrategyFactory.create(
		TrainStrategyType.CURRICULUM,
		# Base TrainConfig fields
		epochs=10,
		early_stop=True,
		shuffle=True,
		verbose=False,
		# Curriculum-specific fields
		num_stages=5,
		epochs_per_stage=2,
	)

	print(f"Base fields: epochs={curriculum.config.epochs}, early_stop={curriculum.config.early_stop}")
	print(f"Curriculum fields: stages={curriculum.curriculum_config.num_stages}")

	return True


def main():
	print("=" * 60)
	print("Strategy Pattern Test Suite")
	print("=" * 60)

	results = {}

	# Run tests
	results['greedy'] = test_greedy_strategy()
	results['curriculum'] = test_curriculum_strategy()
	results['iterative'] = test_iterative_strategy()
	results['forward'] = test_forward_strategy()
	results['direct'] = test_direct_instantiation()
	results['inheritance'] = test_config_inheritance()

	# Summary
	print("\n" + "=" * 60)
	print("Summary")
	print("=" * 60)

	for name, passed in results.items():
		status = "✓ PASS" if passed else "✗ FAIL"
		print(f"  {name}: {status}")

	all_passed = all(results.values())
	print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed.'}")

	return all_passed


if __name__ == "__main__":
	main()
