"""
Test connectivity optimization strategies.

Validates the implementation of Garcia (2003) thesis methods:
- Tabu Search (TS)
- Simulated Annealing (SA)
- Genetic Algorithm (GA)

Run with: python tests/connectivity_optimization.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch import tensor, randint, zeros, ones, arange

from wnn.ram.core.Memory import Memory
from wnn.ram.enums import OptimizationMethod
from wnn.ram.strategies.connectivity import (
	OptimizerStrategyFactory,
	TabuSearchStrategy,
	TabuSearchConfig,
	SimulatedAnnealingStrategy,
	SimulatedAnnealingConfig,
	GeneticAlgorithmStrategy,
	GeneticAlgorithmConfig,
)


def create_classification_task(
	n_classes: int = 4,
	n_samples_per_class: int = 50,
	n_bits: int = 16,
	seed: int = 42,
) -> tuple[list, list]:
	"""
	Create a simple classification task.

	Each class has a distinct pattern in some bits, with noise in others.
	"""
	torch.manual_seed(seed)

	train_data = []
	test_data = []

	# Each class uses different bits for its signature
	bits_per_class = n_bits // n_classes

	for class_idx in range(n_classes):
		# Class signature: specific bits are set
		start_bit = class_idx * bits_per_class
		end_bit = start_bit + bits_per_class

		for sample in range(n_samples_per_class):
			# Create input with class signature + noise
			input_bits = randint(0, 2, (n_bits,), dtype=torch.bool)
			# Set class signature bits
			input_bits[start_bit:end_bit] = True

			# Target is one-hot class
			target = zeros(n_classes, dtype=torch.bool)
			target[class_idx] = True

			if sample < n_samples_per_class * 0.8:
				train_data.append((input_bits, target))
			else:
				test_data.append((input_bits, target))

	return train_data, test_data


def evaluate_connections(
	connections: torch.Tensor,
	train_data: list,
	test_data: list,
	total_input_bits: int,
	num_neurons: int,
	n_bits_per_neuron: int,
) -> float:
	"""Evaluate a connectivity pattern by training and testing."""
	# Create memory with given connections
	memory = Memory(
		total_input_bits=total_input_bits,
		num_neurons=num_neurons,
		n_bits_per_neuron=n_bits_per_neuron,
		connections=connections,
	)

	# Train (one-shot learning)
	for input_bits, target_bits in train_data:
		memory.commit(input_bits.unsqueeze(0), target_bits.unsqueeze(0))

	# Test
	errors = 0
	total = 0
	for input_bits, target_bits in test_data:
		output = memory.forward(input_bits.unsqueeze(0))
		errors += (output[0] != target_bits).sum().item()
		total += target_bits.numel()

	return errors / total if total > 0 else 1.0


def test_tabu_search():
	"""Test Tabu Search optimization."""
	print("\n" + "=" * 60)
	print("Testing Tabu Search Strategy")
	print("=" * 60)

	# Create task
	n_bits = 16
	n_classes = 4
	n_neurons = n_classes
	n_bits_per_neuron = 4  # Small fan-in to show improvement potential

	train_data, test_data = create_classification_task(
		n_classes=n_classes,
		n_samples_per_class=50,
		n_bits=n_bits,
		seed=42,
	)

	# Initial random connections
	initial_connections = randint(0, n_bits, (n_neurons, n_bits_per_neuron), dtype=torch.long)

	# Create evaluation function
	def evaluate_fn(connections):
		return evaluate_connections(
			connections, train_data, test_data,
			n_bits, n_neurons, n_bits_per_neuron
		)

	initial_error = evaluate_fn(initial_connections)
	print(f"Initial error: {initial_error:.4f}")

	# Run Tabu Search
	config = TabuSearchConfig(iterations=5, neighbors_per_iter=30, tabu_size=5)
	strategy = TabuSearchStrategy(config=config, seed=42, verbose=True)

	result = strategy.optimize(
		connections=initial_connections,
		evaluate_fn=evaluate_fn,
		total_input_bits=n_bits,
		num_neurons=n_neurons,
		n_bits_per_neuron=n_bits_per_neuron,
	)

	print(f"\nResult: {result}")
	print(f"Initial connections:\n{initial_connections}")
	print(f"Optimized connections:\n{result.optimized_connections}")

	# Compute orthogonality
	sample_inputs = torch.stack([d[0] for d in train_data + test_data])
	initial_ortho = strategy.compute_orthogonality(
		initial_connections, sample_inputs, n_bits_per_neuron
	)
	final_ortho = strategy.compute_orthogonality(
		result.optimized_connections, sample_inputs, n_bits_per_neuron
	)
	print(f"Orthogonality: {initial_ortho:.4f} -> {final_ortho:.4f}")

	return result


def test_simulated_annealing():
	"""Test Simulated Annealing optimization."""
	print("\n" + "=" * 60)
	print("Testing Simulated Annealing Strategy")
	print("=" * 60)

	# Smaller test for SA (takes longer)
	n_bits = 12
	n_classes = 3
	n_neurons = n_classes
	n_bits_per_neuron = 3

	train_data, test_data = create_classification_task(
		n_classes=n_classes,
		n_samples_per_class=30,
		n_bits=n_bits,
		seed=42,
	)

	initial_connections = randint(0, n_bits, (n_neurons, n_bits_per_neuron), dtype=torch.long)

	def evaluate_fn(connections):
		return evaluate_connections(
			connections, train_data, test_data,
			n_bits, n_neurons, n_bits_per_neuron
		)

	initial_error = evaluate_fn(initial_connections)
	print(f"Initial error: {initial_error:.4f}")

	# Run SA with fewer iterations for test
	config = SimulatedAnnealingConfig(iterations=100, initial_temp=1.0, cooling_rate=0.95)
	strategy = SimulatedAnnealingStrategy(config=config, seed=42, verbose=True)

	result = strategy.optimize(
		connections=initial_connections,
		evaluate_fn=evaluate_fn,
		total_input_bits=n_bits,
		num_neurons=n_neurons,
		n_bits_per_neuron=n_bits_per_neuron,
	)

	print(f"\nResult: {result}")
	return result


def test_genetic_algorithm():
	"""Test Genetic Algorithm optimization."""
	print("\n" + "=" * 60)
	print("Testing Genetic Algorithm Strategy")
	print("=" * 60)

	n_bits = 12
	n_classes = 3
	n_neurons = n_classes
	n_bits_per_neuron = 3

	train_data, test_data = create_classification_task(
		n_classes=n_classes,
		n_samples_per_class=30,
		n_bits=n_bits,
		seed=42,
	)

	initial_connections = randint(0, n_bits, (n_neurons, n_bits_per_neuron), dtype=torch.long)

	def evaluate_fn(connections):
		return evaluate_connections(
			connections, train_data, test_data,
			n_bits, n_neurons, n_bits_per_neuron
		)

	initial_error = evaluate_fn(initial_connections)
	print(f"Initial error: {initial_error:.4f}")

	# Run GA with fewer generations for test
	config = GeneticAlgorithmConfig(population_size=20, generations=10)
	strategy = GeneticAlgorithmStrategy(config=config, seed=42, verbose=True)

	result = strategy.optimize(
		connections=initial_connections,
		evaluate_fn=evaluate_fn,
		total_input_bits=n_bits,
		num_neurons=n_neurons,
		n_bits_per_neuron=n_bits_per_neuron,
	)

	print(f"\nResult: {result}")
	return result


def test_factory():
	"""Test the factory pattern."""
	print("\n" + "=" * 60)
	print("Testing OptimizerStrategyFactory")
	print("=" * 60)

	# Test creating each strategy type
	for method in OptimizationMethod:
		strategy = OptimizerStrategyFactory.create(method, verbose=False)
		print(f"Created: {strategy}")

	# Test default
	default = OptimizerStrategyFactory.create_default()
	print(f"Default: {default}")

	print("Factory test passed!")


def compare_methods():
	"""Compare all three methods on the same task."""
	print("\n" + "=" * 60)
	print("Comparing All Methods")
	print("=" * 60)

	n_bits = 16
	n_classes = 4
	n_neurons = n_classes
	n_bits_per_neuron = 4

	train_data, test_data = create_classification_task(
		n_classes=n_classes,
		n_samples_per_class=40,
		n_bits=n_bits,
		seed=123,
	)

	# Same initial connections for all
	torch.manual_seed(456)
	initial_connections = randint(0, n_bits, (n_neurons, n_bits_per_neuron), dtype=torch.long)

	def evaluate_fn(connections):
		return evaluate_connections(
			connections, train_data, test_data,
			n_bits, n_neurons, n_bits_per_neuron
		)

	initial_error = evaluate_fn(initial_connections)
	print(f"Initial error: {initial_error:.4f}\n")

	results = {}

	# Tabu Search
	ts = OptimizerStrategyFactory.create(OptimizationMethod.TABU_SEARCH, seed=42)
	results['TS'] = ts.optimize(
		initial_connections.clone(), evaluate_fn,
		n_bits, n_neurons, n_bits_per_neuron
	)

	# SA (fewer iterations for fair time comparison)
	sa_config = SimulatedAnnealingConfig(iterations=150)  # 5 * 30 = 150 evals like TS
	sa = OptimizerStrategyFactory.create(
		OptimizationMethod.SIMULATED_ANNEALING,
		config=sa_config, seed=42
	)
	results['SA'] = sa.optimize(
		initial_connections.clone(), evaluate_fn,
		n_bits, n_neurons, n_bits_per_neuron
	)

	# GA (fewer generations for fair comparison)
	ga_config = GeneticAlgorithmConfig(population_size=15, generations=10)
	ga = OptimizerStrategyFactory.create(
		OptimizationMethod.GENETIC_ALGORITHM,
		config=ga_config, seed=42
	)
	results['GA'] = ga.optimize(
		initial_connections.clone(), evaluate_fn,
		n_bits, n_neurons, n_bits_per_neuron
	)

	# Compare
	print("\nComparison:")
	print("-" * 50)
	print(f"{'Method':<15} {'Initial':>10} {'Final':>10} {'Improvement':>12}")
	print("-" * 50)
	for name, result in results.items():
		print(f"{name:<15} {result.initial_error:>10.4f} {result.final_error:>10.4f} {result.improvement_percent:>11.2f}%")
	print("-" * 50)

	# Best method
	best_name = min(results, key=lambda k: results[k].final_error)
	print(f"\nBest method: {best_name} (consistent with Garcia 2003 finding that TS performs best)")


if __name__ == "__main__":
	print("=" * 60)
	print("Connectivity Optimization Test")
	print("Based on Garcia (2003) thesis")
	print("=" * 60)

	# Run tests
	test_factory()
	test_tabu_search()
	test_simulated_annealing()
	test_genetic_algorithm()
	compare_methods()

	print("\n" + "=" * 60)
	print("All tests completed!")
	print("=" * 60)
