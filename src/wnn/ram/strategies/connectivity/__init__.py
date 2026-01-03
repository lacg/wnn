"""
Connectivity Optimization Strategies.

Based on Garcia (2003) thesis on global optimization methods for
choosing connectivity patterns of weightless neural networks.

Usage:
	from wnn.ram.enums import OptimizationMethod
	from wnn.ram.strategies.connectivity import OptimizerStrategyFactory

	# Create strategy (Tabu Search recommended - best results from thesis)
	strategy = OptimizerStrategyFactory.create(OptimizationMethod.TABU_SEARCH)

	# Define evaluation function
	def evaluate_fn(connections):
		# Train and test with these connections
		return error_rate

	# Optimize
	result = strategy.optimize(
		connections=initial_connections,
		evaluate_fn=evaluate_fn,
		total_input_bits=16,
		num_neurons=8,
		n_bits_per_neuron=4,
	)

	print(f"Improved by {result.improvement_percent:.2f}%")
"""

from wnn.ram.strategies.connectivity.base import (
	OptimizerResult,
	OptimizerStrategyBase,
)

from wnn.ram.strategies.connectivity.tabu_search import (
	TabuSearchStrategy,
	TabuSearchConfig,
)
from wnn.ram.strategies.connectivity.simulated_annealing import (
	SimulatedAnnealingStrategy,
	SimulatedAnnealingConfig,
)
from wnn.ram.strategies.connectivity.genetic_algorithm import (
	GeneticAlgorithmStrategy,
	GeneticAlgorithmConfig,
)
from wnn.ram.strategies.connectivity.factory import OptimizerStrategyFactory
from wnn.ram.strategies.connectivity.accelerated import (
	AcceleratedOptimizer,
	OptimizerConfig,
	EvaluationContext,
	OptimizationStrategy,
	create_optimizer,
	RUST_AVAILABLE,
	RUST_CPU_CORES,
)


__all__ = [
	# Base
	'OptimizerResult',
	'OptimizerStrategyBase',
	# Tabu Search
	'TabuSearchStrategy',
	'TabuSearchConfig',
	# Simulated Annealing
	'SimulatedAnnealingStrategy',
	'SimulatedAnnealingConfig',
	# Genetic Algorithm
	'GeneticAlgorithmStrategy',
	'GeneticAlgorithmConfig',
	# Factory
	'OptimizerStrategyFactory',
	# Accelerated (recommended)
	'AcceleratedOptimizer',
	'OptimizerConfig',
	'EvaluationContext',
	'OptimizationStrategy',
	'create_optimizer',
	'RUST_AVAILABLE',
	'RUST_CPU_CORES',
]
