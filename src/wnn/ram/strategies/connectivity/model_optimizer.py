"""
ConnectivityOptimizer - High-level optimizer for RAM model connectivity.

This module provides a clean interface for optimizing the connectivity
of any RAM model (RAMLM, RAMClusterLayer, RAMLayer, etc.) using
GA/TS/SA optimization strategies.

The key insight from Garcia (2003): partial connectivity is THE
generalization mechanism in RAM WNNs. This optimizer finds connectivity
patterns where neurons observe "meaningful" bits that generalize.

Usage:
	from wnn.ram.core import OptimizationMethod
	from wnn.ram.strategies.connectivity import ConnectivityOptimizer

	optimizer = ConnectivityOptimizer(
		method=OptimizationMethod.TABU_SEARCH,
		verbose=True,
	)

	result = optimizer.optimize(
		model=ramlm,
		train_tokens=train_data,
		val_tokens=val_data,
	)

	print(f"Improved by {result.improvement_percent:.2f}%")
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Union

from torch import Tensor

from wnn.ram.core import OptimizationMethod
from wnn.ram.strategies.connectivity.base import OptimizerResult, OverfittingMonitor
from wnn.ram.strategies.connectivity.factory import OptimizerStrategyFactory
from wnn.ram.strategies.connectivity.tabu_search import TabuSearchConfig
from wnn.ram.strategies.connectivity.genetic_algorithm import GeneticAlgorithmConfig
from wnn.ram.strategies.connectivity.simulated_annealing import SimulatedAnnealingConfig


class RAMModel(Protocol):
	"""Protocol for RAM models that can be optimized."""

	@property
	def connections(self) -> Tensor:
		"""Get connectivity matrix [num_neurons, bits_per_neuron]."""
		...

	@connections.setter
	def connections(self, value: Tensor) -> None:
		"""Set connectivity matrix."""
		...

	def reset_memory(self) -> None:
		"""Reset all memory cells to EMPTY, preserving connectivity."""
		...


@dataclass
class OptimizationConfig:
	"""Configuration for connectivity optimization."""

	method: OptimizationMethod = OptimizationMethod.TABU_SEARCH

	# Tabu Search parameters (default - recommended)
	ts_iterations: int = 5
	ts_neighbors_per_iter: int = 30
	ts_tabu_size: int = 5
	ts_mutation_rate: float = 0.001

	# Genetic Algorithm parameters
	ga_population_size: int = 30
	ga_generations: int = 50
	ga_mutation_rate: float = 0.01
	ga_crossover_rate: float = 0.7
	ga_elitism: int = 2

	# Simulated Annealing parameters
	sa_iterations: int = 600
	sa_initial_temp: float = 1.0
	sa_cooling_rate: float = 0.95
	sa_mutation_rate: float = 0.01

	# Common parameters
	early_stop_patience: int = 3
	early_stop_threshold_pct: float = 0.02
	seed: Optional[int] = 42
	verbose: bool = True


@dataclass
class OptimizationResult:
	"""Result of connectivity optimization."""

	initial_cross_entropy: float
	final_cross_entropy: float
	improvement_percent: float
	iterations_run: int
	method: str
	early_stopped: bool = False

	# Detailed history (optional)
	history: list = field(default_factory=list)


class ConnectivityOptimizer:
	"""
	High-level optimizer for RAM model connectivity.

	This class provides a clean interface for optimizing any RAM model's
	connectivity using GA/TS/SA strategies. It handles:

	1. Creating the evaluation function (cross-entropy on validation data)
	2. Running the optimization strategy
	3. Resetting and retraining the model with new connectivity
	4. Tracking and returning results

	The optimization loop:
		for each candidate connectivity:
			1. Set model.connections = candidate
			2. Reset memory (model.reset_memory())
			3. Train (train_fn())
			4. Evaluate (eval_fn() -> cross-entropy)
			5. Return cross-entropy as fitness (lower is better)

	Attributes:
		config: OptimizationConfig with method and hyperparameters
		verbose: Whether to print progress
	"""

	def __init__(
		self,
		method: OptimizationMethod = OptimizationMethod.TABU_SEARCH,
		config: Optional[OptimizationConfig] = None,
		verbose: bool = True,
	):
		"""
		Initialize ConnectivityOptimizer.

		Args:
			method: Optimization method (TABU_SEARCH recommended)
			config: Optional full configuration (overrides method)
			verbose: Print progress during optimization
		"""
		if config is not None:
			self.config = config
		else:
			self.config = OptimizationConfig(method=method, verbose=verbose)

		self.verbose = verbose

	def optimize(
		self,
		model: RAMModel,
		train_fn: Callable[[], None],
		eval_fn: Callable[[], float],
		total_input_bits: Optional[int] = None,
		num_neurons: Optional[int] = None,
		bits_per_neuron: Optional[int] = None,
	) -> OptimizationResult:
		"""
		Optimize model connectivity using the configured strategy.

		Args:
			model: RAM model with .connections property and reset_memory() method
			train_fn: Function that trains the model (called after each reset)
			eval_fn: Function that evaluates the model, returns cross-entropy
			total_input_bits: Total input bits (inferred from model if not provided)
			num_neurons: Number of neurons (inferred from model if not provided)
			bits_per_neuron: Bits per neuron (inferred from model if not provided)

		Returns:
			OptimizationResult with improvement metrics
		"""
		# Infer dimensions from model if not provided
		connections = model.connections
		if total_input_bits is None:
			# Infer from max connection index + 1
			total_input_bits = int(connections.max().item()) + 1
		if num_neurons is None:
			num_neurons = connections.shape[0]
		if bits_per_neuron is None:
			bits_per_neuron = connections.shape[1]

		if self.verbose:
			print(f"=== Connectivity Optimization ===")
			print(f"  Method: {self.config.method.name}")
			print(f"  Neurons: {num_neurons:,}")
			print(f"  Input bits: {total_input_bits}")
			print(f"  Bits per neuron: {bits_per_neuron}")
			print()

		# Create evaluation function for the optimizer
		def evaluate_connectivity(candidate_connections: Tensor) -> float:
			"""Evaluate a connectivity pattern by training and measuring cross-entropy."""
			# Set new connectivity
			model.connections = candidate_connections

			# Reset memory (clear all learned mappings)
			model.reset_memory()

			# Train with new connectivity
			train_fn()

			# Evaluate and return cross-entropy (lower is better)
			return eval_fn()

		# Evaluate initial connectivity
		initial_ce = eval_fn()
		if self.verbose:
			print(f"Initial cross-entropy: {initial_ce:.4f}")

		# Create the optimization strategy
		strategy = self._create_strategy()

		# Run optimization
		result = strategy.optimize(
			connections=connections.clone(),
			evaluate_fn=evaluate_connectivity,
			total_input_bits=total_input_bits,
			num_neurons=num_neurons,
			n_bits_per_neuron=bits_per_neuron,
		)

		# Apply optimized connectivity and do final training
		model.connections = result.optimized_connections
		model.reset_memory()
		train_fn()
		final_ce = eval_fn()

		if self.verbose:
			print()
			print(f"Final cross-entropy: {final_ce:.4f}")
			print(f"Improvement: {result.improvement_percent:.2f}%")

		return OptimizationResult(
			initial_cross_entropy=initial_ce,
			final_cross_entropy=final_ce,
			improvement_percent=result.improvement_percent,
			iterations_run=result.iterations_run,
			method=result.method_name,
			early_stopped=result.early_stopped_overfitting,
			history=result.history,
		)

	def optimize_ramlm(
		self,
		model,  # RAMLM
		train_tokens: list[int],
		val_tokens: list[int],
		global_top_k: int = 1000,
	) -> OptimizationResult:
		"""
		Convenience method for optimizing RAMLM models.

		This wraps optimize() with RAMLM-specific train/eval functions.

		Args:
			model: RAMLM instance
			train_tokens: Training token IDs
			val_tokens: Validation token IDs
			global_top_k: Top-k for FALSE training

		Returns:
			OptimizationResult with improvement metrics
		"""
		def train_fn():
			model.train_epoch(train_tokens, global_top_k=global_top_k, verbose=False)

		def eval_fn() -> float:
			stats = model.evaluate(val_tokens, verbose=False)
			return stats['cross_entropy']

		# Get dimensions from RAMLM
		return self.optimize(
			model=model,
			train_fn=train_fn,
			eval_fn=eval_fn,
			total_input_bits=model.total_input_bits,
			num_neurons=model.layer.total_neurons,
			bits_per_neuron=model.layer.bits_per_neuron,
		)

	def _create_strategy(self):
		"""Create the optimization strategy based on config."""
		config = self.config

		if config.method == OptimizationMethod.TABU_SEARCH:
			strategy_config = TabuSearchConfig(
				iterations=config.ts_iterations,
				neighbors_per_iter=config.ts_neighbors_per_iter,
				tabu_size=config.ts_tabu_size,
				mutation_rate=config.ts_mutation_rate,
				early_stop_patience=config.early_stop_patience,
				early_stop_threshold_pct=config.early_stop_threshold_pct,
			)
		elif config.method == OptimizationMethod.GENETIC_ALGORITHM:
			strategy_config = GeneticAlgorithmConfig(
				population_size=config.ga_population_size,
				generations=config.ga_generations,
				mutation_rate=config.ga_mutation_rate,
				crossover_rate=config.ga_crossover_rate,
				elitism=config.ga_elitism,
				early_stop_patience=config.early_stop_patience,
				early_stop_threshold_pct=config.early_stop_threshold_pct,
			)
		elif config.method == OptimizationMethod.SIMULATED_ANNEALING:
			strategy_config = SimulatedAnnealingConfig(
				iterations=config.sa_iterations,
				initial_temp=config.sa_initial_temp,
				cooling_rate=config.sa_cooling_rate,
				mutation_rate=config.sa_mutation_rate,
			)
		else:
			raise ValueError(f"Unknown optimization method: {config.method}")

		return OptimizerStrategyFactory.create(
			config.method,
			config=strategy_config,
			seed=config.seed,
			verbose=config.verbose,
		)
