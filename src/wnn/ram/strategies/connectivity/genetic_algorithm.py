"""
Genetic Algorithm strategy for connectivity optimization.

From Garcia (2003) thesis:
- Good for exploring large search spaces
- Can reduce memory usage by 89% while maintaining accuracy
- Population-based approach with crossover and mutation
"""

import random
from dataclasses import dataclass
from typing import Callable, Optional

from torch import Tensor

from wnn.ram.strategies.connectivity.base import (
	OptimizerResult,
	OptimizerStrategyBase,
)


@dataclass
class GeneticAlgorithmConfig:
	"""
	Configuration for Genetic Algorithm optimization.

	Parameters from Garcia (2003) experiments:
	- population_size: 30 individuals
	- generations: 50 generations
	- mutation_rate: 0.01 per-connection
	- crossover_rate: 0.7 probability of crossover
	- elitism: 2 best individuals preserved each generation
	"""
	population_size: int = 30
	generations: int = 50
	mutation_rate: float = 0.01
	crossover_rate: float = 0.7
	elitism: int = 2


class GeneticAlgorithmStrategy(OptimizerStrategyBase):
	"""
	Genetic Algorithm optimization for RAM neuron connectivity patterns.

	Key features from Garcia (2003):
	- Population of connectivity patterns evolves over generations
	- Crossover exchanges connection patterns between parents
	- Mutation modifies individual connections
	- Elitism preserves best solutions across generations

	GA achieved 89% memory reduction on T0 topology while
	maintaining classification accuracy (fan-in 16 -> 14.66).
	"""

	def __init__(
		self,
		config: Optional[GeneticAlgorithmConfig] = None,
		seed: Optional[int] = None,
		verbose: bool = False,
	):
		super().__init__(seed=seed, verbose=verbose)
		self._config = config or GeneticAlgorithmConfig()

	@property
	def config(self) -> GeneticAlgorithmConfig:
		return self._config

	@property
	def name(self) -> str:
		return "GeneticAlgorithm"

	def optimize(
		self,
		connections: Tensor,
		evaluate_fn: Callable[[Tensor], float],
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult:
		"""
		Run Genetic Algorithm optimization.

		The algorithm:
		1. Initialize population from variations of initial connections
		2. Evaluate fitness of all individuals
		3. Select parents via tournament selection
		4. Apply crossover and mutation
		5. Preserve elite individuals
		6. Repeat for configured generations
		"""
		self._ensure_rng()
		cfg = self._config

		# Initialize population with variations of initial connections
		population = []
		for i in range(cfg.population_size):
			if i == 0:
				individual = connections.clone()
			else:
				individual, _ = self._generate_neighbor(
					connections, cfg.mutation_rate * 10,
					total_input_bits, num_neurons, n_bits_per_neuron
				)
			population.append(individual)

		# Evaluate initial population
		fitness = [evaluate_fn(ind) for ind in population]

		best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
		best = population[best_idx].clone()
		best_error = fitness[best_idx]
		initial_error = evaluate_fn(connections)

		history = [(0, best_error)]

		if self._verbose:
			print(f"[GA] Initial best error: {best_error:.4f}")

		for generation in range(cfg.generations):
			# Selection and reproduction
			new_population = []

			# Elitism: keep best individuals
			sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
			for i in range(cfg.elitism):
				new_population.append(population[sorted_indices[i]].clone())

			# Generate rest of population
			while len(new_population) < cfg.population_size:
				# Tournament selection
				p1 = self._tournament_select(population, fitness)
				p2 = self._tournament_select(population, fitness)

				# Crossover
				if random.random() < cfg.crossover_rate:
					child = self._crossover(p1, p2, num_neurons)
				else:
					child = p1.clone()

				# Mutation
				child, _ = self._generate_neighbor(
					child, cfg.mutation_rate,
					total_input_bits, num_neurons, n_bits_per_neuron
				)
				new_population.append(child)

			population = new_population[:cfg.population_size]
			fitness = [evaluate_fn(ind) for ind in population]

			# Update best
			gen_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
			if fitness[gen_best_idx] < best_error:
				best = population[gen_best_idx].clone()
				best_error = fitness[gen_best_idx]

			history.append((generation + 1, best_error))

			if self._verbose and (generation + 1) % 10 == 0:
				avg_fitness = sum(fitness) / len(fitness)
				print(f"[GA] Gen {generation + 1}: best={best_error:.4f}, avg={avg_fitness:.4f}")

		improvement_pct = ((initial_error - best_error) / initial_error * 100) if best_error < initial_error else 0.0

		return OptimizerResult(
			initial_connections=connections,
			optimized_connections=best,
			initial_error=initial_error,
			final_error=best_error,
			improvement_percent=improvement_pct,
			iterations_run=cfg.generations,
			method_name=self.name,
			history=history,
		)

	def _tournament_select(
		self,
		population: list,
		fitness: list,
		tournament_size: int = 3
	) -> Tensor:
		"""Tournament selection: pick best from random subset."""
		indices = random.sample(range(len(population)), min(tournament_size, len(population)))
		best_idx = min(indices, key=lambda i: fitness[i])
		return population[best_idx]

	def _crossover(self, parent1: Tensor, parent2: Tensor, num_neurons: int) -> Tensor:
		"""Single-point crossover: exchange neuron connections."""
		child = parent1.clone()
		crossover_point = random.randint(1, num_neurons - 1)
		child[crossover_point:] = parent2[crossover_point:].clone()
		return child

	def __repr__(self) -> str:
		return f"GeneticAlgorithmStrategy(config={self._config}, seed={self._seed}, verbose={self._verbose})"
