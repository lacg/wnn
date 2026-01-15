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
	OptimizerStrategyBase,
	OverfittingControl,
	OverfittingCallback,
)
from wnn.ram.strategies.connectivity.generic_strategies import OptResult
from wnn.ram.strategies.perplexity import PerplexityCalculator


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
	- early_stop_patience: generations without improvement before stopping
	- early_stop_threshold: minimum PPL % improvement required to reset patience
	"""
	population_size: int = 30
	generations: int = 50
	mutation_rate: float = 0.01
	crossover_rate: float = 0.7
	elitism: int = 2
	early_stop_patience: int = 1  # Stop if no improvement for 1 check (5 generations)
	early_stop_threshold_pct: float = 0.02  # Minimum PPL % improvement required (0.02 = 0.02%)


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
		logger: Optional[Callable[[str], None]] = None,
	):
		super().__init__(seed=seed, verbose=verbose, logger=logger)
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
		batch_evaluate_fn: Optional[Callable[[list], list[float]]] = None,
		overfitting_callback: Optional[OverfittingCallback] = None,
		neuron_offsets: Optional[list[int]] = None,
	) -> OptResult[Tensor]:
		"""
		Run Genetic Algorithm optimization with fitness caching.

		The algorithm:
		1. Initialize population from variations of initial connections
		2. Evaluate fitness of all individuals (cached for elite)
		3. Select parents via tournament selection
		4. Apply crossover and mutation
		5. Preserve elite individuals WITH their cached fitness
		6. Repeat for configured generations

		Args:
			batch_evaluate_fn: Optional function to evaluate multiple patterns at once.
				If provided, uses batch evaluation for massive speedup with Rust/joblib.
			overfitting_callback: Optional callback for overfitting detection.
				Called every 5 generations with (best_connectivity, train_fitness).
				Returns OverfittingControl to signal diversity mode or early stop.
			neuron_offsets: Optional list of cumulative connection offsets for each neuron.
				Required for tiered architectures with variable bits per neuron.
				Example: [0, 20, 40, 60, 68, 76] means neuron 0 has connections 0-19,
				neuron 1 has 20-39, neuron 2 has 40-59, neuron 3 has 60-67, etc.
				If not provided, assumes uniform n_bits_per_neuron for all neurons.
		"""
		self._ensure_rng()
		cfg = self._config

		# Compute neuron offsets for tiered architectures
		# For 1D tensors with variable bits per neuron, we need boundaries for crossover
		if neuron_offsets is None and connections.dim() == 1:
			# Assume uniform bits per neuron
			neuron_offsets = [i * n_bits_per_neuron for i in range(num_neurons + 1)]
		elif neuron_offsets is not None:
			# Ensure we have the final offset (total connections)
			if len(neuron_offsets) == num_neurons:
				neuron_offsets = list(neuron_offsets) + [connections.shape[0]]

		# Diversity mode tracking - store original values for restoration
		in_diversity_mode = False
		in_severe_mode = False
		original_population_size = cfg.population_size
		original_elitism = cfg.elitism
		original_mutation_rate = cfg.mutation_rate
		early_stopped_overfitting = False

		# Helper to evaluate only individuals with unknown fitness
		def eval_with_cache(pop_with_fitness: list[tuple]) -> list[tuple]:
			"""Evaluate individuals with None fitness, keep cached values."""
			# Separate known and unknown fitness
			unknown_indices = [i for i, (_, f) in enumerate(pop_with_fitness) if f is None]

			if not unknown_indices:
				return pop_with_fitness  # All cached

			# Extract individuals needing evaluation
			to_eval = [pop_with_fitness[i][0] for i in unknown_indices]

			# Batch evaluate (pass total_pop for accurate logging)
			total_pop = len(pop_with_fitness)
			if batch_evaluate_fn is not None:
				new_fitness = batch_evaluate_fn(to_eval, total_pop=total_pop)
			else:
				new_fitness = [evaluate_fn(ind) for ind in to_eval]

			# Update fitness values
			result = list(pop_with_fitness)
			for idx, fit in zip(unknown_indices, new_fitness):
				result[idx] = (result[idx][0], fit)

			return result

		# Initialize population with variations of initial connections
		# All start with fitness=None (need evaluation)
		population = []
		for i in range(cfg.population_size):
			if i == 0:
				individual = connections.clone()
			else:
				individual, _ = self._generate_neighbor(
					connections, cfg.mutation_rate * 10,
					total_input_bits, num_neurons, n_bits_per_neuron
				)
			population.append((individual, None))  # (individual, fitness)

		# Evaluate initial population
		population = eval_with_cache(population)
		fitness = [f for _, f in population]

		best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
		best = population[best_idx][0].clone()
		best_error = fitness[best_idx]
		initial_error = fitness[0]  # First individual is the original

		history = [(0, best_error)]

		# Early stopping tracking
		patience_counter = 0
		prev_best_for_patience = best_error

		self._log(f"[GA] Initial best error: {best_error:.4f}")

		for generation in range(cfg.generations):
			# Selection and reproduction
			new_population = []

			# Elitism: keep best individuals WITH their cached fitness
			sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
			for i in range(cfg.elitism):
				elite_idx = sorted_indices[i]
				# Clone individual but KEEP the cached fitness
				new_population.append((population[elite_idx][0].clone(), fitness[elite_idx]))

			# Generate rest of population (new children have fitness=None)
			while len(new_population) < cfg.population_size:
				# Tournament selection (using cached fitness)
				p1 = self._tournament_select_cached(population)
				p2 = self._tournament_select_cached(population)

				# Crossover
				if random.random() < cfg.crossover_rate:
					child = self._crossover(p1, p2, num_neurons, neuron_offsets)
				else:
					child = p1.clone()

				# Mutation
				child, _ = self._generate_neighbor(
					child, cfg.mutation_rate,
					total_input_bits, num_neurons, n_bits_per_neuron
				)
				new_population.append((child, None))  # New child needs evaluation

			population = new_population[:cfg.population_size]
			population = eval_with_cache(population)  # Only evaluates None fitness!
			fitness = [f for _, f in population]

			# Update best
			gen_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
			if fitness[gen_best_idx] < best_error:
				best = population[gen_best_idx][0].clone()
				best_error = fitness[gen_best_idx]

			history.append((generation + 1, best_error))

			# Log every generation for progress visibility
			if self._verbose:
				gen_best_error = fitness[gen_best_idx]
				avg_fitness = sum(fitness) / len(fitness)
				self._log(f"[GA] Gen {generation + 1}/{cfg.generations}: current={gen_best_error:.4f}, best={best_error:.4f}, avg={avg_fitness:.4f}")

			# Early stopping check every 5 generations
			# Uses PPL-based improvement: since PPL = exp(CE), small CE changes = large PPL changes
			if (generation + 1) % 5 == 0:
				# Calculate PPL improvement percentage
				ppl_improvement = PerplexityCalculator.ppl_improvement_pct(prev_best_for_patience, best_error)

				if ppl_improvement >= cfg.early_stop_threshold_pct:
					patience_counter = 0
					prev_best_for_patience = best_error
				else:
					patience_counter += 1

				if self._verbose:
					self._log(f"[GA] Early stop check: PPL Δ={ppl_improvement:.2f}%, patience={cfg.early_stop_patience - patience_counter}/{cfg.early_stop_patience}")

				if patience_counter > cfg.early_stop_patience:
					self._log(f"[GA] Early stop at gen {generation + 1}: no PPL improvement >= {cfg.early_stop_threshold_pct}% for {patience_counter * 5} generations")
					break

				# Overfitting callback check
				if overfitting_callback is not None:
					control = overfitting_callback(best, best_error)

					if control.early_stop:
						self._log(f"[GA] Overfitting early stop at gen {generation + 1}")
						early_stopped_overfitting = True
						break

					if control.diversity_mode and not in_diversity_mode:
						# Activate diversity mode: ↑population, same elitism, ↑mutation
						in_diversity_mode = True
						in_severe_mode = control.severe_diversity_mode
						if control.severe_diversity_mode:
							# SEVERE: 2x population, same elitism, 2x mutation
							cfg.population_size = int(original_population_size * 2)
							cfg.mutation_rate = original_mutation_rate * 2
							self._log(f"[GA] SEVERE Diversity ON: pop={cfg.population_size}, elite={original_elitism}, mut={cfg.mutation_rate:.4f}")
						else:
							# MILD: 1.5x population, same elitism, 1.5x mutation
							cfg.population_size = int(original_population_size * 1.5)
							cfg.mutation_rate = original_mutation_rate * 1.5
							self._log(f"[GA] Diversity mode ON: pop={cfg.population_size}, elite={original_elitism}, mut={cfg.mutation_rate:.4f}")

						# Expand population with new random individuals
						while len(population) < cfg.population_size:
							new_ind, _ = self._generate_neighbor(
								best, cfg.mutation_rate * 5,
								total_input_bits, num_neurons, n_bits_per_neuron
							)
							population.append((new_ind, None))
						population = eval_with_cache(population)
						fitness = [f for _, f in population]

					elif control.diversity_mode and in_diversity_mode:
						# Check if severity level changed
						if control.severe_diversity_mode and not in_severe_mode:
							# Escalate to severe
							in_severe_mode = True
							cfg.population_size = int(original_population_size * 2)
							cfg.mutation_rate = original_mutation_rate * 2
							self._log(f"[GA] Escalating to SEVERE: pop={cfg.population_size}, mut={cfg.mutation_rate:.4f}")
							# Expand population
							while len(population) < cfg.population_size:
								new_ind, _ = self._generate_neighbor(
									best, cfg.mutation_rate * 5,
									total_input_bits, num_neurons, n_bits_per_neuron
								)
								population.append((new_ind, None))
							population = eval_with_cache(population)
							fitness = [f for _, f in population]
						elif not control.severe_diversity_mode and in_severe_mode:
							# De-escalate from severe to mild
							in_severe_mode = False
							cfg.population_size = int(original_population_size * 1.5)
							cfg.mutation_rate = original_mutation_rate * 1.5
							self._log(f"[GA] De-escalating to mild: pop={cfg.population_size}, mut={cfg.mutation_rate:.4f}")
							# Shrink population (keep best)
							sorted_pop = sorted(population, key=lambda x: x[1])
							population = sorted_pop[:cfg.population_size]
							fitness = [f for _, f in population]

					elif not control.diversity_mode and in_diversity_mode:
						# Back to normal: restore original hyperparameters
						in_diversity_mode = False
						in_severe_mode = False
						cfg.population_size = original_population_size
						cfg.elitism = original_elitism
						cfg.mutation_rate = original_mutation_rate
						self._log(f"[GA] Diversity mode OFF: restored pop={cfg.population_size}, elite={cfg.elitism}, mut={cfg.mutation_rate:.4f}")
						# Shrink population back to original size (keep best)
						# Population is already list[tuple[Tensor, float]], sort by fitness
						sorted_pop = sorted(population, key=lambda x: x[1])
						population = sorted_pop[:cfg.population_size]
						fitness = [f for _, f in population]

		improvement_pct = ((initial_error - best_error) / initial_error * 100) if best_error < initial_error else 0.0

		return OptResult(
			initial_genome=connections,
			best_genome=best,
			initial_fitness=initial_error,
			final_fitness=best_error,
			improvement_percent=improvement_pct,
			iterations_run=generation + 1,
			method_name=self.name,
			history=history,
			early_stopped=early_stopped_overfitting,
			stop_reason="overfitting" if early_stopped_overfitting else None,
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

	def _tournament_select_cached(
		self,
		population: list[tuple],
		tournament_size: int = 3
	) -> Tensor:
		"""Tournament selection with cached fitness: pick best from random subset."""
		indices = random.sample(range(len(population)), min(tournament_size, len(population)))
		# population is list of (individual, fitness) tuples
		best_idx = min(indices, key=lambda i: population[i][1])
		return population[best_idx][0]  # Return just the individual

	def _crossover(
		self,
		parent1: Tensor,
		parent2: Tensor,
		num_neurons: int,
		neuron_offsets: Optional[list[int]] = None,
	) -> Tensor:
		"""
		Single-point crossover: exchange neuron connections at neuron boundaries.

		For 2D tensors [num_neurons, bits_per_neuron]:
			Crossover at row (neuron) boundary.

		For 1D flattened tensors (tiered architectures):
			Uses neuron_offsets to find neuron boundaries and crossover there.
			This ensures we don't split a neuron's connections between parents.

		Args:
			parent1: First parent connectivity
			parent2: Second parent connectivity
			num_neurons: Total number of neurons
			neuron_offsets: Cumulative connection offsets [0, n0, n0+n1, ...] for 1D tensors
		"""
		child = parent1.clone()
		crossover_point = random.randint(1, num_neurons - 1)

		if parent1.dim() == 1 and neuron_offsets is not None:
			# 1D flattened tensor - use neuron_offsets to find boundary
			# crossover_point is neuron index, convert to connection index
			conn_boundary = neuron_offsets[crossover_point]
			child[conn_boundary:] = parent2[conn_boundary:].clone()
		else:
			# 2D tensor - crossover at row boundary
			child[crossover_point:] = parent2[crossover_point:].clone()

		return child

	def __repr__(self) -> str:
		return f"GeneticAlgorithmStrategy(config={self._config}, seed={self._seed}, verbose={self._verbose})"
