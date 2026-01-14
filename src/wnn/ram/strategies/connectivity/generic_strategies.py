"""
Generic GA and TS strategy base classes.

These provide genome-agnostic optimization algorithms that can be specialized
for different genome types (connectivity patterns, architecture configurations, etc.)
through abstract genome operations.

The core GA/TS loops are implemented here, subclasses provide:
- clone_genome: Copy a genome
- mutate_genome: Generate a neighbor by mutation
- crossover_genomes: Combine two parents (GA only)
"""

import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Any

# Generic genome type
T = TypeVar('T')


@dataclass
class GenericOptResult(Generic[T]):
	"""Result from generic optimization."""

	initial_genome: T
	best_genome: T
	initial_fitness: float
	final_fitness: float
	improvement_percent: float
	iterations_run: int
	method_name: str
	history: List[Tuple[int, float]] = field(default_factory=list)
	early_stopped: bool = False
	# For population seeding
	final_population: Optional[List[T]] = None
	# Accuracy tracking
	initial_accuracy: Optional[float] = None
	final_accuracy: Optional[float] = None


@dataclass
class GAConfig:
	"""Configuration for Genetic Algorithm."""
	population_size: int = 30
	generations: int = 50
	mutation_rate: float = 0.1
	crossover_rate: float = 0.7
	elitism: int = 2
	patience: int = 5
	min_improvement_pct: float = 0.01


@dataclass
class TSConfig:
	"""Configuration for Tabu Search."""
	iterations: int = 100
	neighbors_per_iter: int = 20
	tabu_size: int = 10
	patience: int = 10
	min_improvement_pct: float = 0.01


class GenericGAStrategy(ABC, Generic[T]):
	"""
	Generic Genetic Algorithm strategy.

	Subclasses must implement genome operations:
	- clone_genome: Copy a genome
	- mutate_genome: Generate a mutated variant
	- crossover_genomes: Combine two parents
	- create_random_genome: Create a new random genome

	The core GA loop (selection, crossover, mutation, elitism) is implemented here.
	"""

	def __init__(
		self,
		config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
	):
		self._config = config or GAConfig()
		self._seed = seed
		self._log = logger or print
		self._rng: Optional[random.Random] = None

	@property
	def config(self) -> GAConfig:
		return self._config

	@property
	def name(self) -> str:
		return "GenericGA"

	def _ensure_rng(self) -> None:
		if self._rng is None:
			self._rng = random.Random(self._seed)

	# =========================================================================
	# Abstract genome operations - subclasses must implement
	# =========================================================================

	@abstractmethod
	def clone_genome(self, genome: T) -> T:
		"""Create a deep copy of the genome."""
		...

	@abstractmethod
	def mutate_genome(self, genome: T, mutation_rate: float) -> T:
		"""Create a mutated variant of the genome."""
		...

	@abstractmethod
	def crossover_genomes(self, parent1: T, parent2: T) -> T:
		"""Create a child by combining two parents."""
		...

	@abstractmethod
	def create_random_genome(self) -> T:
		"""Create a new random genome (for population initialization)."""
		...

	# =========================================================================
	# Core GA loop
	# =========================================================================

	def optimize(
		self,
		evaluate_fn: Callable[[T], float],
		initial_genome: Optional[T] = None,
		initial_population: Optional[List[T]] = None,
		batch_evaluate_fn: Optional[Callable[[List[T]], List[float]]] = None,
		accuracy_fn: Optional[Callable[[T], float]] = None,
	) -> GenericOptResult[T]:
		"""
		Run Genetic Algorithm optimization.

		Args:
			evaluate_fn: Function to evaluate a single genome (lower is better)
			initial_genome: Optional seed genome (used if no initial_population)
			initial_population: Optional seed population from previous phase
			batch_evaluate_fn: Optional batch evaluation function
			accuracy_fn: Optional function to compute accuracy

		Returns:
			GenericOptResult with best genome and statistics
		"""
		self._ensure_rng()
		cfg = self._config

		# Initialize population
		population: List[Tuple[T, Optional[float]]] = []  # (genome, fitness)

		if initial_population:
			# Seed from previous phase
			self._log(f"[{self.name}] Seeding population from {len(initial_population)} genomes")
			for genome in initial_population[:cfg.population_size]:
				population.append((self.clone_genome(genome), None))
			# Fill remaining with mutations of best (first) genome
			while len(population) < cfg.population_size:
				mutant = self.mutate_genome(self.clone_genome(initial_population[0]), cfg.mutation_rate * 3)
				population.append((mutant, None))
		elif initial_genome is not None:
			# Seed from single genome
			population.append((self.clone_genome(initial_genome), None))
			for _ in range(cfg.population_size - 1):
				mutant = self.mutate_genome(self.clone_genome(initial_genome), cfg.mutation_rate * 3)
				population.append((mutant, None))
		else:
			# Random initialization
			for _ in range(cfg.population_size):
				population.append((self.create_random_genome(), None))

		# Evaluate initial population
		population = self._evaluate_population(population, batch_evaluate_fn, evaluate_fn)
		fitness_values = [f for _, f in population]

		# Find initial best
		best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
		best = self.clone_genome(population[best_idx][0])
		best_fitness = fitness_values[best_idx]
		initial_fitness = fitness_values[0] if initial_genome else best_fitness

		# Get initial accuracy if function provided
		initial_accuracy = None
		if accuracy_fn is not None:
			initial_accuracy = accuracy_fn(best)

		history = [(0, best_fitness)]
		patience_counter = 0
		prev_best = best_fitness

		self._log(f"[{self.name}] Initial best: {best_fitness:.4f}")

		generation = 0
		for generation in range(cfg.generations):
			# Selection and reproduction
			new_population: List[Tuple[T, Optional[float]]] = []

			# Elitism: keep best individuals with cached fitness
			sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
			for i in range(cfg.elitism):
				elite_idx = sorted_indices[i]
				new_population.append((self.clone_genome(population[elite_idx][0]), fitness_values[elite_idx]))

			# Generate rest of population
			while len(new_population) < cfg.population_size:
				# Tournament selection
				p1 = self._tournament_select(population)
				p2 = self._tournament_select(population)

				# Crossover
				if self._rng.random() < cfg.crossover_rate:
					child = self.crossover_genomes(p1, p2)
				else:
					child = self.clone_genome(p1)

				# Mutation
				child = self.mutate_genome(child, cfg.mutation_rate)
				new_population.append((child, None))  # Needs evaluation

			population = new_population[:cfg.population_size]
			population = self._evaluate_population(population, batch_evaluate_fn, evaluate_fn)
			fitness_values = [f for _, f in population]

			# Update best
			gen_best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
			if fitness_values[gen_best_idx] < best_fitness:
				best = self.clone_genome(population[gen_best_idx][0])
				best_fitness = fitness_values[gen_best_idx]

			history.append((generation + 1, best_fitness))

			# Log progress
			gen_avg = sum(fitness_values) / len(fitness_values)
			self._log(f"[{self.name}] Gen {generation + 1}/{cfg.generations}: "
					  f"best={best_fitness:.4f}, gen_best={fitness_values[gen_best_idx]:.4f}, avg={gen_avg:.4f}")

			# Early stopping check
			improvement_pct = (prev_best - best_fitness) / prev_best * 100 if prev_best > 0 else 0
			if improvement_pct >= cfg.min_improvement_pct:
				patience_counter = 0
				prev_best = best_fitness
			else:
				patience_counter += 1

			if patience_counter >= cfg.patience:
				self._log(f"[{self.name}] Early stop at gen {generation + 1}: no improvement for {cfg.patience} gens")
				break

		# Get final accuracy
		final_accuracy = None
		if accuracy_fn is not None:
			final_accuracy = accuracy_fn(best)

		# Extract final population for seeding next phase
		final_population = [self.clone_genome(g) for g, _ in sorted(population, key=lambda x: x[1])]

		improvement_pct = (initial_fitness - best_fitness) / initial_fitness * 100 if initial_fitness > 0 else 0

		return GenericOptResult(
			initial_genome=initial_genome if initial_genome else population[0][0],
			best_genome=best,
			initial_fitness=initial_fitness,
			final_fitness=best_fitness,
			improvement_percent=improvement_pct,
			iterations_run=generation + 1,
			method_name=self.name,
			history=history,
			early_stopped=patience_counter >= cfg.patience,
			final_population=final_population,
			initial_accuracy=initial_accuracy,
			final_accuracy=final_accuracy,
		)

	def _evaluate_population(
		self,
		population: List[Tuple[T, Optional[float]]],
		batch_fn: Optional[Callable[[List[T]], List[float]]],
		single_fn: Callable[[T], float],
	) -> List[Tuple[T, float]]:
		"""Evaluate individuals with None fitness."""
		unknown_indices = [i for i, (_, f) in enumerate(population) if f is None]

		if not unknown_indices:
			return [(g, f) for g, f in population]  # All cached

		to_eval = [population[i][0] for i in unknown_indices]

		if batch_fn is not None:
			new_fitness = batch_fn(to_eval)
		else:
			new_fitness = [single_fn(g) for g in to_eval]

		result = list(population)
		for idx, fit in zip(unknown_indices, new_fitness):
			result[idx] = (result[idx][0], fit)

		return result

	def _tournament_select(self, population: List[Tuple[T, float]], tournament_size: int = 3) -> T:
		"""Tournament selection: pick best from random subset."""
		indices = self._rng.sample(range(len(population)), min(tournament_size, len(population)))
		best_idx = min(indices, key=lambda i: population[i][1])
		return population[best_idx][0]


class GenericTSStrategy(ABC, Generic[T]):
	"""
	Generic Tabu Search strategy.

	Subclasses must implement genome operations:
	- clone_genome: Copy a genome
	- mutate_genome: Generate a neighbor with move info
	- is_tabu_move: Check if a move reverses a tabu move

	The core TS loop (neighbor generation, tabu filtering, selection) is implemented here.
	"""

	def __init__(
		self,
		config: Optional[TSConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
	):
		self._config = config or TSConfig()
		self._seed = seed
		self._log = logger or print
		self._rng: Optional[random.Random] = None

	@property
	def config(self) -> TSConfig:
		return self._config

	@property
	def name(self) -> str:
		return "GenericTS"

	def _ensure_rng(self) -> None:
		if self._rng is None:
			self._rng = random.Random(self._seed)

	# =========================================================================
	# Abstract genome operations - subclasses must implement
	# =========================================================================

	@abstractmethod
	def clone_genome(self, genome: T) -> T:
		"""Create a deep copy of the genome."""
		...

	@abstractmethod
	def mutate_genome(self, genome: T, mutation_rate: float) -> Tuple[T, Any]:
		"""Create a neighbor. Returns (new_genome, move_info)."""
		...

	@abstractmethod
	def is_tabu_move(self, move: Any, tabu_list: List[Any]) -> bool:
		"""Check if a move is tabu (reverses a recent move)."""
		...

	# =========================================================================
	# Core TS loop
	# =========================================================================

	def optimize(
		self,
		initial_genome: T,
		initial_fitness: float,
		evaluate_fn: Callable[[T], float],
		initial_neighbors: Optional[List[T]] = None,
		batch_evaluate_fn: Optional[Callable[[List[T]], List[float]]] = None,
		accuracy_fn: Optional[Callable[[T], float]] = None,
	) -> GenericOptResult[T]:
		"""
		Run Tabu Search optimization.

		Args:
			initial_genome: Starting genome
			initial_fitness: Fitness of initial genome
			evaluate_fn: Function to evaluate a single genome
			initial_neighbors: Optional seed neighbors from previous phase
			batch_evaluate_fn: Optional batch evaluation function
			accuracy_fn: Optional function to compute accuracy

		Returns:
			GenericOptResult with best genome and statistics
		"""
		self._ensure_rng()
		cfg = self._config

		current = self.clone_genome(initial_genome)
		current_fitness = initial_fitness

		best = self.clone_genome(initial_genome)
		best_fitness = initial_fitness
		start_fitness = initial_fitness

		# Get initial accuracy
		initial_accuracy = None
		if accuracy_fn is not None:
			initial_accuracy = accuracy_fn(best)

		# Tabu list
		tabu_list: deque = deque(maxlen=cfg.tabu_size)

		# Track best neighbors for seeding next phase
		best_neighbors: List[Tuple[T, float]] = [(self.clone_genome(initial_genome), initial_fitness)]

		# Seed with initial neighbors if provided
		if initial_neighbors:
			self._log(f"[{self.name}] Seeding from {len(initial_neighbors)} neighbors")
			# Evaluate seed neighbors
			if batch_evaluate_fn is not None:
				seed_fitness = batch_evaluate_fn(initial_neighbors)
			else:
				seed_fitness = [evaluate_fn(g) for g in initial_neighbors]
			for g, f in zip(initial_neighbors, seed_fitness):
				best_neighbors.append((self.clone_genome(g), f))
			# Sort and keep top k
			best_neighbors.sort(key=lambda x: x[1])
			best_neighbors = best_neighbors[:cfg.neighbors_per_iter]
			# Update best if seed had better
			if best_neighbors[0][1] < best_fitness:
				best = self.clone_genome(best_neighbors[0][0])
				best_fitness = best_neighbors[0][1]
				current = self.clone_genome(best)
				current_fitness = best_fitness

		history = [(0, best_fitness)]
		patience_counter = 0
		prev_best = best_fitness

		self._log(f"[{self.name}] Initial: {start_fitness:.4f}")

		iteration = 0
		for iteration in range(cfg.iterations):
			# Generate neighbors
			neighbor_candidates: List[Tuple[T, Any]] = []
			for _ in range(cfg.neighbors_per_iter):
				neighbor, move = self.mutate_genome(self.clone_genome(current), 1.0)
				if not self.is_tabu_move(move, list(tabu_list)):
					neighbor_candidates.append((neighbor, move))

			if not neighbor_candidates:
				continue

			# Batch evaluate
			if batch_evaluate_fn is not None:
				to_eval = [n for n, _ in neighbor_candidates]
				fitness_values = batch_evaluate_fn(to_eval)
				neighbors = [(n, f, m) for (n, m), f in zip(neighbor_candidates, fitness_values)]
			else:
				neighbors = [(n, evaluate_fn(n), m) for n, m in neighbor_candidates]

			# Select best neighbor
			neighbors.sort(key=lambda x: x[1])
			best_neighbor, best_neighbor_fitness, best_move = neighbors[0]

			# Move to best neighbor (always - TS characteristic)
			current = best_neighbor
			current_fitness = best_neighbor_fitness

			# Add move to tabu list
			if best_move is not None:
				tabu_list.append(best_move)

			# Update global best
			if current_fitness < best_fitness:
				best = self.clone_genome(current)
				best_fitness = current_fitness

			# Track best neighbors for seeding
			for n, f, _ in neighbors[:5]:  # Keep top 5 from each iteration
				best_neighbors.append((self.clone_genome(n), f))
			best_neighbors.sort(key=lambda x: x[1])
			best_neighbors = best_neighbors[:cfg.neighbors_per_iter]

			history.append((iteration + 1, best_fitness))

			# Log progress
			avg_fitness = sum(f for _, f, _ in neighbors) / len(neighbors)
			self._log(f"[{self.name}] Iter {iteration + 1}/{cfg.iterations}: "
					  f"best={best_fitness:.4f}, current={current_fitness:.4f}, avg={avg_fitness:.4f}")

			# Early stopping check
			improvement_pct = (prev_best - best_fitness) / prev_best * 100 if prev_best > 0 else 0
			if improvement_pct >= cfg.min_improvement_pct:
				patience_counter = 0
				prev_best = best_fitness
			else:
				patience_counter += 1

			if patience_counter >= cfg.patience:
				self._log(f"[{self.name}] Early stop at iter {iteration + 1}: no improvement for {cfg.patience} iters")
				break

		# Get final accuracy
		final_accuracy = None
		if accuracy_fn is not None:
			final_accuracy = accuracy_fn(best)

		# Extract final neighbors for seeding
		final_neighbors = [self.clone_genome(g) for g, _ in best_neighbors]

		improvement_pct = (start_fitness - best_fitness) / start_fitness * 100 if start_fitness > 0 else 0

		return GenericOptResult(
			initial_genome=initial_genome,
			best_genome=best,
			initial_fitness=start_fitness,
			final_fitness=best_fitness,
			improvement_percent=improvement_pct,
			iterations_run=iteration + 1,
			method_name=self.name,
			history=history,
			early_stopped=patience_counter >= cfg.patience,
			final_population=final_neighbors,  # For seeding next phase
			initial_accuracy=initial_accuracy,
			final_accuracy=final_accuracy,
		)
