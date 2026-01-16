"""
Generic GA and TS strategy base classes.

These provide genome-agnostic optimization algorithms that can be specialized
for different genome types (connectivity patterns, architecture configurations, etc.)
through abstract genome operations.

The core GA/TS loops are implemented here, subclasses provide:
- clone_genome: Copy a genome
- mutate_genome: Generate a neighbor by mutation
- crossover_genomes: Combine two parents (GA only)

Supports:
- Early stopping with patience and delta logging (EarlyStoppingTracker)
- Overfitting detection via callbacks (OverfittingCallback)
- Diversity mode for escaping local optima
"""

import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Any

# Generic genome type
T = TypeVar('T')


class StopReason(IntEnum):
	"""Reason why optimization stopped early."""
	CONVERGENCE = auto()  # No improvement for patience iterations
	OVERFITTING = auto()  # Overfitting callback triggered early stop
	MAX_ITERATIONS = auto()  # Reached maximum iterations (not early stopped)


@dataclass
class OptimizerResult(Generic[T]):
	"""
	Unified result from optimization (GA, TS, SA).

	This is a generic result type that works with any genome type (Tensor, ClusterGenome, etc.)
	through the type parameter T.

	Naming conventions:
	- Uses 'genome' terminology (more generic than 'connections')
	- Uses 'fitness' terminology (minimization by default, lower is better)

	Attributes:
		initial_genome: Starting genome before optimization
		best_genome: Best genome found during optimization
		initial_fitness: Fitness of initial genome (lower is better)
		final_fitness: Fitness of best genome
		improvement_percent: Percentage improvement ((initial - final) / initial * 100)
		iterations_run: Number of iterations/generations run
		method_name: Name of the optimization method (e.g., "ArchitectureGA")
		history: List of (iteration, best_fitness) tuples for plotting
		early_stopped: Whether optimization stopped early (due to convergence or overfitting)
		stop_reason: Why optimization stopped (StopReason enum)
		final_population: Final population for seeding next phase (GA/TS)
		initial_accuracy: Optional accuracy at start
		final_accuracy: Optional accuracy at end
	"""
	initial_genome: T
	best_genome: T
	initial_fitness: float
	final_fitness: float
	improvement_percent: float
	iterations_run: int
	method_name: str
	history: List[Tuple[int, float]] = field(default_factory=list)
	early_stopped: bool = False
	stop_reason: Optional[StopReason] = None
	# For population seeding between phases
	final_population: Optional[List[T]] = None
	# Accuracy tracking
	initial_accuracy: Optional[float] = None
	final_accuracy: Optional[float] = None

	def __repr__(self) -> str:
		stop_str = f", stop={self.stop_reason.name}" if self.stop_reason else ""
		return (
			f"OptimizerResult("
			f"method={self.method_name}, "
			f"initial={self.initial_fitness:.4f}, "
			f"final={self.final_fitness:.4f}, "
			f"improvement={self.improvement_percent:.2f}%{stop_str})"
		)


@dataclass
class EarlyStoppingConfig:
	"""Configuration for early stopping with patience."""
	patience: int = 5              # Number of checks without improvement before stopping
	check_interval: int = 5        # Check every N iterations/generations
	min_improvement_pct: float = 0.02  # Minimum % improvement required to reset patience


class EarlyStoppingTracker:
	"""
	Reusable early stopping tracker with patience and delta logging.

	Checks improvement at regular intervals (default: every 5 iterations).
	Logs delta improvement and patience counter (e.g., "Δ=0.15%, patience=3/5").
	Stops when patience is exhausted (no improvement for patience * check_interval iterations).

	Usage:
		tracker = EarlyStoppingTracker(config, logger)
		for iteration in range(max_iterations):
			# ... do work ...
			if tracker.check(iteration, current_best_fitness):
				break  # Early stop
	"""

	def __init__(
		self,
		config: EarlyStoppingConfig,
		logger: Callable[[str], None],
		method_name: str = "Optimizer",
	):
		self._config = config
		self._log = logger
		self._method_name = method_name
		self._patience_counter = 0
		self._prev_best: Optional[float] = None
		self._baseline: Optional[float] = None

	def reset(self, initial_fitness: float) -> None:
		"""Reset tracker with initial fitness value."""
		self._patience_counter = 0
		self._prev_best = initial_fitness
		self._baseline = initial_fitness

	def check(self, iteration: int, current_best: float) -> bool:
		"""
		Check if early stopping should occur.

		Args:
			iteration: Current iteration (0-indexed)
			current_best: Current best fitness value (lower is better)

		Returns:
			True if should stop, False otherwise
		"""
		cfg = self._config

		# Only check at specified intervals (1-indexed iteration)
		if (iteration + 1) % cfg.check_interval != 0:
			return False

		# Compute improvement from last check
		if self._prev_best is not None and self._prev_best > 0:
			improvement_pct = (self._prev_best - current_best) / self._prev_best * 100
		else:
			improvement_pct = 0.0

		# Check if improvement meets threshold
		if improvement_pct >= cfg.min_improvement_pct:
			self._patience_counter = 0
			self._prev_best = current_best
		else:
			self._patience_counter += 1

		# Determine status using OverfitThreshold values (negate since improvement is opposite sign)
		# improvement_pct > 0 = improving, OverfitThreshold delta < 0 = healthy
		from wnn.core.thresholds import OverfitThreshold
		delta = -improvement_pct  # Convert to OverfitThreshold convention
		if delta < OverfitThreshold.HEALTHY:  # < -1% (big improvement)
			status = "🟢 HEALTHY"
		elif delta < OverfitThreshold.WARNING:  # -1% to 0% (small improvement)
			status = "⚪ NEUTRAL"
		elif delta < OverfitThreshold.CRITICAL:  # 0% to 3% (stalled/mild regression)
			status = "🟡 WARNING"
		else:  # >= 3% (significant regression)
			status = "🔴 CRITICAL"

		# Log progress with delta, patience, and status
		remaining = cfg.patience - self._patience_counter
		self._log(
			f"[{self._method_name}] Early stop check: "
			f"Δ={improvement_pct:+.2f}%, patience={remaining}/{cfg.patience} {status}"
		)

		# Check if patience exhausted
		if self._patience_counter >= cfg.patience:
			total_iters_without_improvement = self._patience_counter * cfg.check_interval
			self._log(
				f"[{self._method_name}] Early stop: no improvement >= {cfg.min_improvement_pct}% "
				f"for {total_iters_without_improvement} iterations"
			)
			return True

		return False

	@property
	def patience_exhausted(self) -> bool:
		"""Check if patience is exhausted."""
		return self._patience_counter >= self._config.patience


@dataclass
class GAConfig:
	"""Configuration for Genetic Algorithm."""
	population_size: int = 30
	generations: int = 50
	mutation_rate: float = 0.1
	crossover_rate: float = 0.7
	# Dual elitism: keep top N% by CE AND top N% by accuracy (unique)
	# Total elites = 10-20% of population depending on overlap
	elitism_pct: float = 0.1       # 10% by CE + 10% by accuracy
	# Candidate filtering: replace candidates with accuracy below threshold
	min_accuracy: float = 0.0001   # 0.01% minimum accuracy to be viable
	# Early stopping (all configurable via parameters)
	patience: int = 5              # Checks without improvement before stopping
	check_interval: int = 5        # Check every N generations
	min_improvement_pct: float = 0.05  # GA needs diversity, lower threshold (0.05%)


@dataclass
class TSConfig:
	"""Configuration for Tabu Search."""
	iterations: int = 100
	neighbors_per_iter: int = 20
	tabu_size: int = 10
	# Candidate filtering: filter out neighbors with accuracy below threshold
	min_accuracy: float = 0.0001   # 0.01% minimum accuracy to be viable
	# Early stopping (all configurable via parameters)
	patience: int = 5              # Checks without improvement before stopping
	check_interval: int = 5        # Check every N iterations
	min_improvement_pct: float = 0.5  # TS is focused, higher threshold (0.5%)


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
		batch_evaluate_fn: Optional[Callable[[List[T]], List[Tuple[float, float]]]] = None,
		overfitting_callback: Optional[Callable[[T, float], Any]] = None,
	) -> OptimizerResult[T]:
		"""
		Run Genetic Algorithm optimization.

		Args:
			evaluate_fn: Function to evaluate a single genome (lower is better)
			initial_genome: Optional seed genome (used if no initial_population)
			initial_population: Optional seed population from previous phase
			batch_evaluate_fn: Optional batch evaluation function returning List[(CE, accuracy)]
			overfitting_callback: Optional callback for overfitting detection.
				Called every check_interval generations with (best_genome, best_fitness).
				Returns OverfittingControl (or any object with early_stop attribute).
				If early_stop=True, optimization stops with stop_reason=StopReason.OVERFITTING.

		Returns:
			OptimizerResult with best genome and statistics
		"""
		self._ensure_rng()
		cfg = self._config

		# Initialize population with fitness and accuracy tracking
		# Each entry is (genome, fitness, accuracy) where fitness/accuracy can be None (needs evaluation)
		population: List[Tuple[T, Optional[float], Optional[float]]] = []  # (genome, fitness, accuracy)

		if initial_population:
			# Seed from previous phase
			self._log(f"[{self.name}] Seeding population from {len(initial_population)} genomes")
			for genome in initial_population[:cfg.population_size]:
				population.append((self.clone_genome(genome), None, None))
			# Fill remaining with mutations of best (first) genome
			while len(population) < cfg.population_size:
				mutant = self.mutate_genome(self.clone_genome(initial_population[0]), cfg.mutation_rate * 3)
				population.append((mutant, None, None))
		elif initial_genome is not None:
			# Seed from single genome
			population.append((self.clone_genome(initial_genome), None, None))
			for _ in range(cfg.population_size - 1):
				mutant = self.mutate_genome(self.clone_genome(initial_genome), cfg.mutation_rate * 3)
				population.append((mutant, None, None))
		else:
			# Random initialization
			for _ in range(cfg.population_size):
				population.append((self.create_random_genome(), None, None))

		# Evaluate initial population
		population = self._evaluate_population(population, batch_evaluate_fn, evaluate_fn)

		# Filter initial candidates: replace those with accuracy below threshold
		replaced_count = 0
		max_replacements = 3  # Limit replacement attempts
		for attempt in range(max_replacements):
			needs_replacement = []
			for i, (genome, fitness, accuracy) in enumerate(population):
				if accuracy is not None and accuracy < cfg.min_accuracy:
					needs_replacement.append(i)

			if not needs_replacement:
				break

			# Replace with new random genomes
			for idx in needs_replacement:
				new_genome = self.create_random_genome()
				population[idx] = (new_genome, None, None)
				replaced_count += 1

			# Re-evaluate only the replaced genomes
			population = self._evaluate_population(population, batch_evaluate_fn, evaluate_fn)

		if replaced_count > 0:
			self._log(f"[{self.name}] Initial: replaced {replaced_count} candidates with accuracy < {cfg.min_accuracy:.2%}")

		fitness_values = [f for _, f, _ in population]
		accuracy_values = [a for _, _, a in population]

		# Find initial best
		best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
		best = self.clone_genome(population[best_idx][0])
		best_fitness = fitness_values[best_idx]
		initial_fitness = fitness_values[0] if initial_genome else best_fitness
		initial_accuracy = accuracy_values[best_idx]

		history = [(0, best_fitness)]

		# Initialize early stopping tracker
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stopper = EarlyStoppingTracker(early_stop_config, self._log, self.name)
		early_stopper.reset(best_fitness)

		# Track initial diversity (CE spread)
		initial_ce_spread = max(fitness_values) - min(fitness_values) if fitness_values else 0.0

		# Log config and initial best
		self._log(f"[{self.name}] Config: pop={cfg.population_size}, gens={cfg.generations}, "
				  f"elitism={cfg.elitism_pct:.0%} per metric, "
				  f"patience={cfg.patience}, check_interval={cfg.check_interval}, min_delta={cfg.min_improvement_pct}%")
		self._log(f"[{self.name}] Initial best: {best_fitness:.4f}, diversity (CE spread): {initial_ce_spread:.4f}")

		# Tracking for analysis
		elite_wins = 0  # Iterations where elite beat new offspring
		improved_iterations = 0
		# Track elites from first generation for survival analysis
		initial_elite_genomes = None  # Will be set after first generation

		generation = 0
		for generation in range(cfg.generations):
			# Selection and reproduction
			new_population: List[Tuple[T, Optional[float], Optional[float]]] = []

			# Dual elitism: keep top N% by CE AND top N% by accuracy (unique)
			n_per_metric = max(1, int(len(population) * cfg.elitism_pct))

			# Top by CE (lower is better)
			ce_sorted = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
			ce_elite_indices = set(ce_sorted[:n_per_metric])

			# Top by accuracy (higher is better) - only if we have accuracy values
			has_accuracy = any(a is not None for a in accuracy_values)
			if has_accuracy:
				acc_sorted = sorted(range(len(accuracy_values)),
									key=lambda i: -(accuracy_values[i] or 0))
				acc_elite_indices = set(acc_sorted[:n_per_metric])
			else:
				acc_elite_indices = set()

			# Combine unique elites (preserves CE order for ties)
			all_elite_indices = list(ce_elite_indices)
			for idx in acc_elite_indices:
				if idx not in ce_elite_indices:
					all_elite_indices.append(idx)

			overlap_count = len(ce_elite_indices & acc_elite_indices)
			total_elites = len(all_elite_indices)

			# Track initial elites (first generation only) for survival analysis
			if generation == 0:
				initial_elite_genomes = [
					(self.clone_genome(population[idx][0]), fitness_values[idx])
					for idx in all_elite_indices
				]
				# Log elite composition
				self._log(f"[{self.name}] Elitism: {len(ce_elite_indices)} by CE + "
						  f"{len(acc_elite_indices)} by Acc ({overlap_count} overlap) = {total_elites} unique elites")

			# Add elites to new population
			for i, elite_idx in enumerate(all_elite_indices):
				elite_genome = self.clone_genome(population[elite_idx][0])
				elite_fitness = fitness_values[elite_idx]
				elite_accuracy = accuracy_values[elite_idx]
				new_population.append((elite_genome, elite_fitness, elite_accuracy))

				# Log elite source
				in_ce = elite_idx in ce_elite_indices
				in_acc = elite_idx in acc_elite_indices
				source = "CE+Acc" if (in_ce and in_acc) else ("CE" if in_ce else "Acc")
				acc_str = f", Acc={elite_accuracy:.2%}" if elite_accuracy is not None else ""
				self._log(f"[Elite {i + 1}/{total_elites}] CE={elite_fitness:.4f}{acc_str} ({source})")

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
				new_population.append((child, None, None))  # Needs evaluation

			population = new_population[:cfg.population_size]
			population = self._evaluate_population(population, batch_evaluate_fn, evaluate_fn, generation, cfg.generations)

			# Filter candidates: replace those with accuracy below threshold
			replaced_count = 0
			max_replacements = 3  # Limit replacement attempts per generation
			for attempt in range(max_replacements):
				needs_replacement = []
				for i, (genome, fitness, accuracy) in enumerate(population):
					# Skip elites (first total_elites entries)
					if i < total_elites:
						continue
					# Check if below minimum accuracy threshold
					if accuracy is not None and accuracy < cfg.min_accuracy:
						needs_replacement.append(i)

				if not needs_replacement:
					break

				# Replace with new random genomes
				for idx in needs_replacement:
					new_genome = self.create_random_genome()
					population[idx] = (new_genome, None, None)
					replaced_count += 1

				# Re-evaluate only the replaced genomes
				population = self._evaluate_population(population, batch_evaluate_fn, evaluate_fn, generation, cfg.generations)

			if replaced_count > 0:
				self._log(f"[{self.name}] Replaced {replaced_count} candidates with accuracy < {cfg.min_accuracy:.2%}")

			fitness_values = [f for _, f, _ in population]
			accuracy_values = [a for _, _, a in population]

			# Update best
			gen_best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
			if fitness_values[gen_best_idx] < best_fitness:
				best = self.clone_genome(population[gen_best_idx][0])
				best_fitness = fitness_values[gen_best_idx]

			history.append((generation + 1, best_fitness))

			# Compute new_best (best among new candidates, excluding elites)
			new_candidate_fitness = fitness_values[total_elites:]  # Non-elite fitness values
			new_best = min(new_candidate_fitness) if new_candidate_fitness else float('inf')

			# Track elite wins: did any elite beat the best new candidate?
			elite_fitness = fitness_values[:total_elites]
			best_elite = min(elite_fitness) if elite_fitness else float('inf')
			if best_elite <= new_best:
				elite_wins += 1

			# Track improvement
			prev_best = history[-2][1] if len(history) >= 2 else history[-1][1]
			if best_fitness < prev_best:
				improved_iterations += 1

			# Log progress
			gen_avg = sum(fitness_values) / len(fitness_values)
			self._log(f"[{self.name}] Gen {generation + 1}/{cfg.generations}: "
					  f"best={best_fitness:.4f}, new_best={new_best:.4f}, avg={gen_avg:.4f}")

			# Early stopping check (checks at configured intervals)
			if early_stopper.check(generation, best_fitness):
				break

			# Overfitting callback check (same interval as early stopping)
			if overfitting_callback is not None and (generation + 1) % cfg.check_interval == 0:
				control = overfitting_callback(best, best_fitness)
				if hasattr(control, 'early_stop') and control.early_stop:
					self._log(f"[{self.name}] Overfitting early stop at gen {generation + 1}")
					# Return early with overfitting stop reason
					final_population = [self.clone_genome(g) for g, _, _ in sorted(population, key=lambda x: x[1])]
					improvement_pct = (initial_fitness - best_fitness) / initial_fitness * 100 if initial_fitness > 0 else 0
					return OptimizerResult(
						initial_genome=initial_genome if initial_genome else population[0][0],
						best_genome=best,
						initial_fitness=initial_fitness,
						final_fitness=best_fitness,
						improvement_percent=improvement_pct,
						iterations_run=generation + 1,
						method_name=self.name,
						history=history,
						early_stopped=True,
						stop_reason=StopReason.OVERFITTING,
						final_population=final_population,
						initial_accuracy=initial_accuracy,
						final_accuracy=accuracy_values[gen_best_idx] if accuracy_values else None,
					)

		# Get final accuracy from best genome's cached accuracy
		best_idx_final = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
		final_accuracy = accuracy_values[best_idx_final] if accuracy_values else None

		# Extract final population for seeding next phase (sorted by fitness)
		final_population = [self.clone_genome(g) for g, _, _ in sorted(population, key=lambda x: x[1])]

		# Compute final diversity
		final_ce_spread = max(fitness_values) - min(fitness_values) if fitness_values else 0.0

		# Count elite survivals (how many initial elites made it to final population)
		elite_survivals = 0
		if initial_elite_genomes:
			final_fitness_set = set(f for _, f, _ in population)
			for _, elite_fit in initial_elite_genomes:
				if elite_fit in final_fitness_set:
					elite_survivals += 1

		# Log analysis summary
		total_gens = generation + 1
		elite_win_rate = elite_wins / total_gens * 100 if total_gens > 0 else 0
		improvement_rate = improved_iterations / total_gens * 100 if total_gens > 0 else 0
		diversity_change = final_ce_spread - initial_ce_spread

		self._log(f"\n[{self.name}] Analysis Summary:")
		self._log(f"  CE improvement: {initial_fitness:.4f} → {best_fitness:.4f} ({(1 - best_fitness/initial_fitness)*100:+.2f}%)")
		self._log(f"  CE spread: {initial_ce_spread:.4f} → {final_ce_spread:.4f} ({diversity_change:+.4f})")
		self._log(f"  Elite survivals: {elite_survivals}/{len(initial_elite_genomes) if initial_elite_genomes else 0}")
		self._log(f"  Elite win rate: {elite_wins}/{total_gens} ({elite_win_rate:.1f}%)")
		self._log(f"  Improvement rate: {improved_iterations}/{total_gens} ({improvement_rate:.1f}%)")

		improvement_pct = (initial_fitness - best_fitness) / initial_fitness * 100 if initial_fitness > 0 else 0

		return OptimizerResult(
			initial_genome=initial_genome if initial_genome else population[0][0],
			best_genome=best,
			initial_fitness=initial_fitness,
			final_fitness=best_fitness,
			improvement_percent=improvement_pct,
			iterations_run=generation + 1,
			method_name=self.name,
			history=history,
			early_stopped=early_stopper.patience_exhausted,
			stop_reason=StopReason.CONVERGENCE if early_stopper.patience_exhausted else None,
			final_population=final_population,
			initial_accuracy=initial_accuracy,
			final_accuracy=final_accuracy,
		)

	def _evaluate_population(
		self,
		population: List[Tuple[T, Optional[float], Optional[float]]],
		batch_fn: Optional[Callable[[List[T]], List[Tuple[float, float]]]],
		single_fn: Callable[[T], float],
		generation: int = 0,
		total_generations: int = 0,
	) -> List[Tuple[T, float, Optional[float]]]:
		"""
		Evaluate individuals with None fitness, tracking accuracy.

		Args:
			population: List of (genome, fitness, accuracy) tuples
			batch_fn: Optional batch evaluation function returning List[(CE, accuracy)]
			single_fn: Single genome evaluation function (CE only, for fallback)
			generation: Current generation (0-indexed, for logging)
			total_generations: Total generations (for logging)

		Returns:
			Updated population with fitness and accuracy filled in
		"""
		unknown_indices = [i for i, (_, f, _) in enumerate(population) if f is None]

		if not unknown_indices:
			return [(g, f, a) for g, f, a in population]  # All cached

		to_eval = [population[i][0] for i in unknown_indices]

		# Batch evaluate - returns (CE, accuracy) tuples
		if batch_fn is not None:
			results = batch_fn(to_eval)
			new_fitness = [ce for ce, _ in results]
			new_accuracy = [acc for _, acc in results]
		else:
			# Fallback to single evaluation (no accuracy)
			new_fitness = [single_fn(g) for g in to_eval]
			new_accuracy = [None] * len(to_eval)

		# Note: Real-time per-genome logging happens in evaluate_batch (adaptive_cluster.py)
		# with timing info. We don't duplicate logging here.

		result = list(population)
		for idx, fit, acc in zip(unknown_indices, new_fitness, new_accuracy):
			result[idx] = (result[idx][0], fit, acc)

		return result

	def _tournament_select(self, population: List[Tuple[T, float, Optional[float]]], tournament_size: int = 3) -> T:
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
		batch_evaluate_fn: Optional[Callable[[List[T]], List[Tuple[float, float]]]] = None,
		overfitting_callback: Optional[Callable[[T, float], Any]] = None,
	) -> OptimizerResult[T]:
		"""
		Run Tabu Search optimization.

		Args:
			initial_genome: Starting genome
			initial_fitness: Fitness of initial genome
			evaluate_fn: Function to evaluate a single genome
			initial_neighbors: Optional seed neighbors from previous phase
			batch_evaluate_fn: Optional batch evaluation function returning List[(CE, accuracy)]
			overfitting_callback: Optional callback for overfitting detection.
				Called every check_interval iterations with (best_genome, best_fitness).
				Returns OverfittingControl (or any object with early_stop attribute).
				If early_stop=True, optimization stops with stop_reason=StopReason.OVERFITTING.

		Returns:
			OptimizerResult with best genome and statistics
		"""
		self._ensure_rng()
		cfg = self._config

		current = self.clone_genome(initial_genome)
		current_fitness = initial_fitness

		best = self.clone_genome(initial_genome)
		best_fitness = initial_fitness
		start_fitness = initial_fitness

		# Track accuracy (populated from batch evaluation results)
		best_accuracy = None  # Accuracy of best genome

		# Tabu list
		tabu_list: deque = deque(maxlen=cfg.tabu_size)

		# Track best neighbors for seeding next phase
		best_neighbors: List[Tuple[T, float]] = [(self.clone_genome(initial_genome), initial_fitness)]

		# Seed with initial neighbors if provided
		if initial_neighbors:
			self._log(f"[{self.name}] Seeding from {len(initial_neighbors)} neighbors")
			# Evaluate seed neighbors - batch_evaluate_fn returns (CE, accuracy) tuples
			if batch_evaluate_fn is not None:
				results = batch_evaluate_fn(initial_neighbors)
				seed_fitness = [ce for ce, _ in results]
				seed_accuracy = [acc for _, acc in results]
			else:
				seed_fitness = [evaluate_fn(g) for g in initial_neighbors]
				seed_accuracy = [None] * len(initial_neighbors)
			# Note: Real-time per-genome logging happens in evaluate_batch
			# Filter seed neighbors by minimum accuracy
			filtered_count = 0
			for g, f, a in zip(initial_neighbors, seed_fitness, seed_accuracy):
				if a is None or a >= cfg.min_accuracy:
					best_neighbors.append((self.clone_genome(g), f))
				else:
					filtered_count += 1
			if filtered_count > 0:
				self._log(f"[{self.name}] Filtered {filtered_count} seed neighbors with accuracy < {cfg.min_accuracy:.2%}")
			# Seed summary: best, seed_best, avg
			viable_fitness = [f for g, f, a in zip(initial_neighbors, seed_fitness, seed_accuracy) if a is None or a >= cfg.min_accuracy]
			seed_best = min(viable_fitness) if viable_fitness else min(seed_fitness)
			seed_avg = sum(viable_fitness) / len(viable_fitness) if viable_fitness else sum(seed_fitness) / len(seed_fitness)
			self._log(f"[{self.name}] Seed summary: best={best_fitness:.4f}, seed_best={seed_best:.4f}, avg={seed_avg:.4f}")
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

		# Track initial diversity (CE spread) from best_neighbors
		neighbor_fitness = [f for _, f in best_neighbors]
		initial_ce_spread = max(neighbor_fitness) - min(neighbor_fitness) if len(neighbor_fitness) > 1 else 0.0

		# Analysis tracking
		elite_wins = 0  # Iterations where current (seed/elite) beat all new neighbors
		improved_iterations = 0  # Iterations with global best improvement

		# Initialize early stopping tracker
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stopper = EarlyStoppingTracker(early_stop_config, self._log, self.name)
		early_stopper.reset(best_fitness)

		# Log config at startup
		self._log(f"[{self.name}] Config: neighbors={cfg.neighbors_per_iter}, iters={cfg.iterations}, "
				  f"patience={cfg.patience}, check_interval={cfg.check_interval}, min_delta={cfg.min_improvement_pct}%")
		self._log(f"[{self.name}] Initial: {start_fitness:.4f}, diversity (CE spread): {initial_ce_spread:.4f}")

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

			# Batch evaluate fitness - batch_evaluate_fn returns (CE, accuracy) tuples
			if batch_evaluate_fn is not None:
				to_eval = [n for n, _ in neighbor_candidates]
				results = batch_evaluate_fn(to_eval)
				fitness_values = [ce for ce, _ in results]
				accuracy_values = [acc for _, acc in results]
			else:
				fitness_values = [evaluate_fn(n) for n, _ in neighbor_candidates]
				accuracy_values = [None] * len(neighbor_candidates)

			# Build neighbors list with fitness and accuracy
			neighbors = [(n, f, m, a) for (n, m), f, a in zip(neighbor_candidates, fitness_values, accuracy_values)]

			# Filter out neighbors with accuracy below threshold
			viable_neighbors = [
				(n, f, m, a) for n, f, m, a in neighbors
				if a is None or a >= cfg.min_accuracy
			]
			filtered_count = len(neighbors) - len(viable_neighbors)
			if filtered_count > 0:
				self._log(f"[{self.name}] Filtered {filtered_count} neighbors with accuracy < {cfg.min_accuracy:.2%}")

			# If all neighbors filtered, keep the best one anyway (avoid getting stuck)
			if not viable_neighbors:
				self._log(f"[{self.name}] Warning: all neighbors filtered, keeping best anyway")
				viable_neighbors = sorted(neighbors, key=lambda x: x[1])[:1]

			neighbors = viable_neighbors

			# Note: Real-time per-genome logging happens in evaluate_batch (adaptive_cluster.py)

			# Select best neighbor (sort by fitness)
			neighbors.sort(key=lambda x: x[1])
			best_neighbor, best_neighbor_fitness, best_move, best_neighbor_acc = neighbors[0]

			# Move to best neighbor (always - TS characteristic)
			current = best_neighbor
			current_fitness = best_neighbor_fitness

			# Add move to tabu list
			if best_move is not None:
				tabu_list.append(best_move)

			# Track elite wins: current solution beat all new neighbors
			# (Note: we compare *before* the move, so check history)
			prev_best = history[-1][1] if history else start_fitness
			if best_neighbor_fitness >= prev_best:
				elite_wins += 1

			# Update global best (and accuracy)
			prev_global_best = best_fitness
			if current_fitness < best_fitness:
				best = self.clone_genome(current)
				best_fitness = current_fitness
				best_accuracy = best_neighbor_acc
				improved_iterations += 1

			# Track best neighbors for seeding
			for n, f, _, _ in neighbors[:5]:  # Keep top 5 from each iteration
				best_neighbors.append((self.clone_genome(n), f))
			best_neighbors.sort(key=lambda x: x[1])
			best_neighbors = best_neighbors[:cfg.neighbors_per_iter]

			history.append((iteration + 1, best_fitness))

			# Log progress: best (global), iter_best (this iteration), avg
			iter_best = best_neighbor_fitness  # Best among neighbors this iteration
			avg_fitness = sum(f for _, f, _, _ in neighbors) / len(neighbors)
			self._log(f"[{self.name}] Iter {iteration + 1}/{cfg.iterations}: "
					  f"best={best_fitness:.4f}, iter_best={iter_best:.4f}, avg={avg_fitness:.4f}")

			# Early stopping check (checks at configured intervals)
			if early_stopper.check(iteration, best_fitness):
				break

			# Overfitting callback check (same interval as early stopping)
			if overfitting_callback is not None and (iteration + 1) % cfg.check_interval == 0:
				control = overfitting_callback(best, best_fitness)
				if hasattr(control, 'early_stop') and control.early_stop:
					self._log(f"[{self.name}] Overfitting early stop at iter {iteration + 1}")
					# Return early with overfitting stop reason
					final_neighbors = [self.clone_genome(g) for g, _ in best_neighbors]
					improvement_pct = (start_fitness - best_fitness) / start_fitness * 100 if start_fitness > 0 else 0
					return OptimizerResult(
						initial_genome=initial_genome,
						best_genome=best,
						initial_fitness=start_fitness,
						final_fitness=best_fitness,
						improvement_percent=improvement_pct,
						iterations_run=iteration + 1,
						method_name=self.name,
						history=history,
						early_stopped=True,
						stop_reason=StopReason.OVERFITTING,
						final_population=final_neighbors,
						initial_accuracy=None,  # Not available without extra eval
						final_accuracy=best_accuracy,
					)

		# Extract final neighbors for seeding
		final_neighbors = [self.clone_genome(g) for g, _ in best_neighbors]

		# Compute final diversity
		final_neighbor_fitness = [f for _, f in best_neighbors]
		final_ce_spread = max(final_neighbor_fitness) - min(final_neighbor_fitness) if len(final_neighbor_fitness) > 1 else 0.0

		# Log analysis summary
		total_iters = iteration + 1
		elite_win_rate = elite_wins / total_iters * 100 if total_iters > 0 else 0
		improvement_rate = improved_iterations / total_iters * 100 if total_iters > 0 else 0
		diversity_change = final_ce_spread - initial_ce_spread

		self._log(f"\n[{self.name}] Analysis Summary:")
		self._log(f"  CE improvement: {start_fitness:.4f} → {best_fitness:.4f} ({(1 - best_fitness/start_fitness)*100:+.2f}%)")
		self._log(f"  CE spread: {initial_ce_spread:.4f} → {final_ce_spread:.4f} ({diversity_change:+.4f})")
		self._log(f"  Elite win rate: {elite_wins}/{total_iters} ({elite_win_rate:.1f}%)")
		self._log(f"  Improvement rate: {improved_iterations}/{total_iters} ({improvement_rate:.1f}%)")

		improvement_pct = (start_fitness - best_fitness) / start_fitness * 100 if start_fitness > 0 else 0

		return OptimizerResult(
			initial_genome=initial_genome,
			best_genome=best,
			initial_fitness=start_fitness,
			final_fitness=best_fitness,
			improvement_percent=improvement_pct,
			iterations_run=iteration + 1,
			method_name=self.name,
			history=history,
			early_stopped=early_stopper.patience_exhausted,
			stop_reason=StopReason.CONVERGENCE if early_stopper.patience_exhausted else None,
			final_population=final_neighbors,  # For seeding next phase
			initial_accuracy=None,  # Not available without extra eval
			final_accuracy=best_accuracy,
		)
