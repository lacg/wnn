"""
Base Evaluator — Abstract base class for all genome evaluators.

Provides shared behavior:
- EvalResult: Standardized evaluation result with backward-compatible tuple unpacking
- OffspringSearchResult: Named tuple for search_offspring results
- AdaptationConfig: Configuration for training-time architecture adaptation
- BaseEvaluator: ABC with data rotation, seed management, generation tracking,
  adaptation config, search methods, and convenience methods

Subclasses implement the actual evaluation logic:
- BitwiseEvaluator: Bitwise RAM with Rust+Metal pipeline
- TieredEvaluator: Tiered RAM with Rust hybrid CPU+GPU
- MultiStageEvaluator: Multi-stage orchestrator
"""

import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, NamedTuple

from wnn.ram.core.RAMClusterLayer import bits_needed
from wnn.ram.strategies.connectivity.adaptive_cluster import (
	ClusterGenome,
	AdaptiveClusterConfig,
	PhaseType,
)


@dataclass
class AdaptationConfig:
	"""Configuration for training-time architecture adaptation.

	Uses **relative** thresholds that scale with architecture size.
	See docs/ADAPTATION_MECHANISMS.md for design rationale.
	"""

	# Synaptogenesis (connection level)
	synaptogenesis_enabled: bool = False
	prune_entropy_ratio: float = 0.3       # Prune if entropy < median * ratio
	grow_fill_utilization: float = 0.5     # Grow if fill > expected_fill * factor
	grow_error_baseline: float = 0.35      # Grow if error > this (below random 0.5)
	min_bits: int = 4
	max_bits: int = 24

	# Neurogenesis (cluster level)
	neurogenesis_enabled: bool = False

	# Axonogenesis (connection rewiring)
	axonogenesis_enabled: bool = False
	axon_entropy_threshold: float = 0.3
	axon_improvement_factor: float = 1.5
	axon_rewire_count: int = 2
	cluster_error_factor: float = 0.7      # Add if error > 0.5 * factor
	cluster_fill_utilization: float = 0.5  # Add if mean_fill > expected * factor
	neuron_prune_percentile: float = 0.1   # Bottom 10% candidates for removal
	neuron_removal_factor: float = 0.5     # Only remove if score < mean * factor
	max_growth_ratio: float = 1.5          # Max neurons = initial * ratio
	min_neurons: int = 3
	max_neurons_per_pass: int = 3

	# Schedule (cosine annealing, warmup-excluded)
	warmup_generations: int = 10
	cooldown_iterations: int = 5
	stabilize_fraction: float = 0.25       # Last 25% of post-warmup: frozen
	total_generations: int = 250

	# Shared
	passes_per_eval: int = 1
	stats_sample_size: int = 10_000


@dataclass
class EvalResult:
	"""Standardized evaluation result for all evaluators.

	Supports backward-compatible tuple unpacking:
	- When bit_accuracy is None: unpacks as (ce, accuracy)
	- When bit_accuracy is set: unpacks as (ce, accuracy, bit_accuracy)

	Two-stage fields (cluster_ce, within_ce, etc.) are for future use.
	"""
	ce: float                              # Cross-entropy (lower = better)
	accuracy: float                        # Token accuracy (higher = better)
	f1_macro: Optional[float] = None       # F1-macro (higher = better)
	fpr: Optional[float] = None            # False positive rate macro (IDS, lower = better)
	bit_accuracy: Optional[float] = None   # Weighted bit accuracy (bitwise only)
	cluster_ce: Optional[float] = None     # Stage 1 CE (two-stage only)
	cluster_accuracy: Optional[float] = None  # Stage 1 accuracy (two-stage only)
	within_ce: Optional[float] = None      # Stage 2 CE (two-stage only)
	within_accuracy: Optional[float] = None  # Stage 2 accuracy (two-stage only)

	def __iter__(self):
		"""Yield (ce, accuracy[, bit_accuracy]) for backward-compat tuple unpacking.

		For f1_macro/fpr, use named attribute access (result.f1_macro, result.fpr).
		"""
		yield self.ce
		yield self.accuracy
		if self.bit_accuracy is not None:
			yield self.bit_accuracy

	def __getitem__(self, idx: int):
		"""Index-based access: result[0]=ce, result[1]=accuracy, result[2]=bit_accuracy."""
		if idx == 0:
			return self.ce
		elif idx == 1:
			return self.accuracy
		elif idx == 2:
			return self.bit_accuracy
		raise IndexError(f"EvalResult index {idx} out of range")

	def __len__(self) -> int:
		"""Length: 2 if bit_accuracy is None, 3 otherwise."""
		return 3 if self.bit_accuracy is not None else 2


class OffspringSearchResult(NamedTuple):
	"""Result of offspring search with counts for tracking.

	Attributes:
		genomes: List of ClusterGenome objects (viable + fallback if return_best_n)
		evaluated: Total candidates evaluated
		viable: Candidates that passed accuracy threshold (before fallback)
	"""
	genomes: list[ClusterGenome]
	evaluated: int
	viable: int


class BaseEvaluator(ABC):
	"""Abstract base for all genome evaluators.

	Shared behavior:
	- Data rotation (train/eval subset cycling)
	- Seed management
	- Generation tracking and logging
	- Adaptation config (synaptogenesis, neurogenesis, axonogenesis)
	- Convenience methods (evaluate_single, evaluate_single_with_accuracy)

	Subclasses implement evaluation logic via evaluate_batch() and evaluate_batch_full().
	"""

	def __init__(
		self,
		train_tokens: list[int],
		eval_tokens: list[int],
		vocab_size: int,
		context_size: int,
		num_parts: int = 3,
		num_eval_parts: int = 1,
		seed: Optional[int] = None,
		memory_mode: int = 0,
		neuron_sample_rate: float = 1.0,
		adapt_config: Optional[AdaptationConfig] = None,
		log_path: Optional[str] = None,
	):
		self._vocab_size = vocab_size
		self._context_size = context_size
		self._num_parts = num_parts
		self._num_eval_parts = num_eval_parts
		self._memory_mode = memory_mode
		self._neuron_sample_rate = neuron_sample_rate
		self._adapt_config = adapt_config
		self._generation = 0
		self._log_path = log_path
		self._live_progress: Optional[dict] = None
		self._current_phase: str = "evaluate_batch"
		self._experiment_id: Optional[int] = None

		if seed is None:
			seed = int(time.time() * 1000) % (2**32)
		self._seed = seed

		self._train_tokens = train_tokens
		self._eval_tokens = eval_tokens

		# Compute input dimensions
		self._bits_per_token = bits_needed(vocab_size)
		self._total_input_bits = context_size * self._bits_per_token

		# Per-generation metrics history for correlation tracking
		self._generation_log: list[tuple] = []

	# ── Live progress (Python-level, for observer thread) ─────────────────

	def get_live_progress(self) -> Optional[dict]:
		"""Return current live progress dict, or None if not in a search."""
		return self._live_progress

	def set_experiment_context(self, experiment_id: int) -> None:
		"""Set the experiment ID for live progress reporting."""
		self._experiment_id = experiment_id

	def _update_live_progress(
		self, phase: str, evaluated: int, target_count: int, viable: int,
		best_ce: float, best_acc: float, elapsed_secs: float,
		generation: Optional[int] = None, total_generations: Optional[int] = None,
	) -> None:
		"""Update the live progress dict (read by observer thread)."""
		self._live_progress = {
			"experiment_id": self._experiment_id or 0,
			"generation": generation or 0,
			"total_generations": total_generations or 0,
			"phase": phase,
			"evaluated": evaluated,
			"target_count": target_count,
			"viable": viable,
			"best_ce": best_ce,
			"best_acc": best_acc,
			"elapsed_secs": elapsed_secs,
		}

	def _clear_live_progress(self) -> None:
		"""Clear live progress after search completes."""
		self._live_progress = None

	# ── Rotation (abstract — each subclass delegates to its Rust cache) ───

	@abstractmethod
	def next_train_idx(self) -> int: ...

	@abstractmethod
	def next_eval_idx(self) -> int: ...

	def random_train_idx(self, rng: Optional[random.Random] = None) -> int:
		"""Pick a random train subset index (uniform, no state advancement)."""
		if rng is None:
			rng = random.Random()
		return rng.randint(0, self._num_parts - 1)

	# ── Rust progress logging ─────────────────────────────────────────────

	def _setup_progress_env(
		self,
		num_genomes: int,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		log_type: str = "Eval",
		offset: int = 0,
	) -> bool:
		"""Set WNN_PROGRESS_* env vars for Rust per-genome logging.

		Returns True if env vars were set (caller should call _cleanup_progress_env).
		Only activates when log_path is available and batch has >1 genome.
		"""
		if num_genomes <= 1 or not self._log_path:
			return False
		os.environ["WNN_LOG_PATH"] = self._log_path
		os.environ["WNN_PROGRESS_LOG"] = "1"
		os.environ["WNN_PROGRESS_GEN"] = str((generation + 1) if generation is not None else 0)
		os.environ["WNN_PROGRESS_TOTAL_GENS"] = str(total_generations or 1)
		os.environ["WNN_PROGRESS_TYPE"] = log_type
		os.environ["WNN_PROGRESS_TOTAL"] = str(num_genomes)
		os.environ["WNN_PROGRESS_OFFSET"] = str(offset)
		return True

	def _cleanup_progress_env(self) -> None:
		"""Remove WNN_PROGRESS_* env vars after Rust call."""
		os.environ.pop("WNN_PROGRESS_LOG", None)

	# ── Core evaluation (abstract) ────────────────────────────────────────

	@abstractmethod
	def evaluate_batch(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		**kwargs,
	) -> list[EvalResult]: ...

	@abstractmethod
	def evaluate_batch_full(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
	) -> list[EvalResult]: ...

	# ── Convenience (shared) ──────────────────────────────────────────────

	def evaluate_single(
		self,
		genome: ClusterGenome,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
	) -> float:
		"""Evaluate a single genome, returning CE only."""
		return self.evaluate_batch(
			[genome], train_subset_idx, eval_subset_idx,
		)[0].ce

	def evaluate_single_with_accuracy(
		self,
		genome: ClusterGenome,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
	) -> tuple[float, float]:
		"""Evaluate a single genome, returning (CE, accuracy)."""
		r = self.evaluate_batch(
			[genome], train_subset_idx, eval_subset_idx,
		)[0]
		return (r.ce, r.accuracy)

	def evaluate_single_full(self, genome: ClusterGenome) -> tuple[float, float]:
		"""Evaluate a single genome with full data, returning (CE, accuracy)."""
		r = self.evaluate_batch_full([genome])[0]
		return (r.ce, r.accuracy)

	# ── Generation tracking (shared) ──────────────────────────────────────

	def set_generation(self, gen: int, total_generations: int | None = None) -> None:
		"""Set current generation for adaptive evaluation warmup tracking."""
		self._generation = gen
		if total_generations is not None and self._adapt_config is not None:
			self._adapt_config.total_generations = total_generations

	# ── Properties (shared) ───────────────────────────────────────────────

	@property
	def vocab_size(self) -> int:
		return self._vocab_size

	@property
	def context_size(self) -> int:
		return self._context_size

	@property
	def total_input_bits(self) -> int:
		return self._total_input_bits

	@property
	def num_parts(self) -> int:
		return self._num_parts

	@property
	def num_eval_parts(self) -> int:
		return self._num_eval_parts

	@property
	def adapt_config(self) -> Optional[AdaptationConfig]:
		return self._adapt_config

	@property
	def generation_log(self) -> list[tuple]:
		"""Per-generation metrics from evaluate_batch calls."""
		return self._generation_log

	def clear_generation_log(self) -> None:
		"""Clear per-generation metrics (call between phases)."""
		self._generation_log.clear()

	# ── Search methods (shared, overridable for Rust fast paths) ─────────

	def _tournament_select(
		self,
		population: list[tuple[ClusterGenome, float]],
		tournament_size: int,
		rng: random.Random,
		fitness_scores: Optional[list[float]] = None,
	) -> ClusterGenome:
		"""Tournament selection using fitness scores when available.

		When fitness_scores is provided, selection uses those (aligning with
		the GA's fitness calculator — e.g. HarmonicRank).  Otherwise falls
		back to CE (population[i][1]).
		"""
		k = min(tournament_size, len(population))
		indices = rng.sample(range(len(population)), k)
		if fitness_scores is not None and len(fitness_scores) == len(population):
			best_idx = min(indices, key=lambda i: fitness_scores[i])
		else:
			best_idx = min(indices, key=lambda i: population[i][1])
		return population[best_idx][0].clone()

	def _assortative_select(
		self,
		population: list[tuple[ClusterGenome, float]],
		reference: ClusterGenome,
		tournament_size: int,
		rng: random.Random,
		fitness_scores: Optional[list[float]] = None,
	) -> ClusterGenome:
		"""Assortative tournament selection: pick most architecturally similar to reference.

		Runs an expanded tournament (2× size), picks candidate closest to reference
		in neuron count + mean bits (NEAT-style speciation).
		"""
		k = min(tournament_size * 2, len(population))
		indices = rng.sample(range(len(population)), k)

		ref_n = sum(reference.neurons_per_cluster)
		ref_b = sum(reference.bits_per_neuron) / max(1, len(reference.bits_per_neuron))

		def distance(idx: int) -> float:
			g = population[idx][0]
			n = sum(g.neurons_per_cluster)
			b = sum(g.bits_per_neuron) / max(1, len(g.bits_per_neuron))
			max_n = max(ref_n, n, 1)
			max_b = max(ref_b, b, 1.0)
			return abs(ref_n - n) / max_n + 0.5 * abs(ref_b - b) / max_b

		best_idx = min(indices, key=distance)
		return population[best_idx][0].clone()

	def _mutate_genome_phased(
		self,
		genome: ClusterGenome,
		phase_type: int,
		bits_mutation_rate: float,
		neurons_mutation_rate: float,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		rng: random.Random,
	) -> ClusterGenome:
		"""Mutate a genome using ClusterGenome.mutate() with phase-aware rate dispatch."""
		pt = PhaseType(phase_type)
		if pt == PhaseType.NEURONS:
			rate = neurons_mutation_rate
		elif pt == PhaseType.CLUSTER:
			rate = bits_mutation_rate  # CLUSTER mutation uses bits rate for all dimensions
		else:
			rate = bits_mutation_rate
		config = AdaptiveClusterConfig(
			min_bits=min_bits, max_bits=max_bits,
			min_neurons=min_neurons, max_neurons=max_neurons,
			max_bit_delta=getattr(self, '_max_bit_delta', 0),
		)
		return genome.mutate(pt, rate, config, self.total_input_bits, rng)

	def search_neighbors(
		self,
		genome: ClusterGenome,
		target_count: int,
		max_attempts: int,
		accuracy_threshold: float,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		bits_mutation_rate: float = 0.1,
		neurons_mutation_rate: float = 0.05,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		seed: Optional[int] = None,
		log_path: Optional[str] = None,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
		phase_type: int = 0,
	) -> list[ClusterGenome]:
		"""Search for neighbor genomes above accuracy threshold.

		Uses ClusterGenome.mutate() for phase-aware mutation, then
		self.evaluate_batch() for evaluation. Subclasses may override
		to delegate to Rust fast paths.
		"""
		self._current_phase = "ts_neighbors"
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if eval_subset_idx is None:
			eval_subset_idx = 0
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)

		rng = random.Random(seed)
		passed: list[ClusterGenome] = []
		all_candidates: list[ClusterGenome] = []
		evaluated = 0
		viable_count = 0
		batch_size = 50
		best_ce_so_far = float('inf')
		best_acc_so_far = 0.0
		start_time = time.time()

		# Track fingerprints to avoid duplicate neighbors
		known_fps: set = set()
		if hasattr(genome, 'fingerprint'):
			known_fps.add(genome.fingerprint())

		while len(passed) < target_count and evaluated < max_attempts:
			remaining = target_count - len(passed)
			batch_n = min(remaining + 5, batch_size, max_attempts - evaluated)
			if batch_n <= 0:
				break

			batch = []
			for _ in range(batch_n):
				child = self._mutate_genome_phased(
					genome, phase_type,
					bits_mutation_rate, neurons_mutation_rate,
					min_bits, max_bits, min_neurons, max_neurons, rng,
				)
				# Re-mutate if duplicate (up to 3 retries)
				if hasattr(child, 'fingerprint'):
					for _ in range(3):
						if child.fingerprint() not in known_fps:
							break
						child = self._mutate_genome_phased(
							genome, phase_type,
							bits_mutation_rate, neurons_mutation_rate,
							min_bits, max_bits, min_neurons, max_neurons, rng,
						)
					known_fps.add(child.fingerprint())
				batch.append(child)

			results = self.evaluate_batch(
				batch, train_subset_idx, eval_subset_idx,
				logger=logger, generation=generation,
				total_generations=total_generations,
			)
			evaluated += len(results)

			for r in results:
				best_ce_so_far = min(best_ce_so_far, r.ce)
				best_acc_so_far = max(best_acc_so_far, r.accuracy)
			self._update_live_progress(
				"ts_neighbors", evaluated, target_count, viable_count,
				best_ce_so_far, best_acc_so_far, time.time() - start_time,
				generation, total_generations,
			)

			for g, r in zip(batch, results):
				g._cached_fitness = (r.ce, r.accuracy)
				if r.bit_accuracy is not None:
					g._cached_bit_acc = r.bit_accuracy
				if r.accuracy >= accuracy_threshold:
					viable_count += 1
					passed.append(g)
					if len(passed) >= target_count:
						break
				else:
					all_candidates.append(g)

		self._clear_live_progress()
		if len(passed) < target_count and return_best_n:
			all_candidates.sort(key=lambda g: (-g._cached_fitness[1], g._cached_fitness[0]))
			need = target_count - len(passed)
			passed.extend(all_candidates[:need])

		return passed

	def search_offspring(
		self,
		population: list[tuple[ClusterGenome, float]],
		target_count: int,
		max_attempts: int,
		accuracy_threshold: float,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		bits_mutation_rate: float = 0.1,
		neurons_mutation_rate: float = 0.1,
		crossover_rate: float = 0.7,
		tournament_size: int = 3,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		seed: Optional[int] = None,
		log_path: Optional[str] = None,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
		phase_type: int = 0,
		fitness_scores: Optional[list[float]] = None,
		cluster_crossover_ratio: float = 0.0,
		pool_shuffle_ratio: float = 0.0,
		assortative_mating_ratio: float = 0.0,
	) -> OffspringSearchResult:
		"""Search for GA offspring above accuracy threshold.

		Uses ClusterGenome.crossover() + ClusterGenome.mutate() for
		phase-aware genetic operations, then self.evaluate_batch().
		Subclasses may override to delegate to Rust fast paths.

		When fitness_scores is provided, tournament selection uses those
		instead of raw CE — aligning parent selection with elite selection.

		cluster_crossover_ratio: fraction of crossovers using CLUSTER mode (0.0-1.0).
		pool_shuffle_ratio: fraction using old pool-and-shuffle crossover (0.0-1.0).
		"""
		if not population:
			return OffspringSearchResult(genomes=[], evaluated=0, viable=0)

		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if eval_subset_idx is None:
			eval_subset_idx = 0
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)

		self._current_phase = "ga_offspring"
		pt = PhaseType(phase_type)
		rng = random.Random(seed)
		passed: list[ClusterGenome] = []
		all_candidates: list[ClusterGenome] = []
		evaluated = 0
		viable_count = 0
		batch_size = 50
		best_ce_so_far = float('inf')
		best_acc_so_far = 0.0
		start_time = time.time()

		# Build fingerprint set from current population to avoid duplicates
		known_fps: set = set()
		for g, _ in population:
			if hasattr(g, 'fingerprint'):
				known_fps.add(g.fingerprint())

		while len(passed) < target_count and evaluated < max_attempts:
			remaining = target_count - len(passed)
			batch_n = min(remaining + 5, batch_size, max_attempts - evaluated)
			if batch_n <= 0:
				break

			batch = []
			i = 0
			while i < batch_n:
				parent1 = self._tournament_select(population, tournament_size, rng, fitness_scores)
				if rng.random() < crossover_rate and len(population) > 1:
					# Assortative mating: pick p2 similar to p1 (NEAT-style)
					if assortative_mating_ratio > 0.0 and rng.random() < assortative_mating_ratio:
						parent2 = self._assortative_select(population, parent1, tournament_size, rng, fitness_scores)
					else:
						parent2 = self._tournament_select(population, tournament_size, rng, fitness_scores)
					xo_pt = PhaseType.CLUSTER if rng.random() < cluster_crossover_ratio else pt
					use_pool_shuffle = pool_shuffle_ratio > 0.0 and rng.random() < pool_shuffle_ratio
					if use_pool_shuffle:
						# 2→1: pool-shuffle, keep only child1
						child1, _child2 = parent1.crossover_pool_shuffle2(parent2, xo_pt, rng)
						children = [child1]
					else:
						# 2→2: uniform crossover, both children
						child1, child2 = parent1.crossover2(parent2, xo_pt, rng)
						children = [child1, child2]
					for child in children:
						child = self._mutate_genome_phased(
							child, phase_type,
							bits_mutation_rate, neurons_mutation_rate,
							min_bits, max_bits, min_neurons, max_neurons, rng,
						)
						if hasattr(child, 'fingerprint'):
							for _ in range(3):
								if child.fingerprint() not in known_fps:
									break
								child = self._mutate_genome_phased(
									child, phase_type,
									bits_mutation_rate, neurons_mutation_rate,
									min_bits, max_bits, min_neurons, max_neurons, rng,
								)
							known_fps.add(child.fingerprint())
						batch.append(child)
					i += len(children)
				else:
					child = parent1.clone()
					child = self._mutate_genome_phased(
						child, phase_type,
						bits_mutation_rate, neurons_mutation_rate,
						min_bits, max_bits, min_neurons, max_neurons, rng,
					)
					if hasattr(child, 'fingerprint'):
						for _ in range(3):
							if child.fingerprint() not in known_fps:
								break
							child = self._mutate_genome_phased(
								child, phase_type,
								bits_mutation_rate, neurons_mutation_rate,
								min_bits, max_bits, min_neurons, max_neurons, rng,
							)
						known_fps.add(child.fingerprint())
					batch.append(child)
					i += 1
			batch = batch[:batch_n]

			results = self.evaluate_batch(
				batch, train_subset_idx, eval_subset_idx,
				logger=logger, generation=generation,
				total_generations=total_generations,
			)
			evaluated += len(results)

			# Update live progress
			for r in results:
				best_ce_so_far = min(best_ce_so_far, r.ce)
				best_acc_so_far = max(best_acc_so_far, r.accuracy)
			self._update_live_progress(
				"ga_offspring", evaluated, target_count, viable_count,
				best_ce_so_far, best_acc_so_far, time.time() - start_time,
				generation, total_generations,
			)

			for g, r in zip(batch, results):
				g._cached_fitness = (r.ce, r.accuracy)
				if r.bit_accuracy is not None:
					g._cached_bit_acc = r.bit_accuracy
				if r.accuracy >= accuracy_threshold:
					viable_count += 1
					passed.append(g)
					if len(passed) >= target_count:
						break
				else:
					all_candidates.append(g)

		self._clear_live_progress()
		if len(passed) < target_count and return_best_n:
			all_candidates.sort(key=lambda g: (-g._cached_fitness[1], g._cached_fitness[0]))
			need = target_count - len(passed)
			passed.extend(all_candidates[:need])

		return OffspringSearchResult(genomes=passed, evaluated=evaluated, viable=viable_count)

	def search_neighbors_batch(
		self,
		sources: list[tuple[ClusterGenome, int]],
		max_attempts_multiplier: int,
		accuracy_threshold: float,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		bits_mutation_rate: float = 0.1,
		neurons_mutation_rate: float = 0.05,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		seed: Optional[int] = None,
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
		phase_type: int = 0,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
	) -> list[list[ClusterGenome]]:
		"""Search neighbors for multiple source genomes.

		Default: per-source loop calling search_neighbors().
		MultiStageEvaluator overrides for single-batch optimization.
		"""
		results = []
		for genome, target_count in sources:
			max_attempts = target_count * max_attempts_multiplier
			neighbors = self.search_neighbors(
				genome, target_count, max_attempts,
				accuracy_threshold,
				min_bits, max_bits, min_neurons, max_neurons,
				bits_mutation_rate, neurons_mutation_rate,
				train_subset_idx, eval_subset_idx,
				seed, return_best_n=return_best_n,
				mutable_clusters=mutable_clusters,
				phase_type=phase_type,
				logger=logger, generation=generation,
				total_generations=total_generations,
			)
			results.append(neighbors)
		return results
