"""
Architecture optimization strategies using generic GA/TS base classes.

These implement ClusterGenome-specific operations while reusing the core
GA/TS algorithms from generic_strategies.py.

Strategies:
- ArchitectureGAStrategy: GA-based architecture optimization
- ArchitectureTSStrategy: Tabu Search architecture optimization
- GridSearchStrategy: One-shot evaluation of neuron × bit configurations

Features:
- Rust/Metal batch evaluation support for parallel genome evaluation
- Population seeding between phases (GA → TS → GA → ...)
- Checkpoint/resume support for long optimization runs
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from wnn.ram.strategies.connectivity.generic_strategies import (
	GenericGAStrategy,
	GenericTSStrategy,
	GAConfig,
	TSConfig,
	OptimizerResult,
	StopReason,
	EarlyStoppingConfig,
	EarlyStoppingTracker,
)
from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.architecture.genome_log import (
	GenomeLogType,
	format_genome_log,
	format_gen_prefix,
)
from wnn.ram.strategies.filters import PercentileFilter, FilterMode

# Optional tracker integration for genome tracking
try:
	from wnn.ram.experiments.tracker import TierConfig, GenomeConfig, GenomeRole
	HAS_GENOME_TRACKING = True
except ImportError:
	HAS_GENOME_TRACKING = False
	TierConfig = None
	GenomeConfig = None
	GenomeRole = None

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		RustParallelEvaluator,
		AdaptiveClusterConfig,
	)


# =============================================================================
# Shared Mixin for Architecture Strategies
# =============================================================================

class ArchitectureStrategyMixin:
	"""
	Mixin providing common functionality for GA and TS architecture strategies.

	Reduces code duplication by extracting:
	- Metal cleanup logic
	- Shutdown checking
	- genome_to_config conversion
	- Result building with stop_reason
	"""

	_shutdown_check: Optional[Callable[[], bool]]
	_log: Any  # Logger

	def _cleanup_metal(self, iteration: int, log_interval: int = 10) -> None:
		"""Run GC and reset Metal evaluators to prevent buffer accumulation."""
		import gc
		gc.collect()
		try:
			import ram_accelerator
			ram_accelerator.reset_metal_evaluators()
			if iteration % log_interval == 0:
				import resource
				# macOS: ru_maxrss is in bytes
				rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
				self._log.info(f"[{self.name}] GC + Metal reset at iteration {iteration}, RSS: {rss_mb:.0f} MB")
		except Exception:
			pass  # Ignore if accelerator not available

	def _check_and_set_shutdown(self, shutdown_flag: list[bool]) -> bool:
		"""
		Check if shutdown is requested and update the shutdown flag.

		Args:
			shutdown_flag: List with single bool element (mutable reference)

		Returns:
			True if shutdown was requested
		"""
		if self._shutdown_check and self._shutdown_check():
			shutdown_flag[0] = True
			return True
		return False

	def _genome_to_config_impl(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""
		Convert a ClusterGenome to a GenomeConfig for tracking.

		Finds contiguous runs of clusters with the same (neurons, mean_bits) config.
		This enables proper tracking of which cluster indices belong to which tier.

		The tier index is assigned based on the order of first appearance
		(earliest cluster index = tier 0), preserving the original tier config order.

		Per-neuron bits are averaged per cluster to produce a representative bits value.
		"""
		if not HAS_GENOME_TRACKING or GenomeConfig is None or TierConfig is None:
			return None

		# Find contiguous runs of clusters with same (neurons, mean_bits)
		runs: list[tuple[int, int, int, int]] = []  # (start, end, neurons, mean_bits)
		if len(genome.neurons_per_cluster) == 0:
			return GenomeConfig(tiers=[])

		neuron_offsets = genome.cluster_neuron_offsets

		def _cluster_mean_bits(c: int) -> int:
			"""Compute rounded mean bits for cluster c."""
			start_n = neuron_offsets[c]
			end_n = neuron_offsets[c + 1]
			if end_n == start_n:
				return 0
			return round(sum(genome.bits_per_neuron[start_n:end_n]) / (end_n - start_n))

		current_neurons = genome.neurons_per_cluster[0]
		current_bits = _cluster_mean_bits(0)
		run_start = 0

		for i in range(1, len(genome.neurons_per_cluster)):
			neurons = genome.neurons_per_cluster[i]
			bits = _cluster_mean_bits(i)
			if neurons != current_neurons or bits != current_bits:
				# End current run, start new one
				runs.append((run_start, i, current_neurons, current_bits))
				current_neurons = neurons
				current_bits = bits
				run_start = i

		# Don't forget the last run
		runs.append((run_start, len(genome.neurons_per_cluster), current_neurons, current_bits))

		# Assign tier indices based on first appearance of each (neurons, bits) config
		config_to_tier: dict[tuple[int, int], int] = {}
		next_tier = 0
		for _, _, neurons, bits in runs:
			key = (neurons, bits)
			if key not in config_to_tier:
				config_to_tier[key] = next_tier
				next_tier += 1

		# Create TierConfig for each contiguous run
		tiers = []
		for start, end, neurons, bits in runs:
			tier_idx = config_to_tier[(neurons, bits)]
			tiers.append(TierConfig(
				tier=tier_idx,
				clusters=end - start,
				neurons=neurons,
				bits=bits,
				start_cluster=start,
				end_cluster=end,
			))

		return GenomeConfig(tiers=tiers)

	def _determine_stop_reason(
		self,
		shutdown_requested: bool,
		early_stopper: Any,
	) -> Optional[StopReason]:
		"""Determine the stop reason based on shutdown flag and early stopper state."""
		if shutdown_requested:
			return StopReason.SHUTDOWN
		elif hasattr(early_stopper, 'patience_exhausted') and early_stopper.patience_exhausted:
			return StopReason.CONVERGENCE
		return None

	def _run_validation_summary(self, result: OptimizerResult) -> OptimizerResult:
		"""Run full-data validation on top genomes and update result.

		Shared between GA and TS architecture strategies.
		Evaluates top 20% of final population on full validation data,
		logs summary, records phase results, and returns updated result
		with best-CE genome.
		"""
		evaluator = self._cached_evaluator
		if evaluator is None or not result.final_population:
			return result

		self._log.info("")
		self._log.info("=" * 60)
		self._log.info(f"[{self.name}] VALIDATION SUMMARY (Full Dataset)")
		self._log.info("=" * 60)

		# Get top 20% for full evaluation
		top_k = max(1, int(len(result.final_population) * 0.2))
		top_genomes = result.final_population[:top_k]

		# Evaluate on full validation data
		full_results = evaluator.evaluate_batch_full(top_genomes)

		# Use fitness calculator to extract three independent bests
		val_population = [
			(genome, r[0], r[1])
			for genome, r in zip(top_genomes, full_results)
		]
		bests = self._fitness_calculator.bests(val_population)

		self._log.info(f"  Best by CE:       CE={bests.best_ce.ce:.4f}, Acc={bests.best_ce.accuracy:.4%}")
		self._log.info(f"  Best by Accuracy: CE={bests.best_acc.ce:.4f}, Acc={bests.best_acc.accuracy:.4%}")

		if bests.best_fitness.genome is bests.best_ce.genome:
			self._log.info(f"  Best by Fitness:  (same as Best by CE)")
		elif bests.best_fitness.genome is bests.best_acc.genome:
			self._log.info(f"  Best by Fitness:  (same as Best by Accuracy)")
		else:
			self._log.info(f"  Best by Fitness:  CE={bests.best_fitness.ce:.4f}, Acc={bests.best_fitness.accuracy:.4%}")

		top_k_ce = sum(r[0] for r in full_results) / len(full_results)
		top_k_acc = sum(r[1] for r in full_results) / len(full_results)
		self._log.info(f"  Top-{top_k} Mean:    CE={top_k_ce:.4f}, Acc={top_k_acc:.4%}")
		self._log.info("=" * 60)

		# Record phase results via tracker
		if self._tracker and self._tracker_experiment_id:
			try:
				self._tracker.record_phase_result(
					experiment_id=self._tracker_experiment_id,
					metric_type="best_ce",
					ce=bests.best_ce.ce,
					accuracy=bests.best_ce.accuracy,
					improvement_pct=(result.initial_fitness - bests.best_ce.ce) / result.initial_fitness * 100 if result.initial_fitness else 0.0,
				)
				self._tracker.record_phase_result(
					experiment_id=self._tracker_experiment_id,
					metric_type="best_acc",
					ce=bests.best_acc.ce,
					accuracy=bests.best_acc.accuracy,
					improvement_pct=(result.initial_fitness - bests.best_acc.ce) / result.initial_fitness * 100 if result.initial_fitness else 0.0,
				)
				self._tracker.record_phase_result(
					experiment_id=self._tracker_experiment_id,
					metric_type="top_k_mean",
					ce=top_k_ce,
					accuracy=top_k_acc,
					improvement_pct=(result.initial_fitness - top_k_ce) / result.initial_fitness * 100 if result.initial_fitness else 0.0,
				)
			except Exception as e:
				self._log.debug(f"[{self.name}] Failed to record phase results: {e}")

		# Update result with validation bests
		improvement_pct = (result.initial_fitness - bests.best_ce.ce) / result.initial_fitness * 100 if result.initial_fitness != 0 else 0.0
		return OptimizerResult(
			initial_genome=result.initial_genome,
			best_genome=bests.best_ce.genome,
			initial_fitness=result.initial_fitness,
			final_fitness=bests.best_ce.ce,
			improvement_percent=improvement_pct,
			iterations_run=result.iterations_run,
			method_name=result.method_name,
			history=result.history,
			early_stopped=result.early_stopped,
			stop_reason=result.stop_reason,
			final_population=result.final_population,
			initial_accuracy=result.initial_accuracy,
			final_accuracy=bests.best_acc.accuracy,
			final_threshold=result.final_threshold,
		)

	def _apply_percentile_filter(
		self,
		offspring: list[tuple['ClusterGenome', float, Optional[float]]],
	) -> list[tuple['ClusterGenome', float, Optional[float]]]:
		"""Apply fitness percentile filter to offspring/neighbors (3-tuple format).

		Shared between GA (offspring) and TS (neighbors).
		"""
		cfg = self._config
		fitness_calculator = self._fitness_calculator

		if cfg.fitness_calculator_type == FitnessCalculatorType.CE:
			# CE mode: filter by CE only
			ce_filter = PercentileFilter(
				percentile=cfg.fitness_percentile,
				mode=FilterMode.LOWER_IS_BETTER,
				metric_name="CE",
			)
			offspring_2t = [(g, ce) for g, ce, _ in offspring]
			filter_result = ce_filter.apply(offspring_2t, key=lambda g, f: f)
			kept_ids = {id(g) for g, _ in filter_result.kept}
			offspring = [t for t in offspring if id(t[0]) in kept_ids]
		else:
			# HARMONIC_RANK or NORMALIZED: filter by fitness score
			fitness_scores = fitness_calculator.fitness(offspring)
			offspring_with_fitness = list(zip(offspring, fitness_scores))

			fitness_filter = PercentileFilter(
				percentile=cfg.fitness_percentile,
				mode=FilterMode.LOWER_IS_BETTER,
				metric_name=fitness_calculator.name,
			)
			filter_input = [((g, ce, acc), score) for (g, ce, acc), score in offspring_with_fitness]
			filter_result = fitness_filter.apply(filter_input, key=lambda t, f: f)
			offspring = [t for t, _ in filter_result.kept]

		if filter_result.filtered:
			self._log.debug(
				f"[{self.name}] {filter_result.metric_name} filter: kept {filter_result.kept_count}/{filter_result.total_count} "
				f"(threshold={filter_result.threshold_value:.4f})"
			)

		return offspring


# =============================================================================
# Checkpoint System for Resume Support
# =============================================================================

@dataclass
class CheckpointConfig:
	"""Configuration for checkpoint saving."""
	enabled: bool = True
	interval: int = 50                       # Save every N iterations
	checkpoint_dir: Optional[Path] = None    # Directory for checkpoint files
	filename_prefix: str = "checkpoint"      # Prefix for checkpoint filenames


class CheckpointManager:
	"""
	Reusable checkpoint manager for optimization runs.

	Usage:
		# Create manager
		manager = CheckpointManager(
			config=CheckpointConfig(checkpoint_dir=Path("checkpoints")),
			phase_name="Phase 1a: GA Neurons",
			optimizer_type="GA",
			total_iterations=1000,
			logger=print,
		)

		# In optimization loop:
		for iteration in range(1000):
			# ... do optimization ...

			# Save checkpoint every N iterations
			manager.maybe_save(
				iteration=iteration,
				population=population,
				best_genome=best_genome,
				best_fitness=(ce, acc),
				current_threshold=threshold,
				extra_state={"patience": patience_counter},
			)

		# To resume:
		if manager.has_checkpoint():
			state = manager.load()
			start_iteration = state['current_iteration'] + 1
			population = state['population']
	"""

	def __init__(
		self,
		config: CheckpointConfig,
		phase_name: str,
		optimizer_type: str,
		total_iterations: int,
		logger: Optional[Callable[[str], None]] = None,
	):
		self._config = config
		self._phase_name = phase_name
		self._optimizer_type = optimizer_type
		self._total_iterations = total_iterations
		self._logger = logger or (lambda x: None)

		# Create checkpoint directory if needed
		if config.enabled and config.checkpoint_dir:
			config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

	@property
	def checkpoint_path(self) -> Optional[Path]:
		"""Path to the checkpoint file."""
		if not self._config.enabled or not self._config.checkpoint_dir:
			return None
		return self._config.checkpoint_dir / f"{self._config.filename_prefix}_{self._optimizer_type.lower()}.json"

	def has_checkpoint(self) -> bool:
		"""Check if a checkpoint file exists."""
		path = self.checkpoint_path
		return path is not None and path.exists()

	def should_save(self, iteration: int) -> bool:
		"""Check if we should save at this iteration."""
		if not self._config.enabled:
			return False
		# Save at interval (1-indexed), but also at iteration 0 for safety
		return iteration > 0 and (iteration + 1) % self._config.interval == 0

	def maybe_save(
		self,
		iteration: int,
		population: list[tuple['ClusterGenome', float]],
		best_genome: 'ClusterGenome',
		best_fitness: tuple[float, float],
		current_threshold: float,
		config_dict: Optional[dict] = None,
		extra_state: Optional[dict] = None,
	) -> bool:
		"""
		Save checkpoint if at the right interval.

		Args:
			iteration: Current iteration (0-indexed)
			population: List of (genome, ce_fitness) tuples
			best_genome: Best genome found so far
			best_fitness: (CE, accuracy) of best genome
			current_threshold: Current threshold value
			config_dict: Optional config as dict
			extra_state: Optional extra state to save (patience, baseline, etc.)

		Returns:
			True if checkpoint was saved, False otherwise
		"""
		if not self.should_save(iteration):
			return False

		self.save(
			iteration=iteration,
			population=population,
			best_genome=best_genome,
			best_fitness=best_fitness,
			current_threshold=current_threshold,
			config_dict=config_dict,
			extra_state=extra_state,
		)
		return True

	def save(
		self,
		iteration: int,
		population: list[tuple['ClusterGenome', float]],
		best_genome: 'ClusterGenome',
		best_fitness: tuple[float, float],
		current_threshold: float,
		config_dict: Optional[dict] = None,
		extra_state: Optional[dict] = None,
	) -> None:
		"""Save checkpoint now (regardless of interval)."""
		import datetime

		path = self.checkpoint_path
		if path is None:
			return

		# Serialize population
		pop_data = []
		for genome, ce in population:
			gd = self._genome_to_dict(genome)
			# Try to get accuracy from cached fitness
			if hasattr(genome, '_cached_fitness') and genome._cached_fitness:
				gd['fitness'] = list(genome._cached_fitness)
			else:
				gd['fitness'] = [ce, 0.0]
			pop_data.append(gd)

		# Build checkpoint data
		data = {
			'phase_name': self._phase_name,
			'optimizer_type': self._optimizer_type,
			'current_iteration': iteration,
			'total_iterations': self._total_iterations,
			'population': pop_data,
			'best_genome': self._genome_to_dict(best_genome),
			'best_fitness': list(best_fitness),
			'current_threshold': current_threshold,
			'config': config_dict or {},
			'extra_state': extra_state or {},
			'saved_at': datetime.datetime.now().isoformat(),
		}

		# Write atomically (temp file + rename)
		temp_path = path.with_suffix('.tmp')
		with open(temp_path, 'w') as f:
			json.dump(data, f, indent=2)
		temp_path.rename(path)

		self._logger(f"[Checkpoint] Saved at iteration {iteration + 1}/{self._total_iterations}")

	def load(self, genome_class: type) -> dict:
		"""
		Load checkpoint from file.

		Args:
			genome_class: The ClusterGenome class to use for reconstruction

		Returns:
			Dict with:
				- current_iteration: int
				- population: list of (genome, ce) tuples
				- best_genome: ClusterGenome
				- best_fitness: (CE, accuracy)
				- current_threshold: float
				- config: dict
				- extra_state: dict
		"""
		path = self.checkpoint_path
		if path is None or not path.exists():
			raise FileNotFoundError(f"No checkpoint found at {path}")

		with open(path, 'r') as f:
			data = json.load(f)

		# Reconstruct population
		population = []
		for gd in data['population']:
			genome = self._dict_to_genome(gd, genome_class)
			ce = gd['fitness'][0] if gd.get('fitness') else 0.0
			# Restore cached fitness if available
			if gd.get('fitness'):
				genome._cached_fitness = tuple(gd['fitness'])
			population.append((genome, ce))

		# Reconstruct best genome
		best_genome = self._dict_to_genome(data['best_genome'], genome_class)

		self._logger(f"[Checkpoint] Loaded from iteration {data['current_iteration'] + 1}")

		return {
			'current_iteration': data['current_iteration'],
			'population': population,
			'best_genome': best_genome,
			'best_fitness': tuple(data['best_fitness']),
			'current_threshold': data['current_threshold'],
			'config': data.get('config', {}),
			'extra_state': data.get('extra_state', {}),
			'saved_at': data.get('saved_at', ''),
		}

	@staticmethod
	def _genome_to_dict(genome: 'ClusterGenome') -> dict:
		"""Convert a ClusterGenome to a serializable dict."""
		return {
			'bits_per_neuron': list(genome.bits_per_neuron),
			'neurons_per_cluster': list(genome.neurons_per_cluster),
			'connections': list(genome.connections) if genome.connections else None,
		}

	@staticmethod
	def _dict_to_genome(d: dict, genome_class: type) -> 'ClusterGenome':
		"""Convert a dict back to a ClusterGenome.

		Supports both new format (bits_per_neuron) and legacy format (bits_per_cluster).
		"""
		if 'bits_per_neuron' in d:
			return genome_class(
				bits_per_neuron=d['bits_per_neuron'],
				neurons_per_cluster=d['neurons_per_cluster'],
				connections=d.get('connections'),
			)
		else:
			# Legacy format: expand bits_per_cluster to bits_per_neuron
			bits_per_cluster = d['bits_per_cluster']
			neurons_per_cluster = d['neurons_per_cluster']
			bits_per_neuron = []
			for bits, neurons in zip(bits_per_cluster, neurons_per_cluster):
				bits_per_neuron.extend([bits] * neurons)
			return genome_class(
				bits_per_neuron=bits_per_neuron,
				neurons_per_cluster=neurons_per_cluster,
				connections=d.get('connections'),
			)


@dataclass
class ArchitectureConfig:
	"""
	Configuration for architecture optimization.

	Controls both the search space bounds and what gets optimized.
	The optimizer is phase-agnostic - callers control what to optimize
	by setting the optimize_* flags.

	Example usage:
		# Phase 1: Optimize neurons only (bits fixed at default_bits)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=False,
			optimize_neurons=True,
			default_bits=8,  # All genomes start with 8 bits
		)

		# Phase 2: Optimize bits only (pass seed genome from Phase 1)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=True,
			optimize_neurons=False,
		)

		# Phase 3: Optimize connections only (pass seed genome from Phase 2)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=False,
			optimize_neurons=False,
			optimize_connections=True,
		)
	"""
	num_clusters: int
	min_bits: int = 4
	max_bits: int = 24
	min_neurons: int = 3
	max_neurons: int = 30
	# Explicit control over what gets optimized (no magic phase numbers)
	optimize_bits: bool = True
	optimize_neurons: bool = True
	optimize_connections: bool = False
	# Default values for dimensions not being optimized (used in random genome init)
	default_bits: int = 8
	default_neurons: int = 5
	# Token frequencies for frequency-scaled initialization
	token_frequencies: Optional[list[int]] = None
	# Total input bits for connection initialization/mutation
	total_input_bits: Optional[int] = None
	# Per-tier optimization: list of cluster indices that can be mutated (None = all clusters mutable)
	mutable_clusters: Optional[list[int]] = None


class ArchitectureGAStrategy(ArchitectureStrategyMixin, GenericGAStrategy['ClusterGenome']):
	"""
	Genetic Algorithm for architecture (bits, neurons per cluster) optimization.

	Inherits core GA loop from GenericGAStrategy, implements ClusterGenome operations.
	Uses ArchitectureStrategyMixin for shared functionality (Metal cleanup, shutdown, etc.)

	Features:
	- Rust/Metal batch evaluation (default when available)
	- Rust-based offspring search with threshold (when cached_evaluator provided)
	- Population seeding from previous phases
	- Checkpoint/resume support for long runs
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ga_config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
		cached_evaluator: Optional[Any] = None,  # CachedEvaluator for Rust search_offspring
		checkpoint_config: Optional[CheckpointConfig] = None,  # Checkpoint configuration
		phase_name: str = "GA Optimization",  # Phase name for checkpoints
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
	):
		super().__init__(config=ga_config, seed=seed, logger=logger)
		self._arch_config = arch_config
		self._batch_evaluator = batch_evaluator
		# Use cached_evaluator if provided, or check if batch_evaluator has search_offspring
		if cached_evaluator is not None:
			self._cached_evaluator = cached_evaluator
		elif batch_evaluator is not None and hasattr(batch_evaluator, 'search_offspring'):
			self._cached_evaluator = batch_evaluator
		else:
			self._cached_evaluator = None
		self._checkpoint_config = checkpoint_config
		self._phase_name = phase_name
		self._shutdown_check = shutdown_check

	@property
	def name(self) -> str:
		return "ArchitectureGA"

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> 'ClusterGenome':
		"""
		Mutate genome based on optimize_* flags in config.

		The optimizer is phase-agnostic. Callers control what gets optimized
		by setting optimize_bits, optimize_neurons, optimize_connections.

		Mutation delta ranges are 10% of (min + max):
		- Neurons: 10% × (min_neurons + max_neurons)
		- Bits: 10% × (min_bits + max_bits)
		- Connections: 10% × bits_per_token (stays close to token boundaries)

		Bits are mutated per-neuron: for each mutable cluster, each neuron's
		bits_per_neuron is independently mutated.

		When neurons_per_cluster changes, bits_per_neuron is rebuilt to match:
		- Kept neurons retain their (possibly mutated) bits
		- New neurons get bits copied from a random existing neuron in that cluster

		Also adjusts connections when architecture changes.
		"""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		# Calculate delta ranges: 10% of (min + max), minimum 1
		bits_delta_max = max(1, round(0.1 * (cfg.min_bits + cfg.max_bits)))
		neurons_delta_max = max(1, round(0.1 * (cfg.min_neurons + cfg.max_neurons)))

		# Track old architecture for connection adjustment
		old_bits_per_neuron = genome.bits_per_neuron.copy()
		old_neurons = genome.neurons_per_cluster.copy()

		# If only optimizing connections, skip architecture mutation
		if cfg.optimize_connections and not cfg.optimize_bits and not cfg.optimize_neurons:
			if genome.connections is not None and cfg.total_input_bits is not None:
				mutant.connections = self._mutate_connections_only(
					genome.connections.copy(), cfg.total_input_bits, mutation_rate
				)
			return mutant

		# Mutate architecture (bits and/or neurons)
		# If mutable_clusters is set, only mutate clusters in that set (per-tier optimization)
		if cfg.mutable_clusters is not None:
			mutable_set = set(cfg.mutable_clusters)
		else:
			mutable_set = set(range(cfg.num_clusters))

		neuron_offsets = genome.cluster_neuron_offsets

		# Phase 1: Mutate bits per neuron in-place (using original offsets)
		for i in range(cfg.num_clusters):
			if i not in mutable_set:
				continue
			if self._rng.random() < mutation_rate:
				if cfg.optimize_bits:
					n_start = neuron_offsets[i]
					n_end = neuron_offsets[i + 1]
					for n_idx in range(n_start, n_end):
						delta = self._rng.randint(-bits_delta_max, bits_delta_max)
						new_bits = mutant.bits_per_neuron[n_idx] + delta
						mutant.bits_per_neuron[n_idx] = max(cfg.min_bits, min(cfg.max_bits, new_bits))

				if cfg.optimize_neurons:
					delta = self._rng.randint(-neurons_delta_max, neurons_delta_max)
					new_neurons = mutant.neurons_per_cluster[i] + delta
					mutant.neurons_per_cluster[i] = max(cfg.min_neurons, min(cfg.max_neurons, new_neurons))

		# Phase 2: Rebuild bits_per_neuron to match new neurons_per_cluster
		# (handles neurons being added or removed from clusters)
		if cfg.optimize_neurons:
			new_bits_per_neuron = []
			for i in range(cfg.num_clusters):
				n_start = neuron_offsets[i]
				n_end = neuron_offsets[i + 1]
				old_n = n_end - n_start  # original neuron count
				new_n = mutant.neurons_per_cluster[i]
				# Bits that were (possibly mutated) for existing neurons
				cluster_bits = mutant.bits_per_neuron[n_start:n_end]

				if new_n <= old_n:
					# Fewer or same neurons: keep first new_n
					new_bits_per_neuron.extend(cluster_bits[:new_n])
				else:
					# More neurons: keep all existing, add new ones
					new_bits_per_neuron.extend(cluster_bits)
					for _ in range(new_n - old_n):
						# New neuron gets bits from random existing neuron in cluster
						if old_n > 0:
							template = self._rng.randint(0, old_n - 1)
							new_bits_per_neuron.append(cluster_bits[template])
						else:
							new_bits_per_neuron.append(cfg.default_bits)
			mutant.bits_per_neuron = new_bits_per_neuron

		# Adjust connections if they exist and architecture changed
		if genome.connections is not None and cfg.total_input_bits is not None:
			mutant.connections = self._adjust_connections_for_mutation(
				genome, mutant, old_bits_per_neuron, old_neurons, cfg.total_input_bits
			)

		return mutant

	def _mutate_connections_only(
		self,
		connections: list[int],
		total_input_bits: int,
		mutation_rate: float,
	) -> list[int]:
		"""
		Mutate connections without changing architecture.

		Delta range is 10% of bits_per_token (stays close to token boundaries).
		Assumes context × bits_per_token = total_input_bits.
		"""
		# Estimate bits_per_token: typically 16 for GPT-2 vocab
		# Use sqrt heuristic: bits_per_token ≈ log2(vocab) ≈ 16
		bits_per_token = 16  # Could be passed in, but 16 is reasonable default
		conn_delta_max = max(1, round(0.1 * bits_per_token))  # 10% of 16 = 2

		result = connections.copy()
		for i in range(len(result)):
			if self._rng.random() < mutation_rate:
				delta = self._rng.randint(-conn_delta_max, conn_delta_max)
				result[i] = max(0, min(total_input_bits - 1, result[i] + delta))
		return result

	def _adjust_connections_for_mutation(
		self,
		old_genome: 'ClusterGenome',
		new_genome: 'ClusterGenome',
		old_bits_per_neuron: list[int],
		old_neurons: list[int],
		total_input_bits: int,
	) -> list[int]:
		"""Adjust connections when architecture changes during mutation.

		Uses per-neuron bits from old and new genomes with connection_offsets
		for precise indexing into the flat connections array.
		"""
		result = []
		old_neuron_offsets = old_genome.cluster_neuron_offsets
		old_conn_offsets = old_genome.connection_offsets
		new_neuron_offsets = new_genome.cluster_neuron_offsets

		for cluster_idx in range(len(new_genome.neurons_per_cluster)):
			o_neurons = old_neurons[cluster_idx]
			n_neurons = new_genome.neurons_per_cluster[cluster_idx]
			old_n_start = old_neuron_offsets[cluster_idx]
			new_n_start = new_neuron_offsets[cluster_idx]

			for local_neuron in range(n_neurons):
				new_global = new_n_start + local_neuron
				n_bits = new_genome.bits_per_neuron[new_global]

				if local_neuron < o_neurons:
					# Existing neuron - copy and adjust connections
					old_global = old_n_start + local_neuron
					o_bits = old_bits_per_neuron[old_global]
					old_conn_start = old_conn_offsets[old_global]

					for bit_idx in range(n_bits):
						if bit_idx < o_bits:
							# Copy existing connection, with small random mutation
							old_conn = old_genome.connections[old_conn_start + bit_idx]
							# 10% chance of small perturbation
							if self._rng.random() < 0.1:
								delta = self._rng.choice([-2, -1, 1, 2])
								new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
							else:
								new_conn = old_conn
							result.append(new_conn)
						else:
							# New bit position - add random connection
							result.append(self._rng.randint(0, total_input_bits - 1))
				else:
					# New neuron - copy connections from random existing neuron with mutations
					if o_neurons > 0:
						template_local = self._rng.randint(0, o_neurons - 1)
						template_global = old_n_start + template_local
						template_bits = old_bits_per_neuron[template_global]
						template_conn_start = old_conn_offsets[template_global]

						for bit_idx in range(n_bits):
							if bit_idx < template_bits:
								# Copy from template with mutation
								old_conn = old_genome.connections[template_conn_start + bit_idx]
								delta = self._rng.choice([-2, -1, 1, 2])
								new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
								result.append(new_conn)
							else:
								result.append(self._rng.randint(0, total_input_bits - 1))
					else:
						# No existing neurons to copy from - fully random
						for _ in range(n_bits):
							result.append(self._rng.randint(0, total_input_bits - 1))

		return result

	def crossover_genomes(self, parent1: 'ClusterGenome', parent2: 'ClusterGenome') -> 'ClusterGenome':
		"""
		Single-point crossover at cluster boundary.

		Child inherits per-neuron bits for entire clusters from the chosen parent.
		Connections are inherited from the parent whose cluster config is taken.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		self._ensure_rng()
		n = len(parent1.neurons_per_cluster)
		crossover_point = self._rng.randint(1, n - 1)

		# Build child neurons_per_cluster
		child_neurons = parent1.neurons_per_cluster[:crossover_point] + parent2.neurons_per_cluster[crossover_point:]

		# Build child bits_per_neuron using cluster_neuron_offsets
		p1_offsets = parent1.cluster_neuron_offsets
		p2_offsets = parent2.cluster_neuron_offsets

		# Take per-neuron bits from parent1 for clusters [0, crossover_point)
		p1_neuron_end = p1_offsets[crossover_point]
		child_bits = list(parent1.bits_per_neuron[:p1_neuron_end])

		# Take per-neuron bits from parent2 for clusters [crossover_point, n)
		p2_neuron_start = p2_offsets[crossover_point]
		child_bits.extend(parent2.bits_per_neuron[p2_neuron_start:])

		# Build child connections if parents have them
		child_connections = None
		if parent1.connections is not None and parent2.connections is not None:
			p1_conn_offsets = parent1.connection_offsets
			p2_conn_offsets = parent2.connection_offsets

			# Parent1 connections for clusters [0, crossover_point)
			p1_conn_end = p1_conn_offsets[p1_neuron_end]
			child_connections = list(parent1.connections[:p1_conn_end])

			# Parent2 connections for clusters [crossover_point, n)
			p2_conn_start = p2_conn_offsets[p2_neuron_start]
			child_connections.extend(parent2.connections[p2_conn_start:])

		return ClusterGenome(
			bits_per_neuron=child_bits,
			neurons_per_cluster=child_neurons,
			connections=child_connections,
		)

	def create_random_genome(self) -> 'ClusterGenome':
		"""
		Create a random genome based on optimize_* flags.

		- If optimize_bits=True: random bits per neuron in [min_bits, max_bits]
		- If optimize_bits=False: use default_bits for all neurons
		- Same logic for neurons

		Bits are generated per-neuron (flat list), not per-cluster.
		When optimizing connections only, both bits and neurons use defaults.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		self._ensure_rng()
		cfg = self._arch_config

		if cfg.token_frequencies is not None:
			return self._create_frequency_scaled_genome()

		# Initialize neurons: random if optimizing, default otherwise
		if cfg.optimize_neurons:
			neurons = [self._rng.randint(cfg.min_neurons, cfg.max_neurons) for _ in range(cfg.num_clusters)]
		else:
			neurons = [cfg.default_neurons] * cfg.num_clusters

		# Initialize per-neuron bits: random if optimizing, default otherwise
		total_neurons = sum(neurons)
		if cfg.optimize_bits:
			bits_per_neuron = [self._rng.randint(cfg.min_bits, cfg.max_bits) for _ in range(total_neurons)]
		else:
			bits_per_neuron = [cfg.default_bits] * total_neurons

		# Initialize connections if total_input_bits available
		connections = None
		if cfg.total_input_bits is not None:
			from wnn.ram.strategies.connectivity.adaptive_cluster import generate_connections
			connections = generate_connections(bits_per_neuron, cfg.total_input_bits, self._rng.randint(0, 2**63))

		return ClusterGenome(bits_per_neuron=bits_per_neuron, neurons_per_cluster=neurons, connections=connections)

	def _create_frequency_scaled_genome(self) -> 'ClusterGenome':
		"""
		Create genome with bits/neurons scaled by token frequency.

		- If optimize_bits=True: scale bits by frequency (per-neuron)
		- If optimize_bits=False: use default_bits
		- Same logic for neurons

		Bits are expanded to per-neuron (flat list) after computing per-cluster values.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		cfg = self._arch_config
		freqs = cfg.token_frequencies

		# Normalize frequencies to [0, 1]
		max_freq = max(freqs) if freqs else 1
		norm_freqs = [f / max_freq if max_freq > 0 else 0 for f in freqs]

		cluster_bits = []
		neurons = []
		for nf in norm_freqs:
			# Bits: scaled if optimizing, default otherwise
			if cfg.optimize_bits:
				b = int(cfg.min_bits + nf * (cfg.max_bits - cfg.min_bits))
			else:
				b = cfg.default_bits

			# Neurons: scaled if optimizing, default otherwise
			if cfg.optimize_neurons:
				n = int(cfg.min_neurons + nf * (cfg.max_neurons - cfg.min_neurons))
			else:
				n = cfg.default_neurons

			cluster_bits.append(max(cfg.min_bits, min(cfg.max_bits, b)))
			neurons.append(max(cfg.min_neurons, min(cfg.max_neurons, n)))

		# Expand per-cluster bits to per-neuron (flat list)
		bits_per_neuron = []
		for i in range(cfg.num_clusters):
			bits_per_neuron.extend([cluster_bits[i]] * neurons[i])

		# Initialize connections if total_input_bits available
		connections = None
		if cfg.total_input_bits is not None:
			from wnn.ram.strategies.connectivity.adaptive_cluster import generate_connections
			connections = generate_connections(bits_per_neuron, cfg.total_input_bits, self._rng.randint(0, 2**63))

		return ClusterGenome(bits_per_neuron=bits_per_neuron, neurons_per_cluster=neurons, connections=connections)

	# =========================================================================
	# Hooks: Rust-accelerated offspring generation + lifecycle
	# =========================================================================

	def _generate_offspring(self, population, n_needed, threshold, generation):
		"""Generate offspring via Rust search_offspring or Python fallback."""
		if self._cached_evaluator is not None:
			cfg = self._config
			arch_cfg = self._arch_config
			evaluator = self._cached_evaluator

			# Phase-aware mutation rates
			bits_mutation_rate = cfg.mutation_rate if arch_cfg.optimize_bits else 0.0
			neurons_mutation_rate = cfg.mutation_rate if arch_cfg.optimize_neurons else 0.0

			# fitness_percentile controls selectivity: generate a larger pool,
			# rank by fitness, keep only the top fraction → return exactly n_needed.
			# e.g. percentile=0.75 → generate ceil(24/0.75)=32, rank, keep best 24.
			import math
			pct = cfg.fitness_percentile if cfg.fitness_percentile and 0 < cfg.fitness_percentile < 1.0 else None
			generate_count = math.ceil(n_needed / pct) if pct else n_needed

			# Convert 3-tuple population to 2-tuple for Rust
			rust_population = [(g, ce) for g, ce, _ in population]

			search_result = evaluator.search_offspring(
				population=rust_population,
				target_count=generate_count,
				max_attempts=generate_count * 5,
				accuracy_threshold=threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				bits_mutation_rate=bits_mutation_rate,
				neurons_mutation_rate=neurons_mutation_rate,
				crossover_rate=cfg.crossover_rate,
				tournament_size=cfg.tournament_size,
				train_subset_idx=self._phase_train_idx,
				eval_subset_idx=0,
				seed=self._seed_offset + generation,
				generation=generation,
				total_generations=cfg.generations,
				return_best_n=True,
				mutable_clusters=arch_cfg.mutable_clusters,
			)

			# Convert to 3-tuples, rank by fitness, return best n_needed
			offspring = [
				(g, g._cached_fitness[0], g._cached_fitness[1])
				for g in search_result.genomes
				if hasattr(g, '_cached_fitness')
			]

			if pct and len(offspring) > n_needed:
				scores = self._fitness_calculator.fitness(offspring)
				ranked = sorted(zip(offspring, scores), key=lambda x: x[1])
				offspring = [item for item, _ in ranked[:n_needed]]

			return offspring

		# Fallback to Python generation
		return super()._generate_offspring(population, n_needed, threshold, generation)

	def _on_generation_start(self, generation, **ctx):
		"""Metal cleanup, checkpoint save, shutdown check, generation tracking."""
		# Update evaluator generation for adaptive evaluation (Baldwin effect)
		evaluator = self._cached_evaluator or self._batch_evaluator
		if evaluator is not None and hasattr(evaluator, 'set_generation'):
			evaluator.set_generation(generation, total_generations=ctx.get('total_generations'))

		# Metal cleanup (every generation except first)
		if generation > 0 and self._cached_evaluator is not None:
			self._cleanup_metal(generation, log_interval=10)

		# Checkpoint save
		if generation > 0 and generation % 50 == 0 and self._checkpoint_mgr is not None:
			population = ctx.get('population', [])
			self._checkpoint_mgr.save(
				iteration=generation,
				population=[(g, ce) for g, ce, _ in population],
				best_genome=ctx.get('best_genome'),
				best_fitness=(ctx.get('best_fitness'), ctx.get('best_accuracy')),
				current_threshold=ctx.get('threshold', 0.0),
				extra_state={
					'patience_counter': getattr(ctx.get('early_stopper'), '_patience_counter', 0),
				},
			)

		# Shutdown check
		if self._shutdown_check and self._shutdown_check():
			# Save checkpoint before stopping
			if self._checkpoint_mgr is not None:
				population = ctx.get('population', [])
				self._checkpoint_mgr.save(
					iteration=generation,
					population=[(g, ce) for g, ce, _ in population],
					best_genome=ctx.get('best_genome'),
					best_fitness=(ctx.get('best_fitness'), ctx.get('best_accuracy')),
					current_threshold=ctx.get('threshold', 0.0),
				)
			self._log.info(f"[{self.name}] Shutdown requested at generation {generation}, stopping...")
			raise StopIteration("Shutdown requested")

	# =========================================================================
	# Simplified optimize: setup + super() + validation
	# =========================================================================

	def optimize(
		self,
		evaluate_fn: Callable[['ClusterGenome'], float] = None,
		initial_genome: Optional['ClusterGenome'] = None,
		initial_population: Optional[list['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[list['ClusterGenome']], list[tuple[float, float]]]] = None,
		**kwargs,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run GA with optional Rust acceleration.

		Sets up Rust state, delegates to unified loop in base class (which uses
		our _generate_offspring override for Rust-accelerated offspring), then
		runs validation summary on full data.
		"""
		import time

		# Checkpoint manager setup
		self._checkpoint_mgr: Optional[CheckpointManager] = None
		if self._checkpoint_config and self._checkpoint_config.enabled:
			self._checkpoint_mgr = CheckpointManager(
				config=self._checkpoint_config,
				phase_name=self._phase_name,
				optimizer_type="GA",
				total_iterations=self._config.generations,
				logger=self._log.info,
			)
			# Checkpoint resume
			if self._checkpoint_mgr.has_checkpoint():
				from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
				resume_state = self._checkpoint_mgr.load(ClusterGenome)
				self._log.info(f"[{self.name}] Resuming from checkpoint at generation {resume_state['current_iteration'] + 1}")
				# Restore population as initial_population (will be re-evaluated)
				initial_population = [g for g, _ in resume_state['population']]

		# Set up phase state for Rust acceleration
		if self._cached_evaluator is not None:
			self._phase_train_idx = self._cached_evaluator.next_train_idx()
			self._seed_offset = int(time.time() * 1000) % (2**16)
			cfg = self._config

			# Ensure all seed genomes have connections
			if initial_population:
				for g in initial_population:
					if not g.has_connections():
						g.initialize_connections(self._cached_evaluator.total_input_bits)

				# Expand population with mutations if needed (unless seed_only)
				seed_count = len(initial_population)
				need_count = cfg.population_size - seed_count
				if need_count > 0 and not cfg.seed_only and not cfg.fresh_population:
					from wnn.ram.strategies.connectivity.adaptive_cluster import AdaptiveClusterConfig
					arch_cfg = self._arch_config
					mutation_config = AdaptiveClusterConfig(
						min_bits=arch_cfg.min_bits,
						max_bits=arch_cfg.max_bits,
						min_neurons=arch_cfg.min_neurons,
						max_neurons=arch_cfg.max_neurons,
						bits_mutation_rate=0.3 if arch_cfg.optimize_bits else 0.0,
						neurons_mutation_rate=0.3 if arch_cfg.optimize_neurons else 0.0,
					)
					expanded = list(initial_population)
					for i in range(need_count):
						seed = initial_population[i % seed_count]
						mutated = seed.mutate(
							config=mutation_config,
							total_input_bits=self._cached_evaluator.total_input_bits,
						)
						expanded.append(mutated)
					initial_population = expanded

			if cfg.fresh_population:
				initial_population = None

			# Wrap cached evaluator as batch_evaluate_fn
			evaluator = self._cached_evaluator
			phase_train_idx = self._phase_train_idx
			batch_evaluate_fn = lambda genomes, min_accuracy=None: evaluator.evaluate_batch(
				genomes,
				train_subset_idx=phase_train_idx,
				eval_subset_idx=0,
				logger=self._log,
				min_accuracy=min_accuracy,
			)

		elif self._batch_evaluator is not None and batch_evaluate_fn is None:
			batch_evaluate_fn = lambda genomes, min_accuracy=None: self._batch_evaluator.evaluate_batch(
				genomes, logger=self._log, min_accuracy=min_accuracy,
			)

		# Delegate to unified loop (uses our _generate_offspring override)
		result = super().optimize(
			evaluate_fn=evaluate_fn,
			initial_genome=initial_genome,
			initial_population=initial_population,
			batch_evaluate_fn=batch_evaluate_fn,
			**kwargs,
		)

		# Validation summary (Rust path only: full-data evaluation)
		if self._cached_evaluator is not None:
			result = self._run_validation_summary(result)

		return result


class ArchitectureTSStrategy(ArchitectureStrategyMixin, GenericTSStrategy['ClusterGenome']):
	"""
	Tabu Search for architecture (bits, neurons per cluster) optimization.

	Inherits core TS loop from GenericTSStrategy, implements ClusterGenome operations.
	Uses ArchitectureStrategyMixin for shared functionality (Metal cleanup, shutdown, etc.)

	Features:
	- Rust/Metal batch evaluation (default when available)
	- Rust-based neighbor search with threshold (when cached_evaluator provided)
	- Population seeding from previous phases
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ts_config: Optional[TSConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
		cached_evaluator: Optional[Any] = None,  # CachedEvaluator for Rust search_neighbors
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
	):
		super().__init__(config=ts_config, seed=seed, logger=logger)
		self._arch_config = arch_config
		self._batch_evaluator = batch_evaluator
		# Use cached_evaluator if provided, or check if batch_evaluator has search_neighbors
		if cached_evaluator is not None:
			self._cached_evaluator = cached_evaluator
		elif batch_evaluator is not None and hasattr(batch_evaluator, 'search_neighbors'):
			self._cached_evaluator = batch_evaluator
		else:
			self._cached_evaluator = None
		self._shutdown_check = shutdown_check

	@property
	def name(self) -> str:
		return "ArchitectureTS"

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> tuple['ClusterGenome', Any]:
		"""
		Generate a neighbor by mutating multiple clusters based on mutation_rate.

		The optimizer is phase-agnostic. Callers control what gets optimized
		by setting optimize_bits, optimize_neurons, optimize_connections.

		Mutation delta ranges are 10% of (min + max):
		- Neurons: 10% × (min_neurons + max_neurons)
		- Bits: 10% × (min_bits + max_bits)
		- Connections: 10% × bits_per_token

		Bits are mutated per-neuron: for each mutable cluster, each neuron's
		bits_per_neuron is independently mutated.

		Returns (new_genome, move_info) where move_info is a tuple of mutated cluster indices
		for tabu tracking.
		"""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		# Calculate delta ranges: 10% of (min + max), minimum 1
		bits_delta_max = max(1, round(0.1 * (cfg.min_bits + cfg.max_bits)))
		neurons_delta_max = max(1, round(0.1 * (cfg.min_neurons + cfg.max_neurons)))
		bits_per_token = 16  # Reasonable default for GPT-2 vocab
		conn_delta_max = max(1, round(0.1 * bits_per_token))

		# If only optimizing connections, mutate connections based on mutation_rate
		if cfg.optimize_connections and not cfg.optimize_bits and not cfg.optimize_neurons:
			if genome.connections is not None and cfg.total_input_bits is not None:
				mutated_indices = []
				for i in range(len(genome.connections)):
					if self._rng.random() < mutation_rate:
						old_val = mutant.connections[i]
						delta = self._rng.randint(-conn_delta_max, conn_delta_max)
						new_val = max(0, min(cfg.total_input_bits - 1, old_val + delta))
						mutant.connections[i] = new_val
						mutated_indices.append(i)
				move = tuple(mutated_indices) if mutated_indices else None
			else:
				move = None
			return mutant, move

		# Track old architecture for connection adjustment
		old_bits_per_neuron = genome.bits_per_neuron.copy()
		old_neurons = genome.neurons_per_cluster.copy()

		# Mutate multiple clusters based on mutation_rate
		# If mutable_clusters is set, only mutate clusters in that set (per-tier optimization)
		if cfg.mutable_clusters is not None:
			mutable_set = set(cfg.mutable_clusters)
		else:
			mutable_set = set(range(cfg.num_clusters))

		neuron_offsets = genome.cluster_neuron_offsets

		# Phase 1: Mutate bits per neuron and neurons_per_cluster
		mutated_clusters = []
		for i in range(cfg.num_clusters):
			if i not in mutable_set:
				continue
			if self._rng.random() < mutation_rate:
				mutated_clusters.append(i)

				if cfg.optimize_bits:
					# Mutate each neuron's bits independently
					n_start = neuron_offsets[i]
					n_end = neuron_offsets[i + 1]
					for n_idx in range(n_start, n_end):
						delta = self._rng.randint(-bits_delta_max, bits_delta_max)
						new_bits = mutant.bits_per_neuron[n_idx] + delta
						mutant.bits_per_neuron[n_idx] = max(cfg.min_bits, min(cfg.max_bits, new_bits))

				if cfg.optimize_neurons:
					delta = self._rng.randint(-neurons_delta_max, neurons_delta_max)
					new_neurons = mutant.neurons_per_cluster[i] + delta
					mutant.neurons_per_cluster[i] = max(cfg.min_neurons, min(cfg.max_neurons, new_neurons))

		# Phase 2: Rebuild bits_per_neuron to match new neurons_per_cluster
		if cfg.optimize_neurons:
			new_bits_per_neuron = []
			for i in range(cfg.num_clusters):
				n_start = neuron_offsets[i]
				n_end = neuron_offsets[i + 1]
				old_n = n_end - n_start
				new_n = mutant.neurons_per_cluster[i]
				cluster_bits = mutant.bits_per_neuron[n_start:n_end]

				if new_n <= old_n:
					new_bits_per_neuron.extend(cluster_bits[:new_n])
				else:
					new_bits_per_neuron.extend(cluster_bits)
					for _ in range(new_n - old_n):
						if old_n > 0:
							template = self._rng.randint(0, old_n - 1)
							new_bits_per_neuron.append(cluster_bits[template])
						else:
							new_bits_per_neuron.append(cfg.default_bits)
			mutant.bits_per_neuron = new_bits_per_neuron

		# Move is tuple of mutated cluster indices (for tabu tracking)
		move = tuple(mutated_clusters) if mutated_clusters else None

		# Adjust connections if they exist and architecture changed
		if genome.connections is not None and cfg.total_input_bits is not None:
			mutant.connections = self._adjust_connections_for_mutation_ts(
				genome, mutant, old_bits_per_neuron, old_neurons, cfg.total_input_bits
			)

		return mutant, move

	def _adjust_connections_for_mutation_ts(
		self,
		old_genome: 'ClusterGenome',
		new_genome: 'ClusterGenome',
		old_bits_per_neuron: list[int],
		old_neurons: list[int],
		total_input_bits: int,
	) -> list[int]:
		"""Adjust connections when architecture changes during TS mutation.

		Uses per-neuron bits from old and new genomes with connection_offsets
		for precise indexing into the flat connections array.
		"""
		result = []
		old_neuron_offsets = old_genome.cluster_neuron_offsets
		old_conn_offsets = old_genome.connection_offsets
		new_neuron_offsets = new_genome.cluster_neuron_offsets

		for cluster_idx in range(len(new_genome.neurons_per_cluster)):
			o_neurons = old_neurons[cluster_idx]
			n_neurons = new_genome.neurons_per_cluster[cluster_idx]
			old_n_start = old_neuron_offsets[cluster_idx]
			new_n_start = new_neuron_offsets[cluster_idx]

			for local_neuron in range(n_neurons):
				new_global = new_n_start + local_neuron
				n_bits = new_genome.bits_per_neuron[new_global]

				if local_neuron < o_neurons:
					# Existing neuron - copy connections
					old_global = old_n_start + local_neuron
					o_bits = old_bits_per_neuron[old_global]
					old_conn_start = old_conn_offsets[old_global]

					for bit_idx in range(n_bits):
						if bit_idx < o_bits:
							result.append(old_genome.connections[old_conn_start + bit_idx])
						else:
							# New bit position - add random connection
							result.append(self._rng.randint(0, total_input_bits - 1))
				else:
					# New neuron - copy from random existing with slight mutation
					if o_neurons > 0:
						template_local = self._rng.randint(0, o_neurons - 1)
						template_global = old_n_start + template_local
						template_bits = old_bits_per_neuron[template_global]
						template_conn_start = old_conn_offsets[template_global]

						for bit_idx in range(n_bits):
							if bit_idx < template_bits:
								old_conn = old_genome.connections[template_conn_start + bit_idx]
								delta = self._rng.choice([-2, -1, 1, 2])
								new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
								result.append(new_conn)
							else:
								result.append(self._rng.randint(0, total_input_bits - 1))
					else:
						for _ in range(n_bits):
							result.append(self._rng.randint(0, total_input_bits - 1))

		return result

	def is_tabu_move(self, move: Any, tabu_list: list[Any]) -> bool:
		"""
		Check if move overlaps significantly with recent tabu moves.

		Move is now a tuple of mutated cluster indices. A move is tabu if
		it shares more than 50% of clusters with a recent tabu move.
		"""
		if move is None or not move:
			return False

		move_set = set(move)
		for tabu_move in tabu_list:
			if tabu_move is None:
				continue
			tabu_set = set(tabu_move)
			overlap = len(move_set & tabu_set)
			# Tabu if >50% overlap with any recent move
			if overlap > len(move_set) * 0.5:
				return True

		return False

	# =========================================================================
	# Hooks: Rust-accelerated neighbor generation + lifecycle
	# =========================================================================

	def _generate_neighbors(self, best_genome, n_neighbors, threshold, iteration, tabu_list):
		"""Generate neighbors via Rust search_neighbors or Python fallback."""
		if self._cached_evaluator is not None:
			cfg = self._config
			arch_cfg = self._arch_config
			evaluator = self._cached_evaluator

			# Phase-aware mutation rates
			bits_mutation_rate = cfg.mutation_rate if arch_cfg.optimize_bits else 0.0
			neurons_mutation_rate = cfg.mutation_rate if arch_cfg.optimize_neurons else 0.0

			# fitness_percentile: generate larger pool, rank, keep best n_neighbors
			import math
			pct = cfg.fitness_percentile if cfg.fitness_percentile and 0 < cfg.fitness_percentile < 1.0 else None
			generate_count = math.ceil(n_neighbors / pct) if pct else n_neighbors

			self._log.debug(f"[{self.name}] Searching {generate_count} neighbors from best ranked (keeping best {n_neighbors})...")
			neighbors_raw = evaluator.search_neighbors(
				genome=best_genome,
				target_count=generate_count,
				max_attempts=generate_count * 5,
				accuracy_threshold=threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				bits_mutation_rate=bits_mutation_rate,
				neurons_mutation_rate=neurons_mutation_rate,
				train_subset_idx=self._phase_train_idx,
				eval_subset_idx=0,
				seed=self._seed_offset + iteration * 1000,
				generation=iteration,
				total_generations=cfg.iterations,
				return_best_n=True,
				mutable_clusters=arch_cfg.mutable_clusters,
			)

			# Convert to 3-tuples, rank by fitness, return best n_neighbors
			neighbors = [
				(g, g._cached_fitness[0], g._cached_fitness[1])
				for g in neighbors_raw
				if hasattr(g, '_cached_fitness')
			]

			if pct and len(neighbors) > n_neighbors:
				scores = self._fitness_calculator.fitness(neighbors)
				ranked = sorted(zip(neighbors, scores), key=lambda x: x[1])
				neighbors = [item for item, _ in ranked[:n_neighbors]]

			return neighbors

		# Fallback to Python single-path generation
		return super()._generate_neighbors(best_genome, n_neighbors, threshold, iteration, tabu_list)

	def _on_iteration_start(self, iteration, **ctx):
		"""Metal cleanup, shutdown check, generation tracking."""
		# Update evaluator generation for adaptive evaluation (Baldwin effect)
		evaluator = self._cached_evaluator or self._batch_evaluator
		if evaluator is not None and hasattr(evaluator, 'set_generation'):
			evaluator.set_generation(iteration, total_generations=ctx.get('total_generations'))

		# Metal cleanup (every iteration except first)
		if iteration > 0 and self._cached_evaluator is not None:
			self._cleanup_metal(iteration, log_interval=10)

		# Shutdown check
		if self._shutdown_check and self._shutdown_check():
			self._log.info(f"[{self.name}] Shutdown requested at iteration {iteration}, stopping...")
			raise StopIteration("Shutdown requested")

	# =========================================================================
	# Simplified optimize: setup + super() + validation
	# =========================================================================

	def optimize(
		self,
		initial_genome: 'ClusterGenome' = None,
		initial_fitness: Optional[float] = None,
		evaluate_fn: Callable[['ClusterGenome'], float] = None,
		initial_neighbors: Optional[list['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[list['ClusterGenome']], list[tuple[float, float]]]] = None,
		**kwargs,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run TS with optional Rust acceleration.

		Sets up Rust state, delegates to unified loop in base class (which uses
		our _generate_neighbors override for Rust-accelerated search), then
		runs validation summary on full data.

		IMPORTANT: initial_fitness is REQUIRED.
		"""
		import time

		# initial_fitness is REQUIRED - fail fast if missing
		if initial_fitness is None:
			raise ValueError(
				f"[{self.name}] initial_fitness is REQUIRED but was None. "
				"This indicates the previous phase's final_fitness was not properly passed. "
				"Check that: (1) GA saved a checkpoint with final_fitness, "
				"(2) Flow checkpoint loading works correctly, "
				"(3) Flow is not creating new experiments instead of resuming."
			)

		# Set up phase state for Rust acceleration
		if self._cached_evaluator is not None:
			self._phase_train_idx = self._cached_evaluator.next_train_idx()
			self._seed_offset = int(time.time() * 1000) % (2**16)

			# Ensure initial genome has connections
			if not initial_genome.has_connections():
				initial_genome.initialize_connections(self._cached_evaluator.total_input_bits)
			if initial_neighbors:
				for g in initial_neighbors:
					if not g.has_connections():
						g.initialize_connections(self._cached_evaluator.total_input_bits)

			# Wrap cached evaluator as batch_evaluate_fn
			evaluator = self._cached_evaluator
			phase_train_idx = self._phase_train_idx
			batch_evaluate_fn = lambda genomes, min_accuracy=None: evaluator.evaluate_batch(
				genomes,
				train_subset_idx=phase_train_idx,
				eval_subset_idx=0,
				logger=self._log,
				min_accuracy=min_accuracy,
			)

		elif self._batch_evaluator is not None and batch_evaluate_fn is None:
			batch_evaluate_fn = lambda genomes, min_accuracy=None: self._batch_evaluator.evaluate_batch(
				genomes, logger=self._log, min_accuracy=min_accuracy,
			)

		# Delegate to unified loop (uses our _generate_neighbors override)
		result = super().optimize(
			initial_genome=initial_genome,
			initial_fitness=initial_fitness,
			evaluate_fn=evaluate_fn,
			initial_neighbors=initial_neighbors,
			batch_evaluate_fn=batch_evaluate_fn,
			**kwargs,
		)

		# Validation summary (Rust path only: full-data evaluation)
		if self._cached_evaluator is not None:
			result = self._run_validation_summary(result)

		return result


# =============================================================================
# Grid Search Strategy
# =============================================================================

@dataclass
class GridSearchConfig:
	"""Configuration for grid search over neuron × bit combinations."""
	num_clusters: int
	neurons_grid: list[int] = field(default_factory=lambda: [50, 100, 150, 200])
	bits_grid: list[int] = field(default_factory=lambda: [14, 16, 18, 20])
	top_k: int = 3
	population_size: int = 50  # Total genomes in output population
	total_input_bits: Optional[int] = None
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0


class GridSearchStrategy:
	"""
	One-shot evaluation of neuron × bit configurations.

	Unlike GA/TS which iteratively optimize, grid search evaluates
	each (neurons, bits) combination once and ranks by fitness.
	Top-K configurations get proportionally more representation
	in the output population, which seeds the next phase.

	Each grid point is recorded as an iteration for dashboard tracking.

	Evaluation uses the BitwiseEvaluator (Rust + Metal) for fast
	train+eval. All grid configs are batched together so the evaluator
	can leverage parallel CPU training + GPU forward passes.
	"""

	def __init__(
		self,
		config: GridSearchConfig,
		batch_evaluator: Optional[Any] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		self._config = config
		self._batch_evaluator = batch_evaluator
		self._seed = seed
		self._rng = random.Random(seed)
		self._log = logger or (lambda x: None)
		self._shutdown_check = shutdown_check
		self._tracker = None
		self._tracker_experiment_id = None

	@property
	def name(self) -> str:
		return "GridSearch"

	def set_tracker(self, tracker: "ExperimentTracker", experiment_id: int, _unused: Optional[int] = None) -> None:
		"""Set the experiment tracker for iteration recording."""
		self._tracker = tracker
		self._tracker_experiment_id = experiment_id

	def _create_genome(self, neurons_per_cluster: int, bits_per_neuron: int) -> 'ClusterGenome':
		"""Create a genome with uniform neurons/bits and random connections."""
		import numpy as np
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		cfg = self._config
		neurons = [neurons_per_cluster] * cfg.num_clusters
		total_neurons = sum(neurons)
		bits = [bits_per_neuron] * total_neurons

		connections = None
		if cfg.total_input_bits is not None:
			from wnn.ram.strategies.connectivity.adaptive_cluster import generate_connections
			connections = generate_connections(bits, cfg.total_input_bits, self._rng.randint(0, 2**63))

		return ClusterGenome(bits_per_neuron=bits, neurons_per_cluster=neurons, connections=connections)

	def optimize(
		self,
		evaluate_fn: Optional[Callable] = None,
		initial_genome: Optional['ClusterGenome'] = None,
		initial_population: Optional[list['ClusterGenome']] = None,
		**kwargs,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run grid search: evaluate each (neurons, bits) config and rank by fitness.

		All grid genomes are created first, then evaluated in a single batch
		through the Rust+Metal evaluator for maximum throughput.
		"""
		import time
		from wnn.ram.fitness import FitnessCalculatorFactory

		cfg = self._config
		calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weight_ce=cfg.fitness_weight_ce,
			weight_acc=cfg.fitness_weight_acc,
		)

		total_configs = len(cfg.neurons_grid) * len(cfg.bits_grid)
		self._log(f"\n{'='*70}")
		self._log(f"Grid Search ({total_configs} configs)")
		self._log(f"  neurons: {cfg.neurons_grid}")
		self._log(f"  bits:    {cfg.bits_grid}")
		self._log(f"  clusters: {cfg.num_clusters}")
		self._log(f"{'='*70}")

		# Phase 1: Create all genomes upfront
		config_list = []  # [(neurons, bits, genome)]
		for neurons in cfg.neurons_grid:
			for bits in cfg.bits_grid:
				genome = self._create_genome(neurons, bits)
				config_list.append((neurons, bits, genome))

		# Phase 2: Batch evaluate all genomes at once (Rust + Metal)
		all_genomes = [g for _, _, g in config_list]
		t0 = time.time()

		if self._batch_evaluator is not None:
			self._log(f"  Evaluating {total_configs} configs in batch (Rust + Metal)...")
			evals = self._batch_evaluator.evaluate_batch(all_genomes)
		elif evaluate_fn is not None:
			evals = [(evaluate_fn(g), 0.0, 0.0) for g in all_genomes]
		else:
			raise ValueError("GridSearchStrategy requires a batch_evaluator or evaluate_fn")

		batch_elapsed = time.time() - t0
		self._log(f"  Batch evaluation: {batch_elapsed:.1f}s total, "
				  f"{batch_elapsed/total_configs:.1f}s/config")

		# Phase 3: Collect results (grid search is one-shot, no iterations)
		results = []
		for idx, ((neurons, bits, genome), (ce, acc, bit_acc)) in enumerate(
			zip(config_list, evals)
		):
			self._log(f"  [{idx+1}/{total_configs}] n={neurons:3d}, b={bits:2d}: "
					  f"CE={ce:.4f}  Acc={acc:.2%}")

			results.append({
				"neurons": neurons,
				"bits": bits,
				"ce": ce,
				"accuracy": acc,
				"bit_accuracy": bit_acc,
				"elapsed_s": batch_elapsed / total_configs,
				"genome": genome,
			})

		if not results:
			raise ValueError("Grid search produced no results")

		# Phase 4: Rank by fitness
		population = [(r, r["ce"], r["accuracy"]) for r in results]
		fitness_scores = calculator.fitness(population)
		for r, score in zip(results, fitness_scores):
			r["fitness"] = score
		results.sort(key=lambda r: r["fitness"])

		self._log(f"\n{'─'*70}")
		self._log(f"Grid Search Rankings (by {calculator.name}):")
		for i, r in enumerate(results):
			marker = " ★" if i < cfg.top_k else ""
			self._log(f"  {i+1:2d}. n={r['neurons']:3d}, b={r['bits']:2d}: "
					  f"CE={r['ce']:.4f}  Acc={r['accuracy']:.2%}  "
					  f"Fit={r['fitness']:.4f}{marker}")

		# Phase 5: Build output population with proportional representation
		# Better configs get more genomes (each with fresh random connections).
		# Weights: [K, K-1, ..., 1] → best config gets K/sum(weights) of population.
		best_result = results[0]
		best_genome = best_result["genome"]

		top_k = min(cfg.top_k, len(results))
		weights = list(range(top_k, 0, -1))  # [K, K-1, ..., 1]
		total_weight = sum(weights)
		genomes_per_config = [
			max(1, round(w / total_weight * cfg.population_size)) for w in weights
		]

		# Adjust to hit exact population_size
		while sum(genomes_per_config) > cfg.population_size:
			for i in range(len(genomes_per_config) - 1, -1, -1):
				if genomes_per_config[i] > 1:
					genomes_per_config[i] -= 1
					break
		while sum(genomes_per_config) < cfg.population_size:
			genomes_per_config[0] += 1

		# Reuse original evaluated genomes + create new variations for the rest.
		# Each config's first genome is the already-evaluated original;
		# additional genomes get fresh random connections and need evaluation.
		output_population = []
		population_metrics = []
		new_genomes = []       # genomes needing evaluation
		new_genome_indices = []  # their positions in output_population

		for i in range(top_k):
			r = results[i]
			n_genomes = genomes_per_config[i]

			# First genome: reuse the original (already evaluated)
			output_population.append(r["genome"])
			population_metrics.append((r["ce"], r["accuracy"]))

			# Remaining: fresh random connections (need evaluation)
			for _ in range(n_genomes - 1):
				genome = self._create_genome(r["neurons"], r["bits"])
				new_genome_indices.append(len(output_population))
				output_population.append(genome)
				population_metrics.append((0.0, 0.0))  # placeholder
				new_genomes.append(genome)

			self._log(f"  #{i+1:2d} n={r['neurons']:3d}, b={r['bits']:2d} (CE={r['ce']:.4f}) → {n_genomes} genomes (1 original + {n_genomes - 1} new)")

		self._log(f"\nPopulation: {len(output_population)} genomes ({top_k} original + {len(new_genomes)} new)")

		# Evaluate only the NEW genomes (originals already have metrics)
		if new_genomes and self._batch_evaluator is not None:
			self._log(f"  Evaluating {len(new_genomes)} new genomes...")
			t1 = time.time()
			new_evals = self._batch_evaluator.evaluate_batch(new_genomes)
			expand_elapsed = time.time() - t1
			batch_elapsed += expand_elapsed
			self._log(f"  New genome eval: {expand_elapsed:.1f}s")
			# Fill in the placeholder metrics
			for idx_in_new, (ce, acc, _) in enumerate(new_evals):
				pop_idx = new_genome_indices[idx_in_new]
				population_metrics[pop_idx] = (ce, acc)
				# Update best result if a new genome beat the grid search best
				if ce < best_result["ce"]:
					best_result = {"ce": ce, "accuracy": acc, "genome": output_population[pop_idx]}
					best_genome = output_population[pop_idx]

		# Phase 6: Rank full population by fitness and sort
		pop_tuples = [
			(output_population[i], ce, acc)
			for i, (ce, acc) in enumerate(population_metrics)
		]
		pop_fitness_scores = calculator.fitness(pop_tuples)

		# Sort population by fitness score (lower = better)
		sorted_indices = sorted(range(len(output_population)), key=lambda i: pop_fitness_scores[i])
		output_population = [output_population[i] for i in sorted_indices]
		population_metrics = [population_metrics[i] for i in sorted_indices]
		pop_fitness_scores = [pop_fitness_scores[i] for i in sorted_indices]

		# Three independent bests from potentially different genomes
		pop_bests = calculator.bests(pop_tuples)
		best_genome = pop_bests.best_fitness.genome

		# Record 1 iteration with ALL population genomes sorted by fitness
		if self._tracker and self._tracker_experiment_id:
			try:
				avg_ce = sum(ce for ce, _ in population_metrics) / len(population_metrics)
				avg_acc = sum(acc for _, acc in population_metrics) / len(population_metrics)
				iter_id = self._tracker.record_iteration(
					experiment_id=self._tracker_experiment_id,
					iteration_num=1,
					best_ce=pop_bests.best_ce.ce,
					best_accuracy=pop_bests.best_acc.accuracy,
					avg_ce=avg_ce,
					avg_accuracy=avg_acc,
					elapsed_secs=batch_elapsed,
					candidates_total=len(output_population),
				)
				if HAS_GENOME_TRACKING:
					for pos, (genome, (ce, acc), fit) in enumerate(zip(output_population, population_metrics, pop_fitness_scores)):
						genome_config = self._genome_to_config(genome)
						if genome_config:
							genome_id = self._tracker.get_or_create_genome(
								self._tracker_experiment_id, genome_config
							)
							self._tracker.record_genome_evaluation(
								iteration_id=iter_id,
								genome_id=genome_id,
								position=pos,
								role=GenomeRole.INIT,
								ce=ce,
								accuracy=acc,
								fitness_score=fit,
							)
				self._tracker.update_experiment_progress(
					self._tracker_experiment_id,
					current_iteration=1,
					best_ce=pop_bests.best_ce.ce,
					best_accuracy=pop_bests.best_acc.accuracy,
				)
			except Exception as e:
				self._log(f"  Warning: tracker error: {e}")

		# Build OptimizerResult
		worst_ce = results[-1]["ce"]
		improvement = ((worst_ce - pop_bests.best_ce.ce) / worst_ce * 100) if worst_ce > 0 else 0.0

		return OptimizerResult(
			initial_genome=best_genome,
			best_genome=best_genome,
			initial_fitness=worst_ce,
			final_fitness=pop_bests.best_ce.ce,
			improvement_percent=improvement,
			iterations_run=1,
			method_name="GridSearch",
			history=[(1, pop_bests.best_ce.ce)],
			early_stopped=False,
			stop_reason=StopReason.MAX_ITERATIONS,
			final_population=output_population,
			population_metrics=population_metrics,
			initial_accuracy=results[-1]["accuracy"],
			final_accuracy=pop_bests.best_acc.accuracy,
			final_threshold=None,
		)

	def _genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert genome to GenomeConfig for tracker."""
		if not HAS_GENOME_TRACKING:
			return None
		cfg = self._config
		tiers = [TierConfig(
			tier=0,
			clusters=cfg.num_clusters,
			neurons=genome.neurons_per_cluster[0] if genome.neurons_per_cluster else 0,
			bits=genome.bits_per_neuron[0] if genome.bits_per_neuron else 0,
		)]
		return GenomeConfig(tiers=tiers)


# =============================================================================
# Stats-Guided Adaptation Strategies (Neurogenesis, Synaptogenesis, Axonogenesis)
# =============================================================================

@dataclass
class AdaptationConfig:
	"""Configuration for stats-guided adaptation strategy.

	Unlike GA/TS, adaptation strategies deterministically modify architecture
	based on training statistics (neuron/cluster error, fill rates, entropy).
	The "search" is iterative refinement: each round, stats change after
	adaptation, leading to different pruning/growing/rewiring decisions.
	"""
	num_clusters: int
	min_bits: int = 8
	max_bits: int = 25
	min_neurons: int = 3
	max_neurons: int = 33
	total_input_bits: Optional[int] = None
	adaptation_mode: str = "neurogenesis"  # "neurogenesis", "synaptogenesis", "axonogenesis"
	iterations: int = 50
	population_size: int = 50
	patience: int = 5
	check_interval: int = 10
	min_improvement_pct: float = 0.5
	initial_threshold: Optional[float] = None
	threshold_delta: float = 0.01
	threshold_reference: int = 1000
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	min_accuracy_floor: Optional[float] = None


class AdaptationStrategy(ArchitectureStrategyMixin):
	"""
	Stats-guided adaptation strategy (neurogenesis, synaptogenesis, axonogenesis).

	Unlike GA/TS which use random mutation/selection, this strategy deterministically
	modifies architecture based on training statistics. Each iteration:
	  1. Evaluate all genomes with adaptation enabled (train → GPU stats → adapt → retrain → eval)
	  2. Sort by fitness, keep top performers
	  3. Report iteration metrics to dashboard
	  4. Check early stopping

	The Rust evaluator (evaluate_genomes_adaptive) handles the full cycle and
	modifies genomes in-place with adapted architectures.
	"""

	def __init__(
		self,
		config: AdaptationConfig,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		cached_evaluator: Optional[Any] = None,  # BitwiseEvaluator
		phase_name: str = "Adaptation",
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		self._config = config
		self._seed = seed or 42
		self._log = logger or (lambda x: None)
		self._cached_evaluator = cached_evaluator
		self._phase_name = phase_name
		self._shutdown_check = shutdown_check
		self._tracker = None
		self._tracker_experiment_id = None

	@property
	def name(self) -> str:
		mode = self._config.adaptation_mode.capitalize()
		return f"Adaptation({mode})"

	def set_tracker(self, tracker, experiment_id: int, _flow_experiment_id: int = None):
		"""Set V2 tracker for iteration/genome recording."""
		self._tracker = tracker
		self._tracker_experiment_id = experiment_id

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def optimize(
		self,
		evaluate_fn: Optional[Callable] = None,
		initial_population: Optional[list['ClusterGenome']] = None,
		initial_genome: Optional['ClusterGenome'] = None,
		**kwargs,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run stats-guided adaptation.

		Takes a population from the previous phase, iteratively adapts all genomes,
		sorts by fitness, and returns the best.
		"""
		import time as _time
		from wnn.ram.fitness import FitnessCalculatorFactory

		cfg = self._config
		evaluator = self._cached_evaluator
		if evaluator is None:
			raise ValueError(f"{self.name} requires a BitwiseEvaluator (cached_evaluator)")

		# Build population from initial data
		population: list['ClusterGenome'] = []
		if initial_population:
			population = [g.clone() for g in initial_population[:cfg.population_size]]
		elif initial_genome:
			population = [initial_genome.clone()]

		if not population:
			raise ValueError(f"{self.name} requires initial_population or initial_genome")

		# Pad population if needed (clone best genome with fresh connections)
		while len(population) < cfg.population_size and len(population) > 0:
			base = population[len(population) % len(population)]
			clone = base.clone()
			if hasattr(clone, 'initialize_connections') and cfg.total_input_bits:
				clone.initialize_connections(cfg.total_input_bits)
			population.append(clone)

		calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weight_ce=cfg.fitness_weight_ce,
			weight_acc=cfg.fitness_weight_acc,
		)

		# Configure evaluator's adaptation mode for this phase
		original_adapt_config = evaluator._adapt_config
		self._configure_evaluator_adaptation(evaluator)

		self._log(f"\n{'='*70}")
		self._log(f"  {self._phase_name}: {self.name}")
		self._log(f"{'='*70}")
		self._log(f"  Mode: {cfg.adaptation_mode}")
		self._log(f"  Population: {len(population)}")
		self._log(f"  Iterations: {cfg.iterations}")
		self._log(f"  Patience: {cfg.patience} (check every {cfg.check_interval})")
		self._log(f"  Fitness: {calculator.name}")
		self._log("")

		# Early stopping
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stopper = EarlyStoppingTracker(early_stop_config, self._log)

		best_genome = population[0].clone()
		best_ce = float('inf')
		best_acc = 0.0
		evals = [(float('inf'), 0.0)] * len(population)
		fitness_scores = [float('inf')] * len(population)
		history = []
		shutdown_requested = [False]
		start_time = _time.time()

		try:
			for iteration in range(cfg.iterations):
				iter_start = _time.time()

				# Check shutdown
				if self._check_and_set_shutdown(shutdown_requested):
					self._log(f"  Shutdown requested at iteration {iteration}")
					break

				# Set generation for cosine annealing
				evaluator.set_generation(iteration, total_generations=cfg.iterations)

				# Evaluate all genomes (adaptation happens in-place via Rust)
				evals = evaluator.evaluate_batch(
					population,
					generation=iteration,
					total_generations=cfg.iterations,
				)

				# Build (genome, ce, acc) tuples for fitness ranking
				pop_tuples = [
					(genome, ce, acc)
					for genome, (ce, acc, *_rest) in zip(population, evals)
				]

				# Compute fitness and sort
				fitness_scores = calculator.fitness(pop_tuples)
				sorted_indices = sorted(
					range(len(population)),
					key=lambda i: fitness_scores[i],
				)

				# Reorder population by fitness
				population = [population[i] for i in sorted_indices]
				evals = [evals[i] for i in sorted_indices]
				fitness_scores = [fitness_scores[i] for i in sorted_indices]

				# Update bests independently (CE and accuracy tracked separately, like GA/TS)
				pop_bests = calculator.bests(pop_tuples)
				if pop_bests.best_ce.ce < best_ce:
					best_ce = pop_bests.best_ce.ce
					best_genome = pop_bests.best_fitness.genome.clone()
				if pop_bests.best_acc.accuracy > best_acc:
					best_acc = pop_bests.best_acc.accuracy

				history.append((iteration + 1, best_ce))
				iter_elapsed = _time.time() - iter_start

				# Log progress
				avg_ce = sum(e[0] for e in evals) / len(evals)
				avg_acc = sum(e[1] for e in evals) / len(evals) if len(evals[0]) > 1 else 0.0
				self._log(
					f"  [{iteration+1:3d}/{cfg.iterations}] "
					f"best CE={best_ce:.4f} acc={best_acc:.2%} | "
					f"avg CE={avg_ce:.4f} acc={avg_acc:.2%} | "
					f"{iter_elapsed:.1f}s"
				)

				# Record iteration in tracker
				if self._tracker and self._tracker_experiment_id:
					try:
						iter_id = self._tracker.record_iteration(
							experiment_id=self._tracker_experiment_id,
							iteration_num=iteration + 1,
							best_ce=best_ce,
							best_accuracy=best_acc,
							avg_ce=avg_ce,
							avg_accuracy=avg_acc,
							elapsed_secs=iter_elapsed,
							candidates_total=len(population),
						)
						# Record genome evaluations
						if HAS_GENOME_TRACKING:
							for pos, (genome, (ce, acc, *_), fit) in enumerate(
								zip(population, evals, fitness_scores)
							):
								genome_config = self.genome_to_config(genome)
								if genome_config:
									genome_id = self._tracker.get_or_create_genome(
										self._tracker_experiment_id, genome_config
									)
									self._tracker.record_genome_evaluation(
										iteration_id=iter_id,
										genome_id=genome_id,
										position=pos,
										role=GenomeRole.SURVIVOR if pos == 0 else GenomeRole.OFFSPRING,
										ce=ce,
										accuracy=acc,
										fitness_score=fit,
									)
						self._tracker.update_experiment_progress(
							self._tracker_experiment_id,
							current_iteration=iteration + 1,
							best_ce=best_ce,
							best_accuracy=best_acc,
						)
					except Exception as e:
						self._log(f"  Warning: tracker error: {e}")

				# Early stopping check
				if early_stopper.check(iteration, best_ce):
					self._log(f"  Early stopping at iteration {iteration + 1}")
					break

				# Metal cleanup every 10 iterations
				if iteration % 10 == 9:
					self._cleanup_metal(iteration)

		finally:
			# Restore original adaptation config
			evaluator._adapt_config = original_adapt_config

		elapsed = _time.time() - start_time

		# Determine stop reason
		stop_reason = self._determine_stop_reason(shutdown_requested[0], early_stopper)
		if stop_reason is None:
			stop_reason = StopReason.MAX_ITERATIONS

		# Build population metrics for downstream phases
		population_metrics = [(e[0], e[1] if len(e) > 1 else 0.0) for e in evals]

		initial_ce = history[0][1] if history else best_ce
		improvement = ((initial_ce - best_ce) / initial_ce * 100) if initial_ce > 0 else 0.0

		self._log(f"\n{'─'*70}")
		self._log(f"  {self.name} Complete")
		self._log(f"  Best CE: {best_ce:.4f}  Acc: {best_acc:.2%}")
		self._log(f"  Improvement: {improvement:.2f}%")
		self._log(f"  Duration: {elapsed:.1f}s")
		self._log(f"{'─'*70}\n")

		return OptimizerResult(
			initial_genome=best_genome,
			best_genome=best_genome,
			initial_fitness=initial_ce,
			final_fitness=best_ce,
			improvement_percent=improvement,
			iterations_run=len(history),
			method_name=self.name,
			history=history,
			early_stopped=stop_reason in (StopReason.CONVERGENCE, StopReason.SHUTDOWN),
			stop_reason=stop_reason,
			final_population=population,
			population_metrics=population_metrics,
			initial_accuracy=0.0,
			final_accuracy=best_acc,
			final_threshold=cfg.initial_threshold,
		)

	def _configure_evaluator_adaptation(self, evaluator):
		"""Configure the evaluator's adaptation settings for this strategy's mode.

		Creates or modifies the evaluator's _adapt_config to enable only
		the adaptation mode needed by this strategy.
		"""
		from wnn.ram.architecture.bitwise_evaluator import AdaptationConfig as EvalAdaptConfig

		mode = self._config.adaptation_mode
		cfg = self._config

		# Create adaptation config with only the relevant mode enabled
		adapt = EvalAdaptConfig(
			synaptogenesis_enabled=(mode == "synaptogenesis"),
			neurogenesis_enabled=(mode == "neurogenesis"),
			axonogenesis_enabled=(mode == "axonogenesis"),
			min_bits=cfg.min_bits,
			max_bits=cfg.max_bits,
			min_neurons=cfg.min_neurons,
			# Use sensible defaults for adaptation parameters
			warmup_generations=0,  # No warmup — dedicated phase, always active
			cooldown_iterations=0,
			total_generations=cfg.iterations,
			passes_per_eval=1,
			stats_sample_size=10_000,
		)
		evaluator._adapt_config = adapt
