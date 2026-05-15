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

from wnn.ram.strategies.connectivity.adaptive_cluster import PhaseType

import threading


# =============================================================================
# Live Progress Observer
# =============================================================================

class LiveProgressObserver:
	"""Polls Rust evaluator for live progress and POSTs to dashboard.

	Runs in a daemon thread, reading evaluator.get_live_progress() every
	`interval` seconds and sending to the dashboard. Thread-safe: the Rust
	side uses Arc<RwLock> and releases the GIL during search.
	"""

	def __init__(self, evaluator, client, experiment_id: int, interval: float = 5.0):
		self._evaluator = evaluator
		self._client = client
		self._experiment_id = experiment_id
		self._interval = interval
		self._stop_event = threading.Event()
		self._thread = None

	def start(self):
		if not self._client or not self._experiment_id:
			return
		if not hasattr(self._evaluator, 'get_live_progress'):
			return

		def loop():
			while not self._stop_event.wait(self._interval):
				try:
					progress = self._evaluator.get_live_progress()
					if progress and self._client:
						self._client.post_live_progress(self._experiment_id, progress)
				except Exception:
					pass  # Observer must never crash the main thread

		self._thread = threading.Thread(target=loop, daemon=True)
		self._thread.start()

	def stop(self):
		self._stop_event.set()
		if self._thread:
			self._thread.join(timeout=2)
		# Send clear signal
		if self._client and self._experiment_id:
			try:
				self._client.clear_live_progress(self._experiment_id)
			except Exception:
				pass


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
	- Live progress observer for dashboard
	"""

	_shutdown_check: Optional[Callable[[], bool]]
	_log: Any  # Logger
	_dashboard_client: Any = None  # DashboardClient for live progress

	def set_dashboard_client(self, client) -> None:
		"""Set dashboard client for live progress reporting."""
		self._dashboard_client = client

	def _start_live_observer(self) -> Optional[LiveProgressObserver]:
		"""Start a live progress observer if dashboard client is available."""
		evaluator = getattr(self, '_cached_evaluator', None)
		client = self._dashboard_client
		experiment_id = getattr(self, '_tracker_experiment_id', None)
		if not evaluator:
			self._log("[LiveProgress] No evaluator, skipping observer")
		if not client:
			self._log("[LiveProgress] No dashboard client, skipping observer")
		if not experiment_id:
			self._log("[LiveProgress] No experiment_id, skipping observer")
		if evaluator and client and experiment_id:
			# Tell Rust which experiment this is for
			if hasattr(evaluator, 'set_experiment_context'):
				evaluator.set_experiment_context(experiment_id)
			# For BitwiseEvaluator/MultiStageEvaluator: reach the inner cache
			cache = getattr(evaluator, '_cache', None)
			if cache and hasattr(cache, 'set_experiment_context'):
				cache.set_experiment_context(experiment_id)
			observer = LiveProgressObserver(evaluator, client, experiment_id)
			observer.start()
			self._log(f"[LiveProgress] Observer started for experiment {experiment_id}")
			return observer
		return None

	def _stop_live_observer(self, observer: Optional[LiveProgressObserver]) -> None:
		"""Stop a live progress observer if one is active."""
		if observer:
			observer.stop()

	def _derive_phase_type(self) -> 'PhaseType':
		"""Derive PhaseType from optimize_* flags in ArchitectureConfig."""
		cfg = self._arch_config
		if cfg.optimize_connections and not cfg.optimize_bits and not cfg.optimize_neurons:
			return PhaseType.CONNECTIONS
		elif cfg.optimize_neurons and not cfg.optimize_bits:
			return PhaseType.NEURONS
		elif cfg.optimize_bits and not cfg.optimize_neurons:
			return PhaseType.BITS
		else:
			# Both bits+neurons or default: use Neurons phase (most general)
			return PhaseType.NEURONS

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

	def _compute_threshold(self, progress: float = 0.0) -> float:
		"""Compute progressive accuracy threshold at given progress [0, 1].

		Shared across GA, TS, and Adaptation strategies. Config must provide:
		initial_threshold, progressive_threshold, threshold_delta, min_accuracy.
		"""
		cfg = self._config
		start = cfg.initial_threshold if cfg.initial_threshold is not None else getattr(cfg, 'min_accuracy', 0.0)
		if not getattr(cfg, 'progressive_threshold', True):
			return start
		progress = max(0.0, min(1.0, progress))
		return start + progress * cfg.threshold_delta

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

		# Use fitness calculator to extract bests
		bests = self._fitness_calculator.bests(top_genomes, full_results)

		self._log.info(f"  Best by CE:       CE={bests.best_ce.metrics.ce:.4f}, Acc={bests.best_ce.metrics.acc:.4%}")
		self._log.info(f"  Best by Accuracy: CE={bests.best_acc.metrics.ce:.4f}, Acc={bests.best_acc.metrics.acc:.4%}")

		if bests.best_fitness.genome is bests.best_ce.genome:
			self._log.info(f"  Best by Fitness:  (same as Best by CE)")
		elif bests.best_fitness.genome is bests.best_acc.genome:
			self._log.info(f"  Best by Fitness:  (same as Best by Accuracy)")
		else:
			self._log.info(f"  Best by Fitness:  CE={bests.best_fitness.metrics.ce:.4f}, Acc={bests.best_fitness.metrics.acc:.4%}")

		top_k_ce = sum(r.ce for r in full_results) / len(full_results)
		top_k_acc = sum(r.acc for r in full_results) / len(full_results)
		self._log.info(f"  Top-{top_k} Mean:    CE={top_k_ce:.4f}, Acc={top_k_acc:.4%}")
		self._log.info("=" * 60)

		# Record phase results via tracker
		if self._tracker and self._tracker_experiment_id:
			try:
				self._tracker.record_phase_result(
					experiment_id=self._tracker_experiment_id,
					metric_type="best_ce",
					ce=bests.best_ce.metrics.ce,
					accuracy=bests.best_ce.metrics.acc,
					improvement_pct=(result.initial_fitness - bests.best_ce.metrics.ce) / result.initial_fitness * 100 if result.initial_fitness else 0.0,
				)
				self._tracker.record_phase_result(
					experiment_id=self._tracker_experiment_id,
					metric_type="best_acc",
					ce=bests.best_acc.metrics.ce,
					accuracy=bests.best_acc.metrics.acc,
					improvement_pct=(result.initial_fitness - bests.best_acc.metrics.ce) / result.initial_fitness * 100 if result.initial_fitness else 0.0,
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
		improvement_pct = (result.initial_fitness - bests.best_ce.metrics.ce) / result.initial_fitness * 100 if result.initial_fitness != 0 else 0.0
		return OptimizerResult(
			initial_genome=result.initial_genome,
			best_genome=bests.best_ce.genome,
			initial_fitness=result.initial_fitness,
			final_fitness=bests.best_ce.metrics.ce,
			improvement_percent=improvement_pct,
			iterations_run=result.iterations_run,
			method_name=result.method_name,
			history=result.history,
			early_stopped=result.early_stopped,
			stop_reason=result.stop_reason,
			final_population=result.final_population,
			population_metrics=result.population_metrics,
			initial_accuracy=result.initial_accuracy,
			final_accuracy=bests.best_acc.metrics.acc,
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
			offspring_2t = [(t[0], t[1]) for t in offspring]
			filter_result = ce_filter.apply(offspring_2t, key=lambda g, f: f)
			kept_ids = {id(g) for g, _ in filter_result.kept}
			offspring = [t for t in offspring if id(t[0]) in kept_ids]
		else:
			# HARMONIC_RANK or NORMALIZED: filter by fitness score
			offspring_metrics = [t[1] for t in offspring]
			fitness_scores = fitness_calculator.fitness(offspring_metrics)
			offspring_with_fitness = list(zip(offspring, fitness_scores))

			fitness_filter = PercentileFilter(
				percentile=cfg.fitness_percentile,
				mode=FilterMode.LOWER_IS_BETTER,
				metric_name=fitness_calculator.name,
			)
			filter_input = [(item, score) for item, score in offspring_with_fitness]
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
			if hasattr(genome, 'metrics') and genome.metrics is not None:
				gd['fitness'] = genome.metrics.to_dict()
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
				from wnn.ram.metrics import Metrics as _M
				if isinstance(gd['fitness'], dict):
					genome.metrics = _M.from_dict(gd['fitness'])
				else:
					f = gd['fitness']
					genome.metrics = _M(ce=f[0], acc=f[1], f1=f[2] if len(f) > 2 else None, fpr=f[3] if len(f) > 3 else None)
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
	# Cluster-level crossover ratio: 0.0 = all phase-specific, 1.0 = all cluster-level
	cluster_crossover_ratio: float = 0.0
	# Pool-and-shuffle crossover ratio: 0.0 = all uniform (2→2), 1.0 = all pool-and-shuffle (2→1)
	pool_shuffle_ratio: float = 0.0
	# Assortative mating ratio: 0.0 = random p2, 1.0 = always pick most similar p2 (NEAT-style)
	assortative_mating_ratio: float = 0.85


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
		cached_evaluator: Optional[Any] = None,  # BaseEvaluator for Rust search_offspring
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
		self._phase_type = self._derive_phase_type()

	@property
	def name(self) -> str:
		return "ArchitectureGA"

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> 'ClusterGenome':
		"""Phase-aware mutation dispatching to ClusterGenome.mutate()."""
		from wnn.ram.strategies.connectivity.adaptive_cluster import AdaptiveClusterConfig
		self._ensure_rng()
		cfg = self._arch_config
		mutation_config = AdaptiveClusterConfig(
			min_bits=cfg.min_bits, max_bits=cfg.max_bits,
			min_neurons=cfg.min_neurons, max_neurons=cfg.max_neurons,
		)
		tib = cfg.total_input_bits or 64
		return genome.mutate(self._phase_type, mutation_rate, mutation_config, tib, self._rng)

	def crossover_genomes(self, parent1: 'ClusterGenome', parent2: 'ClusterGenome') -> 'ClusterGenome':
		"""Phase-aware crossover dispatching to ClusterGenome.crossover()."""
		self._ensure_rng()
		return parent1.crossover(parent2, self._phase_type, self._rng)

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

			# Phase-aware mutation rates: each phase uses cfg.mutation_rate
			# for its own dimension, 0.0 for others
			if self._phase_type == PhaseType.NEURONS:
				bits_mutation_rate = 0.0
				neurons_mutation_rate = cfg.mutation_rate
			elif self._phase_type == PhaseType.BITS:
				bits_mutation_rate = cfg.mutation_rate
				neurons_mutation_rate = 0.0
			else:  # CONNECTIONS
				bits_mutation_rate = cfg.mutation_rate
				neurons_mutation_rate = 0.0

			# fitness_percentile controls selectivity: generate a larger pool,
			# rank by fitness, keep only the top fraction → return exactly n_needed.
			# e.g. percentile=0.75 → generate ceil(24/0.75)=32, rank, keep best 24.
			import math
			pct = cfg.fitness_percentile if cfg.fitness_percentile and 0 < cfg.fitness_percentile < 1.0 else None
			generate_count = math.ceil(n_needed / pct) if pct else n_needed

			# Convert (genome, Metrics) to (genome, ce_float) for Rust evaluator
			rust_population = [(t[0], t[1].ce) for t in population]

			# Pre-compute fitness scores so tournament selection uses the
			# same metric as elite selection (e.g. HarmonicRank), not raw CE
			fitness_scores = None
			if self._fitness_calculator is not None:
				pop_metrics_list = [t[1] for t in population]
				fitness_scores = self._fitness_calculator.fitness(pop_metrics_list)

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
				logger=self._log,
				generation=generation,
				total_generations=cfg.generations,
				return_best_n=True,
				mutable_clusters=arch_cfg.mutable_clusters,
				phase_type=int(self._phase_type),
				fitness_scores=fitness_scores,
				cluster_crossover_ratio=arch_cfg.cluster_crossover_ratio,
				pool_shuffle_ratio=arch_cfg.pool_shuffle_ratio,
				assortative_mating_ratio=arch_cfg.assortative_mating_ratio,
			)

			# Convert to (genome, Metrics) tuples, rank by fitness, return best n_needed
			offspring = [
				(g, g.metrics)
				for g in search_result.genomes
				if hasattr(g, 'metrics') and g.metrics is not None
			]

			if pct and len(offspring) > n_needed:
				scores = self._fitness_calculator.fitness([t[1] for t in offspring])
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
				population=[(t[0], t[1].ce) for t in population],
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
					population=[(t[0], t[1].ce) for t in population],
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
			# Use explicit train subset if provided (phased_search cycles through subsets),
			# otherwise pick randomly. Ensures different phases use different data.
			if 'train_subset_idx' in kwargs:
				self._phase_train_idx = kwargs.pop('train_subset_idx')
			else:
				self._ensure_rng()
				self._phase_train_idx = self._cached_evaluator.random_train_idx(self._rng)
			self._log.info(f"[{self.name}] Using train subset {self._phase_train_idx}")
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
					)
					mutation_rate = 0.3
					expanded = list(initial_population)
					# Dedup: track known fingerprints to avoid duplicate mutants
					known_fps = set()
					for g in initial_population:
						if hasattr(g, 'fingerprint'):
							known_fps.add(g.fingerprint())
					for i in range(need_count):
						seed = initial_population[i % seed_count]
						mutated = seed.mutate(
							self._phase_type, mutation_rate,
							mutation_config,
							self._cached_evaluator.total_input_bits,
							self._rng,
						)
						# Re-mutate if duplicate (up to 3 retries)
						if hasattr(mutated, 'fingerprint'):
							for _ in range(3):
								fp = mutated.fingerprint()
								if fp not in known_fps:
									break
								mutated = seed.mutate(
									self._phase_type, mutation_rate,
									mutation_config,
									self._cached_evaluator.total_input_bits,
									self._rng,
								)
							known_fps.add(mutated.fingerprint())
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

		# Start live progress observer — covers genesis, re-evaluation, and all generations
		observer = self._start_live_observer()
		try:
			# Delegate to unified loop (uses our _generate_offspring override)
			result = super().optimize(
				evaluate_fn=evaluate_fn,
				initial_genome=initial_genome,
				initial_population=initial_population,
				batch_evaluate_fn=batch_evaluate_fn,
				**kwargs,
			)
		finally:
			self._stop_live_observer(observer)

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
		cached_evaluator: Optional[Any] = None,  # BaseEvaluator for Rust search_neighbors
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
		self._phase_type = self._derive_phase_type()

	@property
	def name(self) -> str:
		return "ArchitectureTS"

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> tuple['ClusterGenome', Any]:
		"""Phase-aware mutation dispatching to ClusterGenome.mutate().

		Returns (new_genome, move_info) where move_info is a hash of the mutated
		architecture (for tabu tracking).
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import AdaptiveClusterConfig
		self._ensure_rng()
		cfg = self._arch_config
		mutation_config = AdaptiveClusterConfig(
			min_bits=cfg.min_bits, max_bits=cfg.max_bits,
			min_neurons=cfg.min_neurons, max_neurons=cfg.max_neurons,
		)
		tib = cfg.total_input_bits or 64
		mutant = genome.mutate(self._phase_type, mutation_rate, mutation_config, tib, self._rng)

		# Compute move info for tabu tracking: tuple of changed cluster indices
		if self._phase_type == PhaseType.NEURONS:
			changed = tuple(c for c in range(len(genome.neurons_per_cluster))
						   if genome.neurons_per_cluster[c] != mutant.neurons_per_cluster[c])
		elif self._phase_type == PhaseType.BITS:
			changed = tuple(c for c in range(len(genome.bits_per_neuron))
						   if genome.bits_per_neuron[c] != mutant.bits_per_neuron[c])
		else:
			# Connections phase: track which clusters had any connection change
			changed_clusters = []
			if genome.connections is not None and mutant.connections is not None:
				g_off = genome.cluster_neuron_offsets
				g_conn_off = genome.connection_offsets
				m_conn_off = mutant.connection_offsets
				for c in range(len(genome.neurons_per_cluster)):
					c_start = g_conn_off[g_off[c]]
					c_end = g_conn_off[g_off[c + 1]]
					m_start = m_conn_off[g_off[c]]
					m_end = m_conn_off[g_off[c + 1]]
					if (c_end - c_start != m_end - m_start or
						genome.connections[c_start:c_end] != mutant.connections[m_start:m_end]):
						changed_clusters.append(c)
			changed = tuple(changed_clusters)
		move = changed if changed else None
		return mutant, move

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

	def _compute_move_info(self, source: 'ClusterGenome', neighbor: 'ClusterGenome') -> Any:
		"""Compute tabu move info by comparing source and neighbor genomes."""
		if self._phase_type == PhaseType.NEURONS:
			changed = tuple(c for c in range(len(source.neurons_per_cluster))
						   if source.neurons_per_cluster[c] != neighbor.neurons_per_cluster[c])
		elif self._phase_type == PhaseType.BITS:
			changed = tuple(c for c in range(len(source.bits_per_neuron))
						   if source.bits_per_neuron[c] != neighbor.bits_per_neuron[c])
		else:
			# Connections phase: track which NEURONS had connection changes
			# (not clusters — single-cluster genomes only have cluster 0,
			# so cluster-level tracking makes every move tabu after the first)
			changed_neurons = []
			if source.connections is not None and neighbor.connections is not None:
				s_off = source.connection_offsets
				n_off = neighbor.connection_offsets
				num_neurons = len(source.bits_per_neuron)
				for n in range(num_neurons):
					s_conns = source.connections[s_off[n]:s_off[n + 1]]
					n_conns = neighbor.connections[n_off[n]:n_off[n + 1]]
					if s_conns != n_conns:
						changed_neurons.append(n)
			changed = tuple(changed_neurons)
		return changed if changed else None

	def _generate_neighbors(self, best_genome, n_neighbors, threshold, iteration, tabu_list):
		"""Generate neighbors via Rust search_neighbors or Python fallback."""
		if self._cached_evaluator is not None:
			cfg = self._config
			arch_cfg = self._arch_config
			evaluator = self._cached_evaluator

			# Phase-aware mutation rates: each phase uses cfg.mutation_rate
			# for its own dimension, 0.0 for others
			if self._phase_type == PhaseType.NEURONS:
				bits_mutation_rate = 0.0
				neurons_mutation_rate = cfg.mutation_rate
			elif self._phase_type == PhaseType.BITS:
				bits_mutation_rate = cfg.mutation_rate
				neurons_mutation_rate = 0.0
			else:  # CONNECTIONS
				bits_mutation_rate = cfg.mutation_rate
				neurons_mutation_rate = 0.0

			# fitness_percentile: generate larger pool, rank, keep best n_neighbors
			# Also over-generate to compensate for tabu filtering
			import math
			pct = cfg.fitness_percentile if cfg.fitness_percentile and 0 < cfg.fitness_percentile < 1.0 else None
			generate_count = math.ceil(n_neighbors / pct) if pct else n_neighbors
			# Over-generate by 50% to compensate for tabu filtering
			if tabu_list:
				generate_count = math.ceil(generate_count * 1.5)

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
				logger=self._log,
				generation=iteration,
				total_generations=cfg.iterations,
				return_best_n=True,
				mutable_clusters=arch_cfg.mutable_clusters,
				phase_type=int(self._phase_type),
			)

			# Convert to (genome, Metrics) tuples, rank by fitness, return best n_neighbors
			neighbors = [
				(g, g.metrics)
				for g in neighbors_raw
				if hasattr(g, 'metrics') and g.metrics is not None
			]

			# Post-filter: remove tabu neighbors
			if tabu_list:
				non_tabu = []
				for t in neighbors:
					move = self._compute_move_info(best_genome, t[0])
					if not self.is_tabu_move(move, tabu_list):
						non_tabu.append(t)
				filtered_count = len(neighbors) - len(non_tabu)
				if filtered_count > 0:
					self._log.debug(f"[{self.name}] Tabu filtered {filtered_count}/{len(neighbors)} neighbors")
				neighbors = non_tabu

			if pct and len(neighbors) > n_neighbors:
				scores = self._fitness_calculator.fitness([t[1] for t in neighbors])
				ranked = sorted(zip(neighbors, scores), key=lambda x: x[1])
				neighbors = [item for item, _ in ranked[:n_neighbors]]

			# Add best neighbor's move to tabu list
			if neighbors:
				best_neighbor = neighbors[0]  # Already ranked or first viable
				if len(neighbors) > 1:
					# Find best by fitness ranking
					best_ranked_neighbors = self._fitness_calculator.rank(
						[t[0] for t in neighbors], [t[1] for t in neighbors]
					)
					best_neighbor = next(
						t for t in neighbors if t[0] is best_ranked_neighbors[0][0]
					)
				move = self._compute_move_info(best_genome, best_neighbor[0])
				if move is not None:
					tabu_list.append(move)

			return neighbors

		# Fallback to Python single-path generation
		return super()._generate_neighbors(best_genome, n_neighbors, threshold, iteration, tabu_list)

	def _generate_neighbors_batch(self, sources, counts, threshold, iteration, tabu_list):
		"""Generate neighbors for multiple sources in a single Rust evaluation call.

		Returns list of offspring lists, one per source. Falls back to per-source
		_generate_neighbors if cached evaluator doesn't support batch search.
		"""
		evaluator = self._cached_evaluator
		if evaluator is None or not hasattr(evaluator, 'search_neighbors_batch'):
			return None  # Signal caller to fall back to per-source loop

		cfg = self._config
		arch_cfg = self._arch_config

		# Phase-aware mutation rates: each phase uses cfg.mutation_rate
		# for its own dimension, 0.0 for others
		if self._phase_type == PhaseType.NEURONS:
			bits_mutation_rate = 0.0
			neurons_mutation_rate = cfg.mutation_rate
		elif self._phase_type == PhaseType.BITS:
			bits_mutation_rate = cfg.mutation_rate
			neurons_mutation_rate = 0.0
		else:  # CONNECTIONS
			bits_mutation_rate = cfg.mutation_rate
			neurons_mutation_rate = 0.0

		import math
		pct = cfg.fitness_percentile if cfg.fitness_percentile and 0 < cfg.fitness_percentile < 1.0 else None

		# Build source list with inflated counts for fitness percentile + tabu filtering
		batch_sources = []
		for source, count in zip(sources, counts):
			gen_count = math.ceil(count / pct) if pct else count
			# Over-generate by 50% to compensate for tabu filtering
			if tabu_list:
				gen_count = math.ceil(gen_count * 1.5)
			batch_sources.append((source, gen_count))

		total_candidates = sum(gc for _, gc in batch_sources)
		self._log.info(
			f"[{self.name}] Batch searching {total_candidates} neighbors "
			f"from {len(sources)} sources"
		)

		def on_progress(batch_num, total_batches, done, total):
			self._log.info(
				f"[{self.name}] Evaluating {done}/{total} candidates "
				f"(sub-batch {batch_num}/{total_batches})"
			)

		if hasattr(evaluator, 'set_progress_callback'):
			evaluator.set_progress_callback(on_progress)
		try:
			results_by_source = evaluator.search_neighbors_batch(
				sources=batch_sources,
				max_attempts_multiplier=1,
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
				return_best_n=True,
				mutable_clusters=arch_cfg.mutable_clusters,
				phase_type=int(self._phase_type),
				logger=self._log,
				generation=iteration,
				total_generations=cfg.iterations,
			)
		finally:
			if hasattr(evaluator, 'set_progress_callback'):
				evaluator.set_progress_callback(None)

		# Convert to tuples, tabu-filter, and apply fitness percentile filtering per source
		all_offspring = []
		total_tabu_filtered = 0
		for (source, _), source_neighbors, target_count in zip(batch_sources, results_by_source, counts):
			neighbors = [
				(g, g.metrics)
				for g in source_neighbors
				if hasattr(g, 'metrics') and g.metrics is not None
			]

			# Post-filter: remove tabu neighbors
			if tabu_list:
				non_tabu = []
				for t in neighbors:
					move = self._compute_move_info(source, t[0])
					if not self.is_tabu_move(move, tabu_list):
						non_tabu.append(t)
				total_tabu_filtered += len(neighbors) - len(non_tabu)
				neighbors = non_tabu

			if pct and len(neighbors) > target_count:
				scores = self._fitness_calculator.fitness([t[1] for t in neighbors])
				ranked = sorted(zip(neighbors, scores), key=lambda x: x[1])
				neighbors = [item for item, _ in ranked[:target_count]]

			# Add best neighbor's move to tabu list
			if neighbors:
				best_neighbor = neighbors[0]
				if len(neighbors) > 1:
					best_ranked_neighbors = self._fitness_calculator.rank(
						[t[0] for t in neighbors], [t[1] for t in neighbors]
					)
					best_neighbor = next(
						t for t in neighbors if t[0] is best_ranked_neighbors[0][0]
					)
				move = self._compute_move_info(source, best_neighbor[0])
				if move is not None:
					tabu_list.append(move)

			all_offspring.append(neighbors)

		if total_tabu_filtered > 0:
			self._log.debug(f"[{self.name}] Tabu filtered {total_tabu_filtered} neighbors across all sources")

		return all_offspring

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
			# Use explicit train subset if provided (phased_search cycles through subsets),
			# otherwise pick randomly. Ensures different phases use different data.
			if 'train_subset_idx' in kwargs:
				self._phase_train_idx = kwargs.pop('train_subset_idx')
			else:
				self._ensure_rng()
				self._phase_train_idx = self._cached_evaluator.random_train_idx(self._rng)
			self._log.info(f"[{self.name}] Using train subset {self._phase_train_idx}")
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

		# Start live progress observer — covers initial eval, seeded neighbors, and all iterations
		observer = self._start_live_observer()
		try:
			# Delegate to unified loop (uses our _generate_neighbors override)
			result = super().optimize(
				initial_genome=initial_genome,
				initial_fitness=initial_fitness,
				evaluate_fn=evaluate_fn,
				initial_neighbors=initial_neighbors,
				batch_evaluate_fn=batch_evaluate_fn,
				**kwargs,
			)
		finally:
			self._stop_live_observer(observer)

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
	top_k: int = 15
	population_size: int = 50  # Total genomes in output population
	total_input_bits: Optional[int] = None
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	fitness_weight_f1: float = 0.0
	fitness_weight_fpr: float = 0.0
	grid_source: str = "random"  # "random" or "leaderboard"


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
		from wnn.ram.metrics import FitnessWeights
		calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weights=FitnessWeights(ce=cfg.fitness_weight_ce, acc=cfg.fitness_weight_acc,
								   f1=cfg.fitness_weight_f1, fpr=cfg.fitness_weight_fpr),
		)

		# Phase 1: Build config list from grid or leaderboard
		is_leaderboard = cfg.grid_source == "leaderboard"
		if is_leaderboard and initial_population:
			config_list = []  # [(neurons, bits, genome)]
			for genome in initial_population:
				n = genome.neurons_per_cluster[0] if genome.neurons_per_cluster else 0
				b = genome.bits_per_neuron[0] if genome.bits_per_neuron else 0
				config_list.append((n, b, genome))
			total_configs = len(config_list)
			self._log(f"\n{'='*70}")
			self._log(f"Grid Search — Leaderboard ({total_configs} genomes)")
			self._log(f"  source: leaderboard top-{total_configs}")
			self._log(f"  clusters: {cfg.num_clusters}")
			self._log(f"{'='*70}")
		else:
			config_list = []
			for neurons in cfg.neurons_grid:
				for bits in cfg.bits_grid:
					genome = self._create_genome(neurons, bits)
					config_list.append((neurons, bits, genome))
			total_configs = len(config_list)
			self._log(f"\n{'='*70}")
			self._log(f"Grid Search ({total_configs} configs)")
			self._log(f"  neurons: {cfg.neurons_grid}")
			self._log(f"  bits:    {cfg.bits_grid}")
			self._log(f"  clusters: {cfg.num_clusters}")
			self._log(f"{'='*70}")

		# Phase 2: Evaluate one config at a time for real-time dashboard progress
		# Fix train subset to a single index so all configs are compared fairly
		# (auto-advancing would evaluate each config on a different subset)
		grid_train_idx = kwargs.get('train_subset_idx', None)
		if grid_train_idx is None and self._batch_evaluator is not None and hasattr(self._batch_evaluator, 'random_train_idx'):
			grid_train_idx = self._batch_evaluator.random_train_idx(self._rng)
		if grid_train_idx is not None:
			self._log(f"  Using fixed train subset {grid_train_idx} for all {total_configs} configs")

		t0 = time.time()
		results = []
		best_ce_so_far = float('inf')
		best_acc_so_far = 0.0
		best_f1_so_far: Optional[float] = None
		best_fpr_so_far: Optional[float] = None

		# Concurrent grid_search: configs are independent — each is a single-
		# genome `evaluate_batch` call. The Rust accelerator releases the GIL
		# inside `evaluate_genomes_parallel_hybrid` via `py.allow_threads`,
		# so Python threads run truly in parallel. GPU dispatches serialize
		# at the Metal queue, but CPU train/eval + buffer alloc + export
		# overlap across threads.
		#
		# WNN_GRID_SEARCH_PARALLEL=N caps the concurrent worker count.
		# Default 4: empirical sweet spot on M4 Max (16 cores + 40-core GPU).
		# Set to 1 to disable concurrency.
		import os as _os
		from concurrent.futures import ThreadPoolExecutor as _TPE
		grid_max_workers = max(1, int(_os.environ.get('WNN_GRID_SEARCH_PARALLEL', '4')))

		def _eval_one_config(idx_neurons_bits_genome):
			idx, neurons, bits, genome = idx_neurons_bits_genome
			if self._shutdown_check and self._shutdown_check():
				return idx, neurons, bits, genome, None
			t_config = time.time()
			if self._batch_evaluator is not None:
				grid_evals = self._batch_evaluator.evaluate_batch(
					[genome], train_subset_idx=grid_train_idx,
				)
				grid_ev = grid_evals[0]
				ce, acc, bit_acc = grid_ev.ce, grid_ev.acc, grid_ev.bit_accuracy or 0.0
				f1_macro = grid_ev.f1
				fpr = grid_ev.fpr
			elif evaluate_fn is not None:
				ce = evaluate_fn(genome)
				acc, bit_acc = 0.0, 0.0
				f1_macro, fpr = None, None
			else:
				raise ValueError("GridSearchStrategy requires a batch_evaluator or evaluate_fn")
			config_elapsed = time.time() - t_config
			return idx, neurons, bits, genome, {
				"neurons": neurons, "bits": bits,
				"ce": ce, "accuracy": acc,
				"bit_accuracy": bit_acc, "f1_macro": f1_macro, "fpr": fpr,
				"elapsed_s": config_elapsed, "genome": genome,
			}

		indexed_configs = [(idx, n, b, g) for idx, (n, b, g) in enumerate(config_list)]

		# Stream results as they complete so the dashboard sees per-config
		# updates in real time. Use as_completed instead of pool.map to
		# avoid buffering all 48 results until the batch finishes.
		def _iter_results(configs_iter):
			"""Yield (idx, neurons, bits, genome, out) tuples in completion order."""
			if grid_max_workers > 1 and len(configs_iter) > 1:
				self._log(f"  Concurrent eval: {len(configs_iter)} configs × parallelism={grid_max_workers} (streaming)")
				from concurrent.futures import as_completed as _as_completed
				with _TPE(max_workers=grid_max_workers) as _pool:
					_futures = [_pool.submit(_eval_one_config, c) for c in configs_iter]
					for _fut in _as_completed(_futures):
						yield _fut.result()
			else:
				for c in configs_iter:
					yield _eval_one_config(c)

		# Process each result as it completes — log + update best_*_so_far +
		# write DB row immediately so the dashboard ticks in real time.
		# iteration_num in the DB tracks COMPLETION order; the per-config grid
		# position is still in the row data (neurons, bits).
		_completion_num = 0
		for _idx, neurons, bits, genome, out in _iter_results(indexed_configs):
			if out is None:
				self._log(f"  Shutdown requested at config {_idx}")
				break
			_completion_num += 1
			ce = out["ce"]; acc = out["accuracy"]
			f1_macro = out["f1_macro"]; fpr = out["fpr"]
			config_elapsed = out["elapsed_s"]
			self._log(f"  [{_completion_num}/{total_configs}] n={neurons:3d}, b={bits:2d}: "
					  f"CE={ce:.4f}  Acc={acc:.2%}  ({config_elapsed:.1f}s)")
			best_ce_so_far = min(best_ce_so_far, ce)
			best_acc_so_far = max(best_acc_so_far, acc)
			results.append(out)

			# Record each config as a separate iteration for real-time dashboard tracking.
			# iteration_num tracks COMPLETION order so the dashboard sees a
			# monotonically growing list. Grid position (neurons, bits) is in
			# the row body if reconstruction is needed.
			if self._tracker and self._tracker_experiment_id:
				try:
					avg_ce = sum(r["ce"] for r in results) / len(results)
					avg_acc = sum(r["accuracy"] for r in results) / len(results)
					cur_f1 = results[-1].get("f1_macro")
					cur_fpr = results[-1].get("fpr")
					if cur_f1 is not None and (best_f1_so_far is None or cur_f1 > best_f1_so_far):
						best_f1_so_far = cur_f1
					if cur_fpr is not None and (best_fpr_so_far is None or cur_fpr < best_fpr_so_far):
						best_fpr_so_far = cur_fpr
					iter_id = self._tracker.record_iteration(
						experiment_id=self._tracker_experiment_id,
						iteration_num=_completion_num,
						best_ce=best_ce_so_far,
						best_accuracy=best_acc_so_far,
						avg_ce=avg_ce,
						avg_accuracy=avg_acc,
						elapsed_secs=config_elapsed,
						candidates_total=total_configs,
						best_f1=best_f1_so_far,
						best_fpr=best_fpr_so_far,
					)
					if HAS_GENOME_TRACKING and iter_id:
						genome_config = self._genome_to_config(genome)
						if genome_config:
							genome_id = self._tracker.get_or_create_genome(
								self._tracker_experiment_id, genome_config
							)
							eval_id = self._tracker.record_genome_evaluation(
								iteration_id=iter_id,
								genome_id=genome_id,
								position=0,
								role=GenomeRole.INIT,
								ce=ce,
								accuracy=acc,
								fitness_score=None,
								f1_macro=f1_macro,
								fpr=fpr,
							)
							results[-1]["eval_id"] = eval_id
					self._tracker.update_experiment_progress(
						self._tracker_experiment_id,
						current_iteration=_completion_num,
						best_ce=best_ce_so_far,
						best_accuracy=best_acc_so_far,
					)
				except Exception as e:
					self._log(f"  Warning: tracker error: {e}")

		batch_elapsed = time.time() - t0
		self._log(f"  Total evaluation: {batch_elapsed:.1f}s, "
				  f"{batch_elapsed/len(results):.1f}s/config avg")

		if not results:
			raise ValueError("Grid search produced no results")

		# Phase 4: Rank by fitness
		from wnn.ram.metrics import Metrics as _M
		rank_metrics = [_M(ce=r["ce"], acc=r["accuracy"], f1=r.get("f1_macro"), fpr=r.get("fpr")) for r in results]
		fitness_scores = calculator.fitness(rank_metrics)
		for r, score in zip(results, fitness_scores):
			r["fitness"] = score
		results.sort(key=lambda r: r["fitness"])

		# Update fitness scores in DB for per-config genome evaluations
		if self._tracker and self._tracker_experiment_id:
			fitness_updates = [
				(r["eval_id"], r["fitness"])
				for r in results if "eval_id" in r
			]
			if fitness_updates:
				try:
					self._tracker.update_genome_evaluation_fitness_batch(fitness_updates)
				except Exception as e:
					self._log(f"  Warning: failed to update fitness scores: {e}")

		self._log(f"\n{'─'*70}")
		self._log(f"Grid Search Rankings (by {calculator.name}):")
		for i, r in enumerate(results):
			marker = " ★" if i < cfg.top_k else ""
			self._log(f"  {i+1:2d}. n={r['neurons']:3d}, b={r['bits']:2d}: "
					  f"CE={r['ce']:.4f}  Acc={r['accuracy']:.2%}  "
					  f"Fit={r['fitness']:.4f}{marker}")

		# Phase 5: Build output population with balanced representation.
		# Generate population_size * 1.1 genomes (10% extra for fitness trimming),
		# distributed evenly across top_k configs with extra going to top-ranked.
		best_result = results[0]
		best_genome = best_result["genome"]

		top_k = min(cfg.top_k, len(results))
		target_total = max(top_k, int(cfg.population_size * 1.1))
		base_per_config = target_total // top_k
		remainder = target_total - base_per_config * top_k
		genomes_per_config = []
		for i in range(top_k):
			extra = 1 if i < remainder else 0
			genomes_per_config.append(max(1, base_per_config + extra))

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
			from wnn.ram.metrics import Metrics as _Metrics
			output_population.append(r["genome"])
			population_metrics.append(_Metrics(ce=r["ce"], acc=r["accuracy"], f1=r.get("f1_macro"), fpr=r.get("fpr")))

			# Remaining: fresh random connections (need evaluation)
			for _ in range(n_genomes - 1):
				genome = self._create_genome(r["neurons"], r["bits"])
				new_genome_indices.append(len(output_population))
				output_population.append(genome)
				population_metrics.append(_Metrics(ce=0.0, acc=0.0))  # placeholder
				new_genomes.append(genome)

			self._log(f"  #{i+1:2d} n={r['neurons']:3d}, b={r['bits']:2d} (CE={r['ce']:.4f}) → {n_genomes} genomes (1 original + {n_genomes - 1} new)")

		self._log(f"\nPopulation: {len(output_population)} genomes ({top_k} original + {len(new_genomes)} new)")

		# Evaluate the NEW genomes — batched by shape (same n,b → same kernel grid).
		# Genomes spawned from the same top-K config share (n, b, num_clusters) →
		# can be dispatched as ng > 1 (single Metal kernel call). Across distinct
		# shape groups, dispatches run concurrently via the same thread pool used
		# in phase 4. Net effect: 3-4× fewer kernel launches (top_k=15, 3-4
		# spawns each → ~15 ng-grouped calls instead of ~45 ng=1 calls), plus
		# the surviving dispatch overhead overlaps across groups.
		num_seed_recorded = 0
		if new_genomes and self._batch_evaluator is not None:
			self._log(f"  Evaluating {len(new_genomes)} new genomes (batched by shape)...")
			t1 = time.time()
			expand_elapsed = 0.0

			# Group new_genomes by (neurons_per_cluster, bits_per_neuron) shape.
			# Connections differ within a group but the kernel grid is uniform.
			from collections import defaultdict as _defaultdict
			shape_groups: dict = _defaultdict(list)  # shape_key -> list[(idx_in_new, genome)]
			for _i, _g in enumerate(new_genomes):
				_shape = (tuple(_g.neurons_per_cluster), tuple(_g.bits_per_neuron))
				shape_groups[_shape].append((_i, _g))

			self._log(f"  {len(new_genomes)} new genomes → {len(shape_groups)} shape groups "
					  f"(avg ng={len(new_genomes)/max(1,len(shape_groups)):.1f}/group)")

			def _eval_group(items):
				# items = [(idx_in_new, genome), ...] — all share the same shape
				t_g = time.time()
				_genomes_only = [_g for _, _g in items]
				_evals = self._batch_evaluator.evaluate_batch(
					_genomes_only, train_subset_idx=grid_train_idx,
				)
				return items, _evals, time.time() - t_g

			# Stream each shape group's results as it completes (same dashboard
			# real-time concern as phase 4 — pool.map would buffer until the
			# whole batch finishes, hiding genomes from the dashboard).
			group_items_list = list(shape_groups.values())
			def _iter_groups(groups_iter):
				if grid_max_workers > 1 and len(groups_iter) > 1:
					from concurrent.futures import as_completed as _as_completed
					with _TPE(max_workers=grid_max_workers) as _pool:
						_futures = [_pool.submit(_eval_group, items) for items in groups_iter]
						for _fut in _as_completed(_futures):
							yield _fut.result()
				else:
					for items in groups_iter:
						yield _eval_group(items)

			# Process each group's results as it completes — per-genome log +
			# tracker writes immediately, dashboard ticks in real time.
			for items, evals_list, group_elapsed in _iter_groups(group_items_list):
				expand_elapsed += group_elapsed
				# Per-genome handling within this shape group
				for (idx_in_new, genome), ev in zip(items, evals_list):
					ce, acc = ev.ce, ev.acc
					pop_idx = new_genome_indices[idx_in_new]
					population_metrics[pop_idx] = _Metrics(ce=ce, acc=acc, f1=ev.f1, fpr=ev.fpr)
					g = output_population[pop_idx]
					neurons = g.neurons_per_cluster[0] if g.neurons_per_cluster else 0
					bits = g.bits_per_neuron[0] if g.bits_per_neuron else 0
					# Per-genome elapsed: amortize group time across members
					per_genome_elapsed = group_elapsed / max(1, len(items))
					self._log(f"  [{idx_in_new+1}/{len(new_genomes)}] n={neurons:3d}, b={bits:2d}: "
							  f"CE={ce:.4f}  Acc={acc:.2%}  ({per_genome_elapsed:.1f}s)")
					if ce < best_result["ce"]:
						best_result = {"ce": ce, "accuracy": acc, "genome": output_population[pop_idx]}
						best_genome = output_population[pop_idx]
					if self._tracker and self._tracker_experiment_id:
						try:
							best_ce_so_far = min(best_ce_so_far, ce)
							best_acc_so_far = max(best_acc_so_far, acc)
							seed_iter_num = len(results) + idx_in_new + 1
							ev_f1 = ev.f1
							ev_fpr = ev.fpr
							if ev_f1 is not None and (best_f1_so_far is None or ev_f1 > best_f1_so_far):
								best_f1_so_far = ev_f1
							if ev_fpr is not None and (best_fpr_so_far is None or ev_fpr < best_fpr_so_far):
								best_fpr_so_far = ev_fpr
							iter_id = self._tracker.record_iteration(
								experiment_id=self._tracker_experiment_id,
								iteration_num=seed_iter_num,
								best_ce=best_ce_so_far,
								best_accuracy=best_acc_so_far,
								avg_ce=ce,
								avg_accuracy=acc,
								elapsed_secs=per_genome_elapsed,
								candidates_total=len(new_genomes),
								best_f1=best_f1_so_far,
								best_fpr=best_fpr_so_far,
							)
							if HAS_GENOME_TRACKING and iter_id:
								genome_config = self._genome_to_config(genome)
								if genome_config:
									genome_id = self._tracker.get_or_create_genome(
										self._tracker_experiment_id, genome_config
									)
									self._tracker.record_genome_evaluation(
										iteration_id=iter_id,
										genome_id=genome_id,
										position=0,
										role=GenomeRole.INIT,
										ce=ce,
										accuracy=acc,
										fitness_score=None,
										f1_macro=ev_f1,
										fpr=ev_fpr,
									)
							self._tracker.update_experiment_progress(
								self._tracker_experiment_id,
								current_iteration=seed_iter_num,
								best_ce=best_ce_so_far,
								best_accuracy=best_acc_so_far,
							)
							num_seed_recorded += 1
						except Exception as e:
							self._log(f"  Warning: seed tracker error: {e}")
			batch_elapsed += expand_elapsed
			self._log(f"  New genome eval total: {expand_elapsed:.1f}s "
					  f"({expand_elapsed/len(new_genomes):.1f}s/genome avg, "
					  f"{len(shape_groups)} group dispatches)")

		# Phase 6: Rank full population by fitness and sort
		pop_fitness_scores = calculator.fitness(population_metrics)

		# Sort population by fitness score (lower = better)
		sorted_indices = sorted(range(len(output_population)), key=lambda i: pop_fitness_scores[i])
		output_population = [output_population[i] for i in sorted_indices]
		population_metrics = [population_metrics[i] for i in sorted_indices]
		pop_fitness_scores = [pop_fitness_scores[i] for i in sorted_indices]

		# Trim to population_size (balanced seeding may overshoot)
		if len(output_population) > cfg.population_size:
			dropped = len(output_population) - cfg.population_size
			self._log(f"  Trimming population: {len(output_population)} → {cfg.population_size} (dropped {dropped} weakest)")
			output_population = output_population[:cfg.population_size]
			population_metrics = population_metrics[:cfg.population_size]
			pop_fitness_scores = pop_fitness_scores[:cfg.population_size]

		# Three independent bests from potentially different genomes (recompute after trim)
		pop_bests = calculator.bests(output_population, population_metrics)
		best_genome = pop_bests.best_fitness.genome

		# Record final iteration with ALL population genomes sorted by fitness
		final_iter_num = len(results) + num_seed_recorded + 1  # After per-config + seed iterations
		if self._tracker and self._tracker_experiment_id:
			try:
				avg_ce = sum(m.ce for m in population_metrics) / len(population_metrics)
				avg_acc = sum(m.acc for m in population_metrics) / len(population_metrics)
				pm_f1s = [m.f1 for m in population_metrics if m.f1 is not None]
				pm_fprs = [m.fpr for m in population_metrics if m.fpr is not None]
				if pm_f1s:
					final_f1 = max(pm_f1s)
					if best_f1_so_far is None or final_f1 > best_f1_so_far:
						best_f1_so_far = final_f1
				if pm_fprs:
					final_fpr = min(pm_fprs)
					if best_fpr_so_far is None or final_fpr < best_fpr_so_far:
						best_fpr_so_far = final_fpr
				iter_id = self._tracker.record_iteration(
					experiment_id=self._tracker_experiment_id,
					iteration_num=final_iter_num,
					best_ce=pop_bests.best_ce.metrics.ce,
					best_accuracy=pop_bests.best_acc.metrics.acc,
					avg_ce=avg_ce,
					avg_accuracy=avg_acc,
					elapsed_secs=batch_elapsed,
					candidates_total=len(output_population),
					best_f1=best_f1_so_far,
					best_fpr=best_fpr_so_far,
				)
				if HAS_GENOME_TRACKING:
					for pos, (genome, m, fit) in enumerate(zip(output_population, population_metrics, pop_fitness_scores)):
						ce, acc = m.ce, m.acc
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
								f1_macro=m.f1,
								fpr=m.fpr,
							)
				self._tracker.update_experiment_progress(
					self._tracker_experiment_id,
					current_iteration=1,
					best_ce=pop_bests.best_ce.metrics.ce,
					best_accuracy=pop_bests.best_acc.metrics.acc,
				)
			except Exception as e:
				self._log(f"  Warning: tracker error: {e}")

		# Build OptimizerResult
		worst_ce = results[-1]["ce"]
		improvement = ((worst_ce - pop_bests.best_ce.metrics.ce) / worst_ce * 100) if worst_ce > 0 else 0.0

		return OptimizerResult(
			initial_genome=best_genome,
			best_genome=best_genome,
			initial_fitness=worst_ce,
			final_fitness=pop_bests.best_ce.metrics.ce,
			improvement_percent=improvement,
			iterations_run=1,
			method_name="GridSearch",
			history=[(1, pop_bests.best_ce.metrics.ce)],
			early_stopped=False,
			stop_reason=StopReason.MAX_ITERATIONS,
			final_population=output_population,
			population_metrics=population_metrics,
			initial_accuracy=results[-1]["accuracy"],
			final_accuracy=pop_bests.best_acc.metrics.acc,
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
	progressive_threshold: bool = True
	min_accuracy: float = 0.0
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	fitness_weight_f1: float = 0.0
	fitness_weight_fpr: float = 0.0
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
		self._rng = random.Random(self._seed)
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
		from wnn.ram.metrics import Metrics as _M

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

		from wnn.ram.metrics import FitnessWeights
		calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weights=FitnessWeights(ce=cfg.fitness_weight_ce, acc=cfg.fitness_weight_acc,
								   f1=cfg.fitness_weight_f1, fpr=cfg.fitness_weight_fpr),
		)

		# Configure evaluator's adaptation mode for this phase
		original_adapt_config = evaluator._adapt_config
		self._configure_evaluator_adaptation(evaluator)

		# Fix train subset for the entire phase so all iterations are comparable
		# (auto-advancing would evaluate each iteration on a different subset,
		# confusing early stopping and fitness tracking)
		phase_train_idx = kwargs.get('train_subset_idx', None)
		if phase_train_idx is None:
			phase_train_idx = evaluator.random_train_idx(self._rng)

		self._log(f"\n{'='*70}")
		self._log(f"  {self._phase_name}: {self.name}")
		self._log(f"{'='*70}")
		self._log(f"  Mode: {cfg.adaptation_mode}")
		self._log(f"  Population: {len(population)}")
		self._log(f"  Iterations: {cfg.iterations}")
		self._log(f"  Patience: {cfg.patience} (check every {cfg.check_interval})")
		self._log(f"  Fitness: {calculator.name}")
		self._log(f"  Train subset: {phase_train_idx}")
		start_threshold = self._compute_threshold(0.0)
		end_threshold = self._compute_threshold(min(1.0, cfg.iterations / cfg.threshold_reference))
		self._log(f"  Threshold: {start_threshold:.4%} → {end_threshold:.4%}")
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
		best_f1_global: Optional[float] = None
		best_fpr_global: Optional[float] = None
		evals = [(float('inf'), 0.0)] * len(population)
		fitness_scores = [float('inf')] * len(population)
		history = []
		shutdown_requested = [False]
		start_time = _time.time()

		try:
			for iteration in range(cfg.iterations):
				iter_start = _time.time()

				# Progressive threshold (shared with GA/TS via mixin)
				current_threshold = self._compute_threshold(iteration / cfg.threshold_reference)

				# Check shutdown
				if self._check_and_set_shutdown(shutdown_requested):
					self._log(f"  Shutdown requested at iteration {iteration}")
					break

				# Set generation for cosine annealing
				evaluator.set_generation(iteration, total_generations=cfg.iterations)

				# Evaluate all genomes (adaptation happens in-place via Rust)
				evals = evaluator.evaluate_batch(
					population,
					train_subset_idx=phase_train_idx,
					generation=iteration,
					total_generations=cfg.iterations,
				)

				# Convert evals to Metrics for fitness ranking
				eval_metrics = [e if isinstance(e, _M) else _M(ce=e.ce, acc=e.acc, f1=e.f1, fpr=e.fpr) for e in evals]

				# Compute fitness and sort
				fitness_scores = calculator.fitness(eval_metrics)
				sorted_indices = sorted(
					range(len(population)),
					key=lambda i: fitness_scores[i],
				)

				# Reorder population by fitness
				population = [population[i] for i in sorted_indices]
				evals = [evals[i] for i in sorted_indices]
				fitness_scores = [fitness_scores[i] for i in sorted_indices]

				# Update bests independently (CE and accuracy tracked separately, like GA/TS)
				pop_bests = calculator.bests(population, eval_metrics)
				if pop_bests.best_ce.metrics.ce < best_ce:
					best_ce = pop_bests.best_ce.metrics.ce
					best_genome = pop_bests.best_fitness.genome.clone()
				if pop_bests.best_acc.metrics.acc > best_acc:
					best_acc = pop_bests.best_acc.metrics.acc

				history.append((iteration + 1, best_ce))
				iter_elapsed = _time.time() - iter_start

				# Log progress
				avg_ce = sum(e.ce for e in evals) / len(evals)
				avg_acc = sum(e.acc for e in evals) / len(evals)
				self._log(
					f"  [{iteration+1:3d}/{cfg.iterations}] "
					f"best CE={best_ce:.4f} acc={best_acc:.2%} | "
					f"avg CE={avg_ce:.4f} acc={avg_acc:.2%} | "
					f"{iter_elapsed:.1f}s"
				)

				# Record iteration in tracker (with patience/threshold for dashboard)
				prev_best = early_stopper._prev_best if early_stopper._prev_best is not None else best_ce
				delta_previous = best_ce - prev_best
				if self._tracker and self._tracker_experiment_id:
					try:
						# Update running global best F1/FPR (like best_ce/best_acc)
						ev_f1s = [e.f1 for e in evals]
						ev_fprs = [e.fpr for e in evals]
						valid_f1s = [v for v in ev_f1s if v is not None]
						valid_fprs = [v for v in ev_fprs if v is not None]
						if valid_f1s:
							iter_best_f1 = max(valid_f1s)
							if best_f1_global is None or iter_best_f1 > best_f1_global:
								best_f1_global = iter_best_f1
						if valid_fprs:
							iter_best_fpr = min(valid_fprs)
							if best_fpr_global is None or iter_best_fpr < best_fpr_global:
								best_fpr_global = iter_best_fpr
						iter_id = self._tracker.record_iteration(
							experiment_id=self._tracker_experiment_id,
							iteration_num=iteration + 1,
							best_ce=best_ce,
							best_accuracy=best_acc,
							avg_ce=avg_ce,
							avg_accuracy=avg_acc,
							elapsed_secs=iter_elapsed,
							candidates_total=len(population),
							fitness_threshold=current_threshold,
							delta_previous=delta_previous,
							patience_counter=early_stopper._patience_counter,
							patience_max=cfg.patience,
							best_f1=best_f1_global,
							best_fpr=best_fpr_global,
						)
						# Record genome evaluations
						if HAS_GENOME_TRACKING:
							for pos, (genome, ev_m, fit) in enumerate(
								zip(population, evals, fitness_scores)
							):
								ce, acc = ev_m.ce, ev_m.acc
								ev_f1 = ev_m.f1
								ev_fpr = ev_m.fpr
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
										f1_macro=ev_f1,
										fpr=ev_fpr,
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
		population_metrics = [e if isinstance(e, _M) else _M(ce=e.ce, acc=e.acc, f1=e.f1, fpr=e.fpr) for e in evals]

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
			final_threshold=self._compute_threshold(min(1.0, (len(history) - 1) / cfg.threshold_reference) if history else 0.0),
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
