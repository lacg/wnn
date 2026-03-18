"""
Single Experiment Runner

Wraps a GA or TS optimization run as a self-contained experiment
with checkpoint saving and dashboard integration.
"""

import gzip
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

from enum import IntEnum

from wnn.ram.fitness import FitnessCalculatorType, FitnessCalculatorFactory
from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.experiments.phased_search import PhaseResult


class ExperimentType(IntEnum):
	"""How we optimize."""
	GA = 0              # Genetic Algorithm
	TS = 1              # Tabu Search
	GRID_SEARCH = 2     # Grid search over neuron × bit configurations
	NEUROGENESIS = 3    # Stats-guided neuron add/remove per cluster
	SYNAPTOGENESIS = 4  # Stats-guided connection prune/grow per neuron
	AXONOGENESIS = 5    # MI-guided connection rewiring
	LAMBDA_SWEEP = 6    # Unigram interpolation lambda sweep (eval-only)


class ClusterType(IntEnum):
	"""What cluster architecture to use."""
	TIERED = 0       # Existing RAMLM with tiered clusters (50K)
	BITWISE = 1      # BitwiseRAMLM with per-bit clusters (16)
	MULTI_STAGE = 2  # Multi-stage: group prediction + within-group prediction


class StageMode(IntEnum):
	"""How Stage 2 receives Stage 1's output in two-stage architecture."""
	SELECTOR = 0       # Stage 1 picks which sub-model (K separate models)
	INPUT_CONCAT = 1   # Stage 1 output bits appended to Stage 2 input


@dataclass
class ExperimentConfig:
	"""Configuration for a single experiment (GA or TS optimization)."""

	name: str
	experiment_type: ExperimentType

	# What to optimize
	optimize_bits: bool = False
	optimize_neurons: bool = False
	optimize_connections: bool = False

	# GA-specific
	generations: int = 250
	population_size: int = 50

	# TS-specific
	iterations: int = 250
	neighbors_per_iter: int = 50

	# Shared
	patience: int = 3
	check_interval: int = 10
	threshold_delta: float = 0.01
	threshold_reference: int = 1000
	threshold_start: float = 0.0  # Starting threshold (fraction, e.g. 0.01 = 1%)

	# Architecture bounds
	min_bits: int = 4
	max_bits: int = 20
	min_neurons: int = 1
	max_neurons: int = 15
	default_bits: int = 8
	default_neurons: int = 5

	# Tier configuration: (count, neurons, bits) or (count, neurons, bits, optimize)
	tier_config: Optional[list[tuple]] = None
	optimize_tier0_only: bool = False

	# Population handling
	seed_only: bool = False
	fresh_population: bool = False

	# Fitness filtering
	fitness_percentile: Optional[float] = None

	# Random seed
	seed: Optional[int] = None

	# Cluster architecture
	cluster_type: ClusterType = ClusterType.TIERED

	# BitwiseRAMLM-specific (only used when cluster_type=BITWISE)
	bitwise_neurons_per_cluster: int = 200
	bitwise_bits_per_neuron: int = 16

	# Bitwise architecture bounds (for per-cluster GA/TS optimization)
	bitwise_min_bits: Optional[int] = None
	bitwise_max_bits: Optional[int] = None
	bitwise_min_neurons: Optional[int] = None
	bitwise_max_neurons: Optional[int] = None

	# Fitness calculator settings
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.NORMALIZED
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	fitness_weight_f1: float = 0.0
	fitness_weight_fpr: float = 0.0
	min_accuracy_floor: float = 0.0

	# Cluster-level crossover ratio: 0.0 = all phase-specific, 1.0 = all cluster-level
	cluster_crossover_ratio: float = 0.0

	# Pool-and-shuffle crossover ratio: 0.0 = all uniform (2→2), 1.0 = all pool-and-shuffle (2→1)
	pool_shuffle_ratio: float = 0.0

	# Assortative mating ratio: 0.0 = random p2, 1.0 = always pick most similar p2 (NEAT-style)
	assortative_mating_ratio: float = 0.85

	# Grid search configuration (only used when experiment_type=GRID_SEARCH)
	neurons_grid: Optional[list[int]] = None   # e.g. [50, 100, 150, 200]
	bits_grid: Optional[list[int]] = None       # e.g. [14, 16, 18, 20]
	grid_top_k: int = 16                        # Top-K configs to seed population (default: all 4×4 grid)

	# Multi-stage configuration (only used when cluster_type=MULTI_STAGE)
	num_stages: int = 2
	stage_k: Optional[list[int]] = None          # K per stage; product >= vocab_size (default: [256, 256])
	stage_cluster_type: Optional[list[str]] = None  # per-stage arch (default: ["bitwise", "bitwise"])
	stage_mode: Optional[list[int]] = None       # StageMode values between stages (len = num_stages-1)
	target_stage: int = 0                        # 0-indexed: which stage this experiment optimizes
	frozen_genomes: Optional[list[Optional[dict]]] = None  # serialized genome per completed stage

	# Lambda sweep configuration (only used when experiment_type=LAMBDA_SWEEP)
	lambda_values: Optional[list[float]] = None     # Unigram lambda values to sweep
	bigram_lambda_values: Optional[list[float]] = None  # KN bigram lambda values to sweep (2D grid if both set)
	s0_checkpoint_id: Optional[int] = None           # Checkpoint ID for stage 0 genome
	s1_checkpoint_id: Optional[int] = None           # Checkpoint ID for stage 1 genome
	genome_type: str = "best_ce"                     # Which genome from checkpoint (best_ce, best_acc, best_fitness)

	def to_dict(self) -> dict[str, Any]:
		"""Convert to dictionary for JSON serialization.

		Note: tier_config is converted to a string format for API compatibility
		(Rust backend expects Option<String>, not a list of tuples).
		"""
		result = asdict(self)
		# Convert tier_config list to string format for API compatibility
		# Format: "100,15,20;400,10,12;rest,5,8" (3-part) or
		#         "100,15,20,true;400,10,12,false;rest,5,8,false" (4-part)
		if result.get("tier_config") is not None:
			tier_parts = []
			for tier in result["tier_config"]:
				count_str = "rest" if tier[0] is None else str(tier[0])
				if len(tier) >= 4:
					# 4-part format with optimize flag
					optimize_str = "true" if tier[3] else "false"
					tier_parts.append(f"{count_str},{tier[1]},{tier[2]},{optimize_str}")
				else:
					# Legacy 3-part format
					tier_parts.append(f"{count_str},{tier[1]},{tier[2]}")
			result["tier_config"] = ";".join(tier_parts)
		# Convert enums to string/int for JSON
		if "experiment_type" in result:
			result["experiment_type"] = result["experiment_type"].name.lower()
		if "cluster_type" in result:
			result["cluster_type"] = result["cluster_type"].name.lower()
		if "fitness_calculator_type" in result:
			result["fitness_calculator_type"] = result["fitness_calculator_type"].name.lower()
		return result

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
		"""Create from dictionary."""
		data = data.copy()  # Don't modify the original
		# Convert experiment_type string back to enum
		if "experiment_type" in data and isinstance(data["experiment_type"], str):
			try:
				data["experiment_type"] = ExperimentType[data["experiment_type"].upper()]
			except KeyError:
				data["experiment_type"] = ExperimentType.GA
		# Convert cluster_type string back to enum
		if "cluster_type" in data and isinstance(data["cluster_type"], str):
			try:
				data["cluster_type"] = ClusterType[data["cluster_type"].upper()]
			except KeyError:
				data["cluster_type"] = ClusterType.TIERED
		# Convert fitness_calculator_type string back to enum
		if "fitness_calculator_type" in data and isinstance(data["fitness_calculator_type"], str):
			try:
				data["fitness_calculator_type"] = FitnessCalculatorType[data["fitness_calculator_type"].upper()]
			except KeyError:
				data["fitness_calculator_type"] = FitnessCalculatorType.NORMALIZED
		# Remove legacy two-stage fields if present
		for legacy_key in ("num_token_groups", "stage2_mode", "frozen_stage1_genome"):
			data.pop(legacy_key, None)
		return cls(**data)


@dataclass
class ExperimentResult:
	"""Result from running an experiment."""

	experiment_name: str
	strategy_type: str
	initial_fitness: Optional[float]
	final_fitness: float
	final_accuracy: Optional[float]
	improvement_percent: float
	iterations_run: int
	best_genome: ClusterGenome
	final_population: Optional[list[ClusterGenome]]
	final_threshold: Optional[float]
	elapsed_seconds: float
	checkpoint_path: Optional[str] = None
	was_shutdown: bool = False  # True if stopped due to external shutdown request
	population_metrics: Optional[list[tuple[float, float]]] = None  # Cached (ce, acc) per genome in final_population
	validation_f1: Optional[float] = None     # F1-macro from final validation (Rust)
	validation_fpr: Optional[float] = None    # FPR from final validation (Rust)
	validation_acc: Optional[float] = None    # Accuracy from final validation (Rust)

	def to_phase_result(self) -> PhaseResult:
		"""Convert to PhaseResult for compatibility."""
		return PhaseResult(
			phase_name=self.experiment_name,
			strategy_type=self.strategy_type,
			final_fitness=self.final_fitness,
			final_accuracy=self.final_accuracy,
			iterations_run=self.iterations_run,
			best_genome=self.best_genome,
			final_population=self.final_population,
			final_threshold=self.final_threshold,
			initial_fitness=self.initial_fitness,
		)


class Experiment:
	"""
	Single experiment runner for GA or TS optimization.

	Wraps the optimizer strategy with:
	- Checkpoint saving to specified directory
	- Dashboard API integration (optional)
	- Clean result encapsulation

	Example usage:
		config = ExperimentConfig(
			name="Phase 1a: GA Neurons",
			experiment_type=ExperimentType.GA,
			optimize_neurons=True,
			generations=250,
		)

		experiment = Experiment(
			config=config,
			evaluator=cached_evaluator,
			logger=log_fn,
			checkpoint_dir=Path("checkpoints"),
		)

		result = experiment.run(
			initial_genome=seed_genome,
			initial_population=seed_population,
		)

		print(f"Best CE: {result.final_fitness:.4f}")
	"""

	def __init__(
		self,
		config: ExperimentConfig,
		evaluator: Any,  # TieredEvaluator or BitwiseEvaluator or MultiStageEvaluator
		logger: Callable[[str], None],
		checkpoint_dir: Optional[Path] = None,
		dashboard_client: Optional[Any] = None,
		experiment_id: Optional[int] = None,
		tracker: Optional[Any] = None,  # ExperimentTracker for V2 tracking
		flow_id: Optional[int] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
		full_evaluator: Optional[Any] = None,  # Separate evaluator for validation (uses validation set)
	):
		"""
		Initialize experiment.

		Args:
			config: Experiment configuration
			evaluator: BaseEvaluator instance for genome evaluation
			logger: Logging function
			checkpoint_dir: Directory for saving checkpoints
			dashboard_client: Optional DashboardClient for API integration
			experiment_id: Optional experiment ID for dashboard integration
			tracker: Optional V2 tracker for direct database writes
			flow_id: Optional flow ID for V2 tracking
			shutdown_check: Optional callable that returns True if shutdown requested
			full_evaluator: Separate evaluator for validation (trains on full train, evals on validation set)
		"""
		self.config = config
		self.evaluator = evaluator
		self.full_evaluator = full_evaluator
		self.log = logger
		self.checkpoint_dir = checkpoint_dir
		self.dashboard_client = dashboard_client
		self.experiment_id = experiment_id
		self.tracker = tracker
		self.flow_id = flow_id
		self.shutdown_check = shutdown_check

		# Derived properties
		self.vocab_size = evaluator.vocab_size
		self.total_input_bits = evaluator.total_input_bits

	def _get_optimizable_clusters(self, tier_config: Optional[list[tuple]]) -> Optional[list[int]]:
		"""Get list of cluster indices that can be mutated based on tier optimize flags.

		Args:
			tier_config: List of tier tuples, optionally with optimize flag as 4th element.

		Returns:
			List of optimizable cluster indices, or None if no tier_config or all tiers optimizable.
		"""
		if not tier_config:
			return None

		# Check if any tier has optimize=False
		has_optimize_flags = any(len(t) >= 4 for t in tier_config)
		if not has_optimize_flags:
			return None  # All tiers optimizable by default

		optimizable = []
		cluster_idx = 0

		for tier in tier_config:
			count = tier[0]
			optimize = tier[3] if len(tier) > 3 else True

			if count is None:
				count = self.vocab_size - cluster_idx

			actual_count = min(count, self.vocab_size - cluster_idx)

			if optimize:
				optimizable.extend(range(cluster_idx, cluster_idx + actual_count))

			cluster_idx += actual_count

			if cluster_idx >= self.vocab_size:
				break

		return optimizable if optimizable else None

	def run(
		self,
		initial_genome: Optional[ClusterGenome] = None,
		initial_fitness: Optional[float] = None,
		initial_population: Optional[list[ClusterGenome]] = None,
		initial_threshold: Optional[float] = None,
		tracker_experiment_id: Optional[int] = None,
		initial_evals: Optional[list[tuple[float, float]]] = None,
		train_subset_idx: Optional[int] = None,
	) -> ExperimentResult:
		"""
		Run the experiment.

		Args:
			initial_genome: Optional starting genome
			initial_fitness: Fitness of initial genome (required for TS)
			initial_population: Population to seed from
			initial_threshold: Starting accuracy threshold
			tracker_experiment_id: V2 experiment ID for tracker (if using V2 tracking)
			initial_evals: Cached eval results from previous phase (avoids re-evaluation for INIT validation)
			train_subset_idx: Which train subset to use (cycled per phase to avoid subset bias)

		Returns:
			ExperimentResult with optimization results
		"""
		cfg = self.config
		start_time = time.time()

		self.log("")
		self.log(f"{'='*60}")
		self.log(f"  {cfg.name}")
		self.log(f"{'='*60}")
		self.log(f"  Type: {cfg.experiment_type.name}")
		self.log(f"  Cluster: {cfg.cluster_type.name}")
		self.log(f"  Optimize: bits={cfg.optimize_bits}, neurons={cfg.optimize_neurons}, connections={cfg.optimize_connections}")
		if train_subset_idx is not None:
			self.log(f"  Train subset: {train_subset_idx} (of {self.evaluator.num_parts})")
		if initial_genome:
			self.log(f"  Starting from: {initial_genome}")
		if initial_population:
			self.log(f"  Seeding from {len(initial_population)} genomes")
		self.log("")

		# Determine strategy type
		is_grid_search = cfg.experiment_type == ExperimentType.GRID_SEARCH
		is_ga = cfg.experiment_type == ExperimentType.GA
		is_adaptation = cfg.experiment_type in (
			ExperimentType.NEUROGENESIS,
			ExperimentType.SYNAPTOGENESIS,
			ExperimentType.AXONOGENESIS,
		)
		adaptation_type_map = {
			ExperimentType.NEUROGENESIS: OptimizerStrategyType.ARCHITECTURE_NEUROGENESIS,
			ExperimentType.SYNAPTOGENESIS: OptimizerStrategyType.ARCHITECTURE_SYNAPTOGENESIS,
			ExperimentType.AXONOGENESIS: OptimizerStrategyType.ARCHITECTURE_AXONOGENESIS,
		}
		if is_grid_search:
			strategy_type = OptimizerStrategyType.ARCHITECTURE_GRID_SEARCH
		elif is_adaptation:
			strategy_type = adaptation_type_map[cfg.experiment_type]
		elif is_ga:
			strategy_type = OptimizerStrategyType.ARCHITECTURE_GA
		else:
			strategy_type = OptimizerStrategyType.ARCHITECTURE_TS

		# Determine num_clusters based on cluster type
		if cfg.cluster_type == ClusterType.MULTI_STAGE:
			# MultiStageEvaluator.num_clusters is stage-aware (reads from target_stage)
			num_clusters = self.evaluator.num_clusters
			stage_label = f"S{cfg.target_stage}"
			self.log(f"  Multi-Stage ({stage_label}): {num_clusters} clusters")
		elif cfg.cluster_type == ClusterType.BITWISE:
			from wnn.ram.core.RAMClusterLayer import bits_needed
			num_clusters = bits_needed(self.vocab_size)
			self.log(f"  Bitwise: {num_clusters} clusters ({cfg.bitwise_neurons_per_cluster} neurons, {cfg.bitwise_bits_per_neuron} bits)")
		else:
			num_clusters = self.vocab_size

		# Build strategy kwargs
		strategy_kwargs = {
			"strategy_type": strategy_type,
			"num_clusters": num_clusters,
			"optimize_bits": cfg.optimize_bits,
			"optimize_neurons": cfg.optimize_neurons,
			"optimize_connections": cfg.optimize_connections,
			"default_bits": cfg.default_bits,
			"default_neurons": cfg.default_neurons,
			"total_input_bits": self.total_input_bits,
			"batch_evaluator": self.evaluator,
			"logger": self.log,
			"patience": cfg.patience,
			"check_interval": cfg.check_interval,
			"threshold_delta": cfg.threshold_delta,
			"threshold_reference": cfg.threshold_reference,
			"initial_threshold": initial_threshold if initial_threshold is not None else (cfg.threshold_start if cfg.threshold_start > 0 else None),
			"fitness_percentile": cfg.fitness_percentile,
			"seed": cfg.seed,
			"min_bits": cfg.bitwise_min_bits if cfg.bitwise_min_bits is not None else cfg.min_bits,
			"max_bits": cfg.bitwise_max_bits if cfg.bitwise_max_bits is not None else cfg.max_bits,
			"min_neurons": cfg.bitwise_min_neurons if cfg.bitwise_min_neurons is not None else cfg.min_neurons,
			"max_neurons": cfg.bitwise_max_neurons if cfg.bitwise_max_neurons is not None else cfg.max_neurons,
			# Fitness calculator settings
			"fitness_calculator_type": cfg.fitness_calculator_type,
			"fitness_weight_ce": cfg.fitness_weight_ce,
			"fitness_weight_acc": cfg.fitness_weight_acc,
			"fitness_weight_f1": cfg.fitness_weight_f1,
			"fitness_weight_fpr": cfg.fitness_weight_fpr,
			"min_accuracy_floor": cfg.min_accuracy_floor if cfg.min_accuracy_floor > 0 else None,
		}

		# Per-tier optimization: determine which clusters are optimizable
		# Check for per-tier optimize flags in tier_config first
		optimizable = self._get_optimizable_clusters(cfg.tier_config)
		if optimizable is not None and len(optimizable) < self.vocab_size:
			strategy_kwargs["mutable_clusters"] = optimizable
			self.log(f"  Per-tier optimization: mutating {len(optimizable)} of {self.vocab_size} clusters")
		elif cfg.optimize_tier0_only and cfg.tier_config:
			# Legacy fallback: tier0-only mode
			tier0_clusters = cfg.tier_config[0][0] or self.vocab_size
			strategy_kwargs["mutable_clusters"] = list(range(tier0_clusters))
			self.log(f"  Tier0-only mode: mutating first {tier0_clusters} clusters")

		# Cluster crossover ratio
		if cfg.cluster_crossover_ratio > 0:
			strategy_kwargs["cluster_crossover_ratio"] = cfg.cluster_crossover_ratio

		# Pool-and-shuffle crossover ratio
		if cfg.pool_shuffle_ratio > 0:
			strategy_kwargs["pool_shuffle_ratio"] = cfg.pool_shuffle_ratio

		# Assortative mating ratio (NEAT-style)
		if cfg.assortative_mating_ratio > 0:
			strategy_kwargs["assortative_mating_ratio"] = cfg.assortative_mating_ratio

		# Type-specific kwargs
		if is_grid_search:
			strategy_kwargs["neurons_grid"] = cfg.neurons_grid
			strategy_kwargs["bits_grid"] = cfg.bits_grid
			strategy_kwargs["grid_top_k"] = cfg.grid_top_k
			strategy_kwargs["population_size"] = cfg.population_size
		elif is_adaptation:
			strategy_kwargs["iterations"] = cfg.iterations
			strategy_kwargs["population_size"] = cfg.population_size
			strategy_kwargs["phase_name"] = cfg.name
		elif is_ga:
			strategy_kwargs["generations"] = cfg.generations
			strategy_kwargs["population_size"] = cfg.population_size
			strategy_kwargs["phase_name"] = cfg.name
			strategy_kwargs["seed_only"] = cfg.seed_only
			strategy_kwargs["fresh_population"] = cfg.fresh_population
		else:
			strategy_kwargs["iterations"] = cfg.iterations
			strategy_kwargs["neighbors_per_iter"] = cfg.neighbors_per_iter
			strategy_kwargs["total_neighbors_size"] = cfg.population_size

		# Pass shutdown check if available
		if self.shutdown_check:
			strategy_kwargs["shutdown_check"] = self.shutdown_check

		# Create strategy
		strategy = OptimizerStrategyFactory.create(**strategy_kwargs)

		# V2 tracking: set tracker on strategy for iteration/genome recording
		# (experiment is created by flow.py, not here)
		if self.tracker and tracker_experiment_id:
			try:
				# Set tracker on strategy - pass experiment_id for iteration recording
				if hasattr(strategy, 'set_tracker'):
					strategy.set_tracker(self.tracker, tracker_experiment_id, tracker_experiment_id)
				self.log(f"  V2 tracking: experiment_id={tracker_experiment_id}")
			except Exception as e:
				self.log(f"  Warning: Failed to set up V2 tracking: {e}")

		# Pass dashboard client for live progress reporting
		if self.dashboard_client and hasattr(strategy, 'set_dashboard_client'):
			strategy.set_dashboard_client(self.dashboard_client)

		# Create fitness calculator for validation genome selection
		fitness_calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weight_ce=cfg.fitness_weight_ce,
			weight_acc=cfg.fitness_weight_acc,
			weight_f1=cfg.fitness_weight_f1,
			weight_fpr=cfg.fitness_weight_fpr,
		)

		# Run INIT validation on seed population
		# Use cached evals from previous phase if available (avoids redundant re-evaluation)
		seed_pop = initial_population or ([initial_genome] if initial_genome else None)
		if seed_pop:
			if initial_evals and len(initial_evals) == len(seed_pop):
				self.log(f"  Using cached metrics for INIT validation ({len(seed_pop)} genomes)")
				init_evals = initial_evals
			else:
				self.log(f"  Evaluating initial population for validation selection ({len(seed_pop)} genomes)...")
				init_evals = self.evaluator.evaluate_batch(seed_pop)
				initial_evals = [(r.ce, r.acc) for r in init_evals]  # Reuse in strategy (plain tuples)
			self._run_validation(
				population=seed_pop,
				evals=init_evals,
				validation_point='init',
				experiment_id=tracker_experiment_id or self.experiment_id,
				flow_id=self.flow_id,
				fitness_calculator=fitness_calculator,
			)

		# Mark experiment as running via dashboard API
		if self.dashboard_client and self.experiment_id:
			try:
				self.dashboard_client.experiment_started(self.experiment_id)
				self.log(f"  Dashboard: experiment {self.experiment_id} marked as running")
			except Exception as e:
				self.log(f"  Warning: Failed to mark experiment as running: {e}")

		# Run optimization with exception handling for phase status
		result = None
		was_shutdown = False
		try:
			# Unified dispatch: all strategies accept the same signature.
			# GA/TS use OptimizationTemplate.optimize() (template method).
			# GridSearch/Adaptation have their own optimize() with **kwargs.
			seed_pop = initial_population or ([initial_genome] if initial_genome else None)
			opt_kwargs = {
				"evaluate_fn": None,
				"initial_genome": initial_genome,
				"initial_population": seed_pop,
				"initial_fitness": initial_fitness,
				"initial_evals": initial_evals,
				"batch_evaluate_fn": None,
				"initial_neighbors": initial_population,  # TS uses this for neighbor seeding
			}
			if train_subset_idx is not None:
				opt_kwargs["train_subset_idx"] = train_subset_idx
			result = strategy.optimize(**opt_kwargs)

			# Check if shutdown was requested (needed for phase status update)
			from wnn.ram.strategies.connectivity.generic_strategies import StopReason
			was_shutdown = result.stop_reason == StopReason.SHUTDOWN if result.stop_reason else False

			# Run FINAL validation (skip if shutdown was requested)
			val_results = {}
			if not was_shutdown and result.final_population:
				# Use cached metrics from the optimizer if available (avoids redundant re-eval)
				if result.population_metrics:
					final_evals = result.population_metrics
				else:
					final_evals = self.evaluator.evaluate_batch(result.final_population)
				val_results = self._run_validation(
					population=result.final_population,
					evals=final_evals,
					validation_point='final',
					experiment_id=tracker_experiment_id or self.experiment_id,
					flow_id=self.flow_id,
					fitness_calculator=fitness_calculator,
				)

			# V2 tracking: update experiment status based on whether shutdown was requested
			if self.tracker and tracker_experiment_id:
				try:
					from wnn.ram.experiments.tracker import TrackerStatus
					exp_status = TrackerStatus.CANCELLED if was_shutdown else TrackerStatus.COMPLETED
					self.tracker.update_experiment_status(tracker_experiment_id, exp_status)
					self.tracker.update_experiment_progress(
						tracker_experiment_id,
						current_iteration=result.iterations_run,
						best_ce=result.final_fitness,
						best_accuracy=result.final_accuracy,
					)
				except Exception as e:
					self.log(f"  Warning: Failed to update V2 experiment status: {e}")

			# Dashboard API: update experiment status (for HTTP/WebSocket clients)
			if self.dashboard_client and self.experiment_id:
				try:
					if was_shutdown:
						self.dashboard_client.update_experiment(self.experiment_id, status="cancelled")
						self.log(f"  Updated dashboard experiment {self.experiment_id} status to cancelled")
					else:
						self.dashboard_client.experiment_completed(
							self.experiment_id,
							best_ce=result.final_fitness,
							best_accuracy=result.final_accuracy,
						)
						self.log(f"  Updated dashboard experiment {self.experiment_id} status to completed")
				except Exception as e:
					self.log(f"  Warning: Failed to update dashboard experiment status: {e}")

		except Exception as e:
			# Mark experiment as failed on exception
			if self.tracker and tracker_experiment_id:
				try:
					from wnn.ram.experiments.tracker import TrackerStatus
					self.tracker.update_experiment_status(tracker_experiment_id, TrackerStatus.FAILED)
				except Exception:
					pass
			# Also update via dashboard API
			if self.dashboard_client and self.experiment_id:
				try:
					self.dashboard_client.update_experiment(self.experiment_id, status="failed")
				except Exception:
					pass
			raise  # Re-raise the original exception

		elapsed = time.time() - start_time

		self.log("")
		self.log(f"{cfg.name} Result:")
		self.log(f"  Best CE: {result.final_fitness:.4f}")
		if result.final_accuracy:
			self.log(f"  Best Accuracy: {result.final_accuracy:.2%}")
		self.log(f"  Iterations: {result.iterations_run}")
		self.log(f"  Duration: {elapsed:.1f}s")

			# Calculate improvement
		improvement = 0.0
		if result.initial_fitness and result.initial_fitness > 0:
			improvement = (result.initial_fitness - result.final_fitness) / result.initial_fitness * 100

		# Extract validation metrics for best genome (used by flow for IDS Results)
		best_val = val_results.get('best_ce') or val_results.get('best_fitness') or {} if val_results else {}

		# Create result
		exp_result = ExperimentResult(
			experiment_name=cfg.name,
			strategy_type="GA" if is_ga else "TS",
			initial_fitness=result.initial_fitness,
			final_fitness=result.final_fitness,
			final_accuracy=result.final_accuracy,
			improvement_percent=improvement,
			iterations_run=result.iterations_run,
			best_genome=result.best_genome,
			final_population=result.final_population,
			final_threshold=result.final_threshold,
			elapsed_seconds=elapsed,
			was_shutdown=was_shutdown,
			population_metrics=result.population_metrics,
			validation_f1=best_val.get('f1'),
			validation_fpr=best_val.get('fpr'),
			validation_acc=best_val.get('acc'),
		)

		# Save checkpoint
		if self.checkpoint_dir:
			checkpoint_path = self._save_checkpoint(exp_result)
			exp_result.checkpoint_path = checkpoint_path

		return exp_result

	def _save_checkpoint(self, result: ExperimentResult) -> str:
		"""Save checkpoint to disk and optionally register with dashboard."""
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

		# Generate filename
		safe_name = result.experiment_name.lower().replace(" ", "_").replace(":", "")
		filename = f"{safe_name}.json.gz"
		filepath = self.checkpoint_dir / filename

		# Convert to PhaseResult for serialization
		phase_result = result.to_phase_result()

		# Save compressed
		data = {
			"phase_result": phase_result.serialize(),
			"_metadata": {
				"elapsed_seconds": result.elapsed_seconds,
				"improvement_percent": result.improvement_percent,
			},
		}
		# Save population metrics for cross-experiment reuse (avoids re-evaluation on resume)
		if result.population_metrics is not None:
			data["population_metrics"] = result.population_metrics

		# Save best_ce and best_acc genomes for combined validation (future-proofing)
		if result.final_population and result.population_metrics:
			try:
				metrics = result.population_metrics
				# Best CE = minimum CE
				best_ce_idx = min(range(len(metrics)), key=lambda i: metrics[i][0])
				data["best_ce_genome"] = result.final_population[best_ce_idx].serialize()
				# Best ACC = maximum accuracy
				best_acc_idx = max(range(len(metrics)), key=lambda i: metrics[i][1])
				data["best_acc_genome"] = result.final_population[best_acc_idx].serialize()
			except Exception:
				pass  # Non-critical, skip silently

		with gzip.open(filepath, 'wt', encoding='utf-8') as f:
			json.dump(data, f, separators=(',', ':'))

		self.log(f"  Checkpoint saved: {filepath}")

		# Register with dashboard if client available
		if self.dashboard_client and self.experiment_id:
			try:
				genome_stats = None
				if result.best_genome:
					stats = result.best_genome.stats()
					genome_stats = {
						"num_clusters": stats.get("num_clusters", 0),
						"total_neurons": stats.get("total_neurons", 0),
						"total_connections": stats.get("total_connections", 0),
						"bits_range": (stats.get("min_bits", 0), stats.get("max_bits", 0)),
						"neurons_range": (stats.get("min_neurons", 0), stats.get("max_neurons", 0)),
					}
					# Add per-tier stats if tier_config is available
					if self.config.tier_config:
						tier_stats = result.best_genome.compute_tier_stats(self.config.tier_config)
						genome_stats["tier_stats"] = tier_stats

				checkpoint_id = self.dashboard_client.checkpoint_created(
					experiment_id=self.experiment_id,
					file_path=str(filepath),
					name=result.experiment_name,
					final_fitness=result.final_fitness,
					final_accuracy=result.final_accuracy,
					iterations_run=result.iterations_run,
					genome_stats=genome_stats,
					is_final=True,
					checkpoint_type="experiment_end",  # Simplified model - no phases
				)
				self.log(f"  Registered checkpoint {checkpoint_id} for experiment {self.experiment_id}")
			except Exception as e:
				self.log(f"  Warning: Failed to register checkpoint with dashboard: {e}")

		return str(filepath)

	def _compute_genome_hash(self, genome: ClusterGenome) -> str:
		"""
		Compute a unique hash for a genome based on its configuration.

		The hash is based on bits_per_neuron, neurons_per_cluster, and connections.
		Two genomes with the same hash will produce identical results.
		"""
		import hashlib

		# Create a string representation of the genome's defining characteristics
		parts = [
			",".join(str(b) for b in genome.bits_per_neuron),
			",".join(str(n) for n in genome.neurons_per_cluster),
		]
		if genome.connections is not None:
			parts.append(",".join(str(c) for c in genome.connections))

		config_str = "|".join(parts)
		return hashlib.sha256(config_str.encode()).hexdigest()[:16]

	def _select_validation_genomes(
		self,
		genomes: list[ClusterGenome],
		evals: list[tuple[float, float]],
		fitness_calculator: Optional[Any] = None,
	) -> list[tuple[ClusterGenome, str, float, float]]:
		"""
		Select genomes for validation: best CE, best Acc, best Fitness.

		Uses FitnessCalculator.bests() for correct, consistent selection
		across all three metrics. Always returns all three types, even if
		they refer to the same genome — deduplication happens via genome_hash
		caching in _run_validation.

		Args:
			genomes: Population of genomes
			evals: List of (ce, acc) tuples from training evaluation
			fitness_calculator: Optional fitness calculator for combined fitness ranking

		Returns:
			List of (genome, label, ce, acc) tuples
		"""
		if not genomes or not evals:
			return []

		# Extract metrics from evals — preserve F1/FPR for IDS fitness calculators
		# evals can be: plain tuples (ce, acc [, f1, fpr]), EvalResult objects, or 2-tuples
		def _extract(e):
			"""Extract (ce, acc, f1_macro, fpr) from various eval formats."""
			ce = e[0] if not hasattr(e, 'ce') else e.ce
			acc = e[1] if not hasattr(e, 'accuracy') else e.acc
			f1 = None
			fpr = None
			if hasattr(e, 'f1_macro'):
				f1 = e.f1
			elif isinstance(e, (tuple, list)) and len(e) > 2:
				f1 = e[2]
			if hasattr(e, 'fpr'):
				fpr = e.fpr
			elif isinstance(e, (tuple, list)) and len(e) > 3:
				fpr = e[3]
			return (ce, acc, f1, fpr)

		extracted = [_extract(e) for e in evals]

		# Use bests() for consistent selection across all three metrics
		if fitness_calculator is not None:
			try:
				# Build 5-element tuples so IDS fitness calculators can use F1/FPR
				population = [(g, ce, acc, f1, fpr) for g, (ce, acc, f1, fpr) in zip(genomes, extracted)]
				pop_bests = fitness_calculator.bests(population)
				return [
					(pop_bests.best_ce.genome, 'best_ce', pop_bests.best_ce.ce, pop_bests.best_ce.acc),
					(pop_bests.best_acc.genome, 'best_acc', pop_bests.best_acc.ce, pop_bests.best_acc.acc),
					(pop_bests.best_fitness.genome, 'best_fitness', pop_bests.best_fitness.ce, pop_bests.best_fitness.acc),
				]
			except Exception:
				pass

		# Fallback without fitness calculator: just CE and Acc
		genome_evals = list(zip(genomes, extracted))
		by_ce = sorted(genome_evals, key=lambda x: x[1][0])
		by_acc = sorted(genome_evals, key=lambda x: -x[1][1])

		best_ce_genome, (best_ce_ce, best_ce_acc, _, _) = by_ce[0]
		best_acc_genome, (best_acc_ce, best_acc_acc, _, _) = by_acc[0]

		return [
			(best_ce_genome, 'best_ce', best_ce_ce, best_ce_acc),
			(best_acc_genome, 'best_acc', best_acc_ce, best_acc_acc),
		]

	def _run_validation(
		self,
		population: list[ClusterGenome],
		evals: list[tuple[float, float]],
		validation_point: str,  # 'init' or 'final'
		experiment_id: int,
		flow_id: Optional[int] = None,
		fitness_calculator: Optional[Any] = None,
	) -> dict:
		"""
		Run full validation on selected genomes from population.

		For each selected genome (best_ce, best_acc, best_fitness):
		1. Check if genome_hash already exists in summaries (via dashboard API)
		2. If found: skip validation, use cached values
		3. If not found: run full validation (train full + eval full)
		4. Store result via dashboard API

		This deduplication means:
		- Init of experiment N reuses final validation from experiment N-1
		- Only genuinely new genomes trigger expensive full validation

		Args:
			population: List of genomes to select from
			evals: Training evaluation results (ce, acc) for each genome
			validation_point: 'init' or 'final'
			experiment_id: Experiment ID for storing summaries
			flow_id: Optional flow ID for organizing summaries
			fitness_calculator: Optional fitness calculator for ranking
		"""
		self.log("")
		self.log("=" * 60)
		self.log(f"  {validation_point.upper()} VALIDATION (Full Dataset)")
		self.log("=" * 60)

		results = {}

		if not population or not evals:
			self.log("  No population to validate")
			return results

		try:
			# Select 1-3 best genomes
			selected = self._select_validation_genomes(population, evals, fitness_calculator)

			if not selected:
				self.log("  No genomes selected for validation")
				return results

			self.log(f"  Selected {len(selected)} genome(s): {[label for _, label, _, _ in selected]}")

			# Process each selected genome
			for genome, genome_type, train_ce, train_acc in selected:
				genome_hash = self._compute_genome_hash(genome)

				# Check if already validated
				cached = None
				if self.dashboard_client:
					try:
						cached = self.dashboard_client.check_cached_validation(genome_hash)
					except Exception:
						pass

				# Initialize IDS metrics
				f1 = None
				fpr_val = None
				cached_threshold_metadata = None
				if cached is not None:
					result = cached
					ce, acc = result[0], result[1]
					# Extract cached IDS metrics (f1_macro, fpr) if available
					f1 = result[2] if len(result) > 2 else None
					fpr_val = result[3] if len(result) > 3 else None
					cached_threshold_metadata = result[4] if len(result) > 4 else None
					if f1 is not None:
						self.log(f"  {genome_type}: CE={ce:.4f}, Acc={acc:.4%}, F1={f1:.4%}, FPR={fpr_val:.4%} (cached)")
					else:
						self.log(f"  {genome_type}: CE={ce:.4f}, Acc={acc:.4%} (cached)")
				else:
					# Run full validation (use full_evaluator if available — validates against held-out set)
					val_evaluator = self.full_evaluator or self.evaluator
					self.log(f"  {genome_type}: Running full validation...")
					full_results = val_evaluator.evaluate_batch_full([genome])
					result = full_results[0]
					ce, acc = result[0], result[1]
					# Extract IDS metrics if available
					f1 = getattr(result, 'f1_macro', None)
					fpr_val = getattr(result, 'fpr', None)
					if f1 is not None:
						self.log(f"  {genome_type}: CE={ce:.4f}, Acc={acc:.4%}, F1={f1:.4%}, FPR={fpr_val:.4%} (validated)")
					else:
						self.log(f"  {genome_type}: CE={ce:.4f}, Acc={acc:.4%} (validated)")

				# Three-threshold validation for single-cluster IDS
				threshold_metadata = None
				val_evaluator = self.full_evaluator or self.evaluator
				is_single_cluster = (
					hasattr(val_evaluator, '_single_cluster') and val_evaluator._single_cluster
				)

				# Use cached threshold_metadata if available (avoids re-running expensive 4-threshold eval)
				if cached_threshold_metadata is not None and is_single_cluster:
					threshold_metadata = cached_threshold_metadata
					self.log(f"    Thresholds: (cached from prior validation)")

				elif is_single_cluster and f1 is not None and cached is None:
					threshold_metadata = {}
					train_threshold = genome.threshold  # from train-calibrated run above

					# 1. Train-calibrated (already computed — override=None calibrates on train scores)
					threshold_metadata['train_cal'] = {
						'f1': f1, 'fpr': fpr_val, 'acc': acc, 'threshold': train_threshold,
					}
					self.log(f"    Train-cal:   F1={f1:.4%}, FPR={fpr_val:.4%}, Acc={acc:.4%}, t={train_threshold:.4f}")

					# 2. Fixed 0.5 — distribution-agnostic baseline
					fixed_results = val_evaluator.evaluate_batch_full([genome], override_threshold=0.5)
					fr = fixed_results[0]
					threshold_metadata['fixed_05'] = {
						'f1': fr.f1, 'fpr': fr.fpr, 'acc': fr.acc,
					}
					self.log(f"    Fixed 0.5:   F1={fr.f1:.4%}, FPR={fr.fpr:.4%}, Acc={fr.acc:.4%}")

					# 3. Test-calibrated: find threshold on holdout, apply to test
					# 3a: override=-1.0 sweeps eval data (43K holdout) for optimal threshold
					self.evaluator.evaluate_batch_full([genome], override_threshold=-1.0)
					holdout_threshold = genome.threshold  # threshold found on holdout

					# 3b: Apply holdout threshold to full validation (175K train → 82K test)
					test_cal_results = val_evaluator.evaluate_batch_full(
						[genome], override_threshold=holdout_threshold,
					)
					tcr = test_cal_results[0]
					threshold_metadata['test_cal'] = {
						'f1': tcr.f1, 'fpr': tcr.fpr, 'acc': tcr.acc, 'threshold': holdout_threshold,
					}
					self.log(f"    Test-cal:    F1={tcr.f1:.4%}, FPR={tcr.fpr:.4%}, Acc={tcr.acc:.4%}, t={holdout_threshold:.4f}")

					# 4. Validation-calibrated (oracle): sweep 82K test for optimal threshold
					# This is the upper bound — in production, you'd update thresholds
					# periodically using recent labeled data (similar effect)
					oracle_results = val_evaluator.evaluate_batch_full([genome], override_threshold=-1.0)
					or_ = oracle_results[0]
					oracle_threshold = genome.threshold
					threshold_metadata['val_cal'] = {
						'f1': or_.f1, 'fpr': or_.fpr, 'acc': or_.acc, 'threshold': oracle_threshold,
					}
					self.log(f"    Val-cal:     F1={or_.f1:.4%}, FPR={or_.fpr:.4%}, Acc={or_.acc:.4%}, t={oracle_threshold:.4f} (oracle)")

					# 5. Platt scaling: fit sigmoid calibration on holdout scores, apply to test
					# Learns P(attack) = sigmoid(a*score + b) from holdout data,
					# absorbing distribution shift + class imbalance between train/test
					try:
						# score_examples trains on full train, returns raw scores for eval set
						holdout_scores = self.evaluator.score_examples(genome)
						holdout_labels_list = self.evaluator._y_test
						if holdout_scores and holdout_labels_list and len(holdout_scores) == len(holdout_labels_list):
							import math

							# Fit Platt scaling: minimize NLL for sigmoid(a*score + b) vs labels
							scores = holdout_scores
							labels = holdout_labels_list
							n = len(scores)
							n_pos = sum(1 for l in labels if l == 1)
							n_neg = n - n_pos

							# Initialize with reasonable values
							a = 1.0
							b = 0.0

							# Newton's method (Platt's algorithm)
							# Target: t_i = (N+ + 1) / (N+ + 2) for positives, 1/(N- + 2) for negatives
							t_pos = (n_pos + 1) / (n_pos + 2)
							t_neg = 1.0 / (n_neg + 2)

							for _ in range(100):  # Newton iterations
								g_a = 0.0  # gradient w.r.t. a
								g_b = 0.0  # gradient w.r.t. b
								h_aa = 0.0  # Hessian
								h_ab = 0.0
								h_bb = 0.0

								for i in range(n):
									fval = a * scores[i] + b
									# Numerically stable sigmoid
									if fval >= 0:
										p = 1.0 / (1.0 + math.exp(-fval))
									else:
										ef = math.exp(fval)
										p = ef / (1.0 + ef)
									p = max(1e-15, min(1 - 1e-15, p))

									t = t_pos if labels[i] == 1 else t_neg
									d = p - t
									g_a += d * scores[i]
									g_b += d
									w = p * (1 - p)
									h_aa += w * scores[i] * scores[i]
									h_ab += w * scores[i]
									h_bb += w

								# Regularize Hessian
								h_aa += 1e-6
								h_bb += 1e-6
								det = h_aa * h_bb - h_ab * h_ab
								if abs(det) < 1e-12:
									break
								da = -(h_bb * g_a - h_ab * g_b) / det
								db = -(h_aa * g_b - h_ab * g_a) / det
								a += da
								b += db
								if abs(da) < 1e-8 and abs(db) < 1e-8:
									break

							# Apply Platt threshold to test: sigmoid(a*score + b) > 0.5
							# This is equivalent to: score > -b/a (when a > 0)
							platt_threshold = -b / a if abs(a) > 1e-10 else 0.5

							platt_results = val_evaluator.evaluate_batch_full(
								[genome], override_threshold=platt_threshold,
							)
							pr = platt_results[0]
							threshold_metadata['platt'] = {
								'f1': pr.f1, 'fpr': pr.fpr, 'acc': pr.acc,
								'threshold': platt_threshold, 'a': a, 'b': b,
							}
							self.log(f"    Platt:       F1={pr.f1:.4%}, FPR={pr.fpr:.4%}, Acc={pr.acc:.4%}, t={platt_threshold:.4f} (a={a:.4f}, b={b:.4f})")
					except Exception as e:
						self.log(f"    Platt:       skipped ({e})")

					# 6. Beta calibration: logit(P) = a*ln(s) - b*ln(1-s) + c
					# Designed for bounded [0,1] scores — handles bimodal WNN distributions
					# (Kull et al. 2017, AISTATS)
					try:
						if holdout_scores and holdout_labels_list:
							import math
							scores = holdout_scores
							labels = holdout_labels_list
							n = len(scores)
							n_pos = sum(1 for l in labels if l == 1)
							n_neg = n - n_pos
							t_pos = (n_pos + 1) / (n_pos + 2)
							t_neg = 1.0 / (n_neg + 2)

							# Transform scores: x1 = ln(s), x2 = -ln(1-s)
							# Clamp to avoid log(0)
							EPS = 1e-10
							x1 = [math.log(max(s, EPS)) for s in scores]
							x2 = [-math.log(max(1 - s, EPS)) for s in scores]

							# Fit logistic regression on (x1, x2) → same Newton method as Platt
							a, b, c = 1.0, 1.0, 0.0
							for _ in range(100):
								g_a = g_b = g_c = 0.0
								h_aa = h_ab = h_ac = h_bb = h_bc = h_cc = 0.0
								for i in range(n):
									fval = a * x1[i] + b * x2[i] + c
									if fval >= 0:
										p = 1.0 / (1.0 + math.exp(-fval))
									else:
										ef = math.exp(fval)
										p = ef / (1.0 + ef)
									p = max(1e-15, min(1 - 1e-15, p))
									t = t_pos if labels[i] == 1 else t_neg
									d = p - t
									g_a += d * x1[i]; g_b += d * x2[i]; g_c += d
									w = p * (1 - p)
									h_aa += w * x1[i]**2; h_ab += w * x1[i] * x2[i]; h_ac += w * x1[i]
									h_bb += w * x2[i]**2; h_bc += w * x2[i]; h_cc += w

								# 3x3 Newton step (solve H * delta = -g)
								h_aa += 1e-6; h_bb += 1e-6; h_cc += 1e-6
								# Use simple iteration instead of full 3x3 inverse
								a -= 0.1 * g_a / h_aa
								b -= 0.1 * g_b / h_bb
								c -= 0.1 * g_c / h_cc
								if abs(g_a / max(h_aa, 1e-10)) < 1e-6 and abs(g_b / max(h_bb, 1e-10)) < 1e-6:
									break

							# Find threshold: a*ln(s) + b*(-ln(1-s)) + c = 0
							# Binary search for the score where this equals 0
							lo, hi = 0.001, 0.999
							for _ in range(100):
								mid = (lo + hi) / 2
								val = a * math.log(mid) + b * (-math.log(1 - mid)) + c
								if val < 0:
									lo = mid
								else:
									hi = mid
							beta_threshold = (lo + hi) / 2

							beta_results = val_evaluator.evaluate_batch_full(
								[genome], override_threshold=beta_threshold,
							)
							br = beta_results[0]
							threshold_metadata['beta'] = {
								'f1': br.f1, 'fpr': br.fpr, 'acc': br.acc,
								'threshold': beta_threshold, 'a': a, 'b': b, 'c': c,
							}
							self.log(f"    Beta:        F1={br.f1:.4%}, FPR={br.fpr:.4%}, Acc={br.acc:.4%}, t={beta_threshold:.4f} (a={a:.3f}, b={b:.3f}, c={c:.3f})")
					except Exception as e:
						self.log(f"    Beta:        skipped ({e})")

					# 7. Empirical table: P(attack|score) computed per discrete score value
					# Zero assumptions — uses WNN's discrete score nature directly
					try:
						if holdout_scores and holdout_labels_list:
							scores = holdout_scores
							labels = holdout_labels_list

							# Build empirical P(attack|score) table from holdout
							from collections import defaultdict
							score_counts = defaultdict(lambda: [0, 0])  # score → [n_normal, n_attack]
							for s, l in zip(scores, labels):
								score_counts[round(s, 6)][l] += 1

							# Find threshold: highest score where P(attack) < 0.5
							sorted_scores = sorted(score_counts.keys())
							empirical_threshold = 0.5
							for s in sorted_scores:
								counts = score_counts[s]
								total = counts[0] + counts[1]
								if total > 0:
									p_attack = counts[1] / total
									if p_attack >= 0.5:
										empirical_threshold = s
										break

							emp_results = val_evaluator.evaluate_batch_full(
								[genome], override_threshold=empirical_threshold,
							)
							er = emp_results[0]
							n_bins = len(sorted_scores)
							threshold_metadata['empirical'] = {
								'f1': er.f1, 'fpr': er.fpr, 'acc': er.acc,
								'threshold': empirical_threshold, 'n_bins': n_bins,
							}
							self.log(f"    Empirical:   F1={er.f1:.4%}, FPR={er.fpr:.4%}, Acc={er.acc:.4%}, t={empirical_threshold:.4f} ({n_bins} bins)")
					except Exception as e:
						self.log(f"    Empirical:   skipped ({e})")

					# 8. Empirical-cumulative: sort-based F1 maximization on holdout scores
					# Unlike per-bin empirical, this considers ALL examples above each threshold
					try:
						if holdout_scores and holdout_labels_list:
							import numpy as np
							scores_arr = np.array(holdout_scores)
							labels_arr = np.array(holdout_labels_list)

							# Sort by score descending — sweep threshold from high to low
							sorted_idx = np.argsort(-scores_arr)
							sorted_labels = labels_arr[sorted_idx]
							sorted_scores = scores_arr[sorted_idx]

							cum_tp = np.cumsum(sorted_labels)           # attacks above threshold
							cum_fp = np.cumsum(1 - sorted_labels)       # normals above threshold
							total_pos = labels_arr.sum()
							total_neg = len(labels_arr) - total_pos

							prec = np.where(cum_tp + cum_fp > 0, cum_tp / (cum_tp + cum_fp), 0)
							rec = np.where(total_pos > 0, cum_tp / total_pos, 0)
							f1_arr = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)

							best_idx = int(np.argmax(f1_arr))
							emp_cum_threshold = float(sorted_scores[best_idx])

							ec_results = val_evaluator.evaluate_batch_full(
								[genome], override_threshold=emp_cum_threshold,
							)
							ec = ec_results[0]
							threshold_metadata['empirical_cumulative'] = {
								'f1': ec.f1, 'fpr': ec.fpr, 'acc': ec.acc,
								'threshold': emp_cum_threshold,
							}
							self.log(f"    Emp-cumul:   F1={ec.f1:.4%}, FPR={ec.fpr:.4%}, Acc={ec.acc:.4%}, t={emp_cum_threshold:.4f}")
					except Exception as e:
						self.log(f"    Emp-cumul:   skipped ({e})")

					# Use test-calibrated as primary metric (most honest)
					f1 = tcr.f1
					fpr_val = tcr.fpr
					acc = tcr.acc

				# Collect results keyed by genome_type
				results[genome_type] = {'ce': ce, 'acc': acc, 'f1': f1, 'fpr': fpr_val}

				# Always store summary via dashboard API (even if cached)
				# This ensures each (experiment_id, validation_point, genome_type) has a record
				if self.dashboard_client and self.experiment_id:
					try:
						self.dashboard_client.create_validation_summary(
							experiment_id=self.experiment_id,
							validation_point=validation_point,
							genome_type=genome_type,
							genome_hash=genome_hash,
							ce=ce,
							accuracy=acc,
							flow_id=flow_id,
							f1_macro=f1,
							fpr=fpr_val,
							threshold_metadata=json.dumps(threshold_metadata) if threshold_metadata else None,
						)
					except Exception as e:
						self.log(f"  Warning: Failed to save {genome_type} summary: {e}")

				# Submit to best genomes leaderboard (final validation only)
				if self.dashboard_client and validation_point == 'final':
					try:
						# Determine task_type from cluster type / IDS metrics
						task_type = "ids" if f1 is not None else "lm"
						# Determine stage from experiment name (e.g., "S0: GA Neurons" → "stage_0")
						exp_name = self.config.name if self.config else ""
						if exp_name.startswith("S1:") or exp_name.startswith("S1 "):
							stage = "stage_1"
						elif exp_name.startswith("S2:") or exp_name.startswith("S2 "):
							stage = "stage_2"
						else:
							stage = "stage_0"
						# Determine metric from genome_type
						metric = genome_type.replace("best_", "")  # "ce", "acc", "fitness"
						if metric == "acc":
							metric = "accuracy"
						# Serialize genome data for storage
						genome_data = {
							"config_hash": genome_hash[:16],
							"tiers_json": str(genome),
							"total_clusters": len(genome.neurons_per_cluster),
							"total_neurons": sum(genome.neurons_per_cluster),
							"architecture_type": task_type,
						}
						if genome.connections is not None:
							genome_data["connections_json"] = ",".join(str(c) for c in genome.connections)
						if threshold_metadata:
							genome_data["threshold_metadata"] = threshold_metadata
						self.dashboard_client.submit_best_genomes([{
							"task_type": task_type,
							"stage": stage,
							"metric": metric,
							"genome_hash": genome_hash,
							"ce": ce,
							"accuracy": acc,
							"f1_macro": f1,
							"fpr": fpr_val,
							"flow_id": flow_id,
							"experiment_id": self.experiment_id,
							"genome_data": genome_data,
						}])
					except Exception as e:
						self.log(f"  Warning: leaderboard submit failed: {e}")

			self.log("=" * 60)
			self.log("")

		except Exception as e:
			self.log(f"  Warning: {validation_point} validation failed: {e}")

		return results
