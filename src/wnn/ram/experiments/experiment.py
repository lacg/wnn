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
from typing import Any, Callable, Literal, Optional

from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.experiments.phased_search import PhaseResult


@dataclass
class ExperimentConfig:
	"""Configuration for a single experiment (GA or TS optimization)."""

	name: str
	experiment_type: Literal["ga", "ts"]

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
	patience: int = 10
	check_interval: int = 10

	# Architecture bounds
	min_bits: int = 4
	max_bits: int = 20
	min_neurons: int = 1
	max_neurons: int = 15
	default_bits: int = 8
	default_neurons: int = 5

	# Tier configuration
	tier_config: Optional[list[tuple[Optional[int], int, int]]] = None
	optimize_tier0_only: bool = False

	# Population handling
	seed_only: bool = False
	fresh_population: bool = False

	# Fitness filtering
	fitness_percentile: Optional[float] = None

	# Random seed
	seed: Optional[int] = None

	# Fitness calculator settings
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.NORMALIZED
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0

	def to_dict(self) -> dict[str, Any]:
		"""Convert to dictionary for JSON serialization.

		Note: tier_config is converted to a string format for API compatibility
		(Rust backend expects Option<String>, not a list of tuples).
		"""
		result = asdict(self)
		# Convert tier_config list to string format for API compatibility
		# Format: "100,15,20;400,10,12;rest,5,8"
		if result.get("tier_config") is not None:
			tier_parts = []
			for tier in result["tier_config"]:
				if tier[0] is None:
					tier_parts.append(f"rest,{tier[1]},{tier[2]}")
				else:
					tier_parts.append(f"{tier[0]},{tier[1]},{tier[2]}")
			result["tier_config"] = ";".join(tier_parts)
		# Convert fitness_calculator_type enum to string for JSON
		if "fitness_calculator_type" in result:
			result["fitness_calculator_type"] = result["fitness_calculator_type"].name.lower()
		return result

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
		"""Create from dictionary."""
		# Convert fitness_calculator_type string back to enum
		if "fitness_calculator_type" in data and isinstance(data["fitness_calculator_type"], str):
			data = data.copy()  # Don't modify the original
			try:
				data["fitness_calculator_type"] = FitnessCalculatorType[data["fitness_calculator_type"].upper()]
			except KeyError:
				data["fitness_calculator_type"] = FitnessCalculatorType.NORMALIZED
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
			experiment_type="ga",
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
		evaluator: Any,  # CachedEvaluator
		logger: Callable[[str], None],
		checkpoint_dir: Optional[Path] = None,
		dashboard_client: Optional[Any] = None,
		experiment_id: Optional[int] = None,
		tracker: Optional[Any] = None,  # ExperimentTracker for V2 tracking
		flow_id: Optional[int] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
		run_init_validation: bool = False,  # Run full validation on seed population before optimization
	):
		"""
		Initialize experiment.

		Args:
			config: Experiment configuration
			evaluator: CachedEvaluator instance for genome evaluation
			logger: Logging function
			checkpoint_dir: Directory for saving checkpoints
			dashboard_client: Optional DashboardClient for API integration
			experiment_id: Optional experiment ID for dashboard integration
			tracker: Optional V2 tracker for direct database writes
			flow_id: Optional flow ID for V2 tracking
			shutdown_check: Optional callable that returns True if shutdown requested
			run_init_validation: If True, run full validation on seed population before optimization starts
		"""
		self.config = config
		self.evaluator = evaluator
		self.log = logger
		self.checkpoint_dir = checkpoint_dir
		self.dashboard_client = dashboard_client
		self.experiment_id = experiment_id
		self.tracker = tracker
		self.flow_id = flow_id
		self.shutdown_check = shutdown_check
		self.run_init_validation = run_init_validation

		# Derived properties
		self.vocab_size = evaluator.vocab_size
		self.total_input_bits = evaluator.total_input_bits

	def run(
		self,
		initial_genome: Optional[ClusterGenome] = None,
		initial_fitness: Optional[float] = None,
		initial_population: Optional[list[ClusterGenome]] = None,
		initial_threshold: Optional[float] = None,
		tracker_experiment_id: Optional[int] = None,
	) -> ExperimentResult:
		"""
		Run the experiment.

		Args:
			initial_genome: Optional starting genome
			initial_fitness: Fitness of initial genome (required for TS)
			initial_population: Population to seed from
			initial_threshold: Starting accuracy threshold
			tracker_experiment_id: V2 experiment ID for tracker (if using V2 tracking)

		Returns:
			ExperimentResult with optimization results
		"""
		cfg = self.config
		start_time = time.time()

		self.log("")
		self.log(f"{'='*60}")
		self.log(f"  {cfg.name}")
		self.log(f"{'='*60}")
		self.log(f"  Type: {cfg.experiment_type.upper()}")
		self.log(f"  Optimize: bits={cfg.optimize_bits}, neurons={cfg.optimize_neurons}, connections={cfg.optimize_connections}")
		if initial_genome:
			self.log(f"  Starting from: {initial_genome}")
		if initial_population:
			self.log(f"  Seeding from {len(initial_population)} genomes")
		self.log("")

		# Determine strategy type
		is_ga = cfg.experiment_type == "ga"
		strategy_type = (
			OptimizerStrategyType.ARCHITECTURE_GA if is_ga
			else OptimizerStrategyType.ARCHITECTURE_TS
		)

		# Build strategy kwargs
		strategy_kwargs = {
			"strategy_type": strategy_type,
			"num_clusters": self.vocab_size,
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
			"initial_threshold": initial_threshold,
			"fitness_percentile": cfg.fitness_percentile,
			"seed": cfg.seed,
			"min_bits": cfg.min_bits,
			"max_bits": cfg.max_bits,
			"min_neurons": cfg.min_neurons,
			"max_neurons": cfg.max_neurons,
			# Fitness calculator settings
			"fitness_calculator_type": cfg.fitness_calculator_type,
			"fitness_weight_ce": cfg.fitness_weight_ce,
			"fitness_weight_acc": cfg.fitness_weight_acc,
		}

		# Tier0-only optimization
		self.log(f"  DEBUG: optimize_tier0_only={cfg.optimize_tier0_only}, tier_config={cfg.tier_config}")
		if cfg.optimize_tier0_only and cfg.tier_config:
			tier0_clusters = cfg.tier_config[0][0] or self.vocab_size
			strategy_kwargs["mutable_clusters"] = tier0_clusters
			self.log(f"  Tier0-only mode: mutating first {tier0_clusters} clusters")

		# Type-specific kwargs
		if is_ga:
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

		# Run init validation on seed population if requested (first experiment only)
		if self.run_init_validation and tracker_experiment_id:
			self._run_init_validation(
				initial_genome=initial_genome,
				initial_population=initial_population,
				experiment_id=tracker_experiment_id,
			)

		# Run optimization with exception handling for phase status
		result = None
		was_shutdown = False
		try:
			if is_ga:
				seed_pop = initial_population or ([initial_genome] if initial_genome else None)
				result = strategy.optimize(
					evaluate_fn=None,
					initial_population=seed_pop,
				)
			else:
				result = strategy.optimize(
					evaluate_fn=None,
					initial_genome=initial_genome,
					initial_fitness=initial_fitness,
					initial_neighbors=initial_population,
				)

			# Check if shutdown was requested (needed for phase status update)
			from wnn.ram.strategies.connectivity.generic_strategies import StopReason
			was_shutdown = result.stop_reason == StopReason.SHUTDOWN if result.stop_reason else False

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

		except Exception as e:
			# Mark experiment as failed on exception
			if self.tracker and tracker_experiment_id:
				try:
					from wnn.ram.experiments.tracker import TrackerStatus
					self.tracker.update_experiment_status(tracker_experiment_id, TrackerStatus.FAILED)
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

	def _run_init_validation(
		self,
		initial_genome: Optional[ClusterGenome],
		initial_population: Optional[list[ClusterGenome]],
		experiment_id: int,
	) -> None:
		"""
		Run full validation on seed population before optimization starts.

		This provides a baseline to compare against experiment end results.
		Logs init_best_ce, init_best_acc, init_top_k_mean.
		"""
		self.log("")
		self.log("=" * 60)
		self.log("  INIT BASELINE VALIDATION (Full Dataset)")
		self.log("=" * 60)

		# Collect genomes to evaluate
		genomes_to_eval = []
		if initial_population:
			genomes_to_eval = list(initial_population)
		elif initial_genome:
			genomes_to_eval = [initial_genome]

		if not genomes_to_eval:
			self.log("  No seed genomes to evaluate, skipping init validation")
			return

		# Evaluate on full validation data
		self.log(f"  Evaluating {len(genomes_to_eval)} seed genomes on full validation set...")
		try:
			full_results = self.evaluator.evaluate_batch_full(genomes_to_eval)
			full_evals = list(zip(genomes_to_eval, full_results))

			# Sort by CE and Acc
			by_ce = sorted(full_evals, key=lambda x: x[1][0])  # ascending CE (lower = better)
			by_acc = sorted(full_evals, key=lambda x: -x[1][1])  # descending acc (higher = better)

			# Best by CE
			_, (init_best_ce_ce, init_best_ce_acc) = by_ce[0]

			# Best by Acc
			_, (init_best_acc_ce, init_best_acc_acc) = by_acc[0]

			# Top-K mean (top 20% or at least 1)
			top_k = max(1, int(len(full_evals) * 0.2))
			top_k_ce = sum(ce for _, (ce, _) in full_evals[:top_k]) / top_k
			top_k_acc = sum(acc for _, (_, acc) in full_evals[:top_k]) / top_k

			# Log results
			self.log(f"  Init Best CE:       CE={init_best_ce_ce:.4f}, Acc={init_best_ce_acc:.4%}")
			self.log(f"  Init Best Accuracy: CE={init_best_acc_ce:.4f}, Acc={init_best_acc_acc:.4%}")
			self.log(f"  Init Top-{top_k} Mean:   CE={top_k_ce:.4f}, Acc={top_k_acc:.4%}")
			self.log("=" * 60)
			self.log("")

			# Record as initial progress on experiment
			if self.tracker and experiment_id:
				try:
					self.tracker.update_experiment_progress(
						experiment_id,
						current_iteration=0,
						best_ce=init_best_ce_ce,
						best_accuracy=init_best_ce_acc,
					)
					self.log("  Init validation results recorded to experiment")
				except Exception as e:
					self.log(f"  Warning: Failed to record init validation results: {e}")

		except Exception as e:
			self.log(f"  Warning: Init validation failed: {e}")
