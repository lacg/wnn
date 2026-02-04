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

from wnn.ram.fitness import FitnessCalculatorType, FitnessCalculatorFactory
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

		# Create fitness calculator for validation genome selection
		fitness_calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weight_ce=cfg.fitness_weight_ce,
			weight_acc=cfg.fitness_weight_acc,
		)

		# Run INIT validation on seed population
		# First evaluate to get baseline metrics, then run full validation on best genomes
		seed_pop = initial_population or ([initial_genome] if initial_genome else None)
		if seed_pop:
			self.log("  Evaluating initial population for validation selection...")
			init_evals = self.evaluator.evaluate_batch(seed_pop)
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

			# Run FINAL validation (skip if shutdown was requested)
			if not was_shutdown and result.final_population:
				# Get final evals from the last iteration
				final_evals = self.evaluator.evaluate_batch(result.final_population)
				self._run_validation(
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

		The hash is based on bits_per_cluster, neurons_per_cluster, and connections.
		Two genomes with the same hash will produce identical results.
		"""
		import hashlib

		# Create a string representation of the genome's defining characteristics
		parts = [
			",".join(str(b) for b in genome.bits_per_cluster),
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

		Always returns all three types, even if they refer to the same genome.
		This allows the frontend to display all markers. Duplicate computation
		is avoided via genome_hash caching in _run_validation.

		Args:
			genomes: Population of genomes
			evals: List of (ce, acc) tuples from training evaluation
			fitness_calculator: Optional fitness calculator for combined fitness ranking

		Returns:
			List of (genome, label, ce, acc) tuples
		"""
		if not genomes or not evals:
			return []

		genome_evals = list(zip(genomes, evals))

		# Sort by CE (lower = better) and Acc (higher = better)
		by_ce = sorted(genome_evals, key=lambda x: x[1][0])
		by_acc = sorted(genome_evals, key=lambda x: -x[1][1])

		# Best by CE (always included)
		best_ce_genome, (best_ce_ce, best_ce_acc) = by_ce[0]
		selected = [(best_ce_genome, 'best_ce', best_ce_ce, best_ce_acc)]

		# Best by Acc (always included, even if same genome as best_ce)
		best_acc_genome, (best_acc_ce, best_acc_acc) = by_acc[0]
		selected.append((best_acc_genome, 'best_acc', best_acc_ce, best_acc_acc))

		# Best by Fitness (if using combined fitness)
		if fitness_calculator is not None:
			try:
				scored = [(g, fitness_calculator.calculate_fitness(ce, acc), ce, acc)
						  for g, (ce, acc) in genome_evals]
				by_fitness = sorted(scored, key=lambda x: -x[1])
				best_fitness_genome, _, bf_ce, bf_acc = by_fitness[0]
				selected.append((best_fitness_genome, 'best_fitness', bf_ce, bf_acc))
			except Exception:
				pass

		return selected

	def _run_validation(
		self,
		population: list[ClusterGenome],
		evals: list[tuple[float, float]],
		validation_point: str,  # 'init' or 'final'
		experiment_id: int,
		flow_id: Optional[int] = None,
		fitness_calculator: Optional[Any] = None,
	) -> None:
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

		if not population or not evals:
			self.log("  No population to validate")
			return

		try:
			# Select 1-3 best genomes
			selected = self._select_validation_genomes(population, evals, fitness_calculator)

			if not selected:
				self.log("  No genomes selected for validation")
				return

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

				if cached is not None:
					ce, acc = cached
					self.log(f"  {genome_type}: CE={ce:.4f}, Acc={acc:.4%} (cached, skipping storage)")
				else:
					# Run full validation
					self.log(f"  {genome_type}: Running full validation...")
					full_results = self.evaluator.evaluate_batch_full([genome])
					ce, acc = full_results[0]
					self.log(f"  {genome_type}: CE={ce:.4f}, Acc={acc:.4%} (validated)")

					# Store summary via dashboard API only when validation was performed
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
							)
						except Exception as e:
							self.log(f"  Warning: Failed to save {genome_type} summary: {e}")

			self.log("=" * 60)
			self.log("")

		except Exception as e:
			self.log(f"  Warning: {validation_point} validation failed: {e}")
