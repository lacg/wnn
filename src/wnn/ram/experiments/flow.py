"""
Flow Orchestration

A Flow is a sequence of experiments (like the current 6-phase pass).
This module provides:
- FlowConfig for defining flows
- Flow class for executing flows
- Factory methods for common patterns (standard-6-phase)
"""

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.experiments.experiment import Experiment, ExperimentConfig, ExperimentResult
from wnn.ram.experiments.dashboard_client import DashboardClient, FlowConfig as APIFlowConfig
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


class FlowStoppedError(Exception):
	"""Raised when a flow is stopped due to shutdown request."""
	pass


@dataclass
class FlowConfig:
	"""
	Configuration for a flow (sequence of experiments).

	A flow defines:
	- A name and description
	- A list of experiment configurations in sequence
	- Optional seed checkpoint to start from
	- Shared parameters (tier config, etc.)
	"""

	name: str
	experiments: list[ExperimentConfig]
	description: Optional[str] = None
	seed_checkpoint_path: Optional[str] = None

	# Shared architecture config
	tier_config: Optional[list[tuple[Optional[int], int, int]]] = None
	optimize_tier0_only: bool = False
	context_size: int = 4

	# Shared optimization params
	patience: int = 10
	check_interval: int = 10
	fitness_percentile: Optional[float] = None

	# Fitness calculator settings
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.NORMALIZED
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0

	# Random seed
	seed: Optional[int] = None

	@classmethod
	def standard_6_phase(
		cls,
		name: str,
		ga_generations: int = 250,
		ts_iterations: int = 250,
		population_size: int = 50,
		neighbors_per_iter: int = 50,
		patience: int = 10,
		phase_order: Literal["neurons_first", "bits_first"] = "neurons_first",
		tier_config: Optional[list[tuple[Optional[int], int, int]]] = None,
		optimize_tier0_only: bool = False,
		context_size: int = 4,
		fitness_percentile: Optional[float] = None,
		fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.NORMALIZED,
		fitness_weight_ce: float = 1.0,
		fitness_weight_acc: float = 1.0,
		seed: Optional[int] = None,
		description: Optional[str] = None,
		seed_checkpoint_path: Optional[str] = None,
	) -> "FlowConfig":
		"""
		Create a standard 6-phase flow configuration.

		This matches the existing PhasedSearchRunner's behavior:
		- Phase 1a/1b: GA/TS for first dimension (neurons or bits)
		- Phase 2a/2b: GA/TS for second dimension
		- Phase 3a/3b: GA/TS for connections

		Args:
			name: Flow name
			ga_generations: GA generations per phase
			ts_iterations: TS iterations per phase
			population_size: GA population / TS neighbor cache size
			neighbors_per_iter: TS neighbors per iteration
			patience: Early stopping patience
			phase_order: "neurons_first" or "bits_first"
			tier_config: Tiered architecture config
			optimize_tier0_only: Only mutate tier0 clusters
			fitness_percentile: Fitness percentile filter
			fitness_calculator_type: Fitness calculator type (NORMALIZED, HARMONIC_RANK, etc.)
			fitness_weight_ce: Weight for CE in fitness calculation
			fitness_weight_acc: Weight for accuracy in fitness calculation
			seed: Random seed
			description: Optional description
			seed_checkpoint_path: Optional checkpoint to seed from

		Returns:
			FlowConfig for standard 6-phase search
		"""
		if phase_order == "bits_first":
			phases = [
				("Phase 1a: GA Bits", "ga", True, False, False),
				("Phase 1b: TS Bits", "ts", True, False, False),
				("Phase 2a: GA Neurons", "ga", False, True, False),
				("Phase 2b: TS Neurons", "ts", False, True, False),
				("Phase 3a: GA Connections", "ga", False, False, True),
				("Phase 3b: TS Connections", "ts", False, False, True),
			]
		else:
			phases = [
				("Phase 1a: GA Neurons", "ga", False, True, False),
				("Phase 1b: TS Neurons", "ts", False, True, False),
				("Phase 2a: GA Bits", "ga", True, False, False),
				("Phase 2b: TS Bits", "ts", True, False, False),
				("Phase 3a: GA Connections", "ga", False, False, True),
				("Phase 3b: TS Connections", "ts", False, False, True),
			]

		experiments = []
		for phase_name, exp_type, opt_bits, opt_neurons, opt_conns in phases:
			config = ExperimentConfig(
				name=phase_name,
				experiment_type=exp_type,
				optimize_bits=opt_bits,
				optimize_neurons=opt_neurons,
				optimize_connections=opt_conns,
				generations=ga_generations,
				population_size=population_size,
				iterations=ts_iterations,
				neighbors_per_iter=neighbors_per_iter,
				patience=patience,
				tier_config=tier_config,
				optimize_tier0_only=optimize_tier0_only,
				fitness_percentile=fitness_percentile,
				fitness_calculator_type=fitness_calculator_type,
				fitness_weight_ce=fitness_weight_ce,
				fitness_weight_acc=fitness_weight_acc,
				seed=seed,
			)
			experiments.append(config)

		return cls(
			name=name,
			experiments=experiments,
			description=description or f"Standard 6-phase search ({phase_order})",
			seed_checkpoint_path=seed_checkpoint_path,
			tier_config=tier_config,
			optimize_tier0_only=optimize_tier0_only,
			context_size=context_size,
			patience=patience,
			fitness_percentile=fitness_percentile,
			fitness_calculator_type=fitness_calculator_type,
			fitness_weight_ce=fitness_weight_ce,
			fitness_weight_acc=fitness_weight_acc,
			seed=seed,
		)

	def to_api_config(self) -> APIFlowConfig:
		"""Convert to API FlowConfig for dashboard registration."""
		# Convert tier_config to string format for API compatibility
		tier_config_str = None
		if self.tier_config is not None:
			tier_parts = []
			for tier in self.tier_config:
				if tier[0] is None:
					tier_parts.append(f"rest,{tier[1]},{tier[2]}")
				else:
					tier_parts.append(f"{tier[0]},{tier[1]},{tier[2]}")
			tier_config_str = ";".join(tier_parts)

		return APIFlowConfig(
			name=self.name,
			experiments=[exp.to_dict() for exp in self.experiments],
			description=self.description,
			params={
				"tier_config": tier_config_str,
				"optimize_tier0_only": self.optimize_tier0_only,
				"context_size": self.context_size,
				"patience": self.patience,
				"fitness_percentile": self.fitness_percentile,
				"fitness_calculator": self.fitness_calculator_type.name.lower(),
				"fitness_weight_ce": self.fitness_weight_ce,
				"fitness_weight_acc": self.fitness_weight_acc,
				"seed": self.seed,
			},
		)


@dataclass
class FlowResult:
	"""Result from running a complete flow."""

	flow_name: str
	experiment_results: list[ExperimentResult]
	final_genome: ClusterGenome
	final_fitness: float
	final_accuracy: Optional[float]
	total_elapsed_seconds: float
	flow_id: Optional[int] = None

	def get_best_by_accuracy(self) -> Optional[ExperimentResult]:
		"""Get the experiment result with best accuracy."""
		best = None
		best_acc = -1.0
		for result in self.experiment_results:
			if result.final_accuracy and result.final_accuracy > best_acc:
				best = result
				best_acc = result.final_accuracy
		return best


class Flow:
	"""
	Flow executor for running a sequence of experiments.

	Example usage:
		config = FlowConfig.standard_6_phase(
			name="Pass 1",
			patience=10,
			tier_config=[(100, 15, 20), (400, 10, 12), (None, 5, 8)],
		)

		flow = Flow(
			config=config,
			evaluator=cached_evaluator,
			logger=log_fn,
			checkpoint_dir=Path("checkpoints/pass1"),
		)

		result = flow.run()
		print(f"Final CE: {result.final_fitness:.4f}")
	"""

	def __init__(
		self,
		config: FlowConfig,
		evaluator: Any,  # CachedEvaluator
		logger: Callable[[str], None],
		checkpoint_dir: Optional[Path] = None,
		dashboard_client: Optional[DashboardClient] = None,
		flow_id: Optional[int] = None,
		tracker: Optional[Any] = None,  # ExperimentTracker for V2 tracking
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
	):
		"""
		Initialize flow.

		Args:
			config: Flow configuration
			evaluator: CachedEvaluator instance
			logger: Logging function
			checkpoint_dir: Directory for checkpoints
			dashboard_client: Optional dashboard client for API integration
			flow_id: Existing flow ID (skip creating new flow if provided)
			tracker: Optional V2 tracker for direct database writes
			shutdown_check: Optional callable that returns True if shutdown requested
		"""
		self.config = config
		self.evaluator = evaluator
		self.log = logger
		self.checkpoint_dir = checkpoint_dir
		self.dashboard_client = dashboard_client
		self.tracker = tracker
		self.shutdown_check = shutdown_check

		self._flow_id: Optional[int] = flow_id
		self._experiment_ids: dict[int, int] = {}  # idx -> experiment_id
		self._results: list[ExperimentResult] = []

	def run(
		self,
		resume_from: Optional[int] = None,
		seed_genome: Optional[ClusterGenome] = None,
		seed_population: Optional[list[ClusterGenome]] = None,
		seed_threshold: Optional[float] = None,
	) -> FlowResult:
		"""
		Run the flow.

		Args:
			resume_from: Experiment index to resume from (0-indexed)
			seed_genome: Initial genome to seed first experiment
			seed_population: Initial population to seed first experiment
			seed_threshold: Initial threshold to continue from

		Returns:
			FlowResult with all experiment results
		"""
		cfg = self.config
		start_time = time.time()

		# Check for empty flow
		if not cfg.experiments:
			raise ValueError("Flow has no experiments configured. Add experiments before running.")

		self.log("")
		self.log("=" * 70)
		self.log(f"  FLOW: {cfg.name}")
		self.log("=" * 70)
		if cfg.description:
			self.log(f"  {cfg.description}")
		self.log(f"  Experiments: {len(cfg.experiments)}")
		if resume_from:
			self.log(f"  Resuming from experiment {resume_from}")
		self.log("")

		# Register with dashboard if client available (skip if flow_id already set)
		if self.dashboard_client and self._flow_id is None:
			try:
				seed_checkpoint_id = None
				if cfg.seed_checkpoint_path:
					# Look up checkpoint ID from path
					seed_checkpoint_id = self.dashboard_client.find_checkpoint_by_path(
						cfg.seed_checkpoint_path
					)
					if seed_checkpoint_id:
						self.log(f"Found seed checkpoint ID: {seed_checkpoint_id}")

				self._flow_id = self.dashboard_client.create_flow(
					cfg.to_api_config(),
					seed_checkpoint_id=seed_checkpoint_id,
				)
				self.dashboard_client.flow_started(self._flow_id)
				self.log(f"Registered flow {self._flow_id} with dashboard")
			except Exception as e:
				self.log(f"Warning: Failed to register flow with dashboard: {e}")
		elif self._flow_id is not None:
			self.log(f"Using existing flow {self._flow_id}")

		# Create all experiments upfront with pending status (for both new and existing flows)
		# This ensures experiments exist in DB before we start running them
		if self.tracker and self._flow_id:
			for idx, exp_config in enumerate(cfg.experiments):
				# Check if experiment already exists for this flow/sequence
				existing_exp = self.tracker.get_experiment_by_flow_sequence(self._flow_id, idx)
				if existing_exp:
					self._experiment_ids[idx] = existing_exp["id"]
					self.log(f"Found existing experiment {existing_exp['id']}: {exp_config.name} (sequence_order={idx})")
				else:
					opt_target = "bits" if exp_config.optimize_bits else "neurons" if exp_config.optimize_neurons else "connections"
					phase_type = f"{'ga' if exp_config.experiment_type == 'ga' else 'ts'}_{opt_target}"
					max_iters = exp_config.generations if exp_config.experiment_type == "ga" else exp_config.iterations

					exp_id = self.tracker.create_pending_experiment(
						name=exp_config.name,
						flow_id=self._flow_id,
						sequence_order=idx,
						phase_type=phase_type,
						max_iterations=max_iters,
					)
					self._experiment_ids[idx] = exp_id
					self.log(f"Created pending experiment {exp_id}: {exp_config.name} (sequence_order={idx})")

		# Load seed from checkpoint if specified
		if cfg.seed_checkpoint_path and not seed_genome:
			seed_genome, seed_population, seed_threshold = self._load_seed_checkpoint(
				cfg.seed_checkpoint_path
			)

		# Create initial genome from tier config if not seeded
		if seed_genome is None and cfg.tier_config:
			seed_genome = self._create_tiered_genome()

		# Run experiments
		start_idx = resume_from or 0
		current_genome = seed_genome
		current_population = seed_population
		current_threshold = seed_threshold
		current_fitness: Optional[float] = None
		stopped_at_idx: Optional[int] = None  # Track where we stopped for checkpoint

		try:
			for idx, exp_config in enumerate(cfg.experiments):
				if idx < start_idx:
					# Load checkpoint for skipped experiments
					result = self._load_experiment_checkpoint(idx)
					if result:
						self._results.append(result)
						current_genome = result.best_genome
						current_population = result.final_population
						current_threshold = result.final_threshold
						current_fitness = result.final_fitness
						self.log(f"Loaded checkpoint for experiment {idx}: CE={current_fitness:.4f}")
					else:
						# Checkpoint not found - try to query database for completed phase results
						self.log(f"Warning: No checkpoint found for experiment {idx}, querying database...")
						db_result = self._load_from_database(idx, exp_config)
						if db_result:
							current_genome, current_population, current_threshold, current_fitness = db_result
							self.log(f"Loaded from database for experiment {idx}: CE={current_fitness:.4f}")
						else:
							# Cannot skip this experiment without its results
							raise ValueError(
								f"Cannot resume from experiment {start_idx}: "
								f"No checkpoint or database results found for experiment {idx} ({exp_config.name}). "
								f"Either run from the beginning or provide a valid checkpoint."
							)
					continue

				# Create experiment checkpoint directory
				exp_checkpoint_dir = None
				if self.checkpoint_dir:
					exp_checkpoint_dir = self.checkpoint_dir / f"exp_{idx:02d}"

				# Create experiment in database for this config spec
				# Each config spec becomes its own experiment with proper name and sequence_order
				experiment_id = None
				tracker_experiment_id = None

				# Convert tier_config to string format for DB
				tier_config_str = None
				if cfg.tier_config is not None:
					tier_parts = []
					for tier in cfg.tier_config:
						if tier[0] is None:
							tier_parts.append(f"rest,{tier[1]},{tier[2]}")
						else:
							tier_parts.append(f"{tier[0]},{tier[1]},{tier[2]}")
					tier_config_str = ";".join(tier_parts)

				# Determine phase type string for tracking
				opt_target = "bits" if exp_config.optimize_bits else "neurons" if exp_config.optimize_neurons else "connections"
				phase_type = f"{'ga' if exp_config.experiment_type == 'ga' else 'ts'}_{opt_target}"

				# Check if experiment already exists (created when flow was queued from dashboard)
				existing_experiment_id = self._experiment_ids.get(idx)
				if existing_experiment_id:
					experiment_id = existing_experiment_id
					tracker_experiment_id = existing_experiment_id
					# Update existing experiment status to running
					if self.dashboard_client:
						try:
							self.dashboard_client.experiment_started(experiment_id)
							self.log(f"Started experiment {experiment_id}: {exp_config.name} (existing)")
						except Exception as e:
							self.log(f"Warning: Failed to update experiment status: {e}")
					elif self.tracker:
						try:
							self.tracker.update_experiment_status(experiment_id, "running")
							self.log(f"Started experiment {experiment_id}: {exp_config.name} (existing)")
						except Exception as e:
							self.log(f"Warning: Failed to update experiment status via tracker: {e}")
				elif self.tracker:
					try:
						# Create experiment with config spec name and sequence_order
						tracker_experiment_id = self.tracker.start_experiment(
							name=exp_config.name,  # Use config spec name, NOT flow name
							flow_id=self._flow_id,
							sequence_order=idx,
							tier_config=tier_config_str,
							context_size=cfg.context_size,
							population_size=exp_config.population_size,
							phase_type=phase_type,
							max_iterations=exp_config.generations if exp_config.experiment_type == "ga" else exp_config.iterations,
						)
						experiment_id = tracker_experiment_id
						self._experiment_ids[idx] = experiment_id
						self.log(f"Created experiment {experiment_id}: {exp_config.name} (sequence_order={idx})")
					except Exception as e:
						self.log(f"Warning: Failed to create experiment via tracker: {e}")

				# Fallback to dashboard_client if tracker not available
				if not experiment_id and self.dashboard_client:
					try:
						experiment_id = self.dashboard_client.create_experiment(
							name=exp_config.name,
							log_path=str(exp_checkpoint_dir) if exp_checkpoint_dir else "",
							config=exp_config.to_dict(),
						)
						self._experiment_ids[idx] = experiment_id

						# Link experiment to flow
						if self._flow_id:
							self.dashboard_client.link_experiment_to_flow(
								flow_id=self._flow_id,
								experiment_id=experiment_id,
								sequence_order=idx,
							)
						self.log(f"Created experiment {experiment_id}: {exp_config.name} (via dashboard)")
					except Exception as e:
						self.log(f"Warning: Failed to create experiment in dashboard: {e}")

				# Check for shutdown before starting experiment
				if self.shutdown_check and self.shutdown_check():
					self.log(f"Shutdown requested, stopping flow before experiment {idx}")
					stopped_at_idx = idx
					raise FlowStoppedError("Shutdown requested")

				# Create and run experiment
				# Run init validation on first experiment only (Phase 1a)
				is_first_experiment = (idx == 0 and start_idx == 0)
				experiment = Experiment(
					config=exp_config,
					evaluator=self.evaluator,
					logger=self.log,
					checkpoint_dir=exp_checkpoint_dir,
					dashboard_client=self.dashboard_client,
					experiment_id=experiment_id,
					tracker=self.tracker,
					flow_id=self._flow_id,
					shutdown_check=self.shutdown_check,
					run_init_validation=is_first_experiment,  # Only validate seed on first experiment
				)

				result = experiment.run(
					initial_genome=current_genome,
					initial_fitness=current_fitness if exp_config.experiment_type == "ts" else None,
					initial_population=current_population,
					initial_threshold=current_threshold,
					tracker_experiment_id=tracker_experiment_id,  # Pass this experiment's ID
				)

				self._results.append(result)

				# Check if experiment was stopped due to shutdown
				if result.was_shutdown:
					self.log(f"Experiment {idx} stopped due to shutdown, stopping flow")
					stopped_at_idx = idx
					raise FlowStoppedError("Shutdown requested during experiment")

				# Also check shutdown_check after experiment completes
				# (in case shutdown was requested while experiment was finishing)
				if self.shutdown_check and self.shutdown_check():
					self.log(f"Shutdown detected after experiment {idx}, stopping flow")
					stopped_at_idx = idx
					raise FlowStoppedError("Shutdown requested after experiment")

				# Update state for next experiment
				current_genome = result.best_genome
				current_population = result.final_population
				current_threshold = result.final_threshold
				current_fitness = result.final_fitness

			# Flow completed successfully
			if self.dashboard_client and self._flow_id:
				try:
					self.dashboard_client.flow_completed(self._flow_id)
				except Exception as e:
					self.log(f"Warning: Failed to mark flow completed: {e}")

		except FlowStoppedError:
			# Flow was stopped gracefully (shutdown requested)
			self.log("Flow stopped due to shutdown request")

			# Save checkpoint to database so we can resume later
			if self.checkpoint_dir and current_genome and self.dashboard_client and self._flow_id:
				try:
					checkpoint_id = self._save_stop_checkpoint_to_db(
						stopped_at_idx=stopped_at_idx or len(self._results),
						current_genome=current_genome,
						current_fitness=current_fitness,
						current_population=current_population,
						current_threshold=current_threshold,
					)
					if checkpoint_id:
						# Update flow's seed_checkpoint_id so it resumes from here
						self.dashboard_client.set_flow_checkpoint(self._flow_id, checkpoint_id)
						self.log(f"Checkpoint saved to database (id={checkpoint_id})")
				except Exception as e:
					self.log(f"Warning: Failed to save stop checkpoint: {e}")

			if self.dashboard_client and self._flow_id:
				try:
					self.dashboard_client.update_flow(self._flow_id, status="cancelled")
				except Exception:
					pass
			raise

		except Exception as e:
			# Flow failed
			if self.dashboard_client and self._flow_id:
				try:
					self.dashboard_client.flow_failed(self._flow_id, str(e))
				except Exception:
					pass
			raise

		elapsed = time.time() - start_time

		# Get final result (handle edge case of no results)
		if not self._results:
			raise ValueError("Flow completed but no experiment results were recorded.")
		final_result = self._results[-1]

		self.log("")
		self.log("=" * 70)
		self.log(f"  FLOW COMPLETE: {cfg.name}")
		self.log("=" * 70)
		self.log(f"  Final CE: {final_result.final_fitness:.4f}")
		if final_result.final_accuracy:
			self.log(f"  Final Accuracy: {final_result.final_accuracy:.2%}")
		self.log(f"  Total Duration: {elapsed:.1f}s")
		self.log("")

		return FlowResult(
			flow_name=cfg.name,
			experiment_results=self._results,
			final_genome=final_result.best_genome,
			final_fitness=final_result.final_fitness,
			final_accuracy=final_result.final_accuracy,
			total_elapsed_seconds=elapsed,
			flow_id=self._flow_id,
		)

	def _load_seed_checkpoint(
		self,
		checkpoint_path: str,
	) -> tuple[Optional[ClusterGenome], Optional[list[ClusterGenome]], Optional[float]]:
		"""Load seed data from checkpoint file."""
		try:
			path = Path(checkpoint_path)

			if path.suffix == '.gz':
				with gzip.open(path, 'rt', encoding='utf-8') as f:
					data = json.load(f)
			else:
				with open(path, 'r') as f:
					data = json.load(f)

			genome = None
			population = None
			threshold = None

			# Check for phase_result format
			if 'phase_result' in data:
				pr = data['phase_result']
				if 'best_genome' in pr:
					genome = ClusterGenome.deserialize(pr['best_genome'])
				if 'final_population' in pr and pr['final_population']:
					population = [ClusterGenome.deserialize(g) for g in pr['final_population']]
				if 'final_threshold' in pr:
					threshold = pr['final_threshold']

			# Check for final format
			elif 'final' in data:
				final = data['final']
				if 'genome' in final:
					genome = ClusterGenome.deserialize(final['genome'])
				if 'final_population' in final and final['final_population']:
					population = [ClusterGenome.deserialize(g) for g in final['final_population']]
				if 'final_threshold' in final:
					threshold = final['final_threshold']

			if genome:
				self.log(f"Loaded seed from checkpoint: {checkpoint_path}")
				if population:
					self.log(f"  Population: {len(population)} genomes")
				if threshold:
					self.log(f"  Threshold: {threshold:.4%}")

			return genome, population, threshold

		except Exception as e:
			self.log(f"Warning: Failed to load seed checkpoint: {e}")
			return None, None, None

	def _save_stop_checkpoint_to_db(
		self,
		stopped_at_idx: int,
		current_genome: ClusterGenome,
		current_fitness: Optional[float],
		current_population: Optional[list[ClusterGenome]],
		current_threshold: Optional[float],
	) -> Optional[int]:
		"""Save a checkpoint to the database when the flow is stopped.

		Returns:
			Checkpoint ID if successful, None otherwise.
		"""
		# Get the experiment_id for the stopped experiment (or the last completed one)
		experiment_id = self._experiment_ids.get(stopped_at_idx) or (
			self._experiment_ids.get(stopped_at_idx - 1) if stopped_at_idx > 0 else None
		)
		if not self.dashboard_client or not experiment_id:
			self.log("Warning: Cannot save stop checkpoint - no dashboard client or experiment_id")
			return None

		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

		# Save checkpoint file
		checkpoint_name = f"stopped_at_exp_{stopped_at_idx}"
		checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json.gz"

		data = {
			"stopped_at_experiment": stopped_at_idx,
			"total_experiments": len(self.config.experiments),
			"completed_experiments": len(self._results),
			"current_genome": current_genome.serialize() if current_genome else None,
			"current_fitness": current_fitness,
			"current_population": [g.serialize() for g in current_population] if current_population else None,
			"current_threshold": current_threshold,
			"flow_name": self.config.name,
		}

		with gzip.open(checkpoint_path, 'wt', encoding='utf-8') as f:
			json.dump(data, f, separators=(',', ':'))

		self.log(f"Checkpoint file saved: {checkpoint_path}")

		# Register in database
		try:
			checkpoint_id = self.dashboard_client.checkpoint_created(
				experiment_id=experiment_id,
				file_path=str(checkpoint_path),
				name=checkpoint_name,
				final_fitness=current_fitness,
				final_accuracy=None,
				iterations_run=stopped_at_idx,
				genome_stats={"stopped": True, "resume_from": stopped_at_idx},
				is_final=False,
				checkpoint_type="user",  # Mark as user-initiated stop checkpoint
			)
			self.log(f"  Registered checkpoint {checkpoint_id} in database")
			self.log(f"  Resume from experiment {stopped_at_idx}")
			return checkpoint_id
		except Exception as e:
			self.log(f"Warning: Failed to register checkpoint in database: {e}")
			return None

	def _load_experiment_checkpoint(self, idx: int) -> Optional[ExperimentResult]:
		"""Load checkpoint for a specific experiment."""
		if not self.checkpoint_dir:
			return None

		exp_dir = self.checkpoint_dir / f"exp_{idx:02d}"
		if not exp_dir.exists():
			return None

		# Find the checkpoint file
		checkpoints = list(exp_dir.glob("*.json.gz"))
		if not checkpoints:
			return None

		checkpoint_path = checkpoints[0]  # Take first (should be only one)

		try:
			with gzip.open(checkpoint_path, 'rt', encoding='utf-8') as f:
				data = json.load(f)

			pr = data.get('phase_result', {})
			metadata = data.get('_metadata', {})

			return ExperimentResult(
				experiment_name=pr.get('phase_name', f"Experiment {idx}"),
				strategy_type=pr.get('strategy_type', 'unknown'),
				initial_fitness=pr.get('initial_fitness'),
				final_fitness=pr.get('final_fitness', 0.0),
				final_accuracy=pr.get('final_accuracy'),
				improvement_percent=metadata.get('improvement_percent', 0.0),
				iterations_run=pr.get('iterations_run', 0),
				best_genome=ClusterGenome.deserialize(pr['best_genome']),
				final_population=[ClusterGenome.deserialize(g) for g in pr.get('final_population', [])] if pr.get('final_population') else None,
				final_threshold=pr.get('final_threshold'),
				elapsed_seconds=metadata.get('elapsed_seconds', 0.0),
				checkpoint_path=str(checkpoint_path),
			)

		except Exception as e:
			self.log(f"Warning: Failed to load checkpoint for experiment {idx}: {e}")
			return None

	def _load_from_database(
		self,
		idx: int,
		exp_config: 'ExperimentConfig',
	) -> Optional[tuple[ClusterGenome, Optional[list[ClusterGenome]], Optional[float], float]]:
		"""
		Query database for completed phase results when checkpoint is missing.

		Returns tuple of (genome, population, threshold, fitness) or None if not found.
		"""
		if not self.dashboard_client or not self._flow_id:
			return None

		try:
			# Get experiments for this flow
			flow = self.dashboard_client.get_flow(self._flow_id)
			if not flow:
				return None

			# Find completed phases for this flow's experiments
			# We need to match by phase type and sequence
			experiments = self.dashboard_client.list_experiments(flow_id=self._flow_id)
			if not experiments:
				return None

			# Look for a completed phase matching this experiment's type
			phase_type = "ga" if exp_config.experiment_type == "ga" else "ts"
			optimize_what = "neurons" if exp_config.optimize_neurons else ("bits" if exp_config.optimize_bits else "connections")
			expected_phase_type = f"{phase_type}_{optimize_what}"

			for exp in experiments:
				phases = self.dashboard_client.get_phases(exp['id'])
				for phase in phases:
					if phase.get('status') == 'completed' and phase.get('phase_type') == expected_phase_type:
						best_ce = phase.get('best_ce')
						best_acc = phase.get('best_accuracy')

						if best_ce is not None:
							self.log(f"Found completed phase in database: {phase.get('name')} CE={best_ce:.4f}")

							# We have the fitness, but we need the genome too
							# Try to find a checkpoint registered in the database
							checkpoints = self.dashboard_client.list_checkpoints(experiment_id=exp['id'])
							for ckpt in checkpoints:
								if ckpt.get('file_path'):
									try:
										genome, population, threshold = self._load_seed_checkpoint(ckpt['file_path'])
										if genome:
											return (genome, population, threshold, best_ce)
									except Exception:
										pass

							# If no checkpoint with genome found, we can't proceed
							self.log(f"Warning: Found completed phase but no checkpoint with genome data")
							return None

			return None

		except Exception as e:
			self.log(f"Warning: Failed to query database for experiment {idx}: {e}")
			return None

	def _create_tiered_genome(self) -> ClusterGenome:
		"""Create a genome with tiered configuration."""
		if not self.config.tier_config:
			return ClusterGenome.create_uniform(
				num_clusters=self.evaluator.vocab_size,
				bits=8,
				neurons=5,
			)

		bits_per_cluster = []
		neurons_per_cluster = []
		cluster_idx = 0

		for num_clusters, neurons, bits in self.config.tier_config:
			if num_clusters is None:
				count = self.evaluator.vocab_size - cluster_idx
			else:
				count = min(num_clusters, self.evaluator.vocab_size - cluster_idx)

			bits_per_cluster.extend([bits] * count)
			neurons_per_cluster.extend([neurons] * count)
			cluster_idx += count

			if cluster_idx >= self.evaluator.vocab_size:
				break

		# Pad if needed
		while len(bits_per_cluster) < self.evaluator.vocab_size:
			bits_per_cluster.append(8)
			neurons_per_cluster.append(5)

		genome = ClusterGenome(
			bits_per_cluster=bits_per_cluster[:self.evaluator.vocab_size],
			neurons_per_cluster=neurons_per_cluster[:self.evaluator.vocab_size],
		)

		# Initialize connections
		if not genome.has_connections():
			genome.initialize_connections(self.evaluator.total_input_bits)

		return genome
