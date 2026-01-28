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

from wnn.ram.experiments.experiment import Experiment, ExperimentConfig, ExperimentResult
from wnn.ram.experiments.dashboard_client import DashboardClient, FlowConfig as APIFlowConfig
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


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

	# Shared optimization params
	patience: int = 10
	check_interval: int = 10
	fitness_percentile: Optional[float] = None

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
		fitness_percentile: Optional[float] = None,
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
			patience=patience,
			fitness_percentile=fitness_percentile,
			seed=seed,
		)

	def to_api_config(self) -> APIFlowConfig:
		"""Convert to API FlowConfig for dashboard registration."""
		return APIFlowConfig(
			name=self.name,
			experiments=[exp.to_dict() for exp in self.experiments],
			description=self.description,
			params={
				"tier_config": self.tier_config,
				"optimize_tier0_only": self.optimize_tier0_only,
				"patience": self.patience,
				"fitness_percentile": self.fitness_percentile,
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
	):
		"""
		Initialize flow.

		Args:
			config: Flow configuration
			evaluator: CachedEvaluator instance
			logger: Logging function
			checkpoint_dir: Directory for checkpoints
			dashboard_client: Optional dashboard client for API integration
		"""
		self.config = config
		self.evaluator = evaluator
		self.log = logger
		self.checkpoint_dir = checkpoint_dir
		self.dashboard_client = dashboard_client

		self._flow_id: Optional[int] = None
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

		# Register with dashboard if client available
		if self.dashboard_client:
			try:
				seed_checkpoint_id = None
				if cfg.seed_checkpoint_path:
					# TODO: Look up checkpoint ID from path
					pass

				self._flow_id = self.dashboard_client.create_flow(
					cfg.to_api_config(),
					seed_checkpoint_id=seed_checkpoint_id,
				)
				self.dashboard_client.flow_started(self._flow_id)
			except Exception as e:
				self.log(f"Warning: Failed to register flow with dashboard: {e}")

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
					continue

				# Create experiment checkpoint directory
				exp_checkpoint_dir = None
				if self.checkpoint_dir:
					exp_checkpoint_dir = self.checkpoint_dir / f"exp_{idx:02d}"

				# Create and run experiment
				experiment = Experiment(
					config=exp_config,
					evaluator=self.evaluator,
					logger=self.log,
					checkpoint_dir=exp_checkpoint_dir,
					dashboard_client=self.dashboard_client,
					experiment_id=None,  # TODO: Get from dashboard
				)

				result = experiment.run(
					initial_genome=current_genome,
					initial_fitness=current_fitness if exp_config.experiment_type == "ts" else None,
					initial_population=current_population,
					initial_threshold=current_threshold,
				)

				self._results.append(result)

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

		except Exception as e:
			# Flow failed
			if self.dashboard_client and self._flow_id:
				try:
					self.dashboard_client.flow_failed(self._flow_id, str(e))
				except Exception:
					pass
			raise

		elapsed = time.time() - start_time

		# Get final result
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
