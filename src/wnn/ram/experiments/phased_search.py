"""
Phased Architecture Search Runner

Encapsulates the three-phase optimization approach:
1. Phase 1: Optimize neurons (bits fixed)
2. Phase 2: Optimize bits (neurons from Phase 1)
3. Phase 3: Optimize connections (architecture from Phase 2)

Each phase uses GA then TS refinement.
"""

import gzip
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.core.reporting import OptimizationResultsTable, PhaseComparisonTable, PhaseMetrics
from wnn.ram.core import bits_needed
from wnn.ram.architecture.cached_evaluator import CachedEvaluator

# Optional dashboard integration
try:
	from wnn.ram.experiments.dashboard_client import DashboardClient, FlowConfig as APIFlowConfig
	HAS_DASHBOARD = True
except ImportError:
	HAS_DASHBOARD = False
	DashboardClient = None


@dataclass
class PhasedSearchConfig:
	"""Configuration for phased architecture search."""

	# Data configuration
	context_size: int = 4
	token_parts: int = 3  # Number of subsets for rotation

	# GA configuration
	ga_generations: int = 100
	population_size: int = 50

	# TS configuration
	ts_iterations: int = 200
	neighbors_per_iter: int = 50

	# Early stopping
	patience: int = 10
	check_interval: int = 10  # Check every N generations/iterations

	# Architecture defaults (used when tiered config not specified)
	default_bits: int = 8
	default_neurons: int = 5

	# Tiered architecture configuration
	# Format: list of (num_clusters, neurons, bits) tuples
	# e.g., [(100, 15, 20), (400, 10, 12), (None, 5, 8)]
	# None in num_clusters means "rest of vocabulary"
	tier_config: Optional[list[tuple[Optional[int], int, int]]] = None

	# Phase order: "neurons_first" (default) or "bits_first"
	phase_order: str = "neurons_first"

	# Tier0-only optimization: only mutate clusters in tier0
	optimize_tier0_only: bool = False

	# Seed only: use seed genomes as-is without mutation expansion (for pass 2+)
	seed_only: bool = False

	# Fitness percentile filter (None = disabled, 0.75 = keep top 75% by fitness)
	fitness_percentile: Optional[float] = None

	# Random seed (None = time-based)
	rotation_seed: Optional[int] = None

	# Logging
	log_path: Optional[str] = None

	def get_tier_boundaries(self, vocab_size: int) -> list[int]:
		"""
		Get tier boundaries as cluster indices.

		Returns:
			List of boundary indices. For tier_config [(100, n, b), (400, n, b), (None, n, b)]:
			returns [100, 500, vocab_size] meaning tier0=[0:100), tier1=[100:500), tier2=[500:vocab_size)
		"""
		if not self.tier_config:
			return [vocab_size]  # Single tier = all clusters

		boundaries = []
		cumulative = 0
		for tier_spec in self.tier_config:
			num_clusters, _, _ = tier_spec
			if num_clusters is None:
				# "rest" tier - goes to end
				boundaries.append(vocab_size)
			else:
				cumulative += num_clusters
				boundaries.append(min(cumulative, vocab_size))
		return boundaries

	def get_tier0_clusters(self, vocab_size: int) -> int:
		"""Get the number of clusters in tier0 (for tier0-only optimization)."""
		if not self.tier_config:
			return vocab_size  # No tiers = all clusters
		num_clusters = self.tier_config[0][0]
		return num_clusters if num_clusters is not None else vocab_size

	def create_tiered_genome(self, vocab_size: int) -> 'ClusterGenome':
		"""
		Create a ClusterGenome with tiered bits/neurons configuration.

		Uses tier_config if specified, otherwise creates uniform genome
		with default_bits and default_neurons.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		if not self.tier_config:
			return ClusterGenome.create_uniform(
				num_clusters=vocab_size,
				bits=self.default_bits,
				neurons=self.default_neurons,
			)

		# Build per-cluster bits and neurons arrays
		bits_per_cluster = []
		neurons_per_cluster = []
		cluster_idx = 0

		for tier_spec in self.tier_config:
			num_clusters, neurons, bits = tier_spec
			if num_clusters is None:
				# Fill rest of vocabulary
				count = vocab_size - cluster_idx
			else:
				count = min(num_clusters, vocab_size - cluster_idx)

			bits_per_cluster.extend([bits] * count)
			neurons_per_cluster.extend([neurons] * count)
			cluster_idx += count

			if cluster_idx >= vocab_size:
				break

		# Pad if needed (shouldn't happen with proper config)
		while len(bits_per_cluster) < vocab_size:
			bits_per_cluster.append(self.default_bits)
			neurons_per_cluster.append(self.default_neurons)

		return ClusterGenome(
			bits_per_cluster=bits_per_cluster[:vocab_size],
			neurons_per_cluster=neurons_per_cluster[:vocab_size],
		)

	def create_tiered_random_genome(
		self,
		vocab_size: int,
		rng: 'random.Random',
		bits_variation: int = 2,
		neurons_variation: int = 2,
		min_bits: int = 4,
		max_bits: int = 24,
		min_neurons: int = 1,
		max_neurons: int = 20,
	) -> 'ClusterGenome':
		"""
		Create a tiered genome with random variations within tier bounds.

		Each cluster gets bits/neurons randomly varied within ±variation of
		the tier's target value, respecting min/max bounds.

		Args:
			vocab_size: Number of clusters
			rng: Random number generator for reproducibility
			bits_variation: Max deviation from tier's bits (default ±2)
			neurons_variation: Max deviation from tier's neurons (default ±2)
			min_bits/max_bits: Hard bounds for bits
			min_neurons/max_neurons: Hard bounds for neurons

		Returns:
			ClusterGenome with randomized tiered configuration
		"""
		import random
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		if not self.tier_config:
			# No tier config - create uniform with random variation
			bits = max(min_bits, min(max_bits, self.default_bits + rng.randint(-bits_variation, bits_variation)))
			neurons = max(min_neurons, min(max_neurons, self.default_neurons + rng.randint(-neurons_variation, neurons_variation)))
			return ClusterGenome.create_uniform(
				num_clusters=vocab_size,
				bits=bits,
				neurons=neurons,
			)

		# Build per-cluster bits and neurons arrays with random variation
		bits_per_cluster = []
		neurons_per_cluster = []
		cluster_idx = 0

		for tier_spec in self.tier_config:
			num_clusters, tier_neurons, tier_bits = tier_spec
			if num_clusters is None:
				count = vocab_size - cluster_idx
			else:
				count = min(num_clusters, vocab_size - cluster_idx)

			for _ in range(count):
				# Random bits within ±variation of tier target, clamped to bounds
				b = tier_bits + rng.randint(-bits_variation, bits_variation)
				b = max(min_bits, min(max_bits, b))
				bits_per_cluster.append(b)

				# Random neurons within ±variation of tier target, clamped to bounds
				n = tier_neurons + rng.randint(-neurons_variation, neurons_variation)
				n = max(min_neurons, min(max_neurons, n))
				neurons_per_cluster.append(n)

			cluster_idx += count
			if cluster_idx >= vocab_size:
				break

		# Pad if needed
		while len(bits_per_cluster) < vocab_size:
			bits_per_cluster.append(self.default_bits)
			neurons_per_cluster.append(self.default_neurons)

		return ClusterGenome(
			bits_per_cluster=bits_per_cluster[:vocab_size],
			neurons_per_cluster=neurons_per_cluster[:vocab_size],
		)

	def to_yaml(self) -> str:
		"""Convert config to YAML string."""
		return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)

	def save_yaml(self, filepath: str) -> None:
		"""Save config to YAML file."""
		path = Path(filepath)
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, 'w') as f:
			f.write(self.to_yaml())

	@classmethod
	def from_yaml(cls, yaml_str: str) -> 'PhasedSearchConfig':
		"""Create config from YAML string."""
		data = yaml.safe_load(yaml_str)
		return cls(**data)

	@classmethod
	def load_yaml(cls, filepath: str) -> 'PhasedSearchConfig':
		"""Load config from YAML file."""
		with open(filepath, 'r') as f:
			return cls.from_yaml(f.read())


@dataclass
class PhaseResult:
	"""Result from a single optimization phase."""
	phase_name: str
	strategy_type: str  # "GA" or "TS"
	final_fitness: float
	final_accuracy: Optional[float]
	iterations_run: int
	best_genome: ClusterGenome
	final_population: Optional[list[ClusterGenome]]
	final_threshold: Optional[float]
	initial_fitness: Optional[float] = None
	initial_accuracy: Optional[float] = None

	def serialize(self) -> dict[str, Any]:
		"""Serialize phase result to dictionary."""
		data: dict[str, Any] = {
			"phase_name": self.phase_name,
			"strategy_type": self.strategy_type,
			"final_fitness": self.final_fitness,
			"final_accuracy": self.final_accuracy,
			"iterations_run": self.iterations_run,
			"best_genome": self.best_genome.serialize(),
			"final_threshold": self.final_threshold,
			"initial_fitness": self.initial_fitness,
			"initial_accuracy": self.initial_accuracy,
		}
		if self.final_population is not None:
			data["final_population"] = [g.serialize() for g in self.final_population]
		return data

	@classmethod
	def deserialize(cls, data: dict[str, Any]) -> 'PhaseResult':
		"""Deserialize phase result from dictionary."""
		final_population = None
		if "final_population" in data and data["final_population"] is not None:
			final_population = [ClusterGenome.deserialize(g) for g in data["final_population"]]

		return cls(
			phase_name=data["phase_name"],
			strategy_type=data["strategy_type"],
			final_fitness=data["final_fitness"],
			final_accuracy=data["final_accuracy"],
			iterations_run=data["iterations_run"],
			best_genome=ClusterGenome.deserialize(data["best_genome"]),
			final_population=final_population,
			final_threshold=data.get("final_threshold"),
			initial_fitness=data.get("initial_fitness"),
			initial_accuracy=data.get("initial_accuracy"),
		)

	def save(self, filepath: str, **metadata: Any) -> None:
		"""
		Save phase result to a compressed JSON file (.json.gz).

		Args:
			filepath: Output file path (auto-adds .gz if not present)
			**metadata: Additional metadata to include
		"""
		data: dict[str, Any] = {
			"phase_result": self.serialize(),
		}
		if metadata:
			data["_metadata"] = metadata

		path = Path(filepath)
		# Auto-add .gz extension for compression
		if not path.suffix == '.gz':
			path = path.with_suffix(path.suffix + '.gz')
		path.parent.mkdir(parents=True, exist_ok=True)

		# Write compressed (no indent for better compression)
		with gzip.open(path, 'wt', encoding='utf-8') as f:
			json.dump(data, f, separators=(',', ':'))

	@classmethod
	def load(cls, filepath: str) -> tuple['PhaseResult', dict[str, Any]]:
		"""
		Load phase result from a JSON file (compressed or uncompressed).

		Args:
			filepath: Input file path

		Returns:
			Tuple of (PhaseResult, metadata dict)
		"""
		path = Path(filepath)

		# Try compressed first, then uncompressed
		if path.suffix == '.gz' or path.with_suffix(path.suffix + '.gz').exists():
			gz_path = path if path.suffix == '.gz' else path.with_suffix(path.suffix + '.gz')
			with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
				data = json.load(f)
		else:
			with open(path, 'r') as f:
				data = json.load(f)

		result = cls.deserialize(data["phase_result"])
		metadata = data.get("_metadata", {})
		return result, metadata


# Phase name constants for checkpoint files (without extension - .json.gz added automatically)
# "neurons_first" order (default): neurons -> bits -> connections
PHASE_NAMES_NEURONS_FIRST = {
	"1a": "phase_1a_ga_neurons",
	"1b": "phase_1b_ts_neurons",
	"2a": "phase_2a_ga_bits",
	"2b": "phase_2b_ts_bits",
	"3a": "phase_3a_ga_connections",
	"3b": "phase_3b_ts_connections",
}

# "bits_first" order: bits -> neurons -> connections
PHASE_NAMES_BITS_FIRST = {
	"1a": "phase_1a_ga_bits",
	"1b": "phase_1b_ts_bits",
	"2a": "phase_2a_ga_neurons",
	"2b": "phase_2b_ts_neurons",
	"3a": "phase_3a_ga_connections",
	"3b": "phase_3b_ts_connections",
}

# For backwards compatibility
PHASE_NAMES = PHASE_NAMES_NEURONS_FIRST


class PhasedSearchRunner:
	"""
	Runs the phased architecture search.

	Encapsulates all the common logic for running phases, allowing
	different search strategies (standard, coarse-to-fine) to reuse the same code.
	"""

	def __init__(
		self,
		config: PhasedSearchConfig,
		logger: Callable[[str], None],
		checkpoint_dir: Optional[str] = None,
		dashboard_client: Optional["DashboardClient"] = None,
	):
		"""
		Initialize the runner.

		Args:
			config: Search configuration
			logger: Logging function
			checkpoint_dir: Directory to save/load checkpoints (None = no checkpoints)
			dashboard_client: Optional dashboard client for real-time updates
		"""
		self.config = config
		self.log = logger
		self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
		self.dashboard_client = dashboard_client
		self.evaluator: Optional[CachedEvaluator] = None
		self.vocab_size: int = 0
		self.total_input_bits: int = 0
		self.token_frequencies: list[int] = []
		self.results: dict[str, PhaseResult] = {}
		self._rotation_seed: Optional[int] = None
		self._flow_id: Optional[int] = None
		self._experiment_id: Optional[int] = None

	def save_checkpoint(self, phase_key: str, result: PhaseResult) -> Optional[str]:
		"""
		Save checkpoint after a phase completes (compressed .json.gz).

		Args:
			phase_key: Phase identifier (e.g., "1a", "1b", "2a")
			result: The phase result to save

		Returns:
			Path to saved checkpoint file, or None if checkpointing disabled
		"""
		if self.checkpoint_dir is None:
			return None

		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		phase_names = self._get_phase_names()
		# Use .json extension - save() will auto-add .gz
		filename = f"{phase_names.get(phase_key, phase_key)}.json"
		filepath = self.checkpoint_dir / filename

		result.save(
			str(filepath),
			phase_key=phase_key,
			rotation_seed=self._rotation_seed,
		)
		# Log actual saved path (with .gz)
		actual_path = filepath.with_suffix('.json.gz')
		self.log(f"  Checkpoint saved: {actual_path}")

		# Register with dashboard if client available
		if self.dashboard_client and self._experiment_id:
			try:
				is_final = phase_key == "3b"  # Final phase
				genome_stats = result.best_genome.stats() if result.best_genome else None
				self.dashboard_client.checkpoint_created(
					experiment_id=self._experiment_id,
					file_path=str(actual_path.absolute()),
					name=result.phase_name,
					final_fitness=result.final_fitness,
					final_accuracy=result.final_accuracy,
					iterations_run=result.iterations_run,
					genome_stats=genome_stats,
					is_final=is_final,
				)
			except Exception as e:
				self.log(f"  Warning: Failed to register checkpoint with dashboard: {e}")

		return str(actual_path)

	def load_checkpoint(self, phase_key: str) -> Optional[PhaseResult]:
		"""
		Load checkpoint for a specific phase (supports .json.gz and .json).

		Args:
			phase_key: Phase identifier (e.g., "1a", "1b", "2a")

		Returns:
			PhaseResult if checkpoint exists, None otherwise
		"""
		if self.checkpoint_dir is None:
			return None

		phase_names = self._get_phase_names()
		base_filename = phase_names.get(phase_key, phase_key)

		# Try compressed first, then uncompressed
		gz_path = self.checkpoint_dir / f"{base_filename}.json.gz"
		json_path = self.checkpoint_dir / f"{base_filename}.json"

		if gz_path.exists():
			result, metadata = PhaseResult.load(str(gz_path))
			self.log(f"  Loaded checkpoint: {gz_path}")
			return result
		elif json_path.exists():
			result, metadata = PhaseResult.load(str(json_path))
			self.log(f"  Loaded checkpoint: {json_path}")
			return result

		return None

	def get_resume_phase(self, resume_from: Optional[str] = None) -> Optional[str]:
		"""
		Determine which phase to resume from based on existing checkpoints.

		Args:
			resume_from: Explicit phase to resume from (e.g., "1b", "2a")
			            If None, finds the latest completed phase.

		Returns:
			Phase key to resume from, or None to start fresh
		"""
		if self.checkpoint_dir is None:
			return None

		if resume_from:
			# Verify the previous phase checkpoint exists
			return resume_from

		# Find latest completed phase (check both .json.gz and .json)
		phase_order = ["1a", "1b", "2a", "2b", "3a", "3b"]
		phase_names = self._get_phase_names()
		latest = None
		for phase_key in phase_order:
			base = phase_names.get(phase_key, phase_key)
			gz_exists = (self.checkpoint_dir / f"{base}.json.gz").exists()
			json_exists = (self.checkpoint_dir / f"{base}.json").exists()
			if gz_exists or json_exists:
				latest = phase_key
		return latest

	def setup(
		self,
		train_tokens: list[int],
		eval_tokens: list[int],
		vocab_size: int,
	) -> None:
		"""
		Initialize evaluator and prepare for search.

		Args:
			train_tokens: Training token IDs
			eval_tokens: Evaluation token IDs
			vocab_size: Vocabulary size
		"""
		import time

		self.vocab_size = vocab_size

		# Compute token frequencies
		self.log("Computing token frequencies from training data...")
		freq_counter = Counter(train_tokens)
		self.token_frequencies = [freq_counter.get(i, 0) for i in range(vocab_size)]
		self.log(f"  Tokens with freq > 0: {sum(1 for f in self.token_frequencies if f > 0):,}")

		# Create cluster ordering (sorted by frequency, most frequent first)
		cluster_order = sorted(range(vocab_size), key=lambda i: -self.token_frequencies[i])

		# Determine rotation seed (store for use in strategies)
		rotation_seed = self.config.rotation_seed
		if rotation_seed is None:
			rotation_seed = int(time.time() * 1000) % (2**32)
		self._rotation_seed = rotation_seed

		# Create cached evaluator
		self.log("")
		self.log("Creating cached evaluator with per-iteration rotation...")
		self.evaluator = CachedEvaluator(
			train_tokens=train_tokens,
			eval_tokens=eval_tokens,
			vocab_size=vocab_size,
			context_size=self.config.context_size,
			cluster_order=cluster_order,
			num_parts=self.config.token_parts,
			num_negatives=5,
			empty_value=0.0,
			seed=rotation_seed,
			log_path=self.config.log_path,
			use_hybrid=True,  # Parallel batch training + GPU eval (fast)
		)
		self.log(f"  {self.evaluator}")

		# Set WNN_LOG_PATH for Rust progress logging to same log file
		import os
		if self.config.log_path:
			os.environ["WNN_LOG_PATH"] = self.config.log_path

		# Compute input bits
		bits_per_token = bits_needed(vocab_size)
		self.total_input_bits = self.config.context_size * bits_per_token
		self.log("")
		self.log(f"Input encoding: {self.config.context_size} tokens × {bits_per_token} bits = {self.total_input_bits} bits")

	def _get_phase_names(self) -> dict[str, str]:
		"""Get the phase names dict based on configured phase_order."""
		if self.config.phase_order == "bits_first":
			return PHASE_NAMES_BITS_FIRST
		return PHASE_NAMES_NEURONS_FIRST

	def evaluate_population_full(
		self,
		population: list[ClusterGenome],
		k: int = 10,
	) -> PhaseMetrics | None:
		"""
		Evaluate a population on FULL validation data and return metrics.

		Returns PhaseMetrics with:
		- top-k mean CE and Acc
		- best CE genome's CE and Acc
		- best Acc genome's CE and Acc

		All evaluated on full validation data for fair comparison.

		Args:
			population: List of genomes to evaluate
			k: Number of genomes for top-k mean (default 10)

		Returns:
			PhaseMetrics or None if population is empty
		"""
		if not population:
			return None

		# Evaluate all genomes on full validation data
		full_evals: list[tuple[ClusterGenome, float, float]] = []
		for genome in population:
			ce, acc = self.evaluator.evaluate_single_full(genome)
			full_evals.append((genome, ce, acc))

		# Sort by CE (ascending) for best CE and top-k
		by_ce = sorted(full_evals, key=lambda x: x[1])

		# Sort by Acc (descending) for best Acc
		by_acc = sorted(full_evals, key=lambda x: -x[2])

		# Top-k mean (by CE)
		top_k = by_ce[:k]
		top_k_ce = sum(e[1] for e in top_k) / len(top_k)
		top_k_acc = sum(e[2] for e in top_k) / len(top_k)

		# Best by CE
		best_ce_genome, best_ce_ce, best_ce_acc = by_ce[0]

		# Best by Acc
		best_acc_genome, best_acc_ce, best_acc_acc = by_acc[0]

		return PhaseMetrics(
			phase_name="",  # Will be set by caller
			top_k_ce=top_k_ce,
			top_k_acc=top_k_acc,
			best_ce_ce=best_ce_ce,
			best_ce_acc=best_ce_acc,
			best_acc_ce=best_acc_ce,
			best_acc_acc=best_acc_acc,
			k=min(k, len(top_k)),
		)

	def evaluate_single_genome_as_metrics(
		self,
		genome: ClusterGenome,
		phase_name: str = "",
	) -> PhaseMetrics:
		"""
		Evaluate a single genome on full validation and return as PhaseMetrics.

		For baseline where we only have one genome, all three metrics
		(top-k, best CE, best Acc) are the same.

		Args:
			genome: Genome to evaluate
			phase_name: Name for the phase

		Returns:
			PhaseMetrics with all metrics set to the same values
		"""
		ce, acc = self.evaluator.evaluate_single_full(genome)
		return PhaseMetrics(
			phase_name=phase_name,
			top_k_ce=ce,
			top_k_acc=acc,
			best_ce_ce=ce,
			best_ce_acc=acc,
			best_acc_ce=ce,
			best_acc_acc=acc,
			k=1,
		)

	def _get_dashboard_phase_config(self, phase_index: int) -> Optional[dict]:
		"""
		Fetch the current config for a phase from the dashboard.

		This enables dynamic modification of pending phases while a flow is running.
		Users can edit, delete, or add phases via the dashboard UI.

		Args:
			phase_index: Index of the phase in the flow's experiments array

		Returns:
			Phase config dict if phase exists, None if deleted or dashboard unavailable
		"""
		if not self.dashboard_client or not self._flow_id:
			return None

		try:
			flow = self.dashboard_client.get_flow(self._flow_id)
			experiments = flow.get("config", {}).get("experiments", [])

			if phase_index >= len(experiments):
				return None  # Phase was deleted or doesn't exist

			return experiments[phase_index]
		except Exception as e:
			self.log(f"Warning: Failed to fetch phase config from dashboard: {e}")
			return None

	def _get_additional_phases_from_dashboard(self, original_count: int) -> list[dict]:
		"""
		Check if any new phases were added to the flow via dashboard.

		Args:
			original_count: Number of phases originally configured

		Returns:
			List of new phase configs (empty if none added)
		"""
		if not self.dashboard_client or not self._flow_id:
			return []

		try:
			flow = self.dashboard_client.get_flow(self._flow_id)
			experiments = flow.get("config", {}).get("experiments", [])

			if len(experiments) > original_count:
				return experiments[original_count:]
			return []
		except Exception as e:
			self.log(f"Warning: Failed to check for additional phases: {e}")
			return []

	def run_phase(
		self,
		phase_name: str,
		strategy_type: OptimizerStrategyType,
		optimize_bits: bool,
		optimize_neurons: bool,
		optimize_connections: bool,
		initial_genome: Optional[ClusterGenome] = None,
		initial_fitness: Optional[float] = None,
		initial_population: Optional[list[ClusterGenome]] = None,
		initial_threshold: Optional[float] = None,
	) -> PhaseResult:
		"""
		Run a single optimization phase.

		Args:
			phase_name: Display name for logging
			strategy_type: GA or TS
			optimize_bits: Whether to optimize bits per cluster
			optimize_neurons: Whether to optimize neurons per cluster
			optimize_connections: Whether to optimize connections
			initial_genome: Starting genome (for seeding)
			initial_fitness: Fitness of initial genome (required for TS)
			initial_population: Population to seed from
			initial_threshold: Starting accuracy threshold

		Returns:
			PhaseResult with optimization results
		"""
		cfg = self.config

		self.log("")
		self.log(f"{'='*60}")
		self.log(f"  {phase_name}")
		self.log(f"{'='*60}")
		self.log(f"  optimize_bits={optimize_bits}, optimize_neurons={optimize_neurons}")
		self.log(f"  optimize_connections={optimize_connections}")
		self.log(f"  Per-iteration rotation: {self.evaluator.num_parts} subsets")
		if initial_genome:
			self.log(f"  Starting from previous best: {initial_genome}")
		if initial_population:
			self.log(f"  Seeding from {len(initial_population)} genomes from previous phase")
		self.log("")

		# Determine if GA or TS
		is_ga = strategy_type == OptimizerStrategyType.ARCHITECTURE_GA

		# Build strategy kwargs
		strategy_kwargs = {
			"strategy_type": strategy_type,
			"num_clusters": self.vocab_size,
			"optimize_bits": optimize_bits,
			"optimize_neurons": optimize_neurons,
			"optimize_connections": optimize_connections,
			"default_bits": cfg.default_bits,
			"default_neurons": cfg.default_neurons,
			"token_frequencies": self.token_frequencies,
			"total_input_bits": self.total_input_bits,
			"batch_evaluator": self.evaluator,
			"logger": self.log,
			"patience": cfg.patience,
			"check_interval": cfg.check_interval,  # Check every N generations/iterations
			"initial_threshold": initial_threshold,
			"fitness_percentile": cfg.fitness_percentile,  # Fitness percentile filter (None = disabled)
			"seed": self._rotation_seed,  # Use rotation_seed for strategy RNG
		}

		# Tier0-only optimization: only mutate top N clusters
		if cfg.optimize_tier0_only:
			tier0_clusters = cfg.get_tier0_clusters(self.vocab_size)
			strategy_kwargs["mutable_clusters"] = tier0_clusters
			self.log(f"  Tier0-only mode: mutating only first {tier0_clusters} clusters")

		if is_ga:
			strategy_kwargs["generations"] = cfg.ga_generations
			strategy_kwargs["population_size"] = cfg.population_size
			strategy_kwargs["phase_name"] = phase_name
			strategy_kwargs["seed_only"] = cfg.seed_only
			# Add checkpoint config if checkpoint_dir is set
			if self.checkpoint_dir:
				from wnn.ram.strategies.connectivity.architecture_strategies import CheckpointConfig
				# Create per-phase checkpoint directory
				phase_checkpoint_dir = self.checkpoint_dir / phase_name.replace(" ", "_").replace(":", "")
				strategy_kwargs["checkpoint_config"] = CheckpointConfig(
					enabled=True,
					interval=50,  # Save every 50 generations
					checkpoint_dir=phase_checkpoint_dir,
					filename_prefix="ga_checkpoint",
				)
			# First GA phase (no population from previous phase): generate fresh population
			if not initial_population:
				if cfg.tier_config:
					# Generate tiered genomes that respect tier structure
					# Only add variation to dimensions being optimized
					import random
					rng = random.Random(self._rotation_seed)

					# If bits are not being optimized, use exact tier bits (no variation)
					# This ensures minimal config groups during neurons-only optimization
					bits_variation = 2 if optimize_bits else 0
					neurons_variation = 2 if optimize_neurons else 0

					tiered_population = [
						cfg.create_tiered_random_genome(
							self.vocab_size, rng,
							bits_variation=bits_variation,
							neurons_variation=neurons_variation,
						)
						for _ in range(cfg.population_size)
					]
					# Initialize connections for all genomes
					for g in tiered_population:
						if not g.has_connections():
							g.initialize_connections(self.total_input_bits)
					initial_population = tiered_population
					varied_dims = []
					if bits_variation > 0: varied_dims.append("bits")
					if neurons_variation > 0: varied_dims.append("neurons")
					variation_desc = f" (varied: {', '.join(varied_dims)})" if varied_dims else " (exact tier config)"
					self.log(f"  Generated {len(tiered_population)} tiered genomes{variation_desc}")
				else:
					# No tier config - use uniform random genomes
					strategy_kwargs["fresh_population"] = True
		else:
			strategy_kwargs["iterations"] = cfg.ts_iterations
			strategy_kwargs["neighbors_per_iter"] = cfg.neighbors_per_iter
			strategy_kwargs["total_neighbors_size"] = cfg.population_size

		strategy = OptimizerStrategyFactory.create(**strategy_kwargs)

		# Run optimization
		if is_ga:
			# GA: use initial_population if available, else [initial_genome], else None (fresh)
			if initial_population:
				seed_pop = initial_population
			elif initial_genome is not None:
				seed_pop = [initial_genome]
			else:
				seed_pop = None
			result = strategy.optimize(
				evaluate_fn=None,
				initial_population=seed_pop,
			)
		else:
			# TS: requires initial_genome
			if initial_genome is not None:
				result = strategy.optimize(
					evaluate_fn=None,
					initial_genome=initial_genome,
					initial_fitness=initial_fitness,
					initial_neighbors=initial_population,
				)
			else:
				result = strategy.optimize(evaluate_fn=None)

		self.log("")
		self.log(f"{phase_name} Result:")
		self.log(f"  Best fitness (CE): {result.final_fitness:.4f}")
		self.log(f"  Best genome: {result.best_genome}")
		self.log(f"  Generations/Iterations: {result.iterations_run}")

		return PhaseResult(
			phase_name=phase_name,
			strategy_type="GA" if is_ga else "TS",
			final_fitness=result.final_fitness,
			final_accuracy=result.final_accuracy,
			iterations_run=result.iterations_run,
			best_genome=result.best_genome,
			final_population=result.final_population,
			final_threshold=result.final_threshold,
			initial_fitness=result.initial_fitness,
			initial_accuracy=result.initial_accuracy,
		)

	def print_phase_comparison(
		self,
		phase_metrics_list: list[PhaseMetrics],
	) -> None:
		"""
		Print cumulative phase comparison table.

		All metrics are evaluated on FULL validation data for fair comparison.

		Args:
			phase_metrics_list: List of PhaseMetrics (baseline + completed phases)
		"""
		table = PhaseComparisonTable("Phase Comparison (Full Validation)")
		for pm in phase_metrics_list:
			table.add_phase(pm)
		self.log("")
		table.print(self.log)

	def get_phase_metrics(
		self,
		phase_result: PhaseResult,
		k: int = 10,
	) -> PhaseMetrics:
		"""
		Evaluate a phase's population on FULL validation and return PhaseMetrics.

		Args:
			phase_result: The phase result with final_population
			k: Number of genomes for top-k mean

		Returns:
			PhaseMetrics with all metrics evaluated on full validation
		"""
		pop = phase_result.final_population

		if pop and len(pop) > 0:
			# Evaluate all genomes on full validation data
			full_evals: list[tuple[ClusterGenome, float, float]] = []
			for genome in pop:
				ce, acc = self.evaluator.evaluate_single_full(genome)
				full_evals.append((genome, ce, acc))

			# Sort by CE (ascending) for best CE and top-k
			by_ce = sorted(full_evals, key=lambda x: x[1])

			# Sort by Acc (descending) for best Acc
			by_acc = sorted(full_evals, key=lambda x: -x[2])

			# Top-k mean (by CE)
			top_k = by_ce[:k]
			top_k_ce = sum(e[1] for e in top_k) / len(top_k)
			top_k_acc = sum(e[2] for e in top_k) / len(top_k)

			# Best by CE
			_, best_ce_ce, best_ce_acc = by_ce[0]

			# Best by Acc
			_, best_acc_ce, best_acc_acc = by_acc[0]

			return PhaseMetrics(
				phase_name=phase_result.phase_name,
				top_k_ce=top_k_ce,
				top_k_acc=top_k_acc,
				best_ce_ce=best_ce_ce,
				best_ce_acc=best_ce_acc,
				best_acc_ce=best_acc_ce,
				best_acc_acc=best_acc_acc,
				k=min(k, len(top_k)),
			)
		else:
			# No population - use best genome for all metrics
			ce, acc = self.evaluator.evaluate_single_full(phase_result.best_genome)
			return PhaseMetrics(
				phase_name=phase_result.phase_name,
				top_k_ce=ce,
				top_k_acc=acc,
				best_ce_ce=ce,
				best_ce_acc=acc,
				best_acc_ce=ce,
				best_acc_acc=acc,
				k=1,
			)

	def run_all_phases(
		self,
		seed_genome: Optional[ClusterGenome] = None,
		seed_population: Optional[list[ClusterGenome]] = None,
		seed_threshold: Optional[float] = None,
		resume_from: Optional[str] = None,
	) -> dict[str, Any]:
		"""
		Run all phases: 1a -> 1b -> 2a -> 2b -> 3a -> 3b -> final evaluation.

		Phase order depends on config.phase_order:
		- "neurons_first" (default): neurons -> bits -> connections
		- "bits_first": bits -> neurons -> connections

		Args:
			seed_genome: Optional genome to seed Phase 1a from (used if no population)
			seed_population: Optional population to seed Phase 1a from (from previous pass)
			seed_threshold: Optional progressive threshold to continue from (from previous pass)
			resume_from: Phase to resume from (e.g., "1b", "2a"). If provided,
			            loads checkpoint from previous phase and continues.

		Returns:
			Dictionary with all results
		"""
		results: dict[str, Any] = {}
		completed_phases: list[PhaseResult] = []

		# Determine phase configuration based on phase_order
		if self.config.phase_order == "bits_first":
			# bits -> neurons -> connections
			phase_specs = [
				("1a", "Phase 1a: GA Bits Only", OptimizerStrategyType.ARCHITECTURE_GA, True, False, False),
				("1b", "Phase 1b: TS Bits Only (refine)", OptimizerStrategyType.ARCHITECTURE_TS, True, False, False),
				("2a", "Phase 2a: GA Neurons Only", OptimizerStrategyType.ARCHITECTURE_GA, False, True, False),
				("2b", "Phase 2b: TS Neurons Only (refine)", OptimizerStrategyType.ARCHITECTURE_TS, False, True, False),
				("3a", "Phase 3a: GA Connections Only", OptimizerStrategyType.ARCHITECTURE_GA, False, False, True),
				("3b", "Phase 3b: TS Connections Only (refine)", OptimizerStrategyType.ARCHITECTURE_TS, False, False, True),
			]
			self.log(f"Phase order: bits → neurons → connections")
		else:
			# neurons -> bits -> connections (default)
			phase_specs = [
				("1a", "Phase 1a: GA Neurons Only", OptimizerStrategyType.ARCHITECTURE_GA, False, True, False),
				("1b", "Phase 1b: TS Neurons Only (refine)", OptimizerStrategyType.ARCHITECTURE_TS, False, True, False),
				("2a", "Phase 2a: GA Bits Only", OptimizerStrategyType.ARCHITECTURE_GA, True, False, False),
				("2b", "Phase 2b: TS Bits Only (refine)", OptimizerStrategyType.ARCHITECTURE_TS, True, False, False),
				("3a", "Phase 3a: GA Connections Only", OptimizerStrategyType.ARCHITECTURE_GA, False, False, True),
				("3b", "Phase 3b: TS Connections Only (refine)", OptimizerStrategyType.ARCHITECTURE_TS, False, False, True),
			]
			self.log(f"Phase order: neurons → bits → connections")

		# Phase order for resume logic
		phase_order = ["1a", "1b", "2a", "2b", "3a", "3b"]
		start_idx = 0

		# Handle resume
		if resume_from:
			if resume_from not in phase_order:
				raise ValueError(f"Invalid resume_from phase: {resume_from}. Must be one of {phase_order}")
			start_idx = phase_order.index(resume_from)
			self.log(f"Resuming from phase {resume_from}")

			# Load all previous phase results from checkpoints
			for i, phase_key in enumerate(phase_order[:start_idx]):
				prev_result = self.load_checkpoint(phase_key)
				if prev_result is None:
					raise ValueError(f"Cannot resume from {resume_from}: missing checkpoint for {phase_key}")
				completed_phases.append(prev_result)
				self.log(f"  Loaded checkpoint for phase {phase_key}: CE={prev_result.final_fitness:.4f}")

		# =====================================================================
		# Dashboard Integration: Create flow and experiment
		# =====================================================================
		if self.dashboard_client:
			try:
				# Build flow config for dashboard
				flow_name = f"Phased Search ({self.config.phase_order})"
				if self.checkpoint_dir:
					flow_name = f"{self.checkpoint_dir.name}"

				api_config = APIFlowConfig(
					name=flow_name,
					experiments=[{
						"name": spec[1],
						"experiment_type": "ga" if "GA" in spec[1] else "ts",
						"optimize_bits": spec[3],
						"optimize_neurons": spec[4],
						"optimize_connections": spec[5],
					} for spec in phase_specs],
					description=f"Phase order: {self.config.phase_order}, Patience: {self.config.patience}",
					params={
						"phase_order": self.config.phase_order,
						"patience": self.config.patience,
						"ga_generations": self.config.ga_generations,
						"ts_iterations": self.config.ts_iterations,
						"tier_config": self.config.tier_config,
					},
				)

				self._flow_id = self.dashboard_client.create_flow(api_config)
				self.dashboard_client.flow_started(self._flow_id)

				# Create experiment for this run
				self._experiment_id = self.dashboard_client.create_experiment(
					name=flow_name,
					log_path=str(self.config.log_path) if self.config.log_path else "",
					config={"phase_order": self.config.phase_order},
				)

				# Link experiment to flow
				self.dashboard_client.link_experiment_to_flow(
					flow_id=self._flow_id,
					experiment_id=self._experiment_id,
					sequence_order=0,
				)

				self.log(f"Dashboard: Created flow {self._flow_id}, experiment {self._experiment_id}")
			except Exception as e:
				self.log(f"Warning: Failed to register with dashboard: {e}")
				self._flow_id = None
				self._experiment_id = None

		# =====================================================================
		# Baseline: Evaluate initial genome on full validation
		# =====================================================================
		# Cumulative list of PhaseMetrics for comparison table
		phase_metrics_list: list[PhaseMetrics] = []

		if start_idx <= 0:
			# Create tiered genome if config specifies tiers, else uniform
			if seed_genome:
				baseline_genome = seed_genome
				self.log("Evaluating seed genome as baseline...")
			elif self.config.tier_config:
				baseline_genome = self.config.create_tiered_genome(self.vocab_size)
				self.log("Evaluating tiered genome as baseline...")
				self.log(f"  Tier config: {self.config.tier_config}")
			else:
				from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
				baseline_genome = ClusterGenome.create_uniform(
					num_clusters=self.vocab_size,
					bits=self.config.default_bits,
					neurons=self.config.default_neurons,
				)
				self.log("Evaluating default genome as baseline...")

			# Initialize connections if not present (critical for reproducible evaluation)
			if not baseline_genome.has_connections():
				baseline_genome.initialize_connections(self.total_input_bits)
				self.log(f"  Initialized {baseline_genome.total_connections()} connections")

			baseline_ce, baseline_acc = self.evaluator.evaluate_single_full(baseline_genome)
			self.log(f"  Baseline CE: {baseline_ce:.4f}, Acc: {baseline_acc:.2%}")
			self.log("")

			# Add baseline to cumulative metrics
			baseline_metrics = PhaseMetrics(
				phase_name="Baseline",
				top_k_ce=baseline_ce,
				top_k_acc=baseline_acc,
				best_ce_ce=baseline_ce,
				best_ce_acc=baseline_acc,
				best_acc_ce=baseline_ce,
				best_acc_acc=baseline_acc,
				k=1,
			)
			phase_metrics_list.append(baseline_metrics)

			# Use tiered genome as seed if no explicit seed provided
			if seed_genome is None and self.config.tier_config:
				seed_genome = baseline_genome
		else:
			# When resuming, we don't have the baseline - use None
			baseline_ce, baseline_acc = None, None

		# =====================================================================
		# Run all phases dynamically based on phase_specs
		# =====================================================================
		prev_result: Optional[PhaseResult] = None

		for idx, (phase_key, phase_name, strategy_type, opt_bits, opt_neurons, opt_conns) in enumerate(phase_specs):
			if start_idx > idx:
				prev_result = completed_phases[idx]
				continue

			# =====================================================================
			# Dynamic Config: Check dashboard for phase updates before each phase
			# =====================================================================
			dashboard_phase = self._get_dashboard_phase_config(idx)
			if dashboard_phase is not None:
				# Phase exists in dashboard - check for modifications
				db_type = dashboard_phase.get("experiment_type", "ga")
				db_opt_bits = dashboard_phase.get("optimize_bits", False)
				db_opt_neurons = dashboard_phase.get("optimize_neurons", False)
				db_opt_conns = dashboard_phase.get("optimize_connections", False)

				# Check if config changed
				new_strategy_type = OptimizerStrategyType.ARCHITECTURE_GA if db_type == "ga" else OptimizerStrategyType.ARCHITECTURE_TS
				if (new_strategy_type != strategy_type or
					db_opt_bits != opt_bits or
					db_opt_neurons != opt_neurons or
					db_opt_conns != opt_conns):
					self.log(f"Dashboard: Phase {phase_key} config updated!")
					self.log(f"  Old: {strategy_type.name}, bits={opt_bits}, neurons={opt_neurons}, conns={opt_conns}")
					self.log(f"  New: {new_strategy_type.name}, bits={db_opt_bits}, neurons={db_opt_neurons}, conns={db_opt_conns}")
					strategy_type = new_strategy_type
					opt_bits = db_opt_bits
					opt_neurons = db_opt_neurons
					opt_conns = db_opt_conns
					# Update phase_name to reflect change
					opt_target = "Bits" if opt_bits else ("Neurons" if opt_neurons else "Connections")
					phase_name = f"Phase {phase_key}: {'GA' if db_type == 'ga' else 'TS'} {opt_target}"
			elif self.dashboard_client and self._flow_id:
				# Dashboard is available but phase was deleted - skip it
				self.log(f"Dashboard: Phase {phase_key} was deleted - skipping")
				continue

			# Determine initial genome/population
			if idx == 0:
				init_genome = seed_genome
				init_population = seed_population
				init_threshold = seed_threshold
				init_fitness = None
			else:
				init_genome = prev_result.best_genome
				init_population = prev_result.final_population
				init_threshold = prev_result.final_threshold
				init_fitness = prev_result.final_fitness if strategy_type == OptimizerStrategyType.ARCHITECTURE_TS else None

			# Notify dashboard that phase is starting
			dashboard_phase_id = None
			if self.dashboard_client and self._experiment_id:
				try:
					# Build phase_type string (e.g., "ga_neurons", "ts_bits")
					opt_target = "neurons" if opt_neurons else ("bits" if opt_bits else "connections")
					phase_type_str = f"{'ga' if strategy_type == OptimizerStrategyType.ARCHITECTURE_GA else 'ts'}_{opt_target}"
					dashboard_phase_id = self.dashboard_client.phase_started(
						experiment_id=self._experiment_id,
						name=phase_name,
						phase_type=phase_type_str,
					)
				except Exception as e:
					self.log(f"Warning: Failed to notify dashboard of phase start: {e}")

			phase_result = self.run_phase(
				phase_name=phase_name,
				strategy_type=strategy_type,
				optimize_bits=opt_bits,
				optimize_neurons=opt_neurons,
				optimize_connections=opt_conns,
				initial_genome=init_genome,
				initial_fitness=init_fitness,
				initial_population=init_population,
				initial_threshold=init_threshold,
			)

			# Notify dashboard that phase completed
			if self.dashboard_client and dashboard_phase_id:
				try:
					self.dashboard_client.phase_completed(dashboard_phase_id)
				except Exception as e:
					self.log(f"Warning: Failed to notify dashboard of phase completion: {e}")

			# Store result in dict (use phase1/2/3 naming for backwards compat)
			phase_num = (idx // 2) + 1
			phase_type = "ga" if strategy_type == OptimizerStrategyType.ARCHITECTURE_GA else "ts"
			results[f"phase{phase_num}_{phase_type}"] = {
				"fitness": phase_result.final_fitness,
				"accuracy": phase_result.final_accuracy,
				"iterations": phase_result.iterations_run,
			}

			completed_phases.append(phase_result)

			# Evaluate phase on full validation and add to cumulative table
			self.log("")
			self.log(f"Evaluating {phase_name} population on full validation...")
			phase_metrics = self.get_phase_metrics(phase_result, k=10)
			phase_metrics_list.append(phase_metrics)

			# Print cumulative comparison table
			self.print_phase_comparison(phase_metrics_list)

			self.save_checkpoint(phase_key, phase_result)
			prev_result = phase_result

		# =====================================================================
		# Check for additional phases added via dashboard
		# =====================================================================
		original_phase_count = len(phase_specs)
		additional_phases = self._get_additional_phases_from_dashboard(original_phase_count)

		for add_idx, add_phase in enumerate(additional_phases):
			phase_idx = original_phase_count + add_idx
			phase_key = f"extra_{add_idx + 1}"

			db_type = add_phase.get("experiment_type", "ga")
			opt_bits = add_phase.get("optimize_bits", False)
			opt_neurons = add_phase.get("optimize_neurons", False)
			opt_conns = add_phase.get("optimize_connections", False)

			strategy_type = OptimizerStrategyType.ARCHITECTURE_GA if db_type == "ga" else OptimizerStrategyType.ARCHITECTURE_TS
			opt_target = "Bits" if opt_bits else ("Neurons" if opt_neurons else "Connections")
			phase_name = add_phase.get("name", f"Extra Phase {add_idx + 1}: {'GA' if db_type == 'ga' else 'TS'} {opt_target}")

			self.log(f"Dashboard: Running additional phase {phase_key}: {phase_name}")

			# Notify dashboard that phase is starting
			dashboard_phase_id = None
			if self.dashboard_client and self._experiment_id:
				try:
					opt_target_lower = "neurons" if opt_neurons else ("bits" if opt_bits else "connections")
					phase_type_str = f"{db_type}_{opt_target_lower}"
					dashboard_phase_id = self.dashboard_client.phase_started(
						experiment_id=self._experiment_id,
						name=phase_name,
						phase_type=phase_type_str,
					)
				except Exception as e:
					self.log(f"Warning: Failed to notify dashboard of phase start: {e}")

			init_genome = prev_result.best_genome if prev_result else seed_genome
			init_population = prev_result.final_population if prev_result else seed_population
			init_threshold = prev_result.final_threshold if prev_result else seed_threshold
			init_fitness = prev_result.final_fitness if prev_result and strategy_type == OptimizerStrategyType.ARCHITECTURE_TS else None

			phase_result = self.run_phase(
				phase_name=phase_name,
				strategy_type=strategy_type,
				optimize_bits=opt_bits,
				optimize_neurons=opt_neurons,
				optimize_connections=opt_conns,
				initial_genome=init_genome,
				initial_fitness=init_fitness,
				initial_population=init_population,
				initial_threshold=init_threshold,
			)

			# Notify dashboard that phase completed
			if self.dashboard_client and dashboard_phase_id:
				try:
					self.dashboard_client.phase_completed(dashboard_phase_id)
				except Exception as e:
					self.log(f"Warning: Failed to notify dashboard of phase completion: {e}")

			results[f"extra_{add_idx + 1}"] = {
				"fitness": phase_result.final_fitness,
				"accuracy": phase_result.final_accuracy,
				"iterations": phase_result.iterations_run,
			}

			completed_phases.append(phase_result)

			# Evaluate and add to metrics
			self.log("")
			self.log(f"Evaluating {phase_name} population on full validation...")
			phase_metrics = self.get_phase_metrics(phase_result, k=10)
			phase_metrics_list.append(phase_metrics)
			self.print_phase_comparison(phase_metrics_list)

			self.save_checkpoint(phase_key, phase_result)
			prev_result = phase_result

		# Get final result (last phase)
		p3b = completed_phases[-1]

		# =====================================================================
		# Final Evaluation with FULL training tokens
		# =====================================================================
		self.log("")
		self.log(f"{'='*60}")
		self.log("  Final Evaluation with FULL Training Data")
		self.log(f"{'='*60}")

		final_ce, final_acc = self.evaluator.evaluate_single_full(p3b.best_genome)
		self.log("")
		self.log(f"  Final genome: {p3b.best_genome}")
		self.log(f"  Final CE (full data): {final_ce:.4f}")
		self.log(f"  Final Accuracy: {final_acc:.2%}")
		self.log(f"  Final PPL: {2.71828 ** final_ce:.1f}")

		results["final"] = {
			"fitness": final_ce,
			"accuracy": final_acc,
			"genome": p3b.best_genome.serialize(),  # Full genome for loading
			"genome_stats": p3b.best_genome.stats(),  # Stats for quick reference
			"final_population": [g.serialize() for g in p3b.final_population] if p3b.final_population else None,
			"final_threshold": p3b.final_threshold,  # Progressive threshold for next pass
		}

		# =====================================================================
		# Final Summary - Print cumulative phase comparison table
		# =====================================================================
		self.log("")
		self.log("=" * 90)
		self.log("  FINAL RESULTS (All metrics on FULL validation data)")
		self.log("=" * 90)

		# Print the cumulative phase comparison table (already evaluated on full validation)
		self.print_phase_comparison(phase_metrics_list)

		# Log the best genomes
		self.log("")
		self.log(f"Final best genome (by CE): {p3b.best_genome}")

		# Find best accuracy across all phases from phase_metrics_list
		if len(phase_metrics_list) > 1:  # Skip baseline-only case
			# Find phase with best accuracy (excluding baseline)
			best_acc_phase = max(phase_metrics_list[1:], key=lambda pm: pm.best_acc_acc)
			best_acc_ce = best_acc_phase.best_acc_ce
			best_acc_acc = best_acc_phase.best_acc_acc

			# Check if best-by-acc is different from final phase
			final_metrics = phase_metrics_list[-1]
			if best_acc_acc != final_metrics.best_acc_acc:
				self.log(f"Best accuracy found in: {best_acc_phase.phase_name}")
				self.log(f"  CE={best_acc_ce:.4f}, Acc={best_acc_acc:.2%}")

				# Store in results
				results["best_by_accuracy"] = {
					"phase": best_acc_phase.phase_name,
					"fitness": best_acc_ce,
					"accuracy": best_acc_acc,
				}

		# Store for later access
		self.results = results
		self._final_genome = p3b.best_genome
		self._completed_phases = completed_phases

		# Mark flow as completed in dashboard
		if self.dashboard_client and self._flow_id:
			try:
				self.dashboard_client.flow_completed(self._flow_id)
				self.log(f"Dashboard: Flow {self._flow_id} marked completed")
			except Exception as e:
				self.log(f"Warning: Failed to mark flow completed: {e}")

		return results

	@property
	def final_genome(self) -> Optional[ClusterGenome]:
		"""Get the final optimized genome."""
		return getattr(self, '_final_genome', None)
