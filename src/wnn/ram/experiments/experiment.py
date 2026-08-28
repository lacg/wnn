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

from wnn.ram.metrics import IDSMetrics, Metrics, GenomeType, FitnessWeights
from wnn.ram.fitness import FitnessCalculatorType, FitnessCalculatorFactory
from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.experiments.phased_search import PhaseResult



def _compute_per_class_breakdown(predictions, y_test_multi, class_names):
	"""Compute per-attack-class detection rate from binary predictions + multi-class labels.

	For each class index in y_test_multi, counts how many of its rows were
	predicted as attack (binary 1). For Benign rows this rate IS the FPR;
	for attack subclasses it IS the recall (detection rate).
	"""
	import numpy as np
	preds = np.asarray(predictions)
	multi = np.asarray(y_test_multi)
	out = {}
	for cls_idx, cls_name in enumerate(class_names):
		mask = (multi == cls_idx)
		n = int(mask.sum())
		if n == 0:
			continue
		n_pred_attack = int(((preds == 1) & mask).sum())
		out[cls_name] = {
			"count": n,
			"predicted_attack": n_pred_attack,
			"rate": float(n_pred_attack / n),
		}
	return out


class ExperimentType(IntEnum):
	"""How we optimize. GA/TS/LAMARCKIAN are the three strategies; the *dimension*
	they act on is a parameter (optimize_neurons/bits/connections for GA/TS,
	genesis_mode for LAMARCKIAN) — so each strategy is ONE type, not many."""
	GA = 0              # Genetic Algorithm
	TS = 1              # Tabu Search
	GRID_SEARCH = 2     # Grid search over neuron × bit configurations
	# DEPRECATED (3-5): pre-unification stats-guided adaptation, one type per
	# mode. Kept as back-compat aliases (historical DB rows / flows still load);
	# all new flows use LAMARCKIAN + genesis_mode. The dispatch routes these to
	# the unified ARCHITECTURE_LAMARCKIAN strategy with the matching mode.
	NEUROGENESIS = 3
	SYNAPTOGENESIS = 4
	AXONOGENESIS = 5
	LAMBDA_SWEEP = 6    # Unigram interpolation lambda sweep (eval-only)
	LAMARCKIAN = 7      # Unified stats-guided adaptation; genesis_mode picks
	                    # neurogenesis | synaptogenesis | axonogenesis


# Deprecated genesis ExperimentTypes → their genesis_mode (unification back-compat).
_GENESIS_ALIAS_MODE = {
	ExperimentType.NEUROGENESIS: "neurogenesis",
	ExperimentType.SYNAPTOGENESIS: "synaptogenesis",
	ExperimentType.AXONOGENESIS: "axonogenesis",
}


class GridSource(IntEnum):
	"""Where grid search gets its initial genomes."""
	RANDOM = 0       # Generate n×b grid configs with random connections
	LEADERBOARD = 1  # Load top genomes from best_genomes leaderboard


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

	# Architecture family (peer to ids/bitwise/multi_stage). "controller" routes
	# run() through the recurrent-controller strategy path; default keeps the IDS
	# behaviour unchanged. phase_type carries the raw dashboard phase string
	# (e.g. "ga_neurons", "lamarckian_memory") so the controller path can resolve
	# the exact (strategy kind, optimization dimension).
	architecture_type: str = "tiered"
	phase_type: str = ""

	# What to optimize (GA/TS dimension params)
	optimize_bits: bool = False
	optimize_neurons: bool = False
	optimize_connections: bool = False

	# LAMARCKIAN dimension param — which genesis operator to apply:
	# "neurogenesis" | "synaptogenesis" | "axonogenesis". Mirrors optimize_*.
	genesis_mode: str = "neurogenesis"

	# GA-specific
	generations: int = 250
	population_size: int = 50

	# TS-specific
	iterations: int = 250
	neighbors_per_iter: int = 50

	# Shared
	patience: int = 3
	check_interval: int = 10
	# Magnitude-aware patience (shared core, 11/07/2026): recover patience
	# proportional to physical-metric gains (F1↑/FPR↓ for IDS; err°/stable%
	# for controllers) instead of 1-per-improving-check on the rank-WHM.
	# LM flows fall back to the WHM check (no F1/FPR) regardless of this flag.
	magnitude_aware_patience: bool = True
	# Random-search baseline (Review C): GA loop with zero selection pressure —
	# offspring are fresh random genomes; eval protocol/budget identical to GA.
	random_search: bool = False
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
	# Rank-combine step: "harmonic" | "arithmetic" | "zscore" | "desirability".
	fitness_aggregation: str = "harmonic"
	fitness_zrank_clamp: float = 3.0
	# ABSOLUTE power-shape anchors for the two BOUNDED columns. None = the frozen
	# binary 0.80. Unlike ce these are not a unit bug, so they are settable to
	# make the multiclass-anchor question an A/B rather than an assertion.
	fitness_f1_anchor: Optional[float] = None
	fitness_acc_anchor: Optional[float] = None
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
	grid_source: GridSource = GridSource.RANDOM

	# Multi-stage configuration (only used when cluster_type=MULTI_STAGE)
	num_stages: int = 2
	stage_k: Optional[list[int]] = None          # K per stage; product >= vocab_size (default: [256, 256])
	stage_cluster_type: Optional[list[str]] = None  # per-stage arch (default: ["bitwise", "bitwise"])
	stage_mode: Optional[list[int]] = None       # StageMode values between stages (len = num_stages-1)
	target_stage: int = 0                        # 0-indexed: which stage this experiment optimizes
	frozen_genomes: Optional[list[Optional[dict]]] = None  # serialized genome per completed stage

	# Support-tiered sizing (MCST arm — docs/MCST_TIERED_ARM_SPEC.md).
	ids_tier_sizing: bool = False                # per-class centres from train supports
	ids_tier_neuron_cap: int = 250               # JOINT neuron cap (hier: S1 gets cap - S0 winner)
	tier_prev_stage_neurons: Optional[int] = None  # set by Flow at the stage boundary (S1 budget rule)
	tier_next_stage_classes: Optional[int] = None  # set by Flow for S0 (feasibility guard: reserve floors)

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
		dataset_key: Optional[str] = None,  # e.g. "ciciot2023_8b_random" — scopes cache lookups
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
		self.dataset_key = dataset_key

		# Derived properties
		self.vocab_size = evaluator.vocab_size
		self.total_input_bits = evaluator.total_input_bits

	def _resolve_ce_anchor(self, cfg) -> Optional[float]:
		"""ABSOLUTE desirability CE half-anchor for this task, or None.

		None only when the flow is not running the desirability combine at all.
		Otherwise the anchor is ALWAYS derived — there is no parameter, no
		default to inherit and nothing for an operator to select. CE is the one
		column with no absolute meaning, so its scale must come from the task
		itself; leaving that to a human choice is how a cohort ends up scored on
		another task's ruler, which happened once and must not be possible again.

		The derivation lives in ram_accelerator over the train labels already
		resident in the Rust cache. An evaluator that cannot produce one RAISES:
		falling back to a binary anchor on a 10-class task is the precise
		failure this exists to prevent, and it would be invisible in the logs.
		"""
		if cfg.fitness_aggregation != "desirability":
			return None
		if not hasattr(self.evaluator, "desirability_ce_anchor"):
			raise RuntimeError(
				f"aggregation='desirability' needs a per-task CE anchor, but evaluator "
				f"{type(self.evaluator).__name__} cannot derive one. Use an IDS evaluator "
				"with in-memory storage.")
		anchor = self.evaluator.desirability_ce_anchor()
		self.log(
			f"  [desirability] ce half-anchor {anchor:.6f} — derived from this task's own "
			f"train-label base-rate entropy (automatic; no per-run setting)")
		return anchor

	def _tier_centres(self, cfg: 'ExperimentConfig', num_clusters: int):
		"""(neuron_centres, bits_centres, grid_total_cap) from this stage's own
		train-label supports, or None with a LOUD line when the evaluator does
		not expose labels (falls back to the uniform grid). MCST arm —
		docs/MCST_TIERED_ARM_SPEC.md §1."""
		from wnn.ram.experiments.tier_sizing import (
			allocate_neurons, bits_centres, NEURON_FLOOR,
		)
		y = getattr(self.evaluator, '_y_train', None)
		if not y:
			self.log("  [tier] ⚠️ ids_tier_sizing requested but the evaluator exposes no "
			         "_y_train — FALLING BACK to the uniform grid")
			return None
		counts = [0] * num_clusters
		for v in y:
			iv = int(v)
			if 0 <= iv < num_clusters:
				counts[iv] += 1
		cap = cfg.ids_tier_neuron_cap
		if cfg.target_stage > 0 and cfg.tier_prev_stage_neurons:
			cap = max(NEURON_FLOOR * num_clusters, cap - cfg.tier_prev_stage_neurons)
			self.log(f"  [tier] S{cfg.target_stage} budget = {cfg.ids_tier_neuron_cap} − "
			         f"{cfg.tier_prev_stage_neurons} (frozen S{cfg.target_stage - 1} winner) → {cap}")
		ncent = allocate_neurons(counts, cap, NEURON_FLOOR)
		bcent = bits_centres(counts, len(y))
		total_cap = cap
		if cfg.tier_next_stage_classes:
			total_cap = cap - NEURON_FLOOR * cfg.tier_next_stage_classes
			self.log(f"  [tier] S0 grid capped at {total_cap} total neurons "
			         f"(reserving {NEURON_FLOOR}×{cfg.tier_next_stage_classes} for the next stage's floors)")
		zero = [i for i, c in enumerate(counts) if c == 0]
		if zero:
			self.log(f"  [tier] ⚠️ classes with ZERO train rows: {zero} — floored at "
			         f"{NEURON_FLOOR}n/{10}b, they cannot learn")
		self.log(f"  [tier] supports={counts}")
		self.log(f"  [tier] neuron centres={ncent} (Σ{sum(ncent)}, cap {cap}) · bits centres={bcent}")
		return ncent, bcent, total_cap

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
		is_controller = cfg.architecture_type == "controller"
		is_grid_search = cfg.experiment_type == ExperimentType.GRID_SEARCH
		is_ga = cfg.experiment_type == ExperimentType.GA
		# LAMARCKIAN (unified) or a deprecated per-mode alias → one strategy.
		is_adaptation = (cfg.experiment_type == ExperimentType.LAMARCKIAN
		                 or cfg.experiment_type in _GENESIS_ALIAS_MODE)
		# Resolve the genesis mode: explicit param for LAMARCKIAN; derived for aliases.
		genesis_mode = (cfg.genesis_mode if cfg.experiment_type == ExperimentType.LAMARCKIAN
		                else _GENESIS_ALIAS_MODE.get(cfg.experiment_type, "neurogenesis"))
		if is_grid_search:
			strategy_type = OptimizerStrategyType.ARCHITECTURE_GRID_SEARCH
		elif is_adaptation:
			strategy_type = OptimizerStrategyType.ARCHITECTURE_LAMARCKIAN
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

		# Typed configs (D6d: the kwargs dict is gone — every knob is explicit)
		from wnn.ram.strategies.connectivity.architecture_strategies import (
			ArchitectureConfig,
			GridSearchConfig,
			AdaptationConfig,
			CheckpointConfig,
		)
		from wnn.ram.strategies.connectivity.framework import GAConfig, TSConfig

		resolved_initial_threshold = (
			initial_threshold if initial_threshold is not None
			else (cfg.threshold_start if cfg.threshold_start > 0 else None)
		)
		min_bits = cfg.bitwise_min_bits if cfg.bitwise_min_bits is not None else cfg.min_bits
		max_bits = cfg.bitwise_max_bits if cfg.bitwise_max_bits is not None else cfg.max_bits
		min_neurons = cfg.bitwise_min_neurons if cfg.bitwise_min_neurons is not None else cfg.min_neurons
		max_neurons = cfg.bitwise_max_neurons if cfg.bitwise_max_neurons is not None else cfg.max_neurons
		min_accuracy_floor = cfg.min_accuracy_floor if cfg.min_accuracy_floor > 0 else 0.0

		# Support-tiered sizing (MCST): per-class centres from this stage's own
		# train-label supports. Computed once here; feeds the tiered grid AND
		# the per-cluster GA mutation bounds.
		tier = self._tier_centres(cfg, num_clusters) if cfg.ids_tier_sizing else None

		# Per-tier optimization: determine which clusters are optimizable
		mutable_clusters = None
		optimizable = self._get_optimizable_clusters(cfg.tier_config)
		if optimizable is not None and len(optimizable) < self.vocab_size:
			mutable_clusters = optimizable
			self.log(f"  Per-tier optimization: mutating {len(optimizable)} of {self.vocab_size} clusters")
		elif cfg.optimize_tier0_only and cfg.tier_config:
			tier0_clusters = cfg.tier_config[0][0] or self.vocab_size
			mutable_clusters = list(range(tier0_clusters))
			self.log(f"  Tier0-only mode: mutating first {tier0_clusters} clusters")

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			optimize_bits=cfg.optimize_bits,
			optimize_neurons=cfg.optimize_neurons,
			optimize_connections=cfg.optimize_connections,
			default_bits=cfg.default_bits,
			default_neurons=cfg.default_neurons,
			total_input_bits=self.total_input_bits,
			mutable_clusters=mutable_clusters,
			cluster_crossover_ratio=cfg.cluster_crossover_ratio,
			pool_shuffle_ratio=cfg.pool_shuffle_ratio,
			assortative_mating_ratio=cfg.assortative_mating_ratio,
			neuron_bounds_per_cluster=(
				[(max(1, int(c * 0.5 + 0.5)), max(1, int(c * 1.5 + 0.5))) for c in tier[0]]
				if tier else None),
			bits_bounds_per_cluster=(
				[(max(10, min(34, int(b * 0.5 + 0.5))), max(10, min(34, int(b * 1.5 + 0.5))))
				 for b in tier[1]]
				if tier else None),
		)

		resolved_ce_anchor = self._resolve_ce_anchor(cfg)

		phase_name = None
		checkpoint_config = None
		if is_grid_search:
			opt_config = GridSearchConfig(
				num_clusters=num_clusters,
				neurons_grid=cfg.neurons_grid or [5, 55, 105, 155, 205, 255, 300],
				bits_grid=cfg.bits_grid or [4, 8, 11, 15, 18, 21, 24],
				top_k=cfg.grid_top_k,
				population_size=cfg.population_size,
				total_input_bits=self.total_input_bits,
				fitness_calculator_type=cfg.fitness_calculator_type,
				fitness_weight_ce=cfg.fitness_weight_ce,
				fitness_weight_acc=cfg.fitness_weight_acc,
				fitness_weight_f1=cfg.fitness_weight_f1,
				fitness_weight_fpr=cfg.fitness_weight_fpr,
				fitness_aggregation=cfg.fitness_aggregation,
				zrank_clamp=cfg.fitness_zrank_clamp,
				ce_anchor=resolved_ce_anchor,
				f1_anchor=cfg.fitness_f1_anchor,
				acc_anchor=cfg.fitness_acc_anchor,
				grid_source=cfg.grid_source.name.lower(),
				tier_neuron_centres=tier[0] if tier else None,
				tier_bits_centres=tier[1] if tier else None,
				tier_total_cap=tier[2] if tier else None,
			)
		elif is_adaptation:
			phase_name = cfg.name
			opt_config = AdaptationConfig(
				num_clusters=num_clusters,
				min_bits=min_bits,
				max_bits=max_bits,
				min_neurons=min_neurons,
				max_neurons=max_neurons,
				total_input_bits=self.total_input_bits,
				adaptation_mode=genesis_mode,
				iterations=cfg.iterations,
				population_size=cfg.population_size,
				patience=cfg.patience,
				check_interval=cfg.check_interval,
				min_improvement_pct=0.05,
				initial_threshold=resolved_initial_threshold,
				threshold_delta=cfg.threshold_delta,
				threshold_reference=cfg.threshold_reference,
				fitness_calculator_type=cfg.fitness_calculator_type,
				fitness_weight_ce=cfg.fitness_weight_ce,
				fitness_weight_acc=cfg.fitness_weight_acc,
				fitness_weight_f1=cfg.fitness_weight_f1,
				fitness_weight_fpr=cfg.fitness_weight_fpr,
				fitness_aggregation=cfg.fitness_aggregation,
				zrank_clamp=cfg.fitness_zrank_clamp,
				ce_anchor=resolved_ce_anchor,
				f1_anchor=cfg.fitness_f1_anchor,
				acc_anchor=cfg.fitness_acc_anchor,
				min_accuracy_floor=min_accuracy_floor,
			)
		elif is_ga:
			phase_name = cfg.name
			# Per-generation population checkpoint → crash-resumable GA. Without this
			# the strategy's CheckpointManager is never built and a killed worker
			# loses all in-RAM generations (the gen-75 loss that motivated this).
			if self.checkpoint_dir:
				checkpoint_config = CheckpointConfig(
					enabled=True,
					target_loss_seconds=100.0,
					max_interval=10,
					checkpoint_dir=Path(self.checkpoint_dir),
					filename_prefix="ga_checkpoint",
				)
			opt_config = GAConfig(
				generations=cfg.generations,
				population_size=cfg.population_size,
				patience=cfg.patience,
				magnitude_aware_patience=cfg.magnitude_aware_patience,
				random_search=cfg.random_search,
				check_interval=cfg.check_interval,
				min_improvement_pct=0.05,
				initial_threshold=resolved_initial_threshold,
				threshold_delta=cfg.threshold_delta,
				threshold_reference=cfg.threshold_reference,
				fitness_percentile=cfg.fitness_percentile,
				seed_only=cfg.seed_only,
				fresh_population=cfg.fresh_population,
				fitness_calculator_type=cfg.fitness_calculator_type,
				fitness_weight_ce=cfg.fitness_weight_ce,
				fitness_weight_acc=cfg.fitness_weight_acc,
				fitness_weight_f1=cfg.fitness_weight_f1,
				fitness_weight_fpr=cfg.fitness_weight_fpr,
				fitness_aggregation=cfg.fitness_aggregation,
				zrank_clamp=cfg.fitness_zrank_clamp,
				ce_anchor=resolved_ce_anchor,
				f1_anchor=cfg.fitness_f1_anchor,
				acc_anchor=cfg.fitness_acc_anchor,
				min_accuracy_floor=min_accuracy_floor,
			)
		else:
			opt_config = TSConfig(
				iterations=cfg.iterations,
				neighbors_per_iter=cfg.neighbors_per_iter,
				total_neighbors_size=cfg.population_size,
				patience=cfg.patience,
				check_interval=cfg.check_interval,
				min_improvement_pct=0.5,
				initial_threshold=resolved_initial_threshold,
				threshold_delta=cfg.threshold_delta,
				threshold_reference=cfg.threshold_reference,
				fitness_percentile=cfg.fitness_percentile,
				fitness_calculator_type=cfg.fitness_calculator_type,
				fitness_weight_ce=cfg.fitness_weight_ce,
				fitness_weight_acc=cfg.fitness_weight_acc,
				fitness_weight_f1=cfg.fitness_weight_f1,
				fitness_weight_fpr=cfg.fitness_weight_fpr,
				fitness_aggregation=cfg.fitness_aggregation,
				zrank_clamp=cfg.fitness_zrank_clamp,
				ce_anchor=resolved_ce_anchor,
				f1_anchor=cfg.fitness_f1_anchor,
				acc_anchor=cfg.fitness_acc_anchor,
				min_accuracy_floor=min_accuracy_floor,
			)

		# Create strategy. Controller flows build a recurrent-controller strategy
		# via the WnnType factory; everything else uses OptimizerStrategyFactory.
		controller_batch_fn = None
		controller_init_pop = None
		if is_controller:
			from wnn.control.flow_adapter import resolve_phase, batch_eval_fn_for
			from wnn.ram.strategies.wnn_factory import create_strategy, WnnType
			_kind, _dim = resolve_phase(cfg.phase_type)
			strategy = create_strategy(
				WnnType.CONTROLLER, _kind, _dim,
				spec=self.evaluator.spec, batch_evaluator=self.evaluator,
				seed=getattr(cfg, "seed", None) or getattr(self.config, "seed", None))
			controller_batch_fn = batch_eval_fn_for(cfg.phase_type, self.evaluator)
			_mk = (getattr(strategy, "create_random_genome", None)
			       or getattr(strategy, "random_genome", None)
			       or getattr(strategy, "seed_genome", None))
			controller_init_pop = [_mk() for _ in range(max(1, cfg.population_size))]
			self.log(f"  Controller: {strategy.name}, pop {len(controller_init_pop)}, phase '{cfg.phase_type}'")
		else:
			strategy = OptimizerStrategyFactory.create(
				strategy_type,
				opt_config,
				arch_config=arch_config,
				seed=cfg.seed,
				logger=self.log,
				batch_evaluator=self.evaluator,
				shutdown_check=self.shutdown_check,
				checkpoint_config=checkpoint_config,
				phase_name=phase_name,
			)

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
			weights=FitnessWeights(
				ce=cfg.fitness_weight_ce, acc=cfg.fitness_weight_acc,
				f1=cfg.fitness_weight_f1, fpr=cfg.fitness_weight_fpr,
			),
			aggregation=cfg.fitness_aggregation,
			zrank_clamp=cfg.fitness_zrank_clamp,
			ce_anchor=resolved_ce_anchor,
			f1_anchor=cfg.fitness_f1_anchor,
			acc_anchor=cfg.fitness_acc_anchor,
		)

		# Run INIT validation on seed population
		# Use cached evals from previous phase if available (avoids redundant re-evaluation)
		seed_pop = initial_population or ([initial_genome] if initial_genome else None)
		if seed_pop and not is_controller:
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
			if is_controller:
				# Controller strategies have no Rust cached evaluator → drive optimize()
				# with the evaluator's batch fn (train+score for GA/TS/Lamarckian, or
				# score-only for MEMORY/paradigm B) + the factory-built initial pop.
				opt_kwargs["batch_evaluate_fn"] = controller_batch_fn
				opt_kwargs["evaluate_fn"] = lambda g: controller_batch_fn([g])[0].ce
				opt_kwargs["initial_population"] = controller_init_pop
				opt_kwargs["initial_genome"] = controller_init_pop[0] if controller_init_pop else None
				opt_kwargs["initial_neighbors"] = controller_init_pop
			if train_subset_idx is not None:
				opt_kwargs["train_subset_idx"] = train_subset_idx
			result = strategy.optimize(**opt_kwargs)

			# Check if shutdown was requested (needed for phase status update)
			from wnn.ram.strategies.connectivity.generic_strategies import StopReason
			was_shutdown = result.stop_reason == StopReason.SHUTDOWN if result.stop_reason else False

			# Run FINAL validation (skip if shutdown was requested; controller flows
			# have no IDS f1/fpr validation — fitness is closed-loop reward).
			val_results = {}
			if not was_shutdown and result.final_population and not is_controller:
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
			data["population_metrics"] = [
				m.to_dict() if hasattr(m, 'to_dict') else m
				for m in result.population_metrics
			]

		# Save best_ce and best_acc genomes for combined validation (future-proofing)
		if result.final_population and result.population_metrics:
			try:
				metrics = result.population_metrics
				# Best CE = minimum CE
				best_ce_idx = min(range(len(metrics)), key=lambda i: metrics[i].ce if hasattr(metrics[i], 'ce') else metrics[i][0])
				data["best_ce_genome"] = result.final_population[best_ce_idx].serialize()
				# Best ACC = maximum accuracy
				best_acc_idx = max(range(len(metrics)), key=lambda i: metrics[i].acc if hasattr(metrics[i], 'acc') else metrics[i][1])
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
			"""Extract (ce, acc, f1, fpr) from Metrics or legacy tuple."""
			if hasattr(e, 'ce'):
				# Metrics object
				return (e.ce, e.acc, e.f1, e.fpr)
			elif isinstance(e, dict):
				return (e['ce'], e['acc'], e.get('f1'), e.get('fpr'))
			else:
				# Legacy tuple
				return (e[0], e[1],
						e[2] if len(e) > 2 else None,
						e[3] if len(e) > 3 else None)

		extracted = [_extract(e) for e in evals]

		from wnn.ram.metrics import Metrics, GenomeType

		# Build Metrics for each genome
		metrics_list = [IDSMetrics(ce=ce, acc=acc, f1=f1, fpr=fpr) for ce, acc, f1, fpr in extracted]

		# Use bests() for consistent selection across all metrics
		if fitness_calculator is not None:
			try:
				pop_bests = fitness_calculator.bests(genomes, metrics_list)
				return [
					(pop_bests.best_ce.genome, GenomeType.BEST_CE, pop_bests.best_ce.metrics),
					(pop_bests.best_acc.genome, GenomeType.BEST_ACC, pop_bests.best_acc.metrics),
					(pop_bests.best_f1.genome, GenomeType.BEST_F1, pop_bests.best_f1.metrics),
					(pop_bests.best_fpr.genome, GenomeType.BEST_FPR, pop_bests.best_fpr.metrics),
					(pop_bests.best_fitness.genome, GenomeType.BEST_FITNESS, pop_bests.best_fitness.metrics),
				]
			except Exception as e:
				import traceback
				self.log(f"  Warning: fitness_calculator.bests() failed: {e}")
				self.log(f"  {traceback.format_exc().strip()}")

		# Fallback: select by CE and Acc
		best_ce_idx = min(range(len(metrics_list)), key=lambda i: metrics_list[i].ce)
		best_acc_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i].acc)

		return [
			(genomes[best_ce_idx], GenomeType.BEST_CE, metrics_list[best_ce_idx]),
			(genomes[best_acc_idx], GenomeType.BEST_ACC, metrics_list[best_acc_idx]),
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

			self.log(f"  Selected {len(selected)} genome(s): {[gt.value for _, gt, _ in selected]}")

			# Process each selected genome
			for genome, genome_type, train_metrics in selected:
				genome_hash = self._compute_genome_hash(genome)

				# Check if already validated (scoped by dataset to prevent cross-dataset cache poisoning)
				cached = None
				if self.dashboard_client:
					try:
						cached = self.dashboard_client.check_cached_validation(genome_hash, self.dataset_key)
					except Exception:
						pass

				val_evaluator = self.full_evaluator or self.evaluator

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
					# Option B: invalidate cache if per_class is missing — re-run for completeness
					_needs_per_class = (val_evaluator is not None
						and getattr(val_evaluator, "_y_test_multi", None) is not None)
					if _needs_per_class and cached_threshold_metadata is not None:
						try:
							_cached_tm = cached_threshold_metadata if isinstance(cached_threshold_metadata, dict) else json.loads(cached_threshold_metadata)
							if "per_class" not in _cached_tm:
								self.log(f"  {genome_type.value}: cached but missing per_class — re-validating")
								cached = None
								cached_threshold_metadata = None
						except Exception:
							pass
					# Multiclass: invalidate caches written before the K-class
					# metrics stage (plain CE/acc path — no decode modes).
					_is_mc_cached = (
						getattr(val_evaluator, '_classification', None) == 'multi'
						and not getattr(val_evaluator, '_single_cluster', False)
					)
					if _is_mc_cached and cached is not None:
						_mc_cached_ok = False
						if cached_threshold_metadata is not None:
							try:
								_cached_tm = cached_threshold_metadata if isinstance(cached_threshold_metadata, dict) else json.loads(cached_threshold_metadata)
								_mc_cached_ok = "argmax" in _cached_tm
							except Exception:
								pass
						if not _mc_cached_ok:
							self.log(f"  {genome_type.value}: cached but missing multiclass decode modes — re-validating")
							cached = None
							cached_threshold_metadata = None
					if f1 is not None:
						self.log(f"  {genome_type.value}: CE={ce:.4f}, Acc={acc:.4%}, F1={f1:.4%}, FPR={fpr_val:.4%} (cached)")
					else:
						self.log(f"  {genome_type.value}: CE={ce:.4f}, Acc={acc:.4%} (cached)")
				else:
					# Run full validation (use full_evaluator if available — validates against held-out set)
					val_evaluator = self.full_evaluator or self.evaluator
					self.log(f"  {genome_type.value}: Running full validation...")
					_is_sc = hasattr(val_evaluator, '_single_cluster') and val_evaluator._single_cluster
					_is_mc = (
						getattr(val_evaluator, '_classification', None) == 'multi'
						and not getattr(val_evaluator, '_single_cluster', False)
					)
					if _is_sc or _is_mc:
						# IDS single-cluster: defer the headline f1/fpr_val/acc to the
						# threshold-sweep block below so that train_cal metrics, the
						# per-class breakdown, and all six other thresholds all come
						# from a SINGLE training pass. Avoids the train_cal-vs-per-class
						# mismatch we saw on 8b runs (e.g. r112: threshold-table FPR
						# 3.68% vs per-class Benign 4.84% — same threshold, different
						# trainings, ~1pp drift from neuron-sample stochasticity).
						# Multiclass (K clusters): same reasoning — the headline argmax
						# metrics come from the decode-mode block's single pass.
						ce, acc, f1, fpr_val = None, None, None, None
					else:
						full_results = val_evaluator.evaluate_batch_full([genome])
						result = full_results[0]  # Metrics object
						ce, acc = result.ce, result.acc
						f1 = result.f1
						fpr_val = result.fpr
						if f1 is not None:
							self.log(f"  {genome_type.value}: CE={ce:.4f}, Acc={acc:.4%}, F1={f1:.4%}, FPR={fpr_val:.4%} (validated)")
						else:
							self.log(f"  {genome_type.value}: CE={ce:.4f}, Acc={acc:.4%} (validated)")

				# Three-threshold validation for single-cluster IDS
				threshold_metadata = None
				val_evaluator = self.full_evaluator or self.evaluator
				is_single_cluster = (
					hasattr(val_evaluator, '_single_cluster') and val_evaluator._single_cluster
				)
				is_multiclass = (
					getattr(val_evaluator, '_classification', None) == 'multi'
					and not getattr(val_evaluator, '_single_cluster', False)
				)

				# Use cached threshold_metadata if available (avoids re-running expensive 4-threshold eval)
				if cached_threshold_metadata is not None and (is_single_cluster or is_multiclass):
					threshold_metadata = cached_threshold_metadata
					self.log(f"    Thresholds: (cached from prior validation)")

				elif is_multiclass and cached is None:
					# Multiclass (K clusters): argmax + benign-margin decode modes
					# from a SINGLE training pass (mirrors the single-cluster
					# 7-mode block below). Metrics are ALWAYS computed on the
					# EVAL set; taus are calibrated Rust-side on train margins
					# (margin_train_cal) / val margins (margin_val_cal —
					# Protocol v2, only when a val partition exists). Each mode
					# entry carries macro_f1/benign_fpr/acc/ce (+ f1/fpr
					# aliases), the K×K confusion matrix, and the per-class
					# precision/recall/F1/support breakdown.
					try:
						import time as _time
						_t0 = _time.time()
						_mc = val_evaluator.evaluate_multiclass_at_thresholds(genome)
						threshold_metadata = _mc['modes']
						if 'margin_val_cal' in threshold_metadata:
							self.log(f"    [PROTOCOL-V2] margin_val_cal tau calibrated on val partition; "
									 f"test partition is report-only")
						# Headline metrics = argmax decode (the same rule the
						# GA-search fitness used).
						_am = threshold_metadata['argmax']
						ce, acc, f1, fpr_val = _am['ce'], _am['acc'], _am['macro_f1'], _am['benign_fpr']
						_tc_tau = threshold_metadata.get('margin_train_cal', {}).get('tau')
						if _tc_tau is not None:
							genome.threshold = _tc_tau
						self.log(f"  {genome_type.value}: CE={ce:.4f}, Acc={acc:.4%}, MacroF1={f1:.4%}, BenignFPR={fpr_val:.4%} (validated, argmax)")
						for _mode in ('argmax', 'margin_fixed0', 'margin_train_cal', 'margin_val_cal',
						              'argmax_platt', 'argmax_beta',
						              'argmax_classnorm', 'margin_classnorm'):
							_md = threshold_metadata.get(_mode)
							if not isinstance(_md, dict):
								continue
							_tau_str = f", tau={_md['tau']:.4f}" if 'tau' in _md else ""
							self.log(f"    {_mode + ':':<17}MacroF1={_md['macro_f1']:.4%}, BenignFPR={_md['benign_fpr']:.4%}, "
									 f"Acc={_md['acc']:.4%}, wF1={_md['weighted_f1']:.4%}{_tau_str}")
						self.log(f"    Scoring:     {len(threshold_metadata)} decode modes from one training pass "
								 f"({_time.time() - _t0:.1f}s)")
					except Exception as e:
						self.log(f"    Multiclass decode sweep failed ({e}) — falling back to evaluate_batch_full")
						threshold_metadata = None
						# Fallback: argmax headline metrics from evaluate_batch_full so
						# downstream code (results dict, summary writer) has valid numbers.
						_fb_results = val_evaluator.evaluate_batch_full([genome])
						_fb = _fb_results[0]
						ce, acc = _fb.ce, _fb.acc
						f1 = _fb.f1
						fpr_val = _fb.fpr

				elif is_single_cluster and cached is None:
					# All 7 threshold modes from a SINGLE training pass. Old path
					# trained 9× per genome (7 evaluate_batch_full + score_examples
					# + score_train_examples). The Rust-side evaluate_at_thresholds
					# trains once, returns eval+train scores, and computes metrics
					# at the thresholds we hand it. Calibrations (Platt/Beta/
					# Empirical/Emp-cumul/train_cal) are derived in Python from
					# train_scores, then per-mode metrics come from the Rust helper
					# compute_binary_metrics_at_threshold_py. Per-class breakdown
					# reuses the same eval_scores (no extra forward pass).
					threshold_metadata = {}
					try:
						import ram_accelerator
						import time as _time
						_t0 = _time.time()
						# Single training pass: returns eval/train scores + metrics
						# at the requested thresholds (-1.0 oracle, 0.5 fixed).
						eval_scores, train_scores, val_scores, anchor_metrics = val_evaluator.evaluate_at_thresholds(
							genome, [-1.0, 0.5],
						)
						oracle_metrics, fixed_metrics = anchor_metrics
						train_labels_list = val_evaluator._y_train
						eval_labels_list = val_evaluator._y_test
						val_labels_list = getattr(val_evaluator, "_y_val", None)
						normal_class = getattr(val_evaluator, "_normal_class", 0)
						# Protocol v2 (3-way splits): calibrate thresholds on the VAL
						# partition, report on TEST. Active only when the cache scored
						# a val partition AND the evaluator carries matching val labels.
						_protocol_v2 = (
							val_scores is not None
							and val_labels_list is not None
							and len(val_scores) == len(val_labels_list)
						)
						if _protocol_v2:
							self.log(f"    [PROTOCOL-V2] threshold calibrations on val partition "
									 f"(n={len(val_scores)}); test partition is report-only")
						_score_secs = _time.time() - _t0
						self.log(f"    Scoring:     train+eval scored in {_score_secs:.1f}s "
								 f"(was {_score_secs * 10:.0f}s with 10× train passes incl. headline call)")
					except Exception as e:
						self.log(f"    Threshold sweep failed ({e}) — falling back to evaluate_batch_full")
						threshold_metadata = None
						eval_scores = None
						train_scores = None
						val_scores = None
						_protocol_v2 = False
						# Fallback: get headline metrics from evaluate_batch_full so downstream
						# code (results dict, dashboard summary writer) still has valid numbers.
						_fb_results = val_evaluator.evaluate_batch_full([genome])
						_fb = _fb_results[0]
						ce, acc = _fb.ce, _fb.acc
						f1 = _fb.f1
						fpr_val = _fb.fpr

					if threshold_metadata is not None and eval_scores is not None and train_scores is not None:
						def _metrics_at(t):
							ce_t, acc_t, f1_t, fpr_t = ram_accelerator.compute_binary_metrics_at_threshold_py(
								eval_scores, eval_labels_list, float(t), normal_class,
							)
							return ce_t, acc_t, f1_t, fpr_t

						# 1. Train-calibrated — primary metric. Sweep F1-optimal threshold on
						# TRAIN scores (from this same training pass), apply to eval scores.
						# This replaces the old line-1007 evaluate_batch_full call so train_cal
						# F1/FPR/Acc and the per-class Benign rate now come from identical
						# scores (same training, same threshold, same predictions).
						train_threshold, _train_f1_unused, _train_fpr_unused = (
							ram_accelerator.find_optimal_threshold_f1_py(
								train_scores, train_labels_list,
							)
						)
						genome.threshold = train_threshold  # keep this for downstream code that reads genome.threshold
						_tc_ce, _tc_acc, _tc_f1, _tc_fpr = _metrics_at(train_threshold)
						# Promote train_cal as the headline metrics for this genome (matches the
						# pre-fix semantics where line 1007 produced these numbers).
						ce, acc, f1, fpr_val = _tc_ce, _tc_acc, _tc_f1, _tc_fpr
						threshold_metadata['train_cal'] = {
							'f1': _tc_f1, 'fpr': _tc_fpr, 'acc': _tc_acc, 'threshold': train_threshold,
						}
						self.log(f"  {genome_type.value}: CE={_tc_ce:.4f}, Acc={_tc_acc:.4%}, F1={_tc_f1:.4%}, FPR={_tc_fpr:.4%} (validated)")
						self.log(f"    Train-cal:   F1={_tc_f1:.4%}, FPR={_tc_fpr:.4%}, Acc={_tc_acc:.4%}, t={train_threshold:.4f}")

						# 2. Fixed 0.5 — distribution-agnostic baseline
						threshold_metadata['fixed_05'] = {
							'f1': fixed_metrics.f1, 'fpr': fixed_metrics.fpr, 'acc': fixed_metrics.acc,
						}
						self.log(f"    Fixed 0.5:   F1={fixed_metrics.f1:.4%}, FPR={fixed_metrics.fpr:.4%}, Acc={fixed_metrics.acc:.4%}")

						# 3. Validation-calibrated. Protocol v2: F1-optimal threshold on the
						# VAL partition scores, applied to TEST scores (the oracle anchor from
						# the -1.0 sentinel is unused in v2 mode). Legacy 2-way: oracle
						# threshold on the report-set scores themselves (unchanged).
						if _protocol_v2:
							val_cal_threshold, _vc_f1_unused, _vc_fpr_unused = (
								ram_accelerator.find_optimal_threshold_f1_py(
									val_scores, val_labels_list,
								)
							)
							_vc_ce, _vc_acc, _vc_f1, _vc_fpr = _metrics_at(val_cal_threshold)
							threshold_metadata['val_cal'] = {
								'f1': _vc_f1, 'fpr': _vc_fpr, 'acc': _vc_acc,
								'threshold': val_cal_threshold,
							}
							self.log(f"    Val-cal:     F1={_vc_f1:.4%}, FPR={_vc_fpr:.4%}, Acc={_vc_acc:.4%}, t={val_cal_threshold:.4f} (val partition)")
						else:
							threshold_metadata['val_cal'] = {
								'f1': oracle_metrics.f1, 'fpr': oracle_metrics.fpr, 'acc': oracle_metrics.acc,
								'threshold': oracle_metrics.threshold,
							}
							self.log(f"    Val-cal:     F1={oracle_metrics.f1:.4%}, FPR={oracle_metrics.fpr:.4%}, Acc={oracle_metrics.acc:.4%}, t={oracle_metrics.threshold:.4f} (oracle)")

						# 4-7. Calibrations fit on VAL scores under Protocol v2 (3-way splits),
						# on TRAINING scores for legacy 2-way flows → applied to the report
						# set via the cheap metric helper.
						if _protocol_v2:
							cal_scores = val_scores
							cal_labels_list = val_labels_list
						else:
							cal_scores = train_scores
							cal_labels_list = train_labels_list
						try:
							if cal_scores and cal_labels_list and len(cal_scores) == len(cal_labels_list):
								# 4. Platt scaling
								platt_threshold, a, b = ram_accelerator.fit_platt_scaling_py(cal_scores, cal_labels_list)
								_, p_acc, p_f1, p_fpr = _metrics_at(platt_threshold)
								threshold_metadata['platt'] = {
									'f1': p_f1, 'fpr': p_fpr, 'acc': p_acc,
									'threshold': platt_threshold, 'a': a, 'b': b,
								}
								self.log(f"    Platt:       F1={p_f1:.4%}, FPR={p_fpr:.4%}, Acc={p_acc:.4%}, t={platt_threshold:.4f} (a={a:.4f}, b={b:.4f})")

								# 5. Beta calibration
								beta_threshold, ba, bb, bc = ram_accelerator.fit_beta_calibration_py(cal_scores, cal_labels_list)
								_, b_acc, b_f1, b_fpr = _metrics_at(beta_threshold)
								threshold_metadata['beta'] = {
									'f1': b_f1, 'fpr': b_fpr, 'acc': b_acc,
									'threshold': beta_threshold, 'a': ba, 'b': bb, 'c': bc,
								}
								self.log(f"    Beta:        F1={b_f1:.4%}, FPR={b_fpr:.4%}, Acc={b_acc:.4%}, t={beta_threshold:.4f} (a={ba:.3f}, b={bb:.3f}, c={bc:.3f})")

								# 6. Empirical table
								empirical_threshold, n_bins = ram_accelerator.fit_empirical_threshold_py(cal_scores, cal_labels_list)
								_, e_acc, e_f1, e_fpr = _metrics_at(empirical_threshold)
								threshold_metadata['empirical'] = {
									'f1': e_f1, 'fpr': e_fpr, 'acc': e_acc,
									'threshold': empirical_threshold, 'n_bins': n_bins,
								}
								self.log(f"    Empirical:   F1={e_f1:.4%}, FPR={e_fpr:.4%}, Acc={e_acc:.4%}, t={empirical_threshold:.4f} ({n_bins} bins)")

								# 7. Empirical-cumulative: GA-fitness-optimal sweep on calibration scores.
								# Distinct from train_cal (pure F1) because it uses the flow's actual
								# fitness weights — so this column reports the threshold the optimizer
								# was implicitly targeting, while train_cal reports the F1-only ideal.
								w_ce = float(self.config.fitness_weight_ce)
								w_f1 = float(self.config.fitness_weight_f1)
								w_fpr = float(self.config.fitness_weight_fpr)
								w_acc = float(self.config.fitness_weight_acc)
								emp_cum_result = ram_accelerator.find_optimal_threshold_fitness_py(
									cal_scores, cal_labels_list, w_ce, w_f1, w_fpr, w_acc,
								)
								emp_cum_threshold = emp_cum_result[0]
								_, c_acc, c_f1, c_fpr = _metrics_at(emp_cum_threshold)
								threshold_metadata['empirical_cumulative'] = {
									'f1': c_f1, 'fpr': c_fpr, 'acc': c_acc,
									'threshold': emp_cum_threshold,
									'w_ce': w_ce, 'w_f1': w_f1, 'w_fpr': w_fpr, 'w_acc': w_acc,
								}
								self.log(f"    Emp-cumul:   F1={c_f1:.4%}, FPR={c_fpr:.4%}, Acc={c_acc:.4%}, t={emp_cum_threshold:.4f} (weights ce={w_ce:.2f} f1={w_f1:.2f} fpr={w_fpr:.2f} acc={w_acc:.2f})")
						except Exception as e:
							self.log(f"    Calibration: skipped ({e})")

						# Per-class breakdown at ALL threshold modes — same eval_scores
						# already in memory, threshold-and-bucket per mode.
						if (val_evaluator is not None
							and getattr(val_evaluator, "_y_test_multi", None) is not None
							and getattr(val_evaluator, "_class_names", None) is not None):
							try:
								import numpy as _np
								_pc_t0 = _time.time()
								scores_arr = _np.asarray(eval_scores, dtype=_np.float64)
								n_modes_done = 0
								for mode_key, mode_data in list(threshold_metadata.items()):
									if not isinstance(mode_data, dict):
										continue
									thr = mode_data.get("threshold")
									if thr is None:
										thr = 0.5 if mode_key == "fixed_05" else train_threshold
									preds = (scores_arr >= float(thr)).astype(int).tolist()
									pc = _compute_per_class_breakdown(
										preds,
										val_evaluator._y_test_multi,
										val_evaluator._class_names,
									)
									mode_data["per_class"] = pc
									n_modes_done += 1
								# Back-compat: top-level per_class mirrors train_cal's
								if "train_cal" in threshold_metadata and isinstance(threshold_metadata["train_cal"], dict):
									if "per_class" in threshold_metadata["train_cal"]:
										threshold_metadata["per_class"] = threshold_metadata["train_cal"]["per_class"]
								self.log(f"    Per-class:   computed at {n_modes_done} thresholds "
										 f"({_time.time()-_pc_t0:.1f}s)")
							except Exception as _e:
								self.log(f"    Per-class:   skipped ({_e})")

					# Use train-calibrated as primary metric (threshold from training, eval on val)
					# f1, fpr_val, acc already set from train_cal above

				# Collect results keyed by genome_type (use .value for dict key)
				results[genome_type.value] = {'ce': ce, 'acc': acc, 'f1': f1, 'fpr': fpr_val}

				# Always store summary via dashboard API (even if cached)
				# This ensures each (experiment_id, validation_point, genome_type) has a record
				if self.dashboard_client and self.experiment_id:
					try:
						self.dashboard_client.create_validation_summary(
							experiment_id=self.experiment_id,
							validation_point=validation_point,
							genome_type=genome_type.value,
							genome_hash=genome_hash,
							ce=ce,
							accuracy=acc,
							flow_id=flow_id,
							f1_macro=f1,
							fpr=fpr_val,
							threshold_metadata=json.dumps(threshold_metadata) if threshold_metadata else None,
						)
					except Exception as e:
						self.log(f"  Warning: Failed to save {genome_type.value} summary: {e}")

				# Submit to best genomes leaderboard (final validation only)
				if self.dashboard_client and validation_point == 'final':
					try:
						task_type = "ids" if f1 is not None else "lm"
						exp_name = self.config.name if self.config else ""
						if exp_name.startswith("S1:") or exp_name.startswith("S1 "):
							stage = "stage_1"
						elif exp_name.startswith("S2:") or exp_name.startswith("S2 "):
							stage = "stage_2"
						else:
							stage = "stage_0"
						metric_map = {
							GenomeType.BEST_CE: "ce",
							GenomeType.BEST_ACC: "accuracy",
							GenomeType.BEST_F1: "f1_macro",
							GenomeType.BEST_FPR: "fpr",
							GenomeType.BEST_FITNESS: "fitness",
						}
						metric = metric_map.get(genome_type, "ce")
						# Use proper JSON for tiers_json (replaces legacy str(genome) repr).
						# Allows downstream tools to reconstruct the genome without gzipped
						# checkpoints, and stores full per-neuron bits.
						if hasattr(genome, "to_json_dict"):
							_tiers_json = json.dumps(genome.to_json_dict())
						else:
							_tiers_json = str(genome)  # back-compat for non-ClusterGenome types
						base_genome_data = {
							"config_hash": genome_hash[:16],
							"tiers_json": _tiers_json,
							"total_clusters": len(genome.neurons_per_cluster),
							"total_neurons": sum(genome.neurons_per_cluster),
							"architecture_type": task_type,
						}
						if genome.connections is not None:
							base_genome_data["connections_json"] = ",".join(str(c) for c in genome.connections)

						# Submit one leaderboard entry per threshold mode
						submissions = []
						# If threshold_metadata is available, submit from it (has all modes including train_cal)
						# Otherwise fall back to the main validation values as train_cal
						if threshold_metadata:
							for mode_key, mode_data in threshold_metadata.items():
								if isinstance(mode_data, dict) and 'f1' in mode_data:
									submissions.append({
										"task_type": task_type, "stage": stage, "metric": metric,
										"genome_hash": genome_hash,
										"ce": ce,  # CE is threshold-independent
										"accuracy": mode_data.get('acc', acc),
										"f1_macro": mode_data.get('f1'),
										"fpr": mode_data.get('fpr'),
										"flow_id": flow_id, "experiment_id": self.experiment_id,
										"genome_data": {**base_genome_data, "threshold_mode": mode_key},
									})
						else:
							# No threshold modes — submit with default train_cal
							submissions.append({
								"task_type": task_type, "stage": stage, "metric": metric,
								"genome_hash": genome_hash,
								"ce": ce, "accuracy": acc, "f1_macro": f1, "fpr": fpr_val,
								"flow_id": flow_id, "experiment_id": self.experiment_id,
								"genome_data": {**base_genome_data, "threshold_mode": "train_cal"},
							})
						# Quality gate: only submit genomes meeting IDS goals
						# (at least one of: F1 >= 87%, Acc >= 87%, FPR <= 12%)
						if task_type == "ids":
							submissions = [
								s for s in submissions
								if (s.get("f1_macro") is not None and s["f1_macro"] >= 0.87)
								or (s.get("accuracy") is not None and s["accuracy"] >= 0.87)
								or (s.get("fpr") is not None and s["fpr"] <= 0.12)
							]
						if submissions:
							self.dashboard_client.submit_best_genomes(submissions)
					except Exception as e:
						self.log(f"  Warning: leaderboard submit failed: {e}")

			self.log("=" * 60)
			self.log("")

		except Exception as e:
			self.log(f"  Warning: {validation_point} validation failed: {e}")

		return results
