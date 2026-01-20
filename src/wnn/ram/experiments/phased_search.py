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
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.core.reporting import OptimizationResultsTable
from wnn.ram.core import bits_needed
from wnn.ram.architecture.cached_evaluator import CachedEvaluator


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

	# Architecture defaults
	default_bits: int = 8
	default_neurons: int = 5

	# CE percentile filter (None = disabled, 0.75 = keep top 75% by CE)
	ce_percentile: Optional[float] = None

	# Random seed (None = time-based)
	rotation_seed: Optional[int] = None

	# Logging
	log_path: Optional[str] = None

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
PHASE_NAMES = {
	"1a": "phase_1a_ga_neurons",
	"1b": "phase_1b_ts_neurons",
	"2a": "phase_2a_ga_bits",
	"2b": "phase_2b_ts_bits",
	"3a": "phase_3a_ga_connections",
	"3b": "phase_3b_ts_connections",
}


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
	):
		"""
		Initialize the runner.

		Args:
			config: Search configuration
			logger: Logging function
			checkpoint_dir: Directory to save/load checkpoints (None = no checkpoints)
		"""
		self.config = config
		self.log = logger
		self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
		self.evaluator: Optional[CachedEvaluator] = None
		self.vocab_size: int = 0
		self.total_input_bits: int = 0
		self.token_frequencies: list[int] = []
		self.results: dict[str, PhaseResult] = {}
		self._rotation_seed: Optional[int] = None

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
		# Use .json extension - save() will auto-add .gz
		filename = f"{PHASE_NAMES.get(phase_key, phase_key)}.json"
		filepath = self.checkpoint_dir / filename

		result.save(
			str(filepath),
			phase_key=phase_key,
			rotation_seed=self._rotation_seed,
		)
		# Log actual saved path (with .gz)
		actual_path = filepath.with_suffix('.json.gz')
		self.log(f"  Checkpoint saved: {actual_path}")
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

		base_filename = PHASE_NAMES.get(phase_key, phase_key)

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
		latest = None
		for phase_key in phase_order:
			base = PHASE_NAMES.get(phase_key, phase_key)
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
		)
		self.log(f"  {self.evaluator}")

		# Compute input bits
		bits_per_token = bits_needed(vocab_size)
		self.total_input_bits = self.config.context_size * bits_per_token
		self.log("")
		self.log(f"Input encoding: {self.config.context_size} tokens × {bits_per_token} bits = {self.total_input_bits} bits")

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
			"initial_threshold": initial_threshold,
			"ce_percentile": cfg.ce_percentile,  # CE percentile filter (None = disabled)
			"seed": self._rotation_seed,  # Use rotation_seed for strategy RNG
		}

		if is_ga:
			strategy_kwargs["generations"] = cfg.ga_generations
			strategy_kwargs["population_size"] = cfg.population_size
		else:
			strategy_kwargs["iterations"] = cfg.ts_iterations
			strategy_kwargs["neighbors_per_iter"] = cfg.neighbors_per_iter
			strategy_kwargs["total_neighbors_size"] = cfg.population_size

		strategy = OptimizerStrategyFactory.create(**strategy_kwargs)

		# Run optimization
		if initial_genome is not None:
			if is_ga:
				seed_pop = initial_population if initial_population else [initial_genome]
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

	def print_progress(
		self,
		title: str,
		phase_results: list[PhaseResult],
		baseline_ce: Optional[float] = None,
		baseline_acc: Optional[float] = None,
	) -> None:
		"""Print a progress table with current results."""
		table = OptimizationResultsTable(title)
		if baseline_ce is not None:
			table.add_stage("Initial (default genome)", ce=baseline_ce, accuracy=baseline_acc)
		for pr in phase_results:
			table.add_stage(pr.phase_name, ce=pr.final_fitness, accuracy=pr.final_accuracy)
		self.log("")
		table.print(self.log)

	def run_all_phases(
		self,
		seed_genome: Optional[ClusterGenome] = None,
		resume_from: Optional[str] = None,
	) -> dict[str, Any]:
		"""
		Run all phases: 1a -> 1b -> 2a -> 2b -> 3a -> 3b -> final evaluation.

		Args:
			seed_genome: Optional genome to seed Phase 1a from
			resume_from: Phase to resume from (e.g., "1b", "2a"). If provided,
			            loads checkpoint from previous phase and continues.

		Returns:
			Dictionary with all results
		"""
		results: dict[str, Any] = {}
		completed_phases: list[PhaseResult] = []

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
		# Baseline: Evaluate initial genome on full validation
		# =====================================================================
		if start_idx <= 0:
			# Create default genome if no seed provided
			if seed_genome:
				baseline_genome = seed_genome
				self.log("Evaluating seed genome as baseline...")
			else:
				from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
				baseline_genome = ClusterGenome.create_uniform(
					num_clusters=self.vocab_size,
					bits=self.config.default_bits,
					neurons=self.config.default_neurons,
				)
				self.log("Evaluating default genome as baseline...")

			baseline_ce, baseline_acc = self.evaluator.evaluate_single_full(baseline_genome)
			self.log(f"  Baseline CE: {baseline_ce:.4f}, Acc: {baseline_acc:.2%}")
			self.log("")
		else:
			# When resuming, we don't have the baseline - use None
			baseline_ce, baseline_acc = None, None

		# =====================================================================
		# Phase 1a: GA Neurons Only
		# =====================================================================
		if start_idx <= 0:
			p1a = self.run_phase(
				phase_name="Phase 1a: GA Neurons Only",
				strategy_type=OptimizerStrategyType.ARCHITECTURE_GA,
				optimize_bits=False,
				optimize_neurons=True,
				optimize_connections=False,
				initial_genome=seed_genome,
			)
			results["phase1_ga"] = {
				"fitness": p1a.final_fitness,
				"accuracy": p1a.final_accuracy,
				"iterations": p1a.iterations_run,
			}
			completed_phases.append(p1a)
			self.print_progress("After Phase 1a", completed_phases, baseline_ce, baseline_acc)
			self.save_checkpoint("1a", p1a)
		else:
			p1a = completed_phases[0]

		# =====================================================================
		# Phase 1b: TS Neurons Only
		# =====================================================================
		if start_idx <= 1:
			p1b = self.run_phase(
				phase_name="Phase 1b: TS Neurons Only (refine)",
				strategy_type=OptimizerStrategyType.ARCHITECTURE_TS,
				optimize_bits=False,
				optimize_neurons=True,
				optimize_connections=False,
				initial_genome=p1a.best_genome,
				initial_fitness=p1a.final_fitness,
				initial_population=p1a.final_population,
				initial_threshold=p1a.final_threshold,
			)
			results["phase1_ts"] = {
				"fitness": p1b.final_fitness,
				"accuracy": p1b.final_accuracy,
				"iterations": p1b.iterations_run,
			}
			completed_phases.append(p1b)
			self.print_progress("After Phase 1b", completed_phases, baseline_ce, baseline_acc)
			self.save_checkpoint("1b", p1b)
		else:
			p1b = completed_phases[1]

		# =====================================================================
		# Phase 2a: GA Bits Only
		# =====================================================================
		if start_idx <= 2:
			p2a = self.run_phase(
				phase_name="Phase 2a: GA Bits Only",
				strategy_type=OptimizerStrategyType.ARCHITECTURE_GA,
				optimize_bits=True,
				optimize_neurons=False,
				optimize_connections=False,
				initial_genome=p1b.best_genome,
				initial_population=p1b.final_population,
				initial_threshold=p1b.final_threshold,
			)
			results["phase2_ga"] = {
				"fitness": p2a.final_fitness,
				"accuracy": p2a.final_accuracy,
				"iterations": p2a.iterations_run,
			}
			completed_phases.append(p2a)
			self.print_progress("After Phase 2a", completed_phases, baseline_ce, baseline_acc)
			self.save_checkpoint("2a", p2a)
		else:
			p2a = completed_phases[2]

		# =====================================================================
		# Phase 2b: TS Bits Only
		# =====================================================================
		if start_idx <= 3:
			p2b = self.run_phase(
				phase_name="Phase 2b: TS Bits Only (refine)",
				strategy_type=OptimizerStrategyType.ARCHITECTURE_TS,
				optimize_bits=True,
				optimize_neurons=False,
				optimize_connections=False,
				initial_genome=p2a.best_genome,
				initial_fitness=p2a.final_fitness,
				initial_population=p2a.final_population,
				initial_threshold=p2a.final_threshold,
			)
			results["phase2_ts"] = {
				"fitness": p2b.final_fitness,
				"accuracy": p2b.final_accuracy,
				"iterations": p2b.iterations_run,
			}
			completed_phases.append(p2b)
			self.print_progress("After Phase 2b", completed_phases, baseline_ce, baseline_acc)
			self.save_checkpoint("2b", p2b)
		else:
			p2b = completed_phases[3]

		# =====================================================================
		# Phase 3a: GA Connections Only
		# =====================================================================
		if start_idx <= 4:
			p3a = self.run_phase(
				phase_name="Phase 3a: GA Connections Only",
				strategy_type=OptimizerStrategyType.ARCHITECTURE_GA,
				optimize_bits=False,
				optimize_neurons=False,
				optimize_connections=True,
				initial_genome=p2b.best_genome,
				initial_population=p2b.final_population,
				initial_threshold=p2b.final_threshold,
			)
			results["phase3_ga"] = {
				"fitness": p3a.final_fitness,
				"accuracy": p3a.final_accuracy,
				"iterations": p3a.iterations_run,
			}
			completed_phases.append(p3a)
			self.print_progress("After Phase 3a", completed_phases, baseline_ce, baseline_acc)
			self.save_checkpoint("3a", p3a)
		else:
			p3a = completed_phases[4]

		# =====================================================================
		# Phase 3b: TS Connections Only
		# =====================================================================
		if start_idx <= 5:
			p3b = self.run_phase(
				phase_name="Phase 3b: TS Connections Only (refine)",
				strategy_type=OptimizerStrategyType.ARCHITECTURE_TS,
				optimize_bits=False,
				optimize_neurons=False,
				optimize_connections=True,
				initial_genome=p3a.best_genome,
				initial_fitness=p3a.final_fitness,
				initial_population=p3a.final_population,
				initial_threshold=p3a.final_threshold,
			)
			results["phase3_ts"] = {
				"fitness": p3b.final_fitness,
				"accuracy": p3b.final_accuracy,
				"iterations": p3b.iterations_run,
			}
			completed_phases.append(p3b)
			self.save_checkpoint("3b", p3b)
		else:
			p3b = completed_phases[5]

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
		}

		# =====================================================================
		# Final Summary - Re-evaluate all phases on FULL validation data
		# =====================================================================
		self.log("")
		self.log("=" * 78)
		self.log("  FINAL RESULTS (Full Validation)")
		self.log("=" * 78)

		# Re-evaluate each phase's best genome on full validation for apples-to-apples comparison
		self.log("")
		self.log("Re-evaluating all phases on full validation data...")
		full_eval_results: list[tuple[float, float]] = []
		for pr in completed_phases:
			ce, acc = self.evaluator.evaluate_single_full(pr.best_genome)
			full_eval_results.append((ce, acc))
			self.log(f"  {pr.phase_name}: CE={ce:.4f}, Acc={acc:.2%}")

		comparison = OptimizationResultsTable("Complete Phased Search (Full Validation) - Best by CE")
		if baseline_ce is not None:
			comparison.add_stage("Initial (default genome)", ce=baseline_ce, accuracy=baseline_acc)
		for pr, (full_ce_val, full_acc_val) in zip(completed_phases, full_eval_results):
			comparison.add_stage(pr.phase_name, ce=full_ce_val, accuracy=full_acc_val)
		comparison.print(self.log)

		self.log("")
		self.log(f"Final best genome (by CE): {p3b.best_genome}")

		# Find best genome by accuracy across all phases
		best_acc_idx = max(range(len(full_eval_results)), key=lambda i: full_eval_results[i][1])
		best_acc_genome = completed_phases[best_acc_idx].best_genome
		best_acc_ce, best_acc_acc = full_eval_results[best_acc_idx]

		# Only print second table if best-by-acc is different from best-by-CE
		if best_acc_genome != p3b.best_genome or best_acc_acc != full_eval_results[-1][1]:
			self.log("")
			self.log("-" * 78)
			self.log("")
			comparison_acc = OptimizationResultsTable("Best by Accuracy (Full Validation)")
			if baseline_ce is not None:
				comparison_acc.add_stage("Initial (default genome)", ce=baseline_ce, accuracy=baseline_acc)
			comparison_acc.add_stage(
				f"Best Acc: {completed_phases[best_acc_idx].phase_name}",
				ce=best_acc_ce,
				accuracy=best_acc_acc
			)
			comparison_acc.print(self.log)
			self.log("")
			self.log(f"Best genome (by Accuracy): {best_acc_genome}")

			# Store both in results
			results["best_by_accuracy"] = {
				"phase": completed_phases[best_acc_idx].phase_name,
				"fitness": best_acc_ce,
				"accuracy": best_acc_acc,
				"genome": best_acc_genome.serialize(),
				"genome_stats": best_acc_genome.stats(),
			}

		# Store for later access
		self.results = results
		self._final_genome = p3b.best_genome
		self._completed_phases = completed_phases

		return results

	@property
	def final_genome(self) -> Optional[ClusterGenome]:
		"""Get the final optimized genome."""
		return getattr(self, '_final_genome', None)
