"""
Phased Architecture Search Runner

Encapsulates the three-phase optimization approach:
1. Phase 1: Optimize neurons (bits fixed)
2. Phase 2: Optimize bits (neurons from Phase 1)
3. Phase 3: Optimize connections (architecture from Phase 2)

Each phase uses GA then TS refinement.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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

	# Random seed (None = time-based)
	rotation_seed: Optional[int] = None

	# Logging
	log_path: Optional[str] = None


@dataclass
class PhaseResult:
	"""Result from a single optimization phase."""
	phase_name: str
	strategy_type: str  # "GA" or "TS"
	final_fitness: float
	final_accuracy: Optional[float]
	iterations_run: int
	best_genome: ClusterGenome
	final_population: Optional[List[ClusterGenome]]
	final_threshold: Optional[float]
	initial_fitness: Optional[float] = None
	initial_accuracy: Optional[float] = None


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
	):
		"""
		Initialize the runner.

		Args:
			config: Search configuration
			logger: Logging function
		"""
		self.config = config
		self.log = logger
		self.evaluator: Optional[CachedEvaluator] = None
		self.vocab_size: int = 0
		self.total_input_bits: int = 0
		self.token_frequencies: List[int] = []
		self.results: Dict[str, PhaseResult] = {}

	def setup(
		self,
		train_tokens: List[int],
		eval_tokens: List[int],
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

		# Determine rotation seed
		rotation_seed = self.config.rotation_seed
		if rotation_seed is None:
			rotation_seed = int(time.time() * 1000) % (2**32)

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
		initial_population: Optional[List[ClusterGenome]] = None,
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

	def print_progress(self, title: str, phase_results: List[PhaseResult]) -> None:
		"""Print a progress table with current results."""
		table = OptimizationResultsTable(title)
		for pr in phase_results:
			table.add_stage(pr.phase_name, ce=pr.final_fitness, accuracy=pr.final_accuracy)
		self.log("")
		table.print(self.log)

	def run_all_phases(
		self,
		seed_genome: Optional[ClusterGenome] = None,
	) -> Dict[str, Any]:
		"""
		Run all phases: 1a -> 1b -> 2a -> 2b -> 3a -> 3b -> final evaluation.

		Args:
			seed_genome: Optional genome to seed Phase 1a from

		Returns:
			Dictionary with all results
		"""
		results: Dict[str, Any] = {}
		completed_phases: List[PhaseResult] = []

		# =====================================================================
		# Phase 1a: GA Neurons Only
		# =====================================================================
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
		self.print_progress("After Phase 1a", completed_phases)

		# =====================================================================
		# Phase 1b: TS Neurons Only
		# =====================================================================
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
		self.print_progress("After Phase 1b", completed_phases)

		# =====================================================================
		# Phase 2a: GA Bits Only
		# =====================================================================
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
		self.print_progress("After Phase 2a", completed_phases)

		# =====================================================================
		# Phase 2b: TS Bits Only
		# =====================================================================
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
		self.print_progress("After Phase 2b", completed_phases)

		# =====================================================================
		# Phase 3a: GA Connections Only
		# =====================================================================
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
		self.print_progress("After Phase 3a", completed_phases)

		# =====================================================================
		# Phase 3b: TS Connections Only
		# =====================================================================
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
			"genome_stats": p3b.best_genome.stats(),
		}

		# =====================================================================
		# Final Summary
		# =====================================================================
		self.log("")
		self.log("=" * 78)
		self.log("  FINAL RESULTS")
		self.log("=" * 78)

		comparison = OptimizationResultsTable("Complete Phased Search")
		comparison.add_stage("Initial", ce=p1a.initial_fitness, accuracy=p1a.initial_accuracy)
		for pr in completed_phases:
			comparison.add_stage(pr.phase_name, ce=pr.final_fitness, accuracy=pr.final_accuracy)
		comparison.add_stage("Final (Full Data)", ce=final_ce, accuracy=final_acc)
		comparison.print(self.log)

		self.log("")
		self.log(f"Final best genome: {p3b.best_genome}")

		# Store for later access
		self.results = results
		self._final_genome = p3b.best_genome
		self._completed_phases = completed_phases

		return results

	@property
	def final_genome(self) -> Optional[ClusterGenome]:
		"""Get the final optimized genome."""
		return getattr(self, '_final_genome', None)
