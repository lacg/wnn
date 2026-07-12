"""Shared optimization configs (GA / TS / SA) — single source of truth for
fitness ranking, threshold progression, and early stopping knobs."""

from dataclasses import dataclass, field
from typing import Optional

from wnn.ram.fitness import FitnessCalculatorType, FitnessCalculatorFactory


@dataclass
class OptimizationConfig:
	"""Shared configuration for all optimization strategies (GA, TS, etc.).

	Single source of truth for fitness ranking, threshold progression,
	early stopping, and percentile filtering.
	"""
	mutation_rate: float = 0.1
	# Threshold continuity: start threshold passed from previous phase
	initial_threshold: Optional[float] = None
	min_accuracy: float = 0.0
	threshold_delta: float = 0.01
	threshold_reference: int = 1000
	progressive_threshold: bool = True
	# Fitness percentile filter (None = disabled)
	fitness_percentile: Optional[float] = None
	# Fitness calculator: unified ranking for all selection/sorting
	# HARMONIC_RANK = harmonic mean of CE+Acc ranks (default)
	# CE = pure CE ranking
	# NORMALIZED = normalized [0,1] weighted sum
	# NORMALIZED_HARMONIC = normalized values with harmonic mean
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	# IDS weights (used by HARMONIC_RANK / NORMALIZED / NORMALIZED_HARMONIC).
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	fitness_weight_f1: float = 0.0
	fitness_weight_fpr: float = 0.0
	# Controller weights (used by CONTROLLER_HARMONIC only). Explicit names;
	# do NOT alias to IDS weights — schema clarity > sprawl avoidance.
	fitness_weight_err_sq: float = 1.0
	fitness_weight_stable: float = 0.0
	fitness_weight_jerk:   float = 0.0
	fitness_weight_mono:   float = 0.0
	fitness_weight_steady: float = 0.0
	fitness_weight_effort: float = 0.0
	min_accuracy_floor: float = 0.0
	# Early stopping
	patience: int = 5
	check_interval: int = 10
	min_improvement_pct: float = 0.1
	# Magnitude-aware patience (controller fitness redesign (a), 16/06/2026 —
	# docs/controller_fitness_patience_redesign.md; generalized to IDS
	# 11/07/2026 via the shared check_magnitude_metrics core). The rank-WHM
	# fitness is magnitude-blind: a real physical jump barely moves it, so the
	# patience tracker watching WHM mis-early-stops. When this is True the
	# early-stopper watches the MAGNITUDE of the domain's physical metrics —
	# err°/stable% for controllers, F1/FPR for IDS — and recovers patience
	# PROPORTIONALLY (err halved → recover ~2; stable 20→70% → recover ~3.5).
	# SELECTION is unchanged (still rank-WHM) → cross-run comparability
	# preserved. Off by default here; the IDS worker defaults it ON via the
	# magnitude_aware_patience flow param (SP wave-1 restart, 11/07/2026).
	magnitude_aware_patience: bool = False
	mag_patience_eps_err: float = 0.5        # ε_err floor (deg) — guards div-0 near 0°
	mag_patience_stable_offset: float = 0.05  # s0 additive — tames stable=0 in the ratio
	mag_patience_delta: float = 0.05         # δ noise gate — ρ below 1+δ counts as no-improvement
	mag_patience_rho_cap: float = 0.0        # ρ recovery cap (0 ⇒ use `patience` as the cap)
	# IDS variant knobs (check_magnitude_ids watches F1↑/FPR↓; 11/07/2026).
	# F1 moves are fractions of a point late-search, so the noise gate is 1%.
	mag_patience_delta_ids: float = 0.01     # δ noise gate for the F1/FPR ratios
	mag_patience_eps_fpr: float = 0.005      # ε additive stabilizer for the FPR ratio

	@property
	def fitness_weights(self) -> 'FitnessWeights':
		from wnn.ram.metrics import FitnessWeights
		return FitnessWeights(ce=self.fitness_weight_ce, acc=self.fitness_weight_acc,
							  f1=self.fitness_weight_f1, fpr=self.fitness_weight_fpr)

	def create_fitness_calculator(self) -> 'FitnessCalculator':
		"""Create a FitnessCalculator from this config. CONTROLLER_HARMONIC
		uses its OWN explicitly-named weight fields (fitness_weight_err_sq /
		_stable / _jerk / _mono); other types use the ce/acc/f1/fpr family."""
		from wnn.ram.fitness import FitnessCalculatorType
		extra = {}
		if self.fitness_calculator_type == FitnessCalculatorType.CONTROLLER_HARMONIC:
			extra = dict(
				weight_err_sq=self.fitness_weight_err_sq,
				weight_stable=self.fitness_weight_stable,
				weight_jerk=self.fitness_weight_jerk,
				weight_mono=self.fitness_weight_mono,
				weight_steady=self.fitness_weight_steady,
				weight_effort=self.fitness_weight_effort,
			)
		return FitnessCalculatorFactory.create(
			self.fitness_calculator_type,
			weights=self.fitness_weights,
			min_accuracy_floor=self.min_accuracy_floor if self.min_accuracy_floor > 0 else None,
			**extra,
		)


@dataclass


@dataclass
class GAConfig(OptimizationConfig):
	"""Configuration for Genetic Algorithm."""
	population_size: int = 50
	generations: int = 50
	crossover_rate: float = 0.7
	tournament_size: int = 3
	# Elitism: keep the top elitism_pct of the population by fitness (unified ranking).
	# 0.2 = 20% kept (the formula is now `int(pop * elitism_pct)` — no hidden ×2).
	elitism_pct: float = 0.2
	# GA-specific early stopping threshold (lower than TS because GA needs diversity)
	min_improvement_pct: float = 0.05
	# Fresh population: ignore initial_population and generate random genomes
	fresh_population: bool = False
	# Seed only: use seed genomes as-is without generating mutations to fill population
	seed_only: bool = False
	# Random immigrants (diversity preservation, plan controller_break_90_v2 E1):
	# each offspring slot has this probability of being a FRESH random genome
	# (create_random_genome) instead of a bred child. Counters premature
	# convergence (one lineage fixating the population by ~gen 50). 0.0 = off.
	immigrant_fraction: float = 0.0
	# Random-search baseline (RAID'26 Review C): every offspring slot is a fresh
	# random genome — zero selection pressure, everything else (evaluation
	# protocol, μ+λ best-of-pool tracking, patience, validation checkpoints)
	# identical to the GA, so best-fitness-vs-evaluations curves are directly
	# comparable at matched compute budget.
	random_search: bool = False


@dataclass


@dataclass
class TSConfig(OptimizationConfig):
	"""Configuration for Tabu Search optimization."""
	iterations: int = 100
	neighbors_per_iter: int = 50
	tabu_size: int = 10
	# Total neighbors cache for seeding next phase (top K by fitness)
	total_neighbors_size: int = 50
	# TS-specific early stopping threshold (higher than GA because TS is more focused)
	min_improvement_pct: float = 0.5
	# Cooperative multi-start: fraction of top genomes used as neighbor sources.
	# 0.0 = single best (classic TS), 0.2 = top 20% of cache as reference set.
	# Based on Crainic, Toulouse & Gendreau (1997) cooperative TS taxonomy.
	diversity_sources_pct: float = 0.2


@dataclass


@dataclass
class SAConfig(OptimizationConfig):
	"""Configuration for Simulated Annealing optimization.

	Hyperparameters from Garcia (2003), carried over from the original
	connectivity SA (IJCNN 2004 lineage):
	- iterations: 600 for convergence
	- initial_temp: 1.0 (best of {1, 0.5, 0.1})
	- cooling_rate: 0.95 (best of {0.99, 0.95, 0.9, 0.85})

	Modernization: `chains` independent annealing chains run in lockstep so
	every iteration evaluates all chain proposals in ONE batch_evaluate_fn
	call (Rust/Metal batch evaluation), and the chain states form the final
	population carried to the next phase.
	"""
	iterations: int = 600
	initial_temp: float = 1.0
	cooling_rate: float = 0.95
	# Parallel independent annealing chains (also the carried population size)
	chains: int = 20
	# SA-specific early stopping threshold (matches TS)
	min_improvement_pct: float = 0.5
