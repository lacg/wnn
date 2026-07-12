"""ControllerGAStrategy — the connectivity GA for the drone controller.

Option A: subclass the existing `GenericGAStrategy` so we reuse the whole proven
GA loop (tournament selection, elitism, early-stopping, fitness caching) AND the
dashboard tracker (record_iteration / record_genome_evaluations_batch fire
automatically whenever a tracker is attached via set_tracker — drop-in at worker
time). We only implement the four genome operations + point the loop at the
controller's evaluator.

What this evolves: the input-bit connectivity of a `FiniteStateGenome` (the
forced full-state recurrent wiring is preserved by the genome's own operators —
see genome.py). Cells are NOT evolved here; they are trained per genome by the
inner loop (C1 reward-gated imitation / C2 reinforce-own-wins) inside
`ControllerEvaluator.evaluate_batch`. (Evolving the cells instead = paradigm
"GA-Memory" / B — a different genome.)

IDS-specific machinery is neutralised by config, not by code:
  - progressive_threshold=False, min_accuracy=0 → the accuracy-floor viability
    filter never rejects a genome (controller "accuracy" = stable_rate starts at
    0 and must be allowed through).
  - fitness_calculator_type=CONTROLLER → rank by closed-loop reward.
  - F1/FPR stay None (the loop already guards `m.f1 is not None`).
  - GPU shape-grouping lumps all genomes into one group (they ARE one shape) —
    harmless; the real GPU win is in ControllerEvaluator.score_population.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from wnn.ram.strategies.connectivity.generic_strategies import GenericGAStrategy, GAConfig
from wnn.ram.fitness import FitnessCalculatorType

from .evaluator import ControllerSpec
from .genome import FiniteStateGenome


def default_controller_ga_config(
	population_size: int = 12,
	generations: int = 6,
	# Canonical GA-Neurons values (GAConfig defaults) — NOT reinvented:
	#   elitism_pct=0.2 (= 20% kept; formula is int(pop*elitism_pct), no hidden ×2);
	#   tournament_size=3; mutation_rate=0.1.
	# population_size/generations are problem-scaled down from the IDS 50/50
	# (the controller's K-seed inner loop is far costlier per genome).
	mutation_rate: float = 0.1,
	crossover_rate: float = 0.7,
	tournament_size: int = 3,
	elitism_pct: float = 0.2,
	# Controller multi-objective weights (29/05/2026). Defaults (err_sq=1.0,
	# others=0) reproduce the CONTROLLER calculator exactly — purely single-
	# objective on integrated err². Setting any of stable/jerk/mono > 0
	# auto-switches to the harmonic-rank calculator (CONTROLLER_HARMONIC).
	weight_err_sq: float = 1.0,
	weight_stable: float = 0.0,
	weight_jerk:   float = 0.0,
	weight_mono:   float = 0.0,
	weight_steady: float = 0.0,
	weight_effort: float = 0.0,
) -> GAConfig:
	"""GAConfig wired for the controller: reward ranking, no accuracy floor.
	Hyperparameters match the canonical GA-Neurons phase; only the reward
	calculator + disabled accuracy-floor are controller-specific."""
	multi_obj = (weight_stable > 0 or weight_jerk > 0 or weight_mono > 0
	             or weight_steady > 0 or weight_effort > 0)
	return GAConfig(
		population_size=population_size,
		generations=generations,
		mutation_rate=mutation_rate,
		crossover_rate=crossover_rate,
		tournament_size=tournament_size,
		elitism_pct=elitism_pct,
		# Disable the IDS accuracy-floor viability filter (stable_rate starts 0).
		progressive_threshold=False,
		min_accuracy=0.0,
		min_accuracy_floor=0.0,
		# Rank by single-obj reward (default) or multi-obj harmonic (any weight > 0).
		fitness_calculator_type=(FitnessCalculatorType.CONTROLLER_HARMONIC
		                         if multi_obj else
		                         FitnessCalculatorType.CONTROLLER),
		# Controller-specific weight fields (added to GAConfig 29/05/2026).
		# These have NO effect when fitness_calculator_type=CONTROLLER.
		fitness_weight_err_sq=weight_err_sq,
		fitness_weight_stable=weight_stable,
		fitness_weight_jerk=weight_jerk,
		fitness_weight_mono=weight_mono,
		fitness_weight_steady=weight_steady,
		fitness_weight_effort=weight_effort,
	)


class ControllerGAStrategy(GenericGAStrategy):
	"""Connectivity GA over FiniteStateGenome, scored by closed-loop reward."""

	def __init__(
		self,
		spec: ControllerSpec,
		ga_config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional[Any] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		super().__init__(config=ga_config or default_controller_ga_config(),
		                 seed=seed, logger=logger)
		self._spec = spec
		self._batch_evaluator = batch_evaluator       # ControllerEvaluator (for the streaming path's getattr)
		self._cached_evaluator = None                 # no Rust search_offspring → use batch_fn path
		self._checkpoint_config = None
		self._shutdown_check = shutdown_check
		# Own numpy RNG for genome ops (the base's self._rng is a random.Random;
		# FiniteStateGenome's operators use numpy). Seeded for reproducibility.
		self._np_rng = np.random.default_rng(0 if seed is None else seed)
		self._random_counter = 0

	@property
	def name(self) -> str:
		return "ControllerGA"

	# ---- the four genome operations the loop calls as black boxes -----------

	def clone_genome(self, genome: FiniteStateGenome) -> FiniteStateGenome:
		return genome.clone()

	def mutate_genome(self, genome: FiniteStateGenome, mutation_rate: float) -> FiniteStateGenome:
		return genome.mutate(self._np_rng, mutation_rate)

	def crossover_genomes(self, parent1: FiniteStateGenome, parent2: FiniteStateGenome) -> FiniteStateGenome:
		return FiniteStateGenome.crossover(parent1, parent2, self._np_rng)

	def create_random_genome(self) -> FiniteStateGenome:
		# Distinct seed per genome so the initial population is diverse.
		self._random_counter += 1
		base = (0 if self._seed is None else self._seed) * 100_000 + self._random_counter
		return FiniteStateGenome.random(self._spec, seed=base)


__all__ = ["ControllerGAStrategy", "default_controller_ga_config"]
