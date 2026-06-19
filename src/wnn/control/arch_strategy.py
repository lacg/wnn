"""ControllerArchGAStrategy — phase-aware architecture GA for the controller.

The variable-shape counterpart to ControllerGAStrategy (ga_strategy.py). Where
that one evolves a fixed-shape FiniteStateGenome's connectivity only, this drives
a RecurrentArchGenome through ONE optimization dimension at a time — NEURONS,
BITS, or CONNECTIONS — mirroring how the IDS ArchitectureGAStrategy uses
optimize_neurons / optimize_bits / optimize_connections.

Phase gating is structural, not config-flag based: the genome's own
`mutate(dimension, …)` dispatch isolates each dimension, so a GA-Neurons phase
physically cannot drift bits or connections. `create_random_genome` likewise
randomizes ONLY the optimized dimension and pins the rest to the seed spec, so
the initial population explores the right axis.

Reuses the whole proven GenericGAStrategy loop (tournament, elitism, early-stop,
dashboard tracker) and the controller's reward-based ranking + disabled
accuracy-floor (see default_controller_ga_config). The evaluator
(ControllerEvaluator) is now variable-shape-aware: it materializes each genome's
own spec + connectivity and scores shape-uniform groups on the GPU.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from wnn.ram.strategies.connectivity.generic_strategies import (
	GenericGAStrategy, GenericTSStrategy, GAConfig, TSConfig,
)
from wnn.ram.strategies.optimization_dimension import OptimizationDimension
from wnn.ram.fitness import FitnessCalculatorType

from .evaluator import ControllerSpec, arch_shape_from_spec
from .ga_strategy import default_controller_ga_config
from .recurrent_genome import RecurrentArchGenome, RecurrentArchShape, RecurrentArchConfig


# ---- shared genome-op helpers (used by both GA and TS controller strategies) --

def _random_arch_genome(shape: RecurrentArchShape, dimension: OptimizationDimension,
                        cfg: RecurrentArchConfig, base: tuple[int, int, int, int],
                        rng: np.random.Generator) -> RecurrentArchGenome:
	"""Build a random genome that varies ONLY the optimized dimension, pinning the
	rest to the seed dims `base = (state_neurons, output_neurons, state_suffix,
	output_suffix)`. This is what makes a GA-Bits population vary bits but share
	neuron counts, etc. (the IDS optimize_* gating, applied to initialization)."""
	sn, on, ssuf, osuf = base
	q = shape.output_quantum
	if dimension in (OptimizationDimension.NEURONS, OptimizationDimension.CLUSTER):
		sn = int(rng.integers(cfg.min_state_neurons, cfg.max_state_neurons + 1))
		lo_b, hi_b = max(1, cfg.min_output_neurons // q), max(1, cfg.max_output_neurons // q)
		on = int(rng.integers(lo_b, hi_b + 1)) * q
	if dimension in (OptimizationDimension.BITS, OptimizationDimension.CLUSTER):
		ssuf = int(rng.integers(cfg.min_suffix, min(cfg.max_suffix, shape.state_input_space) + 1))
		osuf = int(rng.integers(cfg.min_suffix, min(cfg.max_suffix, shape.output_input_space) + 1))
	return RecurrentArchGenome.random(shape, sn, on, ssuf, osuf, rng)


def _arch_move_info(dimension: OptimizationDimension, before: RecurrentArchGenome,
                    after: RecurrentArchGenome):
	"""Tabu move descriptor. NEURONS/BITS → directional axis tokens (axis, ±1) so
	only the REVERSE is tabu; CONNECTIONS → the set of changed neuron tokens
	(overlap-based tabu, mirroring IDS). None if nothing changed."""
	tok: list = []
	if dimension == OptimizationDimension.NEURONS:
		if after.state_neurons != before.state_neurons:
			tok.append(("SN", 1 if after.state_neurons > before.state_neurons else -1))
		if after.output_neurons != before.output_neurons:
			tok.append(("ON", 1 if after.output_neurons > before.output_neurons else -1))
	elif dimension == OptimizationDimension.BITS:
		if after.state_suffix_width != before.state_suffix_width:
			tok.append(("SB", 1 if after.state_suffix_width > before.state_suffix_width else -1))
		if after.output_suffix_width != before.output_suffix_width:
			tok.append(("OB", 1 if after.output_suffix_width > before.output_suffix_width else -1))
	else:  # CONNECTIONS — counts unchanged, compare per-neuron suffixes
		tok += [("S", i) for i in range(before.state_neurons)
		        if before.state_sampled[i] != after.state_sampled[i]]
		tok += [("O", i) for i in range(before.output_neurons)
		        if before.output_sampled[i] != after.output_sampled[i]]
	return tuple(tok) if tok else None


def _arch_is_tabu(dimension: OptimizationDimension, move, tabu_list: list) -> bool:
	if move is None or not move:
		return False
	if dimension in (OptimizationDimension.NEURONS, OptimizationDimension.BITS):
		# Tabu iff this move reverses a recent one (opposite sign on a shared axis).
		reversed_axes = {(axis, -sign) for (axis, sign) in move}
		return any(tm and reversed_axes & set(tm) for tm in tabu_list)
	# CONNECTIONS: tabu iff >50% of changed neurons overlap a recent move.
	move_set = set(move)
	return any(tm and len(move_set & set(tm)) > len(move_set) * 0.5 for tm in tabu_list)


def default_controller_arch_config(spec: ControllerSpec) -> RecurrentArchConfig:
	"""Mutation bounds centered on `spec`, scaled to a sensible search box.

	state_neurons ∈ [2, 4×spec], output levels ∈ [¼×, 4×] spec, suffix widths
	bounded by the per-layer input space. Deltas are small (the small-neighborhood
	rule) so each generation is a local move, not a random jump."""
	q = spec.num_motors
	levels = spec.levels_per_motor
	# Forced prefix = prefix_factor·state_neurons (1-bit MSB-only state since the
	# 08/06/2026 migration; the literal 2· here was a stale 2-bit assumption that
	# silently collapsed seed suffixes to ~0 → non-uniform widths → invalid genomes).
	pf = arch_shape_from_spec(spec).prefix_factor
	state_suffix = spec.state_bits_per_neuron - pf * spec.state_neurons
	out_suffix = spec.output_bits_per_neuron - pf * spec.state_neurons
	max_suffix = max(state_suffix, out_suffix, 8) * 2
	return RecurrentArchConfig(
		min_state_neurons=2,
		max_state_neurons=max(4, 4 * spec.state_neurons),
		min_output_neurons=max(q, (levels // 4) * q),
		max_output_neurons=max(4 * levels, levels) * q,
		min_suffix=1,
		max_suffix=max_suffix,
		state_neuron_delta=1,
		output_block_delta=1,
		suffix_delta=2,
	)


def _filter_inherited_cells(child: "RecurrentArchGenome", parent: "RecurrentArchGenome",
                            *, changed_state=None, changed_output=None):
	"""Inherit `parent`'s cells into `child`, dropping entries that are invalid:

	  1. neuron_idx out of `child`'s neuron count, OR
	  2. address out of `child`'s bit-derived space (`1 << bits_per_neuron`),
	     ALSO capped at 2^64 — the controller's cell memory is keyed by u64
	     (compute_address_sparse → u64, write_*_cell takes u64). When a neuron has
	     >64 bits (2·neurons + suffix), 1<<bits exceeds u64, so addresses in
	     [2^64, 1<<bits) are physically UNADDRESSABLE and must be dropped here —
	     otherwise they reach the PyO3 write_output_cell(u64) and raise
	     OverflowError ("int too big to convert") at conversion, before the Rust
	     body's own out-of-range guard can drop them. (The 05/06 lamarckian crash.)
	  3. CONNECTIVITY-AWARE — the neuron's connections changed (neuron_idx in
	     `changed_state` / `changed_output`): the SAME address now indexes a
	     DIFFERENT input pattern, so the inherited cell is semantically stale →
	     drop it (only that neuron re-learns; unchanged neurons keep their cells).

	Returns an empty MemoryPayload if `parent` has no cells (so the MEMORY mutator
	gets a valid-but-empty universe instead of None)."""
	from .recurrent_genome import MemoryPayload
	if parent.cells is None:
		return MemoryPayload([], [], [], [])
	U64 = 1 << 64  # controller cell memory is u64-keyed; addresses ≥ this can't exist
	state_max = min(1 << child.state_bits_per_neuron, U64)
	output_max = min(1 << child.output_bits_per_neuron, U64)
	cs = changed_state or set()
	co = changed_output or set()
	nsu, nsv = [], []
	for (n, a), v in zip(parent.cells.state_universe, parent.cells.state_values):
		if n < child.state_neurons and a < state_max and n not in cs:
			nsu.append((n, a)); nsv.append(v)
	nou, nov = [], []
	for (n, a), v in zip(parent.cells.output_universe, parent.cells.output_values):
		if n < child.output_neurons and a < output_max and n not in co:
			nou.append((n, a)); nov.append(v)
	return MemoryPayload(nsu, nou, nsv, nov)


class ControllerArchGAStrategy(GenericGAStrategy):
	"""Single-dimension architecture GA over RecurrentArchGenome, ranked by reward."""

	def __init__(
		self,
		spec: ControllerSpec,
		dimension: OptimizationDimension,
		arch_config: Optional[RecurrentArchConfig] = None,
		ga_config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional[Any] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,
		lamarckian: bool = False,
	):
		# Set before super().__init__ — the base reads self.name (→ _dimension).
		self._dimension = dimension
		# Lamarckian: carry learned cells across generations (warm-start + write-back)
		# instead of re-training from scratch each eval. Opt-in (run_phased_ga --lamarckian).
		self._lamarckian = lamarckian
		super().__init__(config=ga_config or default_controller_ga_config(),
		                 seed=seed, logger=logger)
		self._spec = spec
		self._arch_config = arch_config or default_controller_arch_config(spec)
		self._shape: RecurrentArchShape = arch_shape_from_spec(spec)
		self._batch_evaluator = batch_evaluator
		self._cached_evaluator = None        # no Rust search_offspring → batch_fn path
		self._checkpoint_config = None
		self._shutdown_check = shutdown_check
		self._np_rng = np.random.default_rng(0 if seed is None else seed)
		self._random_counter = 0
		# Seed-spec dims (the pinned values for non-optimized dimensions).
		self._seed_state_neurons = spec.state_neurons
		self._seed_output_neurons = spec.num_motors * spec.levels_per_motor
		# Forced prefix = prefix_factor·state_neurons (1-bit since 08/06/2026; the
		# literal 2· was a stale 2-bit assumption → seed suffix computed as ~0,
		# collapsing connectivity into non-uniform widths → invalid genomes).
		self._seed_state_suffix = spec.state_bits_per_neuron - self._shape.prefix_factor * spec.state_neurons
		self._seed_output_suffix = spec.output_bits_per_neuron - self._shape.prefix_factor * spec.state_neurons

	@property
	def name(self) -> str:
		return f"ControllerGA-{self._dimension.name.title()}"

	# ---- the four genome operations the loop calls as black boxes -----------

	def clone_genome(self, genome: RecurrentArchGenome) -> RecurrentArchGenome:
		return genome.clone()

	def mutate_genome(self, genome: RecurrentArchGenome, mutation_rate: float) -> RecurrentArchGenome:
		return genome.mutate(self._dimension, mutation_rate, self._arch_config, self._np_rng)

	def crossover_genomes(self, parent1: RecurrentArchGenome,
	                      parent2: RecurrentArchGenome) -> RecurrentArchGenome:
		child = RecurrentArchGenome.crossover(parent1, parent2, self._np_rng)
		if self._lamarckian:
			# Inherit parent1's cells, connectivity-aware: drop cells of neurons whose
			# suffix changed (parent2 blocks mixed in) — they index a different input now.
			cs = {i for i in range(min(child.state_neurons, parent1.state_neurons))
			      if child.state_sampled[i] != parent1.state_sampled[i]}
			co = {i for i in range(min(child.output_neurons, parent1.output_neurons))
			      if child.output_sampled[i] != parent1.output_sampled[i]}
			child.cells = _filter_inherited_cells(child, parent1, changed_state=cs, changed_output=co)
		return child

	def _lamarckian_evaluate_batch(self, genomes, *a, **kw):
		"""Lamarckian eval: train (warm-started from inherited cells) + score, and
		WRITE BACK the trained cells onto each genome so offspring inherit them.
		Returns only the Metrics list (the GA loop ranks on these). 1-seed per genome
		(write-back needs a single canonical trained state — distinct from the K-fold
		evaluate_batch path; keep lamarckian ON for a whole stage)."""
		# H4: advance the curriculum generation once per GA generation (this batch
		# eval runs once per gen with the whole population). No-op unless the
		# evaluator has axis_curriculum_gens set.
		self._batch_evaluator.advance_generation()
		pairs = self._batch_evaluator.evaluate_for_adaptation(list(genomes), write_back=True)
		return [m for (m, _stats) in pairs]

	def _seed_dims(self) -> tuple[int, int, int, int]:
		return (self._seed_state_neurons, self._seed_output_neurons,
		        self._seed_state_suffix, self._seed_output_suffix)

	def create_random_genome(self) -> RecurrentArchGenome:
		"""Randomize ONLY the optimized dimension; pin the rest to the seed spec
		(so a GA-Bits population varies bits but shares neuron counts, etc.)."""
		self._random_counter += 1
		rng = np.random.default_rng(
			(0 if self._seed is None else self._seed) * 100_000 + self._random_counter)
		return _random_arch_genome(self._shape, self._dimension, self._arch_config,
		                           self._seed_dims(), rng)


def default_controller_ts_config(
	iterations: int = 6,
	neighbors_per_iter: int = 12,
	tabu_size: int = 10,
) -> TSConfig:
	"""TSConfig wired for the controller: reward ranking, no accuracy floor (the
	TS analog of default_controller_ga_config). Iterations/neighbors scaled down
	from the IDS 100/50 — the controller's K-seed inner loop is far costlier."""
	return TSConfig(
		iterations=iterations,
		neighbors_per_iter=neighbors_per_iter,
		tabu_size=tabu_size,
		progressive_threshold=False,
		min_accuracy=0.0,
		min_accuracy_floor=0.0,
		fitness_calculator_type=FitnessCalculatorType.CONTROLLER,
	)


class ControllerArchTSStrategy(GenericTSStrategy):
	"""Single-dimension architecture Tabu Search over RecurrentArchGenome.

	Same phase-isolated mutation as the GA, but local search with a tabu list.
	Uses the base GenericTSStrategy Python neighbor loop (no Rust search) — it
	calls mutate_genome (→ neighbor + move) and is_tabu_move per candidate.
	"""

	def __init__(
		self,
		spec: ControllerSpec,
		dimension: OptimizationDimension,
		arch_config: Optional[RecurrentArchConfig] = None,
		ts_config: Optional[TSConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional[Any] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		self._dimension = dimension   # set before super (base reads self.name)
		super().__init__(config=ts_config or default_controller_ts_config(),
		                 seed=seed, logger=logger)
		self._spec = spec
		self._arch_config = arch_config or default_controller_arch_config(spec)
		self._shape: RecurrentArchShape = arch_shape_from_spec(spec)
		self._batch_evaluator = batch_evaluator
		self._cached_evaluator = None     # no Rust search_neighbors → Python loop
		self._checkpoint_config = None
		self._shutdown_check = shutdown_check
		self._np_rng = np.random.default_rng(0 if seed is None else seed)
		self._seed_state_neurons = spec.state_neurons
		self._seed_output_neurons = spec.num_motors * spec.levels_per_motor
		# Forced prefix = prefix_factor·state_neurons (1-bit since 08/06/2026; the
		# literal 2· was a stale 2-bit assumption → seed suffix computed as ~0,
		# collapsing connectivity into non-uniform widths → invalid genomes).
		self._seed_state_suffix = spec.state_bits_per_neuron - self._shape.prefix_factor * spec.state_neurons
		self._seed_output_suffix = spec.output_bits_per_neuron - self._shape.prefix_factor * spec.state_neurons

	@property
	def name(self) -> str:
		return f"ControllerTS-{self._dimension.name.title()}"

	def clone_genome(self, genome: RecurrentArchGenome) -> RecurrentArchGenome:
		return genome.clone()

	def mutate_genome(self, genome: RecurrentArchGenome, mutation_rate: float):
		"""Neighbor + move descriptor (for tabu tracking)."""
		neighbor = genome.mutate(self._dimension, mutation_rate, self._arch_config, self._np_rng)
		return neighbor, _arch_move_info(self._dimension, genome, neighbor)

	def is_tabu_move(self, move, tabu_list: list) -> bool:
		return _arch_is_tabu(self._dimension, move, tabu_list)

	def seed_genome(self) -> RecurrentArchGenome:
		"""The seed-spec genome — a convenient TS starting point / initial_genome."""
		return _random_arch_genome(
			self._shape, self._dimension, self._arch_config,
			(self._seed_state_neurons, self._seed_output_neurons,
			 self._seed_state_suffix, self._seed_output_suffix), self._np_rng)


class _ControllerMemoryOps:
	"""Shared paradigm-B plumbing for MEMORY-dimension strategies (GA + TS):
	record the address universe ONCE over a fixed seed architecture, then seed
	each genome with random QSR cell values on that shared universe. Mixed into a
	ControllerArch{GA,TS}Strategy, which provides _shape/_arch_config/_seed_*/_seed."""

	def _init_memory(self, thresholds, universe, record_episodes, record_steps) -> None:
		self._thresholds = thresholds
		self._universe = universe          # (state_universe, output_universe) | None → lazy
		self._record_episodes = record_episodes
		self._record_steps = record_steps
		self._seed_genome: Optional[RecurrentArchGenome] = None
		self._mem_counter = 0

	def _seed_dims(self) -> tuple:
		return (self._seed_state_neurons, self._seed_output_neurons,
		        self._seed_state_suffix, self._seed_output_suffix)

	def _ensure_universe(self) -> None:
		if self._universe is not None and self._seed_genome is not None:
			return
		from .evaluator import spec_from_arch
		from .ga_memory import record_address_universe
		# Fixed-arch seed genome (connectivity sampled once; never mutated in MEMORY).
		self._seed_genome = _random_arch_genome(
			self._shape, OptimizationDimension.CONNECTIONS, self._arch_config,
			self._seed_dims(), np.random.default_rng(0 if self._seed is None else self._seed))
		sc, oc = self._seed_genome.to_connections()
		spec = spec_from_arch(self._seed_genome, self._spec)
		th, ev = self._thresholds, self._batch_evaluator
		if th is None and ev is not None:
			ev._ensure_ga_ready()
			th = ev.thresholds
		steps = self._record_steps or (getattr(getattr(ev, "episode_config", None),
		                                        "steps_per_episode", None) or 1000)
		self._universe = record_address_universe(
			spec, th, sc, oc, num_episodes=self._record_episodes, steps=steps,
			seed=0 if self._seed is None else self._seed)

	def _make_cell_genome(self) -> RecurrentArchGenome:
		from .recurrent_genome import MemoryPayload
		self._ensure_universe()
		self._mem_counter += 1
		rng = np.random.default_rng(
			(0 if self._seed is None else self._seed) * 100_000 + self._mem_counter)
		su, ou = self._universe
		g = self._seed_genome.clone()
		g.cells = MemoryPayload(
			list(su), list(ou),
			[int(v) for v in rng.integers(0, 4, len(su))],
			[int(v) for v in rng.integers(0, 4, len(ou))])
		return g


class ControllerMemoryGAStrategy(_ControllerMemoryOps, ControllerArchGAStrategy):
	"""Paradigm B / GA: FIXED architecture, evolve QSR cell VALUES over a recorded
	universe. Per-cell crossover_memory aligns (shared universe). Score with NO
	training — drive optimize() with batch_evaluate_fn=evaluator.score_genomes."""

	def __init__(self, spec: ControllerSpec, *, thresholds: Optional[list] = None,
	             universe: Optional[tuple] = None, record_episodes: int = 8,
	             record_steps: Optional[int] = None, **kw):
		super().__init__(spec, OptimizationDimension.MEMORY, **kw)
		self._init_memory(thresholds, universe, record_episodes, record_steps)

	def create_random_genome(self) -> RecurrentArchGenome:
		return self._make_cell_genome()

	def crossover_genomes(self, parent1: RecurrentArchGenome,
	                      parent2: RecurrentArchGenome) -> RecurrentArchGenome:
		return RecurrentArchGenome.crossover_memory(parent1, parent2, self._np_rng)


class ControllerMemoryTSStrategy(_ControllerMemoryOps, ControllerArchTSStrategy):
	"""Paradigm B / Tabu Search: local search over cell VALUES (fixed arch).
	Move = the set of cell positions whose value changed; tabu by >50% overlap
	(re-stirring the same cells). Score with NO training (score_genomes)."""

	def __init__(self, spec: ControllerSpec, *, thresholds: Optional[list] = None,
	             universe: Optional[tuple] = None, record_episodes: int = 8,
	             record_steps: Optional[int] = None, **kw):
		super().__init__(spec, OptimizationDimension.MEMORY, **kw)
		self._init_memory(thresholds, universe, record_episodes, record_steps)

	def seed_genome(self) -> RecurrentArchGenome:
		return self._make_cell_genome()

	def mutate_genome(self, genome: RecurrentArchGenome, mutation_rate: float):
		neighbor = genome.mutate(OptimizationDimension.MEMORY, mutation_rate,
		                         self._arch_config, self._np_rng)
		move = self._memory_move(genome, neighbor)
		return neighbor, move

	@staticmethod
	def _memory_move(before: RecurrentArchGenome, after: RecurrentArchGenome):
		if before.cells is None or after.cells is None:
			return None
		tok = [("S", i) for i in range(len(before.cells.state_values))
		       if before.cells.state_values[i] != after.cells.state_values[i]]
		tok += [("O", i) for i in range(len(before.cells.output_values))
		        if before.cells.output_values[i] != after.cells.output_values[i]]
		return tuple(tok) if tok else None

	def is_tabu_move(self, move, tabu_list: list) -> bool:
		if not move:
			return False
		s = set(move)
		return any(tm and len(s & set(tm)) > len(s) * 0.5 for tm in tabu_list)


class ControllerMixedGAStrategy(_ControllerMemoryOps, ControllerArchGAStrategy):
	"""ONE GA that evolves ALL FOUR dimensions at once. Each offspring is produced
	by a uniformly-random operator among {NEURONS, BITS, CONNECTIONS, MEMORY} — so
	with 20% elitism + 80% offspring you get ~20% per operator each generation.

	Full neuroevolution of the unified genome: the genome carries cells (universe
	recorded ONCE for the seed arch, then REMAPPED through arch mutations by the 4a
	machinery — no per-genome re-recording). Scored with NO training — drive
	optimize() with batch_evaluate_fn=evaluator.score_genomes. Crossover is disabled
	(set crossover_rate=0.0) because variable-shape, cell-carrying genomes don't
	align for recombination; offspring come purely from the four mutation operators."""

	_DIMS = (OptimizationDimension.NEURONS, OptimizationDimension.BITS,
	         OptimizationDimension.CONNECTIONS, OptimizationDimension.MEMORY)

	def __init__(self, spec: ControllerSpec, *, thresholds: Optional[list] = None,
	             universe: Optional[tuple] = None, record_episodes: int = 8,
	             record_steps: Optional[int] = None, **kw):
		super().__init__(spec, OptimizationDimension.CLUSTER, **kw)  # _dimension unused
		self._init_memory(thresholds, universe, record_episodes, record_steps)

	def create_random_genome(self) -> RecurrentArchGenome:
		return self._make_cell_genome()

	def mutate_genome(self, genome: RecurrentArchGenome, mutation_rate: float) -> RecurrentArchGenome:
		dim = self._DIMS[int(self._np_rng.integers(0, len(self._DIMS)))]
		child = genome.mutate(dim, mutation_rate, self._arch_config, self._np_rng)
		try:
			child.assert_valid()  # never let a malformed genome poison the population
		except AssertionError:
			return genome.clone()
		return child

	def crossover_genomes(self, parent1: RecurrentArchGenome,
	                      parent2: RecurrentArchGenome) -> RecurrentArchGenome:
		"""Average-shape variable crossover (the user's spec, 2026-05-28).

		Child takes the ELEMENT-WISE AVERAGE of parent shapes (state_neurons,
		output_neurons rounded to output_quantum, suffix widths). Per-neuron
		suffixes are sampled from parent positions and resized to the new width.

		Cells: inherited from one parent (uniformly picked) and FILTERED to
		drop entries whose neuron_idx is out of bounds for the child or whose
		address exceeds the child's bit-derived address space. This gives the
		child a VALID subset of cells the MEMORY mutator can refine — without
		this, cells=None crashes the MEMORY mutator (`_mutate_memory` requires
		cells.universe to exist).

		Plan B fallback: `RecurrentArchGenome.crossover` (one parent's shape
		+ mixed-block suffixes). Preserves cells, doesn't interpolate shape.
		Used if `crossover_average` raises or produces an invalid genome.
		"""
		try:
			child = RecurrentArchGenome.crossover_average(parent1, parent2, self._np_rng)
			# Inherit + filter cells from one parent so the MEMORY mutator has
			# valid input. Cells inherited from a different-shape parent are
			# partially stale (addresses derive from parent's suffixes, child's
			# connections differ), but the VALID subset (neuron_idx + address
			# in bounds) gives the GA something to refine. The runtime won't
			# hit most of these addresses, but the cells that ARE hit on the
			# rare matching pattern contribute signal.
			src = parent1 if self._np_rng.random() < 0.5 else parent2
			child.cells = self._filter_inherited_cells(child, src)
			child.assert_valid()
			return child
		except AssertionError:
			# Pathological average-shape recombination — fall back to the
			# Plan B one-parent-shape variant which is structurally safe.
			try:
				child = RecurrentArchGenome.crossover(parent1, parent2, self._np_rng)
				child.assert_valid()
				return child
			except AssertionError:
				# Both crossover variants failed — clone the stronger parent.
				return parent1.clone()

	def _filter_inherited_cells(self, child: RecurrentArchGenome,
	                            parent: RecurrentArchGenome,
	                            changed_state=None, changed_output=None):
		return _filter_inherited_cells(child, parent,
		                               changed_state=changed_state,
		                               changed_output=changed_output)


def controller_ts_neurons(spec: ControllerSpec, **kw) -> ControllerArchTSStrategy:
	"""TS-Neurons: local search over state-neuron count + output levels."""
	return ControllerArchTSStrategy(spec, OptimizationDimension.NEURONS, **kw)


def controller_ts_bits(spec: ControllerSpec, **kw) -> ControllerArchTSStrategy:
	"""TS-Bits: local search over sampled-suffix width per layer."""
	return ControllerArchTSStrategy(spec, OptimizationDimension.BITS, **kw)


def controller_ts_connections(spec: ControllerSpec, **kw) -> ControllerArchTSStrategy:
	"""TS-Connectivity: local search over which sampled input bits each neuron reads."""
	return ControllerArchTSStrategy(spec, OptimizationDimension.CONNECTIONS, **kw)


def controller_ga_neurons(spec: ControllerSpec, **kw) -> ControllerArchGAStrategy:
	"""GA-Neurons: evolve state-neuron count (memory capacity) + output levels
	(PWM resolution); bits + connectivity held at the seed."""
	return ControllerArchGAStrategy(spec, OptimizationDimension.NEURONS, **kw)


def controller_ga_bits(spec: ControllerSpec, **kw) -> ControllerArchGAStrategy:
	"""GA-Bits: evolve sampled-suffix width per layer (synaptogenesis); neuron
	counts + connectivity held at the seed."""
	return ControllerArchGAStrategy(spec, OptimizationDimension.BITS, **kw)


def controller_ga_connections(spec: ControllerSpec, **kw) -> ControllerArchGAStrategy:
	"""GA-Connectivity: evolve which sampled input bits each neuron reads
	(axonogenesis); shape held at the seed. Equivalent role to the legacy
	ControllerGAStrategy but on the variable-shape genome."""
	return ControllerArchGAStrategy(spec, OptimizationDimension.CONNECTIONS, **kw)


# ---- WnnType-factory registration -------------------------------------------
# Self-register the controller family so wnn_factory.create_strategy can build
# these without the ram-level factory ever importing control (registry pattern).

from wnn.ram.strategies.wnn_factory import WnnType, StrategyKind, register_wnn_type

_GA_BY_DIM = {
	OptimizationDimension.NEURONS: controller_ga_neurons,
	OptimizationDimension.BITS: controller_ga_bits,
	OptimizationDimension.CONNECTIONS: controller_ga_connections,
}
_TS_BY_DIM = {
	OptimizationDimension.NEURONS: controller_ts_neurons,
	OptimizationDimension.BITS: controller_ts_bits,
	OptimizationDimension.CONNECTIONS: controller_ts_connections,
}


# Lamarckian dimension → genesis mode. MEMORY = cells acquired by training +
# inherited (write-back/warm-start), fixed arch. CONNECTIONS (axonogenesis) needs
# per-input-bit entropy from new Rust instrumentation (step 4b-5).
_GENESIS_MODE_BY_DIM = {
	OptimizationDimension.NEURONS: "neurogenesis",
	OptimizationDimension.BITS: "synaptogenesis",
	OptimizationDimension.CONNECTIONS: "axonogenesis",
	OptimizationDimension.MEMORY: "memory",
}


def _controller_strategy_builder(kind: StrategyKind, dimension: OptimizationDimension,
                                 *, spec: ControllerSpec, **kwargs):
	"""WnnType.CONTROLLER builder. GA/TS over the architecture dimensions and
	Lamarckian neuro/synapto-genesis are wired; MEMORY (paradigm B) and
	Lamarckian axonogenesis are pending."""
	if kind == StrategyKind.GA and dimension in _GA_BY_DIM:
		return _GA_BY_DIM[dimension](spec, **kwargs)
	if kind == StrategyKind.TS and dimension in _TS_BY_DIM:
		return _TS_BY_DIM[dimension](spec, **kwargs)
	if kind == StrategyKind.LAMARCKIAN:
		mode = _GENESIS_MODE_BY_DIM.get(dimension)
		if mode is not None:
			# axonogenesis (CONNECTIONS) builds fine; at RUN time it needs the
			# WnnController.last_*_layer_input getters — record_input_entropy raises
			# a clear "run maturin develop --release" until the accelerator is rebuilt
			# (held until the 46M flow finishes).
			from .arch_adaptation import ControllerAdaptationStrategy  # lazy: avoid import cycle
			return ControllerAdaptationStrategy(spec, genesis_mode=mode, **kwargs)
	if dimension == OptimizationDimension.MEMORY:
		# Paradigm B on the unified genome: evolve cell values over a recorded
		# universe (fixed arch). Score with evaluator.score_genomes (no training).
		if kind == StrategyKind.GA:
			return ControllerMemoryGAStrategy(spec, **kwargs)
		if kind == StrategyKind.TS:
			return ControllerMemoryTSStrategy(spec, **kwargs)
		raise NotImplementedError(
			f"controller MEMORY is wired for GA + TS (paradigm B); {kind.value} not built.")
	raise ValueError(f"unsupported controller strategy: kind={kind}, dimension={dimension}")


register_wnn_type(WnnType.CONTROLLER, _controller_strategy_builder)


__all__ = [
	"ControllerArchGAStrategy",
	"ControllerArchTSStrategy",
	"ControllerMemoryGAStrategy",
	"ControllerMemoryTSStrategy",
	"ControllerMixedGAStrategy",
	"default_controller_arch_config",
	"default_controller_ts_config",
	"controller_ga_neurons",
	"controller_ga_bits",
	"controller_ga_connections",
	"controller_ts_neurons",
	"controller_ts_bits",
	"controller_ts_connections",
]
