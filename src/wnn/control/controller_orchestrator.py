"""
ControllerOrchestrator — the controller's phased stage machine on the SHARED
PhasedOrchestrator (the same skeleton IDS's _SearchOrchestrator uses).

Replaces the hand-rolled Stage 1-4 loop / carry / resume-skip that `_run_one`
used to spell out inline. Each stage is a `PhaseSpec`; `run_phase` delegates to
the existing, tested stage helpers (`_run_arch_phase` / `_run_memory_phase` /
the curriculum variants) — this migration changes ORCHESTRATION, not the
per-stage math.

Emergency/cancellation: the controller keeps its own COOPERATIVE-cancel machinery
(module-level signal handlers set the Rust cancel flag; the per-stage hook wired by
`_wire_cancel` — ControllerCancelMixin on the shared GenericGAStrategy core —
snapshots the LIVE mid-stage population, does the adaptive crash-save, and dumps at
the next gen boundary). That intra-stage granularity is strictly richer than the
old base carry-only EmergencyDump, which has been retired for both strands.

Spec threading: the base `CarryState` carries genome/population/threshold; the
controller additionally derives each stage's `ControllerSpec` from the previous
winner (`_spec_from_best`). That spec rides in `carry.extra["spec"]` (the base
loop never touches `extra`, so the subclass owns it). `carry.extra["base"]` is
the grid-winner spec = the shape base for `_spec_from_best`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from wnn.ram.strategies.phased.orchestrator import (
	PhasedOrchestrator, PhaseSpec, PhaseOutcome)
from wnn.ram.strategies.phased.carry import CarryState
from wnn.ram.strategies.phased.codecs import ControllerGenomeCodec
from wnn.ram.strategies.optimization_dimension import OptimizationDimension


# Stage table: (key, display name, kind, dimension-or-None). NEURONS is the only
# stage with curriculum variants; BITS/CONNECTIONS are plain arch phases; MEMORY
# freezes the arch and evolves cells. Order IS the pipeline order.
_STAGES = [
	("neurons",     "NEURONS",     "neurons", OptimizationDimension.NEURONS),
	("bits",        "BITS",        "arch",    OptimizationDimension.BITS),
	("connections", "CONNECTIONS", "arch",    OptimizationDimension.CONNECTIONS),
	("memory",      "MEMORY",      "memory",  None),
]


class ControllerOrchestrator(PhasedOrchestrator):
	"""Runs Stages 1-4 for one phased controller search via the shared skeleton."""

	def __init__(self, args, ec, seed: int, seeds, tracker,
	             base_spec, skip_stages: set, eid_fn: Callable[[int], Any]):
		# checkpoint_dir=None: the controller owns its own per-stage checkpoint
		# (_save_stage_checkpoint) and cooperative-cancel dump (ControllerCancelMixin,
		# wired per-stage by _wire_cancel). The base gives us ONLY the loop + carry
		# (the base EmergencyDump was retired — the strategy hook is the sole dumper).
		super().__init__(checkpoint_dir=None, codec=ControllerGenomeCodec(),
		                 log=print)
		self._args = args
		self._ec = ec
		self._seed = seed
		self._seeds = seeds
		self._tracker = tracker
		self._base_spec = base_spec
		self._skip_stages = skip_stages
		self._eid = eid_fn
		# Outputs the driver reads back after run_all.
		self.stage_results: list = []
		self.stage_holdouts: dict = {}
		self._res_by_stage: dict = {}
		# Per-stage-number report row (label, spec, metrics, dt, iters) — recorded
		# for stages that RUN and for --skip-stages skips (None metrics). The driver
		# assembles its ordered 5-row result from this; resume-sliced-out stages it
		# fills itself. row_for_stage(sn) reads it back.
		self._row_by_stage: dict = {}

	# ---- phase specs ------------------------------------------------------
	def phase_specs(self) -> list[PhaseSpec]:
		"""One PhaseSpec per stage; per-stage gens/patience live in the payload."""
		args = self._args
		gp = {
			1: (args.neurons_gens, args.neurons_patience),
			2: (args.bits_gens, args.bits_patience),
			3: (args.conns_gens, args.conns_patience),
			4: (args.memory_gens, args.memory_patience),
		}
		specs = []
		for i, (key, name, kind, dim) in enumerate(_STAGES, start=1):
			gens, patience = gp[i]
			specs.append(PhaseSpec(key=key, name=name, payload={
				"stage_num": i, "kind": kind, "dimension": dim,
				"gens": gens, "patience": patience}))
		return specs

	# ---- the one abstract hook -------------------------------------------
	def run_phase(self, spec: PhaseSpec, carry: CarryState, index: int) -> Optional[PhaseOutcome]:
		from wnn.control.phased_ga import (
			_run_arch_phase, _run_memory_phase, _run_axis_curriculum,
			_run_difficulty_curriculum, _run_adaptive_difficulty_curriculum,
			_stage_header, _print_stage_result,
			_save_stage_checkpoint, _maybe_holdout, _spec_from_best)
		p = spec.payload
		stage_num, name, kind = p["stage_num"], spec.name, p["kind"]
		cur_spec = carry.extra["spec"]

		# --skip-stages bits,connections → skip (carry + spec pass through). MEMORY
		# and NEURONS are never skippable. Record a None-metrics placeholder row so
		# the driver's ordered 5-row result keeps the stage (with its carried spec).
		if kind == "arch" and spec.key in self._skip_stages:
			print(f"[skip-stages] skipping Stage {stage_num} ({name}) — carrying population through")
			self._row_by_stage[stage_num] = (name.title(), cur_spec, None, 0.0, 0)
			return None

		_stage_header(stage_num, name, p["gens"], p["patience"], cur_spec)
		# Stage identity + crash-save wiring is self-contained in the phase function
		# (_run_arch_phase / _run_memory_phase → _wire_cancel), which derives the
		# per-stage emergency path from `args`.

		init_pop = carry.population
		args, ec, seed, tr, eid = self._args, self._ec, self._seed, self._tracker, self._eid(stage_num)
		if kind == "neurons":
			res, ev, dt = self._run_neurons(cur_spec, init_pop, eid,
				_run_arch_phase, _run_axis_curriculum,
				_run_difficulty_curriculum, _run_adaptive_difficulty_curriculum)
		elif kind == "arch":
			res, ev, dt = _run_arch_phase(args, ec, cur_spec, p["dimension"],
				p["gens"], p["patience"], seed, initial_population=init_pop,
				tracker=tr, experiment_id=eid)
		else:  # memory
			res, ev, dt = _run_memory_phase(args, ec, cur_spec, p["gens"],
				p["patience"], seed, initial_population=self._filter_cells(init_pop),
				tracker=tr, experiment_id=eid)

		m = _print_stage_result(stage_num, name, res, p["gens"], dt, ev)
		_save_stage_checkpoint(args, stage_num, name.lower(), cur_spec, res, m)
		ho = _maybe_holdout(args, ec, cur_spec, res, self._seeds, name)
		if ho is not None:
			self.stage_holdouts[name] = ho
		iters = res.iterations_run if res is not None else 0
		row = (name.title(), cur_spec, m, dt, iters)
		self.stage_results.append(row)
		self._row_by_stage[stage_num] = row
		self._res_by_stage[stage_num] = res
		self._release_prior_populations(stage_num)

		# Derive the next stage's spec from this winner's shape (unchanged on skip
		# because a skipped phase returns None above → carry.extra untouched).
		if res is not None and res.best_genome is not None:
			carry.extra["spec"] = _spec_from_best(res.best_genome, self._base_spec)
		if res is None:
			return None
		return PhaseOutcome(
			best_genome=res.best_genome,
			final_population=getattr(res, "final_population", None),
			iterations_run=iters,
			final_fitness=getattr(res, "final_fitness", None),
			strategy_type=name)

	# ---- helpers ----------------------------------------------------------
	def _run_neurons(self, spec, init_pop, eid, arch_fn, axis_fn, diff_fn, adapt_fn):
		"""Dispatch Stage 1 to the selected NEURONS variant (plain / one of the
		three curricula), all seeded from the full carried population."""
		args, ec, seed, tr = self._args, self._ec, self._seed, self._tracker
		if getattr(args, "axis_curriculum", False):
			return axis_fn(args, ec, spec, seed, initial_population=init_pop,
			               tracker=tr, experiment_id=eid)
		if getattr(args, "difficulty_adaptive", False):
			return adapt_fn(args, ec, spec, seed, initial_population=init_pop,
			                tracker=tr, experiment_id=eid)
		if getattr(args, "difficulty_curriculum", False):
			return diff_fn(args, ec, spec, seed, initial_population=init_pop,
			               tracker=tr, experiment_id=eid)
		return arch_fn(args, ec, spec, OptimizationDimension.NEURONS,
		               args.neurons_gens, args.neurons_patience, seed,
		               initial_population=init_pop, tracker=tr, experiment_id=eid)

	def _filter_cells(self, carried_pop):
		"""MEMORY mutation needs cells: keep only genomes that carry them; if none
		remain, return None so the strategy builds random cell genomes over the
		recorded universe (the winning arch still rides via the stage spec)."""
		if not carried_pop:
			return carried_pop
		with_cells = [g for g in carried_pop if getattr(g, "cells", None) is not None]
		if not with_cells:
			print("  [memory] carried population has no cells (no Lamarckian write-back) "
			      "— MEMORY starts from random cell genomes over the recorded universe.")
			return None
		if len(with_cells) < len(carried_pop):
			print(f"  [memory] dropping {len(carried_pop) - len(with_cells)} cell-less "
			      f"genomes from the carried population ({len(with_cells)} kept).")
		return with_cells

	# How many genomes per stage survive the release, for the union stage selection.
	# 3 (Luiz, 09/08/2026): "ALL top 3 should be ranked and fitness ranked together."
	# Cost is linear and small — see _release_prior_populations.
	STAGE_SELECT_TOP_K = 3

	def _release_prior_populations(self, current_stage: int) -> None:
		"""Trim `final_population` to the top-K for every stage before `current_stage`.

		Each GAResult pins 50 genomes, and a genome carries its trained cells:
		measured 21/07/2026 at 5-13.6M cells each, i.e. 120-330 MB per genome.
		Holding all four stages therefore pinned ~200 never-read genomes — tens of
		GB, and the dominant term behind the 44 GB phys_footprint (peak 55 GB).

		The population's one job during the run is seeding the NEXT phase (its own
		docstring says so), and by the time stage N is recorded, stage N-1 has
		already been consumed: the carry handed it to this stage as
		`initial_population`, and `carry.extra["spec"]` was derived from
		`best_genome` before we got here. Resume is unaffected — it restores from
		the on-disk stage checkpoints (`_save_stage_checkpoint` /
		`--resume-from-emergency`), never from this dict.

		WHY TOP-K SURVIVES RATHER THAN NOTHING (09/08/2026). One reader appears
		AFTER every stage has run: `_select_headline_stage`, which ranks stage
		candidates in ONE population (commit 77d5bde0). Releasing the whole
		population left it exactly one candidate per stage, which has two
		consequences it was never meant to have:

		  * RANK COMPRESSION. With 3 candidates a rank-WHM maps every metric to
		    {1,2,3}, so a 22% error win scores identically to a hairline lead and
		    a low-weight term (jerk .20) can outvote a high-weight one (err .40).
		  * WRONG GENOME. The survivor is `best_genome`, but the published triple
		    and the exported committee member are `final_population[0]` — and they
		    differ (measured 3/3 stages on CMT_mpc_s31337004, 5/8 across earlier
		    checkpoints). Selection was ranking a genome that is neither reported
		    nor shipped.

		Keeping the top K (=STAGE_SELECT_TOP_K) fixes both: pop[0] is in the
		candidate set by construction, and K·stages candidates give the ranking
		room to separate. Cost is bounded and small — K=3 against a population of
		50 is 6% of what used to be pinned, ~0.7-2 GB across three stages at the
		measured 120-330 MB/genome, versus the tens of GB that motivated this
		release in the first place.

		`best_genome` / `initial_genome` are KEPT as before.
		"""
		k = max(1, int(getattr(self._args, "stage_select_top_k", self.STAGE_SELECT_TOP_K)))
		for sn, res in self._res_by_stage.items():
			if sn < current_stage and res is not None:
				pop = getattr(res, "final_population", None)
				res.final_population = list(pop[:k]) if pop else None
				res.population_metrics = None

	def best_result(self):
		"""The MEMORY-stage result (final winner + population) for the caller."""
		return self._res_by_stage.get(4)

	def result_for_stage(self, stage_num: int):
		"""This stage's GAResult, or None if it never ran. Used by the val-based
		stage selection (`_select_headline_stage`): every stage's `best_genome`
		survives `_release_prior_populations` by design, so each stage's winner is
		still scoreable here even though its population is gone."""
		return self._res_by_stage.get(stage_num)

	def row_for_stage(self, stage_num: int):
		"""The (label, spec, metrics, dt, iters) report row for a stage, or None if
		that stage never entered run_phase (resume sliced it out) — the driver fills
		those with a placeholder."""
		return self._row_by_stage.get(stage_num)
