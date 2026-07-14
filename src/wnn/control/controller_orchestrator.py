"""
ControllerOrchestrator — the controller's phased stage machine on the SHARED
PhasedOrchestrator (the same skeleton IDS's _SearchOrchestrator uses).

Replaces the hand-rolled Stage 1-4 loop / carry / resume-skip that `_run_one`
used to spell out inline. Each stage is a `PhaseSpec`; `run_phase` delegates to
the existing, tested stage helpers (`_run_arch_phase` / `_run_memory_phase` /
the curriculum variants) — this migration changes ORCHESTRATION, not the
per-stage math.

Emergency/cancellation: constructed with `emergency_dumps=False`, so the base's
synchronous signal-handler dump is OFF and the controller keeps its own, richer
COOPERATIVE-cancel machinery intact (module-level signal handlers set the Rust
cancel flag; the per-generation hook installed by `_run_arch_phase` snapshots the
LIVE mid-stage population + does the adaptive periodic crash-save and the
dump-at-next-gen-boundary). That preserves intra-stage recovery granularity the
base carry-only dump would regress.

Spec threading: the base `CarryState` carries genome/population/threshold; the
controller additionally derives each stage's `ControllerSpec` from the previous
winner (`_spec_from_best`). That spec rides in `carry.extra["spec"]` (the base
loop never touches `extra`, so the subclass owns it). `carry.extra["base"]` is
the grid-winner spec = the shape base for `_spec_from_best`.
"""

from __future__ import annotations

from pathlib import Path
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
	             base_spec, skip_stages: set, eid_fn: Callable[[int], Any],
	             emergency_dir: Path):
		# checkpoint_dir=None + emergency_dumps=False: the controller owns its own
		# per-stage checkpoint (_save_stage_checkpoint) and cooperative-cancel dump,
		# so the base does neither. The base gives us ONLY the loop + carry.
		super().__init__(checkpoint_dir=None, codec=ControllerGenomeCodec(),
		                 log=print, emergency_dumps=False)
		self._args = args
		self._ec = ec
		self._seed = seed
		self._seeds = seeds
		self._tracker = tracker
		self._base_spec = base_spec
		self._skip_stages = skip_stages
		self._eid = eid_fn
		self._emergency_dir = emergency_dir
		# Outputs the driver reads back after run_all.
		self.stage_results: list = []
		self.stage_holdouts: dict = {}
		self._res_by_stage: dict = {}

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

		# --skip-stages bits,connections → skip (carry + spec pass through). MEMORY
		# and NEURONS are never skippable.
		if kind == "arch" and spec.key in self._skip_stages:
			print(f"[skip-stages] skipping Stage {stage_num} ({name}) — carrying population through")
			return None

		cur_spec = carry.extra["spec"]
		_stage_header(stage_num, name, p["gens"], p["patience"], cur_spec)
		# Stage identity + crash-save wiring is self-contained in the phase function
		# (_run_arch_phase / _run_memory_phase → _wire_cancel), which derives the
		# per-stage emergency path from `args` — no out-of-band _set_current_stage.

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
		self.stage_results.append((name.title(), cur_spec, m, dt, iters))
		self._res_by_stage[stage_num] = res

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

	def best_result(self):
		"""The MEMORY-stage result (final winner + population) for the caller."""
		return self._res_by_stage.get(4)
