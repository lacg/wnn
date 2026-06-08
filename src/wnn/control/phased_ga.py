"""Phased-GA controller search (5 stages, total 2000 GA gens + ~20 grid points).

Stages run sequentially; each warm-starts from the previous stage's best
genome. Mirrors the canonical IDS phased-search pattern (grid → ga_neurons →
ga_bits → ga_connections → ga_memory) but tailored to the variable-shape
RecurrentArchGenome + ControllerEvaluator stack.

Stage 0 (Grid):       (state_neurons × bits) × levels=16 — direct cell-scored.
Stage 1 (NEURONS):    ControllerArchGAStrategy(NEURONS)     — 400g, patience 20.
Stage 2 (BITS):       ControllerArchGAStrategy(BITS)        — 400g, patience 20.
Stage 3 (CONNECTIONS):ControllerArchGAStrategy(CONNECTIONS) — 400g, patience 20.
Stage 4 (MEMORY):     ControllerMemoryGAStrategy             — 800g, patience 40.

Warm-start chain: the best genome's (state_neurons, output_neurons, state_bits,
output_bits) becomes the seed ControllerSpec of the next stage. The next stage's
strategy re-records its universe for its own seed arch via _ensure_universe.

Smoke test (tiny budget, end-to-end):
  python tests/run_phased_ga.py \
    --grid-state-neurons 4 8 --grid-bits 16 24 \
    --neurons-gens 5 --bits-gens 5 --conns-gens 5 --memory-gens 5 \
    --pop 12 --eval-episodes 2 --steps 200 --universe-episodes 2

Production run (the spec):
  RAYON_NUM_THREADS=3 python tests/run_phased_ga.py \
    --grid-state-neurons 8 12 16 20 24 --grid-bits 18 24 30 36 --levels 16 \
    --neurons-gens 400 --neurons-patience 20 \
    --bits-gens 400 --bits-patience 20 \
    --conns-gens 400 --conns-patience 20 \
    --memory-gens 800 --memory-patience 40 \
    --pop 200 --elitism 0.2 --crossover-rate 0.5 \
    --eval-episodes 20 --steps 1500 --tilt 15 \
    --universe-episodes 8 --seed 42
"""

from __future__ import annotations

import argparse
import math
import pickle
import signal
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================================
# Emergency dump on SIGTERM (added 31/05/2026 after Plan A v3 lost 19h of work)
# ============================================================================
#
# Behavior:
#   1. SIGTERM (or SIGINT) sets the process-wide Rust cancel flag via
#      ram_accelerator.set_cancel_flag(). Rust evaluators (controller GPU and,
#      separately, IDS Rust) poll this flag at safe boundaries (~25 ms on
#      controller GPU, per-genome on CPU paths). They return whatever they
#      have so far instead of running to completion.
#   2. The next call to the GA's _on_generation_start hook sees the flag,
#      dumps the current stage + population + spec + best genome to a pickle,
#      and raises StopIteration. The strategy catches StopIteration, marks
#      the run as shutdown-requested, and returns cleanly.
#   3. The pickle schema mirrors --save-winner so the resume path can load
#      it like any other stage checkpoint.
#
# Response time:
#   - GPU evaluator: ~25 ms (between dispatch chunks of EPISODES_PER_CHUNK)
#   - CPU training: ~5-10 s (between genomes in the train loop; will tighten
#     when controller_training.rs gets per-episode polling)
#   - Worst case (mid-genome): one genome's train + score time
#
# Resume CLI:
#   --resume-from-emergency PATH   load the dump
#   --resume-same-stage            continue the same stage from the dumped
#                                  generation (default when --resume-from-
#                                  emergency is set)
#   --resume-next-stage            skip the dumped stage entirely and start
#                                  the NEXT stage with the dumped best as
#                                  warm-start

_emergency_state: dict = {
	"stage_num":  None,
	"stage_name": None,
	"spec":       None,
	"population": [],
	"best_genome": None,
	"generation": 0,
	"save_path":  None,
	"args":       None,
}


def _sigterm_handler(signum, _frame) -> None:
	"""Process-wide signal handler. Sets the Rust cancel flag so in-flight
	Rust calls return promptly with partial results. The actual state dump
	happens at the next GA generation boundary (in the patched
	_on_generation_start)."""
	name = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT"}.get(signum, str(signum))
	print(f"\n[{name}] Cancellation requested. Setting Rust cancel flag — "
	      f"will dump state and exit at next safe point.", flush=True)
	# Mark the Python-level PROPER-cancel witness FIRST, then set the Rust flag.
	# The evaluator cancel-guard reads sigterm_received() to tell a real shutdown
	# (return sentinels → GA unwinds → emergency dump + exit) from a spurious
	# flag (reset + retry). Without this the guard sees the Rust flag set but no
	# witness → classifies the real SIGTERM as SPURIOUS → resets + keeps running,
	# i.e. the process IGNORES SIGTERM. Set the witness before the Rust flag so
	# there is no window where a poll observes is_cancelled() without the witness.
	try:
		from wnn.control import cancel_state
		cancel_state.mark_sigterm(signum)
	except Exception as e:
		print(f"[{name}] Could not mark proper-cancel witness: {e}", flush=True)
	try:
		import ram_accelerator
		ram_accelerator.set_cancel_flag()
	except Exception as e:
		print(f"[{name}] Could not set Rust cancel flag: {e}", flush=True)


def _install_signal_handlers() -> None:
	"""Wire SIGTERM + SIGINT to the cooperative-cancellation path."""
	signal.signal(signal.SIGTERM, _sigterm_handler)
	signal.signal(signal.SIGINT,  _sigterm_handler)
	# Make sure no prior process left the Rust flag set.
	try:
		import ram_accelerator
		ram_accelerator.reset_cancel_flag()
	except Exception:
		pass
	# Clear the Python proper-cancel witness too (symmetry with the Rust reset;
	# matters on in-process re-entry / resume so a stale witness can't make the
	# guard treat a later spurious flag as a real shutdown).
	try:
		from wnn.control import cancel_state
		cancel_state.reset_sigterm()
	except Exception:
		pass


def _set_current_stage(stage_num: int, stage_name: str, spec, args, save_path) -> None:
	"""Update the module-level emergency state so the GA hook knows what to
	dump if cancellation hits during this stage."""
	_emergency_state["stage_num"]  = stage_num
	_emergency_state["stage_name"] = stage_name
	_emergency_state["spec"]       = spec
	_emergency_state["args"]       = args
	_emergency_state["save_path"]  = save_path
	_emergency_state["population"] = []
	_emergency_state["best_genome"] = None
	_emergency_state["generation"] = 0


def _dump_emergency_state() -> None:
	"""Pickle the current emergency state. Schema mirrors _save_winner so
	the resume path can load it like any other stage checkpoint."""
	path = _emergency_state.get("save_path")
	if path is None:
		print("[emergency-dump] No save_path set — cannot dump.", flush=True)
		return
	args = _emergency_state["args"]
	payload = {
		"stage_num":   _emergency_state["stage_num"],
		"stage_name":  _emergency_state["stage_name"],
		"spec":        _emergency_state["spec"],
		"population":  _emergency_state["population"],
		"best_genome": _emergency_state["best_genome"],
		"generation":  _emergency_state["generation"],
		"fitness_weights": {
			"err_sq": args.fit_weight_err_sq,
			"stable": args.fit_weight_stable,
			"jerk":   args.fit_weight_jerk,
			"mono":   args.fit_weight_mono,
		},
		"meta": {
			"saved_at_unix":   time.time(),
			"saved_at_iso":    time.strftime("%Y-%m-%dT%H:%M:%S%z"),
			"emergency_dump":  True,
			"levels":          args.levels,
			"tilt_deg":        args.tilt,
			"steps":           args.steps,
			"eval_episodes":   args.eval_episodes,
		},
	}
	p = Path(path)
	p.parent.mkdir(parents=True, exist_ok=True)
	with open(p, "wb") as f:
		pickle.dump(payload, f)
	print(f"\n[emergency-dump] Stage {payload['stage_num']} ({payload['stage_name']}) "
	      f"gen {payload['generation']}, {len(payload['population'])} genomes → {p}",
	      flush=True)


def _install_emergency_hook(strat) -> None:
	"""Monkey-patch the strategy's _on_generation_start to (a) record the
	current population in the module-level emergency state, and (b) check
	the Rust cancel flag and dump+abort if set."""
	original = strat._on_generation_start
	def wrapped(generation, **ctx):
		# Capture current population (start-of-gen snapshot; carries elites +
		# selected offspring ready for the next gen — ideal for resume).
		_emergency_state["population"]  = list(ctx.get("population", []))
		_emergency_state["best_genome"] = ctx.get("best_genome")
		_emergency_state["generation"]  = generation
		# Check cancel and bail if requested.
		try:
			import ram_accelerator
			if ram_accelerator.is_cancelled():
				_dump_emergency_state()
				raise StopIteration
		except StopIteration:
			raise
		except Exception:
			# Don't let the cancel-check infrastructure itself break the GA.
			pass
		return original(generation, **ctx)
	strat._on_generation_start = wrapped

from wnn.control.evaluator import (
	ControllerSpec, ControllerEvaluator, arch_shape_from_spec, spec_from_arch,
	fit_thresholds_from_pid_rollouts,
)
from wnn.control.arch_strategy import (
	ControllerArchGAStrategy, ControllerMemoryGAStrategy,
	default_controller_arch_config,
)
from wnn.control.ga_strategy import default_controller_ga_config
from wnn.control.ga_memory import record_address_universe
from wnn.control.recurrent_genome import RecurrentArchGenome, MemoryPayload
from wnn.control.training import EpisodeConfig, make_pid_action_fn
from wnn.control.dagger import eval_closed_loop_reset
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.reward_gated import RewardGatedConfig
from wnn.ram.strategies.optimization_dimension import OptimizationDimension
from wnn.seeds import resolve_seed_set, log_seed_set, record_seed_set


# -----------------------------------------------------------------------------
# Shape / spec plumbing
# -----------------------------------------------------------------------------

def _make_spec(state_neurons: int, levels: int, bits: int) -> ControllerSpec:
	"""Build a ControllerSpec from a (state_neurons, levels, bits) grid point.
	`bits` becomes BOTH state_bits_per_neuron and output_bits_per_neuron, matching
	the grid-search convention (the GA can later split them in the BITS phase)."""
	return ControllerSpec(
		num_motors=4, levels_per_motor=levels, bits_per_feature=8, input_window_k=4,
		state_neurons=state_neurons,
		state_bits_per_neuron=bits, output_bits_per_neuron=bits,
		delta_control=False,
	)


def _spec_from_best(best: RecurrentArchGenome, base: ControllerSpec) -> ControllerSpec:
	"""ControllerSpec carrying the previous stage's WINNING shape. The next stage
	resets its seed/arch dims from this so create_random_genome/MEMORY universe
	recording pin to the right reference."""
	return spec_from_arch(best, base)


def _filter_cells_for_arch(payload: MemoryPayload,
                           target: RecurrentArchGenome) -> MemoryPayload:
	"""Drop (neuron_idx, address) entries that fall outside `target`'s shape.
	Identical logic to ControllerMixedGAStrategy._filter_inherited_cells but
	usable without instantiating the mixed-GA strategy.

	Used by the grid search: the FIRST grid point records a fresh universe; each
	subsequent point reuses that universe filtered to its own neuron-count + bit
	bounds. (Cells inherited from a different shape are partially stale but the
	VALID subset gives the scorer well-formed input.)"""
	state_max_addr = 1 << target.state_bits_per_neuron
	output_max_addr = 1 << target.output_bits_per_neuron
	new_state_univ, new_state_vals = [], []
	for (n, a), v in zip(payload.state_universe, payload.state_values):
		if n < target.state_neurons and a < state_max_addr:
			new_state_univ.append((n, a))
			new_state_vals.append(v)
	new_output_univ, new_output_vals = [], []
	for (n, a), v in zip(payload.output_universe, payload.output_values):
		if n < target.output_neurons and a < output_max_addr:
			new_output_univ.append((n, a))
			new_output_vals.append(v)
	return MemoryPayload(new_state_univ, new_output_univ, new_state_vals, new_output_vals)


# -----------------------------------------------------------------------------
# Stage 0 — Grid search
# -----------------------------------------------------------------------------

def stage0_grid(args, ec: EpisodeConfig, seed: int):
	"""Grid over (state_neurons × bits). Returns the winning (spec, best_genome,
	best_metrics, wall_time, thresholds) for warm-starting Stage 1.

	**Validity filter**: (sn, b) is valid iff b > 2·sn (need ≥1 suffix bit after
	the forced full-state prefix). Invalid combos are silently filtered BEFORE
	enumeration — no spammy '[skip]' lines, and the displayed count reflects only
	valid points.

	**Scoring**: uses `ev.evaluate_batch` (which TRAINS via reward-gated
	adaptation) instead of `ev.score_genomes` (which scores untrained cells).
	Random cells produce identical scores across architectures (the prior bug —
	all grid points scored CE=387 because cells were never trained). With
	training, each grid point gets a meaningful per-architecture score, so the
	grid actually differentiates shape quality. Per-grid-point cost rises ~10×
	but for ~10-20 grid points the total is still minutes, and the warm-start
	to Stage 1 is genuinely informed.

	Cells: genomes are constructed with cells=None (no payload). evaluate_batch
	builds the controller, trains it via reward-gated adaptation, scores it.
	The trained cells are available on the controller for inspection but the
	returned Metrics is what we rank by.
	"""
	t0 = time.time()
	# Pre-filter valid grid points (skip silently — keep the visible count honest).
	#
	# Each neuron's bits split as: prefix (2·state_neurons for QSR state encoding,
	# the SAME bits for both layers since output also samples state)  + suffix
	# (sampled input bits). For the grid to be meaningful we need:
	#   bits > 2·state_neurons + min_suffix - 1   (i.e., suffix ≥ min_suffix)
	# A suffix of 1-3 bits is technically valid but provides too few input
	# samples per neuron to learn useful patterns — default min_suffix=4 ensures
	# the grid only enumerates architectures with MEANINGFUL input sampling.
	# Bumpable via --grid-min-suffix if you want to inspect smaller-suffix
	# configurations explicitly.
	min_suffix = args.grid_min_suffix
	all_pairs = [(sn, b) for sn in args.grid_state_neurons for b in args.grid_bits]
	valid_pairs = [(sn, b) for (sn, b) in all_pairs if (b - 2 * sn) >= min_suffix]
	n_skipped = len(all_pairs) - len(valid_pairs)
	print(f"\n{'='*72}\n  STAGE 0: GRID SEARCH "
	      f"({len(valid_pairs)} valid pts of {len(all_pairs)} requested, "
	      f"levels={args.levels}, min_suffix={min_suffix})\n{'='*72}")
	if n_skipped:
		print(f"  [grid] {n_skipped} pts skipped (bits − 2·state_neurons < {min_suffix}; "
		      f"need ≥{min_suffix} suffix bits for meaningful input sampling)")

	if not valid_pairs:
		raise RuntimeError(
			f"Grid search has zero valid points — every (sn, b) pair in the requested "
			f"grid produces fewer than {min_suffix} suffix bits (bits − 2·state_neurons "
			f"< {min_suffix}). Each neuron's bits split as 2·state_neurons (forced state "
			f"prefix) + suffix (sampled input bits). Either: (1) increase --grid-bits "
			f"(needs values ≥ 2·max(state_neurons) + {min_suffix}); (2) reduce "
			f"--grid-state-neurons (max should be ≤ (min(bits) − {min_suffix}) / 2); "
			f"or (3) lower --grid-min-suffix (currently {min_suffix}). "
			f"Requested sn={list(args.grid_state_neurons)}, "
			f"bits={list(args.grid_bits)}."
		)

	# Build a representative spec just to fit thresholds (any valid shape works —
	# thresholds come from PID rollouts which are arch-independent). Use the
	# smallest VALID grid point.
	probe_sn, probe_b = valid_pairs[0]
	probe_spec = _make_spec(probe_sn, args.levels, probe_b)
	thresholds = fit_thresholds_from_pid_rollouts(probe_spec, num_episodes=10, seed=seed)

	rng_master = np.random.default_rng(seed)
	results = []  # (spec, genome, metrics)
	for sn, b in valid_pairs:
		spec = _make_spec(sn, args.levels, b)
		shape = arch_shape_from_spec(spec)
		suffix = b - 2 * sn
		rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
		genome = RecurrentArchGenome.random(
			shape, state_neurons=sn,
			output_neurons=spec.num_motors * spec.levels_per_motor,
			state_suffix=suffix, output_suffix=suffix, rng=rng,
		)
		# No pre-attached cells — evaluate_batch will train them via
		# reward-gated adaptation, producing a genuine per-architecture score.
		genome.cells = None
		ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes,
		                         seed=seed, episode_config=ec, thresholds=thresholds,
		                         rg_config=_rg_config(args, ec, seed),
		                         max_train_workers=args.train_workers,
		                         num_eval_folds=args.num_eval_folds)
		m = ev.evaluate_batch([genome])[0]
		results.append((spec, genome, m))
		print(f"  [{len(results):>2}/{len(valid_pairs):>2}] "
		      f"sn={sn:>2} b={b:>2} suffix={suffix:>2}  "
		      f"CE={m.ce:>8.4f}  err={m.mean_attitude_error_deg:>6.2f}°  stable={m.acc*100:>5.1f}%")

	if not results:
		raise RuntimeError("Grid search produced no results (all valid points failed).")

	# Winner: lowest CE (= highest reward).
	winner_spec, winner_genome, winner_metrics = min(results, key=lambda r: r[2].ce)
	dt = time.time() - t0
	print(f"\n  GRID WINNER: sn={winner_spec.state_neurons} b={winner_spec.state_bits_per_neuron} "
	      f"levels={winner_spec.levels_per_motor}  CE={winner_metrics.ce:.4f}  "
	      f"err={winner_metrics.mean_attitude_error_deg:.2f}°  stable={winner_metrics.acc*100:.1f}%  "
	      f"({dt:.0f}s)")
	return winner_spec, winner_genome, winner_metrics, dt, thresholds


# -----------------------------------------------------------------------------
# Stages 1-4 — single-dimension GA phases (warm-started)
# -----------------------------------------------------------------------------

def _build_ga_config(args, gens: int, patience: int):
	"""GAConfig per stage. The controller defaults (reward ranking, no acc floor)
	+ our per-stage overrides (pop/gens/patience/elitism/crossover)."""
	gacfg = default_controller_ga_config(
		population_size=args.pop, generations=gens,
		weight_err_sq=args.fit_weight_err_sq,
		weight_stable=args.fit_weight_stable,
		weight_jerk=args.fit_weight_jerk,
		weight_mono=args.fit_weight_mono,
	)
	gacfg.patience = patience
	gacfg.elitism_pct = args.elitism
	gacfg.crossover_rate = args.crossover_rate
	gacfg.check_interval = args.check_interval
	return gacfg


def _stage_header(idx: int, name: str, gens: int, patience: int, spec: ControllerSpec):
	bar = "=" * 72
	print(f"\n{bar}\n  STAGE {idx}: {name} ({gens} gens, patience {patience})\n{bar}")
	print(f"  seed-spec: state_neurons={spec.state_neurons}, "
	      f"output_neurons={spec.num_motors * spec.levels_per_motor}, "
	      f"state_bits={spec.state_bits_per_neuron}, "
	      f"output_bits={spec.output_bits_per_neuron}, "
	      f"levels={spec.levels_per_motor}")


def _print_stage_result(idx: int, name: str, res, gens: int, dt: float, ev: ControllerEvaluator):
	"""Re-evaluate the winning genome so we can report the full metric tuple
	(CE, err, stable_rate) using the evaluator that drove the stage."""
	best = res.best_genome
	if best is None:
		print(f"  STAGE {idx} ({name}): NO BEST GENOME (iter={res.iterations_run})")
		return None
	# Pick the right scorer: MEMORY-stage genomes carry cells → score_genomes (no
	# training). Architecture-stage genomes carry no cells → evaluate_batch trains.
	if getattr(best, "cells", None) is not None:
		m = ev.score_genomes([best])[0]
	else:
		m = ev.evaluate_batch([best])[0]
	sn, on = best.state_neurons, best.output_neurons
	sb, ob = best.state_bits_per_neuron, best.output_bits_per_neuron
	print(f"  STAGE {idx} ({name}) done: gen {res.iterations_run}/{gens}  "
	      f"CE={m.ce:.4f}  err={m.mean_attitude_error_deg:.2f}°  stable={m.acc*100:.1f}%  "
	      f"arch sn={sn} on={on} sb={sb} ob={ob}  ({dt:.0f}s, "
	      f"{dt/max(res.iterations_run,1):.1f}s/gen)")
	return m


def _rg_config(args, ec: EpisodeConfig, seed: int) -> RewardGatedConfig:
	"""Reward-gated inner-train config — exposed knobs let the smoke test shrink
	the per-genome training cost (default: full 8 rounds × 24 episodes_per_round).
	None for any flag → upstream default."""
	rg = RewardGatedConfig(seed=seed, episode_config=ec)
	if args.rg_rounds is not None:
		rg.num_rounds = args.rg_rounds
	if args.rg_episodes_per_round is not None:
		rg.episodes_per_round = args.rg_episodes_per_round
	if args.rg_eval_episodes is not None:
		rg.eval_episodes = args.rg_eval_episodes
	rg.steps_per_episode = args.steps   # match the outer eval steps for consistency
	rg.progress = False                  # quiet the per-round inner logging
	return rg


def _run_arch_phase(args, ec: EpisodeConfig, spec: ControllerSpec,
                    dimension: OptimizationDimension, gens: int, patience: int,
                    seed: int, warm_start_genome=None, initial_population=None,
                    tracker=None, experiment_id=None):
	"""Generic Stage 1-3 driver: build an ArchGAStrategy on the given dimension
	and run optimize(). Returns (result, evaluator, wall_time).

	29/05/2026 — warm_start_genome: when provided, seeded as initial_population[0].
	Without this, the GA randomizes the optimized dimension while only PINNING
	the prior winner's spec — so Stage 1's specific bits/conns aren't in
	Stage 2's initial pop, causing a cold-start regression
	(e.g. v7 Stage 1 ended at err=6.93° but Stage 2 Gen 1 was err=9.41°
	until the GA re-evolved good bits). With warm-start, the prior winner is
	always in the elite list and the stage's best can never go below the
	previous stage's best."""
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes,
	                         seed=seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=args.num_eval_folds)
	arch_cfg = default_controller_arch_config(spec)
	# Widen the search box to admit the grid winner + room to mutate. The default
	# max_state_neurons is 4·spec.state_neurons; honor the user's grid maximum so
	# the GA can climb past the seed if it likes.
	arch_cfg.max_state_neurons = max(arch_cfg.max_state_neurons,
	                                 4 * max(args.grid_state_neurons))
	# Hard floor on state_neurons from the grid (added 30/05/2026 for Plan A v2).
	# Without this, GA mutations can take sn below the grid minimum, undoing the
	# anchor we set when --grid-state-neurons specifies a tight range.
	arch_cfg.min_state_neurons = max(arch_cfg.min_state_neurons,
	                                 min(args.grid_state_neurons))
	gacfg = _build_ga_config(args, gens, patience)
	strat = ControllerArchGAStrategy(spec, dimension, arch_config=arch_cfg,
	                                 ga_config=gacfg, seed=seed, batch_evaluator=ev,
	                                 lamarckian=getattr(args, "lamarckian", False))
	# Dashboard wiring (no-op for the standalone CLI): attach the tracker so the
	# GenericGAStrategy loop auto-fires record_iteration / record_genome_evaluations_batch
	# per generation — including mean_attitude_error_deg — under this stage's experiment.
	if tracker is not None and experiment_id is not None:
		strat.set_tracker(tracker, experiment_id)
	# Install the emergency-dump hook BEFORE optimize() runs so any SIGTERM
	# during this stage trips the cooperative-cancel path.
	_install_emergency_hook(strat)
	t = time.time()
	# Lamarckian: route batch eval through write-back (carry cells across gens).
	_batch_fn = strat._lamarckian_evaluate_batch if getattr(args, "lamarckian", False) else ev.evaluate_batch
	optimize_kwargs = {
		"evaluate_fn": lambda g: ev.evaluate_batch([g])[0].ce,
		"batch_evaluate_fn": _batch_fn,
	}
	# Resume support (added 31/05/2026): if a full population was passed in
	# (from an emergency-dump pickle), use it directly. Otherwise fall back to
	# the single warm-start genome — the two paths are mutually exclusive.
	if initial_population is not None:
		# Make sure the warm-start genome is at the front of the population so
		# it ends up in the elite slate of gen 0.
		pop = list(initial_population)
		if warm_start_genome is not None:
			pop = [warm_start_genome] + pop
		optimize_kwargs["initial_population"] = pop
	elif warm_start_genome is not None:
		optimize_kwargs["initial_population"] = [warm_start_genome]
	res = strat.optimize(**optimize_kwargs)
	return res, ev, time.time() - t


def _run_memory_phase(args, ec: EpisodeConfig, spec: ControllerSpec,
                      gens: int, patience: int, seed: int,
                      initial_population=None,
                      tracker=None, experiment_id=None):
	"""Stage 4: arch FROZEN at `spec`; evolve QSR cell VALUES over a recorded
	universe. The strategy auto-records the universe on its own seed arch via
	_ensure_universe (called inside _make_cell_genome).

	initial_population (added 30/05/2026 for Plan B memory-only refinement):
	list of seed genomes injected at the start of the GA. Use with a saved
	Plan A run's final_population to refine the entire evolved pool under a
	new fitness weight schema — strictly stronger than seeding with just the
	single winner because 200 evolved genomes carry more diversity than 1
	winner + 199 random ones."""
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes,
	                         seed=seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=args.num_eval_folds)
	arch_cfg = default_controller_arch_config(spec)
	arch_cfg.max_state_neurons = max(arch_cfg.max_state_neurons,
	                                 4 * max(args.grid_state_neurons))
	arch_cfg.min_state_neurons = max(arch_cfg.min_state_neurons,
	                                 min(args.grid_state_neurons))
	gacfg = _build_ga_config(args, gens, patience)
	strat = ControllerMemoryGAStrategy(
		spec, arch_config=arch_cfg, ga_config=gacfg,
		seed=seed, batch_evaluator=ev, thresholds=thresholds,
		record_episodes=args.universe_episodes, record_steps=args.steps,
	)
	# Dashboard wiring (no-op for the standalone CLI): per-gen iteration recording
	# under the MEMORY stage's experiment (see _run_arch_phase note).
	if tracker is not None and experiment_id is not None:
		strat.set_tracker(tracker, experiment_id)
	# Emergency-dump hook for cooperative cancellation (mirrors arch_phase).
	_install_emergency_hook(strat)
	t = time.time()
	# MEMORY paradigm: cells ARE the genome → score_genomes (no training).
	optimize_kwargs = dict(
		evaluate_fn=lambda g: ev.score_genomes([g])[0].ce,
		batch_evaluate_fn=ev.score_genomes,
	)
	if initial_population is not None:
		optimize_kwargs["initial_population"] = list(initial_population)
	res = strat.optimize(**optimize_kwargs)
	return res, ev, time.time() - t


def _save_winner(path: str, args, spec: ControllerSpec,
                 best_genome, final_population, metrics) -> None:
	"""Pickle the final-stage WINNER + the entire FINAL POPULATION + spec +
	provenance to PATH.

	Used by Plan A → Plan B chaining: Plan B (run_memory_refinement.py) loads
	the full population as `initial_population=` for the memory-only refinement
	GA. Strictly stronger than seeding with just the winner because the 200
	evolved genomes carry the search's accumulated diversity — Plan B's GA
	starts at the END of Plan A's exploration instead of one snapshot of it.

	The pickle also keeps `best_genome` as a convenience (Plan B falls back to
	it if `population` is empty, e.g. legacy single-genome saves)."""
	payload = {
		"spec":         spec,
		"best_genome":  best_genome,
		"population":   list(final_population) if final_population is not None else [],
		"metrics":      metrics,
		"fitness_weights": {
			"err_sq": args.fit_weight_err_sq,
			"stable": args.fit_weight_stable,
			"jerk":   args.fit_weight_jerk,
			"mono":   args.fit_weight_mono,
		},
		"meta": {
			"saved_at_unix": time.time(),
			"saved_at_iso":  time.strftime("%Y-%m-%dT%H:%M:%S%z"),
			"levels":        args.levels,
			"tilt_deg":      args.tilt,
			"steps":         args.steps,
			"eval_episodes": args.eval_episodes,
		},
	}
	p = Path(path)
	p.parent.mkdir(parents=True, exist_ok=True)
	with open(p, "wb") as f:
		pickle.dump(payload, f)
	pop_n = len(payload["population"])
	print(f"\n[save-winner] wrote {p}  (spec sn={spec.state_neurons} "
	      f"sb={spec.state_bits_per_neuron} ob={spec.output_bits_per_neuron}, "
	      f"population={pop_n} genomes)")


# -----------------------------------------------------------------------------
# Baselines (PID + reference numbers from prior runs)
# -----------------------------------------------------------------------------

def _pid_baseline(ec: EpisodeConfig, episodes: int, seed: int):
	"""PID score on the held-out episode set, for the final-summary 'vs PID' row."""
	pid = AttitudePID(AttitudePIDConfig())
	from wnn.control.training import make_pid_action_fn
	_, m = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, ec, episodes, seed)
	return m


def _maybe_holdout(args, ec, spec, res, seeds, label: str):
	"""Per-stage held-out (REPORT ONLY) — fires after each stage if --report-seed is set,
	so we see the held-out N→B→C→M trajectory, not just the final. Never feeds selection.

	Returns the held-out winner metric (with .mean_attitude_error_deg / .acc / .fitness)
	so a caller (the dashboard flow_runner) can record it on the stage's experiment;
	returns None when no report-seed is set or the stage was skipped/failed."""
	if getattr(args, "report_seed", None) is None or res is None or res.best_genome is None:
		return None
	try:
		return _holdout_report(args, ec, spec, res.best_genome, res.final_population,
		                       args.report_seed, seeds.train, stage_label=label)
	except Exception as e:
		print(f"  [report-seed] {label} held-out failed: {e}")
		return None


def _holdout_report(args, ec: EpisodeConfig, spec, best_genome, final_population,
                    report_seed: int, train_seed: int, stage_label: str = "final"):
	"""TRUE held-out — REPORT ONLY. Re-eval the whole final population on a FRESH
	report_seed for DESCRIPTIVE statistics. The held-out is NEVER used to select a
	genome or feed any phase — selecting on it would leak val into the population
	(overfitting to the held-out draw). The reported RESULT is the during-search
	winner (selected on train/val) measured ONCE here; pop mean±std is context only.

	The per-stage/per-gen metric is K-fold on the TRAIN seed (the GA's OPTIMISTIC
	selection metric — doesn't reproduce, see project_controller_eval_variance), so
	this fresh-seed measurement of the chosen winner is the honest paper number."""
	import statistics
	if report_seed == train_seed:
		print(f"  [report-seed] WARNING: report_seed == train_seed ({train_seed}) — NOT a held-out.")
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=report_seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes,
	                         seed=report_seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, report_seed),
	                         max_train_workers=args.train_workers)
	pop = list(final_population) if final_population else [best_genome]
	# MEMORY-stage winners carry cells → score (no retrain); arch winners → train+eval.
	use_score = getattr(best_genome, "cells", None) is not None
	metrics = ev.score_genomes(pop) if use_score else ev.evaluate_batch(pop)
	stables = [m.acc * 100 for m in metrics]
	errs = [m.mean_attitude_error_deg for m in metrics]
	ds = metrics[0]            # final_population[0] = the during-search winner = THE RESULT
	pop_max = max(stables)     # descriptive only — NOT selected (would leak)
	pid_m = _pid_baseline(ec, args.eval_episodes, report_seed)
	def _ms(xs):
		return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)
	ms_s, ms_e = _ms(stables), _ms(errs)
	bar = "=" * 72
	print(f"\n{bar}\n  HELD-OUT REPORT [{stage_label}] (report-only) — population ({len(pop)} genomes) on "
	      f"FRESH seed {report_seed}, train/select seed {train_seed}\n{bar}")
	print(f"  RESULT — during-search winner (held-out):  stable={ds.acc*100:.1f}%  "
	      f"err={ds.mean_attitude_error_deg:.2f}°  reward={ds.fitness:.2f}")
	print(f"  population (held-out, descriptive):        stable={ms_s[0]:.1f}±{ms_s[1]:.1f}%  "
	      f"err={ms_e[0]:.2f}±{ms_e[1]:.2f}°   (pop max stable={pop_max:.1f}% — NOT selected, would leak)")
	print(f"  vs PID  (held-out):                        stable={pid_m['stable_rate']*100:.1f}%  "
	      f"err={pid_m['mean_attitude_error_deg']:.2f}°")
	print(bar)
	return ds


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _save_stage_checkpoint(args, stage_num: int, stage_name: str,
                            spec: ControllerSpec, res, metrics) -> None:
	"""Per-stage checkpoint: pickle the stage's winner + final population if
	--save-stage-checkpoints DIR is set. Lets a future reboot resume from the
	last finished stage instead of restarting from scratch (added 30/05/2026
	after Plan A v1 lost ~5.5h in an OOM-triggered reboot mid-Stage-3).

	No-op when --save-stage-checkpoints unset, so existing runs are unaffected."""
	if not getattr(args, "save_stage_checkpoints", None):
		return
	if res is None or res.best_genome is None:
		return
	out_dir = Path(args.save_stage_checkpoints)
	out_dir.mkdir(parents=True, exist_ok=True)
	path = out_dir / f"stage{stage_num}_{stage_name.lower()}.pkl"
	# Re-use _save_winner so the schema matches the final-winner pickle.
	# Plan B can load any stage checkpoint just like a final winner.
	_save_winner(str(path), args, spec, res.best_genome, res.final_population, metrics)
	# Annotate the stage identity so --resume-from-emergency jumps to the RIGHT
	# next stage. The resume logic reads `stage_num`; _save_winner's schema omits
	# it, which would otherwise default to stage 1 and re-run finished stages.
	try:
		with open(path, "rb") as f:
			payload = pickle.load(f)
		payload["stage_num"] = stage_num
		payload["stage_name"] = stage_name
		with open(path, "wb") as f:
			pickle.dump(payload, f)
	except Exception as e:
		print(f"  [stage-checkpoint] could not annotate stage_num on {path}: {e}")


def _run_one(args, ec: EpisodeConfig, seeds, resume_state: dict | None = None,
             tracker=None, stage_experiment_ids=None, stage_holdouts=None):
	"""One full phased run for a SeedSet. Returns the per-stage metrics + final
	best genome for the multi-run aggregator (when --runs>1).

	resume_state (added 31/05/2026): when set, skip earlier stages and start
	from the resume point. Schema mirrors the emergency-dump pickle plus a
	`resume_mode` field:
	  * "same" — continue the dumped stage from its population (most common)
	  * "next" — skip the dumped stage; warm-start the next stage from the
	            dumped best_genome
	Skipped stages get None placeholders in the result list.

	Dashboard hooks (all optional — the standalone CLI passes none, so behaviour
	is unchanged):
	  * tracker — an ExperimentTracker; attached to each GA stage's strategy so
	    the loop records per-generation iterations (incl. mean_attitude_error_deg).
	  * stage_experiment_ids — indexed [0=grid,1=neurons,2=bits,3=connections,
	    4=memory]; the per-stage experiment row the tracker writes under.
	  * stage_holdouts — an out-dict the caller pre-allocates; populated with the
	    per-stage held-out winner metric keyed by stage label (NEURONS/BITS/...).
	"""
	# Use the train seed for everything inside the run; the test/val seeds are
	# only for the final report (PID baseline + held-out reference).
	seed = seeds.train

	def _eid(stage_idx: int):
		"""Experiment id for a stage (or None when not running under a tracker)."""
		if stage_experiment_ids is None:
			return None
		try:
			return stage_experiment_ids[stage_idx]
		except (IndexError, KeyError, TypeError):
			return None

	def _record_ho(label: str, ho):
		"""Stash a stage's held-out metric into the caller's out-dict (if any)."""
		if stage_holdouts is not None and ho is not None:
			stage_holdouts[label] = ho

	# Arch stages to skip (--skip-stages bits,connections). A skipped stage leaves
	# the carried population + prev_best untouched, so they flow to the next stage.
	skip_stages = {s.strip().lower() for s in (getattr(args, "skip_stages", "") or "").split(",") if s.strip()}

	# Resume planning.
	resume_start_stage = 1
	resume_population  = None
	resume_spec        = None
	resume_warm_genome = None
	if resume_state is not None:
		dumped_stage = int(resume_state.get("stage_num") or 1)
		mode = (resume_state.get("resume_mode") or "same").lower()
		if mode == "same":
			resume_start_stage = dumped_stage
			resume_population  = resume_state.get("population") or None
			resume_spec        = resume_state.get("spec")
			resume_warm_genome = resume_state.get("best_genome")
		elif mode == "next":
			resume_start_stage = min(dumped_stage + 1, 4)
			# Carry the FULL dumped population forward: one phase FEEDS the next —
			# the next stage continues evolving the previous stage's whole population,
			# NOT a rebuild from the single winner. best_genome/spec set the seed spec.
			resume_population  = resume_state.get("population") or None
			resume_warm_genome = resume_state.get("best_genome")
			resume_spec        = resume_state.get("spec")  # falls back to dumped spec
		else:
			raise ValueError(f"unknown resume_mode {mode!r}; expected 'same' or 'next'")
		print(f"\n[resume] dumped stage={dumped_stage} mode={mode!r} → starting at "
		      f"stage {resume_start_stage} (pop={len(resume_state.get('population') or [])} genomes)")

	# Stage 0 — grid. Skipped on resume (only its winner_spec matters and the
	# resume's captured `spec` carries that forward).
	if resume_state is None:
		winner_spec, _winner_genome, m0, dt0, _thr = stage0_grid(args, ec, seed)
		stage_results = [("Grid", winner_spec, m0, dt0,
		                  len(args.grid_state_neurons) * len(args.grid_bits))]
	else:
		winner_spec = resume_spec
		if winner_spec is None:
			raise ValueError("resume_state missing both spec and best_genome — cannot determine winner_spec")
		stage_results = [("Grid (skipped on resume)", winner_spec, None, 0.0, 0)]

	# Compute the emergency-dump path for this run. Lives next to the per-stage
	# checkpoint dir if set, else falls back to /tmp.
	_emergency_dir = (Path(args.save_stage_checkpoints) if args.save_stage_checkpoints
	                  else Path("/tmp/wnn-phased-ga-emergency"))
	def _stage_emergency_path(stage_num: int, stage_name: str) -> str:
		return str(_emergency_dir / f"emergency_stage{stage_num}_{stage_name.lower()}.pkl")

	# ---- Stage 1 — NEURONS -------------------------------------------------
	spec1 = winner_spec
	if resume_start_stage > 1:
		print(f"[resume] skipping Stage 1 (Neurons)")
		res1, ev1, dt1, m1 = None, None, 0.0, None
	else:
		_stage_header(1, "NEURONS", args.neurons_gens, args.neurons_patience, spec1)
		_set_current_stage(1, "neurons", spec1, args, _stage_emergency_path(1, "neurons"))
		init_pop1 = resume_population if (resume_state and resume_start_stage == 1) else None
		warm1     = resume_warm_genome if (resume_state and resume_start_stage == 1) else None
		res1, ev1, dt1 = _run_arch_phase(args, ec, spec1, OptimizationDimension.NEURONS,
		                                 args.neurons_gens, args.neurons_patience, seed,
		                                 warm_start_genome=warm1, initial_population=init_pop1,
		                                 tracker=tracker, experiment_id=_eid(1))
		m1 = _print_stage_result(1, "NEURONS", res1, args.neurons_gens, dt1, ev1)
		_save_stage_checkpoint(args, 1, "neurons", spec1, res1, m1)
		_record_ho("NEURONS", _maybe_holdout(args, ec, spec1, res1, seeds, "NEURONS"))

	# Track the most-recent best_genome through the chain — skipped stages
	# pass it forward without modification so the next non-skipped stage can
	# warm-start from it.
	base = winner_spec
	prev_best = res1.best_genome if res1 is not None else resume_warm_genome
	# Population carried into the next stage. Updated only by a stage that RUNS;
	# a skipped stage leaves it unchanged so the population passes straight
	# through (e.g. --skip-stages bits,connections carries Neurons → Memory).
	# Subsumes the old per-stage res-or-resume_population fallback.
	carried_pop = (res1.final_population if (res1 is not None and getattr(res1, "final_population", None))
	               else resume_population)

	# ---- Stage 2 — BITS ----------------------------------------------------
	if res1 is not None:
		spec2 = _spec_from_best(res1.best_genome, base) if res1.best_genome is not None else spec1
	else:
		# Stage 1 was skipped on resume — derive Stage 2's spec from the
		# carried-forward best (or fall back to spec1).
		spec2 = _spec_from_best(prev_best, base) if prev_best is not None else spec1
	if resume_start_stage > 2 or "bits" in skip_stages:
		reason = "resume" if resume_start_stage > 2 else "skip-stages"
		print(f"[{reason}] skipping Stage 2 (Bits) — carrying population through")
		res2, ev2, dt2, m2 = None, None, 0.0, None
	else:
		_stage_header(2, "BITS", args.bits_gens, args.bits_patience, spec2)
		_set_current_stage(2, "bits", spec2, args, _stage_emergency_path(2, "bits"))
		# CARRY the FULL carried population into Stage 2 (one phase feeds the next;
		# do NOT rebuild from the winner).
		init_pop2 = carried_pop
		warm2     = prev_best
		res2, ev2, dt2 = _run_arch_phase(args, ec, spec2, OptimizationDimension.BITS,
		                                 args.bits_gens, args.bits_patience, seed,
		                                 warm_start_genome=warm2, initial_population=init_pop2,
		                                 tracker=tracker, experiment_id=_eid(2))
		m2 = _print_stage_result(2, "BITS", res2, args.bits_gens, dt2, ev2)
		_save_stage_checkpoint(args, 2, "bits", spec2, res2, m2)
		_record_ho("BITS", _maybe_holdout(args, ec, spec2, res2, seeds, "BITS"))
		prev_best = res2.best_genome if res2.best_genome is not None else prev_best
		if getattr(res2, "final_population", None):
			carried_pop = res2.final_population

	# ---- Stage 3 — CONNECTIONS --------------------------------------------
	if res2 is not None:
		spec3 = _spec_from_best(res2.best_genome, base) if res2.best_genome is not None else spec2
	else:
		spec3 = _spec_from_best(prev_best, base) if prev_best is not None else spec2
	if resume_start_stage > 3 or "connections" in skip_stages:
		reason = "resume" if resume_start_stage > 3 else "skip-stages"
		print(f"[{reason}] skipping Stage 3 (Connections) — carrying population through")
		res3, ev3, dt3, m3 = None, None, 0.0, None
	else:
		_stage_header(3, "CONNECTIONS", args.conns_gens, args.conns_patience, spec3)
		_set_current_stage(3, "connections", spec3, args, _stage_emergency_path(3, "connections"))
		# CARRY the FULL carried population into Stage 3.
		init_pop3 = carried_pop
		warm3     = prev_best
		res3, ev3, dt3 = _run_arch_phase(args, ec, spec3, OptimizationDimension.CONNECTIONS,
		                                 args.conns_gens, args.conns_patience, seed,
		                                 warm_start_genome=warm3, initial_population=init_pop3,
		                                 tracker=tracker, experiment_id=_eid(3))
		m3 = _print_stage_result(3, "CONNECTIONS", res3, args.conns_gens, dt3, ev3)
		_save_stage_checkpoint(args, 3, "connections", spec3, res3, m3)
		_record_ho("CONNECTIONS", _maybe_holdout(args, ec, spec3, res3, seeds, "CONNECTIONS"))
		prev_best = res3.best_genome if res3.best_genome is not None else prev_best
		if getattr(res3, "final_population", None):
			carried_pop = res3.final_population

	# ---- Stage 4 — MEMORY (arch FROZEN) -----------------------------------
	if res3 is not None:
		spec4 = _spec_from_best(res3.best_genome, base) if res3.best_genome is not None else spec3
	else:
		spec4 = _spec_from_best(prev_best, base) if prev_best is not None else spec3
	_stage_header(4, "MEMORY", args.memory_gens, args.memory_patience, spec4)
	_set_current_stage(4, "memory", spec4, args, _stage_emergency_path(4, "memory"))
	# CARRY the FULL carried population into Stage 4 (MEMORY). With
	# --skip-stages bits,connections this is the NEURONS final population.
	init_pop4 = carried_pop
	res4, ev4, dt4 = _run_memory_phase(args, ec, spec4, args.memory_gens, args.memory_patience,
	                                   seed, initial_population=init_pop4,
	                                   tracker=tracker, experiment_id=_eid(4))
	m4 = _print_stage_result(4, "MEMORY", res4, args.memory_gens, dt4, ev4)
	_save_stage_checkpoint(args, 4, "memory", spec4, res4, m4)
	_record_ho("MEMORY", _maybe_holdout(args, ec, spec4, res4, seeds, "MEMORY"))

	# PID baseline on the val seed (the held-out reference).
	pid_m = _pid_baseline(ec, args.eval_episodes, seeds.val)

	# Aggregate per-stage tuples. Skipped stages report iters=0 and metrics=None
	# so the final-summary block degrades gracefully.
	def _iters(r):
		return r.iterations_run if r is not None else 0
	stage_results += [
		("Neurons",     spec1, m1, dt1, _iters(res1)),
		("Bits",        spec2, m2, dt2, _iters(res2)),
		("Connections", spec3, m3, dt3, _iters(res3)),
		("Memory",      spec4, m4, dt4, _iters(res4)),
	]
	# final_population: memory-stage population, sorted by fitness. Used by
	# --save-winner so Plan B can warm-start its GA from Plan A's evolved pool.
	return stage_results, res4.best_genome, res4.final_population, pid_m


def _print_final_summary(args, stage_results, best_final, pid_m, total_dt: float):
	"""Final report block: per-stage outcomes + baselines + reference numbers."""
	bar = "=" * 72
	print(f"\n{bar}\n  PHASED-GA RESULT (5 stages, target "
	      f"{args.neurons_gens+args.bits_gens+args.conns_gens+args.memory_gens} GA gens "
	      f"+ {len(args.grid_state_neurons)*len(args.grid_bits)} grid)\n{bar}")
	# Stage rows.
	labels = ["Grid", "Neurons", "Bits", "Conns", "Memory"]
	target_gens = [None, args.neurons_gens, args.bits_gens, args.conns_gens, args.memory_gens]
	for (label, spec, m, dt, iters), target in zip(stage_results, target_gens):
		sn = spec.state_neurons
		b_s = spec.state_bits_per_neuron
		b_o = spec.output_bits_per_neuron
		if label == "Grid":
			print(f"  Stage 0 (Grid):    winner sn={sn} b={b_s} levels={spec.levels_per_motor}  "
			      f"CE={m.ce:.4f}  err={m.mean_attitude_error_deg:.2f}°  stable={m.acc*100:.1f}%  ({dt:.0f}s)")
		else:
			ce = "n/a" if m is None else f"{m.ce:.4f}"
			err = "n/a" if m is None else f"{m.mean_attitude_error_deg:.2f}°"
			stab = "n/a" if m is None else f"{m.acc*100:.1f}%"
			gens_str = f"{iters}/{target}" if target else f"{iters}"
			print(f"  Stage   ({label:<11}): gen {gens_str:<10}  "
			      f"CE={ce:<8}  err={err:<8} stable={stab:<6} "
			      f"arch sn={sn} sb={b_s} ob={b_o} on={best_final.output_neurons if best_final else '?'}  ({dt:.0f}s)")

	# Final winner (= memory stage).
	final_label, final_spec, final_m, _, _ = stage_results[-1]
	print("  " + "─" * 60)
	if final_m is not None:
		print(f"  FINAL: err={final_m.mean_attitude_error_deg:.2f}°  "
		      f"stable={final_m.acc*100:.0f}%  reward={final_m.fitness:.2f}")
	# Baselines.
	print(f"  vs PID:  {pid_m['mean_attitude_error_deg']:.2f}° / "
	      f"{pid_m['stable_rate']*100:.0f}% / {pid_m['mean_reward']:.2f}")
	print(f"  vs MLP:  9.66° / 26.7% / -59.17  (run_mlp_ga.py 3-way held-out baseline)")
	print(f"  vs prior ga_memory: 7.14° / 30% / -32.19  (frozen-arch baseline)")
	print(f"  vs C-mix-3:        13.69°  (mixed-GA partial result, killed at gen 599)")
	print(f"\n  Total wall time: {total_dt/60:.1f} min ({total_dt:.0f}s)")


def build_arg_parser() -> argparse.ArgumentParser:
	"""Build the phased-GA CLI parser. Factored out of main() so the dashboard
	flow_runner can obtain the full set of defaults via parse_args([]) and then
	override per-flow — keeping the CLI defaults the single source of truth."""
	ap = argparse.ArgumentParser()
	# Grid (Stage 0).
	ap.add_argument("--grid-state-neurons", type=int, nargs="+",
	                default=[8, 12, 16, 20, 24],
	                help="state_neurons axis for Stage 0 grid")
	ap.add_argument("--grid-bits", type=int, nargs="+",
	                default=[18, 24, 30, 36],
	                help="bits-per-neuron axis for Stage 0 grid")
	ap.add_argument("--grid-min-suffix", type=int, default=4,
	                help="minimum sampled-input-bit suffix per neuron — pre-filter "
	                     "grid (sn, b) pairs to skip configurations where "
	                     "(b − 2·sn) < this value. Default 4 ensures each neuron has "
	                     "at least 4 input bits to sample patterns from, not just the "
	                     "forced QSR state prefix. Use 1 to allow any positive suffix.")
	ap.add_argument("--levels", type=int, default=16, help="PWM levels per motor (fixed dim)")
	# Stages 1-4.
	ap.add_argument("--neurons-gens", type=int, default=400)
	ap.add_argument("--neurons-patience", type=int, default=20)
	ap.add_argument("--bits-gens", type=int, default=400)
	ap.add_argument("--bits-patience", type=int, default=20)
	ap.add_argument("--conns-gens", type=int, default=400)
	ap.add_argument("--conns-patience", type=int, default=20)
	ap.add_argument("--memory-gens", type=int, default=800)
	ap.add_argument("--memory-patience", type=int, default=40)
	# Skip arch stages by name (e.g. "bits,connections"). A skipped stage passes
	# its incoming population + best straight through to the next stage — so
	# `--skip-stages bits,connections` runs grid → NEURONS → MEMORY, carrying the
	# Neurons population into Memory. Motivated by the 07/06 finding that under
	# --lamarckian the NEURONS stage already optimizes neurons+connections+memory
	# jointly (grid covers bits), making BITS/CONNECTIONS ~28h of dead weight.
	ap.add_argument("--skip-stages", type=str, default="",
	                help="Comma-separated arch stages to SKIP: any of bits,connections "
	                     "(neurons/memory/grid always run; grid skips only via --resume). "
	                     "e.g. --skip-stages bits,connections → grid→neurons→memory.")
	# Shared GA hyperparams.
	ap.add_argument("--pop", type=int, default=200, help="per-stage population")
	# elitism = fraction kept as elites (0.2 = 20%); formula int(pop*elitism), no hidden ×2.
	ap.add_argument("--elitism", type=float, default=0.2)
	# Lamarckian: arch phases carry learned cells across generations (warm-start +
	# write-back) instead of re-training from scratch — preserves the WNN's memory
	# through N/B/C mutations. 1-seed eval (write-back needs one canonical state).
	ap.add_argument("--lamarckian", action="store_true",
	                help="Carry learned cells across arch-phase generations (memory preservation).")
	ap.add_argument("--crossover-rate", type=float, default=0.5)
	# Evaluation / episode.
	ap.add_argument("--eval-episodes", type=int, default=20)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	# Initial-condition severity (match a curriculum stage, e.g. Stage A = 5/0.5/0.3).
	ap.add_argument("--body-rate", type=float, default=0.5, help="max initial body rate (rad/s)")
	ap.add_argument("--yaw-rate", type=float, default=0.3, help="max initial yaw rate (rad/s)")
	# Early-stop cadence: patience checks every check_interval gens (per-phase patience
	# set via --*-patience). Faster pace = smaller check_interval + smaller patience.
	ap.add_argument("--check-interval", type=int, default=10, help="gens between patience checks")
	ap.add_argument("--universe-episodes", type=int, default=8)
	# Inner reward-gated train knobs (production: leave None → 8 rounds × 24 eps);
	# smoke tests pass tiny values to keep per-genome training under a few seconds.
	ap.add_argument("--rg-rounds", type=int, default=None,
	                help="Reward-gated inner-train rounds (default: 8)")
	ap.add_argument("--rg-episodes-per-round", type=int, default=None,
	                help="Episodes per reward-gated round (default: 24)")
	ap.add_argument("--rg-eval-episodes", type=int, default=None,
	                help="Eval episodes within reward-gated (default: 20)")
	# Multi-objective fitness weights (29/05/2026). Defaults (err_sq=1.0, others=0)
	# reproduce single-objective behavior on integrated err². Setting any of
	# stable/jerk/mono > 0 switches to the harmonic-rank calculator (rank-based
	# weighted harmonic mean — mirrors IDS HARMONIC_RANK). Use --fit-weight-stable
	# to add stability ranking; --fit-weight-jerk and --fit-weight-mono are
	# RESERVED (metrics not yet populated by the eval path — calculator will
	# warn-once and ignore if used).
	ap.add_argument("--fit-weight-err-sq", type=float, default=1.0,
	                help="Weight on integrated err² in the harmonic-rank fitness (default 1.0).")
	ap.add_argument("--fit-weight-stable", type=float, default=0.0,
	                help="Weight on stable_rate (acc) in the harmonic-rank fitness (default 0). >0 activates multi-objective.")
	ap.add_argument("--fit-weight-jerk", type=float, default=0.0,
	                help="Weight on motor_jerk_mean. RESERVED (Metrics field not yet populated).")
	ap.add_argument("--fit-weight-mono", type=float, default=0.0,
	                help="Weight on mono_violations_total. RESERVED (Metrics field not yet populated).")
	# Parallelism — the ControllerEvaluator's per-genome ThreadPool. Defaults to
	# 1 inside ControllerEvaluator (no concurrency), which leaves 15/16 cores
	# idle when the GA evaluates 200+ genome populations. 4-8 is the sweet spot
	# on the M4 Max (16 cores, leaves headroom for Rayon-inside-step + the IDS
	# worker on RAYON_NUM_THREADS=3). Found 29/05/2026 during c-mix-4 RCA.
	ap.add_argument("--train-workers", type=int, default=4,
	                help="ControllerEvaluator.max_train_workers; 4 = sweet spot on M4 Max with IDS worker co-resident")
	# Plan A → Plan B chaining: save the final (post-memory) genome to disk so
	# run_memory_refinement.py can load it and refine its cells under a new
	# fitness weight schema (e.g. stability-dominant). Pickle, not JSON, because
	# the genome graph contains MemoryPayload + RecurrentArchShape nested
	# dataclasses; pickle is one-line, JSON would need custom encoders.
	ap.add_argument("--report-seed", type=int, default=None,
		help="TRUE held-out: after the run, re-eval the final winner on this fresh seed "
		     "(must differ from the train/select seed). The honest paper number.")
	ap.add_argument("--save-winner", type=str, default=None,
	                help="Path to pickle the final-stage winner + FULL FINAL POPULATION "
	                     "(spec + best_genome + all evolved genomes + cells + provenance). "
	                     "For Plan B chain: pair Plan A's --save-winner X with "
	                     "tests/run_memory_refinement.py --load-winner X — Plan B seeds "
	                     "its GA from Plan A's evolved pool (not random init).")
	# Per-stage checkpoint save (added 30/05/2026 after Plan A v1 lost 5.5h of
	# work in an OOM-triggered reboot mid-Stage-3). When set, writes
	# {DIR}/stage{N}_{name}.pkl after each stage completes. Same pickle schema
	# as --save-winner, so any stage checkpoint is loadable by
	# run_memory_refinement.py for analysis or re-launch.
	ap.add_argument("--save-stage-checkpoints", type=str, default=None,
	                help="Directory: dump per-stage pickle after each phase finishes. "
	                     "Survives reboots — Plan B / future re-launches can load any "
	                     "intermediate stage.")
	# K-fold cross-validation for the controller GA fitness eval (added
	# 30/05/2026 after Plan A v1 Stage-1 showed 3.65° / 10pp generalization
	# gap from single-pool episode overfit). K=1 reproduces legacy behavior.
	ap.add_argument("--num-eval-folds", type=int, default=5,
	                help="K episode pools per genome eval. DEFAULT 5 (project rule: kfold always "
	                     "5, never 1). For the lamarckian/adaptation path the K folds ACCUMULATE "
	                     "into one controller (cells compound via warm-start chaining — "
	                     "_train_genome_accumulate); folds are random episode seeds, not a finite "
	                     "partition, so this is 'more rollouts', not a CV leak. K=1 only for debug. "
	                     "See docs/controller_kfold_design.md + CLAUDE.md 'K-fold: Always 5'.")
	# Resume from emergency dump (added 31/05/2026). The dump pickle is written
	# by the SIGTERM handler at the next safe GA gen boundary; it captures the
	# current stage's spec + population + best genome. Use --resume-mode to
	# choose whether to continue the dumped stage or skip to the next.
	ap.add_argument("--resume-from-emergency", type=str, default=None,
	                help="Path to an emergency-dump pickle (see _dump_emergency_state). "
	                     "When set, Stage 0 (grid) is skipped and the run starts at the "
	                     "stage selected by --resume-mode.")
	ap.add_argument("--resume-mode", type=str, default="same",
	                choices=["same", "next"],
	                help="'same' (default): continue the dumped stage from its dumped "
	                     "population. 'next': skip the dumped stage and warm-start the "
	                     "next stage from the dumped best_genome.")

	# Seed plumbing (3-way + multi-run, matches run_ga_memory.py / run_mlp_ga.py).
	ap.add_argument("--seed", type=int, default=42, help="legacy single-seed (used when base-seed unset)")
	ap.add_argument("--base-seed", type=int, default=None,
	                help="Master seed for the 3-way SeedSet protocol; default = UTC timestamp.")
	ap.add_argument("--runs", type=int, default=1)
	ap.add_argument("--train-seed", type=int, default=None)
	ap.add_argument("--test-seed", type=int, default=None)
	ap.add_argument("--val-seed", type=int, default=None)
	return ap


def main():
	args = build_arg_parser().parse_args()

	# Install SIGTERM/SIGINT handlers BEFORE any Rust work begins so that
	# SIGTERM during stage 0 grid or any subsequent stage is caught and
	# triggers a clean emergency dump.
	_install_signal_handlers()

	# Load emergency-dump pickle if --resume-from-emergency is set. The loaded
	# state is forwarded to _run_one via the resume_state arg.
	resume_state = None
	if args.resume_from_emergency:
		resume_path = Path(args.resume_from_emergency)
		if not resume_path.exists():
			raise FileNotFoundError(f"--resume-from-emergency {resume_path} does not exist")
		with open(resume_path, "rb") as f:
			resume_state = pickle.load(f)
		resume_state["resume_mode"] = args.resume_mode
		print(f"[main] Loaded emergency dump from {resume_path}")
		print(f"[main]   stage_num={resume_state.get('stage_num')} "
		      f"stage_name={resume_state.get('stage_name')!r} "
		      f"generation={resume_state.get('generation')} "
		      f"pop={len(resume_state.get('population') or [])}")

	t_start = time.time()
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate,
	)

	print(f"Phased-GA controller search: "
	      f"grid ({len(args.grid_state_neurons)}×{len(args.grid_bits)}={len(args.grid_state_neurons)*len(args.grid_bits)}) "
	      f"+ {args.neurons_gens}n + {args.bits_gens}b + {args.conns_gens}c + {args.memory_gens}m  "
	      f"(target {args.neurons_gens+args.bits_gens+args.conns_gens+args.memory_gens} GA gens)")
	print(f"Pop={args.pop} elitism={args.elitism:.0%} crossover={args.crossover_rate:.0%} "
	      f"eval_episodes={args.eval_episodes} steps={args.steps} tilt={args.tilt}° "
	      f"levels={args.levels}")

	val_runs = []
	for run_i in range(args.runs):
		# When the user supplies --seed only (no --base-seed), reuse --seed AS the
		# base so single-run smoke tests keep their explicit determinism. When
		# --base-seed IS set, the seed registry generates train/test/val triples.
		base = args.base_seed if args.base_seed is not None else args.seed
		s = resolve_seed_set(base=base, run_index=run_i,
		                     train=args.train_seed, test=args.test_seed, val=args.val_seed)
		log_seed_set(s)
		record_seed_set(s, script="run_phased_ga", extra={
			"grid_sn": args.grid_state_neurons, "grid_bits": args.grid_bits,
			"levels": args.levels, "pop": args.pop,
			"neurons_gens": args.neurons_gens, "bits_gens": args.bits_gens,
			"conns_gens": args.conns_gens, "memory_gens": args.memory_gens,
		})
		stage_results, best_final, final_population, pid_m = _run_one(args, ec, s,
		                                                              resume_state=resume_state)
		val_runs.append((stage_results, best_final, final_population, pid_m))

	# Single-run path: print the per-run summary directly.
	stage_results, best_final, final_population, pid_m = val_runs[-1]
	_print_final_summary(args, stage_results, best_final, pid_m, time.time() - t_start)

	# Held-out (REPORT ONLY) now fires PER-STAGE inside _run_one (N→B→C→M trajectory);
	# the MEMORY-stage per-stage held-out IS the final number, so nothing to add here.

	if args.save_winner is not None and best_final is not None:
		# stage_results[-1] is the Memory stage tuple (name, spec, metrics, dt, iters).
		mem_spec    = stage_results[-1][1]
		mem_metrics = stage_results[-1][2]
		_save_winner(args.save_winner, args, mem_spec,
		             best_final, final_population, mem_metrics)

	# Multi-run aggregation: mean±std of the FINAL (memory-stage) metrics.
	if args.runs > 1:
		print(f"\n{'='*72}\n  MULTI-RUN ({args.runs} runs) — Stage 4 mean±std\n{'='*72}")
		errs = [r[0][-1][2].mean_attitude_error_deg for r in val_runs if r[0][-1][2] is not None]
		stables = [r[0][-1][2].acc for r in val_runs if r[0][-1][2] is not None]
		rewards = [r[0][-1][2].fitness for r in val_runs if r[0][-1][2] is not None]
		if errs:
			a = np.array(errs); s = np.array(stables); r = np.array(rewards)
			print(f"  err     : {a.mean():.2f} ± {a.std():.2f}°")
			print(f"  stable  : {s.mean()*100:.1f} ± {s.std()*100:.1f}%")
			print(f"  reward  : {r.mean():.2f} ± {r.std():.2f}")

	return 0


if __name__ == "__main__":
	sys.exit(main())
