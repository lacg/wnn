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
import sys
import time

import numpy as np

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
	"""20-point (default) grid over (state_neurons × bits). Returns the winning
	(spec, best_genome, best_metrics, wall_time) for warm-starting Stage 1."""
	t0 = time.time()
	print(f"\n{'='*72}\n  STAGE 0: GRID SEARCH "
	      f"({len(args.grid_state_neurons)}×{len(args.grid_bits)}={len(args.grid_state_neurons)*len(args.grid_bits)} pts, "
	      f"levels={args.levels})\n{'='*72}")

	# Build a representative spec just to fit thresholds (any shape works — they
	# come from PID rollouts which are arch-independent). Use the SMALLEST grid
	# point to keep PID-fit time minimal.
	probe_spec = _make_spec(min(args.grid_state_neurons), args.levels, min(args.grid_bits))
	thresholds = fit_thresholds_from_pid_rollouts(probe_spec, num_episodes=10, seed=seed)

	# Shared universe: recorded ONCE on the first grid point's shape. Every
	# subsequent grid point reuses the filtered subset.
	shared_universe = None
	first_spec = None
	rng_master = np.random.default_rng(seed)

	results = []  # (spec, genome, metrics)
	for i, sn in enumerate(args.grid_state_neurons):
		for j, b in enumerate(args.grid_bits):
			spec = _make_spec(sn, args.levels, b)
			shape = arch_shape_from_spec(spec)
			suffix = b - 2 * sn  # bits = forced_prefix (2·sn) + sampled suffix
			if suffix < 1:
				print(f"  [skip] sn={sn} b={b}: bits<prefix (suffix={suffix})")
				continue
			rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
			genome = RecurrentArchGenome.random(
				shape, state_neurons=sn,
				output_neurons=spec.num_motors * spec.levels_per_motor,
				state_suffix=suffix, output_suffix=suffix, rng=rng,
			)
			# Record a fresh universe on the FIRST point; reuse + filter thereafter.
			if shared_universe is None:
				sc, oc = genome.to_connections()
				su, ou = record_address_universe(
					spec, thresholds, sc, oc,
					num_episodes=args.universe_episodes,
					steps=args.steps, seed=seed,
				)
				shared_universe = (su, ou)
				first_spec = spec
				print(f"  [grid] recorded universe on first point sn={sn} b={b}: "
				      f"{len(su)} state / {len(ou)} output cells")
			su, ou = shared_universe
			seed_payload = MemoryPayload(
				list(su), list(ou),
				[int(v) for v in rng.integers(0, 4, len(su))],
				[int(v) for v in rng.integers(0, 4, len(ou))],
			)
			genome.cells = _filter_cells_for_arch(seed_payload, genome)
			# Score (no training — cells provide the lookup).
			ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes,
			                         seed=seed, episode_config=ec, thresholds=thresholds)
			m = ev.score_genomes([genome])[0]
			results.append((spec, genome, m))
			print(f"  [{len(results):>2}/{len(args.grid_state_neurons)*len(args.grid_bits):>2}] "
			      f"sn={sn:>2} b={b:>2} cells=(s{len(genome.cells.state_universe)}/o{len(genome.cells.output_universe)})  "
			      f"CE={m.ce:>8.4f}  err={m.mean_attitude_error_deg:>6.2f}°  stable={m.acc*100:>5.1f}%")

	if not results:
		raise RuntimeError("Grid search produced no valid points (all skipped).")

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
	gacfg = default_controller_ga_config(population_size=args.pop, generations=gens)
	gacfg.patience = patience
	gacfg.elitism_pct = args.elitism
	gacfg.crossover_rate = args.crossover_rate
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
                    seed: int):
	"""Generic Stage 1-3 driver: build an ArchGAStrategy on the given dimension
	and run optimize(). Returns (result, evaluator, wall_time)."""
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes,
	                         seed=seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, seed))
	arch_cfg = default_controller_arch_config(spec)
	# Widen the search box to admit the grid winner + room to mutate. The default
	# max_state_neurons is 4·spec.state_neurons; honor the user's grid maximum so
	# the GA can climb past the seed if it likes.
	arch_cfg.max_state_neurons = max(arch_cfg.max_state_neurons,
	                                 4 * max(args.grid_state_neurons))
	gacfg = _build_ga_config(args, gens, patience)
	strat = ControllerArchGAStrategy(spec, dimension, arch_config=arch_cfg,
	                                 ga_config=gacfg, seed=seed, batch_evaluator=ev)
	t = time.time()
	res = strat.optimize(evaluate_fn=lambda g: ev.evaluate_batch([g])[0].ce,
	                     batch_evaluate_fn=ev.evaluate_batch)
	return res, ev, time.time() - t


def _run_memory_phase(args, ec: EpisodeConfig, spec: ControllerSpec,
                      gens: int, patience: int, seed: int):
	"""Stage 4: arch FROZEN at `spec`; evolve QSR cell VALUES over a recorded
	universe. The strategy auto-records the universe on its own seed arch via
	_ensure_universe (called inside _make_cell_genome)."""
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes,
	                         seed=seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, seed))
	arch_cfg = default_controller_arch_config(spec)
	arch_cfg.max_state_neurons = max(arch_cfg.max_state_neurons,
	                                 4 * max(args.grid_state_neurons))
	gacfg = _build_ga_config(args, gens, patience)
	strat = ControllerMemoryGAStrategy(
		spec, arch_config=arch_cfg, ga_config=gacfg,
		seed=seed, batch_evaluator=ev, thresholds=thresholds,
		record_episodes=args.universe_episodes, record_steps=args.steps,
	)
	t = time.time()
	# MEMORY paradigm: cells ARE the genome → score_genomes (no training).
	res = strat.optimize(evaluate_fn=lambda g: ev.score_genomes([g])[0].ce,
	                     batch_evaluate_fn=ev.score_genomes)
	return res, ev, time.time() - t


# -----------------------------------------------------------------------------
# Baselines (PID + reference numbers from prior runs)
# -----------------------------------------------------------------------------

def _pid_baseline(ec: EpisodeConfig, episodes: int, seed: int):
	"""PID score on the held-out episode set, for the final-summary 'vs PID' row."""
	pid = AttitudePID(AttitudePIDConfig())
	from wnn.control.training import make_pid_action_fn
	_, m = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, ec, episodes, seed)
	return m


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _run_one(args, ec: EpisodeConfig, seeds):
	"""One full phased run for a SeedSet. Returns the per-stage metrics + final
	best genome for the multi-run aggregator (when --runs>1)."""
	# Use the train seed for everything inside the run; the test/val seeds are
	# only for the final report (PID baseline + held-out reference).
	seed = seeds.train

	# Stage 0 — grid.
	winner_spec, _winner_genome, m0, dt0, _thr = stage0_grid(args, ec, seed)
	stage_results = [("Grid", winner_spec, m0, dt0, len(args.grid_state_neurons) * len(args.grid_bits))]

	# Stage 1 — NEURONS (warm-start from grid winner spec).
	spec1 = winner_spec
	_stage_header(1, "NEURONS", args.neurons_gens, args.neurons_patience, spec1)
	res1, ev1, dt1 = _run_arch_phase(args, ec, spec1, OptimizationDimension.NEURONS,
	                                 args.neurons_gens, args.neurons_patience, seed)
	m1 = _print_stage_result(1, "NEURONS", res1, args.neurons_gens, dt1, ev1)

	# Stage 2 — BITS (warm-start from Stage 1's best genome shape).
	base = winner_spec
	spec2 = _spec_from_best(res1.best_genome, base) if res1.best_genome is not None else spec1
	_stage_header(2, "BITS", args.bits_gens, args.bits_patience, spec2)
	res2, ev2, dt2 = _run_arch_phase(args, ec, spec2, OptimizationDimension.BITS,
	                                 args.bits_gens, args.bits_patience, seed)
	m2 = _print_stage_result(2, "BITS", res2, args.bits_gens, dt2, ev2)

	# Stage 3 — CONNECTIONS (warm-start from Stage 2's best).
	spec3 = _spec_from_best(res2.best_genome, base) if res2.best_genome is not None else spec2
	_stage_header(3, "CONNECTIONS", args.conns_gens, args.conns_patience, spec3)
	res3, ev3, dt3 = _run_arch_phase(args, ec, spec3, OptimizationDimension.CONNECTIONS,
	                                 args.conns_gens, args.conns_patience, seed)
	m3 = _print_stage_result(3, "CONNECTIONS", res3, args.conns_gens, dt3, ev3)

	# Stage 4 — MEMORY (arch FROZEN at Stage 3's winning shape).
	spec4 = _spec_from_best(res3.best_genome, base) if res3.best_genome is not None else spec3
	_stage_header(4, "MEMORY", args.memory_gens, args.memory_patience, spec4)
	res4, ev4, dt4 = _run_memory_phase(args, ec, spec4, args.memory_gens, args.memory_patience, seed)
	m4 = _print_stage_result(4, "MEMORY", res4, args.memory_gens, dt4, ev4)

	# PID baseline on the val seed (the held-out reference).
	pid_m = _pid_baseline(ec, args.eval_episodes, seeds.val)

	stage_results += [
		("Neurons",     spec1, m1, dt1, res1.iterations_run),
		("Bits",        spec2, m2, dt2, res2.iterations_run),
		("Connections", spec3, m3, dt3, res3.iterations_run),
		("Memory",      spec4, m4, dt4, res4.iterations_run),
	]
	return stage_results, res4.best_genome, pid_m


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


def main():
	ap = argparse.ArgumentParser()
	# Grid (Stage 0).
	ap.add_argument("--grid-state-neurons", type=int, nargs="+",
	                default=[8, 12, 16, 20, 24],
	                help="state_neurons axis for Stage 0 grid")
	ap.add_argument("--grid-bits", type=int, nargs="+",
	                default=[18, 24, 30, 36],
	                help="bits-per-neuron axis for Stage 0 grid")
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
	# Shared GA hyperparams.
	ap.add_argument("--pop", type=int, default=200, help="per-stage population")
	ap.add_argument("--elitism", type=float, default=0.2)
	ap.add_argument("--crossover-rate", type=float, default=0.5)
	# Evaluation / episode.
	ap.add_argument("--eval-episodes", type=int, default=20)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	ap.add_argument("--universe-episodes", type=int, default=8)
	# Inner reward-gated train knobs (production: leave None → 8 rounds × 24 eps);
	# smoke tests pass tiny values to keep per-genome training under a few seconds.
	ap.add_argument("--rg-rounds", type=int, default=None,
	                help="Reward-gated inner-train rounds (default: 8)")
	ap.add_argument("--rg-episodes-per-round", type=int, default=None,
	                help="Episodes per reward-gated round (default: 24)")
	ap.add_argument("--rg-eval-episodes", type=int, default=None,
	                help="Eval episodes within reward-gated (default: 20)")
	# Seed plumbing (3-way + multi-run, matches run_ga_memory.py / run_mlp_ga.py).
	ap.add_argument("--seed", type=int, default=42, help="legacy single-seed (used when base-seed unset)")
	ap.add_argument("--base-seed", type=int, default=None,
	                help="Master seed for the 3-way SeedSet protocol; default = UTC timestamp.")
	ap.add_argument("--runs", type=int, default=1)
	ap.add_argument("--train-seed", type=int, default=None)
	ap.add_argument("--test-seed", type=int, default=None)
	ap.add_argument("--val-seed", type=int, default=None)
	args = ap.parse_args()

	t_start = time.time()
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
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
		stage_results, best_final, pid_m = _run_one(args, ec, s)
		val_runs.append((stage_results, best_final, pid_m))

	# Single-run path: print the per-run summary directly.
	stage_results, best_final, pid_m = val_runs[-1]
	_print_final_summary(args, stage_results, best_final, pid_m, time.time() - t_start)

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
