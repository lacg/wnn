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
import os
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
#   2. The next call to the GA's _on_generation_start hook (ControllerCancelMixin
#      on the shared base) sees the flag, dumps the current stage + population +
#      spec + best genome to a schema-2 yaml.gz checkpoint, and raises
#      StopIteration. The strategy catches StopIteration, marks the run as
#      shutdown-requested, and returns cleanly.
#   3. The checkpoint schema mirrors --save-winner so the resume path can load
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
#   --resume-mode same             continue the same stage from the dumped
#                                  population (default)
#   --resume-mode next             skip the dumped stage entirely and start
#                                  the NEXT stage with the dumped best as
#                                  warm-start

# --- Emergency / in-stage crash-save lives on the shared cooperative-cancel core.
# Per-strategy state is held by ControllerCancelMixin (its per-gen hook on
# GenericGAStrategy._checkpoint_and_maybe_stop); the phase driver wires the
# checkpoint manager + stage identity through `_wire_cancel` (below). Only the
# OS-signal layer (_sigterm_handler / _install_signal_handlers) lives here.

def _emergency_dir_for(args) -> Path:
	"""Where stage crash-save / emergency-dump checkpoints land: next to the
	per-stage checkpoint dir when --save-stage-checkpoints is set, else /tmp
	(so cancel-dump protection is always on, even without --save-stage-checkpoints)."""
	return (Path(args.save_stage_checkpoints) if args.save_stage_checkpoints
	        else Path("/tmp/wnn-phased-ga-emergency"))


def _stage_emergency_path(args, stage_num: int, stage_name: str) -> str:
	"""Per-stage emergency/crash-save checkpoint path (schema mirrors --save-winner
	so the resume path loads it like any other stage checkpoint)."""
	return str(_emergency_dir_for(args) / f"emergency_stage{stage_num}_{stage_name.lower()}.pkl")



# --- Unified checkpoint store (D1, 11/06/2026) ----------------------------
# Controller checkpoints write schema-2 yaml.gz via wnn.control.checkpoint_io
# (genomes natively serialized — no pickle); legacy pickles still load.
from wnn.control.checkpoint_io import (
	load_controller_checkpoint as _ctl_load_optional,
	save_controller_checkpoint as _ctl_save,
	save_controller_checkpoint_async as _ctl_save_async,
)

# Background checkpoint-writer threads (between-stage saves are resume-only, off
# the next stage's critical path → written async). Joined at run end so a normal
# exit never loses an in-flight write.
_PENDING_SAVES: list = []

# In-stage crash protection is now owned by each strategy's own
# PhasedCheckpointManager (armed by `_wire_cancel`, joined inside the phase driver
# right after optimize()). No module-level periodic manager anymore.

def _join_pending_saves() -> None:
	"""Join the between-stage async writers at run end so a normal exit never
	loses an in-flight save."""
	for t in _PENDING_SAVES:
		try:
			t.join()
		except Exception:
			pass
	_PENDING_SAVES.clear()


def _ctl_load(path):
	payload = _ctl_load_optional(path)
	if payload is None:
		raise FileNotFoundError(path)
	return payload


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
		from wnn.control import _accel as ram_accelerator
		ram_accelerator.set_cancel_flag()
	except Exception as e:
		print(f"[{name}] Could not set Rust cancel flag: {e}", flush=True)


def _install_signal_handlers() -> None:
	"""Wire SIGTERM + SIGINT to the cooperative-cancellation path."""
	signal.signal(signal.SIGTERM, _sigterm_handler)
	signal.signal(signal.SIGINT,  _sigterm_handler)
	# Make sure no prior process left the Rust flag set.
	try:
		from wnn.control import _accel as ram_accelerator
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


def _wire_cancel(strat, args, stage_num: int, stage_name: str) -> None:
	"""Wire the SHARED cooperative-cancel core onto a controller strategy before
	optimize(). Sets:

	  * `_checkpoint_mgr` — an adaptive in-stage crash-save manager
	    (PhasedCheckpointManager, async). ALWAYS armed (writes to /tmp when
	    --save-stage-checkpoints is unset), so the SIGTERM/cancel dump + hard-crash
	    protection are always on, matching the old always-armed periodic hook. A
	    hard crash loses ≤~one slow gen; a cooperative cancel dumps the live
	    population for --resume-from-emergency.
	  * `_shutdown_check` — polls the Rust cancel flag; when set, the base saves
	    once more and raises StopIteration to unwind the GA cleanly.
	  * `_stage_num` / `_stage_name` / `_checkpoint_meta` — the stage identity +
	    provenance the mixin's _build_checkpoint stamps into the resume payload.

	Re-armed per stage (each stage's first gen establishes its own cadence
	baseline)."""
	from wnn.control import _accel as ram_accelerator
	from wnn.ram.strategies.phased import (
		PhasedCheckpointManager, SaveCadence, ControllerGenomeCodec)
	strat._shutdown_check = lambda: ram_accelerator.is_cancelled()
	strat._stage_num = stage_num
	strat._stage_name = stage_name
	strat._checkpoint_meta = {
		"levels":        args.levels,
		"tilt_deg":      args.tilt,
		"steps":         args.steps,
		"eval_episodes": args.eval_episodes,
	}
	budget = getattr(args, "checkpoint_target_loss_seconds", None)
	max_int = getattr(args, "checkpoint_max_interval", 10)
	strat._checkpoint_mgr = PhasedCheckpointManager(
		Path(_stage_emergency_path(args, stage_num, stage_name)),
		ControllerGenomeCodec(), SaveCadence(budget, max_int), async_save=True)

from wnn.control.evaluator import (
	ControllerSpec, ControllerEvaluator, arch_shape_from_spec, spec_from_arch,
	fit_thresholds_from_pid_rollouts,
)
from wnn.control.arch_strategy import (
	ControllerArchGAStrategy, ControllerArchTSStrategy,
	ControllerMemoryGAStrategy, ControllerMemoryTSStrategy,
	default_controller_arch_config,
)
from wnn.control.ga_strategy import default_controller_ga_config
from wnn.control.ga_memory import record_address_universe
from wnn.control.recurrent_genome import RecurrentArchGenome, MemoryPayload
from wnn.control.controller_grid_search import _steady_str
from wnn.control.training import EpisodeConfig, make_pid_action_fn
from wnn.control.dagger import eval_closed_loop_reset
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.reward_gated import RewardGatedConfig
from wnn.ram.strategies.optimization_dimension import OptimizationDimension
from wnn.seeds import resolve_seed_set, log_seed_set, record_seed_set


# Arch dimension → pipeline stage number (NEURONS is always Stage 1, BITS 2,
# CONNECTIONS 3; MEMORY is Stage 4, wired directly in _run_memory_phase). Lets a
# phase function self-derive its stage identity for _wire_cancel from `dimension`
# alone.
_STAGE_BY_DIM = {
	OptimizationDimension.NEURONS:     1,
	OptimizationDimension.BITS:        2,
	OptimizationDimension.CONNECTIONS: 3,
}


# -----------------------------------------------------------------------------
# Shape / spec plumbing
# -----------------------------------------------------------------------------

def grid_output_neuron_axis(args) -> list[int]:
	"""The Stage-0 output-neuron axis, in OUTPUT-NEURON units.

	output_neurons IS the PWM decode resolution: evaluator.spec_from_arch derives
	`levels_per_motor = output_neurons // num_motors`, so 64 = 4 motors × 16 levels.
	Sweeping it sweeps thermometer precision, not just capacity.

	Default (flag absent) = the single point implied by --levels, i.e. exactly the
	pre-flag behaviour. Every value must be a whole multiple of the OUTPUT QUANTUM,
	else the floor division silently truncates the resolution the user asked for.

	The quantum is num_motors, DOUBLED under BINARY: the antagonist E/I decode needs
	an even levels_per_motor for a symmetric split (odd L drifts neutral off 0.5),
	so arch_shape_from_spec sets output_quantum = 2·num_motors there. Validating
	against num_motors alone would let a BINARY run request an odd level count that
	neurogenesis can never actually hold."""
	num_motors = int(getattr(args, "_geometry_num_motors", 4))
	quantum = num_motors * 2 if getattr(args, "memory_mode", "") == "BINARY" else num_motors
	requested = getattr(args, "grid_output_neurons", None)
	if not requested:
		return [num_motors * int(args.levels)]
	bad = [on for on in requested if on <= 0 or on % quantum]
	if bad:
		raise ValueError(
			f"--grid-output-neurons {bad} is not a positive multiple of the output quantum "
			f"{quantum} (num_motors={num_motors}"
			f"{', doubled for BINARY even-levels' if quantum != num_motors else ''}). "
			f"output_neurons = num_motors · levels_per_motor, so use e.g. "
			f"{' '.join(str(num_motors * lv) for lv in (16, 24, 32))} for 16/24/32 levels.")
	return [int(on) for on in requested]


def grid_point_count(args) -> int:
	"""Stage-0 grid cardinality across all three axes (before validity filtering)."""
	return (len(args.grid_state_neurons) * len(args.grid_bits)
	        * len(grid_output_neuron_axis(args)))


def apply_output_neuron_ceiling(args, arch_cfg) -> None:
	"""Apply --max-output-neurons, then reconcile it with the Stage-0 output axis.

	The ceiling stays a HARD bound: output cells are the other half of the
	per-genome memory that balloons NEURONS into OOM (24/07/2026 — the uncapped
	default 4·levels·q let a NEURONS population thrash the box to 0 free RAM →
	jetsam). The GA may still shrink below it.

	If --grid-output-neurons asks the grid to explore ABOVE that ceiling we refuse
	loudly instead of clamping, because clamping would report "searched 128 output
	neurons" while having searched 64 — a silent truncation of the stated design."""
	if getattr(args, "max_output_neurons", None):
		arch_cfg.max_output_neurons = min(arch_cfg.max_output_neurons, int(args.max_output_neurons))
		arch_cfg.min_output_neurons = min(arch_cfg.min_output_neurons, arch_cfg.max_output_neurons)
	grid_max = max(grid_output_neuron_axis(args))
	if grid_max > arch_cfg.max_output_neurons:
		raise ValueError(
			f"--grid-output-neurons tops out at {grid_max}, above the output-neuron ceiling "
			f"{arch_cfg.max_output_neurons} — the grid winner would be clamped straight back "
			f"down. Raise --max-output-neurons to >= {grid_max}, or lower the grid axis.")


def _make_spec(state_neurons: int, levels: int, bits: int,
               delta_control: bool = True, delta_leak: float = 0.95,
               delta_max: float = 0.1,
               obs_tilt_p: bool = False, obs_tilt_i: bool = False,
               obs_peraxis_p: bool = False, obs_peraxis_i: bool = False,
               obs_peraxis_yaw: bool = True,
               obs_pwm: bool = False,
               obs_yaw_err: bool = False, obs_yaw_err_i: bool = False,
               dhat_b: "tuple[float, float, float] | None" = None,
               dhat_l_gain: float = 0.05,
               integral_leak: float = 0.99, integral_scale: float = 1.0,
               dt: float = 0.001,
               decouple_outputs: bool = False, bits_per_feature: int = 8,
               feature_balance_ratio: float = 0.0,
               threshold_gamma: float = 1.0,
               action_repeat: int = 1,
               output_bits: "int | None" = None,
               num_motors: int = 4,
               input_window_k: int = 4,
               memory_mode: str = "QUAD_WEIGHTED",
               output_decode: "str | None" = None) -> ControllerSpec:
	"""Build a ControllerSpec from a (state_neurons, levels, bits) grid point.
	`bits` becomes BOTH state_bits_per_neuron and output_bits_per_neuron, matching
	the grid-search convention (the GA can later split them in the BITS phase).

	delta_control (default True, 08/06/2026): the output decodes to a per-step PWM
	DELTA with a leaky accumulator — a structural integrator that offloads PID's
	I-term out of the (currently untrained) recurrent state. Empirically +5pp
	stability (71→76% @leak=0.95) vs the old hardcoded False. It's a PARTIAL fix
	(the policy is still memoryless — see project_controller_stability_diagnosis);
	the full fix is training the state as a learned integrator. The grid spec's
	delta_control/leak propagate to all later stages via spec_from_arch(base).

	input_window_k (CLI --input-window-k, default 4): how many past timesteps of
	sensor features the address window carries. It grows the input POOL linearly
	(k·nf·bits_per_feature) but NOT the address space — that stays 2^(prefix+suffix)
	— so the cost is sampling coverage, not memory. Pair a raise with more neurons.
	It is a shared scalar, not a gene: the batched evaluator requires num_motors,
	bits_per_feature and input_window_k to agree across every genome in a pass."""
	return ControllerSpec(
		num_motors=num_motors, levels_per_motor=levels, bits_per_feature=bits_per_feature,
		input_window_k=input_window_k,
		state_neurons=state_neurons,
		state_bits_per_neuron=bits, output_bits_per_neuron=(output_bits if output_bits is not None else bits),
		delta_control=delta_control, delta_leak=delta_leak, delta_max=delta_max,
		obs_tilt_p=obs_tilt_p, obs_tilt_i=obs_tilt_i,
		obs_peraxis_p=obs_peraxis_p, obs_peraxis_i=obs_peraxis_i,
		obs_peraxis_yaw=obs_peraxis_yaw,
		obs_pwm=obs_pwm,
		obs_yaw_err=obs_yaw_err, obs_yaw_err_i=obs_yaw_err_i,
		dhat_b=dhat_b, dhat_l_gain=dhat_l_gain,
		integral_leak=integral_leak, integral_scale=integral_scale,
		dt=dt,
		decouple_outputs=decouple_outputs,
		feature_balance_ratio=feature_balance_ratio,
		threshold_gamma=threshold_gamma,
		action_repeat=action_repeat,
		memory_mode=memory_mode,
		output_decode=output_decode,
	)


def _spec_from_best(best: RecurrentArchGenome, base: ControllerSpec) -> ControllerSpec:
	"""ControllerSpec carrying the previous stage's WINNING shape. The next stage
	resets its seed/arch dims from this so create_random_genome/MEMORY universe
	recording pin to the right reference."""
	return spec_from_arch(best, base)




# -----------------------------------------------------------------------------
# Stage 0 — Grid search
# -----------------------------------------------------------------------------

def _geometry_from_args(args, base_seed: int):
	"""Overactuated residual mode (Phase 2 step 4): build the TRUE-vehicle
	GeometryConfig + AllocResidualConfig from --geometry preset + perturbation
	magnitudes. Presets and the tilt/position perturbation math live in Rust
	(AttitudeSim) — Python only reads the resulting rows back (geometry_rows),
	so there is exactly one implementation. Per-rotor perturbation draws are
	seeded from the base seed → reproducible true-vehicle tables per run.
	Returns (GeometryConfig|None, AllocResidualConfig|None)."""
	preset = getattr(args, "geometry", None)
	if not preset:
		return None, None
	from wnn.control._accel import AttitudeSim
	from wnn.control.training import AllocResidualConfig, GeometryConfig
	sim = AttitudeSim()
	if preset == "octo-x":
		sim.set_geometry_octo_x(0.075, 2.4, 0.05)
	elif preset == "canted-hex":
		sim.set_geometry_canted_hex(0.075, 2.4, 0.05, float(args.geometry_cant))
	elif preset == "quad-plus":
		sim.set_geometry_quad_plus(0.075, 2.4, 0.05)
	else:
		raise SystemExit(f"--geometry: unknown preset {preset!r}")
	nominal = [list(r) for r in sim.geometry_rows()]
	n = len(nominal)
	rng = np.random.default_rng(((base_seed or 0) * 2654435761 + 0x9E0) % (2**63))
	tilt_mag = float(getattr(args, "geometry_tilt_err", 0.0))
	pos_mag = float(getattr(args, "geometry_pos_err", 0.0))
	if tilt_mag > 0.0 or pos_mag > 0.0:
		tilts = [float(rng.uniform(-tilt_mag, tilt_mag)) for _ in range(n)]
		poss = [[float(rng.uniform(-pos_mag, pos_mag)) for _ in range(3)] for _ in range(n)]
		sim.perturb_geometry(tilts, poss)
	true_rows = [list(r) for r in sim.geometry_rows()]
	asym = None
	asym_mag = float(getattr(args, "rotor_asym", 0.0))
	if asym_mag > 0.0:
		asym = [float(1.0 + rng.uniform(-asym_mag, asym_mag)) for _ in range(n)]
	geo = GeometryConfig(rows=true_rows, rotor_asym=asym)
	ar = AllocResidualConfig(
		nominal_rows=nominal,
		scale=float(getattr(args, "alloc_scale", 1.0)),
		clamp=float(getattr(args, "alloc_clamp", 0.15)),
		tau_max=float(getattr(args, "alloc_tau_max", 0.144)))
	return geo, ar


def stage0_grid(args, ec: EpisodeConfig, seed: int):
	"""Grid over (state_neurons × bits). Returns
	(winner_spec, seed_population, winner_metrics, wall_time, thresholds).

	Delegates to ControllerGridSearch (the shared GenericGridSearch core): every
	valid (sn, b) shape is evaluated once through ONE shared mixed-shape evaluator,
	ranked by the controller fitness calculator (NOT raw CE), and the top-K shapes
	are expanded into a full seed population (size --pop) that seeds Stage 1 —
	instead of discarding all but the single winner. `seed_population[0]` is the
	fitness-best genome; its spec is `winner_spec`.
	"""
	from wnn.control.controller_grid_search import ControllerGridSearch
	gs = ControllerGridSearch(args, ec, seed)
	outcome = gs.run()
	winner_spec = outcome.best_point.spec
	return winner_spec, outcome.seed_population, outcome.best_metrics, gs.elapsed, gs.thresholds


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
		weight_steady=args.fit_weight_steady,
		weight_effort=getattr(args, "fit_weight_effort", 0.0),
	)
	gacfg.patience = patience
	gacfg.elitism_pct = args.elitism
	gacfg.crossover_rate = args.crossover_rate
	gacfg.check_interval = args.check_interval
	# Magnitude-aware patience (controller redesign (a), 16/06/2026). Opt-in; when
	# off the early-stopper keeps watching the rank-WHM (comparable with the cohort
	# + C10 sweep). When on, it watches err°/stable% magnitude — recovers patience
	# proportional to real improvement so genuine jumps don't get mis-early-stopped.
	gacfg.magnitude_aware_patience = args.magnitude_aware_patience
	# E1 random immigrants: fraction of each gen's offspring drawn fresh from
	# create_random_genome (applies to BOTH arch and memory stages — the memory
	# strategy's create_random_genome makes fresh random cell-genomes).
	gacfg.immigrant_fraction = args.immigrants
	return gacfg


def _build_ts_config(args, gens: int, patience: int):
	"""TSConfig per stage (--strategy ts, 19/07/2026 single-layer promotion).
	Mirrors _build_ga_config: same controller fitness weights + patience knobs
	(all live on the shared OptimizationConfig base). iterations ⇢ gens and
	neighbors_per_iter ⇢ pop, so a TS stage consumes a comparable eval budget
	per 'generation' to the GA it replaces."""
	from wnn.control.arch_strategy import default_controller_ts_config
	from wnn.ram.fitness import FitnessCalculatorType
	tscfg = default_controller_ts_config(iterations=gens, neighbors_per_iter=args.pop)
	multi_obj = (args.fit_weight_stable > 0 or args.fit_weight_jerk > 0
	             or args.fit_weight_mono > 0 or args.fit_weight_steady > 0
	             or getattr(args, "fit_weight_effort", 0.0) > 0)
	if multi_obj:
		tscfg.fitness_calculator_type = FitnessCalculatorType.CONTROLLER_HARMONIC
	tscfg.fitness_weight_err_sq = args.fit_weight_err_sq
	tscfg.fitness_weight_stable = args.fit_weight_stable
	tscfg.fitness_weight_jerk = args.fit_weight_jerk
	tscfg.fitness_weight_mono = args.fit_weight_mono
	tscfg.fitness_weight_steady = args.fit_weight_steady
	tscfg.fitness_weight_effort = getattr(args, "fit_weight_effort", 0.0)
	tscfg.patience = patience
	tscfg.check_interval = args.check_interval
	tscfg.magnitude_aware_patience = args.magnitude_aware_patience
	return tscfg


def _stage_header(idx: int, name: str, gens: int, patience: int, spec: ControllerSpec):
	bar = "=" * 72
	print(f"\n{bar}\n  STAGE {idx}: {name} ({gens} gens, patience {patience})\n{bar}")
	print(f"  seed-spec: state_neurons={spec.state_neurons}, "
	      f"output_neurons={spec.num_motors * spec.levels_per_motor}, "
	      f"state_bits={spec.state_bits_per_neuron}, "
	      f"output_bits={spec.output_bits_per_neuron}, "
	      f"levels={spec.levels_per_motor}")


def _log_split_pressure(res, label: str):
	"""Telemetry (22/06): surface the splitting trainer's integral-counter bottleneck.
	Reads g.pressure = (split_saturation, split_wish_bits) stamped at evaluation
	(evaluator.py). Disambiguates why the controller plateaus at a steady-state offset:
	  saturation>0      -> CAPACITY-bound  (separator observed, no free state -> grow state_neurons)  [Hyp1]
	  wish_bits non-empty -> CONNECTIVITY-bound (a state neuron should observe a bit it doesn't)        [Hyp2]
	  both ~0 + offset persists -> SELECTION/REWARD (integrator not selected/rewarded)                  [Hyp3, by elimination]
	"""
	def _p(g):
		p = getattr(g, "pressure", None)
		return (int(p[0]) if p else 0, tuple(p[1]) if (p and len(p) > 1) else ())
	best = getattr(res, "best_genome", None)
	pop = list(getattr(res, "final_population", None) or ([best] if best is not None else []))
	if not pop:
		return
	b_sat, b_wb = _p(best) if best is not None else (0, ())
	sats = [_p(g)[0] for g in pop]
	all_wb = set()
	for g in pop:
		all_wb.update(_p(g)[1])
	n_sat = sum(1 for s in sats if s > 0)
	mean_sat = sum(sats) / len(sats) if sats else 0.0
	print(f"  [split-pressure {label}] best: sn={getattr(best, 'state_neurons', '?')} "
	      f"saturation={b_sat} wish_bits={len(b_wb)}{sorted(b_wb)[:8]}  | "
	      f"pop: {n_sat}/{len(pop)} saturated (mean={mean_sat:.1f} max={max(sats) if sats else 0}) "
	      f"distinct_wish_bits={len(all_wb)}")


def _print_stage_result(idx: int, name: str, res, gens: int, dt: float, ev: ControllerEvaluator):
	"""Re-evaluate the winning genome so we can report the full metric tuple
	(CE, err, stable_rate) using the evaluator that drove the stage."""
	best = res.best_genome
	if best is None:
		print(f"  STAGE {idx} ({name}): NO BEST GENOME (iter={res.iterations_run})")
		return None
	# Pick the right scorer: MEMORY-stage genomes carry cells → score_genomes (no
	# training). Architecture-stage genomes carry no cells → evaluate_batch trains.
	# Residual mode (ec.geometry): ALWAYS score-only (no DAGGER; empty = neutral).
	_residual = getattr(getattr(ev, "episode_config", None), "geometry", None) is not None
	if _residual or getattr(best, "cells", None) is not None:
		m = ev.score_genomes([best])[0]
	else:
		m = ev.evaluate_batch([best])[0]
	sn, on = best.state_neurons, best.output_neurons
	sb, ob = best.state_bits_per_neuron, best.output_bits_per_neuron
	print(f"  STAGE {idx} ({name}) done: gen {res.iterations_run}/{gens}  "
	      f"steady={_steady_str(m)}  err={m.mean_attitude_error_deg:.2f}°  "
	      f"stable={m.acc*100:.1f}%  "
	      f"arch sn={sn} on={on} sb={sb} ob={ob}  ({dt:.0f}s, "
	      f"{dt/max(res.iterations_run,1):.1f}s/gen)")
	_log_split_pressure(res, name)
	return m


def _parse_teacher_list(spec: str, flag: str) -> list[str]:
	"""'lqr,pid' → ['lqr', 'pid']; empty/None → []. Validates names early so a
	typo fails at launch, not genome-eval time."""
	names = [s.strip() for s in (spec or "").split(",") if s.strip()]
	bad = [n for n in names if n not in ("pid", "lqr", "mpc", "lqi", "mpcof")]
	if bad:
		raise SystemExit(f"{flag}: unknown teacher(s) {bad} (choices: pid, lqr, mpc, lqi, mpcof)")
	return names


def _rg_config(args, ec: EpisodeConfig, seed: int) -> RewardGatedConfig:
	"""Reward-gated inner-train config — exposed knobs let the smoke test shrink
	the per-genome training cost (default: full 8 rounds × 24 episodes_per_round).
	None for any flag → upstream default."""
	rg = RewardGatedConfig(seed=seed, episode_config=ec)
	rg.teacher = getattr(args, "teacher", "pid")   # DAGGER expert: pid|lqr|mpc
	# Hybrid teachers (task #11): per-round curriculum + per-episode blend.
	rg.teacher_schedule = _parse_teacher_list(
		getattr(args, "teacher_schedule", ""), "--teacher-schedule")
	rg.teacher_blend = _parse_teacher_list(
		getattr(args, "teacher_blend", ""), "--teacher-blend")
	# Pure BC (19/07/2026): teacher drives the training rollouts (see reward_gated).
	rg.expert_drives = bool(getattr(args, "expert_drives", False))
	if args.rg_rounds is not None:
		rg.num_rounds = args.rg_rounds
	if args.rg_episodes_per_round is not None:
		rg.episodes_per_round = args.rg_episodes_per_round
	if args.rg_eval_episodes is not None:
		rg.eval_episodes = args.rg_eval_episodes
	rg.steps_per_episode = args.steps   # match the outer eval steps for consistency
	rg.progress = False                  # quiet the per-round inner logging
	# 04/08/2026: the beam-search top-k is the one knob that targets the COST DRIVER
	# at sn>0. sn=0 skips the per-motor solve entirely (controller.rs:2323
	# solve_motors=0), which is ~100% of the measured 150x gap, so nothing else moves
	# a stateful cell's runtime much. Exposed to make that measurable; default None
	# leaves RewardGatedConfig's 4 untouched, so every prior cohort reproduces.
	if getattr(args, "topk_per_neuron", None) is not None:
		rg.topk_per_neuron = args.topk_per_neuron
	# L4 magnitude-priority writes (Rust trainer, sn=0 path; defaults = legacy).
	rg.write_priority_err = bool(getattr(args, "write_priority_err", False))
	rg.write_err_floor_deg = float(getattr(args, "write_err_floor", 0.0) or 0.0)
	return rg


def _run_arch_phase(args, ec: EpisodeConfig, spec: ControllerSpec,
                    dimension: OptimizationDimension, gens: int, patience: int,
                    seed: int, initial_population=None,
                    tracker=None, experiment_id=None, fixed_axes=None):
	"""Generic Stage 1-3 driver: build an ArchGAStrategy on the given dimension
	and run optimize(). Returns (result, evaluator, wall_time).

	Seeding is always a FULL population (`initial_population`): the grid's top-K
	seed pool for Stage 1, or the previous stage's carried final population for
	Stages 2-3 — with the winner already at index 0 so it lands in gen-0 elites.
	Single-genome warm-starting was removed (a normal run never begins from one
	genome; the prior winner is inside the carried population by construction)."""
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed,
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes,
	                         seed=seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=args.num_eval_folds)
	# H4 axis curriculum. The 7-phase driver (_run_axis_curriculum) passes a FIXED
	# per-sub-phase mask via fixed_axes → this evaluator trains+scores on exactly
	# those axes for the whole sub-phase. (Legacy single-stage per-gen ramp is kept
	# behind the elif but is no longer used by the 7-phase path.) The held-out
	# evaluator in _maybe_holdout sets neither → always full 3-axis.
	if fixed_axes is not None:
		ev.fixed_axes = fixed_axes
	elif getattr(args, "axis_curriculum", False) and dimension == OptimizationDimension.NEURONS:
		ev.axis_curriculum_gens = gens
	arch_cfg = default_controller_arch_config(spec)
	# Widen the search box to admit the grid winner + room to mutate. The default
	# max_state_neurons is 4·spec.state_neurons; honor the user's grid maximum so
	# the GA can climb past the seed if it likes.
	arch_cfg.max_state_neurons = max(arch_cfg.max_state_neurons,
	                                 4 * max(args.grid_state_neurons))
	# Hard ceiling override (E5.2 curriculum): cap state-neuron growth so a
	# warm-started big seed can't balloon (the default 4·spec would let sn=38
	# grow to 152 → address bits explode → per-genome GPU eval blows up and the
	# NEURONS population goes 0/N-viable under harsh weather). The GA may still
	# shrink/explore below the cap.
	# `is not None` (not truthiness): --max-state-neurons 0 is the single-layer
	# recipe (19/07/2026 promotion) and must clamp the box to sn=0, not be ignored.
	if getattr(args, "max_state_neurons", None) is not None:
		arch_cfg.max_state_neurons = min(arch_cfg.max_state_neurons, int(args.max_state_neurons))
		arch_cfg.min_state_neurons = min(arch_cfg.min_state_neurons, arch_cfg.max_state_neurons)
	# Same ceiling on OUTPUT neurons (= num_motors·levels): output cells are the
	# other half of the per-genome memory that balloons NEURONS into OOM (24/07 the
	# uncapped default 4·levels·q let a NEURONS population thrash the box to 0 free
	# RAM → jetsam). Bounds output-cell growth; the GA may still shrink below it.
	apply_output_neuron_ceiling(args, arch_cfg)
	# Hard floor on state_neurons from the grid (added 30/05/2026 for Plan A v2).
	# Without this, GA mutations can take sn below the grid minimum, undoing the
	# anchor we set when --grid-state-neurons specifies a tight range.
	arch_cfg.min_state_neurons = max(arch_cfg.min_state_neurons,
	                                 min(args.grid_state_neurons))
	# Phase-5c damping: route the CLI gain into the mutation config so saturation
	# pressure grows state measuredly instead of force-growing every offspring.
	arch_cfg.saturation_grow_gain = getattr(args, "saturation_grow_gain", 0.02)
	# Per-genome cell budget (23/07/2026): suppress structural grows once a
	# genome's carried cells reach this — stops the wandering-controller cell
	# balloon (QUAD-dfa OOM loop) at the source.
	arch_cfg.max_cells = getattr(args, "max_cells", 1_000_000_000)
	arch_cfg.strict_cell_budget = bool(getattr(args, "max_cells_strict", False))
	if getattr(args, "strategy", "ga") == "ts":
		# Tabu Search stage (19/07/2026): local search with a tabu list over the
		# same phase-isolated mutation. Cooperative cancel works (the template
		# polls _shutdown_check); the GA-mixin crash-save checkpoint is GA-only.
		# --lamarckian is a GA-strategy kwarg — ignored under TS (loud note).
		if getattr(args, "lamarckian", False):
			print("  [strategy=ts] note: --lamarckian is GA-only; ignored for TS stages")
		tscfg = _build_ts_config(args, gens, patience)
		strat = ControllerArchTSStrategy(spec, dimension, arch_config=arch_cfg,
		                                 ts_config=tscfg, seed=seed, batch_evaluator=ev)
	else:
		gacfg = _build_ga_config(args, gens, patience)
		strat = ControllerArchGAStrategy(spec, dimension, arch_config=arch_cfg,
		                                 ga_config=gacfg, seed=seed, batch_evaluator=ev,
		                                 lamarckian=getattr(args, "lamarckian", False))
	# Dashboard wiring (no-op for the standalone CLI): attach the tracker so the
	# GenericGAStrategy loop auto-fires record_iteration / record_genome_evaluations_batch
	# per generation — including mean_attitude_error_deg — under this stage's experiment.
	if tracker is not None and experiment_id is not None:
		strat.set_tracker(tracker, experiment_id)
	# Wire the shared cooperative-cancel + adaptive crash-save core BEFORE
	# optimize() so any SIGTERM during this stage trips the cooperative path.
	# Stage identity derives from the dimension (NEURONS→1, BITS→2, CONNECTIONS→3).
	_wire_cancel(strat, args, _STAGE_BY_DIM[dimension], dimension.name.lower())
	t = time.time()
	# Lamarckian: route batch eval through write-back (carry cells across gens).
	# Residual mode (Phase 2): NO DAGGER training — score_genomes only (EMPTY
	# memory = neutral residual; the GA evolves connectivity under the composed
	# fitness). --lamarckian is gated off in main() for this mode.
	if getattr(ec, "geometry", None) is not None:
		_batch_fn = ev.score_genomes
		_eval_fn = lambda g: -ev.score_genomes([g])[0].reward
	else:
		_batch_fn = strat._lamarckian_evaluate_batch if getattr(args, "lamarckian", False) else ev.evaluate_batch
		_eval_fn = lambda g: -ev.evaluate_batch([g])[0].reward
	optimize_kwargs = {
		"evaluate_fn": _eval_fn,
		"batch_evaluate_fn": _batch_fn,
	}
	# Seed from the full carried/grid population (winner already at index 0 so it
	# lands in gen-0 elites). Single-genome seeding is gone — the only seed input
	# is a full population (grid top-K for Stage 1, carried pool for Stages 2-3).
	if initial_population is not None:
		optimize_kwargs["initial_population"] = list(initial_population)
	res = strat.optimize(**optimize_kwargs)
	strat._checkpoint_mgr.join()  # flush the last in-stage async crash-save write
	return res, ev, time.time() - t


def _axis_curriculum_schedule(total_gens: int):
	"""The 7-phase combinatorial axis curriculum: master each axis ALONE, then
	each PAIR, then all three. Budget split EQUALLY across the 7 phases (any
	remainder goes to the all-3 finale). Per-phase early-stop (patience ×
	check_interval) lets a converged easy phase bail before its cap. Returns
	[(label, (roll,pitch,yaw) mask, gens), ...]."""
	masks = [
		("roll",            (True,  False, False)),
		("pitch",           (False, True,  False)),
		("yaw",             (False, False, True)),
		("roll+pitch",      (True,  True,  False)),
		("roll+yaw",        (True,  False, True)),
		("pitch+yaw",       (False, True,  True)),
		("roll+pitch+yaw",  (True,  True,  True)),
	]
	per = max(1, total_gens // 7)
	gens = [per] * 6 + [max(1, total_gens - 6 * per)]
	return [(masks[i][0], masks[i][1], gens[i]) for i in range(7)]


def _run_axis_curriculum(args, ec: EpisodeConfig, spec: ControllerSpec,
                         seed: int, initial_population=None,
                         tracker=None, experiment_id=None):
	"""NEURONS stage as a 7-phase combinatorial axis curriculum. Each sub-phase
	runs the NEURONS GA on a FIXED axis mask, warm-started from the previous
	sub-phase's FULL final population (winner prepended → guaranteed in the elite
	slate). Patience resets per sub-phase, so a converged easy phase advances
	instead of early-stopping the whole curriculum. Returns the FINAL (all-3)
	sub-phase's (result, evaluator, total_wall) — so downstream stages + the
	held-out report seed from the genome scored on the real 3-axis problem."""
	schedule = _axis_curriculum_schedule(args.neurons_gens)
	carried_pop = initial_population
	last_res, last_ev, total_dt = None, None, 0.0
	for i, (label, mask, gens) in enumerate(schedule):
		bar = "-" * 72
		print(f"\n{bar}\n  STAGE 1: NEURONS [{i + 1}/7] axes={label} "
		      f"({gens} gens, patience {args.neurons_patience})\n{bar}", flush=True)
		res, ev, dt = _run_arch_phase(
			args, ec, spec, OptimizationDimension.NEURONS, gens, args.neurons_patience,
			seed, initial_population=carried_pop,
			tracker=tracker, experiment_id=experiment_id, fixed_axes=mask)
		total_dt += dt
		last_res, last_ev = res, ev
		if getattr(res, "final_population", None):
			carried_pop = res.final_population  # carry the WHOLE pool (diversity)
	return last_res, last_ev, total_dt


def _difficulty_curriculum_schedule(total_gens: int, n_phases: int = 5, d_start: float = 0.2):
	"""Difficulty curriculum (the WNN-correct 'easier first'): ramp the initial-
	condition MAGNITUDE (tilt, body-rate, yaw, yaw-rate) from d_start×full → full
	over n_phases, ALL 3 axes throughout. Unlike the axis curriculum, the easy
	phases' addresses are a SUBSET of the full region (hover is the centre of the
	perturbed distribution) → cells transfer; and low magnitude = fewer distinct
	addresses visited more often = denser, more confident cell fills. Budget split
	equally; remainder to the full-difficulty finale. Returns [(label, d, gens)...]."""
	n_phases = max(1, n_phases)
	if n_phases > 1:
		ds = [d_start + (1.0 - d_start) * (i / (n_phases - 1)) for i in range(n_phases)]
	else:
		ds = [1.0]
	per = max(1, total_gens // n_phases)
	gens = [per] * (n_phases - 1) + [max(1, total_gens - (n_phases - 1) * per)]
	return [(f"d={ds[i]:.2f}", ds[i], gens[i]) for i in range(n_phases)]


def _scaled_ec(ec: EpisodeConfig, d: float) -> EpisodeConfig:
	"""EpisodeConfig with the initial-condition bounds scaled by difficulty `d`
	(dt / steps unchanged)."""
	import dataclasses
	return dataclasses.replace(
		ec,
		max_initial_tilt_rad=ec.max_initial_tilt_rad * d,
		max_initial_yaw_rad=ec.max_initial_yaw_rad * d,
		max_initial_body_rate=ec.max_initial_body_rate * d,
		max_initial_yaw_rate=ec.max_initial_yaw_rate * d,
	)


def _run_difficulty_curriculum(args, ec: EpisodeConfig, spec: ControllerSpec,
                               seed: int, initial_population=None,
                               tracker=None, experiment_id=None):
	"""NEURONS stage as a DIFFICULTY curriculum — full 3-axis throughout, IC
	magnitude ramps d_start×full → full. Each phase warm-starts from the previous
	phase's full population (winner pinned). The caller reports the held-out on the
	FULL ec (the real task), so this never leaks an easy-phase score."""
	import math as _math
	n_phases = getattr(args, "difficulty_phases", 5)
	d_start = getattr(args, "difficulty_start", 0.2)
	schedule = _difficulty_curriculum_schedule(args.neurons_gens, n_phases, d_start)
	carried_pop = initial_population
	last_res, last_ev, total_dt = None, None, 0.0
	for i, (label, d, gens) in enumerate(schedule):
		ec_d = _scaled_ec(ec, d)
		bar = "-" * 72
		print(f"\n{bar}\n  STAGE 1: NEURONS [{i + 1}/{len(schedule)}] difficulty={label} "
		      f"(tilt≤{_math.degrees(ec_d.max_initial_tilt_rad):.1f}° rate≤{ec_d.max_initial_body_rate:.2f}, "
		      f"{gens} gens, patience {args.neurons_patience})\n{bar}", flush=True)
		res, ev, dt = _run_arch_phase(
			args, ec_d, spec, OptimizationDimension.NEURONS, gens, args.neurons_patience,
			seed, initial_population=carried_pop,
			tracker=tracker, experiment_id=experiment_id)
		total_dt += dt
		last_res, last_ev = res, ev
		if getattr(res, "final_population", None):
			carried_pop = res.final_population
	return last_res, last_ev, total_dt


def _phase_stable(ev, best_genome) -> float:
	"""Mastery signal: the phase winner's stable fraction at the phase's difficulty
	(re-scored on the phase evaluator — same as _print_stage_result)."""
	if best_genome is None:
		return 0.0
	_residual = getattr(getattr(ev, "episode_config", None), "geometry", None) is not None
	m = (ev.score_genomes([best_genome])[0]
	     if _residual or getattr(best_genome, "cells", None) is not None
	     else ev.evaluate_batch([best_genome])[0])
	return float(m.acc)


def _report_thresholds(args, ec, spec, report_seed: int, train_seed: int, use_score: bool):
	"""Thresholds for a REPORT-ONLY scoring pass.

	Thresholds are NOT an output knob — they are the per-feature thermometer
	cut-points for the INPUT sensors, and connections + thresholds together decide
	WHICH ADDRESS each neuron reads (evaluator.py:10-16). They are part of the
	address function.

	A genome carrying trained cells had those cells WRITTEN at addresses computed
	under the TRAIN-seed thresholds (fit at :546 / :887 with seed=<train seed>).
	Refitting on the report seed re-quantizes the inputs, so the same physical state
	maps to a different address and the trained memory is read where nothing was
	written — rebuilding a hash function after inserting the keys. Measured on frozen
	winners replayed over 5 report seeds (01/08/2026, docs/threshold_misalignment_finding.md):
	    1layer_9feat_BINARY_s31337003  48.0+-13.8  ->  86.8+-1.7
	    dfa_9feat_BINARY_s31337003     67.0+- 6.2  ->  87.6+-1.9
	Variance collapsing WHILE the mean rises is a mismatch being removed, not leakage
	— and the train-seed fit has zero contact with the test draw, so it is strictly
	the more conservative variant.

	Only the SCORE-ONLY path is affected. `evaluate_batch` trains fresh at eval time
	under whatever thresholds it is handed, so refitting per report seed is correct
	there and is left alone.

	UNCONDITIONAL since 03/08/2026 (Luiz). This was a --holdout-fixed-thresholds flag,
	default-OFF on 01/08 and default-ON earlier today. Both were wrong for the same
	reason: a flag implies a choice, and there is no run that wants to read a trained
	memory at addresses it was never written to. The old default cost 26 dfa1l cells
	reporting 0-27% stable for architectures that measure 98-100% once aligned.

	There is no escape hatch, deliberately. Reproducing a pre-fix published number is
	a git operation — check out the commit that produced it — not a flag that lets
	today's code emit yesterday's bug. And the both-ways COMPARISON that documents the
	gap lives in scripts/rescore_winners.py, which fits both threshold sets itself and
	never needed this flag.

	Still scoped to the score-only path: evaluate_batch trains fresh under whatever
	thresholds it is handed, so refitting per report seed is correct there.
	"""
	seed = train_seed if use_score else report_seed
	return fit_thresholds_from_pid_rollouts(
		spec, num_episodes=10, seed=seed,
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))


def _shell_holdout_compact(args, ec_eval: EpisodeConfig, spec: ControllerSpec,
                           best_genome, seed_list, train_seed: int):
	"""REPORT-ONLY held-out for one adaptive shell. Re-score the during-search WINNER
	(winner only — no pop sample, to keep the per-shell cost low) on each fresh, UNSEEN
	report-seed at the `ec_eval` difficulty, aggregate mean±std. NEVER gates advance/
	regress: gating on it would make it not held-out (it would leak into selection).
	Returns (mean_stable_pct, sd_pp, mean_err_deg, sd_err_deg) or None."""
	import statistics
	if best_genome is None or not seed_list:
		return None
	rep_eps = getattr(args, "report_episodes", None) or args.eval_episodes
	use_score = (getattr(best_genome, "cells", None) is not None
	             or getattr(ec_eval, "geometry", None) is not None)  # residual: score-only
	stbs, errs = [], []
	for rs in seed_list:
		if rs == train_seed:
			continue  # shares the train seed → not held-out
		thresholds = _report_thresholds(args, ec_eval, spec, rs, train_seed, use_score)
		ev = ControllerEvaluator(spec, num_eval_episodes=rep_eps, seed=rs,
		                         episode_config=ec_eval, thresholds=thresholds,
		                         rg_config=_rg_config(args, ec_eval, rs),
		                         max_train_workers=args.train_workers)
		m = (ev.score_genomes([best_genome]) if use_score else ev.evaluate_batch([best_genome]))[0]
		stbs.append(m.acc * 100.0); errs.append(m.mean_attitude_error_deg)
	if not stbs:
		return None
	sd = lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0
	return statistics.mean(stbs), sd(stbs), statistics.mean(errs), sd(errs)


def _run_adaptive_difficulty_curriculum(args, ec: EpisodeConfig, spec: ControllerSpec,
                                        seed: int, initial_population=None,
                                        tracker=None, experiment_id=None):
	"""Mastery-gated difficulty curriculum with BACKTRACKING (user 21/06). Start at
	d_start; each step runs `dwell_gens` of NEURONS at the current difficulty and reads
	the winner's stable% there. Mastered (>= mastery_threshold) → advance d += step
	toward 1.0; not mastered → regress d -= step to consolidate the easier shell, then
	re-approach (re-attempts accumulate budget on the starving shell). Bounded
	[d_start, 1.0]; total gens capped at neurons_gens. Anti-oscillation: if a level
	fails `max_attempts` times it's the competence frontier → stop. Warm-starts every
	step. Held-out (caller) is always at FULL difficulty."""
	import math as _math
	step   = getattr(args, "difficulty_step", 0.1)
	d_start = getattr(args, "difficulty_start", 0.2)
	thresh = getattr(args, "mastery_threshold", 0.95)
	dwell  = max(1, getattr(args, "dwell_gens", 5))
	max_attempts = max(1, getattr(args, "max_attempts", 4))
	budget = args.neurons_gens
	d, spent = d_start, 0
	carried_pop = initial_population
	last_res, last_ev, total_dt = None, None, 0.0
	attempts: dict = {}
	while spent < budget:
		gens = min(dwell, budget - spent)
		ec_d = _scaled_ec(ec, d)
		bar = "-" * 72
		print(f"\n{bar}\n  STAGE 1: NEURONS [adaptive d={d:.2f}] "
		      f"(tilt≤{_math.degrees(ec_d.max_initial_tilt_rad):.1f}° rate≤{ec_d.max_initial_body_rate:.2f}, "
		      f"{gens} gens, spent {spent}/{budget})\n{bar}", flush=True)
		res, ev, dt = _run_arch_phase(
			args, ec_d, spec, OptimizationDimension.NEURONS, gens, args.neurons_patience,
			seed, initial_population=carried_pop,
			tracker=tracker, experiment_id=experiment_id)
		spent += gens; total_dt += dt; last_res, last_ev = res, ev
		if getattr(res, "final_population", None):
			carried_pop = res.final_population
		stable = _phase_stable(ev, getattr(res, "best_genome", None))
		k = round(d, 2); attempts[k] = attempts.get(k, 0) + 1
		mastered = stable >= thresh
		print(f"    -> d={d:.2f} stable={stable*100:.1f}% (threshold {thresh*100:.0f}%, "
		      f"mastered={mastered}, attempt {attempts[k]}/{max_attempts})", flush=True)
		_log_split_pressure(res, f"d={d:.2f}/a{attempts[k]}")
		# REPORT-ONLY held-out (never gates — gating would un-hold-out it; user 22/06).
		# TEST @d: the honest mirror of the in-search mastery number (gap = overfit).
		# TRANSFER @1.0: the same winner on the full task — a transfer-curve point.
		if getattr(args, "holdout_per_shell", False):
			rseeds = (getattr(args, "report_seeds", None)
			          or ([args.report_seed] if getattr(args, "report_seed", None) is not None else None))
			bg = getattr(res, "best_genome", None)
			if rseeds and bg is not None:
				ht = _shell_holdout_compact(args, ec_d, spec, bg, rseeds, seed)
				if ht is not None:
					print(f"    [held-out TEST @d={d:.2f}] (unseen, NOT gated): "
					      f"stable={ht[0]:.1f}±{ht[1]:.1f}%  err={ht[2]:.2f}±{ht[3]:.2f}°  "
					      f"| overfit gap (in-search−test)={stable*100-ht[0]:+.1f}pp", flush=True)
				hv = _shell_holdout_compact(args, ec, spec, bg, rseeds, seed)
				if hv is not None:
					print(f"    [held-out TRANSFER @d=1.00] (unseen): "
					      f"stable={hv[0]:.1f}±{hv[1]:.1f}%  err={hv[2]:.2f}±{hv[3]:.2f}°", flush=True)
		if mastered:
			if d >= 1.0 - 1e-9:
				print("  [adaptive] mastered FULL difficulty (d=1.0) — done.", flush=True)
				break
			d = min(1.0, round(d + step, 2))
		else:
			if attempts[k] >= max_attempts:
				print(f"  [adaptive] competence frontier at d={d:.2f} "
				      f"({max_attempts} attempts, still <{thresh*100:.0f}%) — stopping.", flush=True)
				break
			if d > d_start + 1e-9:
				d = max(d_start, round(d - step, 2))  # regress to consolidate, then re-approach
	return last_res, last_ev, total_dt


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
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed,
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))
	mem_eps = getattr(args, "memory_eval_episodes", None) or args.eval_episodes
	ev = ControllerEvaluator(spec, num_eval_episodes=mem_eps,
	                         seed=seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=args.num_eval_folds)
	arch_cfg = default_controller_arch_config(spec)
	arch_cfg.max_state_neurons = max(arch_cfg.max_state_neurons,
	                                 4 * max(args.grid_state_neurons))
	# Hard ceiling override (E5.2 curriculum): cap state-neuron growth so a
	# warm-started big seed can't balloon (the default 4·spec would let sn=38
	# grow to 152 → address bits explode → per-genome GPU eval blows up and the
	# NEURONS population goes 0/N-viable under harsh weather). The GA may still
	# shrink/explore below the cap.
	# `is not None` (not truthiness): --max-state-neurons 0 is the single-layer
	# recipe (19/07/2026 promotion) and must clamp the box to sn=0, not be ignored.
	if getattr(args, "max_state_neurons", None) is not None:
		arch_cfg.max_state_neurons = min(arch_cfg.max_state_neurons, int(args.max_state_neurons))
		arch_cfg.min_state_neurons = min(arch_cfg.min_state_neurons, arch_cfg.max_state_neurons)
	# Same ceiling on OUTPUT neurons (the other OOM-driving cell layer — see _run_arch_phase).
	apply_output_neuron_ceiling(args, arch_cfg)
	arch_cfg.min_state_neurons = max(arch_cfg.min_state_neurons,
	                                 min(args.grid_state_neurons))
	# Phase-5c damping: route the CLI gain into the mutation config so saturation
	# pressure grows state measuredly instead of force-growing every offspring.
	arch_cfg.saturation_grow_gain = getattr(args, "saturation_grow_gain", 0.02)
	# Per-genome cell budget (23/07/2026): suppress structural grows once a
	# genome's carried cells reach this — stops the wandering-controller cell
	# balloon (QUAD-dfa OOM loop) at the source.
	arch_cfg.max_cells = getattr(args, "max_cells", 1_000_000_000)
	arch_cfg.strict_cell_budget = bool(getattr(args, "max_cells_strict", False))
	if getattr(args, "strategy", "ga") == "ts":
		# TS over cell VALUES (fixed arch): tabu = >50% overlap of changed cells.
		tscfg = _build_ts_config(args, gens, patience)
		strat = ControllerMemoryTSStrategy(
			spec, arch_config=arch_cfg, ts_config=tscfg,
			seed=seed, batch_evaluator=ev, thresholds=thresholds,
			record_episodes=args.universe_episodes, record_steps=args.steps,
		)
	else:
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
	# Shared cooperative-cancel + crash-save core (mirrors arch_phase). MEMORY is
	# always Stage 4.
	_wire_cancel(strat, args, 4, "memory")
	t = time.time()
	# MEMORY paradigm: cells ARE the genome → score_genomes (no training).
	optimize_kwargs = dict(
		evaluate_fn=lambda g: -ev.score_genomes([g])[0].reward,
		batch_evaluate_fn=ev.score_genomes,
	)
	if initial_population is not None:
		optimize_kwargs["initial_population"] = list(initial_population)
	res = strat.optimize(**optimize_kwargs)
	strat._checkpoint_mgr.join()  # flush the last in-stage async crash-save write
	return res, ev, time.time() - t


def _save_winner(path: str, args, spec: ControllerSpec,
                 best_genome, final_population, metrics,
                 stage_num=None, stage_name=None, async_save: bool = False) -> None:
	"""Persist the WINNER + the entire FINAL POPULATION + spec + provenance to
	PATH (schema-2 yaml.gz, cells packed; no pickle).

	Used by Plan A → Plan B chaining: Plan B (run_memory_refinement.py) loads
	the full population as `initial_population=` for the memory-only refinement
	GA — strictly stronger than seeding with just the winner because the evolved
	genomes carry the search's accumulated diversity.

	`stage_num`/`stage_name` are stamped INTO the payload so a stage checkpoint
	is self-identifying for --resume (no load-then-resave annotation needed).
	`async_save=True` writes on a background thread (between-stage dumps are
	resume-only, off the next stage's critical path); the thread is tracked in
	_PENDING_SAVES and joined at run end."""
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
			"steady": args.fit_weight_steady,
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
	if stage_num is not None:
		payload["stage_num"] = stage_num
	if stage_name is not None:
		payload["stage_name"] = stage_name
	pop_n = len(payload["population"])
	if async_save:
		p, t = _ctl_save_async(path, payload)
		_PENDING_SAVES.append(t)
		print(f"\n[save-winner] writing {p} async  (spec sn={spec.state_neurons} "
		      f"sb={spec.state_bits_per_neuron} ob={spec.output_bits_per_neuron}, "
		      f"population={pop_n} genomes)")
	else:
		p = _ctl_save(Path(path), payload)
		print(f"\n[save-winner] wrote {p}  (spec sn={spec.state_neurons} "
		      f"sb={spec.state_bits_per_neuron} ob={spec.output_bits_per_neuron}, "
		      f"population={pop_n} genomes)")


# -----------------------------------------------------------------------------
# Baselines (PID + reference numbers from prior runs)
# -----------------------------------------------------------------------------

def _pid_baseline(ec: EpisodeConfig, episodes: int, seed: int, folds: int = 5):
	"""Reference-baseline score on the held-out episode set for the final
	summary. Quad: PID via the serial closed loop. Residual mode (ec.geometry):
	the allocator-LQR baseline itself — an all-EMPTY controller composed on it
	scores EXACTLY the baseline (residual ≡ 0), via the production CPU scorer.
	The returned dict carries label='alloc-LQR' so the summary row is honest."""
	geo = getattr(ec, "geometry", None)
	if geo is not None:
		from wnn.control._accel import WnnController, score_controllers_cpu
		from wnn.control.training import sample_ics_flat
		ar = getattr(ec, "alloc_residual", None)
		n = len(geo.rows)
		# ANY arch works: residual ≡ 0 for an EMPTY memory ⇒ score IS the baseline.
		c = WnnController(n, 4, 2, 1, 2, 2, 2, [0.0] * 18, [0] * 4, [0] * (n * 4 * 2))
		q0, w0 = sample_ics_flat(seed, episodes, ec)
		nominal = (ar.nominal_rows if ar is not None and ar.nominal_rows is not None
		           else geo.rows)
		row = score_controllers_cpu(
			[c], q0, w0, episodes, ec.steps_per_episode,
			geometry=[[float(x) for x in r] for r in geo.rows],
			rotor_asym=(None if geo.rotor_asym is None else [float(x) for x in geo.rotor_asym]),
			alloc_rows=[[float(x) for x in r] for r in nominal],
			alloc_q_att=float(ar.q_att) if ar else 12.0,
			alloc_q_rate=float(ar.q_rate) if ar else 1.0,
			alloc_r_ctrl=float(ar.r_ctrl) if ar else 1.0,
			alloc_tau_max=float(ar.tau_max) if ar else 0.144,
			alloc_f_hover=(None if ar is None or ar.f_hover is None else float(ar.f_hover)),
			alloc_lambda=float(ar.pinv_lambda) if ar else 1e-6,
			# PURE allocator baseline: force the residual to 0 (scale=0).
			# CORRECTION 12/07: an EMPTY memory decodes 0.75 (QUAD EMPTY=
			# WEAK_TRUE), NOT 0.5 — composing it adds a +clamp collective
			# offset (≈attitude-neutral on symmetric craft, but +~70% effort).
			# The paper's comparison target is the CLASSICAL allocator, so the
			# baseline row must be the scale=0 rollout.
			residual_scale=0.0,
			residual_clamp=float(ar.clamp) if ar else 0.15,
		)[0]
		return {"stable_rate": row[2], "mean_attitude_error_deg": math.degrees(row[1]),
		        "mean_reward": row[0], "label": "alloc-LQR",
		        "mean_effort": (row[12] if len(row) > 12 else None)}
	# NOT eval_closed_loop_reset: it draws ICs from the RAW seed and redraws motor
	# asymmetry per episode, so it flies episodes no WNN cell ever saw. That printed
	# "vs PID 85.0%" under every cell of this study against a true 90.4±7.5 — the
	# comparison was never on the same aircraft. See classical_baseline's docstring.
	from wnn.control.classical_baseline import HoldoutDraw, pid_metrics
	draw = HoldoutDraw(seed=seed, episodes=episodes,
	                   steps=ec.steps_per_episode, eval_folds=folds)
	return pid_metrics(ec, draw)


def _maybe_holdout(args, ec, spec, res, seeds, label: str):
	"""Per-stage held-out (REPORT ONLY) — fires after each stage if --report-seed is set,
	so we see the held-out N→B→C→M trajectory, not just the final. Never feeds selection.

	Returns the held-out winner metric (with .mean_attitude_error_deg / .acc / .fitness)
	so a caller (the dashboard flow_runner) can record it on the stage's experiment;
	returns None when no report-seed is set or the stage was skipped/failed."""
	if res is None or res.best_genome is None:
		return None
	# Multi-seed (--report-seeds) takes precedence over the single --report-seed:
	# score the winner on EACH fresh seed and aggregate mean±std (robust to the
	# documented single-seed controller-eval variance; see project_controller_eval_variance).
	report_seeds = getattr(args, "report_seeds", None)
	seed_list = report_seeds if report_seeds else (
		[args.report_seed] if getattr(args, "report_seed", None) is not None else None)
	if not seed_list:
		return None
	try:
		results = []
		for rs in seed_list:
			ds = _holdout_report(args, ec, spec, res.best_genome, res.final_population,
			                     rs, seeds.train, stage_label=label)
			if ds is not None:
				results.append(ds)
		if not results:
			return None
		if len(results) == 1:
			return results[0]
		import statistics
		from types import SimpleNamespace
		stbs = [r.acc * 100 for r in results]
		errs = [r.mean_attitude_error_deg for r in results]
		fits = [r.fitness for r in results]
		stys = [getattr(r, "mean_steady_error_deg", None) for r in results]
		stys = [s for s in stys if s is not None]
		mean = statistics.mean
		sd = lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0
		steady_str = f"  steady={mean(stys):.2f}±{sd(stys):.2f}°" if stys else ""
		print(f"  [report-seeds] {label} MULTI-SEED held-out ({len(results)} seeds {seed_list}): "
		      f"stable={mean(stbs):.1f}±{sd(stbs):.1f}%  err={mean(errs):.2f}±{sd(errs):.2f}°{steady_str}")
		# Return the seed-mean as the stage held-out (so downstream recording uses the robust number).
		return SimpleNamespace(acc=mean(stbs) / 100.0, mean_attitude_error_deg=mean(errs), fitness=mean(fits),
		                       mean_steady_error_deg=(mean(stys) if stys else None))
	except Exception as e:
		print(f"  [report-seed] {label} held-out failed: {e}")
		return None


def _select_headline_stage(args, ec: EpisodeConfig, seeds, stage_entries,
                           stage_holdouts: dict) -> str | None:
	"""Publish EVERY stage's held-out triple; headline the stage chosen on `seeds.val`.

	Why this exists (08/08/2026). The run's reported answer used to be hardcoded to
	the MEMORY stage (`--save-winner [-1]`). Across 8 paired runs MEMORY beat NEURONS
	only 2/8, and L4 showed the stages landing within ~0.05° of each other — so a
	fixed stage is arbitrary, and picking the best stage AFTER seeing the report seeds
	is best-of-N inflation (E[max] > max E). Both are fixed by selecting on a draw
	that is disjoint from BOTH the search folds and the report seeds:

	    search folds -> train   |   seeds.val -> picks the stage   |   report seeds -> published

	Selection uses the run's OWN fitness calculator (lower = better, as in the GA), not
	steady — steady is the metric we publish, and selecting on the published metric is
	the bias this function exists to avoid. Nothing is hidden: every stage's report-seed
	triple is printed regardless of which one wins.

	`stage_entries` is [(label, spec, res), ...]. Returns the winning label (or None)."""
	scored = []
	for label, spec, res in stage_entries:
		if res is None or getattr(res, "best_genome", None) is None:
			continue
		try:
			vm = _holdout_report(args, ec, spec, res.best_genome, res.final_population,
			                     seeds.val, seeds.train, stage_label=f"{label}-VAL")
		except Exception as e:
			print(f"  [stage-select] {label}: val scoring failed ({e}) — excluded from selection")
			continue
		if vm is not None:
			scored.append((label, getattr(vm, "fitness", None)))
	print("\n" + "=" * 72)
	print("  STAGE TABLE — every stage published; headline = val-selected")
	print("=" * 72)
	valid = [(l, f) for l, f in scored if f is not None]
	winner = min(valid, key=lambda t: t[1])[0] if valid else None
	for label, _spec, _res in stage_entries:
		ho = stage_holdouts.get(label.upper())
		fit = dict(scored).get(label)
		triple = ("stable=%.1f%% err=%.2f° steady=%s" % (
			ho.acc * 100, ho.mean_attitude_error_deg,
			("%.2f°" % ho.mean_steady_error_deg) if getattr(ho, "mean_steady_error_deg", None) is not None else "n/a")
			) if ho is not None else "(no held-out)"
		mark = "  <- HEADLINE (val-selected)" if label == winner else ""
		fit_s = ("%.4f" % fit) if fit is not None else "n/a"
		print(f"  {label:<10} {triple:<48} val_fit={fit_s}{mark}")
	if winner is None:
		print("  [stage-select] no stage could be scored on val — headline falls back to MEMORY")
		return None
	wh = stage_holdouts.get(winner.upper())
	print(f"  [stage-select] HEADLINE stage={winner} (chosen on val seed {seeds.val}, "
	      f"fitness {dict(scored)[winner]:.4f}; NEVER on the report seeds)")
	if wh is not None:
		print(f"  [stage-select] HEADLINE held-out: stable={wh.acc * 100:.1f}% "
		      f"err={wh.mean_attitude_error_deg:.2f}° steady="
		      + (("%.2f°" % wh.mean_steady_error_deg)
		         if getattr(wh, "mean_steady_error_deg", None) is not None else "n/a"))
	return winner


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
	# Score-only when the winner already carries trained cells (or the residual path);
	# that is exactly the case where refitting thresholds on the report seed would
	# misalign the address function — see _report_thresholds.
	_use_score = (getattr(best_genome, "cells", None) is not None
	              or getattr(ec, "geometry", None) is not None)
	thresholds = _report_thresholds(args, ec, spec, report_seed, train_seed, _use_score)
	# Held-out episode count decoupled from the GA's --eval-episodes (10/06/2026):
	# the search eval runs every generation (cost ∝ episodes), but the held-out is
	# scored ONCE per stage — so it can afford many more episodes to de-quantize
	# the reported stable% (8 eps = 12.5pp steps; 50 eps = 2pp).
	rep_eps = getattr(args, "report_episodes", None) or args.eval_episodes
	ev = ControllerEvaluator(spec, num_eval_episodes=rep_eps,
	                         seed=report_seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, report_seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=getattr(args, "num_eval_folds", 5))  # K=1 is NEVER an option
	pop = list(final_population) if final_population else [best_genome]
	# A (13/06): the held-out RESULT is pop[0] (the during-search winner); the
	# rest are scored ONLY for a descriptive population stat that is explicitly
	# NOT selected (leak-guard). Scoring the whole population is ~Nx waste at the
	# big report-episode count, so cap to the winner + a deterministic sample.
	# --holdout-pop-sample 0 restores full-population descriptive stats.
	ho_sample = getattr(args, "holdout_pop_sample", 8)
	if ho_sample and ho_sample > 0 and len(pop) > ho_sample:
		import random
		rng = random.Random(report_seed)
		pop = [pop[0]] + rng.sample(pop[1:], ho_sample - 1)  # winner FIRST (= RESULT)
	# MEMORY-stage / Lamarckian winners carry cells → score-only on the fresh seed (a
	# TRUE held-out: the cells were trained on train_seed during the search). Residual
	# mode (ec.geometry): ALWAYS score-only (no DAGGER; empty = neutral).
	use_score = (getattr(best_genome, "cells", None) is not None
	             or getattr(ec, "geometry", None) is not None)
	if use_score:
		metrics = ev.score_genomes(pop)
	else:
		# Arch-only winner (e.g. a raw grid winner — no cells). Do NOT retrain on the
		# report seed (train==test, not a held-out) and NEVER at K=1: the ctor default
		# num_eval_folds=1 badly undertrains → the controller diverges (measured TERNARY
		# grid winner 90%→8% stable / 24.7°, a pure K=1 artifact — 15/07/2026). Instead
		# train on the TRAIN seed EXACTLY as the search did (K=num_eval_folds accumulate),
		# THEN score those cells on the fresh report seed → train-on-A → score-on-B.
		train_thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=train_seed,
			geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))
		train_ev = ControllerEvaluator(spec, num_eval_episodes=rep_eps, seed=train_seed,
		                               episode_config=ec, thresholds=train_thr,
		                               rg_config=_rg_config(args, ec, train_seed),
		                               max_train_workers=args.train_workers,
		                               num_eval_folds=getattr(args, "num_eval_folds", 5))
		for _g in pop:
			_g.cells = None
		train_ev._evaluate_core(pop, write_back=True)   # stamp train-seed cells (K-fold accumulate)
		metrics = ev.score_genomes(pop)                 # held-out: score on report seed, no retrain
	stables = [m.acc * 100 for m in metrics]
	errs = [m.mean_attitude_error_deg for m in metrics]
	ds = metrics[0]            # final_population[0] = the during-search winner = THE RESULT
	pop_max = max(stables)     # descriptive only — NOT selected (would leak)
	pid_m = _pid_baseline(ec, rep_eps, report_seed, getattr(args, "num_eval_folds", 5))
	def _ms(xs):
		return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)
	ms_s, ms_e = _ms(stables), _ms(errs)
	bar = "=" * 72
	print(f"\n{bar}\n  HELD-OUT REPORT [{stage_label}] (report-only) — population ({len(pop)} genomes) on "
	      f"FRESH seed {report_seed}, train/select seed {train_seed}\n{bar}")
	# Steadiness (Luiz 08/07): steady-state error + monotonicity violations are
	# computed by the eval + weighted in the fitness — surface them here too.
	_sty = getattr(ds, "mean_steady_error_deg", None)
	_mono = getattr(ds, "mono_violations_total", None)
	_eff = getattr(ds, "mean_effort", None)
	_steady_str = (f"  steady={_sty:.2f}°" if _sty is not None else "") + \
	              (f"  mono_viol={_mono:.0f}" if _mono is not None else "") + \
	              (f"  effort={_eff:.3f}" if _eff is not None else "")
	print(f"  RESULT — during-search winner (held-out):  stable={ds.acc*100:.1f}%  "
	      f"err={ds.mean_attitude_error_deg:.2f}°{_steady_str}  reward={ds.fitness:.2f}")
	print(f"  population (held-out, descriptive):        stable={ms_s[0]:.1f}±{ms_s[1]:.1f}%  "
	      f"err={ms_e[0]:.2f}±{ms_e[1]:.2f}°   (pop max stable={pop_max:.1f}% — NOT selected, would leak)")
	_bl = pid_m.get("label", "PID") if isinstance(pid_m, dict) else "PID"
	_bl_eff = pid_m.get("mean_effort") if isinstance(pid_m, dict) else None
	_bl_sty = pid_m.get("mean_steady_error_deg") if isinstance(pid_m, dict) else None
	print(f"  vs {_bl}  (held-out):                        stable={pid_m['stable_rate']*100:.1f}%  "
	      f"err={pid_m['mean_attitude_error_deg']:.2f}°"
	      + (f"  steady={_bl_sty:.2f}°" if _bl_sty is not None else "")
	      + (f"  effort={_bl_eff:.3f}" if _bl_eff is not None else "")
	      + f"   [pool-seeded, fold 0 — same episodes as the WNN row above]")
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
	# Stage identity is stamped into the payload (self-identifying for --resume),
	# so there's no load-then-resave annotation — which used to re-read the whole
	# checkpoint (the 3-min/1.6GB hit pre-packing). Written async: this is a
	# resume-only dump, and the next stage reads its population from memory.
	_save_winner(str(path), args, spec, res.best_genome, res.final_population, metrics,
	             stage_num=stage_num, stage_name=stage_name, async_save=True)


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

	# Arch stages to skip (--skip-stages bits,connections). A skipped stage leaves
	# the carried population + prev_best untouched, so they flow to the next stage.
	skip_stages = {s.strip().lower() for s in (getattr(args, "skip_stages", "") or "").split(",") if s.strip()}

	# Resume planning.
	resume_start_stage = 1
	resume_population  = None
	resume_spec        = None
	resume_warm_genome = None
	resume_mode        = "same"
	if resume_state is not None:
		# stage_num 0 is legitimate ("before Stage 1" — the --seed-winner
		# curriculum path forces it so mode='next' starts at Stage 1/NEURONS).
		# `or 1` would coerce a valid 0 to 1 (0 is falsy), so handle None explicitly.
		_sn = resume_state.get("stage_num")
		dumped_stage = int(_sn) if _sn is not None else 1
		resume_mode = (resume_state.get("resume_mode") or "same").lower()
		if resume_mode == "same":
			resume_start_stage = dumped_stage
			resume_population  = resume_state.get("population") or None
			resume_spec        = resume_state.get("spec")
			resume_warm_genome = resume_state.get("best_genome")
		elif resume_mode == "next":
			resume_start_stage = min(dumped_stage + 1, 4)
			# Carry the FULL dumped population forward: one phase FEEDS the next —
			# the next stage continues evolving the previous stage's whole population,
			# NOT a rebuild from the single winner. best_genome/spec set the seed spec.
			resume_population  = resume_state.get("population") or None
			resume_warm_genome = resume_state.get("best_genome")
			resume_spec        = resume_state.get("spec")  # falls back to dumped spec
		else:
			raise ValueError(f"unknown resume_mode {resume_mode!r}; expected 'same' or 'next'")
		print(f"\n[resume] dumped stage={dumped_stage} mode={resume_mode!r} → starting at "
		      f"stage {resume_start_stage} (pop={len(resume_state.get('population') or [])} genomes)")

	# Stage 0 — grid. Skipped on resume (only its winner_spec matters and the
	# resume's captured `spec` carries that forward).
	if resume_state is None:
		winner_spec, seed_pop0, m0, dt0, _thr = stage0_grid(args, ec, seed)
		stage_results = [("Grid", winner_spec, m0, dt0, grid_point_count(args))]
		# GRID held-out (REPORT ONLY, same contract as every other stage). The grid
		# winner is what you would ship if you stopped before the GA, so it needs the
		# SAME report-seed measurement — without it the Grid row carried during-search
		# numbers while every later row carried held-out ones, and "does the GA earn
		# its ~7000s over a ~1500s grid?" could not even be asked.
		# `seed_population[0]` is documented (stage0_grid) as the fitness-best genome.
		from types import SimpleNamespace as _SNS
		grid_res = _SNS(best_genome=(seed_pop0[0] if seed_pop0 else None),
		                final_population=seed_pop0)
		grid_ho = _maybe_holdout(args, ec, winner_spec, grid_res, seeds, "GRID")
		if grid_ho is not None:
			if stage_holdouts is not None:
				stage_holdouts["GRID"] = grid_ho
			_grid_stage_entry = ("GRID", winner_spec, grid_res)
		else:
			_grid_stage_entry = None
	else:
		seed_pop0 = None
		_grid_stage_entry = None   # resume skips the grid — nothing to score or publish
		winner_spec = resume_spec
		if winner_spec is None:
			raise ValueError("resume_state missing both spec and best_genome — cannot determine winner_spec")
		stage_results = [("Grid (skipped on resume)", winner_spec, None, 0.0, 0)]

	# ---- Stages 1-4 via the shared PhasedOrchestrator ----------------------
	# The controller's phase sequencing lives ONCE in ControllerOrchestrator (on
	# the same PhasedOrchestrator skeleton IDS uses). The hand-rolled Stage 1-4
	# loop / carry / skip / spec-derivation / per-stage checkpoint+holdout that
	# used to be spelled out inline here now lives in orchestrator.run_phase.
	from wnn.control.controller_orchestrator import ControllerOrchestrator
	from wnn.ram.strategies.phased.carry import CarryState

	orch = ControllerOrchestrator(args, ec, seed, seeds, tracker,
	                              base_spec=winner_spec, skip_stages=skip_stages,
	                              eid_fn=_eid)
	# Seed the carry: the starting population + spec for the first stage that runs.
	#   fresh          → grid top-K pool + grid-winner spec
	#   resume "same"  → the dumped stage's population + spec (re-run that stage)
	#   resume "next"  → the dumped population carried forward + spec derived from
	#                    the dumped winner (the next stage continues the pool)
	if resume_state is None:
		start_spec, start_pop = winner_spec, seed_pop0
	elif resume_mode == "next" and resume_warm_genome is not None:
		start_spec, start_pop = _spec_from_best(resume_warm_genome, winner_spec), resume_population
	else:  # "same" — winner_spec IS resume_spec here (set in the resume block)
		start_spec, start_pop = winner_spec, resume_population
	carry = CarryState(genome=resume_warm_genome, population=start_pop,
	                   extra={"base": winner_spec, "spec": start_spec})
	# Resume slices the phase specs to start at resume_start_stage — the controller
	# owns its checkpoints (emergency dump + _save_stage_checkpoint), so we do NOT
	# use the base run_all resume_from (which reloads base per-phase checkpoints).
	specs = orch.phase_specs()[resume_start_stage - 1:]
	orch.run_all(specs, carry)

	# Fold the orchestrator's per-stage held-outs back into the caller's out-dict.
	if stage_holdouts is not None:
		stage_holdouts.update(orch.stage_holdouts)
	res4 = orch.best_result()

	# Publish every stage; headline the val-selected one (see _select_headline_stage).
	# Report-only: this changes NOTHING about the search, the carry, or --save-winner.
	_all_ho = dict(orch.stage_holdouts)
	if stage_holdouts is not None:
		_all_ho.update(stage_holdouts)
	_entries = [e for e in [_grid_stage_entry] if e is not None]
	for _sn, _lbl in ((1, "NEURONS"), (4, "MEMORY")):
		_r = orch.result_for_stage(_sn)
		_row = orch.row_for_stage(_sn)
		if _r is not None and _row is not None:
			_entries.append((_lbl, _row[1], _r))
	if _entries:
		try:
			_select_headline_stage(args, ec, seeds, _entries, _all_ho)
		except Exception as e:
			print(f"  [stage-select] skipped ({e}) — per-stage held-outs above are unaffected")

	# PID baseline on the val seed (the held-out reference).
	pid_m = _pid_baseline(ec, args.eval_episodes, seeds.val,
	                      getattr(args, "num_eval_folds", 5))

	# Assemble the ordered 5-row result [Grid, Neurons, Bits, Connections, Memory].
	# Stages the orchestrator ran (or skipped via --skip-stages) recorded their row;
	# resume-sliced-out stages (< resume_start_stage) get a None-metrics placeholder
	# so the final-summary + --save-winner ([-1] = Memory) degrade gracefully.
	_LABELS = {1: "Neurons", 2: "Bits", 3: "Connections", 4: "Memory"}
	for sn in (1, 2, 3, 4):
		stage_results.append(orch.row_for_stage(sn)
		                     or (_LABELS[sn], winner_spec, None, 0.0, 0))
	# final_population: memory-stage population, sorted by fitness. Used by
	# --save-winner so Plan B can warm-start its GA from Plan A's evolved pool.
	if res4 is None:
		return stage_results, None, None, pid_m
	return stage_results, res4.best_genome, res4.final_population, pid_m


def _print_final_summary(args, stage_results, best_final, pid_m, total_dt: float):
	"""Final report block: per-stage outcomes + baselines + reference numbers."""
	bar = "=" * 72
	print(f"\n{bar}\n  PHASED-GA RESULT (5 stages, target "
	      f"{args.neurons_gens+args.bits_gens+args.conns_gens+args.memory_gens} GA gens "
	      f"+ {grid_point_count(args)} grid)\n{bar}")
	# Stage rows.
	labels = ["Grid", "Neurons", "Bits", "Conns", "Memory"]
	target_gens = [None, args.neurons_gens, args.bits_gens, args.conns_gens, args.memory_gens]
	for (label, spec, m, dt, iters), target in zip(stage_results, target_gens):
		sn = spec.state_neurons
		b_s = spec.state_bits_per_neuron
		b_o = spec.output_bits_per_neuron
		if label == "Grid":
			print(f"  Stage 0 (Grid):    winner sn={sn} b={b_s} levels={spec.levels_per_motor}  "
			      f"steady={_steady_str(m)}  err={m.mean_attitude_error_deg:.2f}°  "
			      f"stable={m.acc*100:.1f}%  ({dt:.0f}s)")
		else:
			steady = "n/a" if m is None else _steady_str(m)
			err = "n/a" if m is None else f"{m.mean_attitude_error_deg:.2f}°"
			stab = "n/a" if m is None else f"{m.acc*100:.1f}%"
			gens_str = f"{iters}/{target}" if target else f"{iters}"
			print(f"  Stage   ({label:<11}): gen {gens_str:<10}  "
			      f"steady={steady:<8}  err={err:<8} stable={stab:<6} "
			      f"arch sn={sn} sb={b_s} ob={b_o} on={best_final.output_neurons if best_final else '?'}  ({dt:.0f}s)")

	# Final winner (= memory stage).
	final_label, final_spec, final_m, _, _ = stage_results[-1]
	print("  " + "─" * 60)
	if final_m is not None:
		print(f"  FINAL: err={final_m.mean_attitude_error_deg:.2f}°  "
		      f"stable={final_m.acc*100:.0f}%  reward={final_m.fitness:.2f}")
	# Baselines.
	_bl = pid_m.get("label", "PID") if isinstance(pid_m, dict) else "PID"
	# reward is None for the pool-seeded scorer (it returns stability/error/steady,
	# and a fabricated reward in a comparison row is worse than an absent one).
	_rw = pid_m.get("mean_reward")
	print(f"  vs {_bl}:  {pid_m['mean_attitude_error_deg']:.2f}° / "
	      f"{pid_m['stable_rate']*100:.0f}% / "
	      f"{'—' if _rw is None else format(_rw, '.2f')}")
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
	ap.add_argument("--grid-output-neurons", type=int, nargs="+", default=None,
	                help="output_neurons axis for Stage 0 grid (third axis). "
	                     "output_neurons = num_motors·levels_per_motor, so these ARE "
	                     "PWM decode resolutions: 64 96 128 = 16/24/32 levels on a quad. "
	                     "Each value must be a multiple of num_motors. Omitted (default) "
	                     "= one point at num_motors·--levels, i.e. today's behaviour.")
	ap.add_argument("--output-decode", choices=["cumulative", "antagonist"], default=None,
	                help="Output decode TOPOLOGY, independent of --memory-mode. Omitted "
	                     "(default) = the mode's historical choice, so every prior cohort "
	                     "reproduces: BINARY→antagonist, everything else→cumulative. "
	                     "'antagonist' splits each motor's levels into excitatory/inhibitory "
	                     "halves decoded 0.5+(ΣE−ΣI)/levels, which puts the untrained neutral "
	                     "at exactly 0.5 with symmetric authority — QUAD's cumulative neutral "
	                     "is 0.75, a 3:1 asymmetry around hover. 'cumulative' is refused for "
	                     "BINARY (its untrained bank would decode to the floor).")
	ap.add_argument("--input-window-k", type=int, default=4,
	                help="Timesteps of sensor history in the address window (default 4). "
	                     "Grows the input POOL linearly (k·num_features·bits_per_feature); "
	                     "the address space 2^(prefix+suffix) is UNCHANGED, so the cost is "
	                     "sampling coverage, not memory — pair a raise with more neurons.")
	ap.add_argument("--grid-top-k", type=int, default=15,
	                help="Seed Stage 1 from the top-K grid architectures (mixed shape, "
	                     "expanded to --pop), not the single winner. Ranked by the "
	                     "controller fitness calculator (not raw CE). Auto-clamps to the "
	                     "number of valid grid points. Default 15 (matches IDS).")
	ap.add_argument("--grid-min-suffix", type=int, default=4,
	                help="minimum sampled-input-bit suffix per neuron — pre-filter "
	                     "grid (sn, b) pairs to skip configurations where "
	                     "(b − 2·sn) < this value. Default 4 ensures each neuron has "
	                     "at least 4 input bits to sample patterns from, not just the "
	                     "forced QSR state prefix. Use 1 to allow any positive suffix.")
	ap.add_argument("--levels", type=int, default=16, help="PWM levels per motor (fixed dim)")
	# Delta-control: output decodes to a per-step PWM delta with a leaky accumulator
	# = a structural integrator (offloads PID's I-term out of the untrained state).
	# Banked 08/06/2026: +5pp stability vs the old hardcoded False. PARTIAL fix.
	ap.add_argument("--delta-control", action=argparse.BooleanOptionalAction, default=True,
	                help="Output decodes to a leaky-accumulator PWM delta (structural integrator). Default ON.")
	ap.add_argument("--delta-leak", type=float, default=0.95,
	                help="Leak on the delta accumulator (1.0=pure integrator/can run away; <1.0 bounds offset). Default 0.95.")
	# L3 (07/08/2026). delta_max was a ControllerSpec field the CLI never passed, so it
	# was UNREACHABLE from a run — not merely unsearched. Together with --delta-leak it
	# sets the smallest sustainable throttle offset, (delta_max/8)/(1-delta_leak), which
	# is 0.25 pwm at the defaults: the actuation-resolution candidate for the hold floor.
	# See docs/hold_floor_levers_spec.md section L3.
	ap.add_argument("--delta-max", type=float, default=0.1,
	                help="Largest single PWM delta the output alphabet can emit; the "
	                     "alphabet step is delta_max/8. With --delta-leak it sets the "
	                     "smallest sustainable offset (delta_max/8)/(1-delta_leak) — "
	                     "0.25 pwm at the defaults. Default 0.1.")
	# L4 — magnitude-priority output writes (07/08/2026). BINARY is last-writer-
	# wins; the legacy backward walk hands contested cells to the window's
	# EARLIEST record, arbitrary w.r.t. error magnitude. Rust trainer, sn=0 only.
	ap.add_argument("--write-priority-err", action="store_true",
	                help="L4 arm A: commit each BPTT window's records in ascending-|err| "
	                     "order so the HIGHEST-attitude-error record writes last and owns "
	                     "contested cells (default: earliest record wins, arbitrarily).")
	ap.add_argument("--write-err-floor", type=float, default=0.0,
	                help="L4 arm B: skip output commits for records with |attitude err| "
	                     "below this floor (degrees) — the near-hover mass cannot "
	                     "overwrite rare large corrections. 0 = off (default).")
	# H2 observation features (Sajus-inspired, attacks the 5° gap = perception/integral,
	# NOT authority per H1). num_features = 9 + tilt_p + tilt_i + 3·peraxis_p + 3·peraxis_i.
	ap.add_argument("--obs-tilt-p", action=argparse.BooleanOptionalAction, default=False,
	                help="H2: add tilt-to-vertical error feature (gravity ref, accel-only). Default OFF.")
	ap.add_argument("--obs-tilt-i", action=argparse.BooleanOptionalAction, default=False,
	                help="H2: add leaky-integral-of-tilt-error feature (the steady-state killer). Default OFF.")
	ap.add_argument("--obs-peraxis-p", action=argparse.BooleanOptionalAction, default=False,
	                help="H2: add per-axis roll/pitch/yaw error features (3). Default OFF.")
	ap.add_argument("--obs-peraxis-i", action=argparse.BooleanOptionalAction, default=False,
	                help="H2: add leaky-integral per-axis error features (3). Default OFF.")
	ap.add_argument("--obs-peraxis-yaw", action=argparse.BooleanOptionalAction, default=True,
	                help="Include yaw in per-axis features (default ON). --no-obs-peraxis-yaw drops "
	                     "yaw → roll+pitch only (gravity-observable; avoids drifting dead-reckoned-yaw poison).")
	ap.add_argument("--obs-pwm", action=argparse.BooleanOptionalAction, default=False,
	                help="Expose the RAW throttle accumulator (current pwm, num_motors feats) as "
	                     "obs — the DIRECT fix for delta's hidden state (∫error was only a proxy). Default OFF.")
	ap.add_argument("--obs-yaw-err", action=argparse.BooleanOptionalAction, default=False,
	                help="Yaw-anchor (Phase A): add a CLEAN scalar yaw-error feature "
	                     "(target_yaw − anchored heading). yaw_heading is seeded to the episode's "
	                     "true initial yaw (from q0) + dt-integrated → absolute yaw ref. Default OFF.")
	ap.add_argument("--obs-yaw-err-i", action=argparse.BooleanOptionalAction, default=False,
	                help="Yaw-anchor: add the leaky integral of the yaw error (1 feature). Default OFF.")
	ap.add_argument("--obs-dhat", action=argparse.BooleanOptionalAction, default=False,
	                help="L1 (06/08/2026): add the mpcof teacher's DISTURBANCE ESTIMATE d̂ as 3 "
	                     "input features (roll/pitch/yaw estimated external angular accel). The "
	                     "observer runs inside the controller from its OWN throttle accumulator "
	                     "and the gyro finite-difference; the plant constant b comes from "
	                     "ram_controller.calibrate_control_gains on --airframe. Motivation: the D2 "
	                     "decomposition showed students are teacher-grade in RECOVERY but hit an "
	                     "absolute HOLD floor set by disturbance observability "
	                     "(docs/hold_floor_levers_spec.md). REQUIRES --airframe. Default OFF.")
	ap.add_argument("--dhat-l-gain", type=float, default=0.05,
	                help="Observer gain for --obs-dhat (mpcof teacher default 0.05). Not searched.")
	ap.add_argument("--feature-balance-ratio", type=float, default=0.0,
	                help="Feature-balance cap: no input feature may capture more than this ratio × "
	                     "the least-wired feature's connection count (e.g. 1.5). Forbids a salient "
	                     "feature dominating the wiring AND floors under-wired ones (fair share). 0/≤1 = off.")
	ap.add_argument("--suffix-coverage", type=float, default=0.0,
	                help="PER-LAYER suffix sizing: set each layer's sampled-suffix width to this fraction "
	                     "of its feature-input span (output=1 frame ⇒ 0.8×80≈64; state=windowed ⇒ capped). "
	                     "Gives more bits/feature so the GA need not starve features. 0 = off (use --grid-bits).")
	ap.add_argument("--suffix-cap", type=int, default=100,
	                help="Max state-suffix width under --suffix-coverage (the state windowed span is large; "
	                     "cap keeps the u64-hashed address sane). Default 100.")
	ap.add_argument("--decouple-outputs", action=argparse.BooleanOptionalAction, default=False,
	                help="H3: output 4 CONTROLS [T, τ_roll, τ_pitch, τ_yaw] mixed to motors instead of "
	                     "4 raw motor PWMs — orthogonal action space, one knob per axis. Default OFF.")
	ap.add_argument("--axis-curriculum", action=argparse.BooleanOptionalAction, default=False,
	                help="H4: ramp the NEURONS-stage episode axes roll → roll+pitch → all over the "
	                     "neurons-gens (warm-start free; held-out stays full 3-axis). Default OFF.")
	ap.add_argument("--difficulty-curriculum", action=argparse.BooleanOptionalAction, default=False,
	                help="H4-v2 (WNN-correct 'easier first'): ramp the NEURONS-stage IC MAGNITUDE "
	                     "(tilt/rate) d_start×full → full over neurons-gens, ALL 3 axes throughout. "
	                     "Easy-phase addresses are a SUBSET of full (hover=centre) so cells transfer "
	                     "(unlike the axis curriculum). Held-out stays full. Default OFF.")
	ap.add_argument("--difficulty-phases", type=int, default=5,
	                help="Difficulty-curriculum phase count (default 5 → d=0.2,0.4,0.6,0.8,1.0).")
	ap.add_argument("--difficulty-start", type=float, default=0.2,
	                help="Starting difficulty as a fraction of full IC magnitude (default 0.2 ≈ tilt1°/rate0.1).")
	ap.add_argument("--bits-per-feature", type=int, default=8,
	                help="Input thermometer resolution per sensor feature (default 8). Higher = finer "
	                     "address resolution → can sense/correct smaller attitude deviations (attacks the "
	                     "~0.94° hover floor + the high-tilt degradation). The encoding-resolution lever.")
	ap.add_argument("--difficulty-adaptive", action=argparse.BooleanOptionalAction, default=False,
	                help="Mastery-gated difficulty curriculum with BACKTRACKING (vs the fixed-phase ramp): "
	                     "advance d+=step when the level is mastered (stable ≥ --mastery-threshold), regress "
	                     "d-=step to consolidate when it isn't, re-approach. Pours budget into the starving "
	                     "shell. Stops at d=1.0 mastered, neurons-gens budget, or a competence frontier.")
	ap.add_argument("--difficulty-step", type=float, default=0.1,
	                help="Adaptive curriculum step (default 0.1).")
	ap.add_argument("--mastery-threshold", type=float, default=0.95,
	                help="Stable fraction counted as 'mastered' in the adaptive curriculum (default 0.95; "
	                     "set 1.0 for strict below-100%%-regresses).")
	ap.add_argument("--dwell-gens", type=int, default=5,
	                help="Gens per adaptive-curriculum mini-phase before re-checking mastery (default 5).")
	ap.add_argument("--max-attempts", type=int, default=4,
	                help="Adaptive curriculum: failures at one difficulty level before declaring it the "
	                     "competence frontier and stopping (anti-oscillation guard, default 4).")
	ap.add_argument("--holdout-per-shell", action="store_true",
	                help="Adaptive curriculum: after each shell, REPORT (never gate) the winner's held-out "
	                     "stable%%/err on unseen --report-seeds — at the shell difficulty d (TEST: the overfit "
	                     "gap vs the in-search mastery number) and at full d=1.0 (TRANSFER: a transfer-curve "
	                     "point). Report-only; never feeds selection (gating would un-hold-out it).")
	ap.add_argument("--integral-leak", type=float, default=0.99,
	                help="H2: leaky-integral decay for the _i obs features (distinct from --delta-leak). Default 0.99.")
	ap.add_argument("--integral-scale", type=float, default=1.0,
	                help="H2: pre-threshold scale for the integral obs features. Default 1.0.")
	# Option A: train the recurrent STATE as a LEARNED integrator (direct thermo-
	# encoded PID-integral target). Sets WNN_STATE_INTEGRAL_TARGET=1 for the Rust
	# trainer. Best paired with SMALL state_neurons (3-9) — the forced prefix is
	# 2·sn, so big sn makes the state memory huge + slow. Off by default.
	ap.add_argument("--state-integral", action="store_true",
	                help="Train the recurrent state as a learned integrator (direct PID-integral target). Use with small --grid-state-neurons.")
	# DAGGER teacher — the expert the WNN imitates during reward-gated training.
	ap.add_argument("--teacher", choices=["pid", "lqr", "mpc", "lqi", "mpcof"], default="pid",
	                help="DAGGER expert: pid (hand-tuned), lqr (optimal linear, continuous CARE), "
	                     "mpc (constrained receding-horizon), lqi (integral-augmented LQR — LQR's "
	                     "optimal gains + PID's integral channel; STATEFUL, rejects constant torque "
	                     "bias, feeds the Option-A integral target), mpcof (offset-free MPC — MPC + "
	                     "input-disturbance observer; STATEFUL, needs the loop's observe() feed). "
	                     "All in Rust (controller/optimal.rs); LQR/MPC are memoryless so no "
	                     "Option-A integral target.")
	ap.add_argument("--expert-drives", action="store_true",
	                help="Pure behavior cloning: the TEACHER's pwm drives the training rollouts "
	                     "(labels unchanged). With --rg-rounds 1 this is classic one-pass BC — "
	                     "the fastest trainer + the covariate-shift baseline. Default off = DAGGER.")
	# Hybrid teachers (both empty = plain --teacher, bit-exact legacy path).
	ap.add_argument("--teacher-schedule", type=str, default="",
	                help="Hybrid curriculum: comma list of per-ROUND teachers (pid|lqr|mpc), e.g. "
	                     "'lqr,lqr,lqr,lqr,pid,pid,pid,pid' (last entry extends past the list). "
	                     "Empty = constant --teacher.")
	ap.add_argument("--teacher-blend", type=str, default="",
	                help="Hybrid blended labels: comma list cycled per-EPISODE within every round, "
	                     "e.g. 'lqr,pid' alternates labels in each gated batch (repeat a name for "
	                     "other ratios: 'lqr,lqr,pid'). Overrides --teacher-schedule when set.")
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
	# Optimizer strategy (19/07/2026 single-layer promotion): GA (default) or Tabu
	# Search — the same phase-isolated mutation, local search + tabu list instead
	# of a population. Applies to every stage (arch phases + MEMORY). TS ignores
	# --lamarckian (GA-only) and the GA-mixin crash-save (cooperative cancel works).
	ap.add_argument("--strategy", type=str, default="ga", choices=("ga", "ts"),
	                help="Per-stage optimizer: ga (population GA, default) or ts (tabu search).")
	ap.add_argument("--crossover-rate", type=float, default=0.5)
	# E1 random immigrants (plan controller_break_90_v2): probability each offspring
	# slot is a FRESH random genome instead of a bred child. Diversity preservation
	# against premature convergence (seed-bimodal 70-90% held-out). 0.0 = off.
	ap.add_argument("--immigrants", type=float, default=0.0,
	                help="Random-immigrant fraction of each generation's offspring (0.0-0.5 sensible; default off).")
	ap.add_argument("--max-output-neurons", type=int, default=None,
	                help="Hard ceiling on output-neuron count (= num_motors·levels_per_motor) "
	                     "in the NEURONS/MEMORY GA. Bounds output-cell memory (the other half "
	                     "of the per-genome footprint that balloons NEURONS into OOM). E.g. 128 "
	                     "= 32 levels × 4 motors. Overrides the default 4·levels·num_motors.")
	ap.add_argument("--max-state-neurons", type=int, default=None,
	                help="Hard ceiling on state-neuron count in the NEURONS/MEMORY GA "
	                     "(overrides the default 4·seed). Caps address-bit growth too "
	                     "(total state_bits = prefix·sn + suffix). Use with --seed-winner "
	                     "so a big warm-start seed can't balloon into GPU-eval blowups.")
	# E3 threshold-density warp: gamma>1 densifies thermometer thresholds near each
	# feature's hover/median region (finer decode where soft-fails settle). 1.0 = off.
	ap.add_argument("--threshold-gamma", type=float, default=1.0,
	                help="Warp thermometer quantiles toward the median (gamma>1 = denser near hover; 1.0 = uniform/off).")
	# Arm R (action-repeat / Sajus frame-skip): decide every Nth physical step,
	# HOLD the PWM in between. Temporal abstraction for the memoryless controller
	# (the 4-frame window then spans 4N steps; jerk drops). 1 = off (bit-identical).
	ap.add_argument("--action-repeat", type=int, default=1,
	                help="Hold each WNN decision for N physical steps (Sajus frame-skip). 1 = every step (default).")
	# ABI 12 granularity ablation (Luiz 12/07/2026): the controller cell format.
	# TERNARY runs empty_value=0.5 (PLN convention); BINARY decodes antagonist-
	# pair E/I output halves (effective neutral 0.5). QUAD is bit-identical to
	# pre-12. The WNN_STATE_SPLIT split-trainer is mode-aware too (12/07/2026:
	# plants hard TRUE/FALSE on TERNARY/BINARY — cell_mode::plant_cell).
	ap.add_argument("--memory-mode", type=str, default="QUAD_WEIGHTED",
	                choices=["QUAD_WEIGHTED", "QUAD_BINARY", "TERNARY", "BINARY", "QSR", "PLN"],
	                help="Controller cell format (default QUAD_WEIGHTED; TERNARY/BINARY = deterministic "
	                     "granularity ablation arms; QSR/PLN = stochastic decode arms — QSR is a per-timestep "
	                     "coin sampler of QUAD, PLN of TERNARY).")
	# Phase-5c saturation→grow damping (§11b). Lower = gentler state growth under
	# splitting-trainer saturation pressure (default 0.02 ≈ old aggressive behavior
	# at high saturation; 0.005 damps hard so sn grows measuredly, not every gen).
	ap.add_argument("--saturation-grow-gain", type=float, default=0.02,
	                help="5c saturation→state-growth probability gain (lower=gentler; default 0.02).")
	# Per-genome cell budget (23/07/2026). A wandering (poor) controller writes a
	# cell per distinct visited input pattern, and bits-grow replicates cells ×2^d
	# — unbounded, this OOM-looped the QUAD-dfa study cells (200k+ cells/genome).
	# Once a genome's carried Lamarckian cells reach the budget, bits/neuron GROW
	# mutations clamp to shrink-only. Default effectively off.
	ap.add_argument("--max-cells", type=int, default=1_000_000_000,
	                help="per-genome carried-cell budget: structural grows are suppressed at/above "
	                     "this many populated cells (default 1e9 = off).")
	# --max-cells is a THRESHOLD, not a ceiling: a genome under it may still take a
	# legal bits-grow, which replicates its layer x2^delta. Measured overshoot on the
	# dfa1l study: 579,115 cells against a 180,000 budget = 3.22x (suffix_delta=2 => x4).
	# All 8 overshooting cells were QUAD; no BINARY cell tripped 180k at all, so the
	# granularity ablation was not budget-matched. This makes the budget behave like
	# its name by clamping a grow to the largest delta that still fits post-grow.
	# Default off: historical runs stay bit-for-bit reproducible.
	ap.add_argument("--max-cells-strict", action="store_true",
	                help="enforce --max-cells on the POST-grow count (clamp the grow) instead of "
	                     "only suppressing grows once already at/over budget (default: off).")
	# Held-out scoring refits the INPUT thermometer thresholds on the report seed, but
	# a trained genome's cells were written at addresses computed under the TRAIN seed's
	# thresholds — connections + thresholds ARE the address function. Refitting reads the
	# memory at addresses it was never written to. Measured on frozen winners over 5 report
	# seeds: 1layer_9feat_BINARY_s31337003 48.0+-13.8 -> 86.8+-1.7 (docs/threshold_misalignment_finding.md).
	# Default OFF so an in-flight campaign stays internally consistent; completed work is
	# better re-measured with scripts/rescore_winners.py than re-run.
	# Evaluation / episode.
	ap.add_argument("--eval-episodes", type=int, default=20)
	ap.add_argument("--memory-eval-episodes", type=int, default=None,
		help="During-search eval episodes for the MEMORY stage ONLY (default: --eval-episodes). "
		     "MEMORY is the stability-lift stage AND cheap per-gen (shapes collapse), so it can "
		     "afford more episodes for a clean stability gradient while NEURONS stays cheaper. "
		     "13/06/2026: 16-ep eval can't resolve stability → GA optimizes blind; raise this.")
	ap.add_argument("--steps", type=int, default=2000,
	                help="Rollout steps per episode (2 s @ 1 kHz). 2000 is the measured "
	                     "sweet spot + the reward_gated default; trajectory memory scales "
	                     "~linearly with this, so pair large values with --max-*-neurons caps.")
	ap.add_argument("--tilt", type=float, default=15.0)
	# W2.3 train-under-weather: arm the calibrated disturbance ladder in ALL
	# rollouts of this run (training + in-search eval + report). OFF = clean
	# (bit-identical legacy). Anchors @2000/L2: PID+ 99.8 / PD 84.0; every
	# clean-trained WNN scored 0 at L2 (W2.2 brittleness audit, 06/07).
	# AIRFRAME. Omit for the legacy synthetic plant (bit-identical). Presets
	# live in wnn/control/airframe.py and each carries its citation:
	# cf21_brushless (Bitcraze firmware, the focus), cf2x_urdf
	# (gym-pybullet-drones), cf2x_firmware (cross-check).
	ap.add_argument("--airframe", type=str, default=None,
	                help="airframe preset name; omit for the legacy plant")
	ap.add_argument("--disturbance", type=str, default="OFF",
	                choices=["OFF", "L4A", "L4B", "L4C",
	                         "L1", "L2", "L3", "L2D", "L3D"],
	                help="disturbance level for all rollouts (default OFF). USE THE "
	                     "L4* RUNG: every value is cited to a paper or datasheet "
	                     "(docs/disturbance_param_sources.md) — sensor noise fixed at "
	                     "the ADIS16448 datasheet, plant uncertainty as the axis "
	                     "(L4A none / L4B 10%% / L4C 20%%, Molchanov's measured "
	                     "ceiling; 30%% is known-harmful). L1/L2/L3/L2D/L3D are "
	                     "DEPRECATED: unsourced magnitudes, and L2D/L3D add sensor "
	                     "dropout + observation latency that NO surveyed simulator "
	                     "models. Nothing measured on them is submission-grade")
	# Motor-fault experiment (docs/motor_fault_experiment.md): fixed single-motor
	# effectiveness loss, e.g. "1:0.3" = motor 2 at 30% effectiveness. Applied to
	# the disturbance's FIXED multiplier via evaluator.apply_motor_fault, so BOTH
	# training rollouts and scoring see the faulted plant. Requires a disturbance
	# level (the fault rides on the DisturbanceConfig).
	ap.add_argument("--motor-fault", type=str, default=None,
	                help="'idx:factor' fixed motor-effectiveness fault (needs --disturbance != OFF)")
	# Overactuated residual mode (Phase 2 — docs/OVERACTUATED_RESIDUAL_DESIGN.md).
	# Setting --geometry switches the run to N-rotor residual search: the sim
	# flies the (optionally perturbed) TRUE table via step_n, the WNN output is
	# a clamped residual on the allocator-LQR baseline, and ALL stages score
	# WITHOUT DAGGER training (empty memory = neutral residual; the GA learns
	# the mismatch through fitness). --lamarckian/--teacher* are unsupported here.
	ap.add_argument("--geometry", type=str, default=None,
	                choices=["octo-x", "canted-hex", "quad-plus"],
	                help="overactuated airframe preset (enables residual mode)")
	ap.add_argument("--geometry-cant", type=float, default=20.0,
	                help="canted-hex tilt (deg) about each arm")
	ap.add_argument("--geometry-tilt-err", type=float, default=0.0,
	                help="per-rotor tilt-error magnitude (deg, U(-m,m) seeded draws) on the TRUE table")
	ap.add_argument("--geometry-pos-err", type=float, default=0.0,
	                help="per-rotor position-error magnitude (m) on the TRUE table")
	ap.add_argument("--rotor-asym", type=float, default=0.0,
	                help="per-rotor thrust multiplier magnitude (1±m seeded draws) on the TRUE table")
	ap.add_argument("--alloc-scale", type=float, default=1.0,
	                help="residual gain on (wnn-0.5)")
	ap.add_argument("--alloc-clamp", type=float, default=0.15,
	                help="|residual| bound (the safety clamp)")
	ap.add_argument("--alloc-tau-max", type=float, default=0.144,
	                help="allocator-LQR per-axis torque authority (N*m)")
	# Initial-condition severity (match a curriculum stage, e.g. Stage A = 5/0.5/0.3).
	ap.add_argument("--body-rate", type=float, default=0.5, help="max initial body rate (rad/s)")
	ap.add_argument("--yaw-rate", type=float, default=0.3, help="max initial yaw rate (rad/s)")
	# Early-stop cadence: patience checks every check_interval gens (per-phase patience
	# set via --*-patience). Faster pace = smaller check_interval + smaller patience.
	ap.add_argument("--check-interval", type=int, default=10, help="gens between patience checks")
	# Magnitude-aware patience (controller fitness redesign (a),
	# docs/controller_fitness_patience_redesign.md). Default OFF → the early-stopper
	# watches the rank-WHM (comparable with the existing cohort + C10 sweep). When
	# set, it watches err°/stable% MAGNITUDE and recovers patience proportional to
	# real improvement, fixing the premature early-stop where a genuine
	# stable 20%→70% jump barely moved the rank objective. Selection is unchanged.
	# DEFAULT ON for the controller (01/07/2026): magnitude-aware is the right patience
	# for physical err°/stable% metrics; every driver already passed it. --no-... to opt out.
	ap.add_argument("--magnitude-aware-patience", action=argparse.BooleanOptionalAction, default=True,
	                help="Patience watches err°/stable° magnitude (not rank-WHM); recovers ∝ real gain. Default ON.")
	ap.add_argument("--universe-episodes", type=int, default=8)
	# Inner reward-gated train knobs (production: leave None → 8 rounds × 24 eps);
	# smoke tests pass tiny values to keep per-genome training under a few seconds.
	ap.add_argument("--rg-rounds", type=int, default=None,
	                help="Reward-gated inner-train rounds (default: 8)")
	ap.add_argument("--rg-episodes-per-round", type=int, default=None,
	                help="Episodes per reward-gated round (default: 24)")
	ap.add_argument("--rg-eval-episodes", type=int, default=None,
	                help="Eval episodes within reward-gated (default: 20)")
	ap.add_argument("--topk-per-neuron", type=int, default=None,
	                help="Beam-search top-K candidate addresses per neuron in the "
	                     "per-motor EDRA solve (default 4). ONLY bites at sn>0: a "
	                     "memoryless controller skips the solve entirely "
	                     "(controller.rs solve_motors=0), which is ~100%% of the "
	                     "measured 150x sn=8/sn=0 runtime gap. Lower = faster but "
	                     "considers fewer addresses per neuron, so it trades solve "
	                     "QUALITY for speed — a science knob, not a free one. "
	                     "Omitted leaves the default, so prior cohorts reproduce.")
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
	ap.add_argument("--fit-weight-steady", type=float, default=0.0,
	                help="Weight on mean_steady_error_deg (mean attitude err over the last 20%% of steps) "
	                     "in the harmonic-rank fitness. The I-pressure term: isolates the steady-state "
	                     "offset only an integrator can kill. Default 0. >0 activates multi-objective.")
	# Parallelism — the ControllerEvaluator's per-genome ThreadPool. Defaults to
	# 1 inside ControllerEvaluator (no concurrency), which leaves 15/16 cores
	# idle when the GA evaluates 200+ genome populations. 4-8 is the sweet spot
	# on the M4 Max (16 cores, leaves headroom for Rayon-inside-step + the IDS
	# worker on RAYON_NUM_THREADS=3). Found 29/05/2026 during c-mix-4 RCA.
	ap.add_argument("--fit-weight-effort", type=float, default=0.0,
	                help="Σu² allocation-efficiency rank weight (mean per-step Σ pwm²; "
	                     "the overactuated Phase-3 term — misallocation costs effort, "
	                     "not attitude error, on planar airframes; 0 = off)")
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
	ap.add_argument("--report-seeds", type=int, nargs="+", default=None,
		help="MULTI-SEED held-out: re-eval the stage winner on EACH of these fresh seeds and "
		     "report mean±std (robust to the single-seed eval variance). Overrides --report-seed "
		     "when set. The honest, seed-robust paper number.")
	ap.add_argument("--report-episodes", type=int, default=None,
		help="Episodes for the per-stage HELD-OUT eval only (default: --eval-episodes). "
		     "The held-out runs once per stage, so it can afford far more episodes than "
		     "the per-generation search eval — de-quantizes the reported stable%%.")
	ap.add_argument("--holdout-pop-sample", type=int, default=8,
		help="Held-out eval scores the winner + this many sampled genomes (default 8). "
		     "The RESULT is always the winner; the rest are a descriptive (leak-guarded, "
		     "not-selected) stat, so scoring the whole population at report-episodes is ~Nx "
		     "waste. 0 = score the full final population (legacy behavior).")
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
	# Periodic IN-STAGE checkpoint (crash protection during a stage). Adaptive
	# wall-clock cadence (shared SaveCadence): a slow gen (e.g. NEURONS ~40 min/gen)
	# trips the budget every gen → saves every gen; fast gens throttle to
	# --checkpoint-max-interval. Bounds work lost to a hard crash (no signal) to
	# ~one slow gen. Writes to the stage save_path; needs --save-winner or
	# --save-stage-checkpoints to have a path. Set 0 to save EVERY gen regardless
	# of cost; large value (or absence of a save path) effectively disables it.
	ap.add_argument("--checkpoint-target-loss-seconds", type=float, default=300.0,
	                help="Adaptive in-stage save: max wall-clock seconds to risk losing "
	                     "on a hard crash. Default 300 (5 min). 0 = save every gen.")
	ap.add_argument("--checkpoint-max-interval", type=int, default=10,
	                help="Hard cap on generations between in-stage saves (fast-gen throttle). Default 10.")
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
	                help="Path to an emergency-dump checkpoint (written by _wire_cancel, see _stage_emergency_path). "
	                     "When set, Stage 0 (grid) is skipped and the run starts at the "
	                     "stage selected by --resume-mode.")
	ap.add_argument("--resume-mode", type=str, default="same",
	                choices=["same", "next"],
	                help="'same' (default): continue the dumped stage from its dumped "
	                     "population. 'next': skip the dumped stage and warm-start the "
	                     "next stage from the dumped best_genome.")
	ap.add_argument("--seed-winner", type=str, default=None,
	                help="Path to a saved winner.yaml.gz (a controller checkpoint). "
	                     "CURRICULUM warm-start (E5.2): skip the grid and start the "
	                     "NEURONS stage from this winner's architecture + trained memory "
	                     "+ FULL final population, then run the pipeline under THIS run's "
	                     "--disturbance. Use e.g. an L1-trained winner to fine-tune under "
	                     "L2 (train in the rain, refine in the storm). Mutually exclusive "
	                     "with --resume-from-emergency.")
	ap.add_argument("--seed-winner-stage", type=str, default="neurons",
	                choices=["neurons", "bits", "connections", "memory"],
	                help="Which stage the --seed-winner warm-start begins at (grid always "
	                     "skipped). The chain runs from that stage → MEMORY under --disturbance. "
	                     "'neurons' (default, E5.2): full arch re-search (sn varies). "
	                     "'connections': FREEZE neuron-count + bit-width, vary only connectivity "
	                     "(synaptogenesis) — with --lamarckian this DAGGER-retrains the cells "
	                     "under the storm while the core arch is frozen (the true 'fine-tune the "
	                     "L1 shape' test). 'memory': skip straight to the value-GA MEMORY stage "
	                     "which does NOT DAGGER-train — this measures RAW L1→L2 transfer, not "
	                     "fine-tuning (score_genomes on frozen cells).")

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

	# Option A: enable the learned-integral state target in the Rust trainer
	# (read per bptt_train_window call). Set before any training begins.
	if getattr(args, "state_integral", False):
		os.environ["WNN_STATE_INTEGRAL_TARGET"] = "1"
		print("[state-integral] ON — recurrent state trained as a learned integrator "
		      "(WNN_STATE_INTEGRAL_TARGET=1). Use small --grid-state-neurons.")

	# Install SIGTERM/SIGINT handlers BEFORE any Rust work begins so that
	# SIGTERM during stage 0 grid or any subsequent stage is caught and
	# triggers a clean emergency dump.
	_install_signal_handlers()

	# Load emergency-dump pickle if --resume-from-emergency is set. The loaded
	# state is forwarded to _run_one via the resume_state arg.
	resume_state = None
	if args.resume_from_emergency and args.seed_winner:
		raise ValueError("--seed-winner and --resume-from-emergency are mutually exclusive")
	if args.resume_from_emergency:
		resume_path = Path(args.resume_from_emergency)
		if not resume_path.exists():
			raise FileNotFoundError(f"--resume-from-emergency {resume_path} does not exist")
		resume_state = _ctl_load(resume_path)
		resume_state["resume_mode"] = args.resume_mode
		print(f"[main] Loaded emergency dump from {resume_path}")
		print(f"[main]   stage_num={resume_state.get('stage_num')} "
		      f"stage_name={resume_state.get('stage_name')!r} "
		      f"generation={resume_state.get('generation')} "
		      f"pop={len(resume_state.get('population') or [])}")
	elif args.seed_winner:
		# CURRICULUM warm-start (E5.2). A winner.yaml.gz payload has the SAME
		# schema the resume path consumes (spec / best_genome / population), so
		# we reuse that machinery: force stage_num=0 + mode='next' → the grid is
		# skipped and Stage 1 (NEURONS) starts warm-started from the winner's
		# spec + best_genome + FULL population, evolving under THIS run's
		# --disturbance (set on `ec` below).
		seed_path = Path(args.seed_winner)
		if not seed_path.exists():
			raise FileNotFoundError(f"--seed-winner {seed_path} does not exist")
		resume_state = _ctl_load_optional(seed_path)
		if resume_state is None:
			raise ValueError(f"--seed-winner {seed_path} could not be loaded as a controller checkpoint")
		if getattr(resume_state.get("best_genome"), "cells", None) is None:
			raise ValueError(f"--seed-winner {seed_path} carries no trained cells (arch-only) — cannot curriculum-warm-start")
		# stage_num + mode='next' pick the FIRST stage to run (resume_start_stage =
		# min(stage_num+1, 4)): neurons→1, bits→2, connections→3, memory→4. Earlier
		# stages are skipped, carrying the L1 population straight through.
		_sw_stage_map = {"neurons": 0, "bits": 1, "connections": 2, "memory": 3}
		_sw_stage = getattr(args, "seed_winner_stage", "neurons")
		resume_state["stage_num"] = _sw_stage_map.get(_sw_stage, 0)
		resume_state["resume_mode"] = "next"
		_sw_desc = {
			"neurons": "NEURONS warm-started (full arch re-search)",
			"bits": "BITS onward (neuron-count frozen)",
			"connections": "CONNECTIONS onward (neurons+bits FROZEN; Lamarckian rewire+retrain)",
			"memory": "MEMORY-only (value-GA, NO DAGGER — raw transfer)",
		}.get(_sw_stage, _sw_stage)
		print(f"[main] CURRICULUM seed-winner from {seed_path} "
		      f"(pop={len(resume_state.get('population') or [])}, "
		      f"spec={type(resume_state.get('spec')).__name__}) → grid skipped, "
		      f"{_sw_desc} under --disturbance {args.disturbance}")

	t_start = time.time()
	from wnn.control.training import DisturbanceConfig
	from wnn.control.airframe import Airframe as _Airframe
	dist = DisturbanceConfig.preset(args.disturbance, seed=911)
	if getattr(args, "motor_fault", None):
		if dist is None:
			raise SystemExit("--motor-fault requires --disturbance != OFF (the fault "
			                 "rides on the DisturbanceConfig)")
		from wnn.control.evaluator import apply_motor_fault
		apply_motor_fault(dist, args.motor_fault)
		print(f"[FAULT] motor fault {args.motor_fault} armed for ALL rollouts "
		      f"(training AND scoring): fixed motor_asym={dist.motor_asym}")
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate,
		disturbance=dist,
		# Airframe: None keeps the pre-airframe synthetic plant so untouched
		# recipes stay bit-identical; a preset name swaps BOTH the sim and the
		# model-based teachers (they read the same numbers) in one place.
		airframe=(None if not getattr(args, 'airframe', None)
		          else _Airframe.preset(args.airframe)),
	)
	# L1 (--obs-dhat): derive the plant's control effectiveness ONCE, here, and stash
	# it on args so every _make_spec call in this run carries the same constant. It
	# comes from the Rust calibrate_control_gains (the SAME routine the LQR/MPC/MPCOF
	# teachers use) — deriving it in Python would be exactly the duplicated-numerics
	# failure the Rust-first rule exists to prevent.
	args._dhat_b = None
	if getattr(args, "obs_dhat", False):
		if ec.airframe is None:
			raise SystemExit("--obs-dhat requires --airframe: the observer's plant constant b "
			                 "is derived from the airframe, and the synthetic default would "
			                 "make d̂ estimate a vehicle you are not flying.")
		from wnn.control._accel import calibrate_control_gains as _calib
		af = ec.airframe
		args._dhat_b = tuple(_calib(
			dt=float(ec.dt), arm_length=float(af.arm_length), k_thrust=float(af.k_thrust),
			k_drag=float(af.k_drag), inertia=[float(x) for x in af.inertia],
			gravity=float(af.gravity)))
		print(f"[L1] --obs-dhat ON: d̂ observer b={args._dhat_b} (from --airframe "
		      f"{args.airframe}), l_gain={args.dhat_l_gain} → +3 input features")
	if dist is not None:
		print(f"[W2] disturbance={args.disturbance} armed for ALL rollouts "
		      f"(tau_bias={dist.tau_bias[0]:.4f} N·m, gust_sigma={dist.gust_sigma:.4f}, "
		      f"asym_mag=±{dist.motor_asym_mag:.0%}, gyro_sigma={dist.gyro_sigma})")

	# Overactuated residual mode (Phase 2): TRUE-vehicle geometry + alloc baseline.
	_geo_base = args.base_seed if args.base_seed is not None else args.seed
	geo_cfg, alloc_cfg = _geometry_from_args(args, _geo_base or 0)
	if geo_cfg is not None:
		if getattr(args, "lamarckian", False):
			raise SystemExit("--geometry (residual mode) is score-only — --lamarckian "
			                 "trains cells via DAGGER, unsupported (step-3 design).")
		if getattr(args, "teacher_schedule", "") or getattr(args, "teacher_blend", ""):
			raise SystemExit("--geometry (residual mode) ignores DAGGER teachers — "
			                 "drop --teacher-schedule/--teacher-blend.")
		n_rot = len(geo_cfg.rows)
		if getattr(args, "decouple_outputs", False) and n_rot != 4:
			raise SystemExit(f"--decouple-outputs requires 4 motors; --geometry {args.geometry} has {n_rot}.")
		if getattr(args, "action_repeat", 1) != 1:
			raise SystemExit("--geometry residual scoring requires --action-repeat 1 "
			                 "(the CPU scorer composes per decision step).")
		ec.geometry = geo_cfg
		ec.alloc_residual = alloc_cfg
		args._geometry_num_motors = n_rot
		print(f"[GEO] residual mode: {args.geometry} (N={n_rot})  "
		      f"tilt-err=±{args.geometry_tilt_err}°  pos-err=±{args.geometry_pos_err}m  "
		      f"rotor-asym=±{args.rotor_asym:.0%}  residual scale={args.alloc_scale} "
		      f"clamp={args.alloc_clamp}  tau_max={args.alloc_tau_max} N·m  "
		      f"(teacher '{getattr(args, 'teacher', 'pid')}' IGNORED — no DAGGER; "
		      f"empty memory = neutral residual)")

	_on_axis = grid_output_neuron_axis(args)
	_grid_dims = (f"{len(args.grid_state_neurons)}×{len(args.grid_bits)}"
	              + (f"×{len(_on_axis)}" if len(_on_axis) > 1 else ""))
	print(f"Phased-GA controller search: "
	      f"grid ({_grid_dims}={grid_point_count(args)}) "
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
			"grid_on": _on_axis, "input_window_k": args.input_window_k,
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

	_join_pending_saves()  # ensure any async between-stage writes finished before exit
	# A witnessed SIGTERM/SIGINT means this run was terminated mid-flight (the
	# cooperative-cancel path unwound the stages), NOT completed. Exit 143 so the
	# study driver treats it as a watchdog stop (no marker, retry) rather than a
	# finished cell — a bare `return 0` here silently records a truncated run as
	# done and lets a NEURONS-stage triple leak into the MEMORY-stage table.
	from wnn.control import cancel_state
	if cancel_state.sigterm_received():
		return 143
	return 0


if __name__ == "__main__":
	sys.exit(main())
