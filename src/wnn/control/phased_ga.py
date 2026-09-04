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

class HoldoutScoringError(RuntimeError):
	"""The held-out scorer could not score — the run is dead, not degraded.

	Raised instead of printing-and-continuing (14/08/2026). Genuine per-genome
	badness produces a BAD SCORE, never an exception; anything that raises in
	here is code or ABI breakage, and every held-out block after it will break
	the same way. Continuing just spends hours producing a result that can never
	be reported — see the lam0/s31337003 post-mortem in the sweep log.
	"""


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
	# SYNC since 11/08/2026 (was async_save=True). MEASURED on the live sn=8 run
	# (mem_sampler.csv, gen-2 boundary 14:14:58Z): the async path's eager whole-
	# population encode spiked RSS 15.0 -> 28.75 GB — +14 GB EVERY generation.
	# Base + spike is what breaches the watchdog floor whenever the IDS worker is
	# busy at the same moment (the 11:17Z kill). The save fires at the generation
	# boundary on the GA's own thread, where nothing mutates the population — so
	# the sync path's streaming encode (peak = ONE genome, ~100 MB) is safe here;
	# async only ever needed its snapshot for the background WRITE, and blocking
	# instead costs ~1-3 min/gen on multi-GB sn>0 dumps (<1 s at sn=0 scale),
	# throttled further by SaveCadence's wall-clock budget.
	strat._checkpoint_mgr = PhasedCheckpointManager(
		Path(_stage_emergency_path(args, stage_num, stage_name)),
		ControllerGenomeCodec(), SaveCadence(budget, max_int), async_save=False)

from wnn.control.evaluator import (
	ControllerSpec, ControllerEvaluator, arch_shape_from_spec, spec_from_arch,
	fit_thresholds_from_pid_rollouts,
	collect_student_feature_samples,
	calib_episode_config as _calib_ec,
)
from wnn.control.arch_strategy import (
	ControllerArchGAStrategy, ControllerArchTSStrategy,
	ControllerMemoryGAStrategy, ControllerMemoryTSStrategy,
	default_controller_arch_config,
)
from wnn.control.ga_strategy import (default_controller_ga_config,
                                     search_aggregation as _search_aggregation,
                                     select_aggregation as _select_aggregation,
                                     gate_args as _gate_args)
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
               delta_max: float = 0.1, delta_gamma: float = 1.0,
               obs_tilt_p: bool = False, obs_tilt_i: bool = False,
               obs_peraxis_p: bool = False, obs_peraxis_i: bool = False,
               obs_peraxis_yaw: bool = True,
               obs_pwm: bool = False,
               obs_yaw_err: bool = False, obs_yaw_err_i: bool = False,
               dhat_b: "tuple[float, float, float] | None" = None,
               dhat_l_gain: float = 0.05,
               dhat_ff: bool = False, dhat_ff_clamp: float = 0.30,
               obs_collective_cmd: bool = False, obs_alt_err: bool = False,
               obs_vz: bool = False,
               obs_pos_err_xy: bool = False, obs_vel_xy: bool = False,
               integral_leak: float = 0.99, integral_scale: float = 1.0,
               dt: float = 0.001,
               decouple_outputs: bool = False, bits_per_feature: int = 8,
               feature_balance_ratio: float = 0.0,
               conn_policy: str = "spread", conn_policy_min: int = 2,
               conn_mutation_scope: str = "free",
               output_full_window: bool = False, frame_stride: int = 1,
               target_levels: int = 0,
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
		delta_gamma=delta_gamma,
		obs_tilt_p=obs_tilt_p, obs_tilt_i=obs_tilt_i,
		obs_peraxis_p=obs_peraxis_p, obs_peraxis_i=obs_peraxis_i,
		obs_peraxis_yaw=obs_peraxis_yaw,
		obs_pwm=obs_pwm,
		obs_yaw_err=obs_yaw_err, obs_yaw_err_i=obs_yaw_err_i,
		dhat_b=dhat_b, dhat_l_gain=dhat_l_gain, dhat_ff=dhat_ff, dhat_ff_clamp=dhat_ff_clamp,
		obs_collective_cmd=obs_collective_cmd, obs_alt_err=obs_alt_err, obs_vz=obs_vz,
		obs_pos_err_xy=obs_pos_err_xy, obs_vel_xy=obs_vel_xy,
		integral_leak=integral_leak, integral_scale=integral_scale,
		dt=dt,
		decouple_outputs=decouple_outputs,
		feature_balance_ratio=feature_balance_ratio,
		conn_policy=conn_policy, conn_policy_min=conn_policy_min,
		conn_mutation_scope=conn_mutation_scope,
		output_full_window=output_full_window, frame_stride=frame_stride,
		target_levels=target_levels,
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


_grid_for_refit = None   # last ControllerGridSearch — the refit needs its probe spec


def stage0_grid(args, ec: EpisodeConfig, seed: int, thresholds_override=None):
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
	gs = ControllerGridSearch(args, ec, seed, thresholds_override=thresholds_override)
	outcome = gs.run()
	winner_spec = outcome.best_point.spec
	global _grid_for_refit
	_grid_for_refit = gs
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
		# The GA stages ARE the search (`--strategy ts` is not the default), so
		# omitting these here made --fit-weight-alt inert no matter how correctly
		# every layer below forwarded it. 18/08: this was missed on the first fix
		# because the TS sibling was patched and the GA path was verified by
		# calling default_controller_ga_config directly — which proves the
		# FUNCTION forwards the weight, not that this caller passes it.
		weight_alt=getattr(args, "fit_weight_alt", 0.0),
		weight_pos=getattr(args, "fit_weight_pos", 0.0),
		# In-search aggregation (19/08): None = legacy harmonic. Same lesson as
		# the weights above — this caller passing it is what makes the flag real.
		aggregation=_search_aggregation(args),
		zrank_clamp=getattr(args, "zrank_clamp", 3.0),
		# Viability gate (21/08): same caller-must-pass-it lesson as alt/pos.
		gate_stable_min=_gate_args(args)[0],
		gate_err_max=_gate_args(args)[1],
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
	             or getattr(args, "fit_weight_effort", 0.0) > 0
	             or getattr(args, "fit_weight_alt", 0.0) > 0
	             or getattr(args, "fit_weight_pos", 0.0) > 0)
	if multi_obj:
		tscfg.fitness_calculator_type = FitnessCalculatorType.CONTROLLER_HARMONIC
	tscfg.fitness_weight_err_sq = args.fit_weight_err_sq
	tscfg.fitness_weight_stable = args.fit_weight_stable
	tscfg.fitness_weight_jerk = args.fit_weight_jerk
	tscfg.fitness_weight_mono = args.fit_weight_mono
	tscfg.fitness_weight_steady = args.fit_weight_steady
	tscfg.fitness_weight_effort = getattr(args, "fit_weight_effort", 0.0)
	tscfg.fitness_weight_alt = getattr(args, "fit_weight_alt", 0.0)
	tscfg.fitness_weight_pos = getattr(args, "fit_weight_pos", 0.0)
	tscfg.fitness_aggregation = _search_aggregation(args)
	tscfg.fitness_gate_stable_min, tscfg.fitness_gate_err_max = _gate_args(args)
	tscfg.zrank_clamp = getattr(args, "zrank_clamp", 3.0)
	if tscfg.fitness_aggregation != "harmonic":
		# The single-objective CONTROLLER type has no aggregation knob — a
		# non-default aggregation needs the multi-objective calculator even
		# when only err² carries weight (mirrors default_controller_ga_config).
		tscfg.fitness_calculator_type = FitnessCalculatorType.CONTROLLER_HARMONIC
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
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None),
		episode_config=_calib_ec(args, ec))
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes,
	                         seed=seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=args.num_eval_folds,
	                         score_crn=args.score_crn)
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
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None),
		episode_config=_calib_ec(args, ec))


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
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None),
		episode_config=_calib_ec(args, ec))
	mem_eps = getattr(args, "memory_eval_episodes", None) or args.eval_episodes
	ev = ControllerEvaluator(spec, num_eval_episodes=mem_eps,
	                         seed=seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=args.num_eval_folds,
	                         score_crn=args.score_crn)
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
		# Provenance: a checkpoint has to say which fitness produced it. Recording
		# only five of the eight weights meant a resumed or re-scored genome could
		# not be attributed to the arm that made it — and in a sweep whose ONLY
		# varying axis is a weight, that is the whole identity of the run.
		"fitness_weights": {
			"err_sq": args.fit_weight_err_sq,
			"stable": args.fit_weight_stable,
			"jerk":   args.fit_weight_jerk,
			"mono":   args.fit_weight_mono,
			"steady": args.fit_weight_steady,
			"effort": getattr(args, "fit_weight_effort", 0.0),
			"alt":    getattr(args, "fit_weight_alt", 0.0),
			"pos":    getattr(args, "fit_weight_pos", 0.0),
			# The aggregation is part of the fitness identity: identical weights
			# under different combines select DIFFERENT genomes (arm 9). A
			# checkpoint that recorded the weights but not the combine would be
			# the 18/08 "5 of 8 weights" omission again, one field over.
			"aggregation_search": _search_aggregation(args),
			"aggregation_select": _select_aggregation(args),
			"zrank_clamp": getattr(args, "zrank_clamp", 3.0),
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

def _pid_baseline(ec: EpisodeConfig, episodes: int, seed: int, folds: int = 5) -> list:
	"""Reference-baseline rows on the held-out episode set, RIVAL FIRST.

	Row 0 is the comparator the WNN is judged against. Any further rows are
	INFORMATIONAL context printed beside it — never the number a claim rests on.

	Quad: PID twice — estimator-fed (the rival) then oracle-fed (informational),
	per the 13/08/2026 rule. The WNN flies on a raw noisy IMU, so a PID handed
	the true quaternion is an upper bound rather than a rival; the gap between
	the two rows is what state estimation costs the classical controller on this
	plant. Both rows fly the SAME episodes on the SAME aircraft.

	Residual mode (ec.geometry): the allocator-LQR baseline itself — an all-EMPTY
	controller composed on it scores EXACTLY the baseline (residual ≡ 0), via the
	production CPU scorer. label='alloc-LQR' so the summary row is honest. It has
	no estimator twin: that comparator is the ALLOCATOR, not an attitude teacher
	reading a quaternion, so the feed distinction does not apply to it."""
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
		return [{"stable_rate": row[2], "mean_attitude_error_deg": math.degrees(row[1]),
		         "mean_reward": row[0], "label": "alloc-LQR",
		         "mean_effort": (row[12] if len(row) > 12 else None)}]
	# NOT eval_closed_loop_reset: it draws ICs from the RAW seed and redraws motor
	# asymmetry per episode, so it flies episodes no WNN cell ever saw. That printed
	# "vs PID 85.0%" under every cell of this study against a true 90.4±7.5 — the
	# comparison was never on the same aircraft. See classical_baseline's docstring.
	from wnn.control.classical_baseline import HoldoutDraw, TeacherFeed, pid_metrics
	draw = HoldoutDraw(seed=seed, episodes=episodes,
	                   steps=ec.steps_per_episode, eval_folds=folds)
	# RIVAL first. Two rollouts of the same episodes ⇒ ~2x a cost measured in
	# seconds, against a stage measured in hours.
	return [pid_metrics(ec, draw, TeacherFeed(use_estimator=True)),
	        pid_metrics(ec, draw, TeacherFeed(use_estimator=False))]


def _baseline_row_str(row: dict, is_rival: bool) -> str:
	"""One classical-comparator line. `is_rival` decides the role tag, which is
	the part a future reader needs most: an oracle-fed row is context, and
	saying so in the row itself is the only thing that survives being pasted
	into a table months later."""
	role = "RIVAL — the comparison" if is_rival else "informational, upper bound"
	sty, eff = row.get("mean_steady_error_deg"), row.get("mean_effort")
	# 14/08/2026 KEY FIX. alt= used to read mean_position_error_m, which chunk D
	# repointed at the Euclidean 3-D error (NaN on the pre-chunk-D wheel) — runs
	# 7-8 printed "alt=nanm" and lost the rival altitude column 35b1328d added.
	# Altitude has its own key now; pos= is the 3-D error, printed only when it
	# was actually flown (NaN/None = not flown, never a fake perfect hold).
	alt_m, pos_m = row.get("mean_altitude_error_m"), row.get("mean_position_error_m")
	_ok = lambda v: v is not None and not math.isnan(v)
	head = f"  vs {row.get('label', 'PID')}  ({role}):"
	return (f"{head:<52}stable={row['stable_rate']*100:.1f}%  "
	        f"err={row['mean_attitude_error_deg']:.2f}°"
	        + (f"  steady={sty:.2f}°" if sty is not None else "")
	        + (f"  alt={alt_m:.3f}m" if _ok(alt_m) else "")
	        + (f"  pos={pos_m:.3f}m" if _ok(pos_m) else "")
	        + (f"  effort={eff:.3f}" if eff is not None else ""))


def _print_baseline_rows(rows: list) -> None:
	"""The classical comparator block, RIVAL FIRST (see _pid_baseline)."""
	for i, row in enumerate(rows):
		print(_baseline_row_str(row, is_rival=(i == 0)))
	print("     [pool-seeded, fold 0 — same episodes as the WNN row above]")


# The held-out metric row, in report order: (Metrics attribute, label, unit, decimals).
#
# ONE declaration, read by all three surfaces — the per-seed RESULT line, the
# MULTI-SEED aggregate line, and the namespace the aggregate returns. Adding a
# metric to Metrics means adding ONE entry here.
#
# Why it is a list and not three hand-written format strings: this is the FOURTH
# time a metric that was measured, and carried on Metrics, never reached the
# held-out report because each site kept its own list — steady (fixed 05/08),
# pos and alt (14-15/08, where the sweep's pre-registered "rank by held-out
# altitude error" had nothing to rank), and now jerk, which is 20% of C10's
# fitness and the term that won C10 its own sweep. An allowlist that must be
# re-remembered per metric will keep losing the newest one, which is always the
# one an experiment was just designed around.
_HELDOUT_ROW = (
	("mean_steady_error_deg",  "steady",    "°", 2),
	("motor_jerk_mean",        "jerk",      "",  4),
	("mono_violations_total",  "mono_viol", "",  0),
	("mean_altitude_error_m",  "alt",       "m", 3),
	("mean_position_error_m",  "pos",       "m", 3),
	("mean_effort",            "effort",    "",  3),
)


def _heldout_row_str(m) -> str:
	"""Per-seed tail: every metric of the row this scorer actually produced.

	Absent (None) metrics are OMITTED, never printed as 0 — a zero altitude reads
	as a genome holding height perfectly, which is the opposite of not measured."""
	parts = []
	for attr, label, unit, dp in _HELDOUT_ROW:
		value = getattr(m, attr, None)
		if value is not None and not math.isnan(value):
			parts.append(f"  {label}={value:.{dp}f}{unit}")
	return "".join(parts)


def _heldout_row_stats(results) -> dict:
	"""Mean/SD per metric of the row across the report seeds; absent metrics omitted."""
	import statistics
	stats = {}
	for attr, _label, _unit, _dp in _HELDOUT_ROW:
		values = [v for v in (getattr(r, attr, None) for r in results)
		          if v is not None and not math.isnan(v)]
		if values:
			spread = statistics.pstdev(values) if len(values) > 1 else 0.0
			stats[attr] = (statistics.mean(values), spread)
	return stats


def _heldout_row_agg_str(stats: dict) -> str:
	"""MULTI-SEED tail: mean±SD for each metric present in stats, in report order."""
	parts = []
	for attr, label, unit, dp in _HELDOUT_ROW:
		if attr in stats:
			mean_v, sd_v = stats[attr]
			parts.append(f"  {label}={mean_v:.{dp}f}±{sd_v:.{dp}f}{unit}")
	return "".join(parts)


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
		mean = statistics.mean
		sd = lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0
		# The rest of the row (steady/jerk/mono/alt/pos/effort) comes from the single
		# _HELDOUT_ROW declaration, so the printed line and the returned namespace can
		# never disagree about which metrics exist.
		stats = _heldout_row_stats(results)
		print(f"  [report-seeds] {label} MULTI-SEED held-out ({len(results)} seeds {seed_list}): "
		      f"stable={mean(stbs):.1f}±{sd(stbs):.1f}%  err={mean(errs):.2f}±{sd(errs):.2f}°"
		      f"{_heldout_row_agg_str(stats)}")
		# Return the seed-mean as the stage held-out (so downstream recording uses the
		# robust number). EVERY row metric is carried, present-or-None: a caller that
		# ranks by a fitness weight needs the whole row, and the old hand-listed
		# namespace silently dropped jerk, mono and effort.
		aggregate = SimpleNamespace(acc=mean(stbs) / 100.0,
		                            mean_attitude_error_deg=mean(errs), fitness=mean(fits))
		for attr, _label, _unit, _dp in _HELDOUT_ROW:
			setattr(aggregate, attr, stats[attr][0] if attr in stats else None)
		return aggregate
	except HoldoutScoringError:
		raise
	except Exception as e:
		# FAIL FAST (14/08/2026). This used to print and return None. On 14/08 the
		# lam0/s31337003 run hit its first failure here at the GRID stage ~30 min
		# in, logged 34 of them, then completed neurons + memory + every
		# stage-select val draw before dying uncaught 2.5 h later: three hours
		# spent on a result that was already unreportable at minute 30. Killing
		# the run at the first failure turns that into a ~30 min loss, and the
		# stage dump is still on disk for a resume.
		print(f"  [report-seed] {label} held-out failed: {e}")
		raise HoldoutScoringError(f"{label} held-out scoring failed: {e}") from e



def _refit_thresholds_from_student(args, ec, seeds, grid, winner_genome):
	"""One student-state refit round (option A). Returns the new thresholds, or
	None when disabled / not applicable.

	WHY A REFIT OBLIGATES A REGRID. The thermometer defines the ADDRESS function,
	so new thresholds invalidate every learned cell — the paper-critical THRESHOLD
	MISALIGNMENT finding. The caller therefore RE-RUNS the grid under the new
	ladder rather than reusing the winner that produced the samples; that winner's
	only job is to be a realistic state generator.

	WHY ONE ROUND, NOT ITERATION. Each round costs a full grid stage (the cheapest
	stage, but not free) and the fixed point is not guaranteed to exist — the
	student that a wider ladder trains visits different states again. One round
	buys the bulk of the correction (teacher-only -> student-informed); further
	rounds are an open question, not a default.
	"""
	if not getattr(args, "threshold_refit_from_student", False):
		return None
	if winner_genome is None:
		print("  [thr-refit] no grid winner to sample from — keeping the teacher fit")
		return None
	eps = int(getattr(args, "threshold_refit_episodes", 10))
	print(f"  [thr-refit] rolling out the grid winner for {eps} episodes to collect "
	      f"the STUDENT's own feature distribution (the teacher fit under-covers it)")
	try:
		# The grid winner is a RecurrentArchGenome (connectivity + cells), not a
		# buildable ControllerGenome — materialize it against the grid's own spec and
		# ladder, exactly as the evaluator does, so the rollout uses the SAME
		# controller the grid scored.
		from wnn.control.evaluator import controller_genome_from_arch
		cg = controller_genome_from_arch(winner_genome, grid._probe_spec, grid.thresholds)
		extra = collect_student_feature_samples(cg, ec, eps, seeds.train)
	except Exception as e:
		print(f"  [thr-refit] collection failed ({e}) — keeping the teacher fit")
		return None
	n = sum(len(x) for x in extra)
	if n == 0:
		print("  [thr-refit] collector returned nothing — keeping the teacher fit")
		return None
	thr = fit_thresholds_from_pid_rollouts(
		grid._probe_spec, num_episodes=10, seed=seeds.train,
		geometry=getattr(ec, "geometry", None),
		alloc=getattr(ec, "alloc_residual", None),
		episode_config=_calib_ec(args, ec),
		outer_quantile=getattr(args, "threshold_outer_quantile", None),
		extra_samples=extra)
	print(f"  [thr-refit] refit on {n} student samples + the teacher pool; "
	      f"REGRIDDING (the address function moved, so every cell is stale)")
	return thr


def _stage_entries_from_checkpoints(ckpt_dir) -> list:
	"""Rebuild `stage_entries` from a run's saved stage checkpoints.

	Reads `stageN_<name>.yaml.gz` (emergency dumps ignored — they are mid-stage
	snapshots of a stage that also has a final one) and returns them in stage
	order as (LABEL, spec, res-like). The label comes from the checkpoint's own
	`phase_name`, so this never needs to know which stages a run happened to
	execute. Each entry carries the stage's saved `final_population`, which is
	exactly what `_select_headline_stage` slices its top-K from."""
	import re
	from pathlib import Path
	from types import SimpleNamespace
	from wnn.control.evaluator import ControllerSpec
	from wnn.ram.strategies.phased.checkpoint import load_checkpoint
	from wnn.ram.strategies.phased.codecs import ControllerGenomeCodec

	codec = ControllerGenomeCodec()
	found = []
	for p in sorted(Path(ckpt_dir).glob("stage*_*.yaml.gz")):
		m = re.match(r"stage(\d+)_", p.name)
		if m is None:
			continue
		cp = load_checkpoint(str(p), codec)
		if cp is None:
			continue
		spec_d = (cp.extra or {}).get("spec")
		if not isinstance(spec_d, dict):
			print(f"  [recalc] {p.name}: no spec in checkpoint — skipped")
			continue
		pop = list(cp.final_population or [])
		if not pop:
			if cp.best_genome is None:
				print(f"  [recalc] {p.name}: neither population nor best_genome — skipped")
				continue
			pop = [cp.best_genome]
		label = str(cp.phase_name or p.name).upper()
		found.append((int(m.group(1)), label,
		              (label, ControllerSpec(**spec_d),
		               SimpleNamespace(final_population=pop, best_genome=cp.best_genome))))
		print(f"  [recalc] {p.name}: {label} population={len(pop)}")
	found.sort(key=lambda t: t[0])
	return [e for _sn, _lbl, e in found]


def _recalc_headline(args, ec: EpisodeConfig) -> None:
	"""Re-run the headline selection for an ALREADY-FLOWN run, from its checkpoints.

	Nothing is trained and nothing is written: this re-scores the saved candidates
	on the val seeds and prints the same STAGE TABLE / HEADLINE block a live run
	prints. The caller must pass the SAME flags the original run used — the seeds,
	the episodes and the scoring all derive from them, so a mismatched flag would
	silently produce a number that is not comparable with the original."""
	entries = _stage_entries_from_checkpoints(args.recalc_headline)
	if not entries:
		raise SystemExit(f"--recalc-headline: no usable stage checkpoints in {args.recalc_headline}")
	base = args.base_seed if args.base_seed is not None else args.seed
	seeds = resolve_seed_set(base=base, run_index=0, train=args.train_seed,
	                         test=args.test_seed, val=args.val_seed)
	log_seed_set(seeds)
	print(f"\n[recalc] re-selecting the headline over {len(entries)} stages "
	      f"({', '.join(lbl for lbl, _s, _r in entries)}) from {args.recalc_headline}")
	# stage_holdouts is empty: the per-stage report blocks live in the original
	# run's log and are unchanged by re-selection. The selector scores the genome
	# it picks on the report seeds itself, so the headline triple is still the
	# selected genome's own number.
	_select_headline_stage(args, ec, seeds, entries, {})


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

	SELECTION IS A UNION RANKING, NOT PER-STAGE SCORING. Every stage's winner is scored
	on the SAME val draws, then all candidates are ranked together in ONE call to the
	run's own fitness calculator. Why it must be done this way (got wrong 08/08/2026):

	  * `FitnessCalculatorControllerHarmonic` is RANK-based — WHM over per-metric ranks
	    computed WITHIN the list it is handed, and `n == 1 -> 1.0`. A rank means nothing
	    outside the population it came from, so scoring each stage SEPARATELY and
	    comparing the resulting numbers is invalid. It is also perverse: a winner is
	    `pop[0]` of its own population, so it ranks 1 there and every stage scores
	    exactly 1.0 — selection on a constant. Verified empirically.
	  * Handing all candidates to the calculator AS ONE LIST is precisely the case a
	    rank-WHM is valid for: they compete on the same population, so a genome cannot
	    win by having had weaker company.

	MULTI-SEED VAL (5 draws, derived `seeds.val + i`). One val draw would make
	best-of-N selection a lottery — the more candidates screened on a single finite
	draw, the likelier the winner is one that got lucky there. Averaging the metrics
	across 5 val seeds before ranking damps that, exactly as the report block uses 5.
	This is NOT a leak: val is disjoint from both the search folds and the report seeds,
	so the published number stays honestly measured whichever candidate wins.

	    search folds -> train | seeds.val + 0..4 -> ranks the candidates | report seeds -> published

	Only the winner of each stage is scored here (`final_population=None` forces
	`pop=[best_genome]`): the population sample exists for descriptive stats and would
	just multiply cost. Extending this to the full carried populations is a follow-up —
	it needs bounded top-K retention before `_release_prior_populations` frees them.

	`stage_entries` is [(label, spec, res), ...]. Returns the winning label (or None)."""
	import statistics
	from types import SimpleNamespace
	from wnn.control.controller_orchestrator import ControllerOrchestrator
	from wnn.ram.fitness import FitnessCalculatorControllerHarmonic
	VAL_SEEDS = [int(seeds.val) + i for i in range(5)]
	_TOP_K = max(1, int(getattr(args, "stage_select_top_k",
	                            ControllerOrchestrator.STAGE_SELECT_TOP_K)))

	def _mean_metric(ms):
		"""Average the fields the fitness calculator ranks, across val draws."""
		def avg(attr, default=None):
			vals = [getattr(m, attr, None) for m in ms]
			vals = [v for v in vals if v is not None]
			return statistics.mean(vals) if vals else default
		return SimpleNamespace(
			reward=avg("reward", 0.0), stable_rate=avg("stable_rate", 0.0),
			acc=avg("acc", 0.0), mean_attitude_error_deg=avg("mean_attitude_error_deg", 0.0),
			mean_steady_error_deg=avg("mean_steady_error_deg"),
			motor_jerk_mean=avg("motor_jerk_mean"), mono_violations_total=avg("mono_violations_total"),
			mean_effort=avg("mean_effort"),
			# The translation channels are fields the calculator ranks too
			# (17/08/2026). Dropping them here handed the union ranking a None
			# and the alt weight was discarded with a warning — the ONE site
			# noisy enough to notice, which is how the whole gap surfaced.
			mean_altitude_error_m=avg("mean_altitude_error_m"),
			mean_position_error_m=avg("mean_position_error_m"))

	# CANDIDATES = the top-K genomes of EVERY stage, ranked together in one population.
	# `final_population` survives the orchestrator's release trimmed to K
	# (_release_prior_populations), so pop[0] — the genome that IS the published
	# result and the exported member — is always a candidate. Falling back to
	# best_genome keeps older checkpoints and resumed runs scoreable.
	cand: list = []          # [(key, label, spec, genome), ...]
	for label, spec, res in stage_entries:
		if res is None:
			continue
		pop = list(getattr(res, "final_population", None) or [])
		if not pop:
			bg = getattr(res, "best_genome", None)
			if bg is None:
				continue
			pop = [bg]
		for i, g in enumerate(pop[:_TOP_K]):
			cand.append((f"{label}#{i}" if len(pop) > 1 else label, label, spec, g))

	scored: dict = {}
	cand_meta: dict = {}
	for key, label, spec, genome in cand:
		per_seed = []
		for vs in VAL_SEEDS:
			try:
				vm = _holdout_report(args, ec, spec, genome, None,
				                     vs, seeds.train, stage_label=f"{key}-VAL{vs}")
			except HoldoutScoringError:
				raise
			except Exception as e:
				# FAIL FAST — same rule as _maybe_holdout above. Swallowing these
				# is how 25 of the 34 failures on 14/08 went unnoticed until the
				# run died: every candidate was "excluded from selection" until
				# nothing was left and the headline silently fell back to MEMORY.
				print(f"  [stage-select] {key}: val seed {vs} failed ({e})")
				raise HoldoutScoringError(
					f"stage-select {key}: val seed {vs} failed: {e}") from e
			if vm is not None:
				per_seed.append(vm)
		if per_seed:
			scored[key] = _mean_metric(per_seed)
			cand_meta[key] = (label, spec, genome)
		else:
			print(f"  [stage-select] {key}: no val draw scored — excluded from selection")

	# ONE ranking over the union of candidates, using the run's own weights.
	winner, whms = None, {}
	if scored:
		labels = list(scored)
		# Aggregation via _select_aggregation: ARITHMETIC by default (19/08, Luiz —
		# the harmonic combine selected specialists: arm 9's headline won on steady
		# rank-1 ALONE, dead last on jerk at nearly zero cost, 72% of its score from
		# one cell; the weights read as trade-offs, so the selector must honour all
		# of them). When --fit-aggregation is set, the run is coherent under that
		# one mode end-to-end — the fitness A/B's contract.
		calc = FitnessCalculatorControllerHarmonic(
			weight_err_sq=args.fit_weight_err_sq, weight_stable=args.fit_weight_stable,
			weight_jerk=args.fit_weight_jerk, weight_mono=args.fit_weight_mono,
			weight_steady=getattr(args, "fit_weight_steady", 0.0),
			weight_effort=getattr(args, "fit_weight_effort", 0.0),
			weight_alt=getattr(args, "fit_weight_alt", 0.0),
			weight_pos=getattr(args, "fit_weight_pos", 0.0),
			aggregation=_select_aggregation(args),
			zrank_clamp=getattr(args, "zrank_clamp", 3.0),
			gate_stable_min=_gate_args(args)[0],
			gate_err_max=_gate_args(args)[1])
		try:
			vals = calc.fitness([scored[l] for l in labels])
			whms = dict(zip(labels, vals))
			winner = min(whms, key=lambda l: whms[l])
		except Exception as e:
			print(f"  [stage-select] union ranking failed ({e}) — falling back to val steady")
			winner = min(labels, key=lambda l: (scored[l].mean_steady_error_deg
			                                    if scored[l].mean_steady_error_deg is not None else 9e9))

	print("\n" + "=" * 72)
	print("  STAGE TABLE — every stage published; headline = val-selected (union rank)")
	print("=" * 72)
	win_label = cand_meta[winner][0] if winner in cand_meta else winner
	for label, _spec, _res in stage_entries:
		ho = stage_holdouts.get(label.upper())
		# stable/err lead, then the FULL row from the one _HELDOUT_ROW declaration
		# — steady, jerk, mono, alt, pos, effort. This site used to hand-list
		# stable/err/steady and stop, so every stage but the headline reported no
		# altitude at all: a sweep pre-registered on held-out altitude could not
		# read altitude off its own stage table. Exactly the allowlist disease
		# 352035f4 cured for the per-seed rows, still alive one print site over.
		# _heldout_row_str OMITS absent metrics rather than printing 0.0 — a zero
		# altitude reads as a genome holding height perfectly.
		triple = ("stable=%.1f%% err=%.2f°%s" % (
			ho.acc * 100, ho.mean_attitude_error_deg, _heldout_row_str(ho))
			) if ho is not None else "(no held-out)"
		# The stage row publishes pop[0]; each of its top-K candidates gets its own
		# val/whm line underneath, so a reader can see WHY one of them was chosen.
		print(f"  {label:<9} {triple}")
		for key in [k for k in scored if cand_meta[k][0] == label]:
			v = scored[key]
			val_s = ("val %.1f%%/%.2f°/%s" % (
				v.acc * 100, v.mean_attitude_error_deg,
				("%.2f°" % v.mean_steady_error_deg) if v.mean_steady_error_deg is not None else "n/a"))
			# EVERY ranked dimension must be visible in this line, or the table cannot
			# explain its own winner (first live table: a 0%-stable genome out-ranked
			# healthy ones purely on jerk+mono). alt/pos joined 19/08 — their absence
			# meant the alt-weighted arms' markers recorded candidate rows MISSING a
			# ranked dimension, so re-deriving a headline from the marker alone was
			# impossible. Same allowlist disease as the held-out row; same cure.
			aux = "rw %.2f jrk %s mono %s" % (
				v.reward if v.reward is not None else float("nan"),
				("%.4f" % v.motor_jerk_mean) if v.motor_jerk_mean is not None else "n/a",
				("%.1f" % v.mono_violations_total) if v.mono_violations_total is not None else "n/a")
			if getattr(v, "mean_altitude_error_m", None) is not None:
				aux += " alt %.3fm" % v.mean_altitude_error_m
			if getattr(v, "mean_position_error_m", None) is not None:
				aux += " pos %.3fm" % v.mean_position_error_m
			# "fit=", not "whm=": the combine step is selectable, and a label
			# hardcoding one mean would disguise exactly the change that moves
			# headlines (arm 9: MEMORY#0 under WHM, CONNECTIONS#0 under arithmetic).
			fit_s = ("fit=%.4f" % whms[key]) if key in whms else "fit=n/a"
			mark = "  <- HEADLINE" if key == winner else ""
			print(f"    {key:<14} {val_s:<26} {aux:<44} {fit_s}{mark}")
	# The combine word is READ OFF THE CALCULATOR, never hardcoded. It was
	# hardcoded to "arithmetic" until 20/08/2026, so a --fit-aggregation
	# harmonic run printed "ZRank(...), ONE arithmetic weighted-rank" the moment
	# the flag was honoured end-to-end — a footer contradicting the very name
	# beside it. Same failure class as the 18/08 grid label that printed no
	# `alt=`: a log describing the ranking wrongly is how a silent no-op survives
	# review. If the calculator exposes no aggregation, say so rather than guess.
	combine = {"harmonic": "harmonic", "arithmetic": "arithmetic",
	           "zscore": "z-score",
	           "desirability": "desirability"}.get(getattr(calc, "aggregation", None), "unnamed")
	print(f"  (published triple = REPORT seeds, always pop[0]; 'val' = mean over {len(VAL_SEEDS)} "
	      f"disjoint val seeds {VAL_SEEDS[0]}..{VAL_SEEDS[-1]}; fit = {calc.name if scored else 'n/a'}, "
	      f"ONE {combine} weighted-rank over ALL {len(scored)} candidates — top-{_TOP_K} of every stage together)")
	if winner is None:
		print("  [stage-select] no stage could be scored on val — headline falls back to MEMORY")
		return None
	print(f"  [stage-select] HEADLINE stage={win_label} genome={winner} (union rank over "
	      f"{len(scored)} candidates = top-{_TOP_K} of every stage, on {len(VAL_SEEDS)} val "
	      f"seeds; NEVER on the report seeds)")
	# The headline triple must describe the genome that was SELECTED. When that is
	# pop[0] the stage's own report block already holds it; when the ranking picked a
	# runner-up, score THAT genome on the report seeds rather than quoting pop[0]'s
	# numbers under its name (one genome x report seeds — cheap, and the alternative
	# is a mislabelled published result).
	wh = stage_holdouts.get(win_label.upper())
	# `wh is None` covers --recalc-headline, where the per-stage report blocks are
	# not re-derived: score the selected genome rather than print no triple at all.
	if winner in cand_meta and (wh is None
	                            or (not winner.endswith("#0") and winner != win_label)):
		_lbl, _spec, _g = cand_meta[winner]
		try:
			wh = _maybe_holdout(args, ec, _spec,
			                    SimpleNamespace(best_genome=_g, final_population=None),
			                    seeds, label=f"HEADLINE-{winner}")
		except Exception as e:
			print(f"  [stage-select] could not score the selected runner-up ({e}) — "
			      f"headline triple below is stage pop[0], NOT the selected genome")
	if wh is not None:
		# 15/08/2026: pos= belongs HERE too. Every other held-out surface carries it
		# (RESULT line, per-stage multi-seed block, rival rows), but the headline —
		# the one row that gets pasted into a table — printed the triple alone, so a
		# reader could not tell whether the selected genome held position or drifted.
		# When the ranking picks a runner-up, `wh` is that genome's own score, so the
		# position number belongs to the same genome as the triple.
		# The full row, from the one _HELDOUT_ROW declaration. This is THE line an
		# analysis reads to rank arms by a fitness formula, so a metric missing here
		# is a weight that silently cannot be applied — jerk is 20% of C10.
		print(f"  [stage-select] HEADLINE held-out: stable={wh.acc * 100:.1f}% "
		      f"err={wh.mean_attitude_error_deg:.2f}°{_heldout_row_str(wh)}")
	return win_label


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
			geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None),
		episode_config=_calib_ec(args, ec))
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
	pid_rows = _pid_baseline(ec, rep_eps, report_seed, getattr(args, "num_eval_folds", 5))
	def _ms(xs):
		return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)
	ms_s, ms_e = _ms(stables), _ms(errs)
	bar = "=" * 72
	print(f"\n{bar}\n  HELD-OUT REPORT [{stage_label}] (report-only) — population ({len(pop)} genomes) on "
	      f"FRESH seed {report_seed}, train/select seed {train_seed}\n{bar}")
	# Steadiness (Luiz 08/07): steady-state error + monotonicity violations are
	# computed by the eval + weighted in the fitness — surface them here too.
	# Every metric of _HELDOUT_ROW that this scorer produced — see the note on that
	# declaration for why this is no longer a hand-written list per site.
	print(f"  RESULT — during-search winner (held-out):  stable={ds.acc*100:.1f}%  "
	      f"err={ds.mean_attitude_error_deg:.2f}°{_heldout_row_str(ds)}  reward={ds.fitness:.2f}")
	print(f"  population (held-out, descriptive):        stable={ms_s[0]:.1f}±{ms_s[1]:.1f}%  "
	      f"err={ms_e[0]:.2f}±{ms_e[1]:.2f}°   (pop max stable={pop_max:.1f}% — NOT selected, would leak)")
	_print_baseline_rows(pid_rows)
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

	Called for stage_num 0 (GRID, from _run_one) and 1-4 (from
	ControllerOrchestrator.run_phase), so the checkpoint pool covers EVERY stage
	a run produces a winner for — that is what makes cross-stage member analysis
	(e.g. scoring grid/neurons/memory winners against each other) possible
	without re-flying the cohort.

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
	# checkpoint (the 3-min/1.6GB hit pre-packing).
	#
	# SYNC since 11/08/2026 (was async). Async paid for its race-free snapshot by
	# eagerly encoding the WHOLE population on the main thread — a multi-GB
	# transient that landed at the exact moment the box is fullest, and is what
	# pushed avail under the watchdog floor and SIGKILLed the sn=8 run ONE SECOND
	# into writing its FINISHED NEURONS stage (stage1_neurons.yaml.gz.tmp.91742).
	# This call sits BETWEEN stages — sequential code, nothing mutates the
	# population while we write — so the snapshot bought nothing here, and the
	# sync path streams the encode one genome at a time (peak = 1 genome instead
	# of ~50). Cost: a few minutes of wall-clock once per stage, on the stage
	# boundary. (The in-stage crash manager went sync too, later the same day —
	# the sampler MEASURED its eager encode spiking +14 GB at a gen boundary; see
	# `_wire_cancel`.)
	_save_winner(str(path), args, spec, res.best_genome, res.final_population, metrics,
	             stage_num=stage_num, stage_name=stage_name, async_save=False)


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
		# STUDENT-STATE THRESHOLD REFIT (option A). The ladder above was fitted on
		# PID rollouts — a BETTER controller than the student — so it under-covers
		# the excursions the student actually makes. Roll the grid winner out,
		# refit on ITS states, and REGRID: new thresholds move the address
		# function, so every cell trained under the old ladder is stale (the
		# paper-critical THRESHOLD MISALIGNMENT finding). The first winner is a
		# state generator only — discarded, not carried.
		_thr2 = _refit_thresholds_from_student(
			args, ec, seeds, _grid_for_refit, grid_res.best_genome)
		if _thr2 is not None:
			winner_spec, seed_pop0, m0, dt0, _thr = stage0_grid(
				args, ec, seed, thresholds_override=_thr2)
			stage_results = [("Grid", winner_spec, m0, dt0, grid_point_count(args))]
			grid_res = _SNS(best_genome=(seed_pop0[0] if seed_pop0 else None),
			                final_population=seed_pop0)
		grid_ho = _maybe_holdout(args, ec, winner_spec, grid_res, seeds, "GRID")
		if grid_ho is not None:
			# Keep it LOCALLY too: `stage_holdouts` is the flow_runner's out-dict and is
			# None on a standalone run, so guarding the store on it dropped GRID's triple
			# and the stage table printed "(no held-out)" for a stage it had just measured.
			if stage_holdouts is not None:
				stage_holdouts["GRID"] = grid_ho
			_grid_stage_entry = ("GRID", winner_spec, grid_res)
			_grid_holdout = grid_ho
		else:
			_grid_stage_entry = None
			_grid_holdout = None
		# Stage-0 checkpoint (added 08/08/2026). Stages 1-4 dump through the
		# orchestrator's _save_stage_checkpoint; Stage 0 runs BEFORE the orchestrator
		# and had no dump at all, so a cohort's checkpoint pool held NEURONS + MEMORY
		# members but no GRID ones — the "score stage winners from grid, neurons and
		# memory" follow-up was silently impossible without re-flying every run.
		#
		# DELIBERATELY AFTER _maybe_holdout, not before. A raw grid winner is
		# arch-only (cells=None) and an arch-only checkpoint cannot be scored — it is
		# resume-fodder, not a committee member. _holdout_report's arch-only branch
		# stamps train-seed cells INTO these same genome objects (K-fold accumulate,
		# exactly as the search trains) before scoring them on the report seeds, so
		# saving afterwards captures a trained, directly-scoreable controller.
		# Caveat for anyone comparing it: those cells were trained with
		# --report-episodes, not --eval-episodes, so this is the grid winner as the
		# HELD-OUT measured it, not as the search saw it mid-selection.
		# With no report seed set the holdout is skipped and this falls back to the
		# arch-only dump (still useful: --resume, and the top-K pool for warm-start).
		_save_stage_checkpoint(args, 0, "grid", winner_spec, grid_res, m0)
	else:
		seed_pop0 = None
		_grid_stage_entry = None   # resume skips the grid — nothing to score or publish
		_grid_holdout = None
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
	if _grid_holdout is not None:
		_all_ho.setdefault("GRID", _grid_holdout)
	# GRID + every stage the orchestrator actually ran. The stage list comes from
	# the orchestrator's own registry (see ControllerOrchestrator.stage_entries) —
	# naming stages here is what dropped BITS and CONNECTIONS from selection.
	_entries = [e for e in [_grid_stage_entry] if e is not None]
	_entries.extend(orch.stage_entries())
	if _entries:
		try:
			_select_headline_stage(args, ec, seeds, _entries, _all_ho)
		except Exception as e:
			print(f"  [stage-select] skipped ({e}) — per-stage held-outs above are unaffected")

	# PID baseline on the val seed (the held-out reference).
	pid_rows = _pid_baseline(ec, args.eval_episodes, seeds.val,
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
		return stage_results, None, None, pid_rows
	return stage_results, res4.best_genome, res4.final_population, pid_rows


def _print_final_summary(args, stage_results, best_final, pid_rows, total_dt: float):
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
	# rows[0] is the RIVAL; the rest are informational (see _pid_baseline).
	for _i, _row in enumerate(pid_rows):
		# reward is None for the pool-seeded scorer (it returns stability/error/steady,
		# and a fabricated reward in a comparison row is worse than an absent one).
		_rw = _row.get("mean_reward")
		_tag = "" if _i == 0 else "   (informational)"
		print(f"  vs {_row.get('label', 'PID')}:  {_row['mean_attitude_error_deg']:.2f}° / "
		      f"{_row['stable_rate']*100:.0f}% / "
		      f"{'—' if _rw is None else format(_rw, '.2f')}{_tag}")
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
	# --- SCOPE C STAGE 1 (13/08/2026): the vertical channel. All default OFF, so
	#     every pre-stage-1 recipe reproduces bit-identically.
	#     docs/scope_c_full_controller_spec.md
	ap.add_argument("--obs-collective-cmd", action=argparse.BooleanOptionalAction, default=False,
	                help="Stage 1: feed the controller the COMMANDED COLLECTIVE from the outer "
	                     "loop (1 feature). This is what makes the controller composable — any "
	                     "outer loop, including pybullet's DSLPID, can drive it. Default OFF.")
	ap.add_argument("--obs-alt-err", action=argparse.BooleanOptionalAction, default=False,
	                help="Stage 1: feed the controller its ALTITUDE ERROR (target − z, 1 feature). "
	                     "It cannot hold what it cannot see. Requires --translation. Default OFF.")
	ap.add_argument("--obs-vz", action=argparse.BooleanOptionalAction, default=False,
	                help="Stage 1: feed the controller its VERTICAL VELOCITY (1 feature) — the "
	                     "damping channel. Requires --translation. Default OFF.")
	ap.add_argument("--translation", action=argparse.BooleanOptionalAction, default=False,
	                help="Stage 1: integrate vertical translation in the sim "
	                     "(v̇z = (ΣT·cosθ)/m − g). Mass comes from the airframe and is a PLANT "
	                     "parameter — randomized, never a feature. Default OFF (bit-identical to "
	                     "every attitude-only result).")
	ap.add_argument("--calib-airframe", action=argparse.BooleanOptionalAction, default=False,
	                help="Fit the thermometer ladder on the AIRFRAME (and its firmware "
	                     "cascade) instead of the historical synthetic plant. Default OFF "
	                     "= every banked run reproduces. Adopting this is a LINEAGE BREAK "
	                     "(~85%% of addresses move) — see task #11's paired A/B.")
	ap.add_argument("--alt-offset", type=float, default=0.3,
	                help="Stage 1: initial altitude offset bound (m); z0 ~ U(-x, x).")
	ap.add_argument("--init-vz", type=float, default=0.2,
	                help="Stage 1: initial vertical-velocity bound (m/s).")
	ap.add_argument("--collective-jitter", type=float, default=0.1,
	                help="Stage 1: commanded-collective variation as a fraction of hover.")
	ap.add_argument("--mass-jitter", type=float, default=0.15,
	                help="Stage 1: per-episode mass randomization fraction. A PLANT "
	                     "parameter — randomized, never observed (Molchanov randomizes "
	                     "thrust-to-weight and never inputs it).")
	ap.add_argument("--target-altitude", type=float, default=0.0,
	                help="Stage 1: the altitude every episode holds (m).")
	ap.add_argument("--reward-lambda-alt", type=float, default=0.0,
	                help="Stage 1 REWARD SHAPING (λ_alt): weight on the altitude-error term "
	                     "INSIDE the per-step reward, -λ_alt·alt_err². 0.0 = OFF "
	                     "(bit-identical). ⚠️ This λ carries the metres↔radians unit "
	                     "conversion, so its value is tied to the CAPACITY it was swept at "
	                     "— see the rename note on --fit-weight-alt. Prefer the rank weight.")
	ap.add_argument("--obs-pos-err-xy", action=argparse.BooleanOptionalAction, default=False,
	                help="Stage 2: feed the controller its HORIZONTAL POSITION ERROR "
	                     "(e_x, e_y — 2 features; one flag carries BOTH axes, x and y are "
	                     "the same physics rotated 90°). Requires --translation and "
	                     "--xy-offset > 0. Default OFF.")
	ap.add_argument("--obs-vel-xy", action=argparse.BooleanOptionalAction, default=False,
	                help="Stage 2: feed the controller its HORIZONTAL VELOCITY (v_x, v_y — "
	                     "2 features), the damping channel. Same gating. Default OFF.")
	ap.add_argument("--xy-offset", type=float, default=0.0,
	                help="Stage 2: initial horizontal offset bound (m) per axis; episodes "
	                     "start displaced AT REST. 0.0 = the horizontal channel is unarmed "
	                     "(bit-identical to stage 1, and the trainer draws NOTHING so the "
	                     "rng sequence of stage-1 runs is untouched).")
	ap.add_argument("--reward-lambda-pos", type=float, default=0.0,
	                help="Stage 2 REWARD SHAPING (λ_pos): weight on the RADIAL horizontal "
	                     "position error λ_pos·(e_x²+e_y²) INSIDE the per-step reward. "
	                     "0.0 = OFF (bit-identical). Radial, not per-axis — per-axis weights "
	                     "would let the GA learn a compass direction that exists only in the "
	                     "reward. Same unit-carrying caveat as λ_alt; prefer --fit-weight-pos.")
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
	ap.add_argument("--output-full-window", action=__import__("argparse").BooleanOptionalAction, default=False,
	                help="ARM D (14/08): the OUTPUT layer samples the full K-frame window "
	                     "(same layout the state layer reads) instead of frame t-0 only. "
	                     "Requires sn=0 (--grid-state-neurons 0 --max-state-neurons 0).")
	ap.add_argument("--frame-stride", type=int, default=1,
	                help="Frame stride (15/08): the K-window shifts once every N pushes, so it "
	                     "spans N*K steps instead of K. At dt=1ms, k=4/stride=1 is a 4 ms lookback "
	                     "where t-1 barely differs from t-0 (the rate gyro already carries that "
	                     "derivative); stride=10 gives 40 ms. Newest slot always holds the CURRENT "
	                     "frame (sample-and-hold), so reactivity is never traded away. 1 = legacy.")
	ap.add_argument("--target-levels", type=int, default=0,
	                help="TARGET-LEVELS redundancy (arm R, 16/08/2026): output neurons per motor "
	                     "share this many distinct thermometer thresholds (proportional map; the "
	                     "sum decode is unchanged, so redundant-group errors average out while "
	                     "thresholds stay learnable). 0 = legacy. Requires the >=16/08 wheel — "
	                     "the evaluator fails loudly on an older one rather than train legacy targets.")
	ap.add_argument("--conn-policy", choices=["spread", "min1", "min2", "min3", "framed1"], default="spread",
	                help="Connection-creation policy for fresh OUTPUT maps (14/08 specialist "
	                     "programme): spread = legacy uniform; min1 = full FEATURE COVERAGE (every "
	                     "feature gets >=1 threshold — at b=num_features, exactly one each); "
	                     "framed1 = each neuron picks ONE frame "
	                     "(recency weights 2^slot, so 8:4:2:1 at k=4) and min1-covers it; requires "
	                     "--output-full-window. min2/min3 = MIN_PER_CLUSTER(m) — "
	                     "every touched feature gets >= m thresholds (m=2 makes interval "
	                     "detection the floor), unaffordable features dropped, remainder donated.")
	ap.add_argument("--conn-mutation-scope", choices=["free", "window", "feature"], default="free",
	                help="GA-connectivity mutation scope (16/08, Luiz's connectivity types): where a "
	                     "CONNECTIONS-stage rewire may land. free = legacy (anywhere — can leave the "
	                     "feature and even the window); window = never crosses time (at k=1 this "
	                     "degenerates to free); feature = only WHERE on the feature moves, the "
	                     "feature map is frozen at what grid/init chose. Output maps only.")
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
	ap.add_argument("--dob", dest="dhat_ff", action="store_true",
	                help="OUTPUT-SIDE disturbance observer: subtract clamp(d-hat/b) from the "
	                     "policy's motors each step, exactly as the mpcof teacher does "
	                     "(u = policy - d-hat/b) — the one line that makes it post 0.00 steady. "
	                     "The LUT is unchanged; the trim is downstream, which is why L1 "
	                     "(d-hat as an INPUT feature, refuted 4/4) does not settle this. "
	                     "Requires --dhat-b. ~6 flops/axis/step. Report as 'WNN + DOB'.")
	ap.add_argument("--dob-clamp", dest="dhat_ff_clamp", type=float, default=0.30,
	                help="Per-axis bound on the DOB feedforward d-hat/b (teacher default 0.30). "
	                     "b can be small, so an unclamped ratio could peg the actuator.")
	ap.add_argument("--delta-gamma", type=float, default=1.0,
	                help="Non-uniform delta alphabet: the decode's normalized offset t is "
	                     "shaped |t|^gamma before scaling to +/-delta_max. Same range, "
	                     "neutral, level count and FOOTPRINT — resolution concentrated near "
	                     "zero where the hold window lives. gamma=2 makes the finest step ~8x "
	                     "finer at 16 levels with no extra neurons (raising --levels to 64 "
	                     "costs 3x cells for an unreliable gain). 1.0 = the original "
	                     "piecewise-linear map, bit-identical.")
	ap.add_argument("--threshold-refit-from-student", action="store_true",
	                help="After the first GRID pass, roll out the grid winner, refit the "
	                     "thermometer on the STUDENT's own visited states (concatenated "
	                     "with the teacher pool), and RE-RUN the grid under the new "
	                     "ladder. The fitter otherwise rolls out PID, a better controller "
	                     "than the student, so the ladder under-covers the excursions the "
	                     "student actually makes — DAgger covariate shift in the input "
	                     "encoding, where training cannot repair it. Costs one extra GRID "
	                     "stage; the regrid is MANDATORY because a refit moves the address "
	                     "function and staleness every learned cell.")
	ap.add_argument("--threshold-refit-episodes", type=int, default=10,
	                help="Student rollout episodes for the refit (default 10, matching the "
	                     "teacher pool). Too few and the student's samples are swamped by "
	                     "the teacher's — measured, 2 samples/feature moved the ladder "
	                     "1.00x, i.e. the refit becomes a placebo that looks implemented.")
	ap.add_argument("--threshold-outer-quantile", type=float, default=None,
	                help="Coverage margin: outermost thermometer quantiles span "
	                     "[q, 1-q]. Default (None) keeps 1/(b+1), b/(b+1) — 0.111/0.889 at "
	                     "b=8, so ~22%% of the operating distribution saturates to an "
	                     "all-0/all-1 code BY CONSTRUCTION. 0.02 reaches into the tails "
	                     "(measured 1.35x wider ladder). Saturation costs recovery, not "
	                     "just precision: calib=5deg lost stable as well as steady, 2/2 seeds.")
	ap.add_argument("--threshold-calib-tilt", type=float, default=None,
	                help="Initial tilt (DEGREES) for the PID rollouts that calibrate the "
	                     "thermometer thresholds. Default: the run's own --tilt (calibrate on "
	                     "the regime you fly). Set NARROWER than --tilt to concentrate bins "
	                     "near zero where the hold/steady metric lives, at the cost of "
	                     "saturating the transient. Measured saturation vs a flown 5deg "
	                     "distribution: 30deg=11.3%% outside the ladder, 5deg=31.0%%, "
	                     "2.5deg=39.6%%, 1deg=59.1%% (1deg is over the cliff).")
	ap.add_argument("--stage-select-top-k", type=int, default=3,
	                help="How many genomes per stage enter the headline union ranking (default 3). "
	                     "All K x stages candidates are ranked in ONE population, so pop[0] — the "
	                     "genome that IS the published result — always competes, and the ranking is "
	                     "not compressed onto 3 rank slots. K also bounds what survives the "
	                     "orchestrator's population release (~120-330 MB/genome).")
	ap.add_argument("--recalc-headline", type=str, default=None, metavar="CKPT_DIR",
	                help="REPORT-ONLY re-selection: skip the search entirely, rebuild the "
	                     "stage candidates from the stage checkpoints in CKPT_DIR and re-run "
	                     "the val-based headline selection. Every other flag must match the "
	                     "original run (they define the episodes, the seeds and the scoring); "
	                     "nothing is trained and nothing is written. Added 17/08/2026 to "
	                     "re-headline runs flown while CONNECTIONS/BITS were excluded from "
	                     "the candidate pool, without re-flying a 4 h search.")
	ap.add_argument("--fit-weight-steady", type=float, default=0.0,
	                help="Weight on mean_steady_error_deg (mean attitude err over the last 20%% of steps) "
	                     "in the harmonic-rank fitness. The I-pressure term: isolates the steady-state "
	                     "offset only an integrator can kill. Default 0. >0 activates multi-objective.")
	# ⚠️ RENAMED 18/08/2026 — this flag USED to be the reward λ_alt. It is now a RANK
	# weight, and the reward term moved to --reward-lambda-alt. Passing an old λ value
	# (e.g. 16) here would make altitude swamp the rank; _validate_rank_weights refuses
	# anything > 1.0 and names the rename. See the sweep-ladder post-mortem: a rank is
	# SCALE-FREE, so it does not carry the metres↔radians conversion that tied λ_alt to
	# the capacity it was swept at.
	ap.add_argument("--fit-weight-alt", type=float, default=0.0,
	                help="Weight on mean_altitude_error_m in the harmonic-rank fitness — the "
	                     "altitude channel as its own RANK dimension. Being a rank, it is "
	                     "scale-free: metres never compete numerically with radians. Default 0. "
	                     "NOT the reward term — that is --reward-lambda-alt.")
	ap.add_argument("--fit-weight-pos", type=float, default=0.0,
	                help="Weight on mean_position_error_m in the harmonic-rank fitness — the "
	                     "horizontal channel as its own RANK dimension, same reasoning as "
	                     "--fit-weight-alt. Inert until --xy-offset > 0 arms stage 2 (with the "
	                     "channel unarmed the metric is a constant and every genome ties). "
	                     "Default 0. NOT the reward term — that is --reward-lambda-pos.")
	# Fitness aggregation (19/08/2026). Unset = the legacy split the banked runs
	# used: harmonic in-search + arithmetic stage-select. Set = ONE mode end-to-
	# end (grid ranking, GA elitism/incumbent, TS, stage-select) — the coherent
	# contract of the harmonic-vs-zscore fitness A/B. The math for all three
	# lives in ram_core::fitness (the wheel), not in Python.
	ap.add_argument("--fit-aggregation", choices=["harmonic", "arithmetic", "zscore", "desirability"],
	                default=None,
	                help="Rank-combine aggregation, applied EVERYWHERE when set: harmonic = "
	                     "legacy WHM (specialist-friendly: dominated by the best weighted "
	                     "rank); arithmetic = every rank hurts in proportion to its weight; "
	                     "zscore = winsorized robust z — magnitude-aware, 1st by 13° no "
	                     "longer counts the same as 1st by 0.1°. Unset = harmonic in-search "
	                     "+ arithmetic stage-select (the legacy/banked behavior).")
	ap.add_argument("--gate-stable", type=float, default=None,
		help="Viability gate: minimum stable_rate FRACTION (0.70 = 70%%). Set with "
		     "--gate-err or not at all. Candidates below the gate rank behind every "
		     "flyer, ordered by how close they are to flying (Deb's rules); flyers "
		     "rank by the base combine over the FLYING SUBSET only. Applies to all "
		     "three sites: search, stage-select, grid. Approved 21/08/2026: 0.70.")
	ap.add_argument("--gate-err", type=float, default=None,
		help="Viability gate: maximum mean attitude error in DEGREES. Set with "
		     "--gate-stable or not at all. Approved 21/08/2026: 8.0.")
	ap.add_argument("--zrank-clamp", type=float, default=3.0,
	                help="Winsorization bound for --fit-aggregation zscore: per-metric robust "
	                     "z is clamped to ±this, so no single dimension can capture the score "
	                     "however extreme the outlier (the λ_alt lesson). Default 3.0.")
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
	# CRN fitness (03/09/2026, Luiz). DEFAULT ON: every genome is scored on ALL K
	# pools every generation and trains on the SAME fold seeds, so the fitness is a
	# deterministic function of the genome and elites' cached scores are honest.
	# --no-score-crn = the 30/05 per-generation pool ROTATION, under which an elite
	# kept its lucky-pool number for the whole stage (measured: 5-gen CONNECTIONS
	# stages flat at (=) while the population's held-out moved 50pp). Search-only:
	# the held-out report evaluators never see this flag.
	ap.add_argument("--score-crn", action=argparse.BooleanOptionalAction, default=True,
	                help="Common-random-numbers fitness: score every genome on all K pools every "
	                     "generation + shared training seeds (default ON; --no-score-crn = legacy "
	                     "per-generation pool rotation).")
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


def _validate_rank_weights(args) -> None:
	"""Refuse a reward λ passed to a RANK weight — the 18/08/2026 rename's tripwire.

	`--fit-weight-alt` USED to be the reward λ_alt and was routinely passed as 16.
	It is now a rank weight, where every other member of the C10/S16 family lives in
	[0, 1]. A stale caller passing 16 would not crash: altitude would simply take
	~94% of the rank mass and every genome would be selected on altitude alone —
	silent, and exactly the failure the rename exists to prevent. So bound it and
	name the rename in the error."""
	for flag, value, reward_twin in (
		("--fit-weight-alt", getattr(args, "fit_weight_alt", 0.0), "--reward-lambda-alt"),
		("--fit-weight-pos", getattr(args, "fit_weight_pos", 0.0), "--reward-lambda-pos"),
	):
		if float(value) > 1.0:
			raise SystemExit(
				f"{flag}={value} is out of range for a RANK weight (expected 0..1, "
				f"like every other --fit-weight-*).\n"
				f"RENAMED 18/08/2026: {flag} used to be the REWARD lambda and took "
				f"values such as 16. That term is now {reward_twin}.\n"
				f"  • want the reward term back?   {reward_twin} {value}\n"
				f"  • want the rank dimension?     {flag} 0.10   (a weight, not a lambda)\n"
				f"Why: a lambda multiplies metres against radians inside the reward, so its "
				f"tuned value is bound to the capacity it was swept at. A rank is scale-free.")


def main():
	args = build_arg_parser().parse_args()
	_validate_rank_weights(args)

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
		# SCOPE C STAGE 1 (13/08/2026): the vertical channel. --translation off
		# ⇒ every field below is inert and the run is bit-identical to a
		# pre-stage-1 one. Episode axes default to a modest spread so the
		# controller is actually ASKED to correct altitude — a controller that
		# always starts at its target has never had to.
		translation=bool(getattr(args, "translation", False)),
		max_initial_alt_offset_m=float(getattr(args, "alt_offset", 0.3)),
		max_initial_vz=float(getattr(args, "init_vz", 0.2)),
		collective_cmd_jitter=float(getattr(args, "collective_jitter", 0.1)),
		mass_jitter=float(getattr(args, "mass_jitter", 0.15)),
		target_altitude=float(getattr(args, "target_altitude", 0.0)),
		lambda_alt=float(getattr(args, "reward_lambda_alt", 0.0)),
		# SCOPE C STAGE 2 (14/08/2026): the horizontal channel. --xy-offset 0.0
		# ⇒ unarmed, bit-identical to stage 1.
		max_initial_xy_offset_m=float(getattr(args, "xy_offset", 0.0)),
		lambda_pos=float(getattr(args, "reward_lambda_pos", 0.0)),
		calib_airframe=bool(getattr(args, "calib_airframe", False)),
	)
	# STAGE 1 GUARD: the vertical FEATURES read the sim's z/vz, so enabling them
	# without --translation would feed the controller a permanently-zero channel
	# — three wasted features and a silently different address space. Refuse
	# loudly rather than fly a run whose result means nothing.
	if not ec.translation:
		_vert_on = [n for n in ("obs_collective_cmd", "obs_alt_err", "obs_vz")
		            if getattr(args, n, False)]
		if _vert_on:
			raise SystemExit(
				f"{'/'.join('--' + n.replace('_', '-') for n in _vert_on)} require "
				"--translation: without it the sim has no altitude, so those features "
				"would be constant zeros.")
		if float(getattr(args, "reward_lambda_alt", 0.0)) != 0.0:
			raise SystemExit("--fit-weight-alt requires --translation: there is no "
			                 "altitude to reward without it.")
	if ec.translation and ec.airframe is None:
		raise SystemExit("--translation requires --airframe: mass is a PLANT parameter "
		                 "and the synthetic default has none.")
	# STAGE 2 GUARD, same reasoning one level up: the horizontal FEATURES read
	# x/y, which never leave the origin unless episodes start displaced. Refuse
	# a run whose horizontal channel would be constant zeros.
	_horiz_on = [n for n in ("obs_pos_err_xy", "obs_vel_xy") if getattr(args, n, False)]
	if _horiz_on and not (ec.translation and ec.max_initial_xy_offset_m > 0.0):
		raise SystemExit(
			f"{'/'.join('--' + n.replace('_', '-') for n in _horiz_on)} require "
			"--translation AND --xy-offset > 0: without them x/y never leave the "
			"origin, so those features would be constant zeros.")
	if float(getattr(args, "reward_lambda_pos", 0.0)) != 0.0 \
			and not (ec.translation and ec.max_initial_xy_offset_m > 0.0):
		raise SystemExit("--fit-weight-pos requires --translation and --xy-offset > 0: "
		                 "there is no horizontal error to reward without them.")
	# ARM D sanity gate: the Rust constructor refuses sn>0 + full window, but
	# failing at arg-parse beats failing 30 min into the grid.
	if getattr(args, "output_full_window", False):
		_sn = getattr(args, "grid_state_neurons", 0)
		if isinstance(_sn, str):
			sn_axis = [int(x) for x in _sn.split()]
		elif isinstance(_sn, (list, tuple)):
			sn_axis = [int(x) for x in _sn]
		else:
			sn_axis = [int(_sn)]
		if any(v > 0 for v in sn_axis) or int(getattr(args, "max_state_neurons", 0)) > 0:
			raise SystemExit("--output-full-window requires sn=0 everywhere "
			                 "(--grid-state-neurons 0 --max-state-neurons 0): arm D is single-layer.")
	# Connection-creation policy (14/08 specialist programme): parse "min2"/"min3"
	# into (policy, m) ONCE and stash on args — same pattern as _dhat_b below.
	_cp = getattr(args, "conn_policy", "spread")
	if _cp == "framed1":
		args._conn_policy = "framed1"
	elif _cp.startswith("min"):
		args._conn_policy = "min_per_cluster"
	else:
		args._conn_policy = "spread"
	args._conn_policy_min = int(_cp[3:]) if _cp.startswith("min") else 2
	# framed1 is meaningless without arm D: with sn=0 and the legacy layout the
	# output layer sees ONE frame whatever k is, so every neuron would land in
	# the same frame and the arm would silently be min1. Fail at arg-parse.
	if _cp == "framed1" and not getattr(args, "output_full_window", False):
		raise SystemExit("--conn-policy framed1 requires --output-full-window: without it the "
		                 "output layer sees only frame t-0, so per-frame specialisation is a no-op "
		                 "(it would silently run as min1).")
	if args._conn_policy == "min_per_cluster":
		print(f"[conn-policy] {_cp}: fresh OUTPUT maps drawn MIN_PER_CLUSTER"
		      f"(m={args._conn_policy_min}) — touched features get >= m thresholds, "
		      f"unaffordable features dropped, remainder donated")
	if _cp == "framed1":
		_k = int(getattr(args, "input_window_k", 4))
		_w = [2 ** s for s in range(_k)]
		_tot = sum(_w)
		print(f"[conn-policy] framed1: each output neuron covers ONE frame completely "
		      f"(min1 within it); frame drawn with recency weights {_w[::-1]} "
		      f"(newest first) => ~{[round(100*w/_tot) for w in _w[::-1]]}% of neurons per frame")
	if int(getattr(args, "target_levels", 0)) > 0:
		print(f"[target-levels] T={args.target_levels}/motor: output neurons share T coarse "
		      f"thermometer thresholds (proportional map, redundant groups average in the "
		      f"sum decode). Training-side only; requires the >=16/08 wheel.")
	if getattr(args, "conn_mutation_scope", "free") != "free":
		print(f"[conn-scope] {args.conn_mutation_scope}: CONNECTIONS-stage rewires stay in the "
		      f"original bit's {'window (never cross time)' if args.conn_mutation_scope == 'window' else 'thermometer run (feature map FROZEN, only thresholds move)'}; "
		      f"output maps only, state maps keep the free draw. Requires the >=16/08 wheel.")
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
	      f"levels={args.levels} "
	      f"fitness_pools={'CRN(all ' + str(args.num_eval_folds) + ' pools/gen)' if args.score_crn else 'rotation(1 pool/gen)'}")

	# REPORT-ONLY re-selection: no search, no writes — rebuild the candidates from
	# the saved stage checkpoints and re-run the val-based headline selection.
	if getattr(args, "recalc_headline", None):
		_recalc_headline(args, ec)
		return

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
		stage_results, best_final, final_population, pid_rows = _run_one(args, ec, s,
		                                                              resume_state=resume_state)
		val_runs.append((stage_results, best_final, final_population, pid_rows))

	# Single-run path: print the per-run summary directly.
	stage_results, best_final, final_population, pid_rows = val_runs[-1]
	_print_final_summary(args, stage_results, best_final, pid_rows, time.time() - t_start)

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
