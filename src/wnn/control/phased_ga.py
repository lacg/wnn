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

# Periodic IN-STAGE checkpoint (crash protection during a stage). Unlike the
# emergency dump (signal/cancel only) and the between-stage save (stage end),
# this writes the live population on an adaptive wall-clock cadence so a HARD
# crash (segfault / OOM / power loss — no signal) loses at most ~one slow gen.
# Driven by the SHARED PhasedCheckpointManager (same orchestrator IDS uses): it
# owns the cadence + a single in-flight async writer, so writes never overlap on
# the pid-keyed temp file. Re-armed (fresh time baseline) per stage.
_periodic_mgr = None              # PhasedCheckpointManager | None


def _join_periodic_save() -> None:
	"""Block until the manager's in-flight periodic write finishes (≤1)."""
	if _periodic_mgr is not None:
		_periodic_mgr.join()


def _join_pending_saves() -> None:
	_join_periodic_save()
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


def _build_emergency_payload(emergency_dump: bool) -> dict:
	"""Snapshot the current emergency state into a checkpoint payload. Schema
	mirrors _save_winner so the resume path loads it like any other stage
	checkpoint. Shared by the signal/cancel dump (sync) and the periodic
	in-stage save (async)."""
	args = _emergency_state["args"]
	return {
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
			"steady": args.fit_weight_steady,
		},
		"meta": {
			"saved_at_unix":   time.time(),
			"saved_at_iso":    time.strftime("%Y-%m-%dT%H:%M:%S%z"),
			"emergency_dump":  emergency_dump,
			"levels":          args.levels,
			"tilt_deg":        args.tilt,
			"steps":           args.steps,
			"eval_episodes":   args.eval_episodes,
		},
	}


def _dump_emergency_state() -> None:
	"""Write the current emergency state synchronously (must finish before the
	process exits on signal). Joins any in-flight periodic write first so the two
	never race on the pid-keyed temp file."""
	path = _emergency_state.get("save_path")
	if path is None:
		print("[emergency-dump] No save_path set — cannot dump.", flush=True)
		return
	_join_periodic_save()
	payload = _build_emergency_payload(emergency_dump=True)
	p = Path(path)
	_ctl_save(p, payload)
	print(f"\n[emergency-dump] Stage {payload['stage_num']} ({payload['stage_name']}) "
	      f"gen {payload['generation']}, {len(payload['population'])} genomes → {p}",
	      flush=True)


def _maybe_periodic_save(generation: int) -> None:
	"""Adaptive in-stage checkpoint via the shared PhasedCheckpointManager: if the
	cadence is due, async-write the live population to the stage's save_path. Slow
	gens (≥budget) save every gen; fast gens throttle to ≤max_interval. No-op when
	the manager is unarmed (no save path) or the population is empty."""
	if _periodic_mgr is None or not _emergency_state.get("population"):
		return
	payload = _build_emergency_payload(emergency_dump=False)   # cheap (refs only)
	from wnn.control.checkpoint_io import payload_to_checkpoint
	if _periodic_mgr.maybe_save(generation, payload_to_checkpoint(payload)):
		print(f"[checkpoint] in-stage save: stage {payload['stage_num']} "
		      f"gen {generation}, {len(payload['population'])} genomes (async)", flush=True)


def _arm_periodic_cadence(args) -> None:
	"""(Re)build the in-stage checkpoint manager for the stage about to run, with
	a fresh cadence (each stage's first gen establishes its own time baseline).
	No save_path (no --save-winner/--save-stage-checkpoints) → manager stays None
	→ periodic save is a no-op. target_loss_seconds None → save every gen."""
	global _periodic_mgr
	path = _emergency_state.get("save_path")
	if path is None:
		_periodic_mgr = None
		return
	from wnn.ram.strategies.phased import (
		PhasedCheckpointManager, SaveCadence, ControllerGenomeCodec)
	budget = getattr(args, "checkpoint_target_loss_seconds", None)
	max_int = getattr(args, "checkpoint_max_interval", 10)
	_periodic_mgr = PhasedCheckpointManager(
		Path(path), ControllerGenomeCodec(), SaveCadence(budget, max_int),
		async_save=True)


def _install_emergency_hook(strat) -> None:
	"""Monkey-patch the strategy's _on_generation_start to (a) record the
	current population in the module-level emergency state, (b) write an adaptive
	in-stage checkpoint so a hard crash loses ≤~one slow gen, and (c) check the
	Rust cancel flag and dump+abort if set."""
	_arm_periodic_cadence(_emergency_state.get("args"))
	original = strat._on_generation_start
	def wrapped(generation, **ctx):
		# Capture current population (start-of-gen snapshot; carries elites +
		# selected offspring ready for the next gen — ideal for resume).
		_emergency_state["population"]  = list(ctx.get("population", []))
		_emergency_state["best_genome"] = ctx.get("best_genome")
		_emergency_state["generation"]  = generation
		# Periodic crash-protection save (adaptive cadence; no-op if disabled).
		try:
			_maybe_periodic_save(generation)
		except Exception:
			pass  # never let checkpointing break the GA
		# Check cancel and bail if requested.
		try:
			from wnn.control import _accel as ram_accelerator
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

def _make_spec(state_neurons: int, levels: int, bits: int,
               delta_control: bool = True, delta_leak: float = 0.95,
               obs_tilt_p: bool = False, obs_tilt_i: bool = False,
               obs_peraxis_p: bool = False, obs_peraxis_i: bool = False,
               obs_peraxis_yaw: bool = True,
               obs_pwm: bool = False,
               obs_yaw_err: bool = False, obs_yaw_err_i: bool = False,
               integral_leak: float = 0.99, integral_scale: float = 1.0,
               dt: float = 0.001,
               decouple_outputs: bool = False, bits_per_feature: int = 8,
               feature_balance_ratio: float = 0.0,
               threshold_gamma: float = 1.0,
               action_repeat: int = 1,
               output_bits: "int | None" = None) -> ControllerSpec:
	"""Build a ControllerSpec from a (state_neurons, levels, bits) grid point.
	`bits` becomes BOTH state_bits_per_neuron and output_bits_per_neuron, matching
	the grid-search convention (the GA can later split them in the BITS phase).

	delta_control (default True, 08/06/2026): the output decodes to a per-step PWM
	DELTA with a leaky accumulator — a structural integrator that offloads PID's
	I-term out of the (currently untrained) recurrent state. Empirically +5pp
	stability (71→76% @leak=0.95) vs the old hardcoded False. It's a PARTIAL fix
	(the policy is still memoryless — see project_controller_stability_diagnosis);
	the full fix is training the state as a learned integrator. The grid spec's
	delta_control/leak propagate to all later stages via spec_from_arch(base)."""
	return ControllerSpec(
		num_motors=4, levels_per_motor=levels, bits_per_feature=bits_per_feature, input_window_k=4,
		state_neurons=state_neurons,
		state_bits_per_neuron=bits, output_bits_per_neuron=(output_bits if output_bits is not None else bits),
		delta_control=delta_control, delta_leak=delta_leak,
		obs_tilt_p=obs_tilt_p, obs_tilt_i=obs_tilt_i,
		obs_peraxis_p=obs_peraxis_p, obs_peraxis_i=obs_peraxis_i,
		obs_peraxis_yaw=obs_peraxis_yaw,
		obs_pwm=obs_pwm,
		obs_yaw_err=obs_yaw_err, obs_yaw_err_i=obs_yaw_err_i,
		integral_leak=integral_leak, integral_scale=integral_scale,
		dt=dt,
		decouple_outputs=decouple_outputs,
		feature_balance_ratio=feature_balance_ratio,
		threshold_gamma=threshold_gamma,
		action_repeat=action_repeat,
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
	cov = getattr(args, "suffix_coverage", 0.0)
	# valid_pairs are (state_neurons, state_bits, output_bits) — PER-LAYER bits so the output
	# (1 frame of features) and state (windowed) layers can have DIFFERENT suffix widths.
	if cov > 0.0:
		# Per-layer coverage: suffix = cov × that layer's feature-input span, capped. The output
		# feature region is one frame (e.g. 80b → 80%≈64); state is windowed (e.g. 320b, capped).
		_probe = _make_spec(args.grid_state_neurons[0], args.levels, args.grid_state_neurons[0] + min_suffix,
			obs_tilt_p=args.obs_tilt_p, obs_tilt_i=args.obs_tilt_i, obs_peraxis_p=args.obs_peraxis_p,
			obs_peraxis_i=args.obs_peraxis_i, obs_peraxis_yaw=args.obs_peraxis_yaw, obs_pwm=args.obs_pwm,
			obs_yaw_err=args.obs_yaw_err, obs_yaw_err_i=args.obs_yaw_err_i, bits_per_feature=args.bits_per_feature)
		_sh = arch_shape_from_spec(_probe); pf = _sh.prefix_factor
		osuf = min(max(min_suffix, round(cov * _sh.output_input_space)), _sh.output_input_space)
		ssuf = min(max(min_suffix, round(cov * _sh.state_input_space)), args.suffix_cap, _sh.state_input_space)
		valid_pairs = [(sn, pf * sn + ssuf, pf * sn + osuf) for sn in args.grid_state_neurons]
		all_pairs = valid_pairs
		print(f"  [grid] per-layer coverage={cov}: state_suffix={ssuf} (of {_sh.state_input_space}), "
		      f"output_suffix={osuf} (of {_sh.output_input_space}), cap={args.suffix_cap}")
	else:
		all_pairs = [(sn, b, b) for sn in args.grid_state_neurons for b in args.grid_bits]
		valid_pairs = [(sn, sb, ob) for (sn, sb, ob) in all_pairs if (sb - sn) >= min_suffix]  # forced prefix = sn
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
	probe_sn, probe_b, probe_ob = valid_pairs[0]
	probe_spec = _make_spec(probe_sn, args.levels, probe_b, args.delta_control, args.delta_leak, obs_tilt_p=args.obs_tilt_p, obs_tilt_i=args.obs_tilt_i, obs_peraxis_p=args.obs_peraxis_p, obs_peraxis_i=args.obs_peraxis_i, obs_peraxis_yaw=args.obs_peraxis_yaw, obs_pwm=args.obs_pwm, obs_yaw_err=args.obs_yaw_err, obs_yaw_err_i=args.obs_yaw_err_i, integral_leak=args.integral_leak, integral_scale=args.integral_scale, decouple_outputs=args.decouple_outputs, bits_per_feature=args.bits_per_feature, feature_balance_ratio=args.feature_balance_ratio, threshold_gamma=args.threshold_gamma, action_repeat=args.action_repeat, output_bits=probe_ob)
	thresholds = fit_thresholds_from_pid_rollouts(probe_spec, num_episodes=10, seed=seed)

	rng_master = np.random.default_rng(seed)
	results = []  # (spec, genome, metrics)
	from .recurrent_genome import RecurrentArchConfig
	for sn, b, ob in valid_pairs:
		spec = _make_spec(sn, args.levels, b, args.delta_control, args.delta_leak, obs_tilt_p=args.obs_tilt_p, obs_tilt_i=args.obs_tilt_i, obs_peraxis_p=args.obs_peraxis_p, obs_peraxis_i=args.obs_peraxis_i, obs_peraxis_yaw=args.obs_peraxis_yaw, obs_pwm=args.obs_pwm, obs_yaw_err=args.obs_yaw_err, obs_yaw_err_i=args.obs_yaw_err_i, integral_leak=args.integral_leak, integral_scale=args.integral_scale, decouple_outputs=args.decouple_outputs, bits_per_feature=args.bits_per_feature, feature_balance_ratio=args.feature_balance_ratio, threshold_gamma=args.threshold_gamma, action_repeat=args.action_repeat, output_bits=ob)
		shape = arch_shape_from_spec(spec)
		state_suffix = b - shape.prefix_factor * sn   # per-layer forced prefix = prefix_factor·sn
		output_suffix = ob - shape.prefix_factor * sn
		rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
		genome = RecurrentArchGenome.random(
			shape, state_neurons=sn,
			output_neurons=spec.num_motors * spec.levels_per_motor,
			state_suffix=state_suffix, output_suffix=output_suffix, rng=rng,
			config=RecurrentArchConfig(feature_balance_ratio=args.feature_balance_ratio, bits_per_feature=args.bits_per_feature),
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
		      f"sn={sn:>2} sb={b:>3} ob={ob:>3} suf(s/o)={state_suffix}/{output_suffix}  "
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
		weight_steady=args.fit_weight_steady,
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
	_log_split_pressure(res, name)
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
                    tracker=None, experiment_id=None, fixed_axes=None):
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
	if getattr(args, "max_state_neurons", None):
		arch_cfg.max_state_neurons = min(arch_cfg.max_state_neurons, int(args.max_state_neurons))
		arch_cfg.min_state_neurons = min(arch_cfg.min_state_neurons, arch_cfg.max_state_neurons)
	# Hard floor on state_neurons from the grid (added 30/05/2026 for Plan A v2).
	# Without this, GA mutations can take sn below the grid minimum, undoing the
	# anchor we set when --grid-state-neurons specifies a tight range.
	arch_cfg.min_state_neurons = max(arch_cfg.min_state_neurons,
	                                 min(args.grid_state_neurons))
	# Phase-5c damping: route the CLI gain into the mutation config so saturation
	# pressure grows state measuredly instead of force-growing every offspring.
	arch_cfg.saturation_grow_gain = getattr(args, "saturation_grow_gain", 0.02)
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
                         seed: int, warm_start_genome=None, initial_population=None,
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
	warm = warm_start_genome
	last_res, last_ev, total_dt = None, None, 0.0
	for i, (label, mask, gens) in enumerate(schedule):
		bar = "-" * 72
		print(f"\n{bar}\n  STAGE 1: NEURONS [{i + 1}/7] axes={label} "
		      f"({gens} gens, patience {args.neurons_patience})\n{bar}", flush=True)
		res, ev, dt = _run_arch_phase(
			args, ec, spec, OptimizationDimension.NEURONS, gens, args.neurons_patience,
			seed, warm_start_genome=warm, initial_population=carried_pop,
			tracker=tracker, experiment_id=experiment_id, fixed_axes=mask)
		total_dt += dt
		last_res, last_ev = res, ev
		if getattr(res, "final_population", None):
			carried_pop = res.final_population  # carry the WHOLE pool (diversity)
		if getattr(res, "best_genome", None) is not None:
			warm = res.best_genome              # ...with the winner pinned to the front
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
                               seed: int, warm_start_genome=None, initial_population=None,
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
	warm = warm_start_genome
	last_res, last_ev, total_dt = None, None, 0.0
	for i, (label, d, gens) in enumerate(schedule):
		ec_d = _scaled_ec(ec, d)
		bar = "-" * 72
		print(f"\n{bar}\n  STAGE 1: NEURONS [{i + 1}/{len(schedule)}] difficulty={label} "
		      f"(tilt≤{_math.degrees(ec_d.max_initial_tilt_rad):.1f}° rate≤{ec_d.max_initial_body_rate:.2f}, "
		      f"{gens} gens, patience {args.neurons_patience})\n{bar}", flush=True)
		res, ev, dt = _run_arch_phase(
			args, ec_d, spec, OptimizationDimension.NEURONS, gens, args.neurons_patience,
			seed, warm_start_genome=warm, initial_population=carried_pop,
			tracker=tracker, experiment_id=experiment_id)
		total_dt += dt
		last_res, last_ev = res, ev
		if getattr(res, "final_population", None):
			carried_pop = res.final_population
		if getattr(res, "best_genome", None) is not None:
			warm = res.best_genome
	return last_res, last_ev, total_dt


def _phase_stable(ev, best_genome) -> float:
	"""Mastery signal: the phase winner's stable fraction at the phase's difficulty
	(re-scored on the phase evaluator — same as _print_stage_result)."""
	if best_genome is None:
		return 0.0
	m = (ev.score_genomes([best_genome])[0] if getattr(best_genome, "cells", None) is not None
	     else ev.evaluate_batch([best_genome])[0])
	return float(m.acc)


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
	use_score = getattr(best_genome, "cells", None) is not None
	stbs, errs = [], []
	for rs in seed_list:
		if rs == train_seed:
			continue  # shares the train seed → not held-out
		thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=rs)
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
                                        seed: int, warm_start_genome=None, initial_population=None,
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
	carried_pop, warm = initial_population, warm_start_genome
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
			seed, warm_start_genome=warm, initial_population=carried_pop,
			tracker=tracker, experiment_id=experiment_id)
		spent += gens; total_dt += dt; last_res, last_ev = res, ev
		if getattr(res, "final_population", None):
			carried_pop = res.final_population
		if getattr(res, "best_genome", None) is not None:
			warm = res.best_genome
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
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
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
	if getattr(args, "max_state_neurons", None):
		arch_cfg.max_state_neurons = min(arch_cfg.max_state_neurons, int(args.max_state_neurons))
		arch_cfg.min_state_neurons = min(arch_cfg.min_state_neurons, arch_cfg.max_state_neurons)
	arch_cfg.min_state_neurons = max(arch_cfg.min_state_neurons,
	                                 min(args.grid_state_neurons))
	# Phase-5c damping: route the CLI gain into the mutation config so saturation
	# pressure grows state measuredly instead of force-growing every offspring.
	arch_cfg.saturation_grow_gain = getattr(args, "saturation_grow_gain", 0.02)
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
	# Held-out episode count decoupled from the GA's --eval-episodes (10/06/2026):
	# the search eval runs every generation (cost ∝ episodes), but the held-out is
	# scored ONCE per stage — so it can afford many more episodes to de-quantize
	# the reported stable% (8 eps = 12.5pp steps; 50 eps = 2pp).
	rep_eps = getattr(args, "report_episodes", None) or args.eval_episodes
	ev = ControllerEvaluator(spec, num_eval_episodes=rep_eps,
	                         seed=report_seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=_rg_config(args, ec, report_seed),
	                         max_train_workers=args.train_workers)
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
	# MEMORY-stage winners carry cells → score (no retrain); arch winners → train+eval.
	use_score = getattr(best_genome, "cells", None) is not None
	metrics = ev.score_genomes(pop) if use_score else ev.evaluate_batch(pop)
	stables = [m.acc * 100 for m in metrics]
	errs = [m.mean_attitude_error_deg for m in metrics]
	ds = metrics[0]            # final_population[0] = the during-search winner = THE RESULT
	pop_max = max(stables)     # descriptive only — NOT selected (would leak)
	pid_m = _pid_baseline(ec, rep_eps, report_seed)
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
		# stage_num 0 is legitimate ("before Stage 1" — the --seed-winner
		# curriculum path forces it so mode='next' starts at Stage 1/NEURONS).
		# `or 1` would coerce a valid 0 to 1 (0 is falsy), so handle None explicitly.
		_sn = resume_state.get("stage_num")
		dumped_stage = int(_sn) if _sn is not None else 1
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
		if getattr(args, "axis_curriculum", False):
			# H4: 7-phase combinatorial curriculum (singles → pairs → triple).
			res1, ev1, dt1 = _run_axis_curriculum(args, ec, spec1, seed,
			                                      warm_start_genome=warm1, initial_population=init_pop1,
			                                      tracker=tracker, experiment_id=_eid(1))
		elif getattr(args, "difficulty_adaptive", False):
			# H4-v3: mastery-gated difficulty curriculum with backtracking.
			res1, ev1, dt1 = _run_adaptive_difficulty_curriculum(args, ec, spec1, seed,
			                                                     warm_start_genome=warm1, initial_population=init_pop1,
			                                                     tracker=tracker, experiment_id=_eid(1))
		elif getattr(args, "difficulty_curriculum", False):
			# H4-v2: difficulty curriculum (ramp IC magnitude, full 3-axis throughout).
			res1, ev1, dt1 = _run_difficulty_curriculum(args, ec, spec1, seed,
			                                             warm_start_genome=warm1, initial_population=init_pop1,
			                                             tracker=tracker, experiment_id=_eid(1))
		else:
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
	# Delta-control: output decodes to a per-step PWM delta with a leaky accumulator
	# = a structural integrator (offloads PID's I-term out of the untrained state).
	# Banked 08/06/2026: +5pp stability vs the old hardcoded False. PARTIAL fix.
	ap.add_argument("--delta-control", action=argparse.BooleanOptionalAction, default=True,
	                help="Output decodes to a leaky-accumulator PWM delta (structural integrator). Default ON.")
	ap.add_argument("--delta-leak", type=float, default=0.95,
	                help="Leak on the delta accumulator (1.0=pure integrator/can run away; <1.0 bounds offset). Default 0.95.")
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
	# E1 random immigrants (plan controller_break_90_v2): probability each offspring
	# slot is a FRESH random genome instead of a bred child. Diversity preservation
	# against premature convergence (seed-bimodal 70-90% held-out). 0.0 = off.
	ap.add_argument("--immigrants", type=float, default=0.0,
	                help="Random-immigrant fraction of each generation's offspring (0.0-0.5 sensible; default off).")
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
	# Phase-5c saturation→grow damping (§11b). Lower = gentler state growth under
	# splitting-trainer saturation pressure (default 0.02 ≈ old aggressive behavior
	# at high saturation; 0.005 damps hard so sn grows measuredly, not every gen).
	ap.add_argument("--saturation-grow-gain", type=float, default=0.02,
	                help="5c saturation→state-growth probability gain (lower=gentler; default 0.02).")
	# Evaluation / episode.
	ap.add_argument("--eval-episodes", type=int, default=20)
	ap.add_argument("--memory-eval-episodes", type=int, default=None,
		help="During-search eval episodes for the MEMORY stage ONLY (default: --eval-episodes). "
		     "MEMORY is the stability-lift stage AND cheap per-gen (shapes collapse), so it can "
		     "afford more episodes for a clean stability gradient while NEURONS stays cheaper. "
		     "13/06/2026: 16-ep eval can't resolve stability → GA optimizes blind; raise this.")
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	# W2.3 train-under-weather: arm the calibrated disturbance ladder in ALL
	# rollouts of this run (training + in-search eval + report). OFF = clean
	# (bit-identical legacy). Anchors @2000/L2: PID+ 99.8 / PD 84.0; every
	# clean-trained WNN scored 0 at L2 (W2.2 brittleness audit, 06/07).
	ap.add_argument("--disturbance", type=str, default="OFF",
	                choices=["OFF", "L1", "L2", "L3"],
	                help="W2 weather level for all rollouts (default OFF)")
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
	                help="Path to an emergency-dump pickle (see _dump_emergency_state). "
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
	                choices=["neurons", "memory"],
	                help="Which stage the --seed-winner warm-start begins at (E5.2 vs "
	                     "memory-only). 'neurons' (default): grid skipped, NEURONS→…→MEMORY "
	                     "under --disturbance (architecture is re-searched). 'memory': ALSO "
	                     "skip NEURONS/BITS/CONNECTIONS — FREEZE the L1 winner's architecture "
	                     "and fine-tune ONLY the memory (cells) under --disturbance. Tests "
	                     "whether architecture search under the storm helps or hurts vs pure "
	                     "cell fine-tuning of the proven L1 shape.")

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
		# stage_num + mode='next' pick the FIRST stage to run:
		#   'neurons' → stage_num=0 → resume_start_stage=min(0+1,4)=1 (NEURONS→…→MEMORY)
		#   'memory'  → stage_num=3 → resume_start_stage=min(3+1,4)=4 (skip N/B/C; MEMORY only,
		#               freezing the L1 winner's architecture, fine-tuning only its cells)
		_sw_stage = getattr(args, "seed_winner_stage", "neurons")
		resume_state["stage_num"] = 3 if _sw_stage == "memory" else 0
		resume_state["resume_mode"] = "next"
		_sw_desc = ("MEMORY-only (arch FROZEN, cells fine-tuned)" if _sw_stage == "memory"
		            else "NEURONS warm-started")
		print(f"[main] CURRICULUM seed-winner from {seed_path} "
		      f"(pop={len(resume_state.get('population') or [])}, "
		      f"spec={type(resume_state.get('spec')).__name__}) → grid skipped, "
		      f"{_sw_desc} under --disturbance {args.disturbance}")

	t_start = time.time()
	from wnn.control.training import DisturbanceConfig
	dist = DisturbanceConfig.preset(args.disturbance, seed=911)
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate,
		disturbance=dist,
	)
	if dist is not None:
		print(f"[W2] disturbance={args.disturbance} armed for ALL rollouts "
		      f"(tau_bias={dist.tau_bias[0]:.4f} N·m, gust_sigma={dist.gust_sigma:.4f}, "
		      f"asym_mag=±{dist.motor_asym_mag:.0%}, gyro_sigma={dist.gyro_sigma})")

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

	_join_pending_saves()  # ensure any async between-stage writes finished before exit
	return 0


if __name__ == "__main__":
	sys.exit(main())
