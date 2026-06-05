"""Curriculum-GA for the WNN drone controller.

Two modes share most of the code path:

  --mode sweep   Weight sweep, Stage A only (250 steps / tilt=5° / body-rate
                 0.5 / pop=200 / gens=30 / patience=3). Runs each combo
                 sequentially and reports the winner per combo. Used to pick a
                 fitness-weight recipe before committing to the full curriculum.

  --mode full    Single weight set, 5-stage curriculum with warm-start chain.
                 The horizon is FIXED (250 ms); the easy→hard axis is the
                 initial-condition severity (tilt + body-rate):
                   A: tilt=5°  body-rate=0.5
                   B: tilt=15° body-rate=1.0
                   C: tilt=30° body-rate=2.0
                   D: tilt=45° body-rate=3.0
                   E: tilt=60° body-rate=4.0 (production-hard)
                 Each stage's final population seeds the next, so the GA's
                 accumulated diversity isn't lost on stage transitions.
                 Resume a cancelled run with --resume <save_dir>/curriculum_resume.pkl.

Why curriculum-on-INITIAL-CONDITIONS (01/06/2026): the original
curriculum-on-steps was empirically refuted. Attitude is the double-integral
of control torque, so at dt=0.001 a 10-30 ms episode yields only ~0.003-0.22°
of control authority vs the 5° stable threshold — a do-nothing hover scored
identically to a perfect PID (52.5% vs 53.0% stable at 10 ms). "Short horizon"
is not an easy task, it is an UNOBSERVABLE one. The principled easy→hard axis
is disturbance severity at a horizon long enough (≥~100 ms; we use 250 ms) for
control authority to dominate. Reward (−Σerr²) then separates skill at every
stage (PID−hover reward gap grows 2→154 across the schedule); stable_rate
(mean_err≤5°) saturates to ~0 from ~15° up, so fitness must be reward-dominant.

History: superseded the 31/05 curriculum-on-steps (Plan A v5 plateaued at
fit=1.0000/err=12.61°). See project_curriculum_cancel_fix memory.
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

from wnn.control.evaluator import (
	ControllerSpec, ControllerEvaluator, arch_shape_from_spec, spec_from_arch,
	fit_thresholds_from_pid_rollouts,
)
from wnn.control.training import EpisodeConfig
from wnn.control.dagger import eval_closed_loop_reset
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.arch_strategy import (
	ControllerArchGAStrategy, default_controller_arch_config, default_controller_ga_config,
)
from wnn.control.reward_gated import RewardGatedConfig
from wnn.control import cancel_state
from wnn.ram.strategies.optimization_dimension import OptimizationDimension
from wnn.seeds import resolve_seed_set, log_seed_set, record_seed_set


# ============================================================================
# SIGTERM emergency dump (shared with run_phased_ga.py philosophy)
# ============================================================================

_emergency_state: dict = {
	"stage":      None,     # "A".."E" or combo name
	"spec":       None,
	"population": [],
	"best_genome": None,
	"generation": 0,
	"save_path":  None,
	"args":       None,
}


def _sigterm_handler(signum, _frame) -> None:
	name = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT"}.get(signum, str(signum))
	print(f"\n[{name}] Cancellation requested. Marking PROPER cancel + setting Rust "
	      f"cancel flag — will dump state and exit at next safe point.", flush=True)
	# PROPER-cancel signal (the controller analog of the IDS worker's
	# _stop_current_flow). The evaluator's cancel-guard reads this to tell a
	# real SIGTERM apart from a spurious Rust-flag set: proper → honor the stop
	# (sentinels → GA unwinds → dump/resume); spurious → reset + retry.
	cancel_state.mark_sigterm(signum)
	try:
		import ram_accelerator
		ram_accelerator.set_cancel_flag()
	except Exception as e:
		print(f"[{name}] Could not set Rust cancel flag: {e}", flush=True)


def _install_signal_handlers() -> None:
	signal.signal(signal.SIGTERM, _sigterm_handler)
	signal.signal(signal.SIGINT,  _sigterm_handler)
	cancel_state.reset_sigterm()
	try:
		import ram_accelerator
		ram_accelerator.reset_cancel_flag()
	except Exception:
		pass


def _dump_emergency_state() -> None:
	path = _emergency_state.get("save_path")
	if path is None:
		return
	payload = {
		"stage":       _emergency_state["stage"],
		"spec":        _emergency_state["spec"],
		"population":  _emergency_state["population"],
		"best_genome": _emergency_state["best_genome"],
		"generation":  _emergency_state["generation"],
		"meta": {
			"saved_at_unix":  time.time(),
			"saved_at_iso":   time.strftime("%Y-%m-%dT%H:%M:%S%z"),
			"emergency_dump": True,
		},
	}
	p = Path(path)
	p.parent.mkdir(parents=True, exist_ok=True)
	with open(p, "wb") as f:
		pickle.dump(payload, f)
	print(f"\n[emergency-dump] Stage {payload['stage']} gen {payload['generation']}, "
	      f"{len(payload['population'])} genomes → {p}", flush=True)


def _install_emergency_hook(strat) -> None:
	original = strat._on_generation_start
	def wrapped(generation, **ctx):
		# ctx['population'] is a list of (genome, metrics) tuples; resume's
		# initial_population needs RAW genomes (seed_population clones them).
		_pop = ctx.get("population", [])
		_emergency_state["population"]  = [p[0] if isinstance(p, tuple) else p for p in _pop]
		_emergency_state["best_genome"] = ctx.get("best_genome")
		_emergency_state["generation"]  = generation
		# Stop ONLY on a PROPER cancel (real SIGTERM to this process). A
		# spurious Rust-flag set is handled+reset inside the evaluator's
		# cancel-guard (retry), so it must NOT unwind the GA here — that was
		# the bug that let one stray flag collapse every later stage to
		# 180°/-inf and emit a fake "CURRICULUM RESULT".
		if cancel_state.sigterm_received():
			_dump_emergency_state()
			raise StopIteration
		# PER-GENERATION crash checkpoint (full mode): flush curriculum_resume.pkl
		# every gen so a HARD crash (power/OOM/SIGKILL) resumes the CURRENT stage at
		# the CURRENT generation. Bounds crash loss to ~1 gen instead of a full stage.
		if _emergency_state.get("resume_ctx"):
			try:
				_dump_midstage_resume()
			except Exception:
				pass
		return original(generation, **ctx)
	strat._on_generation_start = wrapped


# ============================================================================
# Curriculum stage definition
# ============================================================================

class CurriculumStage:
	"""One curriculum stage. Stages run sequentially with the GA's
	final_population carried forward as initial_population of the next.

	01/06/2026 — curriculum-on-INITIAL-CONDITION-difficulty (replaces the
	curriculum-on-steps, which was empirically refuted: at dt=0.001, a 10-30ms
	episode gives ~0.003-0.22° of control authority vs a 5° threshold, so a
	do-nothing hover scored identically to a perfect PID — 52.5% vs 53.0%
	stable — and the GA had no control-skill gradient). The HORIZON is now
	FIXED at a signal-bearing length; the easy→hard axis is the initial
	disturbance: tilt + body-rate grow per stage. Empirically (250 steps) the
	reward gap PID−hover grows 2→154 across the schedule, so reward gives a
	strong gradient at every stage. NOTE: stable_rate (mean_err≤5°) saturates
	to ~0 from ~15° up — reward (weight_err_sq) is the real driver; weight the
	fitness reward-dominant."""
	__slots__ = ("name", "steps", "tilt_deg", "body_rate", "yaw_rate",
	             "gens", "patience", "eval_episodes")
	def __init__(self, name, steps, tilt_deg, gens, patience, eval_episodes,
	             body_rate=0.5, yaw_rate=0.3):
		self.name           = name
		self.steps          = steps
		self.tilt_deg       = tilt_deg
		self.body_rate      = body_rate    # rad/s, max initial |omega_x/y|
		self.yaw_rate       = yaw_rate     # rad/s, max initial |omega_z|
		self.gens           = gens
		self.patience       = patience
		self.eval_episodes  = eval_episodes


# Fixed signal-bearing horizon. 250 steps × dt=0.001 = 250 ms — long enough for
# control authority (~70° achievable) to dominate the 5° threshold band, so the
# GA ranks on control skill, not initial-condition luck. (At 10-30 ms it ranked
# on luck; see CurriculumStage docstring + project_curriculum_cancel_fix memory.)
FIXED_HORIZON_STEPS = 250

# 5-stage IC-difficulty schedule: tilt 5→15→30→45→60°, body-rate 0.5→4.0 rad/s.
# Used directly by --mode full; --mode sweep runs only Stage A (now itself a
# real, signal-bearing stage at 5°/250 ms: PID 97% vs hover 31% stable).
DEFAULT_CURRICULUM: list[CurriculumStage] = [
	CurriculumStage("A", steps=FIXED_HORIZON_STEPS, tilt_deg=5.0,  body_rate=0.5, yaw_rate=0.3, gens=30, patience=3, eval_episodes=100),
	CurriculumStage("B", steps=FIXED_HORIZON_STEPS, tilt_deg=15.0, body_rate=1.0, yaw_rate=0.6, gens=30, patience=3, eval_episodes=100),
	CurriculumStage("C", steps=FIXED_HORIZON_STEPS, tilt_deg=30.0, body_rate=2.0, yaw_rate=1.2, gens=30, patience=3, eval_episodes=100),
	CurriculumStage("D", steps=FIXED_HORIZON_STEPS, tilt_deg=45.0, body_rate=3.0, yaw_rate=1.8, gens=30, patience=3, eval_episodes=100),
	CurriculumStage("E", steps=FIXED_HORIZON_STEPS, tilt_deg=60.0, body_rate=4.0, yaw_rate=2.4, gens=30, patience=3, eval_episodes=100),
]


def _with_steps(stage: CurriculumStage, steps) -> CurriculumStage:
	"""Clone a stage with a different horizon (steps). Used to run the sweep
	and the full curriculum at different fixed horizons (--sweep-steps /
	--full-steps) without mutating the shared DEFAULT_CURRICULUM. None = keep."""
	if steps is None or steps == stage.steps:
		return stage
	return CurriculumStage(stage.name, steps=steps, tilt_deg=stage.tilt_deg,
	                       body_rate=stage.body_rate, yaw_rate=stage.yaw_rate,
	                       gens=stage.gens, patience=stage.patience,
	                       eval_episodes=stage.eval_episodes)


def _with_overrides(stage: CurriculumStage, steps=None, gens=None, patience=None) -> CurriculumStage:
	"""Clone a stage applying horizon/gens/patience overrides (None = keep). Used by
	--mode full (--full-steps/--full-gens/--full-patience) so the full curriculum can
	run a different GA budget than the sweep WITHOUT mutating shared DEFAULT_CURRICULUM."""
	new_steps = steps if steps is not None else stage.steps
	new_gens = gens if gens is not None else stage.gens
	new_pat = patience if patience is not None else stage.patience
	if (new_steps, new_gens, new_pat) == (stage.steps, stage.gens, stage.patience):
		return stage
	return CurriculumStage(stage.name, steps=new_steps, tilt_deg=stage.tilt_deg,
	                       body_rate=stage.body_rate, yaw_rate=stage.yaw_rate,
	                       gens=new_gens, patience=new_pat,
	                       eval_episodes=stage.eval_episodes)


def _build_ec(stage: CurriculumStage) -> EpisodeConfig:
	"""Episode config for this curriculum stage. Yaw tilt is capped at 45° so
	yaw error never dominates the (roll/pitch) attitude-error objective."""
	return EpisodeConfig(
		dt=0.001, steps_per_episode=stage.steps,
		max_initial_tilt_rad=math.radians(stage.tilt_deg),
		max_initial_yaw_rad=math.radians(min(stage.tilt_deg, 45.0)),
		max_initial_body_rate=stage.body_rate, max_initial_yaw_rate=stage.yaw_rate,
	)


def _build_ga_config(args, gens: int, patience: int, w: dict):
	"""GAConfig with stage gens + patience + the run's fitness weights.
	Mirrors run_phased_ga._build_ga_config but injects per-combo weights."""
	cfg = default_controller_ga_config(
		population_size=args.pop, generations=gens,
		weight_err_sq=w["err"],
		weight_stable=w["stable"],
		weight_jerk=w["jerk"],
		weight_mono=w["mono"],
	)
	cfg.patience       = patience
	cfg.elitism_pct    = args.elitism
	cfg.crossover_rate = args.crossover_rate
	cfg.check_interval = args.check_interval
	return cfg


def _run_one_stage(args, stage: CurriculumStage, weights: dict,
                   spec: ControllerSpec, seed: int,
                   initial_population=None,
                   stage_label: str = "", resume_start_gen: int = 0):
	"""Run ONE curriculum stage. Returns (res, ev, wall_time, fitted_thresholds).

	resume_start_gen > 0 resumes this stage mid-way (the GA continues at that
	generation via _resume_start_gen) — used by per-generation crash recovery."""
	ec = _build_ec(stage)
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
	# Critical fix 31/05/2026: RewardGatedConfig defaults to
	# steps_per_episode=2000 — without overriding it the curriculum's
	# stage steps (10/30/100/...) are IGNORED inside reward_gated_train
	# and every genome trains under 2000-step rollouts. This is the bug
	# that made the first sweep run grind for 30+ minutes on Stage A
	# instead of finishing in ~5 min. Build an rg_config that matches the
	# stage's steps, with sensible defaults for rounds/episodes_per_round.
	rg_config = RewardGatedConfig(seed=seed, episode_config=ec)
	rg_config.steps_per_episode = stage.steps
	rg_config.num_rounds        = 3   # 8 is overkill at Stage A; tune up later
	rg_config.episodes_per_round = 6
	rg_config.progress          = False
	ev = ControllerEvaluator(spec, num_eval_episodes=stage.eval_episodes,
	                         seed=seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=rg_config,
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=args.num_eval_folds)
	arch_cfg = default_controller_arch_config(spec)
	gacfg = _build_ga_config(args, stage.gens, stage.patience, weights)
	strat = ControllerArchGAStrategy(spec, OptimizationDimension.NEURONS,
	                                 arch_config=arch_cfg, ga_config=gacfg,
	                                 seed=seed, batch_evaluator=ev)
	if resume_start_gen > 0:
		strat._resume_start_gen = int(resume_start_gen)   # GA continues at this gen

	# Wire emergency-dump hook BEFORE optimize() runs.
	_emergency_state["stage"] = stage_label or stage.name
	_emergency_state["spec"]  = spec
	_emergency_state["args"]  = args
	if args.save_dir:
		_emergency_state["save_path"] = str(Path(args.save_dir) /
		                                   f"emergency_stage{stage_label or stage.name}.pkl")
	_install_emergency_hook(strat)

	t = time.time()
	optimize_kwargs = {
		"evaluate_fn": lambda g: ev.evaluate_batch([g])[0].ce,
		"batch_evaluate_fn": ev.evaluate_batch,
	}
	if initial_population is not None:
		optimize_kwargs["initial_population"] = list(initial_population)
	res = strat.optimize(**optimize_kwargs)
	wall = time.time() - t
	return res, ev, wall, thresholds


def _stage_summary(stage: CurriculumStage, weights: dict, res, ev, wall: float,
                   prefix: str = ""):
	bar = "─" * 68
	w = weights
	print(f"\n{bar}")
	print(f"  {prefix}Stage {stage.name}: steps={stage.steps:>3}  "
	      f"tilt={stage.tilt_deg:>4.1f}°  body_rate={stage.body_rate:>3.1f}  "
	      f"pop={ev.num_eval_episodes if hasattr(ev, 'num_eval_episodes') else '?'}  "
	      f"gens={stage.gens} pat={stage.patience}")
	print(f"  weights: err²={w['err']:.2f}  stable={w['stable']:.2f}  "
	      f"jerk={w['jerk']:.2f}  mono={w['mono']:.2f}")
	# Best metrics: pull from the strategy's result.
	if res is None or res.best_genome is None:
		print(f"  result: NO BEST (likely cancelled)")
	elif cancel_state.sigterm_received():
		# A proper cancel poisons any fresh evaluate_batch (it returns the
		# 180°/-inf sentinel), which would mislabel this stage's REAL winner.
		# Report the metrics the GA already recorded instead of re-evaluating.
		reward = -res.final_fitness if res.final_fitness is not None else float("nan")
		stable = res.final_accuracy if res.final_accuracy is not None else float("nan")
		print(f"  best: (cancelled — GA-recorded) stable={stable*100:.1f}%  "
		      f"reward={reward:.2f}  iters={res.iterations_run}  wall={wall:.0f}s")
	else:
		# Re-eval the best genome on the same evaluator to pull surface metrics.
		try:
			best_m = ev.evaluate_batch([res.best_genome])[0]
			print(f"  best: err={best_m.mean_attitude_error_deg:.2f}°  "
			      f"stable={best_m.acc*100:.1f}%  reward={best_m.fitness:.2f}  "
			      f"iters={res.iterations_run}  wall={wall:.0f}s")
		except Exception as e:
			print(f"  best: (eval-failed: {e})  iters={res.iterations_run}  wall={wall:.0f}s")
	print(bar)


def _holdout_eval(args, spec, weights, genome, report_seed: int, train_seed: int):
	"""TRUE held-out: re-evaluate the FINAL controller on a FRESH seed's episodes.

	The sweep/full 'best:' numbers are all measured on evaluators seeded with the
	TRAINING seed — fresh episode draws, but the same RNG stream / initial-condition
	+ disturbance distribution the GA selected against. That is in-distribution
	resampling, not a held-out (the IDS analogue of reporting K-fold fitness as the
	result). This builds an independent evaluator at report_seed (≠ train_seed) over
	the final-stage horizon and reports the controller's metrics there — the honest
	number to quote for a paper. PID is evaluated on the SAME report_seed for a fair
	side-by-side.
	"""
	if report_seed == train_seed:
		print(f"  [report-seed] WARNING: report_seed == train_seed ({train_seed}) — "
		      f"NOT a held-out; pick a distinct --report-seed.")
	stage = _with_steps(DEFAULT_CURRICULUM[-1], getattr(args, "full_steps", None))
	ec = _build_ec(stage)
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=report_seed)
	rg_config = RewardGatedConfig(seed=report_seed, episode_config=ec)
	rg_config.steps_per_episode = stage.steps
	rg_config.num_rounds = 3
	rg_config.episodes_per_round = 6
	rg_config.progress = False
	ev = ControllerEvaluator(spec, num_eval_episodes=stage.eval_episodes,
	                         seed=report_seed, episode_config=ec, thresholds=thresholds,
	                         rg_config=rg_config, max_train_workers=args.train_workers,
	                         num_eval_folds=args.num_eval_folds)
	m = ev.evaluate_batch([genome])[0]
	bar = "=" * 72
	print(f"\n{bar}\n  HELD-OUT REPORT — final controller on fresh seed {report_seed} "
	      f"(train seed was {train_seed})\n{bar}")
	print(f"  controller (held-out):  err={m.mean_attitude_error_deg:.2f}°  "
	      f"stable={m.acc*100:.1f}%  reward={m.fitness:.2f}")
	pid = AttitudePID(AttitudePIDConfig())
	from wnn.control.training import make_pid_action_fn
	_, pid_m = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, ec, 100, report_seed)
	print(f"  vs PID    (held-out seed):  err={pid_m['mean_attitude_error_deg']:.2f}°  "
	      f"stable={pid_m['stable_rate']*100:.1f}%  reward={pid_m['mean_reward']:.2f}")
	print(f"  vs MLP-GA baseline:  9.66°  (run_mlp_ga.py 3-way held-out)")
	print(bar)
	return m


# ============================================================================
# Mode 1: Weight sweep at Stage A only
# ============================================================================

# Per Luiz's 31/05/2026 spec: err-heavy with stable second, jerk + mono small
# but non-zero.
SWEEP_COMBOS: list[dict] = [
	# Original curriculum-sweep probes (low jerk/mono region, plus a 0.10 point).
	{"name": "W1",  "err": 0.50, "stable": 0.40, "jerk": 0.05, "mono": 0.05},
	{"name": "W2",  "err": 0.40, "stable": 0.50, "jerk": 0.05, "mono": 0.05},
	{"name": "W3",  "err": 0.60, "stable": 0.30, "jerk": 0.05, "mono": 0.05},
	{"name": "W4",  "err": 0.45, "stable": 0.35, "jerk": 0.10, "mono": 0.10},
	# The full 29/05 phased-GA weight grid (err×stable×jerk×mono, sum=1.0),
	# re-tested under the new short-horizon curriculum. C7 == the fixed Plan A
	# v1-v5 weights; C2 tied the MLP baseline (9.66°) in the old long-horizon run.
	{"name": "C1",  "err": 0.20, "stable": 0.40, "jerk": 0.20, "mono": 0.20},
	{"name": "C2",  "err": 0.20, "stable": 0.50, "jerk": 0.10, "mono": 0.20},
	{"name": "C3",  "err": 0.20, "stable": 0.50, "jerk": 0.20, "mono": 0.10},
	{"name": "C4",  "err": 0.30, "stable": 0.30, "jerk": 0.20, "mono": 0.20},
	{"name": "C5",  "err": 0.30, "stable": 0.40, "jerk": 0.10, "mono": 0.20},
	{"name": "C6",  "err": 0.30, "stable": 0.40, "jerk": 0.20, "mono": 0.10},
	{"name": "C7",  "err": 0.30, "stable": 0.50, "jerk": 0.10, "mono": 0.10},
	{"name": "C8",  "err": 0.40, "stable": 0.20, "jerk": 0.20, "mono": 0.20},
	{"name": "C9",  "err": 0.40, "stable": 0.30, "jerk": 0.10, "mono": 0.20},
	{"name": "C10", "err": 0.40, "stable": 0.30, "jerk": 0.20, "mono": 0.10},
	{"name": "C11", "err": 0.40, "stable": 0.40, "jerk": 0.10, "mono": 0.10},
	{"name": "C12", "err": 0.50, "stable": 0.20, "jerk": 0.10, "mono": 0.20},
	{"name": "C13", "err": 0.50, "stable": 0.20, "jerk": 0.20, "mono": 0.10},
	{"name": "C14", "err": 0.50, "stable": 0.30, "jerk": 0.10, "mono": 0.10},
]


def _grid_seed_spec(args, seed: int) -> ControllerSpec:
	"""Tiny grid to pick a seed spec for the sweep. Matches the run_phased_ga
	Stage 0 minimum filter so we don't waste budget on a hopeless seed."""
	# Hard-code a reasonable seed: sn=12, output=4 motors * 64 levels, b=40
	# matching the v3/v4/v5 grid winner. Curriculum doesn't need the full
	# Stage 0 grid — we want to ablate weights, not architectures.
	return ControllerSpec(
		num_motors=4, levels_per_motor=64,
		state_neurons=12, state_bits_per_neuron=40,
		output_bits_per_neuron=40,
		input_window_k=4, bits_per_feature=8,
	)


def run_sweep(args, seed: int):
	"""4-combo Stage A sweep. Runs each combo with the same seed spec and
	stage A schedule. Prints a final ranking by best-of-stage stable_rate
	+ err.

	Each combo writes its winner pickle to
	{args.save_dir}/sweep_{combo_name}_stageA.pkl so a downstream run can
	warm-start with the winning combo's stage-A population."""
	out_dir = Path(args.save_dir) if args.save_dir else Path("/tmp/curriculum_sweep")
	out_dir.mkdir(parents=True, exist_ok=True)
	# The sweep only RANKS weight combos, so it runs at a leaner population
	# (--sweep-pop, default 50) than the full curriculum (--pop). We temporarily
	# point args.pop at it; main() restores it before the auto-full run.
	_sweep_pop = getattr(args, "sweep_pop", None)
	if _sweep_pop:
		args.pop = _sweep_pop
	# Optional --combos filter: run only the named combos (e.g. for a
	# multi-seed confirmation round on the top-K survivors). Default = all 18.
	_only = getattr(args, "combos", None)
	if _only:
		wanted = {c.strip() for c in _only.split(",") if c.strip()}
		combos_to_run = [c for c in SWEEP_COMBOS if c["name"] in wanted]
		missing = wanted - {c["name"] for c in combos_to_run}
		if missing:
			print(f"  [warn] --combos: unknown names ignored: {sorted(missing)}")
	else:
		combos_to_run = list(SWEEP_COMBOS)
	# Sweep runs Stage A at the (overridable) sweep horizon.
	stage = _with_steps(DEFAULT_CURRICULUM[0], getattr(args, "sweep_steps", None))
	seed_spec = _grid_seed_spec(args, seed)
	print(f"\n{'='*72}")
	print(f"  CURRICULUM SWEEP — Stage A only — {len(combos_to_run)}/{len(SWEEP_COMBOS)} weight combos"
	      f"{' (filtered: ' + _only + ')' if _only else ''}")
	print(f"  seed spec: sn={seed_spec.state_neurons} sb={seed_spec.state_bits_per_neuron} ob={seed_spec.output_bits_per_neuron}")
	print(f"  stage A: steps={stage.steps} tilt={stage.tilt_deg}° pop={args.pop} gens={stage.gens} pat={stage.patience}")
	print(f"  outdir: {out_dir}")
	print(f"{'='*72}")

	combo_results = []
	for combo in combos_to_run:
		combo_label = f"sweep-{combo['name']}"
		print(f"\n{'#'*72}\n# COMBO {combo['name']}: "
		      f"err²={combo['err']:.2f} stable={combo['stable']:.2f} "
		      f"jerk={combo['jerk']:.2f} mono={combo['mono']:.2f}\n{'#'*72}")
		res, ev, wall, _thr = _run_one_stage(args, stage, combo, seed_spec, seed,
		                                     initial_population=None,
		                                     stage_label=combo_label)
		_stage_summary(stage, combo, res, ev, wall, prefix=f"[{combo['name']}] ")
		# Save winner for this combo
		if res is not None and res.best_genome is not None:
			pkl_path = out_dir / f"sweep_{combo['name']}_stageA.pkl"
			with open(pkl_path, "wb") as f:
				pickle.dump({
					"combo":      combo,
					"stage":      "A",
					"spec":       seed_spec,
					"best_genome": res.best_genome,
					"population": list(res.final_population) if res.final_population else [],
				}, f)
			print(f"  [save] winner → {pkl_path}")
		combo_results.append((combo, res, ev, wall))

		# PROPER cancel mid-sweep → stop sweeping. Remaining combos would all
		# return 180°/-inf sentinels, and a partial sweep must NOT auto-launch
		# a full run on a half-measured winner.
		if cancel_state.sigterm_received():
			print(f"\n  [cancel] PROPER cancel (signum={cancel_state.last_signum()}) after "
			      f"combo {combo['name']} — stopping sweep ({len(combo_results)}/{len(combos_to_run)} done). "
			      f"Will NOT auto-launch full curriculum.")
			break

	# Final ranking. When a proper cancel is active, a fresh evaluate_batch is
	# poisoned (180°/-inf sentinel) — rank on the GA-recorded metrics instead.
	print(f"\n{'='*72}\n  SWEEP RESULT — by best-genome stable_rate (then err)\n{'='*72}")
	cancelled = cancel_state.sigterm_received()
	ranked = []
	for combo, res, ev, wall in combo_results:
		if res is None or res.best_genome is None:
			continue
		if cancelled:
			stable = res.final_accuracy if res.final_accuracy is not None else 0.0
			reward = -res.final_fitness if res.final_fitness is not None else float("-inf")
			ranked.append((combo, stable, float("nan"), reward, wall))
			continue
		try:
			m = ev.evaluate_batch([res.best_genome])[0]
			ranked.append((combo, m.acc, m.mean_attitude_error_deg, m.fitness, wall))
		except Exception:
			continue
	ranked.sort(key=lambda r: (-r[1], r[2]))  # stable desc, err asc
	print(f"  {'combo':<6}  {'stable':>8}  {'err':>8}  {'reward':>8}  {'wall':>6}")
	for combo, stable, err, fit, wall in ranked:
		print(f"  {combo['name']:<6}  {stable*100:>7.1f}%  {err:>7.2f}°  {fit:>8.2f}  {wall:>5.0f}s")
	if ranked:
		winner = ranked[0][0]
		print(f"\n  WINNER: {winner['name']} → re-run with --mode full --weights "
		      f"err={winner['err']},stable={winner['stable']},jerk={winner['jerk']},mono={winner['mono']}")
	return ranked


def _is_clear_winner(ranked: list, min_stable_pct: float = 1.0) -> bool:
	"""Heuristic for 'this combo clearly won so launch full curriculum'.
	True iff: (a) at least one combo produced a result, AND (b) the top
	combo's stable_rate exceeds `min_stable_pct` (default 1%) — i.e. some
	signal of learning is present (not all combos hitting 0%). The user
	can always SIGTERM the auto-launched full run if the picked combo
	turns out to be wrong; the heuristic is intentionally generous."""
	if not ranked:
		return False
	top_stable = ranked[0][1]  # stable_rate fraction
	return top_stable * 100.0 >= min_stable_pct


# ============================================================================
# Mode 2: Full 5-stage curriculum
# ============================================================================

def _parse_weights(s: str) -> dict:
	"""Parse 'err=0.5,stable=0.4,jerk=0.05,mono=0.05' into a dict."""
	parts = [p.strip() for p in s.split(",") if p.strip()]
	w = {}
	for p in parts:
		k, v = p.split("=")
		w[k.strip()] = float(v)
	for k in ("err", "stable", "jerk", "mono"):
		if k not in w:
			raise ValueError(f"missing weight {k!r} in {s!r}; expected err,stable,jerk,mono")
	return w


def _resume_path(out_dir: Path) -> Path:
	"""Where the full-curriculum resume checkpoint lives."""
	return out_dir / "curriculum_resume.pkl"


def _make_stage_record(stage: CurriculumStage, res, ev, wall: float) -> dict:
	"""Resume-safe printable record for one completed stage. Reward + stable
	come from the GA's own result (never cancel-poisoned); err needs one
	re-eval, which we skip when a proper cancel is active (it would return the
	180°/-inf sentinel)."""
	rec = {
		"name": stage.name, "steps": stage.steps, "tilt": stage.tilt_deg,
		"iters": getattr(res, "iterations_run", 0) if res else 0,
		"wall": wall, "err": float("nan"),
		"reward": (-res.final_fitness if (res and res.final_fitness is not None) else float("nan")),
		"stable": (res.final_accuracy if (res and res.final_accuracy is not None) else float("nan")),
		"has_best": bool(res and res.best_genome is not None),
	}
	if rec["has_best"] and not cancel_state.sigterm_received():
		try:
			m = ev.evaluate_batch([res.best_genome])[0]
			rec["err"] = m.mean_attitude_error_deg
			rec["reward"] = m.fitness
			rec["stable"] = m.acc
		except Exception:
			pass
	return rec


def _save_resume(out_dir: Path, *, weights, seed, spec, next_index: int,
                 prev_population, prev_best, cumulative_wall: float,
                 stage_records: list, full_steps=None) -> Path:
	"""Persist enough to continue the curriculum from `next_index` on relaunch."""
	path = _resume_path(out_dir)
	with open(path, "wb") as f:
		pickle.dump({
			"weights": weights, "seed": seed, "spec": spec,
			"next_index": next_index,
			"prev_population": list(prev_population) if prev_population else None,
			"prev_best": prev_best,
			"cumulative_wall": cumulative_wall,
			"stage_records": stage_records,
			"full_steps": full_steps,
			"meta": {"saved_at_unix": time.time(),
			         "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
		}, f)
	return path


def _dump_midstage_resume() -> None:
	"""Per-generation crash checkpoint. Writes curriculum_resume.pkl pointing at the
	CURRENT stage (next_index=idx) + mid_stage_gen=current gen + the in-progress
	population, so --resume re-enters the stage at that generation (GA honors
	_resume_start_gen). Atomic (tmp+os.replace) so a crash mid-write can't corrupt it.
	Context is stashed in _emergency_state['resume_ctx'] at each stage start."""
	ctx = _emergency_state.get("resume_ctx")
	if not ctx:
		return
	path = _resume_path(ctx["out_dir"])
	tmp = path.with_suffix(".pkl.tmp")
	with open(tmp, "wb") as f:
		pickle.dump({
			"weights": ctx["weights"], "seed": ctx["seed"], "spec": ctx["spec"],
			"next_index": ctx["idx"],                       # re-enter THIS stage
			"mid_stage_gen": int(_emergency_state.get("generation", 0)),
			"prev_population": _emergency_state.get("population") or ctx.get("prev_population"),
			"prev_best": _emergency_state.get("best_genome") or ctx.get("prev_best"),
			"cumulative_wall": ctx["cumulative_wall"],
			"stage_records": ctx["stage_records"],
			"full_steps": ctx["full_steps"],
			"meta": {"midstage": True, "saved_at_unix": time.time()},
		}, f)
	os.replace(tmp, path)


def _print_curriculum_report(stage_records: list, cumulative_wall: float, seed: int):
	bar = "=" * 72
	print(f"\n{bar}\n  CURRICULUM RESULT — {len(stage_records)}/{len(DEFAULT_CURRICULUM)} stages "
	      f"— total wall {cumulative_wall/60:.1f} min\n{bar}")
	print(f"  {'stage':<6}  {'steps':>5}  {'tilt':>5}  {'iters':>6}  "
	      f"{'stable':>8}  {'err':>8}  {'reward':>8}  {'wall':>6}")
	for r in stage_records:
		if not r.get("has_best"):
			print(f"  {r['name']:<6}  {r['steps']:>5}  {r['tilt']:>4.1f}°  (no best)")
			continue
		print(f"  {r['name']:<6}  {r['steps']:>5}  {r['tilt']:>4.1f}°  "
		      f"{r['iters']:>6}  {r['stable']*100:>7.1f}%  "
		      f"{r['err']:>7.2f}°  {r['reward']:>8.2f}  {r['wall']:>5.0f}s")
	pid = AttitudePID(AttitudePIDConfig())
	from wnn.control.training import make_pid_action_fn
	final_ec = _build_ec(DEFAULT_CURRICULUM[-1])
	_, pid_m = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, final_ec, 100, seed)
	print(f"  vs PID  (full episodes, val seed):  "
	      f"err={pid_m['mean_attitude_error_deg']:.2f}°  stable={pid_m['stable_rate']*100:.1f}%  "
	      f"reward={pid_m['mean_reward']:.2f}")
	print(f"  vs MLP-GA baseline:  9.66°  (run_mlp_ga.py 3-way held-out)")


def run_full_curriculum(args, weights: dict, seed: int):
	"""Run all 5 curriculum stages sequentially, warm-starting each from the
	previous stage's final population.

	Cancellation (01/06/2026): a PROPER cancel (real SIGTERM, surfaced via
	cancel_state) ABORTS the run after the current stage — it dumps a resume
	checkpoint and returns "ABORTED" instead of grinding the remaining stages
	into 180°/-inf sentinels and printing a fake CURRICULUM RESULT (the bug
	that made the 01/06 overnight run degenerate). Relaunch with
	--resume <save_dir>/curriculum_resume.pkl to continue from the next stage.
	"""
	out_dir = Path(args.save_dir) if args.save_dir else Path("/tmp/curriculum_full")
	out_dir.mkdir(parents=True, exist_ok=True)
	spec = _grid_seed_spec(args, seed)

	# Fixed horizon for ALL full-curriculum stages (overridable via --full-steps;
	# None keeps each stage's schedule default = FIXED_HORIZON_STEPS).
	full_steps = getattr(args, "full_steps", None)

	# Resume? Reload prior progress so we continue from the next stage.
	start_index = 0
	prev_population = None
	prev_best = None
	cumulative_wall = 0.0
	stage_records: list = []
	resume_start_gen = 0   # >0 → mid-stage resume of the first stage (per-gen checkpoint)
	resume_file = getattr(args, "resume", None)
	if resume_file:
		with open(resume_file, "rb") as f:
			rs = pickle.load(f)
		weights        = rs.get("weights", weights)
		seed           = rs.get("seed", seed)
		spec           = rs.get("spec", spec)
		start_index    = rs.get("next_index", 0)
		resume_start_gen = int(rs.get("mid_stage_gen", 0))   # 0 if a stage-boundary checkpoint
		prev_population = rs.get("prev_population")
		prev_best       = rs.get("prev_best")
		cumulative_wall = rs.get("cumulative_wall", 0.0)
		stage_records   = list(rs.get("stage_records", []))
		full_steps      = rs.get("full_steps", full_steps)  # keep horizon consistent
		_midtag = f" (mid-stage @ gen {resume_start_gen})" if resume_start_gen else ""
		print(f"\n  [resume] loaded {resume_file} → continuing at stage "
		      f"{DEFAULT_CURRICULUM[start_index].name if start_index < len(DEFAULT_CURRICULUM) else 'DONE'}{_midtag} "
		      f"({len(stage_records)} stage(s) already complete, {cumulative_wall/60:.1f} min prior)")

	print(f"\n{'='*72}")
	print(f"  CURRICULUM FULL — 5 stages — weights err²={weights['err']:.2f} "
	      f"stable={weights['stable']:.2f} jerk={weights['jerk']:.2f} mono={weights['mono']:.2f}")
	print(f"  seed spec: sn={spec.state_neurons} sb={spec.state_bits_per_neuron} ob={spec.output_bits_per_neuron}")
	print(f"  outdir: {out_dir}")
	print(f"{'='*72}")

	for idx in range(start_index, len(DEFAULT_CURRICULUM)):
		stage = _with_overrides(DEFAULT_CURRICULUM[idx], full_steps,
		                        getattr(args, "full_gens", None),
		                        getattr(args, "full_patience", None))
		# Snapshot the population this stage STARTS from, so a proper cancel
		# mid-stage can resume by RE-RUNNING this (incomplete) stage cleanly
		# rather than skipping ahead with a half-evolved population.
		pop_before  = list(prev_population) if prev_population else None
		best_before = prev_best
		# Stash the full resume context so the per-generation hook can flush a
		# complete, atomic curriculum_resume.pkl every gen (crash → resume this
		# stage at the current gen). cumulative_wall/stage_records here are PRE-stage.
		_emergency_state["resume_ctx"] = {
			"out_dir": out_dir, "weights": weights, "seed": seed, "spec": spec,
			"idx": idx, "cumulative_wall": cumulative_wall,
			"stage_records": list(stage_records), "full_steps": full_steps,
			"prev_population": pop_before, "prev_best": best_before,
		}
		_rsg = resume_start_gen if idx == start_index else 0   # mid-stage only on the resumed stage
		res, ev, wall, _thr = _run_one_stage(args, stage, weights, spec, seed,
		                                     initial_population=prev_population,
		                                     stage_label=stage.name, resume_start_gen=_rsg)
		cumulative_wall += wall

		# PROPER cancel → this stage is INCOMPLETE (the GA was unwound mid-run).
		# Dump a resume that re-runs THIS stage from pop_before; do NOT record
		# the partial result or run the remaining stages (they would all return
		# 180°/-inf sentinels and fake a CURRICULUM RESULT — the 01/06 bug).
		if cancel_state.sigterm_received():
			rp = _save_resume(out_dir, weights=weights, seed=seed, spec=spec,
			                  next_index=idx, prev_population=pop_before,
			                  prev_best=best_before, cumulative_wall=cumulative_wall,
			                  stage_records=stage_records, full_steps=full_steps)
			print(f"\n{'='*72}")
			print(f"  CURRICULUM ABORTED during stage {stage.name} (PROPER cancel, "
			      f"signum={cancel_state.last_signum()}) — {len(stage_records)}/"
			      f"{len(DEFAULT_CURRICULUM)} stages complete; stage {stage.name} incomplete.")
			print(f"  Resume with:  --mode full --resume {rp}")
			print(f"{'='*72}")
			_print_curriculum_report(stage_records, cumulative_wall, seed)
			return "ABORTED"

		# Stage completed cleanly — record + persist + carry population forward.
		_stage_summary(stage, weights, res, ev, wall, prefix=f"[Stage {stage.name}] ")
		stage_records.append(_make_stage_record(stage, res, ev, wall))
		if res is not None and res.best_genome is not None:
			pkl_path = out_dir / f"stage{stage.name}_winner.pkl"
			with open(pkl_path, "wb") as f:
				pickle.dump({
					"stage":      stage.name,
					"weights":    weights,
					"spec":       spec,
					"best_genome": res.best_genome,
					"population":  list(res.final_population) if res.final_population else [],
				}, f)
			print(f"  [save] stage {stage.name} winner → {pkl_path}")
			prev_population = list(res.final_population) if res.final_population else None
			prev_best       = res.best_genome
		else:
			print(f"  [warn] stage {stage.name} produced no winner (empty pool). "
			      f"Carrying prev population forward.")
		# Crash-safe checkpoint: write a resume pointer after EVERY clean stage (not
		# just on SIGTERM) so a HARD crash (power/OOM/SIGKILL) auto-resumes from the
		# next stage via --resume. Bounds worst-case loss to the in-progress stage.
		rp = _save_resume(out_dir, weights=weights, seed=seed, spec=spec,
		                  next_index=idx + 1, prev_population=prev_population,
		                  prev_best=prev_best, cumulative_wall=cumulative_wall,
		                  stage_records=stage_records, full_steps=full_steps)
		print(f"  [checkpoint] stage {stage.name} done → resume pointer at stage "
		      f"{idx + 1}/{len(DEFAULT_CURRICULUM)} ({rp.name})")

	# All stages completed cleanly.
	if _resume_path(out_dir).exists():
		try:
			_resume_path(out_dir).unlink()  # tidy: no stale resume after success
		except Exception:
			pass
	_print_curriculum_report(stage_records, cumulative_wall, seed)
	# TRUE held-out: re-eval the final controller on a fresh seed (the honest
	# paper number). Skipped unless --report-seed is given.
	report_seed = getattr(args, "report_seed", None)
	if report_seed is not None and prev_best is not None:
		try:
			_holdout_eval(args, spec, weights, prev_best, report_seed, seed)
		except Exception as e:
			print(f"  [report-seed] held-out eval failed: {e}")
	elif report_seed is not None:
		print(f"  [report-seed] no final controller to evaluate (no winner).")
	return "DONE"


# ============================================================================
# Main
# ============================================================================

def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--mode", choices=["sweep", "full"], required=True)
	ap.add_argument("--weights", type=str, default=None,
	                help="REQUIRED for --mode full. Format: 'err=0.5,stable=0.4,jerk=0.05,mono=0.05'")
	ap.add_argument("--auto-full", action="store_true",
	                help="After --mode sweep finishes, automatically launch the "
	                     "5-stage curriculum at the winning combo's weights if "
	                     "the top combo's stable_rate >= 1%% (i.e. some signal "
	                     "of learning is present). User can SIGTERM the auto-"
	                     "launched run if needed.")
	# Common GA knobs.
	ap.add_argument("--pop", type=int, default=200,
	                help="Population for the FULL curriculum (the real run).")
	ap.add_argument("--sweep-pop", type=int, default=50,
	                help="Leaner population for the weight-screening sweep (default 50). "
	                     "The sweep only ranks combos, so it doesn't need --pop.")
	ap.add_argument("--combos", type=str, default=None,
	                help="Comma-separated combo names (e.g. 'W2,W3,C7') to run a SUBSET "
	                     "of the sweep — used for multi-seed confirmation on the top-K "
	                     "survivors. Default = all 18.")
	# elitism = fraction of population kept as elites (0.2 = 20%). The formula is now
	# int(pop*elitism) — no hidden ×2 (fixed 04/06). IDS uses the same 20%.
	ap.add_argument("--elitism", type=float, default=0.2)
	ap.add_argument("--crossover-rate", type=float, default=0.5)
	ap.add_argument("--train-workers", type=int, default=3)
	ap.add_argument("--num-eval-folds", type=int, default=5)
	# Output.
	ap.add_argument("--save-dir", type=str, default=None,
	                help="Per-stage / per-combo checkpoint dir. Defaults to /tmp/curriculum_{mode}.")
	ap.add_argument("--resume", type=str, default=None,
	                help="Resume a --mode full run from a curriculum_resume.pkl "
	                     "written when a prior run was cancelled (proper SIGTERM). "
	                     "Continues from the next un-run stage.")
	ap.add_argument("--sweep-steps", type=int, default=None,
	                help="Override the fixed horizon (steps) for the sweep's Stage A. "
	                     f"Default = schedule's {FIXED_HORIZON_STEPS}.")
	ap.add_argument("--full-steps", type=int, default=None,
	                help="Override the fixed horizon (steps) applied to ALL full-"
	                     f"curriculum stages. Default = schedule's {FIXED_HORIZON_STEPS}.")
	# Seeds.
	ap.add_argument("--base-seed", type=int, default=None)
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--report-seed", type=int, default=None,
		help="TRUE held-out: after a --mode full run, re-eval the final controller "
		     "on this fresh seed (must differ from the train seed). The honest paper number.")
	ap.add_argument("--full-gens", type=int, default=None,
		help="--mode full only: gens per stage (default = stage's 30). Sweep unaffected.")
	ap.add_argument("--full-patience", type=int, default=None,
		help="--mode full only: patience per stage (default = stage's 3). Sweep unaffected.")
	ap.add_argument("--check-interval", type=int, default=10,
		help="Gens between patience checks (early-stop fires after patience×check_interval flat gens).")
	args = ap.parse_args()

	_install_signal_handlers()

	base = args.base_seed if args.base_seed is not None else args.seed
	seedset = resolve_seed_set(base=base, run_index=0)
	log_seed_set(seedset)
	record_seed_set(seedset, script="run_curriculum_ga", extra={
		"mode": args.mode, "pop": args.pop,
	})
	seed = seedset.train

	# Exit code 130 (SIGINT convention) on a proper cancel so a supervising
	# script can tell "aborted, resume me" from "finished cleanly" (0).
	if args.mode == "sweep":
		full_pop = args.pop                     # run_sweep repoints args.pop at --sweep-pop
		ranked = run_sweep(args, seed)
		args.pop = full_pop                     # restore the FULL-run population for auto-full
		if cancel_state.sigterm_received():
			print("\n  [cancel] sweep was cancelled — not launching full curriculum.")
			return 130
		if args.auto_full and _is_clear_winner(ranked):
			winner = ranked[0][0]
			weights = {"err": winner["err"], "stable": winner["stable"],
			           "jerk": winner["jerk"], "mono": winner["mono"]}
			print(f"\n{'='*72}")
			print(f"  AUTO-FULL: sweep winner {winner['name']!r} clears the launch "
			      f"heuristic — launching 5-stage curriculum now (pop={args.pop})")
			print(f"{'='*72}")
			outcome = run_full_curriculum(args, weights, seed)
			return 130 if outcome == "ABORTED" else 0
		elif args.auto_full:
			print(f"\n  AUTO-FULL: no combo cleared the launch heuristic "
			      f"(top stable_rate < 1%). Not launching full curriculum.")
	else:
		if args.resume is None and args.weights is None:
			print("ERROR: --weights required for --mode full (or pass --resume)", file=sys.stderr)
			return 2
		weights = _parse_weights(args.weights) if args.weights else {}
		outcome = run_full_curriculum(args, weights, seed)
		return 130 if outcome == "ABORTED" else 0
	return 0


if __name__ == "__main__":
	sys.exit(main())
