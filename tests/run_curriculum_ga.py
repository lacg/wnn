"""Curriculum-GA for the WNN drone controller.

Two modes share most of the code path:

  --mode sweep   4-combo weight sweep, Stage A only (steps=10 / tilt=5° /
                 pop=200 / gens=30 / patience=3). Runs each combo
                 sequentially and reports the winner per combo. Used to
                 pick a fitness-weight recipe before committing to the
                 full curriculum.

  --mode full    Single weight set, 5-stage curriculum with warm-start
                 chain:
                   A: steps=10  tilt=5°
                   B: steps=30  tilt=8°
                   C: steps=100 tilt=10°
                   D: steps=300 tilt=12°
                   E: steps=500 tilt=15° (production)
                 Each stage's final population is the initial population
                 of the next. Population is preserved across stages so
                 the GA's accumulated diversity isn't lost on stage
                 transitions.

Why curriculum-on-steps: at short horizons the controller only needs to
"stop the spin" — selection pressure has a dense supply of partial
successes. At 500 steps the controller must stabilize AND hold for ~5s,
and almost no random genome can do that, so the GA sees almost no
positive signal and plateaus. Curriculum builds the "stabilize" skill at
cheap horizons, then progressively requires "hold" at longer ones.

Designed alongside Luiz on 31/05/2026 after Plan A v5 plateaued at
fit=1.0000 / err=12.61° from gen 0 with no GA improvement over 8 gens.
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
	print(f"\n[{name}] Cancellation requested. Setting Rust cancel flag — "
	      f"will dump state and exit at next safe point.", flush=True)
	try:
		import ram_accelerator
		ram_accelerator.set_cancel_flag()
	except Exception as e:
		print(f"[{name}] Could not set Rust cancel flag: {e}", flush=True)


def _install_signal_handlers() -> None:
	signal.signal(signal.SIGTERM, _sigterm_handler)
	signal.signal(signal.SIGINT,  _sigterm_handler)
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
		_emergency_state["population"]  = list(ctx.get("population", []))
		_emergency_state["best_genome"] = ctx.get("best_genome")
		_emergency_state["generation"]  = generation
		try:
			import ram_accelerator
			if ram_accelerator.is_cancelled():
				_dump_emergency_state()
				raise StopIteration
		except StopIteration:
			raise
		except Exception:
			pass
		return original(generation, **ctx)
	strat._on_generation_start = wrapped


# ============================================================================
# Curriculum stage definition
# ============================================================================

class CurriculumStage:
	"""One curriculum stage: a (steps, tilt, gens, patience) tuple plus a
	short name. Stages run sequentially with the GA's final_population
	carried forward as initial_population of the next."""
	__slots__ = ("name", "steps", "tilt_deg", "gens", "patience", "eval_episodes")
	def __init__(self, name, steps, tilt_deg, gens, patience, eval_episodes):
		self.name           = name
		self.steps          = steps
		self.tilt_deg       = tilt_deg
		self.gens           = gens
		self.patience       = patience
		self.eval_episodes  = eval_episodes


# 5-stage default schedule (per Luiz's 31/05/2026 spec). Used directly by
# --mode full; --mode sweep runs only Stage A.
DEFAULT_CURRICULUM: list[CurriculumStage] = [
	CurriculumStage("A", steps=10,  tilt_deg=5.0,  gens=30, patience=3, eval_episodes=100),
	CurriculumStage("B", steps=30,  tilt_deg=8.0,  gens=30, patience=3, eval_episodes=100),
	CurriculumStage("C", steps=100, tilt_deg=10.0, gens=30, patience=3, eval_episodes=100),
	CurriculumStage("D", steps=300, tilt_deg=12.0, gens=30, patience=3, eval_episodes=100),
	CurriculumStage("E", steps=500, tilt_deg=15.0, gens=30, patience=3, eval_episodes=100),
]


def _build_ec(stage: CurriculumStage) -> EpisodeConfig:
	"""Episode config for this curriculum stage."""
	return EpisodeConfig(
		dt=0.001, steps_per_episode=stage.steps,
		max_initial_tilt_rad=math.radians(stage.tilt_deg),
		max_initial_yaw_rad=math.radians(stage.tilt_deg),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
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
	return cfg


def _run_one_stage(args, stage: CurriculumStage, weights: dict,
                   spec: ControllerSpec, seed: int,
                   initial_population=None,
                   stage_label: str = ""):
	"""Run ONE curriculum stage. Returns (res, ev, wall_time, fitted_thresholds)."""
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
	      f"tilt={stage.tilt_deg:>4.1f}°  pop={ev.num_eval_episodes if hasattr(ev, 'num_eval_episodes') else '?'}  "
	      f"gens={stage.gens} pat={stage.patience}")
	print(f"  weights: err²={w['err']:.2f}  stable={w['stable']:.2f}  "
	      f"jerk={w['jerk']:.2f}  mono={w['mono']:.2f}")
	# Best metrics: pull from the strategy's result.
	if res is None or res.best_genome is None:
		print(f"  result: NO BEST (likely cancelled)")
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


# ============================================================================
# Mode 1: Weight sweep at Stage A only
# ============================================================================

# Per Luiz's 31/05/2026 spec: err-heavy with stable second, jerk + mono small
# but non-zero.
SWEEP_COMBOS: list[dict] = [
	{"name": "W1",  "err": 0.50, "stable": 0.40, "jerk": 0.05, "mono": 0.05},
	{"name": "W2",  "err": 0.40, "stable": 0.50, "jerk": 0.05, "mono": 0.05},
	{"name": "W3",  "err": 0.60, "stable": 0.30, "jerk": 0.05, "mono": 0.05},
	{"name": "W4",  "err": 0.45, "stable": 0.35, "jerk": 0.10, "mono": 0.10},
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
	stage = DEFAULT_CURRICULUM[0]  # Stage A
	seed_spec = _grid_seed_spec(args, seed)
	print(f"\n{'='*72}")
	print(f"  CURRICULUM SWEEP — Stage A only — {len(SWEEP_COMBOS)} weight combos")
	print(f"  seed spec: sn={seed_spec.state_neurons} sb={seed_spec.state_bits_per_neuron} ob={seed_spec.output_bits_per_neuron}")
	print(f"  stage A: steps={stage.steps} tilt={stage.tilt_deg}° pop={args.pop} gens={stage.gens} pat={stage.patience}")
	print(f"  outdir: {out_dir}")
	print(f"{'='*72}")

	combo_results = []
	for combo in SWEEP_COMBOS:
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

	# Final ranking
	print(f"\n{'='*72}\n  SWEEP RESULT — by best-genome stable_rate (then err)\n{'='*72}")
	ranked = []
	for combo, res, ev, wall in combo_results:
		if res is None or res.best_genome is None:
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


def run_full_curriculum(args, weights: dict, seed: int):
	"""Run all 5 curriculum stages sequentially, warm-starting each from the
	previous stage's final population."""
	out_dir = Path(args.save_dir) if args.save_dir else Path("/tmp/curriculum_full")
	out_dir.mkdir(parents=True, exist_ok=True)
	# Same seed spec as the sweep — keeps the comparison fair.
	spec = _grid_seed_spec(args, seed)
	print(f"\n{'='*72}")
	print(f"  CURRICULUM FULL — 5 stages — weights err²={weights['err']:.2f} "
	      f"stable={weights['stable']:.2f} jerk={weights['jerk']:.2f} mono={weights['mono']:.2f}")
	print(f"  seed spec: sn={spec.state_neurons} sb={spec.state_bits_per_neuron} ob={spec.output_bits_per_neuron}")
	print(f"  outdir: {out_dir}")
	print(f"{'='*72}")

	prev_population = None
	prev_best = None
	cumulative_wall = 0.0
	stage_summaries = []
	for stage in DEFAULT_CURRICULUM:
		res, ev, wall, _thr = _run_one_stage(args, stage, weights, spec, seed,
		                                     initial_population=prev_population,
		                                     stage_label=stage.name)
		cumulative_wall += wall
		_stage_summary(stage, weights, res, ev, wall, prefix=f"[Stage {stage.name}] ")
		stage_summaries.append((stage, res, ev, wall))
		# Persist per-stage checkpoint.
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
			# Carry the entire evolved population forward — strictly stronger
			# than warm-starting with just the best genome.
			prev_population = list(res.final_population) if res.final_population else None
			prev_best       = res.best_genome
		else:
			print(f"  [warn] stage {stage.name} produced no winner (cancelled or empty pool). "
			      f"Carrying prev population forward.")

	# Final report
	bar = "=" * 72
	print(f"\n{bar}\n  CURRICULUM RESULT — 5 stages — total wall {cumulative_wall/60:.1f} min\n{bar}")
	print(f"  {'stage':<6}  {'steps':>5}  {'tilt':>5}  {'iters':>6}  "
	      f"{'stable':>8}  {'err':>8}  {'reward':>8}  {'wall':>6}")
	for stage, res, ev, wall in stage_summaries:
		if res is None or res.best_genome is None:
			print(f"  {stage.name:<6}  {stage.steps:>5}  {stage.tilt_deg:>4.1f}°  ?")
			continue
		try:
			m = ev.evaluate_batch([res.best_genome])[0]
			print(f"  {stage.name:<6}  {stage.steps:>5}  {stage.tilt_deg:>4.1f}°  "
			      f"{res.iterations_run:>6}  {m.acc*100:>7.1f}%  "
			      f"{m.mean_attitude_error_deg:>7.2f}°  {m.fitness:>8.2f}  {wall:>5.0f}s")
		except Exception:
			print(f"  {stage.name:<6}  (eval failed)")
	# Baselines
	pid = AttitudePID(AttitudePIDConfig())
	from wnn.control.training import make_pid_action_fn
	final_ec = _build_ec(DEFAULT_CURRICULUM[-1])
	_, pid_m = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, final_ec, 100, seed)
	print(f"  vs PID  (full episodes, val seed):  "
	      f"err={pid_m['mean_attitude_error_deg']:.2f}°  stable={pid_m['stable_rate']*100:.1f}%  "
	      f"reward={pid_m['mean_reward']:.2f}")
	print(f"  vs MLP-GA baseline:  9.66°  (run_mlp_ga.py 3-way held-out)")


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
	ap.add_argument("--pop", type=int, default=200)
	ap.add_argument("--elitism", type=float, default=0.2)
	ap.add_argument("--crossover-rate", type=float, default=0.5)
	ap.add_argument("--train-workers", type=int, default=3)
	ap.add_argument("--num-eval-folds", type=int, default=5)
	# Output.
	ap.add_argument("--save-dir", type=str, default=None,
	                help="Per-stage / per-combo checkpoint dir. Defaults to /tmp/curriculum_{mode}.")
	# Seeds.
	ap.add_argument("--base-seed", type=int, default=None)
	ap.add_argument("--seed", type=int, default=42)
	args = ap.parse_args()

	_install_signal_handlers()

	base = args.base_seed if args.base_seed is not None else args.seed
	seedset = resolve_seed_set(base=base, run_index=0)
	log_seed_set(seedset)
	record_seed_set(seedset, script="run_curriculum_ga", extra={
		"mode": args.mode, "pop": args.pop,
	})
	seed = seedset.train

	if args.mode == "sweep":
		ranked = run_sweep(args, seed)
		if args.auto_full and _is_clear_winner(ranked):
			winner = ranked[0][0]
			weights = {"err": winner["err"], "stable": winner["stable"],
			           "jerk": winner["jerk"], "mono": winner["mono"]}
			print(f"\n{'='*72}")
			print(f"  AUTO-FULL: sweep winner {winner['name']!r} clears the launch "
			      f"heuristic — launching 5-stage curriculum now")
			print(f"{'='*72}")
			run_full_curriculum(args, weights, seed)
		elif args.auto_full:
			print(f"\n  AUTO-FULL: no combo cleared the launch heuristic "
			      f"(top stable_rate < 1%). Not launching full curriculum.")
	else:
		if args.weights is None:
			print("ERROR: --weights required for --mode full", file=sys.stderr)
			return 2
		weights = _parse_weights(args.weights)
		run_full_curriculum(args, weights, seed)
	return 0


if __name__ == "__main__":
	sys.exit(main())
