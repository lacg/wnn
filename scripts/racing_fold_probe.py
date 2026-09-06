#!/usr/bin/env python3
"""RACING PREDICTIVITY PROBE — does the score after f of K training folds predict the
score after all K? (06/09/2026, Luiz: "run the predictivity probe before any code".)

WHY THE FOLD IS THE RUNG. Measured on 06/09: ~80% of a CONNECTIONS generation is
DAgger training (~1,520 s for 60 candidates), ~20% is scoring. Each genome trains
K=5 folds x 8 reward-gated rounds; the Rust batch trainer accumulates fold k+1 onto
fold k's cells in ONE call, calling the per-fold trainer afresh for every fold seed
(dagger_train.rs: `for &seed_k in &fold_seeds[i] { dagger_train_inplace_rs(..) }`).
Every piece of per-call state — the improvement gate's running history, the
best-checkpoint snapshot, the curriculum tilt ramp (a function of num_rounds) — is
scoped to ONE fold. So cutting INSIDE a fold would distort the schedule, but cutting
AT a fold boundary is exact: the survivors' remaining folds are the same Rust calls
they would have received anyway, warm-started from the exported cells exactly as the
Lamarckian path warm-starts every generation. Racing therefore needs NO Rust change.

WHAT THIS MEASURES. One realistic generation of offspring (tournament + crossover +
mutation from a banked stage population, via the real strategy), trained fold by
fold with the evaluator's own batched trainer, CRN-scored on all pools after EVERY
fold. Then, for each cut f < K: rank correlation of the real gated fitness at fold
f vs fold K; how much of the true top third the fold-f top third would keep; and
the REGRET — the best true fitness among fold-f survivors vs the best overall.
Plus the EXACTNESS check: the fold-K score from K single-fold calls must equal the
score from the one K-fold call the GA makes today. If it does not, the design is
dead before it is built.

It reuses ControllerEvaluator._train_genomes_rust_batched / _score_fitness /
_train_base_seeds and the strategy's own operators — never a reimplementation.

⚠️ This is a CONTROLLER PROCESS (GPU + CPU). Run it only on an idle box; the chain
wrapper (scripts/racing_fold_probe_chain.sh) enforces that.
"""
import argparse, json, math, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def parse_probe_args():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--ckpt", required=True, help="stage checkpoint (yaml.gz) whose population seeds the offspring")
	p.add_argument("--recipe-args", required=True, help="the phased_ga argv of the recipe, one string")
	p.add_argument("--out", required=True, help="JSON results path")
	p.add_argument("--candidates", type=int, default=60, help="offspring to train (the GA evaluates 60 per gen)")
	p.add_argument("--keep-fraction", type=float, default=1.0 / 3.0)
	p.add_argument("--reference", action="store_true",
	               help="ALSO train every candidate with the single K-fold call the GA makes today "
	                    "and compare fold-K scores (the exactness check; doubles training time)")
	p.add_argument("--dry-run", action="store_true", help="build everything, generate offspring, train nothing")
	return p.parse_args()


def build_episode_config(args):
	"""Mirror of the EpisodeConfig block in phased_ga.main(). Kept identical field for
	field; fold into a shared helper when the box is idle (phased_ga is live-imported)."""
	from wnn.control.training import DisturbanceConfig, EpisodeConfig
	from wnn.control.airframe import Airframe
	dist = DisturbanceConfig.preset(args.disturbance, seed=911)
	return EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate,
		disturbance=dist,
		airframe=(None if not getattr(args, "airframe", None) else Airframe.preset(args.airframe)),
		translation=bool(getattr(args, "translation", False)),
		max_initial_alt_offset_m=float(getattr(args, "alt_offset", 0.3)),
		max_initial_vz=float(getattr(args, "init_vz", 0.2)),
		collective_cmd_jitter=float(getattr(args, "collective_jitter", 0.1)),
		mass_jitter=float(getattr(args, "mass_jitter", 0.15)),
		target_altitude=float(getattr(args, "target_altitude", 0.0)),
		lambda_alt=float(getattr(args, "reward_lambda_alt", 0.0)),
		max_initial_xy_offset_m=float(getattr(args, "xy_offset", 0.0)),
		lambda_pos=float(getattr(args, "reward_lambda_pos", 0.0)),
		calib_airframe=bool(getattr(args, "calib_airframe", False)),
	)


def build_evaluator_and_strategy(args, ec, spec, seed):
	"""Same construction as phased_ga._run_arch_phase for the CONNECTIONS stage."""
	from wnn.control import phased_ga as pg
	from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
	from wnn.control.arch_strategy import ControllerArchGAStrategy, default_controller_arch_config
	from wnn.ram.strategies.optimization_dimension import OptimizationDimension
	thresholds = fit_thresholds_from_pid_rollouts(
		spec, num_episodes=10, seed=seed,
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None),
		episode_config=pg._calib_ec(args, ec))
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes, seed=seed,
	                         episode_config=ec, thresholds=thresholds,
	                         rg_config=pg._rg_config(args, ec, seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=args.num_eval_folds, score_crn=args.score_crn)
	arch_cfg = default_controller_arch_config(spec)
	arch_cfg.max_state_neurons = max(arch_cfg.max_state_neurons, 4 * max(args.grid_state_neurons))
	if getattr(args, "max_state_neurons", None) is not None:
		arch_cfg.max_state_neurons = min(arch_cfg.max_state_neurons, int(args.max_state_neurons))
		arch_cfg.min_state_neurons = min(arch_cfg.min_state_neurons, arch_cfg.max_state_neurons)
	pg.apply_output_neuron_ceiling(args, arch_cfg)
	arch_cfg.min_state_neurons = max(arch_cfg.min_state_neurons, min(args.grid_state_neurons))
	arch_cfg.saturation_grow_gain = getattr(args, "saturation_grow_gain", 0.02)
	arch_cfg.max_cells = getattr(args, "max_cells", 1_000_000_000)
	arch_cfg.strict_cell_budget = bool(getattr(args, "max_cells_strict", False))
	gacfg = pg._build_ga_config(args, args.conns_gens, args.conns_patience)
	strat = ControllerArchGAStrategy(spec, OptimizationDimension.CONNECTIONS, arch_config=arch_cfg,
	                                 ga_config=gacfg, seed=seed, batch_evaluator=ev,
	                                 lamarckian=bool(getattr(args, "lamarckian", False)))
	return ev, strat, gacfg


def metrics_from_scored(scored):
	"""The same Metrics the evaluator hands the fitness calculator (evaluator._evaluate_core)."""
	from wnn.ram.metrics import ControllerMetrics as Metrics
	out = []
	for reward, m in scored:
		g = lambda k: (float(m[k]) if m.get(k) is not None else None)
		out.append(Metrics(
			reward=float(reward), stable_rate=float(m.get("stable_rate", 0.0)), fitness=float(reward),
			mean_attitude_error_deg=float(m.get("mean_attitude_error_deg", 0.0)),
			motor_jerk_mean=g("mean_pwm_jerk"), mono_violations_total=g("mono_violations"),
			mean_steady_error_deg=g("mean_steady_error_deg"), mean_effort=g("mean_effort"),
			mean_position_error_m=g("mean_position_error_m"), mean_altitude_error_m=g("mean_altitude_error_m")))
	return out


_CALC = {}


def fitness_of(gacfg, metrics_list):
	"""The GA's OWN calculator (lower = better), exactly what selection ranks on —
	GAConfig.create_fitness_calculator() is the factory the strategy itself uses."""
	calc = _CALC.get("calc")
	if calc is None:
		calc = _CALC["calc"] = gacfg.create_fitness_calculator()
	return list(calc.fitness(metrics_list))


def make_offspring(strat, population, pop_fitness, n):
	"""Tournament + crossover + mutation via the strategy's own operators — the body of
	GenericGAStrategy._generate_offspring's generator, minus the viability re-draw."""
	cfg = strat._config
	if getattr(strat, "_rng", None) is None:
		# optimize() seeds the operator RNG on entry; we never enter optimize().
		import random
		strat._rng = random.Random(getattr(strat, "_seed", None))
	pop_tuples = [(g, f, None) for g, f in zip(population, pop_fitness)]
	kids = []
	for _ in range(n):
		p1 = strat._tournament_select(pop_tuples, cfg.tournament_size)
		p2 = strat._tournament_select(pop_tuples, cfg.tournament_size)
		child = strat.crossover_genomes(p1, p2) if strat._rng.random() < cfg.crossover_rate else strat.clone_genome(p1)
		kids.append(strat.mutate_genome(child, cfg.mutation_rate))
	return kids


def train_fold_by_fold(ev, genomes, K, log):
	"""Fold k trains from the cells fold k-1 exported: K single-fold Rust calls. Returns
	per-fold scored lists and the wall time of each fold's train+score."""
	from wnn.control import _accel as ra
	N = len(genomes)
	ev._ensure_ga_ready()
	ev._cur_axes = ev._active_axes(0)
	ev._advance_fold()
	base = ev._train_base_seeds(N, 0)
	shape_keys = [ev._shape_key(g) for g in genomes]
	inits = [(getattr(g, "cells", None) or ra.GenomeCells()) for g in genomes]
	per_fold, timing = [], []
	for k in range(K):
		t0 = time.time()
		tasks = [(gi, [base[gi] + k]) for gi in range(N)]
		trained = ev._train_genomes_rust_batched(genomes, tasks, init_override=inits)
		t1 = time.time()
		controllers = [c for (c, _s) in trained]
		scored = ev._score_fitness(controllers, shape_keys)
		t2 = time.time()
		inits = [c.export_cells_handle() for c in controllers]
		per_fold.append(scored)
		timing.append(dict(fold=k + 1, train_s=round(t1 - t0, 1), score_s=round(t2 - t1, 1)))
		log(f"fold {k + 1}/{K}: train {t1 - t0:.0f}s  score {t2 - t1:.0f}s")
	return per_fold, timing


def train_reference(ev, genomes, K, log):
	"""The ONE K-fold call the GA makes today (evaluator._evaluate_core's Rust path)."""
	from wnn.control import _accel as ra
	N = len(genomes)
	base = ev._train_base_seeds(N, 0)
	shape_keys = [ev._shape_key(g) for g in genomes]
	inits = [(getattr(g, "cells", None) or ra.GenomeCells()) for g in genomes]
	t0 = time.time()
	trained = ev._train_genomes_rust_batched(genomes, [(gi, [base[gi] + k for k in range(K)]) for gi in range(N)],
	                                         init_override=inits)
	scored = ev._score_fitness([c for (c, _s) in trained], shape_keys)
	log(f"reference K-fold call: {time.time() - t0:.0f}s")
	return scored


def spearman(a, b):
	def ranks(v):
		order = sorted(range(len(v)), key=lambda i: v[i])
		r = [0.0] * len(v)
		i = 0
		while i < len(order):
			j = i
			while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
				j += 1
			for t in range(i, j + 1):
				r[order[t]] = (i + j) / 2.0 + 1.0
			i = j + 1
		return r
	ra_, rb = ranks(a), ranks(b)
	n = len(a)
	ma, mb = sum(ra_) / n, sum(rb) / n
	num = sum((x - ma) * (y - mb) for x, y in zip(ra_, rb))
	den = math.sqrt(sum((x - ma) ** 2 for x in ra_) * sum((y - mb) ** 2 for y in rb))
	return num / den if den else float("nan")


def analyse(fit_by_fold, keep_fraction):
	"""fit_by_fold[k] = list of GA fitness (lower = better) after fold k+1."""
	K = len(fit_by_fold)
	final = fit_by_fold[-1]
	N = len(final)
	keep = max(1, int(round(N * keep_fraction)))
	true_top = set(sorted(range(N), key=lambda i: final[i])[:keep])
	best_true = min(final)
	rows = []
	for k in range(K - 1):
		f = fit_by_fold[k]
		surv = set(sorted(range(N), key=lambda i: f[i])[:keep])
		best_surv = min(final[i] for i in surv)
		rows.append(dict(cut_after_fold=k + 1, spearman=round(spearman(f, final), 3),
		                 top_kept=len(surv & true_top), top_size=keep,
		                 regret=round(best_surv - best_true, 4),
		                 true_best_survives=(min(range(N), key=lambda i: final[i]) in surv),
		                 train_units=round((N * (k + 1) + keep * (K - k - 1)) / (N * K), 3)))
	return rows


def main():
	pa = parse_probe_args()
	from wnn.control import phased_ga as pg
	from wnn.control.checkpoint_io import load_controller_checkpoint
	args = pg.build_arg_parser().parse_args(pa.recipe_args.split())
	seed = int(args.base_seed)
	log = lambda s: print(f"[racing-probe] {time.strftime('%H:%M:%S')} {s}", flush=True)

	payload = load_controller_checkpoint(pa.ckpt)
	if payload is None:
		raise SystemExit(f"checkpoint not loadable: {pa.ckpt}")
	population = list(payload.get("population") or [])
	spec = payload.get("spec")
	if not population or spec is None:
		raise SystemExit("checkpoint carries no population/spec")
	log(f"loaded {len(population)} genomes from {os.path.basename(pa.ckpt)} (stage {payload.get('stage_name')})")

	ec = build_episode_config(args)
	ev, strat, gacfg = build_evaluator_and_strategy(args, ec, spec, seed)
	K = ev.num_eval_folds
	log(f"evaluator: folds={K} eval_episodes={ev.num_eval} crn={ev.score_crn} rounds/fold={ev.rg_config.num_rounds}")

	# Parents need a fitness for tournament selection: score them as they are (cells
	# ARE the genome for a banked population — no training), rank with the GA's calculator.
	if pa.dry_run:
		# No GPU work in a dry run (it may share the box with a live controller):
		# uniform parent fitness makes tournament selection uniform, which is enough
		# to exercise the operators and the plumbing.
		parent_fit = [0.0] * len(population)
		log("dry-run: parents NOT scored (uniform tournament)")
	else:
		t0 = time.time()
		parent_scored = ev.score_genomes(population)
		parent_fit = fitness_of(gacfg, parent_scored)
		log(f"parents scored ({time.time() - t0:.0f}s); best fitness {min(parent_fit):.4f}")
	kids = make_offspring(strat, population, parent_fit, pa.candidates)
	inherited = sum(1 for g in kids if getattr(g, "cells", None) is not None)
	log(f"{len(kids)} offspring generated ({inherited} carry inherited cells)")
	if pa.dry_run:
		log("dry-run: stopping before training")
		return 0

	per_fold, timing = train_fold_by_fold(ev, kids, K, log)
	metrics_by_fold = [metrics_from_scored(s) for s in per_fold]
	fit_by_fold = [fitness_of(gacfg, m) for m in metrics_by_fold]
	rows = analyse(fit_by_fold, pa.keep_fraction)

	result = dict(
		ckpt=pa.ckpt, stage=payload.get("stage_name"), recipe_args=pa.recipe_args, seed=seed,
		candidates=len(kids), inherited_cells=inherited, folds=K, keep_fraction=pa.keep_fraction,
		timing=timing, cuts=rows,
		per_fold=[[dict(reward=float(r), stable=float(m.get("stable_rate", 0.0)),
		                err=float(m.get("mean_attitude_error_deg", 0.0)),
		                steady=(float(m["mean_steady_error_deg"]) if m.get("mean_steady_error_deg") is not None else None),
		                alt=(float(m["mean_altitude_error_m"]) if m.get("mean_altitude_error_m") is not None else None),
		                fitness=float(f))
		           for (r, m), f in zip(s, fb)] for s, fb in zip(per_fold, fit_by_fold)],
	)
	if pa.reference:
		ref = train_reference(ev, kids, K, log)
		d = [abs(float(a[0]) - float(b[0])) for a, b in zip(per_fold[-1], ref)]
		result["exactness"] = dict(max_abs_reward_delta=max(d), mean_abs_reward_delta=sum(d) / len(d),
		                           identical=all(x == 0.0 for x in d))
		log(f"EXACTNESS fold-by-fold vs one K-fold call: max|Δreward| = {max(d):.6f}  identical={result['exactness']['identical']}")

	os.makedirs(os.path.dirname(os.path.abspath(pa.out)), exist_ok=True)
	with open(pa.out, "w") as fh:
		json.dump(result, fh, indent=1)
	print()
	print("  cut after  spearman  top-third kept  regret(fitness)  true-best survives  train units")
	for r in rows:
		print("  fold %d/%d   %+.3f     %2d/%2d            %.4f          %-5s               %.2f"
		      % (r["cut_after_fold"], K, r["spearman"], r["top_kept"], r["top_size"], r["regret"],
		         r["true_best_survives"], r["train_units"]))
	print(f"\n  written {pa.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
