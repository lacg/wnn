#!/usr/bin/env python3
"""CTRL-21 — CRN comparison-noise probe: measure the per-metric noise of a PAIRED
genome comparison inside one CRN frame, so the MAD floors (CTRL-20) stop being guesses.

THE ESTIMATOR (pre-registered on CTRL-21). Model a scored metric as
	X_g(F) = mu_g + b_F + eps_gF
mu_g the genome's true value, b_F a pool-set effect COMMON to every genome (what CRN
buys), eps_gF the genome-specific residual. The z-combine needs the within-frame
between-genome noise Var(X_a(F) - X_b(F)) = 2 sigma^2. Re-score the SAME genome on
two pool-sets: X_g(A) - X_g(B) = (b_A - b_B) + (eps_gA - eps_gB). The b term is
identical for every genome, so the SD ACROSS GENOMES of D_g = X_g(A) - X_g(B) cancels
it and leaves sqrt(2) sigma — exactly the within-frame comparison noise.
	SD_g(D) IS the floor.   mean(D) is the frame shift CRN protects against.

Per metric this reports mean(D), SD_g(D) (+ SE ~ SD/sqrt(2(n-1))), the metric's
1.4826*MAD across genomes (the spread the z-combine divides by), and noise/spread.
A column whose noise >= its spread is pure noise in that population. Over ALL genomes
and the FLYING subset (stable >= --flying-stable in EVERY frame) — sigma is
heteroscedastic. --frames 3 adds frame C: the three pairwise SDs should agree
(additivity check). NOTE frame A defaults to the run's OWN search pools, on which the
population was SELECTED — so A-B can carry selection optimism that B-C (both fresh)
cannot; with --frames 3 the A-vs-fresh / B-C gap measures exactly that.

MACHINERY, not a reimplementation: the population comes from the banked stage
checkpoint (population, NEVER best_genome — project_checkpoint_best_vs_pop0); the
regime is phased_ga's OWN parser + episode_config_from_args on the run's exact argv
(--run-args-file); thresholds are phased_ga._report_thresholds fit ONCE on the train
seed (the address function the cells were written under, = the MEMORY-stage fit);
scoring is ControllerEvaluator.score_genomes with score_crn=True — score-only, no
training, nothing written back — i.e. the MEMORY stage's own fitness call.
--run-out cross-checks frame A's pools + episode count against the run's logged
'CRN fitness' line, so a wrong argv/seed fails loudly instead of scoring other pools.

Smoke (tiny, safe on a live box):
	crn_noise_probe.py --ckpt <stage4_memory.yaml.gz> --run-args-file <argv.txt> \\
	    --limit-genomes 2 --pools 1 --episodes 5 --out /tmp/x.json
(--pools != 5 marks the output smoke=true; never cite it.)
"""
import argparse
import json
import math
import os
import re
import shlex
import statistics
import sys
import time
from dataclasses import dataclass

METRICS = ("stable_rate", "mean_attitude_error_deg", "mean_steady_error_deg",
           "mean_altitude_error_m", "mean_position_error_m", "motor_jerk_mean",
           "mono_violations_total", "mean_effort", "reward")
MAD_K = 1.4826
FRAME_NAMES = "ABC"
# Frame seeds for the fresh frames: fixed odd offsets from the train seed, so the
# probe is reproducible and never collides with the run's test/val/report seeds.
FRESH_OFFSETS = (0, 0x2C1B3C6D, 0x5E2D58D9)
_CRN_LINE = re.compile(r"CRN fitness: every genome scored on all (\d+) pools \[([0-9, ]+)\].*?\((\d+) episodes each")


@dataclass(frozen=True)
class Regime:
	"""The run's parsed phased_ga namespace + its EpisodeConfig + train seed."""
	args: object
	ec: object
	train_seed: int


def _load_regime(argv_path):
	"""Resolve the run EXACTLY as phased_ga would, from its recorded argv."""
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args
	from wnn.seeds import resolve_seed_set
	with open(argv_path) as f:
		argv = shlex.split(" ".join(l for l in f.read().splitlines() if not l.lstrip().startswith("#")))
	args = build_arg_parser().parse_args(argv)
	ec = episode_config_from_args(args)
	if ec.translation and ec.airframe is None:
		raise SystemExit("--translation requires --airframe (phased_ga main()'s guard)")
	train = resolve_seed_set(base=args.base_seed, run_index=0).train
	return Regime(args, ec, train)


def _load_population(path, limit):
	"""→ (spec, [genomes], meta). The saved final POPULATION, in saved order."""
	from wnn.control.checkpoint_io import load_controller_checkpoint
	payload = load_controller_checkpoint(path)
	if not payload:
		raise SystemExit(f"no checkpoint at {path}")
	pop = list(payload.get("population") or [])
	if not pop:
		raise SystemExit(f"{path} has no final population — refusing best_genome")
	missing = [i for i, g in enumerate(pop) if getattr(g, "cells", None) is None]
	if missing:
		raise SystemExit(f"population members without cells (not score-only): {missing[:10]}")
	meta = {"stage_name": payload.get("stage_name"), "generation": payload.get("generation"),
	        "population_size": len(pop)}
	return payload["spec"], pop[:limit] if limit else pop, meta


def _pools(seed, k):
	"""The pool seeds an evaluator with seed/num_eval_folds=k actually samples from."""
	from wnn.control.evaluator import fold_pool_seed
	return [fold_pool_seed(seed, i) for i in range(k)] if k > 1 else [seed]


def _frame_seeds(a, reg):
	seeds = a.frame_seeds or [(reg.train_seed + o) & 0xFFFFFFFF for o in FRESH_OFFSETS]
	seeds = seeds[:a.frames]
	if len(seeds) < a.frames:
		raise SystemExit(f"--frame-seeds gives {len(seeds)} seeds for --frames {a.frames}")
	return seeds


def _check_disjoint(frames):
	"""Frames must not share a single pool, or D cancels real noise."""
	seen = {}
	for name, pools in frames.items():
		for p in pools:
			if p in seen:
				raise SystemExit(f"pool {p} shared by frames {seen[p]} and {name}")
			seen[p] = name


def _logged_crn(out_path):
	"""(pools, episodes) of the LAST 'CRN fitness' line in the run .out (= MEMORY)."""
	hit = None
	with open(out_path, errors="replace") as f:
		for line in f:
			m = _CRN_LINE.search(line)
			if m:
				hit = m
	if hit is None:
		raise SystemExit(f"no 'CRN fitness' line in {out_path}")
	return [int(x) for x in hit.group(2).split(",")], int(hit.group(3))


def _verify_against_run(a, frame_a_pools, episodes):
	"""Frame A defaults to the run's search pools — prove it on the .out."""
	pools, eps = _logged_crn(a.run_out)
	ok = pools == frame_a_pools and eps == episodes
	print(f"  [verify] run .out MEMORY CRN pools {pools} x {eps} ep — frame A "
	      f"{frame_a_pools} x {episodes} ep: {'MATCH' if ok else 'MISMATCH'}", flush=True)
	if not ok and not a.smoke_ok:
		raise SystemExit("frame A does not reproduce the run's search frame (wrong argv/seed?)")
	return ok


def _thresholds(reg, spec):
	"""The MEMORY stage's fit: PID rollouts on the train seed, calib ec included."""
	from wnn.control.phased_ga import _report_thresholds
	return _report_thresholds(reg.args, reg.ec, spec, reg.train_seed, reg.train_seed, True)


def _evaluator(reg, spec, thresholds, seed, a):
	"""The MEMORY stage's search evaluator, knob for knob, on another pool seed."""
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control.phased_ga import _rg_config
	return ControllerEvaluator(spec, num_eval_episodes=a.episodes, seed=seed,
	                           episode_config=reg.ec, thresholds=thresholds,
	                           rg_config=_rg_config(reg.args, reg.ec, seed),
	                           max_train_workers=reg.args.train_workers,
	                           num_eval_folds=a.pools, score_crn=True)


def _value(m, name):
	v = getattr(m, name, None)
	return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _score_frame(reg, spec, genomes, thresholds, seed, a):
	"""→ [{metric: value}] per genome, in population order (score-only)."""
	t0 = time.time()
	ev = _evaluator(reg, spec, thresholds, seed, a)
	rows = [{k: _value(m, k) for k in METRICS} for m in ev.score_genomes(genomes)]
	print(f"  frame seed {seed}: {len(genomes)} genomes scored in {time.time() - t0:.0f}s", flush=True)
	return rows


def _robust_spread(xs):
	med = statistics.median(xs)
	return MAD_K * statistics.median(abs(x - med) for x in xs)


def _pair_stats(xa, xb):
	"""One metric, one frame pair, one subset → the CTRL-21 numbers."""
	pairs = [(x, y) for x, y in zip(xa, xb) if x is not None and y is not None]
	n = len(pairs)
	if n < 2:
		return {"n": n}
	d = [x - y for x, y in pairs]
	sd = statistics.stdev(d)
	spread = _robust_spread([x for x, _ in pairs] + [y for _, y in pairs])
	return {"n": n, "mean_D": statistics.mean(d), "sd_D": sd,
	        "se_sd_D": sd / math.sqrt(2 * (n - 1)), "mad_spread": spread,
	        "noise_over_spread": (sd / spread) if spread > 0 else None,
	        "spread_zero": spread == 0}  # saturated column: MAD=0, any noise is infinite z


def _flying(frames, thr):
	"""Genome indices stable >= thr in EVERY frame (the saturation regime)."""
	n = len(next(iter(frames.values())))
	return [i for i in range(n)
	        if all((rows[i]["stable_rate"] or 0.0) >= thr for rows in frames.values())]


def _subset_stats(frames, idx):
	"""{pair: {metric: stats}} over the genomes in idx."""
	names = list(frames)
	out = {}
	for i, fa in enumerate(names):
		for fb in names[i + 1:]:
			col = lambda f, k: [frames[f][g][k] for g in idx]
			out[f"{fa}-{fb}"] = {k: _pair_stats(col(fa, k), col(fb, k)) for k in METRICS}
	return out


def _fmt(v, w=10):
	return f"{v:>{w}.4g}" if isinstance(v, (int, float)) else f"{'—':>{w}}"


def _print_subset(label, stats):
	print(f"\n  [{label}]")
	print(f"  {'pair':5} {'metric':26} {'n':>3} {'mean(D)':>10} {'SD_g(D)':>10} "
	      f"{'±SE':>10} {'1.48MAD':>10} {'noise/spr':>10}")
	for pair, cols in stats.items():
		for k, s in cols.items():
			print(f"  {pair:5} {k:26} {s['n']:>3} {_fmt(s.get('mean_D'))} {_fmt(s.get('sd_D'))} "
			      f"{_fmt(s.get('se_sd_D'))} {_fmt(s.get('mad_spread'))} {_fmt(s.get('noise_over_spread'))}")


def _print_report(doc):
	m = doc["meta"]
	print("=" * 100)
	print(f"  CTRL-21 CRN COMPARISON-NOISE PROBE — {m['tag']} | stage {m['stage_name']} "
	      f"gen {m['generation']} | {m['genomes_scored']}/{m['population_size']} genomes")
	print(f"  {m['pools_per_frame']} pools x {m['episodes']} ep per frame | frames "
	      + ", ".join(f"{k}={v}" for k, v in m["frame_seeds"].items())
	      + (" | SMOKE — NOT A RESULT" if m["smoke"] else ""))
	print("  SD_g(D) = the within-frame comparison floor; noise/spr >= 1 ⇒ column is pure noise")
	print("=" * 100)
	_print_subset(f"ALL n={len(doc['subsets']['all']['genomes'])}", doc["subsets"]["all"]["pairs"])
	fly = doc["subsets"]["flying"]
	_print_subset(f"FLYING stable>={m['flying_stable']} n={len(fly['genomes'])}", fly["pairs"])


def _write(doc, path):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	tmp = path + ".tmp"
	with open(tmp, "w") as f:
		json.dump(doc, f, indent=1)
	os.replace(tmp, path)


def _tag(a):
	return a.tag or os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))


def _meta(a, reg, ck_meta, frame_seeds, pools, verified):
	return {"tag": _tag(a), "ckpt": a.ckpt, "run_args_file": a.run_args_file,
	        **ck_meta, "genomes_scored": None, "train_seed": reg.train_seed,
	        "frame_seeds": frame_seeds, "frame_pools": pools, "pools_per_frame": a.pools,
	        "episodes": a.episodes, "steps": reg.ec.steps_per_episode,
	        "flying_stable": a.flying_stable, "run_out_verified": verified,
	        "smoke": a.pools != 5 or bool(a.limit_genomes) or not verified,
	        "estimator": "SD over genomes of D_g = X_g(F1) - X_g(F2) = sqrt(2)*sigma = the "
	                     "within-frame comparison floor; frame A = run search pools "
	                     "(selection-exposed), B/C fresh"}


def _doc(meta, frames, flying):
	allidx = list(range(len(next(iter(frames.values())))))
	meta["genomes_scored"] = len(allidx)
	return {"meta": meta, "per_genome": frames,
	        "subsets": {"all": {"genomes": allidx, "pairs": _subset_stats(frames, allidx)},
	                    "flying": {"genomes": flying, "pairs": _subset_stats(frames, flying)}}}


def _setup(a):
	"""Regime, population, frames → everything the scoring loop needs."""
	reg = _load_regime(a.run_args_file)
	if a.episodes is None:
		a.episodes = getattr(reg.args, "memory_eval_episodes", None) or reg.args.eval_episodes
	seeds = _frame_seeds(a, reg)
	pools = {FRAME_NAMES[i]: _pools(s, a.pools) for i, s in enumerate(seeds)}
	_check_disjoint(pools)
	verified = _verify_against_run(a, pools["A"], a.episodes) if a.run_out else False
	spec, genomes, ck_meta = _load_population(a.ckpt, a.limit_genomes)
	print(f"  loaded {ck_meta['population_size']} genomes from {a.ckpt} "
	      f"(stage {ck_meta['stage_name']}, gen {ck_meta['generation']}); scoring {len(genomes)}", flush=True)
	frame_seeds = {FRAME_NAMES[i]: s for i, s in enumerate(seeds)}
	return reg, spec, genomes, _meta(a, reg, ck_meta, frame_seeds, pools, verified)


def _build_parser():
	ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	ap.add_argument("--ckpt", required=True, help="banked stage checkpoint (its final POPULATION is scored)")
	ap.add_argument("--run-args-file", required=True, help="the run's exact phased_ga argv ('#' lines ignored)")
	ap.add_argument("--run-out", default=None, help="the run's .out — verifies frame A = the search frame")
	ap.add_argument("--tag", default=None, help="default: the checkpoint's directory name")
	ap.add_argument("--frames", type=int, default=2, choices=(2, 3))
	ap.add_argument("--frame-seeds", type=int, nargs="+", default=None,
	                help="evaluator seeds per frame (default: train seed, then fixed fresh offsets)")
	ap.add_argument("--pools", type=int, default=5, help="CRN pools per frame (5 = the run; else SMOKE)")
	ap.add_argument("--episodes", type=int, default=None, help="default: the run's --memory-eval-episodes")
	ap.add_argument("--limit-genomes", type=int, default=0, help="score only the first N (SMOKE)")
	ap.add_argument("--flying-stable", type=float, default=0.70)
	ap.add_argument("--smoke-ok", action="store_true", help="tolerate a frame-A/.out mismatch (smoke only)")
	ap.add_argument("--out", default=None, help="default experiments/crn_noise_probe/PROBE_<tag>.json")
	return ap


def main():
	a = _build_parser().parse_args()
	reg, spec, genomes, meta = _setup(a)
	a.out = a.out or f"experiments/crn_noise_probe/PROBE_{meta['tag']}.json"
	thresholds = _thresholds(reg, spec)
	frames = {}
	for name, seed in meta["frame_seeds"].items():
		frames[name] = _score_frame(reg, spec, genomes, thresholds, seed, a)
	doc = _doc(meta, frames, _flying(frames, a.flying_stable))
	_write(doc, a.out)
	_print_report(doc)
	print(f"# wrote {a.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
