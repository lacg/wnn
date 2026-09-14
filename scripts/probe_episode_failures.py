#!/usr/bin/env python
"""Which EPISODES does a saved controller fail on one report seed — and what do they share?

Motivation (14/09/2026): the s31337003 `_bd` headline (GRID#0) flew 500/500 stable
on the val seeds and 464/500 on the report seeds (20 of the 36 failures on
99990101). The held-out scorer reduced its per-episode buffer to per-genome means,
so the question "what do the failed episodes have in common" was unanswerable.
This probe re-scores ONE genome on ONE report seed through the SAME path the
held-out used (`_report_thresholds` → `ControllerEvaluator.score_genomes`, the GPU
scorer), with the evaluator's per-episode sink armed, then joins the per-episode
flags against the episode's initial conditions and prints the failures next to
the passes on every variable.

Nothing is trained, nothing is written. The recipe flags are passed verbatim
after `--` so the plant/disturbance/features are the run's own
(`episode_config_from_args`), never a hand copy.

Usage:
  PYTHONPATH=<dir with the per-episode-export ram_controller .so>:src/wnn \\
  python scripts/probe_episode_failures.py \\
      --ckpt-dir logs/controller/sweep_ladder/ckpt/<tag> --stage stage0_grid \\
      --report-seed 99990101 --expect-stable 80.0 --expect-err 4.24 \\
      -- <the run's phased_ga flags, minus --save-winner/--save-stage-checkpoints>
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys

import numpy as np


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
	if "--" not in argv:
		raise SystemExit("pass the run's phased_ga flags after `--`")
	i = argv.index("--")
	return argv[:i], argv[i + 1:]


def _probe_args(argv: list[str]) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
	ap.add_argument("--ckpt-dir", required=True, help="the run's --save-stage-checkpoints dir")
	ap.add_argument("--stage", default="stage0_grid",
	                help="checkpoint basename to load (stage0_grid / stage3_connections / stage4_memory)")
	ap.add_argument("--genome-index", type=int, default=0,
	                help="index into the stage's final_population (0 = the stage winner = its #0 candidate)")
	ap.add_argument("--report-seed", type=int, required=True)
	ap.add_argument("--expect-stable", type=float, default=None,
	                help="the run's logged stable%% on this seed — the probe REFUSES to interpret "
	                     "episodes unless its aggregate reproduces it")
	ap.add_argument("--expect-err", type=float, default=None, help="likewise, the logged err (deg)")
	ap.add_argument("--thresholds", choices=("train", "report"), default="train",
	                help="which seed the thermometer thresholds are fit on. `train` = the aligned "
	                     "score-only path (cells were written under train-seed thresholds). `report` "
	                     "reproduces what _holdout_report does on the FIRST report seed of an arch-only "
	                     "stage (14/09/2026 finding): it trains under train-seed thresholds, then scores "
	                     "under report-seed thresholds — the documented address misalignment.")
	return ap.parse_args(argv)


def _load_stage_genome(ckpt_dir: str, stage: str, index: int):
	"""The stage's (spec, genome) exactly as stage-select saw it — via the same
	loader `--recalc-headline` uses."""
	from wnn.control.phased_ga import _stage_entries_from_checkpoints
	entries = _stage_entries_from_checkpoints(ckpt_dir)
	want = stage.split("_", 1)[1].upper() if "_" in stage else stage.upper()
	for label, spec, res in entries:
		if label.upper() == want:
			pop = list(getattr(res, "final_population", None) or [res.best_genome])
			if index >= len(pop):
				raise SystemExit(f"{stage}: population has {len(pop)} genomes, index {index} out of range")
			return label, spec, pop[index]
	raise SystemExit(f"{stage}: not among {[l for l, _, _ in entries]} in {ckpt_dir}")


def _replay_ics(seed: int, n: int, ec) -> list[dict]:
	"""Re-draw the per-episode (roll, pitch, yaw, ωx, ωy, ωz) with the SAME RNG chain
	`sample_ics_flat` uses, so each episode's initial condition is known in the
	angles it was drawn in (the quaternion the scorer receives is derived from
	these). Read-only mirror of the draw ORDER; the scorer still gets its own q0."""
	rng = np.random.default_rng(seed)
	out = []
	for _ in range(n):
		ep = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		r = float(ep.uniform(-ec.max_initial_tilt_rad, ec.max_initial_tilt_rad))
		p = float(ep.uniform(-ec.max_initial_tilt_rad, ec.max_initial_tilt_rad))
		y = float(ep.uniform(-ec.max_initial_yaw_rad, ec.max_initial_yaw_rad))
		ox = float(ep.uniform(-ec.max_initial_body_rate, ec.max_initial_body_rate))
		oy = float(ep.uniform(-ec.max_initial_body_rate, ec.max_initial_body_rate))
		oz = float(ep.uniform(-ec.max_initial_yaw_rate, ec.max_initial_yaw_rate))
		out.append(dict(roll_deg=math.degrees(r), pitch_deg=math.degrees(p), yaw_deg=math.degrees(y),
		                tilt_deg=math.degrees(math.hypot(r, p)),
		                wx=ox, wy=oy, wz=oz, w_rp=math.hypot(ox, oy)))
	return out


def _score_with_sink(args, ec, spec, genome, report_seed: int, train_seed: int,
                     thresholds_on: str = "train"):
	"""The held-out path of `_holdout_report` for ONE genome, sink armed."""
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control.phased_ga import _report_thresholds, _rg_config
	use_score = getattr(genome, "cells", None) is not None or getattr(ec, "geometry", None) is not None
	if not use_score:
		raise SystemExit("genome carries no trained cells — the held-out would retrain on the "
		                 "train seed first; this probe only scores what was scored")
	thresholds = _report_thresholds(args, ec, spec, report_seed, train_seed,
	                                use_score=(thresholds_on == "train"))
	rep_eps = getattr(args, "report_episodes", None) or args.eval_episodes
	ev = ControllerEvaluator(spec, num_eval_episodes=rep_eps, seed=report_seed, episode_config=ec,
	                         thresholds=thresholds, rg_config=_rg_config(args, ec, report_seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=getattr(args, "num_eval_folds", 5))
	ev.per_episode_sink = []
	metrics = ev.score_genomes([genome])
	if not ev.per_episode_sink:
		raise SystemExit("the scorer returned no per-episode rows — is the per-episode-export "
		                 "ram_controller on PYTHONPATH, and was the GPU scorer used?")
	return metrics[0], ev.per_episode_sink[0], rep_eps


def _episode_rows(sink: dict, ics: list[dict]) -> list[dict]:
	rows = []
	for r in sink["episodes"]:
		gi, ep, stable, diverged, err, steady, alt, pos, effort, jerk, steps = r
		ep = int(ep)
		d = dict(ep=ep, stable=bool(stable), diverged=bool(diverged),
		         err_deg=math.degrees(err), steady_deg=math.degrees(steady),
		         alt_m=alt, pos_m=pos, effort=effort, jerk=jerk, steps=int(steps))
		d.update(ics[ep])
		for k in ("z0", "vz0", "mass", "coll", "x0", "y0"):
			v = sink.get(k)
			d[k] = None if v is None else float(v[ep])
		rows.append(d)
	return rows


def _describe(rows: list[dict]) -> None:
	fails = [r for r in rows if not r["stable"]]
	passes = [r for r in rows if r["stable"]]
	hard = sum(1 for r in fails if r["diverged"])
	print(f"\n  episodes {len(rows)}  stable {len(passes)}  failed {len(fails)}  "
	      f"(HARD/diverged {hard}, SOFT/>5° {len(fails) - hard})")
	if not fails:
		return
	print("\n  FAILED EPISODES (each row = one episode; ICs are the drawn values):")
	hdr = "   ep  kind   err°   steady°  alt m   tilt°  roll°  pitch°  yaw°   |w_rp|  wz     z0     vz0    mass   coll"
	print(hdr)
	for r in sorted(fails, key=lambda r: -r["err_deg"]):
		z0 = "  —  " if r["z0"] is None else f"{r['z0']:+.3f}"
		vz = "  —  " if r["vz0"] is None else f"{r['vz0']:+.3f}"
		ms = "  —  " if r["mass"] is None else f"{r['mass']:.3f}"
		cl = "  —  " if r["coll"] is None else f"{r['coll']:.3f}"
		print(f"  {r['ep']:3d}  {'HARD' if r['diverged'] else 'SOFT'}  {r['err_deg']:6.2f}  {r['steady_deg']:6.2f}  "
		      f"{r['alt_m']:.3f}  {r['tilt_deg']:5.2f}  {r['roll_deg']:+5.2f}  {r['pitch_deg']:+5.2f}  "
		      f"{r['yaw_deg']:+5.2f}  {r['w_rp']:.3f}  {r['wz']:+.3f}  {z0}  {vz}  {ms}  {cl}")
	print("\n  FAILED vs PASSED, per variable (mean ± SD; AUC = P(failed > passed), 0.5 = no separation):")
	keys = ["tilt_deg", "roll_deg", "pitch_deg", "yaw_deg", "w_rp", "wx", "wy", "wz",
	        "z0", "vz0", "mass", "coll", "x0", "y0"]
	for k in keys:
		fv = [r[k] for r in fails if r.get(k) is not None]
		pv = [r[k] for r in passes if r.get(k) is not None]
		if not fv or not pv:
			continue
		auc = sum(1.0 if f > p else 0.5 if f == p else 0.0 for f in fv for p in pv) / (len(fv) * len(pv))
		# absolute-value view too: a symmetric IC (±) separates on magnitude, not sign
		fa = [abs(v) for v in fv]; pa = [abs(v) for v in pv]
		auc_abs = sum(1.0 if f > p else 0.5 if f == p else 0.0 for f in fa for p in pa) / (len(fa) * len(pa))
		sd = lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0
		print(f"    {k:9s}  failed {statistics.mean(fv):+8.3f} ± {sd(fv):6.3f}   "
		      f"passed {statistics.mean(pv):+8.3f} ± {sd(pv):6.3f}   AUC {auc:.2f}   AUC|.| {auc_abs:.2f}")


def main() -> None:
	probe_argv, run_argv = _split_argv(sys.argv[1:])
	pa = _probe_args(probe_argv)
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args
	from wnn.seeds import resolve_seed_set
	args = build_arg_parser().parse_args(run_argv)
	ec = episode_config_from_args(args)
	base = args.base_seed if args.base_seed is not None else args.seed
	seeds = resolve_seed_set(base=base, run_index=0, train=args.train_seed,
	                         test=args.test_seed, val=args.val_seed)
	label, spec, genome = _load_stage_genome(pa.ckpt_dir, pa.stage, pa.genome_index)
	print(f"[probe] {label}#{pa.genome_index} from {pa.ckpt_dir} — report seed {pa.report_seed}, "
	      f"train seed {seeds.train}, cells={'yes' if getattr(genome, 'cells', None) is not None else 'no'}")
	print(f"[probe] thresholds fit on the {pa.thresholds.upper()} seed")
	m, sink, n_eps = _score_with_sink(args, ec, spec, genome, pa.report_seed, seeds.train, pa.thresholds)
	st, er = m.acc * 100.0, m.mean_attitude_error_deg
	print(f"[probe] aggregate on {pa.report_seed}: stable={st:.1f}%  err={er:.2f}°  "
	      f"steady={m.mean_steady_error_deg:.2f}°  ({n_eps} episodes)")
	if pa.expect_stable is not None and abs(st - pa.expect_stable) > 0.05:
		raise SystemExit(f"REFUSING to interpret: aggregate stable {st:.1f}% != logged {pa.expect_stable:.1f}% "
		                 "— the probe did not reproduce the held-out (recipe/seed/wheel mismatch)")
	if pa.expect_err is not None and abs(er - pa.expect_err) > 0.005:
		raise SystemExit(f"REFUSING to interpret: aggregate err {er:.2f}° != logged {pa.expect_err:.2f}°")
	if pa.expect_stable is not None:
		print("[probe] aggregate REPRODUCES the logged held-out — per-episode rows are the scorer's own")
	ics = _replay_ics(sink["seed"], n_eps, ec)
	rows = _episode_rows(sink, ics)
	_describe(rows)


if __name__ == "__main__":
	main()
