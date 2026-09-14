#!/usr/bin/env python
"""Re-score the FIRST report seed of every arch-only stage row from the run's own
checkpoint — the 15/07→14/09/2026 threshold misalignment (phased_ga._holdout_report).

WHAT WAS WRONG. For a stage whose winner carried no cells (GRID always; NEURONS /
BITS / CONNECTIONS when not Lamarckian), `_holdout_report` trained the cells on the
train seed under TRAIN-seed thresholds and scored them with an evaluator built on
REPORT-seed thresholds — two address functions for one memory. Only the FIRST
report seed was hit: the write-back left the genome carrying cells, so seeds 2..5
took the aligned score-only path. Fixed at the source on 14/09/2026; this script
repairs the banked rows so the archive is on one rule.

WHAT THIS DOES, per marker (read-only on the run; writes ONLY to the marker json):
  1. rebuild the run's flags from its .out header + tag (steps/tilt/levels, seeds,
     disturbance, airframe, translation) — never a hand-copied EpisodeConfig;
  2. load the stage winner (pop[0]) from `ckpt/<tag>/stageN_<name>.yaml.gz`, the
     same loader --recalc-headline uses; it carries the cells the held-out wrote back;
  3. CLASSIFY by reproduction, bit-exact against the .out's per-seed RESULT line:
       scored under REPORT-seed thresholds == logged  → the row WAS misaligned → repair
       scored under TRAIN-seed  thresholds == logged  → already aligned (Lamarckian) → skip
       neither                                        → REFUSE (recipe/ckpt mismatch)
  4. score seed 1 under TRAIN-seed thresholds (the aligned path) and recompute the
     stage's multi-seed mean±SD from the aligned seed 1 + the logged seeds 2..N;
  5. write `held_<stage>_multiseed_aligned` (same string format the run prints),
     `headline_holdout_aligned` when the headline IS that stage genome, and a
     `holdout_alignment` block with logged vs aligned per-seed values and the status.
     The original fields are never touched.

Nothing is trained. GPU scorer, ~2 scorings per stage row.

Usage:
  PYTHONPATH=src/wnn python scripts/rescore_first_report_seed.py [--only TAG ...]
      [--stages GRID,NEURONS,...] [--dry-run] [--limit N] [--headline-first]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import re
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_FILES = {"GRID": "stage0_grid", "NEURONS": "stage1_neurons", "BITS": "stage2_bits",
               "CONNECTIONS": "stage3_connections", "MEMORY": "stage4_memory"}
TOL_STABLE, TOL_ERR = 0.051, 0.0051   # the .out prints stable to 0.1 pp and err to 0.01°


# ---------------------------------------------------------------- .out parsing
def _find_out(tag: str) -> str | None:
	hits = glob.glob(os.path.join(ROOT, "logs", "controller", "*", f"{tag}.out"))
	return hits[0] if hits else None


def _find_ckpt_dir(tag: str) -> str | None:
	hits = glob.glob(os.path.join(ROOT, "logs", "controller", "*", "ckpt", tag))
	return hits[0] if hits else None


def parse_out(path: str) -> dict:
	"""Header facts + per-stage per-report-seed RESULT rows."""
	L = open(path, errors="replace").read().split("\n")
	f: dict = dict(per_seed={}, report_seeds=None)
	for i, l in enumerate(L[:40]):
		m = re.search(r"\[W2\] disturbance=(\w+)", l)
		if m:
			f["disturbance"] = m.group(1)
		m = re.search(r"eval_episodes=(\d+) steps=(\d+) tilt=([\d.]+)° levels=(\d+)", l)
		if m:
			f["eval_episodes"], f["steps"], f["tilt"], f["levels"] = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))
		m = re.search(r"\[seeds\].*base=(\d+) .*train=(\d+) test=(\d+) val=(\d+)", l)
		if m:
			f["base"], f["train"], f["test"], f["val"] = (int(x) for x in m.groups())
	for i, l in enumerate(L):
		m = re.search(r"HELD-OUT REPORT \[(\w+)\] \(report-only\).*FRESH seed (\d+)", l)
		if not m:
			m2 = re.search(r"\[report-seeds\] \w+ MULTI-SEED held-out \((\d+) seeds \[([\d, ]+)\]\)", l)
			if m2 and f["report_seeds"] is None:
				f["report_seeds"] = [int(x) for x in m2.group(2).split(",")]
			continue
		stage, seed = m.group(1), int(m.group(2))
		for j in range(i, min(i + 8, len(L))):
			r = re.search(r"RESULT — during-search winner \(held-out\): *stable=([\d.]+)%\s+err=([\d.]+)°\s+steady=([\d.]+)°(.*)", L[j])
			if r:
				rest = r.group(4)
				row = dict(stable=float(r.group(1)), err=float(r.group(2)), steady=float(r.group(3)))
				for k in ("jerk", "mono_viol", "alt", "pos", "effort", "reward"):
					mm = re.search(rf"{k}=(-?[\d.]+)", rest)
					if mm:
						row[k] = float(mm.group(1))
				f["per_seed"].setdefault(stage, {})[seed] = row
				break
	return f


# ---------------------------------------------------------------- recipe rebuild
def build_run_args(tag: str, marker: dict, facts: dict):
	"""The run's parsed flags, rebuilt from what the run itself recorded. Anything
	this cannot recover is left at the parser default; the bit-exact reproduction
	gate in classify() refuses the marker if that guess was wrong."""
	from wnn.control.phased_ga import build_arg_parser
	argv = ["--steps", str(facts["steps"]), "--tilt", str(facts["tilt"]), "--levels", str(facts["levels"]),
	        "--disturbance", facts["disturbance"],
	        "--base-seed", str(facts["base"]), "--train-seed", str(facts["train"]),
	        "--test-seed", str(facts["test"]), "--val-seed", str(facts["val"]),
	        "--report-seeds", *[str(s) for s in facts["report_seeds"]],
	        "--num-eval-folds", "5", "--holdout-pop-sample", "8", "--runs", "1"]
	af = re.search(r"(cf21_brushless|cf2x_urdf)", tag)
	if af:
		argv += ["--airframe", af.group(1)]
	elif "_b18n32_" in tag or tag.startswith("AW_"):
		argv += ["--airframe", "cf21_brushless"]   # the alt-weight sweep's airframe (verified by the gate)
	if "afcal" in tag:
		argv += ["--calib-airframe"]
	if "alt=" in (marker.get("headline_holdout") or ""):
		argv += ["--translation"]
	sf = re.search(r"stable_fail=\d+/(\d+)", marker.get("held_grid_multiseed") or marker.get("headline_holdout") or "")
	rep_eps = (int(sf.group(1)) // len(facts["report_seeds"])) if sf else 100
	argv += ["--report-episodes", str(rep_eps), "--eval-episodes", str(facts.get("eval_episodes", 100))]
	return build_arg_parser().parse_args(argv)


# ---------------------------------------------------------------- scoring
def load_stage_genome(entries: list, stage: str):
	"""(spec, pop[0]) of one stage from the entries `_stage_entries_from_checkpoints`
	returned — loaded ONCE per marker (the stage files are ~0.4 GB each)."""
	for label, spec, res in entries:
		if label.upper() == stage:
			pop = list(getattr(res, "final_population", None) or [res.best_genome])
			return spec, pop[0]
	return None, None


def score_seed(args, ec, spec, genome, report_seed: int, train_seed: int, thresholds_on: str):
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control.phased_ga import _report_thresholds, _rg_config
	thr = _report_thresholds(args, ec, spec, report_seed, train_seed, use_score=(thresholds_on == "train"))
	rep_eps = args.report_episodes or args.eval_episodes
	ev = ControllerEvaluator(spec, num_eval_episodes=rep_eps, seed=report_seed, episode_config=ec,
	                         thresholds=thr, rg_config=_rg_config(args, ec, report_seed),
	                         max_train_workers=args.train_workers,
	                         num_eval_folds=getattr(args, "num_eval_folds", 5))
	m = ev.score_genomes([genome])[0]
	row = dict(stable=round(m.acc * 100.0, 4), err=round(m.mean_attitude_error_deg, 4),
	           steady=round(m.mean_steady_error_deg, 4) if m.mean_steady_error_deg is not None else None)
	for k, attr in (("alt", "mean_altitude_error_m"), ("pos", "mean_position_error_m"),
	                ("jerk", "motor_jerk_mean"), ("effort", "mean_effort")):
		v = getattr(m, attr, None)
		if v is not None:
			row[k] = round(float(v), 4)
	mv = getattr(m, "mono_violations_total", None)
	if mv is not None:
		row["mono_viol"] = float(mv)
	return row


def _matches(a: dict, b: dict) -> bool:
	return abs(a["stable"] - b["stable"]) <= TOL_STABLE and abs(a["err"] - b["err"]) <= TOL_ERR


# ---------------------------------------------------------------- aggregation
def _fmt_ms(vals: list[float], nd: int) -> str:
	mean = statistics.mean(vals)
	sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
	return f"{mean:.{nd}f}±{sd:.{nd}f}"


def multiseed_string(stage: str, rows: list[dict], seeds: list[int], n_eps: int) -> str:
	"""The `[report-seeds] STAGE MULTI-SEED held-out (...)` line, rebuilt from per-seed
	rows — the same fields, same order, same precision as the run prints."""
	parts = [f" [report-seeds] {stage} MULTI-SEED held-out ({len(seeds)} seeds [{', '.join(str(s) for s in seeds)}]):"]
	parts.append(f"stable={_fmt_ms([r['stable'] for r in rows], 1)}%")
	parts.append(f"err={_fmt_ms([r['err'] for r in rows], 2)}°")
	parts.append(f"steady={_fmt_ms([r['steady'] for r in rows], 2)}°")
	for k, nd, unit in (("jerk", 4, ""), ("mono_viol", 0, ""), ("alt", 3, "m"), ("pos", 3, "m"), ("effort", 3, "")):
		if all(k in r for r in rows):
			parts.append(f"{k}={_fmt_ms([r[k] for r in rows], nd)}{unit}")
	fails = sum(round((100.0 - r["stable"]) / 100.0 * n_eps) for r in rows)
	parts.append(f"stable_fail={fails}/{n_eps * len(rows)}")
	return "  ".join(parts)


def headline_string(row: dict) -> str:
	s = f" [stage-select] HEADLINE held-out: stable={row['stable']:.1f}% err={row['err']:.2f}°  steady={row['steady']:.2f}°"
	for k, nd, unit in (("jerk", 4, ""), ("mono_viol", 0, ""), ("alt", 3, "m"), ("pos", 3, "m"), ("effort", 3, "")):
		if k in row:
			s += f"  {k}={row[k]:.{nd}f}{unit}"
	return s


def headline_row_from_multiseed(rows: list[dict]) -> dict:
	out = {}
	for k in rows[0]:
		vals = [r[k] for r in rows if r.get(k) is not None]
		out[k] = statistics.mean(vals) if vals else None
	return out


# ---------------------------------------------------------------- per marker
def process_marker(path: str, stages: list[str], dry_run: bool, log) -> str:
	from wnn.control.phased_ga import episode_config_from_args
	marker = json.load(open(path))
	tag = marker.get("tag") or os.path.basename(path)[:-5]
	out = _find_out(tag)
	ck = _find_ckpt_dir(tag)
	if not out or not ck:
		return f"skip: {'no .out' if not out else 'no checkpoint dir'}"
	facts = parse_out(out)
	if not facts.get("report_seeds") or "train" not in facts or "steps" not in facts:
		return "skip: .out header incomplete (no report seeds / seed set / recipe line)"
	hs = re.search(r"stage=(\w+) genome=(\S+)", marker.get("headline_stage") or "")
	head_stage, head_genome = (hs.group(1), hs.group(2)) if hs else ("?", "?")
	try:
		args = build_run_args(tag, marker, facts)
		ec = episode_config_from_args(args)
	except SystemExit as e:
		return f"skip: recipe rebuild refused ({e})"
	align = marker.get("holdout_alignment") or {}
	seeds = facts["report_seeds"]
	s1 = seeds[0]
	n_eps = args.report_episodes
	changed = False
	entries = None
	for stage in stages:
		per = facts["per_seed"].get(stage)
		if not per or s1 not in per:
			continue
		if stage in align and align[stage].get("status", "").startswith(("rescored", "already_aligned")):
			continue
		if entries is None:
			from wnn.control.phased_ga import _stage_entries_from_checkpoints
			entries = _stage_entries_from_checkpoints(ck)
		spec, genome = load_stage_genome(entries, stage)
		if genome is None:
			align[stage] = dict(seed=s1, status="skip: no stage checkpoint")
			continue
		if getattr(genome, "cells", None) is None:
			align[stage] = dict(seed=s1, status="skip: checkpoint genome carries no cells")
			continue
		logged = per[s1]
		t0 = time.time()
		# Classify by reproduction. The expected case goes first so the common
		# outcome costs one scoring: arch-only stages were misaligned (report-seed
		# thresholds reproduce the log); a Lamarckian CONNECTIONS row was aligned.
		first = "train" if stage == "CONNECTIONS" else "report"
		got = {first: score_seed(args, ec, spec, genome, s1, facts["train"], first)}
		if not _matches(got[first], logged):
			other = "report" if first == "train" else "train"
			got[other] = score_seed(args, ec, spec, genome, s1, facts["train"], other)
		rep, tr = got.get("report"), got.get("train")
		if rep is not None and _matches(rep, logged):
			aligned = tr if tr is not None else score_seed(args, ec, spec, genome, s1, facts["train"], "train")
			rows = [aligned] + [per[s] for s in seeds[1:] if s in per]
			ms = multiseed_string(stage, rows, seeds, n_eps)
			align[stage] = dict(seed=s1, status="rescored", logged=logged, aligned=aligned,
			                    multiseed_aligned=ms, secs=round(time.time() - t0, 1))
			marker[f"held_{stage.lower()}_multiseed_aligned"] = ms
			if stage == head_stage:
				hrow = headline_row_from_multiseed(rows)
				marker["headline_holdout_aligned"] = headline_string(hrow)
				align[stage]["headline_genome"] = head_genome
			changed = True
			log(f"  {stage:11s} seed {s1}: MISALIGNED {logged['stable']:.1f}%/{logged['err']:.2f}° → aligned "
			    f"{aligned['stable']:.1f}%/{aligned['err']:.2f}°   row: {ms.split(':',1)[1].strip()[:60]}")
		else:
			if tr is not None and _matches(tr, logged):
				align[stage] = dict(seed=s1, status="already_aligned", logged=logged, secs=round(time.time() - t0, 1))
				log(f"  {stage:11s} seed {s1}: already aligned ({logged['stable']:.1f}%/{logged['err']:.2f}°)")
			else:
				align[stage] = dict(seed=s1, status="refused: neither threshold set reproduces the logged row",
				                    logged=logged, under_report=rep, under_train=tr)
				log(f"  {stage:11s} seed {s1}: REFUSED — logged {logged['stable']:.1f}%/{logged['err']:.2f}°, "
				    f"report-thr {rep['stable']:.1f}%/{rep['err']:.2f}°, train-thr {tr['stable']:.1f}%/{tr['err']:.2f}°")
	if align:
		align["rescored_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
		marker["holdout_alignment"] = align
		if not dry_run:
			tmp = path + ".tmp"
			with open(tmp, "w") as fh:
				json.dump(marker, fh)
			os.replace(tmp, path)
	return "updated" if changed else "no change"


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
	ap.add_argument("--only", nargs="*", default=None, help="marker tags to process (default: all)")
	ap.add_argument("--stages", default="GRID,NEURONS,BITS,CONNECTIONS",
	                help="stage rows to check (MEMORY is score-only and never misaligned)")
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--limit", type=int, default=None)
	ap.add_argument("--headline-first", action="store_true",
	                help="process markers whose HEADLINE is an arch-only stage before the rest")
	ap.add_argument("--markers-glob", default=os.path.join(ROOT, "experiments", "*_markers", "*.json"))
	pa = ap.parse_args()
	stages = [s.strip().upper() for s in pa.stages.split(",") if s.strip()]
	paths = [p for p in sorted(glob.glob(pa.markers_glob)) if "void" not in p]
	if pa.only:
		want = set(pa.only)
		paths = [p for p in paths if os.path.basename(p)[:-5] in want]
	if pa.headline_first:
		def _key(p):
			try:
				hs = json.load(open(p)).get("headline_stage") or ""
			except Exception:
				return 1
			return 0 if re.search(r"stage=(GRID|NEURONS|BITS|CONNECTIONS)", hs) else 1
		paths.sort(key=_key)
	if pa.limit:
		paths = paths[:pa.limit]
	def log(s):
		print(s, flush=True)
	t0 = time.time()
	tally = {}
	for i, p in enumerate(paths, 1):
		tag = os.path.basename(p)[:-5]
		log(f"[{i}/{len(paths)}] {tag}")
		try:
			st = process_marker(p, stages, pa.dry_run, log)
		except Exception as e:  # one bad marker must not stop the sweep; it is listed
			st = f"error: {type(e).__name__}: {e}"
		tally[st.split(':')[0]] = tally.get(st.split(':')[0], 0) + 1
		log(f"  → {st}   ({(time.time() - t0) / 60:.1f} min elapsed)")
	log(f"done: {tally}")


if __name__ == "__main__":
	sys.exit(main())
