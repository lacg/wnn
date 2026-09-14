#!/usr/bin/env python
"""Re-run stage-select (`--recalc-headline`) under the FIXED held-out code for the banked
runs whose published headline was an arch-only stage's #0 genome — the 15/07→14/09/2026
first-report-seed misalignment reached those headlines AND the val scoring that chose
them (a candidate without cells had its first val seed scored misaligned too).

Per marker: rebuild the run's flags from its own .out (rescore_first_report_seed.
build_run_args), point `--recalc-headline` at its checkpoint dir, run
phased_ga._recalc_headline in-process (NOT as `-m wnn.control.phased_ga`, so the
chains' idle gate never sees it), capture the STAGE TABLE / HEADLINE block, and:

  GATE  every candidate that CARRIED cells in the checkpoint must reproduce its
        original `val` line bit-exactly (aligned then, aligned now). A mismatch means
        the rebuilt recipe is not the run's — REFUSE, write nothing.
  WRITE `headline_stage_recalc`, `headline_holdout_recalc` (the selected genome's
        aligned 5-seed mean, the `[report-seeds] HEADLINE-<key> MULTI-SEED` line or the
        stage's aligned row when the pick is unchanged), `stage_select_candidates_recalc`,
        and `headline_recalc` {previous, new, changed, at}. Originals untouched.

Usage:  PYTHONPATH=src/wnn python scripts/recalc_headlines.py [--only TAG ...] [--dry-run]
        [--wait-pid PID]   (start only after that process has exited — GPU courtesy)
        [--headline-stages GRID NEURONS BITS] [--any-genome]

Two blast radii, two passes. The DEFAULT targets are the REPORTING radius: headlines
that ARE an arch-only stage's #0 genome (their published number was misaligned). The
SELECTION radius is wider — the bias only ever DEMOTED an arch-only candidate behind a
cells-carrying one (MEMORY, Lamarckian CONNECTIONS), so a run whose headline is
MEMORY#k may have crowned the wrong genome for ANY k. Second pass (14/09/2026, Luiz):
    --headline-stages MEMORY --any-genome
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import glob
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rescore_first_report_seed as rs  # noqa: E402  (shared recipe rebuild + .out parsing)

ARCH_ONLY = ("GRID", "NEURONS", "BITS")


def targets(paths: list[str], stages: tuple[str, ...] = ARCH_ONLY,
            any_genome: bool = False) -> list[str]:
	"""Markers whose headline is a `stages` genome (#0 only unless `any_genome`),
	banked since the bug landed, not yet re-selected, with checkpoint + .out."""
	out = []
	for p in paths:
		try:
			m = json.load(open(p))
		except Exception:
			continue
		hs = re.search(r"stage=(\w+) genome=(\S+)", m.get("headline_stage") or "")
		if not hs or hs.group(1) not in stages:
			continue
		if not any_genome and hs.group(2) not in (hs.group(1), hs.group(1) + "#0"):
			continue
		done = m.get("done")
		if not isinstance(done, str) or done < "2026-07-15":
			continue
		if "headline_recalc" in m:
			continue
		tag = m.get("tag") or os.path.basename(p)[:-5]
		if rs._find_ckpt_dir(tag) and rs._find_out(tag):
			out.append(p)
	return out


def _val_lines(block: str) -> dict:
	"""{candidate: 'val 100.0%/1.14°/0.66°'} from a STAGE TABLE block or the marker's
	stage_select_candidates string."""
	return {m.group(1): m.group(2) for m in re.finditer(r"(\w+#\d+)\s+val ([\d.]+%/[\d.]+°/[\d.]+°)", block)}


def _val_seed_rows(text: str) -> dict:
	"""{(candidate, val_seed): (stable, err)} from `HELD-OUT REPORT [KEY-VALseed]` blocks."""
	L = text.split("\n")
	out = {}
	for i, l in enumerate(L):
		m = re.search(r"HELD-OUT REPORT \[(\w+#\d+|\w+)-VAL(\d+)\]", l)
		if not m:
			continue
		for j in range(i, min(i + 8, len(L))):
			r = re.search(r"RESULT — during-search winner \(held-out\): *stable=([\d.]+)%\s+err=([\d.]+)°", L[j])
			if r:
				out[(m.group(1), int(m.group(2)))] = (float(r.group(1)), float(r.group(2)))
				break
	return out


def _cells_carrying(ckpt_dir: str) -> set:
	from wnn.control.phased_ga import _stage_entries_from_checkpoints
	from wnn.control.controller_orchestrator import ControllerOrchestrator
	k = int(ControllerOrchestrator.STAGE_SELECT_TOP_K)
	have = set()
	for label, _spec, res in _stage_entries_from_checkpoints(ckpt_dir):
		pop = list(getattr(res, "final_population", None) or [res.best_genome])
		for i, g in enumerate(pop[:k]):
			if getattr(g, "cells", None) is not None:
				have.add(f"{label}#{i}" if len(pop) > 1 else label)
	return have


def _fit_identity(text: str) -> str | None:
	m = re.search(r"fit = (\S+\(.*?\))", text)
	return m.group(1) if m else None


def _apply_fitness_identity(args, ckpt_dir: str, out_path: str) -> None:
	from wnn.ram.strategies.phased.checkpoint import load_checkpoint
	from wnn.ram.strategies.phased.codecs import ControllerGenomeCodec
	stage0 = sorted(glob.glob(os.path.join(ckpt_dir, "stage*_*.yaml.gz")))[0]
	fw = (load_checkpoint(stage0, ControllerGenomeCodec()).extra or {}).get("fitness_weights") or {}
	for k in ("err_sq", "stable", "jerk", "mono", "steady", "effort", "alt", "pos"):
		if k in fw:
			setattr(args, f"fit_weight_{k}", float(fw[k]))
	if fw.get("aggregation_select"):
		args.fit_aggregation = fw["aggregation_select"]
	if fw.get("zrank_clamp") is not None:
		args.zrank_clamp = float(fw["zrank_clamp"])
	ident = _fit_identity(open(out_path, errors="replace").read())
	g = re.search(r"gate=\(([\d.]+),([\d.]+)°\)", ident or "")
	if g:
		args.gate_stable, args.gate_err = float(g.group(1)), float(g.group(2))


def recalc_one(path: str, dry_run: bool, log) -> str:
	from wnn.control.phased_ga import _recalc_headline, episode_config_from_args
	marker = json.load(open(path))
	tag = marker.get("tag") or os.path.basename(path)[:-5]
	ck, out = rs._find_ckpt_dir(tag), rs._find_out(tag)
	facts = rs.parse_out(out)
	if not facts.get("report_seeds") or "train" not in facts:
		return "skip: .out header incomplete"
	args = rs.build_run_args(tag, marker, facts)
	args.recalc_headline = ck
	# The selector ranks with the RUN's fitness identity, not the parser defaults:
	# weights + combine + clamp from the stage-0 checkpoint payload, the viability
	# gate from the .out's `fit = ...` line. The reproduction gate below also checks
	# the printed identity string, so a wrong guess cannot re-select silently.
	_apply_fitness_identity(args, ck, out)
	# Training-side flags the recalc needs ONLY for candidates the checkpoint holds
	# without cells (they are re-trained on the train seed, as the run did at val
	# time). Recovered from the marker/tag; GATE 3 below checks the outcome.
	tm = re.search(r"_(mpcof|lqi|lqr|mpc|pid)_", tag)
	args.teacher = marker.get("teacher") or (tm.group(1) if tm else "mpcof")
	args.teacher_hover = marker.get("teacher_hover") or "legacy"
	ec = episode_config_from_args(args)
	old_hs = re.search(r"stage=(\w+) genome=(\S+)", marker.get("headline_stage") or "")
	prev = f"{old_hs.group(1)}/{old_hs.group(2)}" if old_hs else "?"
	buf = io.StringIO()
	t0 = time.time()
	with contextlib.redirect_stdout(buf):
		_recalc_headline(args, ec)
	text = buf.getvalue()
	hs = re.search(r"\[stage-select\] HEADLINE stage=(\w+) genome=(\S+)", text)
	hh = re.search(r"\[stage-select\] HEADLINE held-out: (.*)", text)
	if not hs:
		return "refused: no HEADLINE line in the recalc output"
	# GATE 1: the fitness identity the selector printed must be the run's own.
	want_id, got_id = _fit_identity(open(out, errors="replace").read()), _fit_identity(text)
	if want_id != got_id:
		return f"refused: fitness identity differs — run {want_id!r} vs recalc {got_id!r}"
	# GATE 2: cells-carrying candidates must reproduce their original val line.
	new_val, old_val = _val_lines(text), _val_lines(marker.get("stage_select_candidates") or "")
	carrying = _cells_carrying(ck)
	bad = [c for c in carrying if c in old_val and c in new_val and old_val[c] != new_val[c]]
	if bad:
		return f"refused: cells-carrying candidates do not reproduce their val line: {bad[:4]}"
	# GATE 3: a candidate the checkpoint holds WITHOUT cells is re-trained here; the
	# run trained it the same way at val time and its val seeds 2..5 were already on
	# the aligned path, so those per-seed rows must reproduce bit-exactly — that is
	# the check on teacher / hover / trainer flags this script had to guess.
	orig_rows, new_rows = _val_seed_rows(open(out, errors="replace").read()), _val_seed_rows(text)
	first_val = min((sd for (_c, sd) in orig_rows), default=None)
	mismatch = [(c, sd) for (c, sd), v in orig_rows.items()
	            if c not in carrying and sd != first_val and (c, sd) in new_rows
	            and (abs(new_rows[(c, sd)][0] - v[0]) > 0.051 or abs(new_rows[(c, sd)][1] - v[1]) > 0.0051)]
	if mismatch:
		return f"refused: re-trained candidates do not reproduce their aligned val seeds: {mismatch[:3]}"
	changed = {c for c in new_val if c in old_val and old_val[c] != new_val[c]}
	winner = hs.group(2)
	ms = re.search(rf"\[report-seeds\] HEADLINE-{re.escape(winner)} MULTI-SEED held-out.*", text)
	if ms:
		holdout = ms.group(0)
	elif hh:
		holdout = " [stage-select] HEADLINE held-out: " + hh.group(1)
	else:
		return "refused: selected genome has no held-out line"
	table = re.search(r"STAGE TABLE.*?(?=\n\s*\[stage-select\] HEADLINE stage=)", text, re.S)
	cands = " ; ".join(l.strip() for l in (table.group(0).split("\n") if table else []) if " val " in l)
	rec = dict(previous=prev, new=f"{hs.group(1)}/{winner}", changed=(prev != f"{hs.group(1)}/{winner}"),
	           val_lines_changed=sorted(changed), cells_carrying=sorted(carrying),
	           at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), secs=round(time.time() - t0, 1))
	log(f"  headline {prev} → {hs.group(1)}/{winner}{'  (CHANGED)' if rec['changed'] else ''}; "
	    f"val lines re-derived for {sorted(changed) or 'none'}; held-out: {holdout.split(':', 1)[1].strip()[:70]}")
	if not dry_run:
		marker["headline_stage_recalc"] = hs.group(0)
		marker["headline_holdout_recalc"] = holdout
		marker["stage_select_candidates_recalc"] = cands
		marker["headline_recalc"] = rec
		tmp = path + ".tmp"
		with open(tmp, "w") as fh:
			json.dump(marker, fh)
		os.replace(tmp, path)
		with open(os.path.join(ROOT, "logs", "controller", "recalc_headlines", f"{tag}.recalc.out"), "w") as fh:
			fh.write(text)
	return "changed" if rec["changed"] else "unchanged"


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
	ap.add_argument("--only", nargs="*", default=None)
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--wait-pid", type=int, default=None)
	ap.add_argument("--markers-glob", default=os.path.join(ROOT, "experiments", "*_markers", "*.json"))
	ap.add_argument("--headline-stages", nargs="+", default=list(ARCH_ONLY),
	                help="headline stages to re-select (default: the arch-only three)")
	ap.add_argument("--any-genome", action="store_true",
	                help="accept headline genome #k for any k, not only #0")
	pa = ap.parse_args()
	if pa.wait_pid:
		while True:
			try:
				os.kill(pa.wait_pid, 0)
			except OSError:
				break
			time.sleep(60)
	os.makedirs(os.path.join(ROOT, "logs", "controller", "recalc_headlines"), exist_ok=True)
	paths = [p for p in sorted(glob.glob(pa.markers_glob)) if "void" not in p]
	if pa.only:
		want = set(pa.only)
		paths = [p for p in paths if os.path.basename(p)[:-5] in want]
	else:
		paths = targets(paths, tuple(pa.headline_stages), pa.any_genome)
	def log(s):
		print(s, flush=True)
	log(f"{len(paths)} markers to re-select")
	tally = {}
	t0 = time.time()
	for i, p in enumerate(paths, 1):
		log(f"[{i}/{len(paths)}] {os.path.basename(p)[:-5]}")
		try:
			st = recalc_one(p, pa.dry_run, log)
		except Exception as e:
			st = f"error: {type(e).__name__}: {e}"
		tally[st.split(':')[0]] = tally.get(st.split(':')[0], 0) + 1
		log(f"  → {st}   ({(time.time() - t0) / 60:.1f} min elapsed)")
	log(f"done: {tally}")


if __name__ == "__main__":
	main()
