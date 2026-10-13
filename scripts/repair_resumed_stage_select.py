#!/usr/bin/env python
"""Re-run stage-select for a banked run whose RESUMED attempt ranked a shrunken pool.

WHY (13/10/2026). Before the resume fix in phased_ga._run_one, a run killed in stage
k and retried with --resume-from-emergency re-entered only stages >= k into
`_select_headline_stage`; the stages the killed attempt had FINISHED (GRID, NEURONS,
... — all checkpointed on disk) were dropped. SL_C_b24n256_..._s31337002_tw20 banked a
headline ranked over 3 MEMORY candidates instead of 9, and its marker carries no
GRID/NEURONS held-out rows (they were printed by the killed attempt's log only).

WHAT THIS DOES — no re-fly, no search:
  * re-runs the union stage-select over top-K of EVERY stage checkpoint
    (`phased_ga._recalc_headline`, bounded top-K loads) on the run's own val seeds,
    with the run's EXACT phased_ga flags (pass them after `--`; they define episodes,
    seeds, thresholds, scorer and the fitness identity);
  * recovers each stage's per-stage MULTI-SEED held-out row from the run's own logs
    (`<tag>.out` and every `<tag>.out.killed*` — the attempt that ran the stage
    printed it; the newest log wins per stage).

GATES (any failure REFUSES and writes nothing to the marker):
  G0  the recalc derived the same [seeds] line the run printed (train/test/val);
  G1  the selector's fitness identity (`fit = ...`) equals the run's;
  G2  every candidate the ORIGINAL pool ranked reproduces its `val` line bit-exactly
      (proves the flags rebuild the run's scorer);
  G3  when the new headline is a stage's #0 genome, its re-scored report-seed
      MULTI-SEED row equals that stage's published row from the run's logs.

WRITES (originals never touched): headline_stage_recalc, headline_holdout_recalc,
stage_select_candidates_recalc, held_stage_multiseed_recalc,
held_grid_multiseed_recalc, held_neurons_multiseed_recalc, headline_recalc{...}.
The recalc stdout is kept at logs/controller/recalc_headlines/<tag>.repair[.refused].out.

Usage (from the repo root; the box's ONE controller slot must be free):
  PYTHONPATH=src /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python \\
      scripts/repair_resumed_stage_select.py --tag <TAG> [--dry-run] -- <phased_ga flags>
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
STAGES = ("GRID", "NEURONS", "BITS", "CONNECTIONS", "MEMORY")


def _one(root: str, pattern: str) -> str | None:
	hits = sorted(glob.glob(os.path.join(root, pattern)))
	return hits[0] if hits else None


def _logs_oldest_first(out_path: str) -> list:
	"""The run's attempts in chronological order: .out.killed1, .killed2, ..., .out."""
	killed = sorted(glob.glob(out_path + ".killed*"),
	                key=lambda p: int(re.sub(r"\D", "", p.rsplit("killed", 1)[1]) or 0))
	return killed + [out_path]


def _squeeze(line: str) -> str:
	"""The marker's normalization (controller_arm_lib.sh stage_multiseed_json)."""
	return re.sub(r" +", " ", line.replace('"', "")).strip()


def stage_rows(logs: list) -> dict:
	"""{STAGE: last '<STAGE> MULTI-SEED held-out' line}, newest attempt wins."""
	rows: dict = {}
	for path in logs:
		text = open(path, errors="replace").read()
		for st in STAGES:
			hits = re.findall(rf"^.* {st} MULTI-SEED held-out.*$", text, re.M)
			if hits:
				rows[st] = _squeeze(hits[-1])
	return rows


def _seeds_line(text: str) -> str | None:
	m = re.search(r"\[seeds\].*?train=\d+ test=\d+ val=\d+", text)
	return re.sub(r"^.*?(train=)", r"\1", m.group(0)) if m else None


def _row_numbers(line: str) -> str:
	"""The metric part of a MULTI-SEED row (after the seed list), for G3."""
	return _squeeze(line.split("]):", 1)[1]) if "]):" in line else _squeeze(line)


def main() -> int:
	argv = sys.argv[1:]
	if "--" not in argv:
		raise SystemExit("pass the run's exact phased_ga flags after `--`")
	cut = argv.index("--")
	ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
	ap.add_argument("--tag", required=True)
	ap.add_argument("--root", default=ROOT,
	                help="repo root holding experiments/ and logs/ (default: this checkout)")
	ap.add_argument("--dry-run", action="store_true",
	                help="check inputs + gates G0 only; no scoring, marker untouched")
	pa = ap.parse_args(argv[:cut])
	flags = argv[cut + 1:]

	import recalc_headlines as rh
	from wnn.control.phased_ga import (build_arg_parser, episode_config_from_args,
	                                   _recalc_headline)

	tag = pa.tag
	marker_p = _one(pa.root, f"experiments/*_markers/{tag}.json")
	out_p = _one(pa.root, f"logs/controller/*/{tag}.out")
	ck = _one(pa.root, f"logs/controller/*/ckpt/{tag}")
	if not (marker_p and out_p and ck):
		raise SystemExit(f"missing input: marker={marker_p} out={out_p} ckpt={ck}")
	marker = json.load(open(marker_p))
	if "headline_recalc" in marker:
		raise SystemExit(f"{tag}: marker already carries headline_recalc — refusing to overwrite")
	logs = _logs_oldest_first(out_p)
	rows = stage_rows(logs)
	run_text = open(out_p, errors="replace").read()
	print(f"[repair] {tag}\n  logs: {[os.path.basename(p) for p in logs]}\n"
	      f"  per-stage rows recovered: {sorted(rows)}")

	args = build_arg_parser().parse_args(flags)
	if os.path.abspath(os.path.join(pa.root, args.save_stage_checkpoints or "")) != os.path.abspath(ck):
		raise SystemExit(f"--save-stage-checkpoints {args.save_stage_checkpoints!r} is not this run's "
		                 f"checkpoint dir {ck} — wrong flags?")
	args.recalc_headline = ck
	ec = episode_config_from_args(args)
	if pa.dry_run:
		print("  [dry-run] flags parse; inputs present; no scoring done")
		return 0

	buf = io.StringIO()
	t0 = time.time()
	with contextlib.redirect_stdout(buf):
		_recalc_headline(args, ec)
	text = buf.getvalue()
	secs = round(time.time() - t0, 1)
	keep_dir = os.path.join(pa.root, "logs", "controller", "recalc_headlines")
	os.makedirs(keep_dir, exist_ok=True)

	def refuse(why: str) -> int:
		with open(os.path.join(keep_dir, f"{tag}.repair.refused.out"), "w") as fh:
			fh.write(text)
		print(f"  REFUSED: {why}")
		return 2

	if _seeds_line(text) != _seeds_line(run_text):                                   # G0
		return refuse(f"seed set differs: run {_seeds_line(run_text)!r} vs recalc {_seeds_line(text)!r}")
	if rh._fit_identity(text) != rh._fit_identity(run_text):                          # G1
		return refuse(f"fitness identity differs: run {rh._fit_identity(run_text)!r} "
		              f"vs recalc {rh._fit_identity(text)!r}")
	old_val = rh._val_lines(marker.get("stage_select_candidates") or "")
	new_val = rh._val_lines(text)
	bad = [c for c in old_val if new_val.get(c) != old_val[c]]                        # G2
	if bad:
		return refuse(f"original candidates do not reproduce their val line: "
		              f"{[(c, old_val[c], new_val.get(c)) for c in bad]}")
	hs = re.search(r"\[stage-select\] HEADLINE stage=(\w+) genome=(\S+)", text)
	if not hs:
		return refuse("no HEADLINE line in the recalc output")
	winner = hs.group(2)
	ms = re.search(rf"\[report-seeds\] HEADLINE-{re.escape(winner)} MULTI-SEED held-out.*", text)
	hh = re.search(r"\[stage-select\] HEADLINE held-out: .*", text)
	holdout = _squeeze(ms.group(0)) if ms else (_squeeze(hh.group(0)) if hh else None)
	if holdout is None:
		return refuse("selected genome has no held-out line")
	if winner.endswith("#0") and hs.group(1) in rows and ms:                          # G3
		want, got = _row_numbers(rows[hs.group(1)]), _row_numbers(ms.group(0))
		if want.split(" stable_fail=")[0] != got.split(" stable_fail=")[0]:
			return refuse(f"{winner} re-scored {got!r} != its published stage row {want!r}")

	table = re.search(r"STAGE TABLE.*?(?=\n\s*\[stage-select\] HEADLINE stage=)", text, re.S)
	cands = " ; ".join(_squeeze(l) for l in (table.group(0).split("\n") if table else []) if " val " in l)
	n_c = re.search(r"union rank over (\d+) candidates", text)
	old_hs = re.search(r"stage=(\w+) genome=(\S+)", marker.get("headline_stage") or "")
	prev = f"{old_hs.group(1)}/{old_hs.group(2)}" if old_hs else "?"
	new = f"{hs.group(1)}/{winner}"
	marker["headline_stage_recalc"] = _squeeze(hs.group(0))
	marker["headline_holdout_recalc"] = holdout
	marker["stage_select_candidates_recalc"] = cands
	marker["held_stage_multiseed_recalc"] = rows
	marker["held_grid_multiseed_recalc"] = rows.get("GRID", "")
	marker["held_neurons_multiseed_recalc"] = rows.get("NEURONS") or rows.get("CONNECTIONS", "")
	marker["headline_recalc"] = dict(
		previous=prev, new=new, changed=(prev != new),
		reason="resume-pool repair: the resumed attempt ranked only the stages it re-ran",
		candidates_before=len(old_val), candidates_after=int(n_c.group(1)) if n_c else len(new_val),
		val_lines_reproduced=sorted(old_val), stage_rows_from=[os.path.basename(p) for p in logs],
		at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), secs=secs)
	tmp = marker_p + ".tmp"
	with open(tmp, "w") as fh:
		json.dump(marker, fh)
	os.replace(tmp, marker_p)
	with open(os.path.join(keep_dir, f"{tag}.repair.out"), "w") as fh:
		fh.write(text)
	print(f"  headline {prev} -> {new}{'  (CHANGED)' if prev != new else ''} over "
	      f"{marker['headline_recalc']['candidates_after']} candidates; held-out: {holdout}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
