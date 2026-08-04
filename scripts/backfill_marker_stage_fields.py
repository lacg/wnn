#!/usr/bin/env python3
"""Repair the mislabelled stage fields in existing controller run markers.

THE BUG (fixed in controller_arm_lib.sh, 04/08/2026). Markers were written with

    held_n = <RESULT line 1>
    held_m = <RESULT line 2>

which is only correct when a run emits ONE held-out RESULT per stage. With
`--report-seeds N` each stage emits N of them, so at N=5 lines 1 and 2 are both
NEURONS (report seeds 1 and 2) and `held_memory` was never the memory stage at all.
Every marker written by that helper carries the wrong value, and those values were
quoted as MEMORY results.

This re-derives both fields from the run's .out by anchoring on the STAGE headers,
and additionally captures the MULTI-SEED aggregate lines — the authoritative numbers
when report seeds are used, because a single RESULT line is one draw and carries no
variance.

Markers whose .out is missing are left ALONE and reported: a marker is a claim that a
run finished, and rewriting one without its evidence would be worse than the bug.

Usage:
  python3 scripts/backfill_marker_stage_fields.py            # report only
  python3 scripts/backfill_marker_stage_fields.py --apply    # rewrite in place
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# marker dir -> log dir
PAIRS = [
	("experiments/p4_markers", "logs/controller/p4"),
	("experiments/p3_markers", "logs/controller/p3"),
	("experiments/p2_markers", "logs/controller/p2"),
	("experiments/l3dfeat_markers", "logs/controller/l3dfeat"),
	("experiments/dfa1l_markers", "logs/controller/dfa1l"),
]
RESULT = "RESULT — during-search winner"
NEURONS_HDR = re.compile(r"STAGE 1 \(NEURONS\) done")
MEMORY_HDR = re.compile(r"STAGE 4 \(MEMORY\) done")


def first_result_after(lines, header):
	"""The first held-out RESULT line following a stage header — the same anchor
	controller_arm_lib.sh now uses, and correct for one report seed or many."""
	seen = False
	for ln in lines:
		if header.search(ln):
			seen = True
		elif seen and RESULT in ln:
			return " ".join(ln.split())
	return ""


def multiseed(lines, stage):
	for ln in reversed(lines):
		if f"{stage} MULTI-SEED held-out" in ln:
			return " ".join(ln.split())
	return ""


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--apply", action="store_true", help="rewrite markers in place")
	args = ap.parse_args()

	fixed = same = no_log = 0
	for mdir, ldir in PAIRS:
		md, ld = ROOT / mdir, ROOT / ldir
		if not md.is_dir():
			continue
		for mp in sorted(md.glob("*.json")):
			try:
				d = json.loads(mp.read_text())
			except Exception as e:
				print(f"  SKIP {mp.name}: unreadable ({e})")
				continue
			if "held_memory" not in d:
				continue
			lp = ld / f"{mp.stem}.out"
			if not lp.exists():
				no_log += 1
				print(f"  NO LOG {mdir}/{mp.name} — left untouched (cannot re-derive without evidence)")
				continue
			lines = lp.read_text(errors="replace").splitlines()
			new = {
				"held_neurons": first_result_after(lines, NEURONS_HDR),
				"held_memory": first_result_after(lines, MEMORY_HDR),
				"held_neurons_multiseed": multiseed(lines, "NEURONS"),
				"held_memory_multiseed": multiseed(lines, "MEMORY"),
			}
			if not new["held_memory"]:
				print(f"  NO MEMORY STAGE {mdir}/{mp.name} — left untouched (run may be truncated)")
				continue
			changed = any(d.get(k, "") != v for k, v in new.items())
			if not changed:
				same += 1
				continue
			fixed += 1
			old = d.get("held_memory", "")
			print(f"  FIX {mdir}/{mp.name}")
			print(f"      was held_memory: {old[:78]}")
			print(f"      now held_memory: {new['held_memory'][:78]}")
			if new["held_memory_multiseed"]:
				print(f"      + multiseed    : {new['held_memory_multiseed'][-58:]}")
			if args.apply:
				d.update(new)
				mp.write_text(json.dumps(d) + "\n")

	print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: {fixed} marker(s) to fix, "
	      f"{same} already correct, {no_log} without a log.")
	if not args.apply and fixed:
		print("Re-run with --apply to rewrite.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
