#!/usr/bin/env python
"""Report a controller fitness-weight sweep run on the FULL phased_ga pipeline
(scripts/run_weight_sweep_phased.sh — grid→GA-neurons→GA-memory, all 4 gated
options). Per combo: combo #, the 4 fitness weights, latest-gen err/stable, the
per-stage HELD-OUT (NEURONS + MEMORY, from --report-seed — the honest numbers),
and wall duration. The MEMORY held-out is the final result per combo.

Each combo is its own phased_ga run under <DIR>/<COMBO>/run.out.

Usage:
  python scripts/report_weight_sweep.py --dir logs/controller/wsweep_phased_20260610
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# The 18 SWEEP_COMBOS (name → err, stable, jerk, mono), in order.
COMBOS = [
	("W1", 0.50, 0.40, 0.05, 0.05), ("W2", 0.40, 0.50, 0.05, 0.05),
	("W3", 0.60, 0.30, 0.05, 0.05), ("W4", 0.45, 0.35, 0.10, 0.10),
	("C1", 0.20, 0.40, 0.20, 0.20), ("C2", 0.20, 0.50, 0.10, 0.20),
	("C3", 0.20, 0.50, 0.20, 0.10), ("C4", 0.30, 0.30, 0.20, 0.20),
	("C5", 0.30, 0.40, 0.10, 0.20), ("C6", 0.30, 0.40, 0.20, 0.10),
	("C7", 0.30, 0.50, 0.10, 0.10), ("C8", 0.40, 0.20, 0.20, 0.20),
	("C9", 0.40, 0.30, 0.10, 0.20), ("C10", 0.40, 0.30, 0.20, 0.10),
	("C11", 0.40, 0.40, 0.10, 0.10), ("C12", 0.50, 0.20, 0.10, 0.20),
	("C13", 0.50, 0.20, 0.20, 0.10), ("C14", 0.50, 0.30, 0.10, 0.10),
]

GEN_RE = re.compile(r"Gen (\d+)/(\d+): .*?stable=([\d.]+)%, err=([\d.]+)°")
HO_HDR_RE = re.compile(r"HELD-OUT REPORT \[(\w+)\]")
HO_RESULT_RE = re.compile(r"RESULT — during-search winner \(held-out\):\s+stable=([\d.]+)%\s+err=([\d.]+)°")
WALL_RE = re.compile(r"Total wall time:\s+([\d.]+) min")
STAGE_RE = re.compile(r"STAGE \d+: (\w+)")


def parse_combo(run_out: Path) -> dict:
	d = {"last_gen": None, "stage": None, "ho": {}, "wall_min": None, "done": False}
	if not run_out.exists():
		return d
	pending_ho_stage = None
	for line in run_out.read_text(errors="ignore").splitlines():
		ms = STAGE_RE.search(line)
		if ms:
			d["stage"] = ms.group(1)
		mg = GEN_RE.search(line)
		if mg:
			d["last_gen"] = (int(mg.group(1)), int(mg.group(2)), float(mg.group(3)), float(mg.group(4)))
		mh = HO_HDR_RE.search(line)
		if mh:
			pending_ho_stage = mh.group(1)
		mr = HO_RESULT_RE.search(line)
		if mr and pending_ho_stage:
			d["ho"][pending_ho_stage] = (float(mr.group(2)), float(mr.group(1)))  # (err°, stable%)
			pending_ho_stage = None
		mw = WALL_RE.search(line)
		if mw:
			d["wall_min"] = float(mw.group(1))
			d["done"] = True
	return d


def round2_survivors(base: Path) -> list[str]:
	"""Survivor names from ROUND1_REPORT.txt (falls back to round2/ subdirs)."""
	rep = base / "ROUND1_REPORT.txt"
	if rep.exists():
		for line in rep.read_text().splitlines():
			if "SURVIVORS" in line and ":" in line:
				return [n.strip() for n in line.split(":", 1)[1].split(",") if n.strip()]
	r2 = base / "round2"
	if r2.exists():
		return sorted(p.name for p in r2.iterdir() if p.is_dir())
	return []


def report_round2(base: Path):
	"""Round-2 table: only the survivors; same columns as round 1 plus the R1
	MEMORY held-out and the R1/R2 average once the combo's round-2 run is done."""
	names = round2_survivors(base)
	if not names:
		print("  (no round-2 survivors found — is ROUND1_REPORT.txt written?)")
		return
	hdr = (f"{'#':>3} {'combo':<5} {'err':>4} {'stb':>4} {'jrk':>4} {'mno':>4} | "
	       f"{'stage':>6} {'lastgen':>8} {'lg_err':>7} {'lg_stb':>7} | "
	       f"{'N_err':>6} {'N_stb':>6} | {'M_err':>6} {'M_stb':>6} | {'dur':>6} | "
	       f"{'R1_M':>11} | {'avg(R1,R2)':>12}")
	done = sum(1 for n in names if parse_combo(base / "round2" / n / "run.out")["done"])
	print(f"  ROUND 2 — survivors on heavier config (pop50/kfold5): {base.name}   ({done}/{len(names)} done)")
	print(hdr)
	print("  " + "-" * len(hdr))
	ranked = []
	for i, name in enumerate(names, 1):
		e, s, j, m = WEIGHTS_BY_NAME[name]
		r2 = parse_combo(base / "round2" / name / "run.out")
		r1 = parse_combo(base / name / "run.out")
		r1m = r1["ho"].get("MEMORY")
		lg = r2["last_gen"]
		stage = (r2["stage"] or "-")[:6]
		lg_s = f"{lg[0]:>3}/{lg[1]:<3}" if lg else "   -   "
		lg_err = f"{lg[3]:.2f}°" if lg else "  -  "
		lg_stb = f"{lg[2]:.1f}%" if lg else "  -  "
		ho = r2["ho"]
		n_err, n_stb = (f"{ho['NEURONS'][0]:.2f}°", f"{ho['NEURONS'][1]:.1f}%") if "NEURONS" in ho else ("  -  ", "  -  ")
		m_err, m_stb = (f"{ho['MEMORY'][0]:.2f}°", f"{ho['MEMORY'][1]:.1f}%") if "MEMORY" in ho else ("  -  ", "  -  ")
		dur = f"{r2['wall_min']:.0f}m" if r2["wall_min"] is not None else ("run" if lg else "  -  ")
		r1_s = f"{r1m[0]:.2f}°/{r1m[1]:.0f}%" if r1m else "     -     "
		if r1m and "MEMORY" in ho:
			ae = (r1m[0] + ho["MEMORY"][0]) / 2
			as_ = (r1m[1] + ho["MEMORY"][1]) / 2
			avg = f"{ae:.2f}°/{as_:.1f}%"
			ranked.append((name, as_, ae))
		else:
			avg = "      -     "
		print(f"  {i:>3} {name:<5} {e:>4.2f} {s:>4.2f} {j:>4.2f} {m:>4.2f} | "
		      f"{stage:>6} {lg_s:>8} {lg_err:>7} {lg_stb:>7} | "
		      f"{n_err:>6} {n_stb:>6} | {m_err:>6} {m_stb:>6} | {dur:>6} | "
		      f"{r1_s:>11} | {avg:>12}")
	if ranked:
		ranked.sort(key=lambda r: (-r[1], r[2]))
		print("\n  Ranking by avg(R1,R2) MEMORY held-out stable (then err):")
		for rk, (name, as_, ae) in enumerate(ranked, 1):
			print(f"    {rk}. {name}:  stable={as_:.1f}%  err={ae:.2f}°")
	print("\n  R1_M = round-1 MEMORY held-out; avg = mean of R1+R2 MEMORY held-out (only once R2 done). "
	      "All held-out on fresh report-seed, matched 5°.")


WEIGHTS_BY_NAME = {name: (e, s, j, m) for (name, e, s, j, m) in COMBOS}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dir", required=True, help="phased weight-sweep dir (<DIR>/<COMBO>/run.out per combo)")
	ap.add_argument("--round2", action="store_true", help="round-2 view: survivors only, + R1 number and R1/R2 average")
	args = ap.parse_args()
	base = Path(args.dir)
	if args.round2:
		report_round2(base)
		return

	rows = []
	done = 0
	for i, (name, e, s, j, m) in enumerate(COMBOS, 1):
		c = parse_combo(base / name / "run.out")
		if c["done"]:
			done += 1
		rows.append((i, name, e, s, j, m, c))

	hdr = (f"{'#':>3} {'combo':<5} {'err':>4} {'stb':>4} {'jrk':>4} {'mno':>4} | "
	       f"{'stage':>6} {'lastgen':>8} {'lg_err':>7} {'lg_stb':>7} | "
	       f"{'N_err':>6} {'N_stb':>6} | {'M_err':>6} {'M_stb':>6} | {'dur':>6}")
	print(f"  Phased weight sweep: {base.name}   ({done}/{len(COMBOS)} done)")
	print(hdr)
	print("  " + "-" * len(hdr))
	for (i, name, e, s, j, m, c) in rows:
		lg = c["last_gen"]
		stage = (c["stage"] or "-")[:6]
		lg_s = f"{lg[0]:>3}/{lg[1]:<3}" if lg else "   -   "
		lg_err = f"{lg[3]:.2f}°" if lg else "  -  "
		lg_stb = f"{lg[2]:.1f}%" if lg else "  -  "
		ho = c["ho"]
		n_err, n_stb = (f"{ho['NEURONS'][0]:.2f}°", f"{ho['NEURONS'][1]:.1f}%") if "NEURONS" in ho else ("  -  ", "  -  ")
		m_err, m_stb = (f"{ho['MEMORY'][0]:.2f}°", f"{ho['MEMORY'][1]:.1f}%") if "MEMORY" in ho else ("  -  ", "  -  ")
		dur = f"{c['wall_min']:.0f}m" if c["wall_min"] is not None else ("run" if lg else "  -  ")
		print(f"  {i:>3} {name:<5} {e:>4.2f} {s:>4.2f} {j:>4.2f} {m:>4.2f} | "
		      f"{stage:>6} {lg_s:>8} {lg_err:>7} {lg_stb:>7} | "
		      f"{n_err:>6} {n_stb:>6} | {m_err:>6} {m_stb:>6} | {dur:>6}")

	# Ranking by MEMORY held-out stable (the final honest number), completed combos only.
	ranked = [(name, c["ho"]["MEMORY"]) for (_i, name, *_w, c) in
	          [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows] if "MEMORY" in c["ho"]]
	if ranked:
		ranked.sort(key=lambda r: (-r[1][1], r[1][0]))  # stable desc, err asc
		print("\n  Ranking by MEMORY held-out (final honest number):")
		for rk, (name, (he, hs)) in enumerate(ranked, 1):
			print(f"    {rk}. {name}:  stable={hs:.1f}%  err={he:.2f}°")
	print("\n  N_/M_ = NEURONS/MEMORY per-stage HELD-OUT (fresh report-seed 99990001, matched 5°). "
	      "MEMORY = final result. lg = latest GA gen (during-search, optimistic).")


if __name__ == "__main__":
	main()
