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
PATIENCE_RE = re.compile(r"patience=(\d+)/(\d+)")
STAGE_HDR_RE = re.compile(r"STAGE \d+: (\w+) \(")
HO_HDR_RE = re.compile(r"HELD-OUT REPORT \[(\w+)\]")
HO_RESULT_RE = re.compile(r"RESULT — during-search winner \(held-out\):\s+stable=([\d.]+)%\s+err=([\d.]+)°")
WALL_RE = re.compile(r"Total wall time:\s+([\d.]+) min")
STAGE_RE = re.compile(r"STAGE \d+: (\w+)")


def parse_combo(run_out: Path) -> dict:
	d = {"last_gen": None, "stage": None, "ho": {}, "wall_min": None, "done": False,
	     "patience": None, "elapsed_min": None}
	if not run_out.exists():
		return d
	import os, time
	try:
		d["elapsed_min"] = (time.time() - os.stat(run_out).st_birthtime) / 60.0
	except Exception:
		pass
	pending_ho_stage = None
	for line in run_out.read_text(errors="ignore").splitlines():
		ms = STAGE_RE.search(line)
		if ms:
			d["stage"] = ms.group(1)
			d["patience"] = None   # patience counter resets per stage
		mg = GEN_RE.search(line)
		if mg:
			d["last_gen"] = (int(mg.group(1)), int(mg.group(2)), float(mg.group(3)), float(mg.group(4)))
		mp = PATIENCE_RE.search(line)
		if mp:
			d["patience"] = (int(mp.group(1)), int(mp.group(2)))
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
	       f"{'stage':>6} {'lastgen':>13} {'lg_err':>7} {'lg_stb':>7} | "
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
		pat = r2["patience"]
		pat_s = f"({pat[0]}/{pat[1]})" if pat else ""
		lg_s = (f"{lg[0]:>3}/{lg[1]:<3}{pat_s}" if lg else "   -   ")
		lg_err = f"{lg[3]:.2f}°" if lg else "  -  "
		lg_stb = f"{lg[2]:.1f}%" if lg else "  -  "
		ho = r2["ho"]
		n_err, n_stb = (f"{ho['NEURONS'][0]:.2f}°", f"{ho['NEURONS'][1]:.1f}%") if "NEURONS" in ho else ("  -  ", "  -  ")
		m_err, m_stb = (f"{ho['MEMORY'][0]:.2f}°", f"{ho['MEMORY'][1]:.1f}%") if "MEMORY" in ho else ("  -  ", "  -  ")
		dur = (f"{r2['wall_min']:.0f}m" if r2["wall_min"] is not None else
		       (f"{r2['elapsed_min']:.0f}m+" if (lg and r2["elapsed_min"] is not None) else "  -  "))
		r1_s = f"{r1m[0]:.2f}°/{r1m[1]:.0f}%" if r1m else "     -     "
		if r1m and "MEMORY" in ho:
			ae = (r1m[0] + ho["MEMORY"][0]) / 2
			as_ = (r1m[1] + ho["MEMORY"][1]) / 2
			avg = f"{ae:.2f}°/{as_:.1f}%"
			ranked.append((name, as_, ae))
		else:
			avg = "      -     "
		print(f"  {i:>3} {name:<5} {e:>4.2f} {s:>4.2f} {j:>4.2f} {m:>4.2f} | "
		      f"{stage:>6} {lg_s:>13} {lg_err:>7} {lg_stb:>7} | "
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

# Round-3 base seeds (must match ROUND3_SEEDS in wsweep_orchestrator.py).
ROUND3_SEEDS = [20260609, 20260610, 20260611]


def round3_survivors(base: Path) -> list[str]:
	"""Survivor names from ROUND2_REPORT.txt (falls back to round3/ subdirs)."""
	rep = base / "ROUND2_REPORT.txt"
	if rep.exists():
		for line in rep.read_text().splitlines():
			if "SURVIVORS" in line and ":" in line:
				return [n.strip() for n in line.split(":", 1)[1].split(",") if n.strip()]
	r3 = base / "round3"
	if r3.exists():
		return sorted((p.name for p in r3.iterdir() if p.is_dir()), key=lambda n: (len(n), n))
	return []


def report_round3(base: Path):
	"""Round-3 table: top-3 survivors × 3 seeds, per-seed progress + per-combo
	mean±std of the MEMORY held-out (the final honest number). Interleaved order
	means each combo accrues seeds in lock-step, so partial means are comparable."""
	import statistics
	names = round3_survivors(base)
	if not names:
		print("  (no round-3 survivors found — is ROUND2_REPORT.txt written?)")
		return
	hdr = (f"{'seed':>6} {'base':>9} | {'stage':>6} {'lastgen':>13} "
	       f"{'lg_err':>7} {'lg_stb':>7} | {'N_err':>6} {'N_stb':>6} | {'M_err':>6} {'M_stb':>6} | {'dur':>6}")
	# count fully-done seeds across all combos
	tot_seeds = len(names) * len(ROUND3_SEEDS)
	done_seeds = 0
	combo_means = []   # (name, mean_err, mean_stb, n_done)
	lines = []
	for name in names:
		e, s, j, m = WEIGHTS_BY_NAME[name]
		# per-combo header: name + the 4 fitness weights, once
		lines.append(f"  {name} — err={e:.2f} stb={s:.2f} jrk={j:.2f} mno={m:.2f}")
		seed_ms = []
		for k, bs in enumerate(ROUND3_SEEDS, 1):
			c = parse_combo(base / "round3" / name / f"seed{bs}" / "run.out")
			if c["done"]:
				done_seeds += 1
			lg = c["last_gen"]
			stage = (c["stage"] or "-")[:6]
			pat = c["patience"]
			pat_s = f"({pat[0]}/{pat[1]})" if pat else ""
			lg_s = (f"{lg[0]:>3}/{lg[1]:<3}{pat_s}" if lg else "   -   ")
			lg_err = f"{lg[3]:.2f}°" if lg else "  -  "
			lg_stb = f"{lg[2]:.1f}%" if lg else "  -  "
			ho = c["ho"]
			n_err, n_stb = ((f"{ho['NEURONS'][0]:.2f}°", f"{ho['NEURONS'][1]:.1f}%")
			                if "NEURONS" in ho else ("  -  ", "  -  "))
			if "MEMORY" in ho:
				m_err, m_stb = f"{ho['MEMORY'][0]:.2f}°", f"{ho['MEMORY'][1]:.1f}%"
				seed_ms.append(ho["MEMORY"])
			else:
				m_err, m_stb = "  -  ", "  -  "
			dur = (f"{c['wall_min']:.0f}m" if c["wall_min"] is not None else
			       (f"{c['elapsed_min']:.0f}m+" if (lg and c["elapsed_min"] is not None) else "  -  "))
			lines.append(f"  {k:>6} {bs:>9} | {stage:>6} {lg_s:>13} "
			             f"{lg_err:>7} {lg_stb:>7} | {n_err:>6} {n_stb:>6} | {m_err:>6} {m_stb:>6} | {dur:>6}")
		if seed_ms:
			me = statistics.mean(v[0] for v in seed_ms)
			ms = statistics.mean(v[1] for v in seed_ms)
			se = statistics.stdev(v[0] for v in seed_ms) if len(seed_ms) > 1 else 0.0
			ss = statistics.stdev(v[1] for v in seed_ms) if len(seed_ms) > 1 else 0.0
			combo_means.append((name, me, ms, len(seed_ms)))
			lines.append(f"  {'MEAN':>6} {'(' + str(len(seed_ms)) + '/3)':>9} | "
			             f"MEMORY held-out: err={me:.2f}±{se:.2f}°  stable={ms:.1f}±{ss:.1f}%")
		else:
			lines.append(f"  {'MEAN':>6} {'(0/3)':>9} | (no seed finished its MEMORY stage yet)")
		lines.append("  " + "·" * len(hdr))
	print(f"  ROUND 3 — top-3 × 3-seed (heaviest: pop50/kfold5/steps1000): {base.name}   "
	      f"({done_seeds}/{tot_seeds} seed-runs done)")
	print(hdr)
	print("  " + "-" * len(hdr))
	for ln in lines:
		print(ln)
	if combo_means:
		combo_means.sort(key=lambda r: (-r[2], r[1]))   # stable desc, err asc (matches orchestrator cull)
		print("\n  Ranking by mean MEMORY held-out stable (then err) — orchestrator's WINNER rule:")
		for rk, (name, me, ms, n) in enumerate(combo_means, 1):
			star = "  ★" if rk == 1 and n == len(ROUND3_SEEDS) else ""
			print(f"    {rk}. {name}:  stable={ms:.1f}%  err={me:.2f}°  ({n}/3 seeds){star}")
	fr = base / "FINAL_REPORT.txt"
	print(f"\n  N_/M_ = NEURONS/MEMORY per-stage HELD-OUT (each seed's own report-seed, matched 5°). "
	      f"MEAN row (MEMORY) = the figure the WINNER is chosen on; N_<M_ err = memory stage overfit.")
	print(f"  FINAL_REPORT.txt: {'WRITTEN — round 3 complete' if fr.exists() else 'not yet (round 3 in progress)'}")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dir", required=True, help="phased weight-sweep dir (<DIR>/<COMBO>/run.out per combo)")
	ap.add_argument("--round2", action="store_true", help="round-2 view: survivors only, + R1 number and R1/R2 average")
	ap.add_argument("--round3", action="store_true", help="round-3 view: top-3 survivors × 3 seeds + mean±std MEMORY held-out")
	args = ap.parse_args()
	base = Path(args.dir)
	if args.round3:
		report_round3(base)
		return
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
		dur = (f"{c['wall_min']:.0f}m" if c["wall_min"] is not None else
		      (f"{c['elapsed_min']:.0f}m+" if (lg and c["elapsed_min"] is not None) else "  -  "))
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
