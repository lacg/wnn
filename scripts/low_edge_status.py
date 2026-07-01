#!/usr/bin/env python3
"""Status for the low-edge LEAN sweep: s16 + pidmix_pwm × grid-bits {12,16,20} × seed {09,10}.
Goal = how lean (eff-sb / cells / duration) before stable% cliffs. Per-substrate stable-vs-eff-bits
curve + the lean summary (min eff-sb still on the ~83% plateau). Read-only. No args."""
import os, re, math, subprocess

ROOT = "logs/controller/LowEdge_20260701"
LOG  = ROOT + ".log"
GATE = "/tmp/wnn_state_integral_done.json"
SUBS = ["s16", "pidmix_pwm"]
BITS = [12, 16, 20]
SEEDS = [20260609, 20260610]

def sh(cmd):
	try: return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
	except Exception: return ""

def parse_cell(sub, gb, seed):
	d = f"{ROOT}/{sub}_b{gb}_seed{seed}"; out = f"{d}/run.out"
	c = {"phase": "pending", "src": "", "stable": None, "err": None, "sb": None,
	     "cells": None, "dur": None}
	if not os.path.exists(out):
		if os.path.exists(f"{d}/done.json"): c["phase"] = "DONE"
		return c
	txt = open(out, encoding="utf-8", errors="ignore").read()
	if os.path.exists(f"{d}/done.json"): c["phase"] = "DONE"
	else:
		pos = {}
		for lbl, pat in (("MEM", "ControllerGA-Memory"), ("NEUR", "ControllerGA-Neurons"), ("grid", "GRID SEARCH")):
			i = txt.rfind(pat)
			if i >= 0: pos[lbl] = i
		gen = re.findall(r"Gen (\d+/\d+)", txt)
		c["phase"] = f"run:{max(pos, key=pos.get) if pos else 'grid'} {gen[-1] if gen else ''}".strip()
	PAT = r"{} MULTI-SEED held-out.*?stable=([\d.]+)[^\d.]+([\d.]+).*?err=([\d.]+)[^\d.]+([\d.]+)"
	m = re.findall(PAT.format("MEMORY"), txt); src = "ho-mem"
	if not m:
		m = re.findall(PAT.format("NEURONS"), txt); src = "ho-neur"
	if m:
		s = m[-1]; c["stable"] = (float(s[0]), float(s[1])); c["err"] = (float(s[2]), float(s[3])); c["src"] = src
	sbm = re.findall(r"STAGE 1 \(NEURONS\) done: .*?sb=(\d+)", txt) or re.findall(r"arch sn=\d+ .*?sb=(\d+)", txt)
	if sbm: c["sb"] = int(sbm[-1])
	cg = re.findall(r"cells\[(\d+)-(\d+)\]", txt)
	if cg: c["cells"] = int(cg[-1][1])
	dm = re.findall(r"Total wall time: ([\d.]+) min", txt)
	if dm: c["dur"] = float(dm[-1])
	if c["phase"].startswith("run:") and c["stable"] and os.path.exists(LOG):
		if re.search(rf"FAIL {sub} b={gb} seed={seed} ", open(LOG, errors='ignore').read()): c["phase"] = "FAIL/OOM"
	return c

def pool(c1, c2, key):
	a, b = c1.get(key), c2.get(key)
	if not a or not b: return None
	(m1, s1), (m2, s2) = a, b
	return (m1+m2)/2, math.sqrt((s1*s1+s2*s2)/2 + ((m1-m2)/2)**2)

def fmt(t, u): return f"{t[0]:.1f}±{t[1]:.1f}{u}" if t else "—"

def main():
	from datetime import datetime, timezone
	now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
	started = os.path.isdir(ROOT) and any(os.path.exists(f"{ROOT}/{s}_b{b}_seed{sd}/run.out")
	                                      for s in SUBS for b in BITS for sd in SEEDS)
	print(f"============ LOW-EDGE LEAN sweep (s16 + pidmix_pwm) — {now} ============")
	if not started:
		gate = "A/B/C DONE ✓" if os.path.exists(GATE) else "waiting on state-integral A/B/C"
		print(f"PARKED — not started yet ({gate}). Chains after {GATE}. No cells yet.")
		return
	mem = sh("vm_stat | awk '/free/{f=$3}/inactive/{i=$3}/Pages active/{a=$3}/speculative/{sp=$3}/wired/{w=$4}END{t=f+i+a+sp+w; printf \"%.0f\",(f+i+sp)/t*100}'").strip()
	rss = sh("ps -eo rss,command | grep phased_ga | grep -v grep | awk '{s+=$1}END{printf \"%.1f\", s/1048576}'").strip()
	cells = {(s, b, sd): parse_cell(s, b, sd) for s in SUBS for b in BITS for sd in SEEDS}
	done = sum(1 for c in cells.values() if c["phase"] == "DONE")
	fail = sum(1 for c in cells.values() if c["phase"] == "FAIL/OOM")
	print(f"mem free: {mem or '?'}%  | controller RSS: {rss or 0}GB  | fill: {done}/12  FAIL/OOM: {fail}")
	for s in SUBS:
		print(f"\n----- {s} -----")
		hdr = f"{'bits':>4s} {'seed':>4s} {'PHASE':<11s} {'STABLE±SD':>11s} {'ERR±SD':>10s} {'sb':>4s} {'cells':>9s} {'SRC':>7s} {'DUR':>6s}"
		print(hdr); print("-"*len(hdr))
		curve = []
		for b in BITS:
			for sd in SEEDS:
				c = cells[(s, b, sd)]
				cl = f"{c['cells']:,}" if c['cells'] else "—"
				dur = f"{c['dur']:.0f}m" if c['dur'] else "—"
				print(f"{b:>4d} {str(sd)[-2:]:>4s} {c['phase']:<11s} {fmt(c['stable'],'%'):>11s} "
				      f"{fmt(c['err'],'°'):>10s} {str(c['sb'] or '—'):>4s} {cl:>9s} {c['src'] or '—':>7s} {dur:>6s}")
			p = pool(cells[(s, b, SEEDS[0])], cells[(s, b, SEEDS[1])], "stable")
			if p:
				sbs = [cells[(s, b, sd)]["sb"] for sd in SEEDS if cells[(s, b, sd)]["sb"]]
				effsb = f"eff-sb~{min(sbs)}-{max(sbs)}" if sbs else ""
				print(f"{'':>4s} {'POOL':>4s} {'(n=8)':<11s} {fmt(p,'%'):>11s} {'':>10s}  {effsb}")
				curve.append((b, p, min(sbs) if sbs else None))
		if curve:
			print(f"  CURVE {s}:  " + "   ".join(f"b{b}(eff{e})={p[0]:.1f}±{p[1]:.1f}" for b, p, e in curve))
	print("\nLEAN read: plateau ~83% (b24 bit-sweep). Watch where stable% first DROPS below ~80% as")
	print("eff-sb/cells shrink = the cliff = min viable memory. Flat = leaner-is-free (FPGA headline).")

if __name__ == "__main__":
	os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	main()
