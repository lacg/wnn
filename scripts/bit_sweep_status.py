#!/usr/bin/env python3
"""Status for the pidmix_pwm folds=5 bit-sweep (grid-bits {24,40,64} x seed {09,10}).
Reports per-cell held-out stable%/err/steady (ho-mem), effective sb, wish_bits/
saturation (the 'wants more bits?' knee signal), distinct-address cells, duration;
plus the POOLED-over-8-held-outs stable-vs-bits CURVE. Read-only. No args."""
import os, re, glob, math, subprocess

ROOT = "logs/controller/BitSweep_pidmix_pwm_20260630"
LOG  = ROOT + ".log"
BITS = [24, 40, 64]
SEEDS = [20260609, 20260610]

def sh(cmd):
	try: return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
	except Exception: return ""

def parse_cell(gb, seed):
	d = f"{ROOT}/pidmix_pwm_b{gb}_seed{seed}"
	out = f"{d}/run.out"
	c = {"dir": d, "phase": "pending", "src": "", "stable": None, "err": None,
	     "steady": None, "sb": None, "wish": None, "sat": None, "cells": None, "dur": None}
	if not os.path.exists(out):
		# not started yet, or started but no run.out — check log
		if os.path.exists(f"{d}/done.json"): c["phase"] = "DONE"
		return c
	txt = open(out, errors="ignore").read()
	if os.path.exists(f"{d}/done.json"): c["phase"] = "DONE"
	else:
		# latest stage by position in the log (mixed-group findall mislabels; use rfind)
		pos = {}
		for lbl, pat in (("MEM", "ControllerGA-Memory"), ("NEUR", "ControllerGA-Neurons"), ("grid", "GRID SEARCH")):
			i = txt.rfind(pat)
			if i >= 0: pos[lbl] = i
		gen = re.findall(r"Gen (\d+/\d+)", txt)
		g = gen[-1] if gen else ""
		c["phase"] = f"run:{max(pos, key=pos.get) if pos else 'grid'} {g}".strip()
	# held-out: prefer MEMORY, else NEURONS
	m = re.findall(r"MEMORY MULTI-SEED held-out .*?: stable=([\d.]+)±([\d.]+)% err=([\d.]+)±([\d.]+). steady=([\d.]+)±([\d.]+)", txt)
	src = "ho-mem"
	if not m:
		m = re.findall(r"NEURONS MULTI-SEED held-out .*?: stable=([\d.]+)±([\d.]+)% err=([\d.]+)±([\d.]+). steady=([\d.]+)±([\d.]+)", txt); src = "ho-neur"
	if m:
		s = m[-1]; c["stable"] = (float(s[0]), float(s[1])); c["err"] = (float(s[2]), float(s[3]))
		c["steady"] = (float(s[4]), float(s[5])); c["src"] = src
	# effective sb + cells from STAGE done lines; wish/sat from split-pressure
	sbm = re.findall(r"STAGE 1 \(NEURONS\) done: .*?sb=(\d+)", txt)
	if sbm: c["sb"] = int(sbm[-1])
	sp = re.findall(r"split-pressure NEURONS\] best: sn=\d+ saturation=(\d+) wish_bits=(\d+)", txt)
	if sp: c["sat"], c["wish"] = int(sp[-1][0]), int(sp[-1][1])
	cg = re.findall(r"cells\[(\d+)-(\d+)\]", txt)
	if cg: c["cells"] = int(cg[-1][1])  # max distinct addresses seen
	dm = re.findall(r"Total wall time: ([\d.]+) min", txt)
	if dm: c["dur"] = float(dm[-1])
	# FAIL: ho-mem present, no done.json, driver logged FAIL b=gb
	if c["phase"].startswith("run:") and c["stable"] and os.path.exists(LOG):
		if re.search(rf"FAIL b={gb} ", open(LOG, errors='ignore').read()): c["phase"] = "FAIL/OOM"
	return c

def pool(c1, c2, key):
	a, b = c1.get(key), c2.get(key)
	if not a or not b: return None
	(m1, s1), (m2, s2) = a, b
	return (m1+m2)/2, math.sqrt((s1*s1+s2*s2)/2 + ((m1-m2)/2)**2)

def fmt(t, unit): return f"{t[0]:.1f}±{t[1]:.1f}{unit}" if t else "—"

def main():
	from datetime import datetime, timezone
	now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
	mem = sh("vm_stat | awk '/free/{f=$3}/inactive/{i=$3}/Pages active/{a=$3}/speculative/{s=$3}/wired/{w=$4}END{tot=f+i+a+s+w; printf \"%.0f\", (f+i+s)/tot*100}'").strip()
	rss = sh("ps -eo rss,command | grep -E 'phased_ga' | grep -v grep | awk '{s+=$1}END{printf \"%.1f\", s/1048576}'").strip()
	print(f"============ BIT-SWEEP pidmix_pwm folds=5 — {now} ============")
	print(f"mem free: {mem or '?'}%  | controller RSS: {rss or 0}GB  | ROOT {ROOT}")
	cells = {(gb, sd): parse_cell(gb, sd) for gb in BITS for sd in SEEDS}
	done = sum(1 for c in cells.values() if c["phase"] == "DONE")
	fail = sum(1 for c in cells.values() if c["phase"] == "FAIL/OOM")
	print(f"fill: {done}/6 done  |  FAIL/OOM: {fail}  |  (eff sb ≈ grid-bits +~20 after NEURONS growth)\n")
	hdr = f"{'bits':>4s} {'seed':>4s} {'PHASE':<11s} {'STABLE±SD':>11s} {'ERR±SD':>10s} {'STEADY±SD':>11s} {'sb':>4s} {'wish/sat':>8s} {'cells':>8s} {'SRC':>7s} {'DUR':>6s}"
	print(hdr); print("-"*len(hdr))
	curve = []
	for gb in BITS:
		for sd in SEEDS:
			c = cells[(gb, sd)]
			ws = f"{c['wish']}/{c['sat']}" if c['wish'] is not None else "—"
			cl = f"{c['cells']:,}" if c['cells'] else "—"
			dur = f"{c['dur']:.0f}m" if c['dur'] else "—"
			print(f"{gb:>4d} {str(sd)[-2:]:>4s} {c['phase']:<11s} {fmt(c['stable'],'%'):>11s} "
			      f"{fmt(c['err'],'°'):>10s} {fmt(c['steady'],'°'):>11s} {str(c['sb'] or '—'):>4s} "
			      f"{ws:>8s} {cl:>8s} {c['src'] or '—':>7s} {dur:>6s}")
		p = pool(cells[(gb, SEEDS[0])], cells[(gb, SEEDS[1])], "stable")
		pe = pool(cells[(gb, SEEDS[0])], cells[(gb, SEEDS[1])], "err")
		if p: print(f"{'':>4s} {'POOL':>4s} {'(n=8)':<11s} {fmt(p,'%'):>11s} {fmt(pe,'°'):>10s}"); curve.append((gb, p))
		print()
	if curve:
		print("CURVE  eff-bits →  ho-mem stable% (pooled over both seeds' 8 held-outs):")
		print("   " + "   ".join(f"b{gb}={p[0]:.1f}±{p[1]:.1f}" for gb, p in curve))
		# knee hint from wish_bits
		anyw = any(cells[(gb, sd)]["wish"] for gb in BITS for sd in SEEDS if cells[(gb, sd)]["wish"] is not None)
		print(f"   wish_bits signal: {'SOME neuron wants more bits (knee not yet reached)' if anyw else 'ZERO across all cells so far (arch never asks for more bits → knee at/below tested widths)'}")

if __name__ == "__main__":
	os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	main()
