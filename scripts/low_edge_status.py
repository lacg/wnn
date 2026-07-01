#!/usr/bin/env python3
"""Status for the low-edge INPUT-BITS lean sweep v2: s16 + pidmix_pwm × input-bits {4,8,12,16} × seed
{09,10}. Convention: total_sb = sn (recurrent prefix) + input-bits (suffix, sampled features). The
sweep FIXES the input-bits (suffix) and lets the neurons phase grow sn. x-axis = input-bits; sn/total/
cells/dur are the lean metrics. Goal = how few INPUT-bits before stable% cliffs. Read-only. No args."""
import os, re, math, subprocess

ROOT = "logs/controller/LowEdge_20260701"
LOG  = ROOT + ".log"
GATE = "/tmp/wnn_state_integral_done.json"
SUBS = ["s16", "pidmix_pwm"]
INPUTS = [4, 8, 12, 16]
SEEDS = [20260609, 20260610]

def sh(cmd):
	try: return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
	except Exception: return ""

def parse_cell(sub, inp, seed):
	d = f"{ROOT}/{sub}_in{inp}_seed{seed}"; out = f"{d}/run.out"
	c = {"phase": "pending", "src": "", "stable": None, "err": None, "sn": None,
	     "sb": None, "cells": None, "dur": None}
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
	arch = re.findall(r"(?:STAGE 4 \(MEMORY\)|STAGE 1 \(NEURONS\)) done: .*?arch sn=(\d+) .*?sb=(\d+)", txt)
	if arch: c["sn"], c["sb"] = int(arch[-1][0]), int(arch[-1][1])
	cg = re.findall(r"cells\[(\d+)-(\d+)\]", txt)
	if cg: c["cells"] = int(cg[-1][1])
	dm = re.findall(r"Total wall time: ([\d.]+) min", txt)
	if dm: c["dur"] = float(dm[-1])
	if c["phase"].startswith("run:") and c["stable"] and os.path.exists(LOG):
		if re.search(rf"FAIL {sub} in={inp} seed={seed} ", open(LOG, errors='ignore').read()): c["phase"] = "FAIL/OOM"
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
	started = os.path.isdir(ROOT) and any(os.path.exists(f"{ROOT}/{s}_in{i}_seed{sd}/run.out")
	                                      for s in SUBS for i in INPUTS for sd in SEEDS)
	print(f"====== LOW-EDGE INPUT-BITS lean sweep (s16 + pidmix_pwm) — {now} ======")
	print("total_sb = sn (recurrent prefix) + input-bits (sampled features). x-axis = input-bits.")
	if not started:
		gate = "A/B/C DONE ✓" if os.path.exists(GATE) else "waiting on state-integral A/B/C"
		print(f"PARKED — not started yet ({gate}). Chains after {GATE}. No cells yet.")
		return
	mem = sh("vm_stat | awk '/free/{f=$3}/inactive/{i=$3}/Pages active/{a=$3}/speculative/{sp=$3}/wired/{w=$4}END{t=f+i+a+sp+w; printf \"%.0f\",(f+i+sp)/t*100}'").strip()
	rss = sh("ps -eo rss,command | grep phased_ga | grep -v grep | awk '{s+=$1}END{printf \"%.1f\", s/1048576}'").strip()
	cells = {(s, i, sd): parse_cell(s, i, sd) for s in SUBS for i in INPUTS for sd in SEEDS}
	done = sum(1 for c in cells.values() if c["phase"] == "DONE")
	fail = sum(1 for c in cells.values() if c["phase"] == "FAIL/OOM")
	print(f"mem free: {mem or '?'}%  | controller RSS: {rss or 0}GB  | fill: {done}/16  FAIL/OOM: {fail}")
	for s in SUBS:
		print(f"\n----- {s} -----")
		hdr = f"{'inbits':>6s} {'seed':>4s} {'PHASE':<11s} {'STABLE±SD':>11s} {'ERR±SD':>10s} {'sn':>4s} {'total':>5s} {'cells':>9s} {'SRC':>7s} {'DUR':>6s}"
		print(hdr); print("-"*len(hdr))
		curve = []
		for inp in INPUTS:
			for sd in SEEDS:
				c = cells[(s, inp, sd)]
				cl = f"{c['cells']:,}" if c['cells'] else "—"
				dur = f"{c['dur']:.0f}m" if c['dur'] else "—"
				print(f"{inp:>6d} {str(sd)[-2:]:>4s} {c['phase']:<11s} {fmt(c['stable'],'%'):>11s} "
				      f"{fmt(c['err'],'°'):>10s} {str(c['sn'] or '—'):>4s} {str(c['sb'] or '—'):>5s} {cl:>9s} {c['src'] or '—':>7s} {dur:>6s}")
			p = pool(cells[(s, inp, SEEDS[0])], cells[(s, inp, SEEDS[1])], "stable")
			if p:
				print(f"{'':>6s} {'POOL':>4s} {'(n=8)':<11s} {fmt(p,'%'):>11s}")
				curve.append((inp, p))
		if curve:
			print(f"  CURVE {s} (input-bits→stable%):  " + "   ".join(f"in{i}={p[0]:.1f}±{p[1]:.1f}" for i, p in curve))
	print("\nLEAN read: bit-sweep anchor = input-16 → 83.5±5.5. Watch where stable% CLIFFS below ~80% as")
	print("input-bits shrink 16→12→8→4 = the floor = min viable INPUT budget (FPGA headline). sn floats free.")

if __name__ == "__main__":
	os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	main()
