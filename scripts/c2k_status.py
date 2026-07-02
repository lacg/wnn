#!/usr/bin/env python3
"""Status for the C2K all-2000-trained committee pool (4 arms × seed {09,10}).
Arms: PWM2K / TILT2K / LEAN2K (input-4 lean) / ANCH2K (yaw-anchor, 30 gens) — all
trained at --steps 2000 (the horizon-drift fix) + immigrants 0.15, matching the
proven E2 LONG recipe. Free pool members: E2 LONG_s09 (done) + LONG_s10 (E2 sweep).
NOTE: held-out numbers here are on the 2000-step ruler (LONG_s09 = 88.8 there);
committee assembly runs the fresh-seed truth serum @2000/@5000 via e4_best_of_k.
Read-only. No args. Parse logic shared with e2_status.py."""
import os, re, math, subprocess

ROOT = "logs/controller/C2K_20260702"
LOG  = ROOT + ".log"
GATE = "/tmp/wnn_rescue_done.json"
ARMS = ["PWM2K", "TILT2K", "LEAN2K", "ANCH2K"]
SEEDS = [20260609, 20260610]
LONG_REF = (88.8, 3.3)   # E2 LONG_s09 ho-mem on the SAME 2000-step ruler

def sh(cmd):
	try: return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
	except Exception: return ""

def parse_cell(arm, seed):
	d = f"{ROOT}/{arm}_seed{seed}"; out = f"{d}/run.out"
	c = {"phase": "pending", "src": "", "stable": None, "err": None, "steady": None,
	     "sn": None, "wish": None, "sat": None, "cells": None, "dur": None}
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
	PAT = r"{} MULTI-SEED held-out.*?stable=([\d.]+)[^\d.]+([\d.]+).*?err=([\d.]+)[^\d.]+([\d.]+).*?steady=([\d.]+)[^\d.]+([\d.]+)"
	m = re.findall(PAT.format("MEMORY"), txt); src = "ho-mem"
	if not m:
		m = re.findall(PAT.format("NEURONS"), txt); src = "ho-neur"
	if m:
		s = m[-1]; c["stable"] = (float(s[0]), float(s[1])); c["err"] = (float(s[2]), float(s[3]))
		c["steady"] = (float(s[4]), float(s[5])); c["src"] = src
	snm = re.findall(r"STAGE 1 \(NEURONS\) done: .*?arch sn=(\d+)", txt) or re.findall(r"arch sn=(\d+)", txt)
	if snm: c["sn"] = int(snm[-1])
	sp = re.findall(r"split-pressure NEURONS\] best: sn=\d+ saturation=(\d+) wish_bits=(\d+)", txt)
	if sp: c["sat"], c["wish"] = int(sp[-1][0]), int(sp[-1][1])
	cg = re.findall(r"cells\[(\d+)-(\d+)\]", txt)
	if cg: c["cells"] = int(cg[-1][1])
	dm = re.findall(r"Total wall time: ([\d.]+) min", txt)
	if dm: c["dur"] = float(dm[-1])
	if c["phase"].startswith("run:") and c["stable"] and os.path.exists(LOG):
		if re.search(rf"FAIL {arm} seed={seed} ", open(LOG, errors='ignore').read()): c["phase"] = "FAIL/OOM"
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
	started = os.path.exists(ROOT) and any(
		os.path.exists(f"{ROOT}/{a}_seed{s}/run.out") for a in ARMS for s in SEEDS)
	print(f"====== C2K all-2000-trained committee pool — {now} ======")
	print(f"ruler = 2000-step ho (LONG_s09 ref {LONG_REF[0]}±{LONG_REF[1]} on it). Pool += free LONG_s09/LONG_s10.")
	if not started:
		gate = "rescue DONE ✓" if os.path.exists(GATE) else "waiting on low-edge seed10 rescue"
		print(f"PARKED — not started yet ({gate}). Chains after {GATE}. No cells yet.")
		return
	mem = sh("vm_stat | awk '/free/{f=$3}/inactive/{i=$3}/Pages active/{a=$3}/speculative/{sp=$3}/wired/{w=$4}END{t=f+i+a+sp+w; printf \"%.0f\",(f+i+sp)/t*100}'").strip()
	rss = sh("ps -eo rss,command | grep phased_ga | grep -v grep | awk '{s+=$1}END{printf \"%.1f\", s/1048576}'").strip()
	print(f"mem free: {mem or '?'}%  | controller RSS: {rss or 0}GB")
	cells = {(a, s): parse_cell(a, s) for a in ARMS for s in SEEDS}
	done = sum(1 for c in cells.values() if c["phase"] == "DONE")
	fail = sum(1 for c in cells.values() if c["phase"] == "FAIL/OOM")
	print(f"fill: {done}/8 done  |  FAIL/OOM: {fail}\n")
	hdr = f"{'arm':<8s} {'seed':>4s} {'PHASE':<13s} {'STABLE±SD':>11s} {'ERR±SD':>10s} {'STEADY±SD':>11s} {'sn':>4s} {'wish/sat':>8s} {'cells':>9s} {'SRC':>7s} {'DUR':>6s}"
	print(hdr); print("-"*len(hdr))
	for a in ARMS:
		for s in SEEDS:
			c = cells[(a, s)]
			ws = f"{c['wish']}/{c['sat']}" if c['wish'] is not None else "—"
			cl = f"{c['cells']:,}" if c['cells'] else "—"
			dur = f"{c['dur']:.0f}m" if c['dur'] else "—"
			print(f"{a:<8s} {str(s)[-2:]:>4s} {c['phase']:<13s} {fmt(c['stable'],'%'):>11s} "
			      f"{fmt(c['err'],'°'):>10s} {fmt(c['steady'],'°'):>11s} {str(c['sn'] or '—'):>4s} "
			      f"{ws:>8s} {cl:>9s} {c['src'] or '—':>7s} {dur:>6s}")
		ps = pool(cells[(a, SEEDS[0])], cells[(a, SEEDS[1])], "stable")
		py = pool(cells[(a, SEEDS[0])], cells[(a, SEEDS[1])], "steady")
		if ps:
			print(f"{'':<8s} {'POOL':>4s} {'(n=8)':<13s} {fmt(ps,'%'):>11s} {'':>10s} {fmt(py,'°'):>11s}"
			      f"  (LONG ref {LONG_REF[0]}: {ps[0]-LONG_REF[0]:+.1f}pp)")
		print()
	print("On 8/8: assemble via E4_STEPS=2000/5000 e4_best_of_k over the pool (+LONG_s09/s10,")
	print("+audition ANCH_s09-@500) — fresh-seed truth serum FIRST, then mean-PWM committees of 6-8.")

if __name__ == "__main__":
	os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	main()
