#!/usr/bin/env python3
"""Assemble the 1-layer-vs-DFA study table from the per-run markers + baselines.

For each (substrate, feature, memory-mode) cell, aggregate the MEMORY-stage
held-out triple across the (up to 5) completed seeds as mean±std, and print it
against the 5 classical baselines. Reads whatever markers exist, so it is a
live progress view mid-campaign, not only a final report.

Usage:  build_dfa_1layer_table.py [--markdir /tmp/wnn_dfa1l]
"""
import argparse
import glob
import json
import os
import re
import statistics

_TRIPLE = re.compile(r"stable=([\d.]+)%\s+err=([\d.]+)°(?:\s+steady=([\d.]+)°)?")


def _parse_triple(s):
	"""'... stable=91.0% err=2.82° steady=2.78° ...' → (stable, err, steady|None)."""
	m = _TRIPLE.search(s or "")
	if not m:
		return None
	return (float(m.group(1)), float(m.group(2)),
	        float(m.group(3)) if m.group(3) else None)


def _fmt(xs):
	xs = [x for x in xs if x is not None]
	if not xs:
		return "   —   "
	m = statistics.mean(xs)
	sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
	return f"{m:5.1f}±{sd:4.1f}"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--markdir", default="/tmp/wnn_dfa1l")
	a = ap.parse_args()

	rows = {}          # (sub, feat, mode) -> list of (stable, err, steady)
	seeds_seen = {}    # cell -> set of seeds
	for f in glob.glob(os.path.join(a.markdir, "*.json")):
		if os.path.basename(f) == "baselines.json":
			continue
		try:
			d = json.load(open(f))
		except Exception:
			continue
		# Prefer the MEMORY-stage held-out (final pipeline output); fall back to
		# NEURONS if MEMORY is missing (run stopped early).
		tri = _parse_triple(d.get("held_memory")) or _parse_triple(d.get("held_neurons"))
		if tri is None:
			continue
		key = (d["substrate"], d["feature"], d["mode"])
		rows.setdefault(key, []).append(tri)
		seeds_seen.setdefault(key, set()).add(d.get("seed"))

	bl_path = os.path.join(a.markdir, "baselines.json")
	baselines = json.load(open(bl_path))["baselines"] if os.path.exists(bl_path) else {}

	print("=" * 78)
	print("  WNN 1-layer vs DFA — held-out (MEMORY stage), mean±std over seeds")
	print("  disturbance L2D, tilt 5°, 100 report-episodes × 2000 steps")
	print("=" * 78)
	hdr = f"  {'cell':28} {'n':>2}  {'stable%':>11} {'err°':>11} {'steady°':>11}"
	print(hdr)
	print("  " + "-" * 74)
	for sub in ("1layer", "dfa"):
		for feat in ("9feat", "10feat"):
			for mode in ("BINARY", "QUAD"):
				key = (sub, feat, mode)
				tris = rows.get(key, [])
				n = len(seeds_seen.get(key, set()))
				label = f"{sub:6} {feat:6} {mode}"
				if not tris:
					print(f"  {label:28} {n:>2}  {'(pending)':>11}")
					continue
				st = _fmt([t[0] for t in tris])
				er = _fmt([t[1] for t in tris])
				sy = _fmt([t[2] for t in tris])
				print(f"  {label:28} {n:>2}  {st:>11} {er:>11} {sy:>11}")
	print("  " + "-" * 74)
	print("  CLASSICAL BASELINES (same held-out set):")
	for name in ("PID", "LQR", "MPC", "LQI", "MPCOF"):
		b = baselines.get(name)
		if b:
			print(f"  {name:28}  -  {b['stable']:5.1f}      {b['err_deg']:5.2f}      "
			      f"{b['steady_deg']:5.2f}")
	print("=" * 78)
	total = sum(len(v) for v in rows.values())
	print(f"  completed runs: {total}/40")


if __name__ == "__main__":
	main()
