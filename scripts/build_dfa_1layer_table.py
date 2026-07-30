#!/usr/bin/env python3
"""Assemble the 1-layer-vs-DFA study table from the per-run markers + baselines.

For each (substrate, feature, memory-mode) cell, aggregate the MEMORY-stage
held-out triple across the (up to 5) completed seeds as mean±std, and print it
against the 5 classical baselines. Reads whatever markers exist, so it is a
live progress view mid-campaign, not only a final report.

Also carries per-cell wall-clock (mean±std hours) and the campaign's total search
compute, so the cost of a cell is visible next to what that cost bought.

Usage:  build_dfa_1layer_table.py [--markdir experiments/dfa1l_markers]
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
	ap.add_argument("--markdir", default="experiments/dfa1l_markers")
	a = ap.parse_args()

	rows = {}          # (sub, feat, mode) -> list of MEMORY-stage (stable, err, steady)
	durs = {}          # cell -> list of wall-clock hours, one per completed seed
	seeds_seen = {}    # cell -> set of seeds with a MEMORY-stage triple
	fallback_seen = {} # cell -> set of seeds that ONLY have a NEURONS-stage triple
	for f in glob.glob(os.path.join(a.markdir, "*.json")):
		if os.path.basename(f) == "baselines.json":
			continue
		try:
			d = json.load(open(f))
		except Exception:
			continue
		# The markdir also holds non-marker study artifacts (baselines_L3D.json,
		# gap decompositions, stress re-scores). Only per-cell markers carry a
		# substrate — anything else would KeyError two lines down.
		if "substrate" not in d:
			continue
		# Only the MEMORY-stage held-out (final pipeline output) may enter the
		# mean±std. A run stopped early (watchdog pause/kill mid-search) can have
		# only a NEURONS-stage triple — a DIFFERENT-stage number that must NOT be
		# folded into the MEMORY mean. Count it as an incomplete cell and flag it.
		key = (d["substrate"], d["feature"], d["mode"])
		mem_tri = _parse_triple(d.get("held_memory"))
		if mem_tri is None:
			if _parse_triple(d.get("held_neurons")) is not None:
				fallback_seen.setdefault(key, set()).add(d.get("seed"))
			continue
		rows.setdefault(key, []).append(mem_tri)
		seeds_seen.setdefault(key, set()).add(d.get("seed"))
		# Wall-clock hours for the WHOLE cell (grid + every GA stage), from the
		# marker's dur_s. Only ever recorded for a run that completed, so it stays
		# in step with the triple above — a cell's n and its duration n are equal.
		dur_s = d.get("dur_s")
		durs.setdefault(key, []).append(dur_s / 3600.0 if dur_s else None)

	bl_path = os.path.join(a.markdir, "baselines.json")
	bl_doc = json.load(open(bl_path)) if os.path.exists(bl_path) else {}
	baselines = bl_doc.get("baselines", {})
	# The seed whose 100 held-out episodes EVERY row in this table was scored on.
	paired_seed = str(bl_doc.get("meta", {}).get("report_seed", ""))

	print("=" * 82)
	print("  WNN 1-layer vs DFA — held-out (MEMORY stage), mean±std over seeds")
	print("  disturbance L2D, tilt 5°, 100 report-episodes × 2000 steps")
	print("=" * 82)
	hdr = f"  {'cell':28} {'n':>2}  {'stable%':>11} {'err°':>11} {'steady°':>11} {'dur (h)':>11}"
	print(hdr)
	print("  " + "-" * 78)
	for sub in ("1layer", "dfa"):
		for feat in ("9feat", "10feat"):
			for mode in ("BINARY", "QUAD"):
				key = (sub, feat, mode)
				tris = rows.get(key, [])
				n = len(seeds_seen.get(key, set()))
				n_fb = len(fallback_seen.get(key, set()) - seeds_seen.get(key, set()))
				flag = f"  ⚠ {n_fb} NEURONS-only run(s) excluded" if n_fb else ""
				label = f"{sub:6} {feat:6} {mode}"
				if not tris:
					pend = "(NEURONS-only)" if n_fb else "(pending)"
					print(f"  {label:28} {n:>2}  {pend:>11}{flag}")
					continue
				st = _fmt([t[0] for t in tris])
				er = _fmt([t[1] for t in tris])
				sy = _fmt([t[2] for t in tris])
				du = _fmt(durs.get(key, []))
				print(f"  {label:28} {n:>2}  {st:>11} {er:>11} {sy:>11} {du:>11}{flag}")
	print("  " + "-" * 78)
	# A baseline row must show its score on the SAME held-out episodes every WNN row
	# was scored on, or the comparison stops being paired. b["stable"] is the mean
	# ACROSS report-seeds, so printing it once the baselines are multi-seed would
	# silently compare WNN-on-100-episodes against baseline-on-N×100-DIFFERENT-episodes
	# — a gap that then moves with how hard the extra draws happened to be, not with
	# either controller. So the row reads per_seed[paired_seed]; the multi-seed spread
	# is a separate footnote, where it cannot be mistaken for the comparison.
	print(f"  CLASSICAL BASELINES (no training; scored on the SAME held-out "
	      f"seed {paired_seed} as every row above):")

	def _paired(b, key, idx):
		"""The baseline's value on the paired seed, not its cross-seed mean."""
		tri = b.get("per_seed", {}).get(paired_seed)
		return tri[idx] if tri else b[key]

	spread = []
	for name in ("PID", "LQR", "MPC", "LQI", "MPCOF"):
		b = baselines.get(name)
		if not b:
			continue
		st, er, sy = (_paired(b, "stable", 0), _paired(b, "err_deg", 1),
		              _paired(b, "steady_deg", 2))
		# Classical controllers are closed-form/solved per episode — there is no
		# search to time, so a duration here would not mean what the WNN column means.
		print(f"  {name:28} {1:>2}  {st:>11.1f} {er:>11.1f} {sy:>11.1f} {'—':>11}")
		if b.get("n_seeds", 1) > 1:
			spread.append((name, b))
	if spread:
		# TEST-SET variance (same controller, different held-out draws). The WNN ±SD
		# above is TRAINING-seed variance on a fixed draw. Different axes: this one
		# says how precise a 100-episode estimate is, that one says how much the
		# training seed moves the result. Never read them as the same error bar.
		n_bl = spread[0][1].get("n_seeds", 1)
		print("  " + "-" * 78)
		print(f"  TEST-SET PRECISION — same controllers across {n_bl} report seeds "
		      f"(NOT the WNN ±SD axis):")
		for name, b in spread:
			print(f"  {name:28} {n_bl:>2}  "
			      f"{b['stable']:5.1f}±{b.get('stable_std', 0.0):4.1f} "
			      f"{b['err_deg']:5.1f}±{b.get('err_std', 0.0):4.1f} "
			      f"{b['steady_deg']:5.1f}±{b.get('steady_std', 0.0):4.1f}")
	print("=" * 82)
	total = sum(len(v) for v in rows.values())
	spent = sum(h for v in durs.values() for h in v if h)
	print(f"  completed runs: {total}/40   |   search compute spent: {spent:,.1f}h")


if __name__ == "__main__":
	main()
