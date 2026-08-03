#!/usr/bin/env python3
"""The dfa1l study table on the ALIGNED axis (train-seed thermometer thresholds).

build_dfa_1layer_table.py reads the per-cell markers, which carry the held-out
triple as phased_ga printed it. Every cell run before 03/08/2026 printed that
triple through a REFIT address function — thresholds fit on the report seed while
the genome's cells were written under the train seed — so the memory was read where
nothing had been written. That is the bug documented in
docs/threshold_misalignment_finding.md.

This script reads experiments/rescore/all.json instead, which holds each frozen
winner replayed BOTH ways by scripts/rescore_winners.py, and reports the aligned
column. Cells not yet rescored are listed rather than silently dropped: a family's
row here may rest on fewer seeds than the marker table's row, and pretending
otherwise would trade one wrong number for another.

Two different +-SDs are in play and must not be mixed:
  * per-cell  : across the 5 REPORT seeds, one frozen winner -> test-set variance,
                the same axis as the classical baselines.
  * per-family: across TRAINING seeds, using each cell's report-seed MEAN -> the
                same axis as the marker table's +-SD, so the two are comparable.
Only the second is printed in the family table, to keep it side-by-side readable.

Usage: python3 scripts/build_dfa_aligned_table.py
"""
import json
import pathlib
import re
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESCORE = ROOT / "experiments" / "rescore" / "all.json"
MARKDIR = ROOT / "experiments" / "dfa1l_markers"
CELL_RE = re.compile(r"^(dfa|1layer)_(9|10)feat_(BINARY|QUAD)_s\d+$")

ORDER = [
	("1layer", "9feat", "BINARY"), ("1layer", "9feat", "QUAD"),
	("1layer", "10feat", "BINARY"), ("1layer", "10feat", "QUAD"),
	("dfa", "9feat", "BINARY"), ("dfa", "9feat", "QUAD"),
	("dfa", "10feat", "BINARY"), ("dfa", "10feat", "QUAD"),
]


def agg(values):
	"""mean +- sample SD; SD is 0.0 at n=1 (printed, never hidden)."""
	if not values:
		return None
	m = st.mean(values)
	s = st.stdev(values) if len(values) > 1 else 0.0
	return m, s


def main():
	data = json.loads(RESCORE.read_text())
	cells = data["cells"]
	by_family = {}
	for c in cells:
		key = (c["substrate"], c["feature"], c["mode"])
		by_family.setdefault(key, []).append(c)

	markers = {p.stem for p in MARKDIR.glob("*.json") if CELL_RE.match(p.stem)}
	rescored = {c["tag"] for c in cells}
	missing = sorted(markers - rescored)

	print("=" * 86)
	print("  WNN 1-layer vs DFA — held-out on the ALIGNED axis (train-seed thresholds)")
	print("  disturbance L2D, tilt 5°, 100 report-episodes × 2000 steps, 5 report seeds/cell")
	print("=" * 86)
	print(f"  {'cell':<28}{'n':>3}{'stable%':>14}{'err°':>12}{'steady°':>12}   vs BROKEN axis")
	print("  " + "-" * 82)
	for key in ORDER:
		fam = by_family.get(key)
		label = f"{key[0]:<7}{key[1]:<7}{key[2]}"
		if not fam:
			print(f"  {label:<28}{'—':>3}{'not yet rescored':>50}")
			continue
		a_st = agg([c["train"]["agg"]["stable"][0] for c in fam])
		a_er = agg([c["train"]["agg"]["err"][0] for c in fam])
		a_sd = agg([c["train"]["agg"]["steady"][0] for c in fam])
		b_st = agg([c["per_seed"]["agg"]["stable"][0] for c in fam])
		b_er = agg([c["per_seed"]["agg"]["err"][0] for c in fam])
		print(f"  {label:<28}{len(fam):>3}"
		      f"{a_st[0]:>9.1f}±{a_st[1]:<4.1f}{a_er[0]:>8.1f}±{a_er[1]:<3.1f}"
		      f"{a_sd[0]:>8.1f}±{a_sd[1]:<3.1f}"
		      f"   was {b_st[0]:>5.1f}% / {b_er[0]:>5.1f}°")
	print("  " + "-" * 82)
	print("  CLASSICAL BASELINES (compute_baselines.py, same 5 report seeds):")
	print(f"  {'PID':<28}{5:>3}{90.4:>9.1f}±{7.5:<4.1f}{4.0:>8.1f}±{0.4:<3.1f}{4.0:>8.1f}±{0.5:<3.1f}")
	print(f"  {'LQR':<28}{5:>3}{100.0:>9.1f}±{0.0:<4.1f}{1.6:>8.1f}±{0.1:<3.1f}{1.3:>8.1f}±{0.2:<3.1f}")
	print(f"  {'MPC':<28}{5:>3}{100.0:>9.1f}±{0.0:<4.1f}{1.7:>8.1f}±{0.1:<3.1f}{1.4:>8.1f}±{0.2:<3.1f}")
	print(f"  {'LQI':<28}{5:>3}{100.0:>9.1f}±{0.0:<4.1f}{1.4:>8.1f}±{0.1:<3.1f}{1.0:>8.1f}±{0.1:<3.1f}")
	print(f"  {'MPCOF':<28}{5:>3}{100.0:>9.1f}±{0.0:<4.1f}{0.8:>8.1f}±{0.0:<3.1f}{0.2:>8.1f}±{0.0:<3.1f}")
	print("=" * 86)
	print(f"  aligned cells: {len(rescored)}   |   markers on disk: {len(markers)}"
	      f"   |   STILL ON THE BROKEN AXIS: {len(missing)}")
	if missing:
		print("  not yet rescored (their marker triple is NOT trustworthy):")
		for t in missing:
			print(f"    {t}")
	print("=" * 86)


if __name__ == "__main__":
	main()
