#!/usr/bin/env python
"""CTRL-69 (Luiz 22/11/2026): the rival rows of docs/ctrl_rival_row_ruling.md §0 on the
installed f32-estimator lineage (ABI 61/62), old (ABI 60, f64 Mahony) vs new, side by side.

Two sources, both TEST (35 CF2.1 units 99990101..99990135 x 100 episodes):
  * MPCOF rows A / B / C: the `[rescore]` lines of the cargo test
    rival_probe_tests::test_units_rival_rows_rescore_f32_lineage (--probe-log). Same probe,
    same fixture builder (scripts/ctrl_indi_val_fixture.py --units test) as the published
    rows (f6e084e8, spec §12.1), minus the published-value gate.
  * PID firmware row (and a cross-check of row A): the per-unit rival lines every flown run
    prints in its MEMORY-stage test-unit HELD-OUT REPORT blocks ("vs PID[est]" = the
    published PID row 91.1 / 3.33 / 2.76 / 0.352; "vs MPCOF>rate(h10)[est]" = row A),
    averaged over the 35 units. Those lines are printed at 0.01 deg / 0.001 m, so these
    means carry that rounding (old ABI-60 means reproduce the published rows to it).
Rivals are deterministic (pool-seeded per unit): every run of a lineage must print the same
rows; the script checks that across all --old-outs / --new-outs and reports the max spread.

  python scripts/ctrl61_rival_rows.py --probe-log P --old-outs A60_s1.out ... --new-outs A61_s1.out ... --md docs/ctrl61_rival_rows.md
"""
import argparse
import re
from collections import defaultdict

UNITS = range(99990101, 99990136)
RIVALS = ("PID[est]", "MPCOF>rate(h10)[est]")


def unit_rows(path: str) -> dict:
	"""{rival: {unit: (stable%, err, steady, pos)}} from the MEMORY-stage test-unit blocks."""
	rows, cur = defaultdict(dict), None
	for line in open(path, errors="replace"):
		m = re.search(r"HELD-OUT REPORT \[([^\]]+)\].*?FRESH seed (\d+)", line)
		if m:
			cur = int(m.group(2)) if m.group(1) == "MEMORY" and int(m.group(2)) in UNITS else None
			continue
		if cur is None:
			continue
		m = re.match(r"\s+vs (\S+)\s.*?stable=([\d.]+)%\s+err=([\d.]+)°\s+steady=([\d.]+)°.*?\bpos=([\d.]+)m", line)
		if m and m.group(1) in RIVALS and cur not in rows[m.group(1)]:
			rows[m.group(1)][cur] = tuple(float(x) for x in m.groups()[1:])
	return rows


def lineage(paths: list) -> tuple:
	"""Mean row per rival over the 35 units (from the first run) + max per-unit spread across runs."""
	all_rows = [unit_rows(p) for p in paths]
	first, spread = all_rows[0], 0.0
	for r in all_rows[1:]:
		for k in RIVALS:
			for u, v in first[k].items():
				spread = max(spread, max(abs(a - b) for a, b in zip(v, r[k][u])))
	out = {}
	for k in RIVALS:
		d = first[k]
		if sorted(d) != list(UNITS):
			raise SystemExit(f"{paths[0]}: {k} has {len(d)} of the 35 test units")
		mean = [sum(v[i] for v in d.values()) / len(d) for i in range(4)]
		fails = round(sum(100.0 - v[0] for v in d.values()))
		out[k] = (*mean, fails)
	return out, spread


def probe_rows(path: str) -> dict:
	rx = re.compile(r"\[rescore\] row (\w) NEW\(f32\): stable ([\d.]+) err ([\d.]+) steady ([\d.]+) pos ([\d.]+) "
	                r"fails (\d+)/\d+ pos_hold ([\d.]+) \| OLD\(ABI 60\): stable ([\d.]+) err ([\d.]+) steady ([\d.]+) pos ([\d.]+) \| (.*)")
	rows = {}
	for line in open(path, errors="replace"):
		m = rx.search(line)
		if m:
			g = m.groups()
			rows[g[0]] = {"name": g[11].strip(), "new": tuple(float(x) for x in g[1:6]), "old": tuple(float(x) for x in g[7:11])}
	if sorted(rows) != ["A", "B", "C"]:
		raise SystemExit(f"{path}: expected [rescore] rows A, B, C, found {sorted(rows)}")
	return rows


def fmt(stable, err, steady, pos, fails=None) -> str:
	f = "" if fails is None else f" ({int(fails)}/3500)"
	return f"{stable:6.2f} / {err:.3f} / {steady:.3f} / {pos:.3f}{f}"


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--probe-log", required=True)
	ap.add_argument("--old-outs", nargs="+", required=True)
	ap.add_argument("--new-outs", nargs="+", required=True)
	ap.add_argument("--md", required=True)
	ap.add_argument("--note", default="")
	a = ap.parse_args()
	pr = probe_rows(a.probe_log)
	old, s_old = lineage(a.old_outs)
	new, s_new = lineage(a.new_outs)
	uo, un = unit_rows(a.old_outs[0]), unit_rows(a.new_outs[0])
	moved = {k: sum(uo[k][u] != un[k][u] for u in UNITS) for k in RIVALS}
	W = 62
	lines = [
		"# Rival rows on the f32-estimator lineage (ABI 61/62) — old vs new (CTRL-69, 22/11/2026)",
		"",
		"Written by `scripts/ctrl61_rival_rows.py` from the driver `logs/controller/ctrl62/ab_r62.sh` after the ABI-62 landing.",
		"**Partition: TEST** = 35 CF2.1 units 99990101..99990135 × 100 episodes = 3500 episodes per row. Deterministic laws:",
		"one row each, no seed spread. Nothing tuned on these numbers. Cells: stable % / err ° / steady ° / pos m (fails).",
		"",
		"```",
		f"{'row':{W}s} | OLD ABI 60 (f64 Mahony)                 | NEW ABI 61/62 (f32 MahonyF32)          ",
		"-" * W + "-+-----------------------------------------+----------------------------------------",
	]
	for k in "ABC":
		o, n = pr[k]["old"], pr[k]["new"]
		lines.append(f"{(k + '  ' + pr[k]['name']):{W}s} | {fmt(100 * o[0] if o[0] <= 1 else o[0], *o[1:], 0)}   | {fmt(*n)}")
	p_o, p_n = old["PID[est]"], new["PID[est]"]
	lines.append(f"{'F  PID firmware cascade, as shipped (run-printed rows)':{W}s} | {fmt(*p_o)}   | {fmt(*p_n)}")
	a_o, a_n = old["MPCOF>rate(h10)[est]"], new["MPCOF>rate(h10)[est]"]
	lines.append(f"{'A* cross-check: MPCOF est-fed, run-printed rows':{W}s} | {fmt(*a_o)}   | {fmt(*a_n)}")
	lines += [
		"```",
		"",
		"- Rows A/B/C: `rival_probe_tests::test_units_rival_rows_rescore_f32_lineage` (cargo test --release, fixture",
		"  `scripts/ctrl_indi_val_fixture.py --units test` on the ABI-61 N0 s1 run log; self-check = this build reproduces",
		"  the installed wheel's est-fed row on unit 99990101). OLD = the published rows (docs/ctrl_rival_row_ruling.md §0,",
		"  spec §12.1, f6e084e8), quoted, not re-flown (the ABI-60 wheel is no longer installed).",
		"- Row F and A*: means of the per-unit rival lines the flown runs print (MEMORY-stage test-unit blocks, 0.01 ° /",
		f"  0.001 m print precision). OLD = {len(a.old_outs)} ABI-60 N0 run(s), NEW = {len(a.new_outs)} ABI-61/62 run(s); max per-unit",
		f"  spread across runs of one lineage: old {s_old:.4f}, new {s_new:.4f} (0 = deterministic, as expected).",
		f"- Units whose printed rival line moved old -> new: PID {moved['PID[est]']}/35, MPCOF est-fed {moved['MPCOF>rate(h10)[est]']}/35",
		"  (at print precision; the probe rows above carry full precision).",
		"- **Comparator (ruling §1): row B (equal-information MPCOF).** From this file on, WNN rows of the ABI-61/62",
		"  lineage are set beside the NEW column, never the OLD one.",
	]
	if a.note:
		lines += ["", a.note]
	open(a.md, "w").write("\n".join(lines) + "\n")
	b = pr["B"]["new"]
	print(f"RIVALS B(eq-info) {b[0]:.2f}/{b[1]:.3f}/{b[2]:.3f}/{b[3]:.3f} ({int(b[4])}) "
	      f"A {pr['A']['new'][1]:.3f}/{pr['A']['new'][2]:.3f} C {pr['C']['new'][1]:.3f}/{pr['C']['new'][2]:.3f} "
	      f"PID {p_n[0]:.2f}/{p_n[1]:.3f}/{p_n[2]:.3f} ({int(p_n[4])}) -> {a.md}")


if __name__ == "__main__":
	main()
