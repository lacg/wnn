#!/usr/bin/env python3
"""unswt table for the hourly tick (Luiz 21/11/2026), same box format and cell as ids_width_tick_table.py.

Columns: ctrl = IDS-56 ms01-encon (production unswt, 7492/7496/7500) | D | BC | V02 = Plane IDS-74 one-setting
A/Bs | Eon = Plane IDS-73 cut E, dedupe on (7600-7602, re-queued by Luiz) | Eoff = IDS-73 cut E, dedupe off (IDS73b).
Cell = GA ga_neurons, best_f1, test_point final, margin_val_cal, TEST.
Classical rows = untuned RF/XGB, VAL-calibrated threshold, mean±SD over seeds r20419-23 on temporal_3way TEST,
from logs/ids/unsw_headroom_1611/bl_temporal_3way_{thermo16,raw}.jsonl.
"""
import json
import sqlite3
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ids_width_tick_table as T  # noqa: E402  (cell, hours, render: one implementation)

CONTROLS = {20421: 7492, 20422: 7496, 20423: 7500}
ARMS = ("ctrl", "D", "BC", "V02", "Eon", "Eoff")
ARM_NAME = {
	"D": "IDS74-unswt-%-ms01-encon-D-r{seed}", "BC": "IDS74-unswt-%-ms01-encon-BC-r{seed}",
	"V02": "IDS74-unswt-%-ms01-encon-V02-r{seed}", "Eon": "IDS73-unswt-%-ms01-encon-cutE-r{seed}",
	"Eoff": "IDS73b-unswt-%-ms01-cutEoff-r{seed}",  # 7615-7617 (Eon repeats) cancelled 21/11
}
BASELINES = HERE / "unsw_headroom_1611"


# Lever screens (Luiz 22/11): paired vs the ADOPTED IDS-73 E + dedupe-on flows 7600-7602.
LEVER_CONTROLS = {20421: 7600, 20422: 7601, 20423: 7602}
LEVER_ARMS = ("E", "C", "D24", "D16", "MDL", "E64")
LEVER_NAME = {
	"C": "IDS64-unswt-%-cutE-rktrain-r{seed}", "D24": "IDS65-unswt-%-cutE-b24-r{seed}",
	"D16": "IDS65-unswt-%-cutE-b16-r{seed}", "MDL": "IDS72-unswt-%-cutMDL-r{seed}",
	"E64": "IDS72-unswt-%-64bWb-%-cutE-r{seed}",
}
LEVER_PLANE = {"E": "IDS-73", "C": "IDS-64", "D24": "IDS-65", "D16": "IDS-65", "MDL": "IDS-72", "E64": "IDS-72"}
LEVER_LEGEND = ("E = adopted production unswt (cut E + dedupe on, 7600-7602) · C = train_only feature ranking · "
	"D24 / D16 = ~910 x 24b / ~1970 x 16b neurons at equal cells · MDL = MDLP cuts (no bit budget) · E64 = cut E at 64-bit encoder")


def flow_for(db, arm: str, seed: int):
	if arm == "ctrl":
		return db.execute("select id, status, started_at, completed_at from flows where id=?", (CONTROLS[seed],)).fetchone()
	if arm == "E":
		return db.execute("select id, status, started_at, completed_at from flows where id=?", (LEVER_CONTROLS[seed],)).fetchone()
	if arm in LEVER_NAME:
		return db.execute("select id, status, started_at, completed_at from flows where name like ? and status != 'cancelled' "
			"order by id asc limit 1", (LEVER_NAME[arm].format(seed=seed),)).fetchone()
	return db.execute("select id, status, started_at, completed_at from flows where name like ? and status != 'cancelled' "
		"order by id asc limit 1", (ARM_NAME[arm].format(seed=seed),)).fetchone()


def classical_rows() -> list[tuple]:
	rows = []
	for enc, label in (("thermo16", "16b"), ("raw", "raw top-20")):
		recs = [json.loads(l) for l in (BASELINES / f"bl_temporal_3way_{enc}.jsonl").read_text().splitlines() if l.strip()]
		for model in ("xgb", "rf"):
			rs = [r for r in recs if r["model"] == model]
			ms = lambda k: "%.2f±%.2f" % (statistics.mean(r[k] for r in rs), statistics.pstdev(r[k] for r in rs))  # noqa: E731
			nodes = statistics.mean(r["nodes"] for r in rs)
			size = "%.0fK nodes" % (nodes / 1e3) if nodes < 1e6 else "%.2fM nodes" % (nodes / 1e6)
			rows.append((f"{model.upper()} {label}", ms("test_f1"), ms("test_fpr").replace("±", "±"), ms("test_acc"), size, "—", "—"))
	return rows


STATE_WORD = {"queued": "q", "flying": "fly", "paused": "pause", "–": "–", "no-row": "?"}
METRICS = (("f1", "F1 %", "%.2f", True), ("fpr", "FPR %", "%.2f", False), ("acc", "acc %", "%.2f", True))
PLANE = {"ctrl": "IDS-56", "D": "IDS-74", "BC": "IDS-74", "V02": "IDS-74", "Eon": "IDS-73", "Eoff": "IDS-73"}
LEGEND = ("ctrl = production unswt until 22/11 (IDS-56 encon) · D = desirability · BC = balance_classes off · "
	"V02 = vote share 0.02 · Eon / Eoff = cut E, dedupe on / off (Eon ADOPTED 22/11)")
COLS = ("arm", "story", "F1 %  s1 | s2 | s3 | mean", "ΔF1", "FPR %  s1 | s2 | s3 | mean", "ΔFPR",
	"acc %  s1 | s2 | s3 | mean", "Δacc", "cells", "hours")


def arm_cells(db, arm: str) -> list[dict]:
	return [T.cell_values(db, arm, s) for s in CONTROLS]


def seeds_and_mean(cells: list[dict], key: str, fmt: str) -> str:
	words = [fmt % c["raw"][key] if "raw" in c else STATE_WORD.get(c.get("state", "–"), c.get("state", "–")) for c in cells]
	done = [c["raw"][key] for c in cells if "raw" in c]
	return " | ".join(words + [fmt % (sum(done) / len(done)) if done else "–"])


def paired_delta(cells: list[dict], ctrl: list[dict], key: str, higher_better: bool) -> str:
	pairs = [(c["raw"][key], k["raw"][key]) for c, k in zip(cells, ctrl) if "raw" in c and "raw" in k]
	if not pairs:
		return "–"
	d = [a - b for a, b in pairs]
	better = sum((x > 0) if higher_better else (x < 0) for x in d)
	return "%+.2f (%d/%d)" % (sum(d) / len(d), better, len(d))


def mean_cells(cells: list[dict]) -> str:
	done = [c["raw"]["cells"] for c in cells if "raw" in c]
	return "%.2fM" % (sum(done) / len(done) / 1e6) if done else "–"


def arm_row(arm: str, cells: list[dict], ctrl: list[dict], plane: dict = None, ctrl_arm: str = "ctrl") -> tuple:
	hours = [float(c["hours"]) for c in cells if "raw" in c]
	row = [arm, (plane or PLANE)[arm]]
	for key, _label, fmt, higher in METRICS:
		row += [seeds_and_mean(cells, key, fmt), "—" if arm == ctrl_arm else paired_delta(cells, ctrl, key, higher)]
	return tuple(row + [mean_cells(cells), "%.1f" % (sum(hours) / len(hours)) if hours else "–"])


def classical_arm_rows() -> list[tuple]:
	return [(name, "IDS-10", f1, "", fpr, "", acc, "", size, "") for name, f1, fpr, acc, size, _b, _h in classical_rows()]


def box(rows: list[tuple], split_after: int) -> str:
	widths = [max(len(r[i]) for r in rows + [COLS]) + 2 for i in range(len(COLS))]
	rule = lambda l, m, r, ch="─": l + m.join(ch * w for w in widths) + r  # noqa: E731
	line = lambda r: "│" + "│".join((" " + r[i]).ljust(widths[i]) if i < 2 else r[i].center(widths[i]) for i in range(len(COLS))) + "│"  # noqa: E731
	out = [rule("┌", "┬", "┐"), line(COLS), rule("├", "┼", "┤")]
	out += [line(r) for r in rows[:split_after]] + [rule("╞", "╪", "╡", "═")] + [line(r) for r in rows[split_after:]]
	return "\n".join(out + [rule("└", "┴", "┘")])


def main():
	T.ARMS, T.SEEDS, T.CONTROLS, T.ARM_NAME, T.flow_for = ARMS, tuple(CONTROLS), CONTROLS, ARM_NAME, flow_for
	db = sqlite3.connect(T.DB, uri=True)
	cells = {a: arm_cells(db, a) for a in ARMS}
	classical = classical_arm_rows()
	wnn = [arm_row(a, cells[a], cells["ctrl"]) for a in ARMS]
	print("unswt (temporal_3way, 16b) · TEST · cell = GA best_f1 · margin_val_cal · seeds r20421-23 · mean = over the "
		"arm's finished seeds · Δ = PAIRED mean vs ctrl (seeds better / paired) · classical = mean±SD · q = queued, fly = running")
	print(LEGEND)
	print(box(classical + wnn, len(classical)))
	lever = {a: arm_cells(db, a) for a in LEVER_ARMS}
	rows = [arm_row(a, lever[a], lever["E"], LEVER_PLANE, "E") for a in LEVER_ARMS]
	print()
	print("unswt LEVER SCREENS (22/11) · same cell and columns · Δ = PAIRED mean vs E (seeds better / paired)")
	print(LEVER_LEGEND)
	print(box(classical + rows, len(classical)))


if __name__ == "__main__":
	main()
