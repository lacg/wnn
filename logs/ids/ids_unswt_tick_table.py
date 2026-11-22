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


def flow_for(db, arm: str, seed: int):
	if arm == "ctrl":
		return db.execute("select id, status, started_at, completed_at from flows where id=?", (CONTROLS[seed],)).fetchone()
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


def arm_cells(db, arm: str) -> list[dict]:
	return [T.cell_values(db, arm, s) for s in CONTROLS]


def per_seed(cells: list[dict], key: str, fmt: str) -> str:
	return " | ".join(fmt % c["raw"][key] if "raw" in c else STATE_WORD.get(c.get("state", "–"), c.get("state", "–"))
		for c in cells)


def paired_delta(cells: list[dict], ctrl: list[dict], key: str) -> str:
	pairs = [(c["raw"][key], k["raw"][key]) for c, k in zip(cells, ctrl) if "raw" in c and "raw" in k]
	if not pairs:
		return "–"
	d = [a - b for a, b in pairs]
	better = sum(x > 0 for x in d) if key == "f1" else sum(x < 0 for x in d)
	return "%+.2f (%d/%d)" % (sum(d) / len(d), better, len(d))


def mean_cells(cells: list[dict]) -> str:
	done = [c["raw"]["cells"] for c in cells if "raw" in c]
	return "%.2fM" % (sum(done) / len(done) / 1e6) if done else "–"


def arm_row(arm: str, cells: list[dict], ctrl: list[dict]) -> tuple:
	is_ctrl = arm == "ctrl"
	hours = [float(c["hours"]) for c in cells if "raw" in c]
	return (arm, PLANE[arm], per_seed(cells, "f1", "%.2f"), "—" if is_ctrl else paired_delta(cells, ctrl, "f1"),
		per_seed(cells, "fpr", "%.2f"), "—" if is_ctrl else paired_delta(cells, ctrl, "fpr"),
		mean_cells(cells), "%.1f" % (sum(hours) / len(hours)) if hours else "–")


PLANE = {"ctrl": "IDS-56", "D": "IDS-74", "BC": "IDS-74", "V02": "IDS-74", "Eon": "IDS-73", "Eoff": "IDS-73"}
LEGEND = ("ctrl = production unswt (IDS-56 encon) · D = desirability · BC = balance_classes off · V02 = vote share 0.02 · "
	"Eon / Eoff = cut E, dedupe on / off")
COLS = ("arm", "story", "F1 %  s1 | s2 | s3", "ΔF1 vs ctrl", "FPR %  s1 | s2 | s3", "ΔFPR vs ctrl", "cells", "hours")


def classical_arm_rows() -> list[tuple]:
	return [(name, "IDS-10", f1 + " (mean±SD)", "", fpr, "", size, "") for name, f1, fpr, _acc, size, _b, _h in classical_rows()]


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
	print("unswt (temporal_3way, 16b) · TEST · cell = GA best_f1 · margin_val_cal · seeds r20421-23 · "
		"Δ = paired mean vs ctrl (seeds better/done) · q = queued, fly = running")
	print(LEGEND)
	print(box(classical + wnn, len(classical)))


if __name__ == "__main__":
	main()
