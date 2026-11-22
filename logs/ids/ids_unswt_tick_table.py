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


def main():
	T.ARMS, T.SEEDS, T.CONTROLS, T.ARM_NAME, T.flow_for = ARMS, tuple(CONTROLS), CONTROLS, ARM_NAME, flow_for
	cols = " | ".join(ARMS)
	T.HEADER = ("seed", f"F1 % ({cols})", f"FPR % ({cols})", f"acc % ({cols})", f"cells/neurons ({cols})",
		f"bits ({cols})", f"hours ({cols})")
	db = sqlite3.connect(T.DB, uri=True)
	per_seed = [[T.cell_values(db, a, s) for a in ARMS] for s in CONTROLS]
	seeds = [T.seed_row(cells, "s%d (r%d)" % (i + 1, s)) for i, (s, cells) in enumerate(zip(CONTROLS, per_seed))]
	print("unswt (temporal_3way, 16b, 34b neurons)   ctrl = IDS-56 encon · D | BC | V02 = Plane IDS-74 · Eon | Eoff = Plane IDS-73 "
		"cut E dedupe on | off · cell = GA best_f1 · margin_val_cal · TEST")
	print(T.render(classical_rows(), seeds, T.mean_row(per_seed)))


if __name__ == "__main__":
	main()
