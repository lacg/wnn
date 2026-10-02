"""READ-ONLY best SINGLE genome per arm, searched EVERYWHERE: 5 genome types x 7 threshold modes x Grid/GA.

Scope: IDSXD/IDSXD2/IDSXD3 on cicids + ciciot (quad 96b, random_3way). CIC-IoT runs started before the
OR-fold fix (30/08/2026 02:27 UTC) are dropped, as in load.py. Held-out TEST, validation_summaries final rows.
A best row is best-of-N (seeds x 2 phases x 5 genome types x 7 modes) — report as 'best found', next to the mean.

Usage: python3 scripts/ids2/best_anywhere.py <cicids|ciciot> [f1_floor ...]
"""
import json
import re
import sqlite3
import sys

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]
NAME_RE = re.compile(r"^(IDSXD[23]?)-(cicids|ciciot)-quad-96b-(.+?)-r(\d+)(-w64fix)?$")
OR_FOLD_FIX = "2026-08-30T02:27"
SQL = """SELECT f.id, f.name, f.started_at, e.phase_type, vs.genome_type, vs.threshold_metadata
	FROM validation_summaries vs JOIN flows f ON f.id = vs.flow_id JOIN experiments e ON e.id = vs.experiment_id
	WHERE f.name LIKE 'IDSXD%' AND f.status = 'completed' AND vs.validation_point = 'final'"""


def load(ds: str) -> list[dict]:
	con = sqlite3.connect(DB, uri=True)
	rows = []
	for fid, name, start, ph, gt, tm in con.execute(SQL):
		m = NAME_RE.match(name)
		if not m or m.group(2) != ds or (ds == "ciciot" and start < OR_FOLD_FIX):
			continue
		t = json.loads(tm)
		for md in MODES:
			v = t.get(md)
			if isinstance(v, dict) and v.get("f1") is not None:
				rows.append(dict(cohort=m.group(1), arm=m.group(3), seed=int(m.group(4)), fid=fid, md=md, gt=gt,
					ph="GS" if ph == "grid_search" else "GA", f1=v["f1"] * 100, fpr=v["fpr"] * 100, acc=v["acc"] * 100))
	return rows


def line(tag: str, r: dict | None) -> str:
	if r is None:
		return f"  {tag:<16} —"
	return (f"  {tag:<16} F1 {r['f1']:6.2f} | FPR {r['fpr']:5.2f} | Acc {r['acc']:6.2f} | {r['ph']} {r['gt']:<12} "
		f"{r['md']:<20} {r['cohort']} r{r['seed']} flow {r['fid']}")


def report(ds: str, floors: list[float]) -> None:
	rows = load(ds)
	arms = sorted({r["arm"] for r in rows})
	print(f"BEST SINGLE GENOMES — {ds}, anywhere (5 gt x 7 modes x GS/GA), held-out TEST")
	for arm in arms + ["ALL"]:
		rs = rows if arm == "ALL" else [r for r in rows if r["arm"] == arm]
		seeds = len({(r["cohort"], r["seed"]) for r in rs})
		print(f"{arm}  ({len(rs)} rows, {seeds} runs)")
		print(line("best F1", max(rs, key=lambda r: (r["f1"], -r["fpr"]))))
		print(line("best Acc", max(rs, key=lambda r: (r["acc"], -r["fpr"]))))
		for fl in floors:
			hi = [r for r in rs if r["f1"] >= fl]
			print(line(f"min FPR@F1>={fl:g}", min(hi, key=lambda r: (r["fpr"], -r["f1"])) if hi else None))


if __name__ == "__main__":
	report(sys.argv[1], [float(x) for x in sys.argv[2:]] or [90.0])
