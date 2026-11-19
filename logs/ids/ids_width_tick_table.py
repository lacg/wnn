#!/usr/bin/env python3
"""unswr width table for the hourly tick (Luiz 18/11/2026): C (64b) | 80 (IDS62) | 96 (IDS63), per seed.

Cell = GA ga_neurons, best_f1, test_point final, margin_val_cal, TEST (test_summaries.threshold_metadata).
Classical rows on top = Luiz's target rows (same random_3way TEST): untuned RF/XGB defaults, VAL-calibrated threshold,
mean±SD over 5 seeds r20419-23, from logs/ids/unsw_headroom_1611/bl_random_3way_{thermo64,raw}.jsonl. Classical source story = Plane IDS-10 (rs=42 single fit + this 5-seed update, 18/11).
"""
import json
import sqlite3
from datetime import datetime, timezone

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
SEEDS = (20421, 20422, 20423)
CONTROLS = {20421: 7567, 20422: 7568, 20423: 7569}
ARMS = ("C", "80", "96")
ARM_NAME = {"80": "IDS62-unswr-%-fixb80-encoff-r{seed}", "96": "IDS63-unswr-%-fixb96-encoff-r{seed}"}
CLASSICAL = (
	("XGB 64b", "95.53±0.00", "0.417±0.000", "99.33±0.00", "6K nodes"),
	("RF 64b", "95.32±0.07", "0.380±0.064", "99.31±0.00", "1.78M nodes"),
	("XGB raw top-20", "95.60±0.00", "0.427±0.000", "99.34±0.00", "6K nodes"),
	("RF raw top-20", "95.81±0.07", "0.288±0.027", "99.39±0.01", "1.16M nodes"),
)
HEADER = ("seed", "F1 % (C | 80 | 96)", "FPR % (C | 80 | 96)", "acc % (C | 80 | 96)",
	"cells/neurons (C | 80 | 96)", "bits (C | 80 | 96)", "hours (C | 80 | 96)")


def flow_for(db, arm: str, seed: int):
	if arm == "C":
		return db.execute("select id, status, started_at, completed_at from flows where id=?", (CONTROLS[seed],)).fetchone()
	return db.execute("select id, status, started_at, completed_at from flows where name like ? and status != 'cancelled' "
		"order by id desc limit 1", (ARM_NAME[arm].format(seed=seed),)).fetchone()


def hours(start: str, end: str | None) -> float:
	t0 = datetime.fromisoformat(start)
	t1 = datetime.fromisoformat(end) if end else datetime.now(timezone.utc)
	return (t1 - t0).total_seconds() / 3600


def result(db, flow_id: int) -> dict | None:
	row = db.execute("select t.threshold_metadata, t.logical_cells, t.genome_hash from test_summaries t "
		"join experiments e on e.id=t.experiment_id where t.flow_id=? and e.phase_type='ga_neurons' "
		"and t.test_point='final' and t.genome_type='best_f1'", (flow_id,)).fetchone()
	if not row:
		return None
	m = json.loads(row[0])["margin_val_cal"]
	tiers = db.execute("select g.tiers_json from genomes g join experiments e on e.id=g.experiment_id "
		"where e.flow_id=? and g.genome_hash=?", (flow_id, row[2])).fetchone()
	bits = json.loads(tiers[0])["bits_per_neuron"] if tiers else []
	return {"f1": m["macro_f1"] * 100, "fpr": m["fpr"] * 100, "acc": m["acc"] * 100, "cells": row[1],
		"neurons": len(bits), "bits": (min(bits), max(bits)) if bits else None}


def cell_values(db, arm: str, seed: int) -> dict:
	"""Formatted per-column strings for one arm/seed, or a status word in every column."""
	f = flow_for(db, arm, seed)
	if not f:
		return {"state": "–"}
	fid, status, started, completed = f
	if status != "completed":
		word = "flying" if status == "running" else status
		extra = "%.1f" % hours(started, None) if status == "running" and started else word
		return {"state": word, "hours": extra}
	r = result(db, fid)
	if not r:
		return {"state": "no-row"}
	lo, hi = r["bits"] or (0, 0)
	return {"f1": "%.2f" % r["f1"], "fpr": "%.3f" % r["fpr"], "acc": "%.2f" % r["acc"],
		"size": "%.2fM/%d" % (r["cells"] / 1e6, r["neurons"]), "bits": str(lo) if lo == hi else "%d-%d" % (lo, hi),
		"hours": "%.1f" % hours(started, completed), "raw": r}


KEYS = ("f1", "fpr", "acc", "size", "bits", "hours")


def seed_row(cells: list[dict], label: str) -> tuple:
	return (label,) + tuple(" | ".join(c.get(k, c.get("state", "–")) for c in cells) for k in KEYS)


def mean_row(per_seed: list[list[dict]]) -> tuple:
	"""PAIRED means: over the seeds where every arm that has any result is done; arms with none show '–'."""
	active = [a for a in range(len(ARMS)) if any("raw" in s[a] for s in per_seed)]
	common = [i for i, s in enumerate(per_seed) if all("raw" in s[a] for a in active)]
	cols = {k: [] for k in KEYS}
	for a in range(len(ARMS)):
		done = [per_seed[i][a] for i in common] if a in active else []
		if not done:
			for k in KEYS:
				cols[k].append("–")
			continue
		mean = lambda key: sum(d["raw"][key] for d in done) / len(done)  # noqa: E731
		lo = min(d["raw"]["bits"][0] for d in done)
		hi = max(d["raw"]["bits"][1] for d in done)
		cols["f1"].append("%.3f" % mean("f1"))
		cols["fpr"].append("%.3f" % mean("fpr"))
		cols["acc"].append("%.2f" % mean("acc"))
		cols["size"].append("%.2fM/%d" % (mean("cells") / 1e6, round(mean("neurons"))))
		cols["bits"].append(str(lo) if lo == hi else "%d-%d" % (lo, hi))
		cols["hours"].append("%.1f" % (sum(float(d["hours"]) for d in done) / len(done)))
	label = "mean s%s (paired)" % ",".join(str(i + 1) for i in common) if common else "mean (none)"
	return (label,) + tuple(" | ".join(cols[k]) for k in KEYS)


def render(classical: list[tuple], seeds: list[tuple], mean: tuple) -> str:
	allrows = classical + seeds + [mean, HEADER]
	widths = [max(len(r[i]) for r in allrows) + 2 for i in range(len(HEADER))]

	def rule(left, mid, right, ch="─"):
		return left + mid.join(ch * w for w in widths) + right

	def line(r):
		return "│" + "│".join((" " + r[i]).ljust(widths[i]) if i == 0 else r[i].center(widths[i])
			for i in range(len(HEADER))) + "│"

	out = [rule("┌", "┬", "┐"), line(HEADER), rule("├", "┼", "┤")]
	out += [line(r) for r in classical]
	out += [rule("╞", "╪", "╡", "═")] + [line(r) for r in seeds]
	out += [rule("├", "┼", "┤"), line(mean), rule("└", "┴", "┘")]
	return "\n".join(out)


def main():
	db = sqlite3.connect(DB, uri=True)
	classical = [(name, f1, fpr, acc, size, "—", "—") for name, f1, fpr, acc, size in CLASSICAL]
	per_seed = [[cell_values(db, a, s) for a in ARMS] for s in SEEDS]
	seeds = [seed_row(cells, "s%d (r%d)" % (i + 1, s)) for i, (s, cells) in enumerate(zip(SEEDS, per_seed))]
	print(render(classical, seeds, mean_row(per_seed)))


if __name__ == "__main__":
	main()
