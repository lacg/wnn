"""Generate the Rule-7 5-table report for SP-* / SP100-* cohorts.

WHY THIS EXISTS: scripts/build_oi_vs_old_report.py auto-detects ONLY cohorts
carrying the `-FIXED-OLD-` rename marker (today: WSWEEP-T20-96b-C35-250n100b),
and scripts/build_xds_5tables.py matches ONLY the `XDS-<cohort>-<w>b-W<x>-C35-
<n>n<b>b-OI-r<seed>` name shape. Neither covers the SP-2027 paper cohorts
(SP100-* live cohort, SP-* memory-mode ablations), so those were previously
hand-assembled. This script is the tooling for them.

Cohort cell = flow-name family (everything before the trailing `-r<seed>`).
Every number comes from validation_summaries.threshold_metadata at the
'final' checkpoint = HELD-OUT report partition. Never from iterations.best_f1
(that is the during-search k-fold metric = train-on-eval leak).

Usage:
  python3 scripts/build_sp_5tables.py --prefix 'SP100-%'
  python3 scripts/build_sp_5tables.py --prefix 'SP-%' --exclude 'SP100-%'
  python3 scripts/build_sp_5tables.py --prefix 'SP100-%' --leaderboard-only
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_URI = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"

GENOMES = ["best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"]
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]
METRIC_TO_GENOME = {"fitness": "best_fitness", "f1_macro": "best_f1",
                    "fpr": "best_fpr", "accuracy": "best_acc", "ce": "best_ce"}
SEED_RE = re.compile(r"-r(\d+)$")


def fam_of(name):
	m = SEED_RE.search(name)
	return (name[: m.start()], int(m.group(1))) if m else (name, 0)


def fmt_pair(values):
	if not values:
		return "    —    "
	if len(values) == 1:
		return f"  {values[0]:5.2f}  "
	return f"{statistics.mean(values):>5.2f}±{statistics.stdev(values):.2f}"


def utc(ts):
	return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def stamp(dt):
	et = dt.astimezone(timezone(timedelta(hours=-4)))
	return f"{dt.strftime('%d/%m/%Y %H:%M')} UTC ({et.strftime('%d/%m/%Y %H:%M')} ET)"


def cfg_line(p):
	"""One-line experiment fingerprint from flows.config_json params."""
	mm = p.get("memory_mode") or "QUAD_WEIGHTED (worker default; param absent)"
	w = (p.get("fitness_weight_ce"), p.get("fitness_weight_acc"),
	     p.get("fitness_weight_f1"), p.get("fitness_weight_fpr"))
	return (f"dataset={p.get('ids_dataset')} split={p.get('ids_split')} "
	        f"bits={p.get('ids_n_bits')} feats={p.get('ids_feature_selection')} "
	        f"class={p.get('ids_classification')} | mem={mm} | "
	        f"caps {p.get('max_neurons')}n/{p.get('max_bits')}b | "
	        f"w(ce/acc/f1/fpr)={w[0]}/{w[1]}/{w[2]}/{w[3]} | "
	        f"kfold={p.get('ids_k_folds')}x{p.get('ids_kfold_per_gen')} gens={p.get('ga_generations')}")


def load(cur, like, exclude):
	"""Pull every completed flow in the prefix + its final held-out metrics."""
	ex = " AND f.name NOT LIKE ? " if exclude else ""
	params = (like, exclude) if exclude else (like,)

	cur.execute(f"SELECT id, name, status, started_at, completed_at, config_json "
	            f"FROM flows f WHERE f.name LIKE ? {ex}", params)
	flows = {r["id"]: dict(r) for r in cur.fetchall()}

	cur.execute(
		f"""SELECT f.id AS fid, f.name, vs.genome_type, e.phase_type, vs.threshold_metadata
		FROM validation_summaries vs
		JOIN flows f ON f.id = vs.flow_id
		JOIN experiments e ON e.id = vs.experiment_id
		WHERE f.name LIKE ? {ex} AND f.status='completed' AND vs.validation_point='final'""",
		params)

	data = defaultdict(lambda: defaultdict(lambda: defaultdict(
		lambda: defaultdict(lambda: {"f1": [], "fpr": [], "acc": []}))))
	pts = defaultdict(list)
	seeds = defaultdict(set)
	for r in cur.fetchall():
		fam, seed = fam_of(r["name"])
		gt = r["genome_type"]
		if gt not in GENOMES or not r["threshold_metadata"]:
			continue
		phase = "GS" if r["phase_type"] == "grid_search" else "GA"
		seeds[(fam, phase, gt)].add(seed)
		tm = json.loads(r["threshold_metadata"])
		for mode in MODES:
			md = tm.get(mode, {})
			if not isinstance(md, dict):
				continue
			f1 = md.get("f1", md.get("f1_macro"))
			if f1 is None or md.get("fpr") is None or md.get("acc") is None:
				continue
			f1, fpr, acc = f1 * 100, md["fpr"] * 100, md["acc"] * 100
			d = data[fam][phase][gt][mode]
			d["f1"].append(f1)
			d["fpr"].append(fpr)
			d["acc"].append(acc)
			pts[fam].append((f1, fpr, acc, phase, gt, mode, seed))

	arch = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"n": [], "b": []})))
	cur.execute(
		f"""SELECT f.name, bg.metric, e.phase_type, g.total_neurons, g.tiers_json
		FROM best_genomes bg
		JOIN genomes g ON g.id = bg.genome_id
		JOIN flows f ON f.id = bg.flow_id
		JOIN experiments e ON e.id = bg.experiment_id
		WHERE f.name LIKE ? {ex} AND f.status='completed' AND bg.threshold_mode='val_cal'""",
		params)
	for r in cur.fetchall():
		fam, _ = fam_of(r["name"])
		gt = METRIC_TO_GENOME.get(r["metric"])
		if not gt:
			continue
		phase = "GS" if r["phase_type"] == "grid_search" else "GA"
		arch[fam][phase][gt]["n"].append(r["total_neurons"])
		try:
			bpn = json.loads(r["tiers_json"]).get("bits_per_neuron", [])
			if bpn:
				arch[fam][phase][gt]["b"].append(statistics.mean(bpn))
		except Exception:
			pass
	return flows, data, pts, seeds, arch


def arch_str(nl, bl):
	def f(v):
		if not v:
			return "—"
		s = statistics.stdev(v) if len(v) > 1 else 0
		return f"{int(round(statistics.mean(v)))}±{int(round(s))}"
	return f"{f(nl)} neurons | {f(bl)} bits"


def header_block(out, title, flows):
	done = [f for f in flows.values() if f["status"] == "completed"]
	run = [f for f in flows.values() if f["status"] == "running"]
	q = [f for f in flows.values() if f["status"] == "queued"]
	paused = [f for f in flows.values() if f["status"] == "paused"]
	durs = [(utc(f["completed_at"]) - utc(f["started_at"])).total_seconds() / 60
	        for f in done if f["started_at"] and f["completed_at"]]
	avg = statistics.mean(durs) if durs else 0
	total = len(done) + len(run) + len(q)
	latest = max((utc(f["completed_at"]) for f in done if f["completed_at"]), default=None)
	out.append(f"# {title}")
	out.append("")
	out.append(f"    Flows : {len(done)}/{total} completed | running: {len(run)} | "
	           f"queued: {len(q)} | paused (not counted in target): {len(paused)}")
	out.append(f"    Total wall (completed) : {sum(durs)/60:.1f}h  |  Avg/run: {avg:.0f}m")
	out.append(f"    Latest done : {stamp(latest) if latest else '—'}")
	remaining = len(run) + len(q)
	if latest and avg and remaining:
		out.append(f"    ETA remaining {remaining} runs : {stamp(latest + timedelta(minutes=avg*remaining))}")
	out.append("")


def best_rows(cfg_pts):
	def pick(fn, key=lambda x: x[0]):
		f = [p for p in cfg_pts if fn(p)]
		return max(f, key=key) if f else None
	return [
		("Best F1 (any FPR)", max(cfg_pts, key=lambda x: x[0]) if cfg_pts else None),
		("Best F1 (FPR<10%)", pick(lambda p: p[1] < 10)),
		("Best F1 (FPR<6%)", pick(lambda p: p[1] < 6)),
		("Best F1 (FPR<5%)", pick(lambda p: p[1] < 5)),
		("Best F1 (FPR<4%)", pick(lambda p: p[1] < 4)),
		("Best F1 (FPR<2%)", pick(lambda p: p[1] < 2)),
		("Best FPR (any F1)", min(cfg_pts, key=lambda x: x[1]) if cfg_pts else None),
		("Best FPR (F1>80%)", pick(lambda p: p[0] > 80, key=lambda x: -x[1])),
		("Best FPR (F1>90%)", pick(lambda p: p[0] > 90, key=lambda x: -x[1])),
		("Best Acc (any FPR)", max(cfg_pts, key=lambda x: x[2]) if cfg_pts else None),
	]


def fmt_best_row(label, p):
	if p is None:
		return f"    {label:<20}|       — |       — |       — | —"
	return (f"    {label:<20}| {p[0]:6.2f}% | {p[1]:6.2f}% | {p[2]:6.2f}% | "
	        f"r{p[6]} {p[3]} {p[4]:<12} {p[5]}")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--prefix", required=True, help="SQL LIKE pattern, e.g. 'SP100-%%'")
	ap.add_argument("--exclude", default=None, help="SQL LIKE pattern to exclude")
	ap.add_argument("--title", default=None)
	ap.add_argument("--leaderboard-only", action="store_true")
	args = ap.parse_args()

	con = sqlite3.connect(DB_URI, uri=True)
	con.row_factory = sqlite3.Row
	cur = con.cursor()
	flows, data, pts, seeds, arch = load(cur, args.prefix, args.exclude)

	out = []
	header_block(out, args.title or f"Cohort {args.prefix}", flows)

	# family -> representative params (config fingerprint)
	fam_params = {}
	fam_status = defaultdict(lambda: defaultdict(int))
	for f in flows.values():
		fam, _ = fam_of(f["name"])
		fam_status[fam][f["status"]] += 1
		if fam not in fam_params:
			fam_params[fam] = json.loads(f["config_json"]).get("params", {})

	for fam in sorted(data.keys()):
		cfg_pts = pts[fam]
		st = fam_status[fam]
		n_done = st.get("completed", 0)
		n_tot = n_done + st.get("running", 0) + st.get("queued", 0)
		out.append("")
		out.append(f"## {fam}  ({n_done}/{n_tot} completed)")
		out.append("")
		out.append(f"    {cfg_line(fam_params.get(fam, {}))}")
		out.append("")
		out.append("### Best individual genomes (all phases x genome types x 7 modes)")
		out.append("")
		out.append("    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)")
		out.append("    --------------------+---------+---------+---------+--------------------------------------")
		for label, p in best_rows(cfg_pts):
			out.append(fmt_best_row(label, p))
		out.append("")
		for gt in GENOMES:
			gs_n = len(seeds[(fam, "GS", gt)])
			ga_n = len(seeds[(fam, "GA", gt)])
			out.append(f"### {gt}  (runs: GS {gs_n} | GA {ga_n})")
			out.append(f"    Grid Search : {arch_str(arch[fam]['GS'][gt]['n'], arch[fam]['GS'][gt]['b'])}")
			out.append(f"    GA Neurons  : {arch_str(arch[fam]['GA'][gt]['n'], arch[fam]['GA'][gt]['b'])}")
			out.append("")
			out.append("    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA")
			out.append("    ---------------------+---------------------+---------------------+--------------------")
			for mode in MODES:
				g = data[fam]["GS"][gt][mode]
				a = data[fam]["GA"][gt][mode]
				out.append(
					f"    {mode:<20} |{fmt_pair(g['f1'])} {fmt_pair(a['f1'])} "
					f"|{fmt_pair(g['fpr']):>10} {fmt_pair(a['fpr']):>9} "
					f"|{fmt_pair(g['acc']):>10} {fmt_pair(a['acc']):>9}")
			out.append("")

	con.close()
	return "\n".join(out)


if __name__ == "__main__":
	print(main())
