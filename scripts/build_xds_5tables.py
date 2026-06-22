"""Generate the 5-table format report for XDS-unsw-temporal (or any XDS cohort).

Adapts scripts/build_c35_5tables.py to XDS, which has multiple (width × weight)
sub-cohorts inside one prefix. Generates ONE 5-table block per (width, weight),
sorted width ascending then weight (a,b,c).

Per-block layout matches docs/ids_results.md exactly:
  - Header: completed/target, total wall, avg/run, latest done
  - Best individual genomes (F1/FPR/Acc with FPR sub-cohort cutoffs)
  - 5 tables (best_fitness, best_f1, best_fpr, best_acc, best_ce)
  - Each table: arch line (Grid Search / GA Neurons mean±std neurons/bits),
    then 7-threshold × Grid/GA F1/FPR/Acc grid

Usage:
  python3 scripts/build_xds_5tables.py                  # unsw-temporal (default)
  python3 scripts/build_xds_5tables.py --cohort unsw-random
  python3 scripts/build_xds_5tables.py --cohort cicids
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

DB = Path("/Users/lacg/wnn/db/wnn.db")

# Weight decoder — from reference_xds_weight_schemes:
#   Wa = ce=0.35, acc=0.30 (CIC-IoT legacy, "original")
#   Wb = ce=0.10, acc=0.20 (paper/PUB50, "balanced")
#   Wc = ce=0.70, acc=0.10 (CE-heavy, NEW probe)
WEIGHT_DESC = {
	"a": "Wa (CIC-IoT legacy, ce=0.35 acc=0.30)",
	"b": "Wb (paper/PUB50, ce=0.10 acc=0.20)",
	"bu": "Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)",
	"c": "Wc (CE-heavy NEW, ce=0.70 acc=0.10)",
}

GENOMES = ["best_fitness", "best_f1", "best_fpr", "best_acc", "best_ce"]
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]


def fmt_pair(values):
	if not values:
		return "    —    "
	if len(values) == 1:
		return f"  {values[0]:5.2f}  "
	return f"{statistics.mean(values):>5.2f}±{statistics.stdev(values):.2f}"


def fmt_pair_compact(values):
	"""Used by the 'Best individual genomes' table — single best point, no mean±std."""
	if not values:
		return "  —  "
	return f"{max(values):.2f}"


def main():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--cohort", default="unsw-temporal",
	                choices=["unsw-temporal", "unsw-random", "cicids", "ciciot"],
	                help="XDS sub-cohort to report on.")
	ap.add_argument("--target", type=int, default=None,
	                help="Target flow count for ETA. Default: all valid (width,weight,seed) cells.")
	args = ap.parse_args()

	# cicids XDS flows are all the `random` split, so the name carries the
	# `-random-` segment (XDS-cicids-random-32b-...). unsw cohorts already encode
	# the split in the cohort name (unsw-random / unsw-temporal).
	if args.cohort == "cicids":
		prefix = "XDS-cicids-random-"
	elif args.cohort == "ciciot":
		# ciciot is a width × weight × ARCH × seed probe sweep (90 cells): two arches
		# (250n100b, 500n34b) share each (width, weight). Capture arch so the two stay
		# in SEPARATE cells (otherwise their means would be silently merged).
		prefix = "XDS-ciciot-subsample-"
	else:
		prefix = f"XDS-{args.cohort}-"
	# NOTE 31/05/2026: switched literal `500n34b` to `\d+n\d+b` so cohorts can be
	# resized without breaking this matcher (UNSW-random was 50n34b on 31/05).
	# Weight group is [a-z]+ (not [abc]) so multi-letter schemes like Wbu match.
	# group(3) = arch ("250n100b"); single-arch cohorts get a constant suffix → unchanged.
	name_re = re.compile(rf"^{re.escape(prefix)}(\d+)b-W([a-z]+)-C35-(\d+n\d+b)-OI-r(\d+)$")

	con = sqlite3.connect(str(DB))
	con.row_factory = sqlite3.Row
	cur = con.cursor()

	# Top-level cohort stats
	cur.execute(
		"SELECT COUNT(*) FROM flows WHERE name LIKE ? AND name NOT LIKE '%PREEMP-OLD%' AND status='completed'",
		(f"{prefix}%",),
	)
	total_completed = cur.fetchone()[0]
	cur.execute(
		"""SELECT (julianday(completed_at)-julianday(started_at))*1440 AS m, completed_at
		FROM flows WHERE name LIKE ? AND name NOT LIKE '%PREEMP-OLD%' AND status='completed'
		ORDER BY completed_at""",
		(f"{prefix}%",),
	)
	rows = list(cur)
	durs = [r["m"] for r in rows if r["m"] is not None]
	avg_dur = statistics.mean(durs) if durs else 0
	total_dur_h = sum(durs) / 60 if durs else 0
	latest_done = rows[-1]["completed_at"] if rows else None
	latest_done_str = (
		datetime.fromisoformat(latest_done.replace("Z", "+00:00")).strftime("%d/%m/%Y %H:%M") + " UTC"
		if latest_done
		else "—"
	)

	# Pull ALL validation_summaries rows for the cohort, classify by (width, weight)
	cur.execute(
		"""SELECT f.name, vs.genome_type, e.phase_type, vs.threshold_metadata
		FROM validation_summaries vs
		JOIN flows f ON f.id=vs.flow_id
		JOIN experiments e ON e.id=vs.experiment_id
		WHERE f.name LIKE ? AND f.name NOT LIKE '%PREEMP-OLD%' AND f.status='completed'
		  AND vs.validation_point='final'""",
		(f"{prefix}%",),
	)

	# data[(width, weight)][phase][genome_type][mode] = {"f1": [...], "fpr": [...], "acc": [...]}
	data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"f1": [], "fpr": [], "acc": []}))))
	# all_pts[(width, weight)] = list of (f1, fpr, acc, phase, gt, mode)
	all_pts = defaultdict(list)
	# Distinct seed counts per (width, weight, phase, gt) for "(GS: N runs | GA: N runs)"
	seen_seeds = defaultdict(set)

	for r in cur:
		m = name_re.match(r["name"])
		if not m:
			continue
		width, weight, arc, seed = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
		cfg = (width, weight, arc)
		phase = "GS" if r["phase_type"] == "grid_search" else "GA"
		gt = r["genome_type"]
		if gt not in GENOMES:
			continue
		seen_seeds[(cfg, phase, gt)].add(seed)
		tm = json.loads(r["threshold_metadata"])
		for mode in MODES:
			md = tm.get(mode, {})
			if not isinstance(md, dict) or md.get("f1") is None:
				continue
			f1, fpr, acc = md["f1"] * 100, md["fpr"] * 100, md["acc"] * 100
			data[cfg][phase][gt][mode]["f1"].append(f1)
			data[cfg][phase][gt][mode]["fpr"].append(fpr)
			data[cfg][phase][gt][mode]["acc"].append(acc)
			if f1 >= 80:
				all_pts[cfg].append((f1, fpr, acc, phase, gt, mode, seed))

	# Architecture: per (width, weight, phase, genome_type) — neuron counts + avg bits
	arch = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"n": [], "b": []})))
	cur.execute(
		"""SELECT f.name, bg.metric, e.phase_type, g.total_neurons, g.tiers_json
		FROM best_genomes bg
		JOIN genomes g ON g.id = bg.genome_id
		JOIN flows f ON f.id = bg.flow_id
		JOIN experiments e ON e.id = bg.experiment_id
		WHERE f.name LIKE ? AND f.name NOT LIKE '%PREEMP-OLD%' AND f.status='completed'
		  AND bg.threshold_mode='val_cal'""",
		(f"{prefix}%",),
	)
	metric_to_genome = {"fitness": "best_fitness", "f1_macro": "best_f1",
	                    "fpr": "best_fpr", "accuracy": "best_acc", "ce": "best_ce"}
	for r in cur:
		m = name_re.match(r["name"])
		if not m:
			continue
		width, weight, arc = int(m.group(1)), m.group(2), m.group(3)
		cfg = (width, weight, arc)
		gt = metric_to_genome.get(r["metric"])
		if not gt:
			continue
		phase = "GS" if r["phase_type"] == "grid_search" else "GA"
		arch[cfg][phase][gt]["n"].append(r["total_neurons"])
		try:
			tiers = json.loads(r["tiers_json"])
			bpn = tiers.get("bits_per_neuron", [])
			if bpn:
				arch[cfg][phase][gt]["b"].append(statistics.mean(bpn))
		except Exception:
			pass

	def arch_str(nlist, blist):
		ns = (
			f"{int(round(statistics.mean(nlist)))}±{int(round(statistics.stdev(nlist) if len(nlist) > 1 else 0))}"
			if nlist
			else "—"
		)
		bs = (
			f"{int(round(statistics.mean(blist)))}±{int(round(statistics.stdev(blist) if len(blist) > 1 else 0))}"
			if blist
			else "—"
		)
		return f"{ns} neurons | {bs} bits"

	# Build output
	out = []
	out.append(f"# XDS-{args.cohort} — width × weight cohort breakdown ({total_completed} non-OLD completed)")
	out.append("")
	out.append(f"    Total non-OLD completed : {total_completed}  |  Total wall: {total_dur_h:.1f}h  |  Avg/run: {avg_dur:.0f}m")
	out.append(f"    Latest done : {latest_done_str}")
	out.append("")
	out.append("    Weight schemes:")
	for k in sorted(WEIGHT_DESC):
		out.append(f"      {WEIGHT_DESC[k]}")
	out.append("")

	# Iterate (width, weight, arch) in stable order
	configs = sorted(data.keys())
	for cfg in configs:
		(w, wt, ar) = cfg
		cfg_data = data[cfg]
		cfg_pts = all_pts[cfg]
		# n_flows for this cell: use distinct seeds × phases
		gs_n_any = max((len(seen_seeds[(cfg, "GS", g)]) for g in GENOMES), default=0)
		ga_n_any = max((len(seen_seeds[(cfg, "GA", g)]) for g in GENOMES), default=0)
		out.append("")
		out.append(f"## XDS-{args.cohort}-{w}b-W{wt}-{ar}  ({gs_n_any} flows × 2 phases, seeds: {sorted({s for g in GENOMES for s in seen_seeds[(cfg, 'GA', g)]})})")
		out.append("")
		out.append(f"    Weight : {WEIGHT_DESC.get(wt, '?')}  |  Arch : {ar}")
		out.append("")

		# Best individual genomes section
		def find_best(filter_fn, key=lambda x: x[0]):
			filtered = [p for p in cfg_pts if filter_fn(p)]
			return max(filtered, key=key) if filtered else None

		bf = max(cfg_pts, key=lambda x: x[0]) if cfg_pts else None
		bf_f14 = find_best(lambda p: p[1] < 14)
		bf_f10 = find_best(lambda p: p[1] < 10)
		bf_f6 = find_best(lambda p: p[1] < 6)
		bf_f5 = find_best(lambda p: p[1] < 5)
		bf_f4 = find_best(lambda p: p[1] < 4)
		bfpr = min(cfg_pts, key=lambda x: x[1]) if cfg_pts else None
		bfpr_f80 = find_best(lambda p: p[0] > 80, key=lambda x: -x[1])
		bacc = max(cfg_pts, key=lambda x: x[2]) if cfg_pts else None

		def fmt_row(label, p):
			if p is None:
				return f"    {label:<25}|       — |       — |       — | —"
			return (f"    {label:<25}| {p[0]:6.2f}% | {p[1]:6.2f}% | {p[2]:6.2f}% | "
			        f"r{p[6]} {p[3]} {p[4]:<14} {p[5]}")

		out.append("### Best individual genomes")
		out.append("")
		out.append("    Metric                   |      F1 |     FPR |     Acc | Source")
		out.append("    -------------------------+---------+---------+---------+-----------------------------------")
		out.append(fmt_row("Best F1 (any FPR)", bf))
		out.append(fmt_row("Best F1 (FPR<14%)", bf_f14))
		out.append(fmt_row("Best F1 (FPR<10%)", bf_f10))
		out.append(fmt_row("Best F1 (FPR<6%)", bf_f6))
		out.append(fmt_row("Best F1 (FPR<5%)", bf_f5))
		out.append(fmt_row("Best F1 (FPR<4%)", bf_f4))
		out.append(fmt_row("Best FPR (any F1)", bfpr))
		out.append(fmt_row("Best FPR (F1>80%)", bfpr_f80))
		out.append(fmt_row("Best Acc (any FPR)", bacc))
		out.append("")

		# 5 tables
		for gt in GENOMES:
			gs_n = len(seen_seeds[(cfg, "GS", gt)])
			ga_n = len(seen_seeds[(cfg, "GA", gt)])
			out.append(f"### {gt}  (GS: {gs_n} runs | GA: {ga_n} runs)")
			gs_neurons = arch[cfg]["GS"][gt]["n"]
			gs_bits    = arch[cfg]["GS"][gt]["b"]
			ga_neurons = arch[cfg]["GA"][gt]["n"]
			ga_bits    = arch[cfg]["GA"][gt]["b"]
			out.append(f"    Grid Search : {arch_str(gs_neurons, gs_bits)}")
			out.append(f"    GA Neurons  : {arch_str(ga_neurons, ga_bits)}")
			out.append("")
			out.append("    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA")
			out.append("    ---------------------+---------------------+---------------------+--------------------")
			for mode in MODES:
				gs_d = cfg_data["GS"][gt][mode]
				ga_d = cfg_data["GA"][gt][mode]
				out.append(
					f"    {mode:<20} |{fmt_pair(gs_d['f1'])} {fmt_pair(ga_d['f1'])} "
					f"|{fmt_pair(gs_d['fpr']):>10} {fmt_pair(ga_d['fpr']):>9} "
					f"|{fmt_pair(gs_d['acc']):>10} {fmt_pair(ga_d['acc']):>9}"
				)
			out.append("")

	con.close()
	return "\n".join(out)


if __name__ == "__main__":
	print(main())
