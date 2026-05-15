"""Generate the 5-table format report for C35 PUB112 (CIC-IoT-2023 neto-sub).

Matches the format of PUB50-ciciot-random in docs/ids_results.md:
  - Top header with Completed/total, durations, ETA
  - Best individual genomes section
  - 5 tables, one per genome_type (best_fitness, best_f1, best_fpr, best_acc, best_ce)
  - Each table: Grid Search vs GA Neurons side-by-side, 7 threshold modes
"""

import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

DB = Path("/Users/lacg/wnn/db/wnn.db")
# CLI args: --250n100b for new architecture, --fixed for u32-fix cohort,
# default for regular C35
import sys
if "--fixed" in sys.argv:
	PATTERN = "WSWEEP-T20-96b-C35-250n100b-FIXED-r%"
	TARGET = 112
	TITLE_SUFFIX = "250n×100b FIXED (canonical TOP20, post-u32-fix)"
elif "--250n100b" in sys.argv:
	PATTERN = "WSWEEP-96b-C35-250n100b-r%"
	TARGET = 112
	TITLE_SUFFIX = "250n×100b (new winner architecture)"
else:
	PATTERN = "WSWEEP-96b-C35-r%"
	TARGET = 112
	TITLE_SUFFIX = "500n×34b (PUB112 baseline)"
GENOMES = ["best_fitness", "best_f1", "best_fpr", "best_acc", "best_ce"]
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]


def fmt_pair(values):
	if not values:
		return "—"
	if len(values) == 1:
		return f"{values[0]:.2f}"
	return f"{statistics.mean(values):>5.2f}±{statistics.stdev(values):.2f}"


def main():
	con = sqlite3.connect(str(DB))
	con.row_factory = sqlite3.Row
	cur = con.cursor()

	# Regular C35 must exclude 250n100b / 800n40b / 1000n50b variants
	if PATTERN == "WSWEEP-96b-C35-r%":
		count_sql = "SELECT COUNT(*) FROM flows WHERE name LIKE ? AND name NOT LIKE '%n40b%' AND name NOT LIKE '%n50b%' AND name NOT LIKE '%n100b%' AND status='completed'"
		durs_sql = """SELECT (julianday(completed_at)-julianday(started_at))*1440 AS m, completed_at
			FROM flows WHERE name LIKE ? AND name NOT LIKE '%n40b%' AND name NOT LIKE '%n50b%' AND name NOT LIKE '%n100b%' AND status='completed' ORDER BY completed_at"""
	else:
		count_sql = "SELECT COUNT(*) FROM flows WHERE name LIKE ? AND status='completed'"
		durs_sql = """SELECT (julianday(completed_at)-julianday(started_at))*1440 AS m, completed_at
			FROM flows WHERE name LIKE ? AND status='completed' ORDER BY completed_at"""
	cur.execute(count_sql, (PATTERN,))
	completed = cur.fetchone()[0]
	cur.execute(durs_sql, (PATTERN,))
	rows = list(cur)
	durs = [r["m"] for r in rows if r["m"] is not None]
	avg_dur = statistics.mean(durs) if durs else 0
	total_dur_h = sum(durs) / 60 if durs else 0
	latest_done = rows[-1]["completed_at"] if rows else None

	# Compute ETA
	remaining = TARGET - completed
	eta_str_utc = eta_str_et = "n/a"
	if latest_done and avg_dur > 0:
		latest_dt = datetime.fromisoformat(latest_done.replace("Z", "+00:00"))
		eta_dt_utc = latest_dt + timedelta(minutes=remaining * avg_dur)
		# ET = UTC - 4 in EDT
		from datetime import timezone

		eta_dt_et = eta_dt_utc - timedelta(hours=4)
		eta_str_utc = eta_dt_utc.strftime("%d/%m/%Y %H:%M") + " UTC"
		eta_str_et = eta_dt_et.strftime("%d/%m/%Y %H:%M") + " ET"
		latest_done_str = latest_dt.strftime("%d/%m/%Y %H:%M") + " UTC"
	else:
		latest_done_str = "—"

	# Pull all phase × genome × threshold data
	cur.execute(
		"""SELECT f.name, vs.genome_type, e.phase_type, vs.threshold_metadata
		FROM validation_summaries vs
		JOIN flows f ON f.id=vs.flow_id
		JOIN experiments e ON e.id=vs.experiment_id
		WHERE f.name LIKE ? AND f.status='completed' AND vs.validation_point='final'""",
		(PATTERN,),
	)

	data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"f1": [], "fpr": [], "acc": []})))
	all_pts = []

	for r in cur:
		phase = "GS" if r["phase_type"] == "grid_search" else "GA"
		gt = r["genome_type"]
		tm = json.loads(r["threshold_metadata"])
		for mode in MODES:
			md = tm.get(mode, {})
			if isinstance(md, dict) and md.get("f1") is not None:
				f1, fpr, acc = md["f1"] * 100, md["fpr"] * 100, md["acc"] * 100
				data[phase][gt][mode]["f1"].append(f1)
				data[phase][gt][mode]["fpr"].append(fpr)
				data[phase][gt][mode]["acc"].append(acc)
				if f1 >= 80:
					all_pts.append((f1, fpr, acc, phase, gt, mode))

	# Architecture: query best_genomes joined with genomes for total_neurons + tiers_json
	# Each best_genome row maps to a (flow, genome_type, phase, threshold_mode). We want
	# avg neurons & bits per (phase, genome_type) across all flows.
	arch = defaultdict(lambda: defaultdict(lambda: {"n": [], "b": []}))
	cur.execute(
		"""SELECT bg.metric, e.phase_type, g.total_neurons, g.tiers_json
		FROM best_genomes bg
		JOIN genomes g ON g.id = bg.genome_id
		JOIN flows f ON f.id = bg.flow_id
		JOIN experiments e ON e.id = bg.experiment_id
		WHERE f.name LIKE ? AND f.status='completed'
		AND bg.threshold_mode='train_cal'""",
		(PATTERN,),
	)
	# Map metric→genome_type label (best_genomes uses just 'fitness', 'f1', etc; we want 'best_X')
	metric_to_genome = {"fitness": "best_fitness", "f1_macro": "best_f1", "fpr": "best_fpr",
	                    "accuracy": "best_acc", "ce": "best_ce"}
	for r in cur:
		gt = metric_to_genome.get(r["metric"])
		if not gt:
			continue
		phase = "GS" if r["phase_type"] == "grid_search" else "GA"
		arch[phase][gt]["n"].append(r["total_neurons"])
		# Compute bits = avg bits_per_neuron from tiers_json
		try:
			tiers = json.loads(r["tiers_json"])
			bpn = tiers.get("bits_per_neuron", [])
			if bpn:
				arch[phase][gt]["b"].append(statistics.mean(bpn))
		except Exception:
			pass

	# Build report
	out = []
	out.append(f"# C35 PUB{TARGET} — CIC-IoT-2023 neto-sub Random 96b {TITLE_SUFFIX} — Live Tracking ({completed}/{TARGET} runs)")
	out.append("")
	out.append(f"    Completed : {completed}/{TARGET}  |  Total wall: {total_dur_h:.1f}h  |  Avg/run: {avg_dur:.0f}m")
	out.append(f"    Latest done : {latest_done_str}")
	out.append(f"    ETA : {eta_str_utc} / {eta_str_et}")
	out.append("")

	# Best individual genomes
	out.append("### Best individual genomes")
	out.append("")
	out.append("    Metric                   |      F1 |     FPR |     Acc | Phase Genome        Threshold       |")
	out.append("    -------------------------+---------+---------+---------+-----------------------------------+")

	def find_best(filter_fn, key=lambda x: x[0]):
		filtered = [p for p in all_pts if filter_fn(p)]
		return max(filtered, key=key) if filtered else None

	bf = max(all_pts, key=lambda x: x[0]) if all_pts else None
	bf_f14 = find_best(lambda p: p[1] < 14)
	bf_f10 = find_best(lambda p: p[1] < 10)
	bf_f6 = find_best(lambda p: p[1] < 6)
	bf_f5 = find_best(lambda p: p[1] < 5)
	bf_f4 = find_best(lambda p: p[1] < 4)
	bfpr = min(all_pts, key=lambda x: x[1]) if all_pts else None
	bfpr_f80 = find_best(lambda p: p[0] > 80, key=lambda x: -x[1])  # lowest FPR with F1>80
	bacc = max(all_pts, key=lambda x: x[2]) if all_pts else None

	def fmt_row(label, p):
		if p is None:
			return f"    {label:<25}|       — |       — |       — | —                                  |"
		return f"    {label:<25}|  {p[0]:.2f}% |  {p[1]:.2f}% |  {p[2]:.2f}% |  {p[3]} {p[4]:<14} {p[5]:<22}|"

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
		gs_runs = len(set([f"{p[3]}-{p[5]}" for p in all_pts if p[4] == gt and p[3] == "GS"])) if False else None
		# count distinct flows per phase × genome
		# Use len(data[phase][gt][train_cal]['f1']) as proxy
		gs_n = len(data["GS"][gt]["train_cal"]["f1"])
		ga_n = len(data["GA"][gt]["train_cal"]["f1"])

		out.append(f"## {gt}  (GS: {gs_n} runs | GA: {ga_n} runs)")

		gs_neurons = arch["GS"][gt]["n"]
		gs_bits = arch["GS"][gt]["b"]
		ga_neurons = arch["GA"][gt]["n"]
		ga_bits = arch["GA"][gt]["b"]

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

		out.append(f"    Grid Search : {arch_str(gs_neurons, gs_bits)}")
		out.append(f"    GA Neurons  : {arch_str(ga_neurons, ga_bits)}")
		out.append("")
		out.append("    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA")
		out.append("    ---------------------+---------------------+---------------------+--------------------")

		for mode in MODES:
			gs_d = data["GS"][gt][mode]
			ga_d = data["GA"][gt][mode]
			f1_gs = fmt_pair(gs_d["f1"])
			f1_ga = fmt_pair(ga_d["f1"])
			fpr_gs = fmt_pair(gs_d["fpr"])
			fpr_ga = fmt_pair(ga_d["fpr"])
			acc_gs = fmt_pair(gs_d["acc"])
			acc_ga = fmt_pair(ga_d["acc"])
			out.append(
				f"    {mode:<20} |{f1_gs} {f1_ga} |{fpr_gs:>10} {fpr_ga:>9} |{acc_gs:>10} {acc_ga:>9}"
			)

		out.append("")

	con.close()
	return "\n".join(out)


if __name__ == "__main__":
	print(main())
