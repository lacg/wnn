"""Per-dataset / per-config IDS leaderboard: best F1, best FPR, best Acc as TRIPLES.

Every "best" is printed with the other two metrics beside it, plus the identity
(seed / phase / genome_type / threshold mode) that produced it, plus n (distinct
completed runs in that config cell) so best-of-N inflation is visible.

Also prints the honest cohort-central number for the same cell: GA x val_cal
mean±std over runs. best-of-N minus mean is the winner's-curse gap.

All numbers: validation_summaries.threshold_metadata, validation_point='final'
= HELD-OUT report partition. Never iterations.best_f1.

Usage:
  python3 scripts/build_ids_leaderboard.py --prefix 'SP100-%' 'SP-%' 'XDS-%'
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict

DB_URI = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
GENOMES = ["best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"]
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]
SEED_RE = re.compile(r"-r(\d+)$")

DATASET_LABEL = {
	("unsw-nb15", "temporal_3way"): "UNSW-NB15 temporal (3way, Protocol v2)",
	("unsw-nb15", "random_3way"): "UNSW-NB15 random (3way, Protocol v2)",
	("unsw-nb15", "temporal"): "UNSW-NB15 temporal (2way, LEGACY)",
	("unsw-nb15", "random"): "UNSW-NB15 random (2way, LEGACY)",
	("cicids2017", "random_3way"): "CICIDS2017 random (3way, Protocol v2)",
	("cicids2017", "random"): "CICIDS2017 random (2way, LEGACY)",
	("ciciot2023_neto_subsample", "random_3way"): "CIC-IoT-2023 neto-subsample random (3way, Protocol v2)",
	("ciciot2023_neto_subsample", "random"): "CIC-IoT-2023 neto-subsample random (2way, LEGACY)",
}


def fam_of(name):
	m = SEED_RE.search(name)
	return (name[: m.start()], int(m.group(1))) if m else (name, 0)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--prefix", nargs="+", required=True)
	ap.add_argument("--rollup", action="store_true",
	                help="Emit the per-DATASET roll-up (best point across all configs in that dataset).")
	ap.add_argument("--fitness-delta", action="store_true",
	                help="Emit the best_fitness Grid-vs-GA delta table instead of the leaderboard.")
	args = ap.parse_args()

	con = sqlite3.connect(DB_URI, uri=True)
	con.row_factory = sqlite3.Row
	cur = con.cursor()

	fam_ds = {}
	fam_seeds = defaultdict(set)
	pts = defaultdict(list)
	ga_valcal = defaultdict(lambda: defaultdict(list))  # fam -> genome_type -> f1 list
	ga_valcal_fpr = defaultdict(lambda: defaultdict(list))
	# fam -> phase -> genome_type -> {f1,fpr,acc} at val_cal (held-out, Protocol v2 val-calibrated)
	pv = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"f1": [], "fpr": [], "acc": []})))

	for like in args.prefix:
		cur.execute("SELECT id, name, config_json FROM flows WHERE name LIKE ? AND status='completed'", (like,))
		meta = {}
		for r in cur.fetchall():
			fam, seed = fam_of(r["name"])
			p = json.loads(r["config_json"]).get("params", {})
			fam_ds.setdefault(fam, (p.get("ids_dataset"), p.get("ids_split"),
			                        p.get("ids_n_bits"), p.get("memory_mode") or "QUAD",
			                        p.get("max_neurons"), p.get("max_bits"),
			                        p.get("ids_classification")))
			fam_seeds[fam].add(seed)
			meta[r["id"]] = None

		cur.execute(
			"""SELECT f.name, vs.genome_type, e.phase_type, vs.threshold_metadata
			FROM validation_summaries vs
			JOIN flows f ON f.id=vs.flow_id
			JOIN experiments e ON e.id=vs.experiment_id
			WHERE f.name LIKE ? AND f.status='completed' AND vs.validation_point='final'""",
			(like,))
		for r in cur.fetchall():
			fam, seed = fam_of(r["name"])
			gt = r["genome_type"]
			if gt not in GENOMES or not r["threshold_metadata"]:
				continue
			phase = "GS" if r["phase_type"] == "grid_search" else "GA"
			tm = json.loads(r["threshold_metadata"])
			for mode in MODES:
				md = tm.get(mode, {})
				if not isinstance(md, dict):
					continue
				f1 = md.get("f1", md.get("f1_macro"))
				if f1 is None or md.get("fpr") is None or md.get("acc") is None:
					continue
				trip = (f1 * 100, md["fpr"] * 100, md["acc"] * 100, phase, gt, mode, seed)
				pts[fam].append(trip)
				if mode == "val_cal":
					pv[fam][phase][gt]["f1"].append(f1 * 100)
					pv[fam][phase][gt]["fpr"].append(md["fpr"] * 100)
					pv[fam][phase][gt]["acc"].append(md["acc"] * 100)
				if phase == "GA" and mode == "val_cal":
					ga_valcal[fam][gt].append(f1 * 100)
					ga_valcal_fpr[fam][gt].append(md["fpr"] * 100)

	if args.fitness_delta:
		out = []
		out.append("    Genome type best_fitness, threshold mode val_cal, HELD-OUT report partition.")
		out.append("    Delta = GA Neurons minus Grid Search (positive F1/Acc = GA better; negative FPR = GA better).")
		out.append("")
		out.append("    config                                     |  n | F1 Grid    F1 GA     dF1  | FPR Grid   FPR GA    dFPR | Acc Grid   Acc GA    dAcc")
		out.append("    -------------------------------------------+----+---------------------------+---------------------------+--------------------------")
		for fam in sorted(pv.keys()):
			g = pv[fam]["GS"]["best_fitness"]
			a = pv[fam]["GA"]["best_fitness"]
			if not g["f1"] or not a["f1"]:
				continue
			def ms(v):
				s = statistics.stdev(v) if len(v) > 1 else 0.0
				return statistics.mean(v), s
			cells = []
			for k in ("f1", "fpr", "acc"):
				gm, gs = ms(g[k])
				am, asd = ms(a[k])
				cells.append(f"{gm:5.2f}±{gs:4.2f} {am:5.2f}±{asd:4.2f} {am-gm:+6.2f}")
			out.append(f"    {fam:<42} | {len(a['f1']):>2} | {cells[0]} | {cells[1]} | {cells[2]}")
		con.close()
		return "\n".join(out)

	# group families by dataset label
	by_ds = defaultdict(list)
	for fam, ds in fam_ds.items():
		by_ds[DATASET_LABEL.get((ds[0], ds[1]), f"{ds[0]} / {ds[1]}")].append(fam)

	if args.rollup:
		out = []
		out.append("    Best POINT per dataset across every config in that dataset. n = runs in the")
		out.append("    winning config. Read as a ceiling, not as the claim (see COHORT rows in 3A/3B).")
		out.append("")
		out.append("    dataset                                            | best      |     F1 |    FPR |    Acc |  n | winning config / source")
		out.append("    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------")
		for ds_label in sorted(by_ds):
			P = [p + (fam,) for fam in by_ds[ds_label] for p in pts[fam]]
			if not P:
				continue
			s5 = [p for p in P if p[1] < 5]
			s4 = [p for p in P if p[1] < 4]
			rows = [("F1", max(P, key=lambda x: x[0])),
			        ("FPR", min(P, key=lambda x: (x[1], -x[0]))),
			        ("Acc", max(P, key=lambda x: x[2]))]
			if s5:
				rows.append(("F1|FPR<5", max(s5, key=lambda x: x[0])))
			if s4:
				rows.append(("F1|FPR<4", max(s4, key=lambda x: x[0])))
			for i, (lbl, p) in enumerate(rows):
				fam = p[7]
				name = ds_label if i == 0 else ""
				out.append(f"    {name:<50} | {lbl:<9} | {p[0]:6.2f} | {p[1]:6.2f} | {p[2]:6.2f} | "
				           f"{len(fam_seeds[fam]):>2} | {fam} r{p[6]} {p[3]} {p[4]} {p[5]}")
			out.append(f"    {'-'*50}-+-----------+--------+--------+--------+----+------------------------")
		con.close()
		return "\n".join(out)

	out = []
	for ds_label in sorted(by_ds):
		out.append("")
		out.append(f"## {ds_label}")
		out.append("")
		out.append("    config                                     |  n | best      |     F1 |    FPR |    Acc | source (seed phase genome mode)")
		out.append("    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------")
		for fam in sorted(by_ds[ds_label]):
			cp = pts[fam]
			n = len(fam_seeds[fam])
			if not cp:
				out.append(f"    {fam:<42} | {n:>2} | (no final validation_summaries rows)")
				continue
			bf1 = max(cp, key=lambda x: x[0])
			bfpr = min(cp, key=lambda x: (x[1], -x[0]))
			bacc = max(cp, key=lambda x: x[2])
			sub5 = [p for p in cp if p[1] < 5]
			sub4 = [p for p in cp if p[1] < 4]
			sub2 = [p for p in cp if p[1] < 2]
			rows = [("F1", bf1), ("FPR", bfpr), ("Acc", bacc)]
			if sub5:
				rows.append(("F1|FPR<5", max(sub5, key=lambda x: x[0])))
			if sub4:
				rows.append(("F1|FPR<4", max(sub4, key=lambda x: x[0])))
			if sub2:
				rows.append(("F1|FPR<2", max(sub2, key=lambda x: x[0])))
			flag = "  <-- n<5: winner's-curse territory" if n < 5 else ""
			for i, (lbl, p) in enumerate(rows):
				name_col = fam if i == 0 else ""
				n_col = f"{n:>2}" if i == 0 else "  "
				out.append(f"    {name_col:<42} | {n_col} | {lbl:<9} | {p[0]:6.2f} | {p[1]:6.2f} | {p[2]:6.2f} | "
				           f"r{p[6]} {p[3]} {p[4]} {p[5]}" + (flag if i == 0 else ""))
			# honest central number
			f1s = ga_valcal[fam].get("best_f1", [])
			fprs = ga_valcal_fpr[fam].get("best_f1", [])
			if f1s:
				sd = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
				sdf = statistics.stdev(fprs) if len(fprs) > 1 else 0.0
				out.append(f"    {'':<42} |    | COHORT    | {statistics.mean(f1s):6.2f} | {statistics.mean(fprs):6.2f} |    --- | "
				           f"GA best_f1 val_cal mean±std over n={len(f1s)}: F1 ±{sd:.2f}, FPR ±{sdf:.2f}")
			out.append(f"    {'-'*42}-+----+-----------+--------+--------+--------+---------------------------------")
	con.close()
	return "\n".join(out)


if __name__ == "__main__":
	print(main())
