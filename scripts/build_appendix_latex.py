"""Generate LaTeX appendix tables for the paper's `\\appendix Full Threshold Mode Breakdown`.

Mirrors the data extraction logic from scripts/build_xds_5tables.py but emits
the LaTeX format used in paper/main.tex appendix (lines ~1475-1583):

  \\begin{table}[ht!]
  \\centering
  \\small
  \\caption{<DATASET> <PARTITION>, <GENOME_LABEL> genome (<N> runs, <BITS>-bit thermometer).
           Grid Search: <Ngs>$\\pm$<SD> neurons / <Bgs>$\\pm$<SD> bits;
           GA Neurons: <Nga>$\\pm$<SD> neurons / <Bga>$\\pm$<SD> bits.}
  \\label{tab:app-<slug>-<genome>}
  \\resizebox{\\textwidth}{!}{%
  \\begin{tabular}{l@{\\quad}r@{\\quad}r@{\\quad}r@{\\quad}r@{\\quad}r@{\\quad}r}
  \\toprule
  Threshold & F1 Grid (\\%) & F1 GA (\\%) & FPR Grid (\\%) & FPR GA (\\%) & Acc Grid (\\%) & Acc GA (\\%) \\\\
  \\midrule
  Train-cal          & <gs> & <ga> & <gs> & <ga> & <gs> & <ga> \\\\
  Fixed 0.5          & ...
  ...
  \\bottomrule
  \\end{tabular}}
  \\end{table}

Usage:
  python3 scripts/build_appendix_latex.py \\
      --flow-ids 2897,2911,2925,2978-3004 \\
      --dataset "UNSW-NB15 temporal" \\
      --slug unsw-temp \\
      --bits 16 \\
      --output paper/snippets/appendix_unsw_temp.tex

Then in main.tex, replace the 5 hand-written tables with:
  \\input{snippets/appendix_unsw_temp.tex}
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DB = Path("/Users/lacg/wnn/db/wnn.db")

GENOMES = ["best_fitness", "best_f1", "best_fpr", "best_acc", "best_ce"]
GENOME_LABEL = {
	"best_fitness": "best-fitness",
	"best_f1": "best-F1",
	"best_fpr": "best-FPR",
	"best_acc": "best-Acc",
	"best_ce": "best-CE",
}
GENOME_SLUG = {
	"best_fitness": "bestfitness",
	"best_f1": "bestf1",
	"best_fpr": "bestfpr",
	"best_acc": "bestacc",
	"best_ce": "bestce",
}
MODES = [
	("train_cal",            "Train-cal"),
	("fixed_05",             "Fixed 0.5"),
	("platt",                "Platt"),
	("beta",                 "Beta"),
	("empirical",            "Empirical"),
	("empirical_cumulative", "Empirical cumul."),
	("val_cal",              "Val\\_cal"),
]


def parse_flow_ids(spec: str) -> list[int]:
	"""Parse '2897,2911,2925,2978-3004' into a flat list of ints."""
	out = []
	for chunk in spec.split(","):
		chunk = chunk.strip()
		if "-" in chunk:
			a, b = chunk.split("-")
			out.extend(range(int(a), int(b) + 1))
		else:
			out.append(int(chunk))
	return sorted(set(out))


def fmt_pair(values: list[float]) -> str:
	"""Format as 'X.XX$\\pm$X.XX' or '---' if empty."""
	if not values:
		return "---"
	if len(values) == 1:
		return f"{values[0]:.2f}$\\pm$0.00"
	m = statistics.mean(values)
	s = statistics.stdev(values)
	return f"{m:.2f}$\\pm${s:.2f}"


def fmt_arch(nlist: list[int], blist: list[float]) -> str:
	"""Format as 'NNN$\\pm$NN neurons / BB$\\pm$B bits'."""
	if not nlist:
		return "--- neurons / --- bits"
	nm = int(round(statistics.mean(nlist)))
	ns = int(round(statistics.stdev(nlist))) if len(nlist) > 1 else 0
	bm = int(round(statistics.mean(blist))) if blist else 0
	bs = int(round(statistics.stdev(blist))) if len(blist) > 1 else 0
	return f"{nm}$\\pm${ns} neurons / {bm}$\\pm${bs} bits"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--flow-ids", required=True,
	                help="Flow IDs, comma-separated, ranges allowed (e.g. '2897,2911,2925,2978-3004')")
	ap.add_argument("--dataset", required=True,
	                help="Dataset label, e.g. 'UNSW-NB15 temporal'")
	ap.add_argument("--slug", required=True,
	                help="Label slug, e.g. 'unsw-temp' (for \\label{tab:app-<slug>-bestfitness})")
	ap.add_argument("--bits", type=int, required=True,
	                help="Thermometer bits per feature (for caption)")
	ap.add_argument("--output", default="-",
	                help="Output file path (default '-' for stdout)")
	args = ap.parse_args()

	flow_ids = parse_flow_ids(args.flow_ids)
	placeholders = ",".join(["?"] * len(flow_ids))

	con = sqlite3.connect(DB)
	con.row_factory = sqlite3.Row
	cur = con.cursor()

	# Pull validation_summaries: per-flow, per-genome_type, per-phase, per-threshold-mode → (f1, fpr, acc)
	cur.execute(
		f"""SELECT vs.flow_id, vs.genome_type, e.phase_type, vs.threshold_metadata
		FROM validation_summaries vs
		JOIN experiments e ON e.id = vs.experiment_id
		WHERE vs.flow_id IN ({placeholders})
		  AND vs.validation_point = 'final'""",
		flow_ids,
	)

	# data[genome_type][phase][mode] = {"f1": [...], "fpr": [...], "acc": [...]}
	data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"f1": [], "fpr": [], "acc": []})))
	# seen_flows[genome_type][phase] = set of flow_ids that contributed
	seen_flows: dict = defaultdict(lambda: defaultdict(set))

	for r in cur:
		gt = r["genome_type"]
		if gt not in GENOMES:
			continue
		phase = "GS" if r["phase_type"] == "grid_search" else "GA"
		seen_flows[gt][phase].add(r["flow_id"])
		tm = json.loads(r["threshold_metadata"])
		for key, _label in MODES:
			md = tm.get(key, {})
			if not isinstance(md, dict) or md.get("f1") is None:
				continue
			data[gt][phase][key]["f1"].append(md["f1"] * 100)
			data[gt][phase][key]["fpr"].append(md["fpr"] * 100)
			data[gt][phase][key]["acc"].append(md["acc"] * 100)

	# Architecture (neurons + bits) from best_genomes + genomes for the val_cal pick per genome type
	metric_to_genome = {
		"fitness": "best_fitness",
		"f1_macro": "best_f1",
		"fpr": "best_fpr",
		"accuracy": "best_acc",
		"ce": "best_ce",
	}
	cur.execute(
		f"""SELECT bg.flow_id, bg.metric, e.phase_type, g.total_neurons, g.tiers_json
		FROM best_genomes bg
		JOIN genomes g ON g.id = bg.genome_id
		JOIN experiments e ON e.id = bg.experiment_id
		WHERE bg.flow_id IN ({placeholders})
		  AND bg.threshold_mode = 'val_cal'""",
		flow_ids,
	)
	arch: dict = defaultdict(lambda: defaultdict(lambda: {"n": [], "b": []}))
	for r in cur:
		gt = metric_to_genome.get(r["metric"])
		if not gt:
			continue
		phase = "GS" if r["phase_type"] == "grid_search" else "GA"
		arch[gt][phase]["n"].append(r["total_neurons"])
		try:
			tiers = json.loads(r["tiers_json"])
			bpn = tiers.get("bits_per_neuron", [])
			if bpn:
				arch[gt][phase]["b"].append(statistics.mean(bpn))
		except Exception:
			pass

	# Emit LaTeX
	out: list[str] = []
	out.append(f"% ============================================================================")
	out.append(f"% Auto-generated by scripts/build_appendix_latex.py")
	out.append(f"% Dataset: {args.dataset}")
	out.append(f"% Slug: {args.slug}")
	out.append(f"% Thermometer bits: {args.bits}")
	out.append(f"% Flow IDs (n={len(flow_ids)}): {args.flow_ids}")
	out.append(f"% ============================================================================")
	out.append("")

	for gt in GENOMES:
		gs_n = len(seen_flows[gt]["GS"])
		ga_n = len(seen_flows[gt]["GA"])
		n_runs = max(gs_n, ga_n)
		gs_arch = fmt_arch(arch[gt]["GS"]["n"], arch[gt]["GS"]["b"])
		ga_arch = fmt_arch(arch[gt]["GA"]["n"], arch[gt]["GA"]["b"])

		out.append(r"\begin{table}[ht!]")
		out.append(r"\centering")
		out.append(r"\small")
		out.append(
			f"\\caption{{{args.dataset}, {GENOME_LABEL[gt]} genome "
			f"({n_runs} runs, {args.bits}-bit thermometer). "
			f"Grid Search: {gs_arch}; GA Neurons: {ga_arch}.}}"
		)
		out.append(f"\\label{{tab:app-{args.slug}-{GENOME_SLUG[gt]}}}")
		out.append(r"\resizebox{\textwidth}{!}{%")
		out.append(r"\begin{tabular}{l@{\quad}r@{\quad}r@{\quad}r@{\quad}r@{\quad}r@{\quad}r}")
		out.append(r"\toprule")
		out.append(
			r"Threshold & F1 Grid (\%) & F1 GA (\%) & FPR Grid (\%) & FPR GA (\%) & Acc Grid (\%) & Acc GA (\%) \\"
		)
		out.append(r"\midrule")

		for mode_key, mode_label in MODES:
			gs_d = data[gt]["GS"][mode_key]
			ga_d = data[gt]["GA"][mode_key]
			cells = [
				fmt_pair(gs_d["f1"]),  fmt_pair(ga_d["f1"]),
				fmt_pair(gs_d["fpr"]), fmt_pair(ga_d["fpr"]),
				fmt_pair(gs_d["acc"]), fmt_pair(ga_d["acc"]),
			]
			out.append(f"{mode_label:<18} & " + " & ".join(cells) + r" \\")

		out.append(r"\bottomrule")
		out.append(r"\end{tabular}}")
		out.append(r"\end{table}")
		out.append("")

	text = "\n".join(out) + "\n"
	if args.output == "-":
		sys.stdout.write(text)
	else:
		Path(args.output).parent.mkdir(parents=True, exist_ok=True)
		Path(args.output).write_text(text)
		print(f"Wrote {args.output} ({len(out)} lines)", file=sys.stderr)


if __name__ == "__main__":
	main()
