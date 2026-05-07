"""Build the WSWEEP weight-sweep report — appends to ids_results.md.

Covers all WSWEEP-96b-* variants and CE10 baseline (BITSWEEP-neto-sub-96b-*)
with deployable corner stats and per-flow listings.
"""

import json
import sqlite3
import statistics
from pathlib import Path

DB_PATH = Path("/Users/lacg/wnn/db/wnn.db")

# (label, pattern, ce, acc, f1, fpr, status_note)
CONFIGS = [
	("CE10 (baseline)", "BITSWEEP-neto-sub-96b-r%", 0.10, 0.20, 0.35, 0.35, ""),
	("CE20 (historical)", "WSWEEP-96b-CE20-r%", 0.20, 0.10, 0.30, 0.40, "paused"),
	("F1H", "WSWEEP-96b-F1H-r%", 0.10, 0.10, 0.50, 0.30, "paused"),
	("FPRH", "WSWEEP-96b-FPRH-r%", 0.10, 0.10, 0.30, 0.50, "paused"),
	("CE40", "WSWEEP-96b-CE40-r%", 0.40, 0.00, 0.30, 0.30, "paused"),
	("DDR", "WSWEEP-96b-DDR-r%", 0.30, 0.30, 0.25, 0.15, ""),
	("C35", "WSWEEP-96b-C35-r%", 0.35, 0.30, 0.30, 0.05, ""),
	("F40", "WSWEEP-96b-F40-r%", 0.30, 0.20, 0.40, 0.10, ""),
	("TRI (paused)", "WSWEEP-96b-TRI-r%", 0.30, 0.30, 0.30, 0.10, "paused"),
	("B05-CE", "WSWEEP-96b-B05-CE-r%", 0.675, 0.225, 0.05, 0.05, "paused"),
	("B05-AC", "WSWEEP-96b-B05-AC-r%", 0.225, 0.675, 0.05, 0.05, "n=1"),
	("B10-CE", "WSWEEP-96b-B10-CE-r%", 0.60, 0.20, 0.10, 0.10, "paused"),
	("B10-AC", "WSWEEP-96b-B10-AC-r%", 0.20, 0.60, 0.10, 0.10, "n=1"),
	("B15-CE", "WSWEEP-96b-B15-CE-r%", 0.525, 0.175, 0.15, 0.15, "paused"),
	("B15-AC", "WSWEEP-96b-B15-AC-r%", 0.175, 0.525, 0.15, 0.15, "n=1"),
	("CE20rev", "WSWEEP-96b-CE20rev-r%", 0.20, 0.20, 0.40, 0.20, "extending"),
	("CE20bal", "WSWEEP-96b-CE20bal-r%", 0.20, 0.15, 0.325, 0.325, "n=1"),
	("F35-CE30", "WSWEEP-96b-F35-CE30-r%", 0.30, 0.20, 0.35, 0.15, "n=1"),
	("F35-CE35", "WSWEEP-96b-F35-CE35-r%", 0.35, 0.20, 0.35, 0.10, "n=1"),
	("F35-CE25-A25", "WSWEEP-96b-F35-CE25-A25-r%", 0.25, 0.25, 0.35, 0.15, "n=1"),
	("F2R-low-eq", "WSWEEP-96b-F2R-low-eq-r%", 0.425, 0.425, 0.10, 0.05, ""),
	("F2R-mid-eq", "WSWEEP-96b-F2R-mid-eq-r%", 0.35, 0.35, 0.20, 0.10, "n=1"),
	("F2R-high-eq", "WSWEEP-96b-F2R-high-eq-r%", 0.275, 0.275, 0.30, 0.15, "n=1"),
]


def stats_pattern(cur, pat, mode="train_cal"):
	cur.execute(
		"""SELECT vs.threshold_metadata FROM validation_summaries vs
		JOIN flows f ON f.id=vs.flow_id JOIN experiments e ON e.id=vs.experiment_id
		WHERE f.name LIKE ? AND f.status='completed' AND vs.validation_point='final'
		AND e.phase_type='ga_neurons' AND vs.genome_type='best_fitness'""",
		(pat,),
	)
	F, P, A = [], [], []
	for r in cur:
		md = json.loads(r["threshold_metadata"]).get(mode, {})
		if isinstance(md, dict) and md.get("f1") is not None:
			F.append(md["f1"] * 100)
			P.append(md["fpr"] * 100)
			A.append(md["acc"] * 100)
	return F, P, A


def deployable_counts(cur, pat):
	cur.execute(
		"""SELECT vs.threshold_metadata FROM validation_summaries vs
		JOIN flows f ON f.id=vs.flow_id JOIN experiments e ON e.id=vs.experiment_id
		WHERE f.name LIKE ? AND f.status='completed' AND vs.validation_point='final'""",
		(pat,),
	)
	sub5 = sub6_F90 = sub6_F905 = 0
	min_fpr = 999.0
	best5 = best_dep = None
	for r in cur:
		tm = json.loads(r["threshold_metadata"])
		for mode in ["train_cal", "fixed_05", "val_cal", "platt", "beta", "empirical", "empirical_cumulative"]:
			md = tm.get(mode, {})
			if isinstance(md, dict) and md.get("f1") is not None and md["f1"] * 100 >= 88:
				f1, fpr, acc = md["f1"] * 100, md["fpr"] * 100, md["acc"] * 100
				if fpr < min_fpr:
					min_fpr = fpr
				if fpr < 5 and f1 >= 90:
					sub5 += 1
					if best5 is None or f1 > best5[0]:
						best5 = (f1, fpr, acc)
				if fpr < 6 and f1 >= 90:
					sub6_F90 += 1
				if fpr < 6 and f1 >= 90.5:
					sub6_F905 += 1
					if best_dep is None or f1 > best_dep[0]:
						best_dep = (f1, fpr, acc)
	return sub5, sub6_F90, sub6_F905, min_fpr if min_fpr < 999 else None, best5, best_dep


def fmt(v):
	if not v:
		return "—"
	return f"{statistics.mean(v):.2f}±{statistics.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def main():
	conn = sqlite3.connect(str(DB_PATH))
	conn.row_factory = sqlite3.Row
	cur = conn.cursor()

	out = []
	out.append("# WSWEEP 96b — Weight-Variant Sweep on neto-sub (1.43M, 46f)")
	out.append("")
	out.append("All variants tested at 96b on the same neto-subsample with K-fold=5 during GA search.")
	out.append("Compares against CE10 baseline (the deployable sub-5% champion).")
	out.append("")
	out.append("## Summary table — best_fitness GA at train_cal")
	out.append("")
	out.append("    Variant                | Weights (ce/acc/f1/fpr) | n | F1 mean        | FPR mean       | Acc mean       | sub-5%(F1≥90) | sub-6%(F1≥90.5) | Lowest FPR (F1≥88)")
	out.append("    -----------------------+-------------------------+---+----------------+----------------+----------------+---------------+-----------------+--------------------")

	for label, pat, ce, acc_w, f1_w, fpr_w, note in CONFIGS:
		F, P, A = stats_pattern(cur, pat)
		s5, s6_90, s6_905, mf, best5, best_dep = deployable_counts(cur, pat)
		n = len(F)
		if n == 0:
			continue
		weights = f"{ce}/{acc_w}/{f1_w}/{fpr_w}"
		mfs = f"{mf:.2f}" if mf is not None else "—"
		out.append(
			f"    {label:<22} | {weights:<23} | {n} | {fmt(F):<14} | {fmt(P):<14} | {fmt(A):<14} | {s5:<13} | {s6_905:<15} | {mfs}"
		)

	out.append("")
	out.append("## Best deployable points per variant (F1≥88, sorted by FPR ascending)")
	out.append("")

	for label, pat, ce, acc_w, f1_w, fpr_w, note in CONFIGS:
		s5, s6_90, s6_905, mf, best5, best_dep = deployable_counts(cur, pat)
		if best5:
			out.append(
				f"    {label:<22}: ★sub-5% champion: F1={best5[0]:.2f} / FPR={best5[1]:.2f} / Acc={best5[2]:.2f}"
			)
		elif best_dep:
			out.append(
				f"    {label:<22}: best F1≥90.5: F1={best_dep[0]:.2f} / FPR={best_dep[1]:.2f} / Acc={best_dep[2]:.2f}"
			)

	out.append("")
	conn.close()
	return "\n".join(out)


if __name__ == "__main__":
	print(main())
