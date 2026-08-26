#!/usr/bin/env python3
"""Regenerate docs/multiclass_baselines/README.md from the measured *.json legs.

The JSONs are the truth (written by run_multiclass_baselines.py); this only
renders them. Re-run after any leg lands or is re-measured — never hand-edit
the README, it is overwritten.
"""

import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent.parent / "docs" / "multiclass_baselines"

# Render order: the screening dataset first, then by ascending scale.
ORDER = [
	"unsw-nb15_temporal_3way",
	"unsw-nb15_random_3way",
	"cicids2017_random_3way",
	"cicids2017_temporal_3way",
	"ciciot2023_neto_subsample_random_3way",
	"ciciot2023_neto_full_random_3way",
]

# Legs measured but NOT comparable, with the reason. They are rendered in their
# own section so a reader cannot pick them up out of the at-a-glance table by
# accident. Keep the evidence in the reason — "excluded" without a measurement
# is just an assertion.
EXCLUDED = {
	"cicids2017_temporal_3way": (
		"**Degenerate protocol — the attack classes are time-disjoint.** CICIDS2017 is "
		"captured Monday-Friday with different attacks each day, so a chronological cut "
		"splits the classes rather than sampling them. Measured train/test label sets do "
		"not intersect on a single attack: train carries BENIGN + the DoS family + "
		"Patators + Web attacks + Infiltration + Heartbleed (12 labels); test carries only "
		"BENIGN, PortScan (79,318), DDoS (64,135) and Bot (940) — none of which appear in "
		"train. Every attack in test is unseen by construction, so the classifier predicts "
		"BENIGN for everything: benign recall 1.000, every attack recall 0.000. This is a "
		"zero-shot task, not a hard supervised one. Use `random_3way` for CICIDS multiclass; "
		"the binary numbers are unusable for the same reason."
	),
}

SUMMARY_COLS = f"| {'dataset':<30} | {'K':>2} | {'test rows':>11} | {'RF macro-F1':>11} | {'XGB macro-F1':>12} | {'RF benFPR':>9} | {'XGB benFPR':>10} |"


def leg_key(leg: dict) -> str:
	"""The ORDER/EXCLUDED key for a loaded leg (dataset + split)."""
	return f"{leg['dataset']}_{leg['split']}"


def load_legs() -> list[dict]:
	"""Load every leg named in ORDER that actually exists on disk."""
	legs = []
	for name in ORDER:
		path = BASE / f"{name}.json"
		if path.exists():
			legs.append(json.load(path.open()))
	return legs


def summary_row(leg: dict) -> str:
	"""One line of the at-a-glance table: the bar each dataset sets."""
	rf, xgb = leg["models"]["rf_multi"], leg["models"]["xgb_multi"]
	label = f"{leg['dataset']} [{leg['split'].replace('_3way','')}]"
	return (
		f"| {label:<30} | {len(leg['classes']):>2} | {leg['rows']['test']:>11,} "
		f"| {rf['macro_f1']*100:>11.2f} | {xgb['macro_f1']*100:>12.2f} "
		f"| {rf['benign_fpr']*100:>9.2f} | {xgb['benign_fpr']*100:>10.2f} |"
	)


def metrics_block(leg: dict) -> list[str]:
	"""Binary-vs-multiclass headline metrics for both models."""
	head = f"{'model':<6}{'binF1':>8}{'binFPR':>9}{'macroF1':>10}{'weightF1':>10}{'acc':>8}{'benFPR':>9}{'fit_s':>9}"
	lines = ["```", head, "-" * len(head)]
	for key in ("rf", "xgb"):
		b, m = leg["models"][f"{key}_binary"], leg["models"][f"{key}_multi"]
		lines.append(
			f"{key.upper():<6}{b['f1']*100:>8.2f}{b['fpr']*100:>9.2f}"
			f"{m['macro_f1']*100:>10.2f}{m['weighted_f1']*100:>10.2f}"
			f"{m['accuracy']*100:>8.2f}{m['benign_fpr']*100:>9.2f}"
			f"{b['fit_seconds'] + m['fit_seconds']:>9.1f}"
		)
	return lines + ["```"]


def recall_block(leg: dict) -> list[str]:
	"""Per-class recall, rarest class first — where multiclass actually fails."""
	rf, xgb = leg["models"]["rf_multi"], leg["models"]["xgb_multi"]
	support = rf["support"]
	head = f"{'class':<28}{'support':>11}{'RF rec':>9}{'XGB rec':>9}"
	lines = ["```", head, "-" * len(head)]
	for cls in sorted(support, key=lambda c: support[c]):
		lines.append(
			f"{cls:<28}{support[cls]:>11,}"
			f"{rf['per_class_recall'].get(cls, 0)*100:>9.1f}"
			f"{xgb['per_class_recall'].get(cls, 0)*100:>9.1f}"
		)
	return lines + ["```"]


def leg_section(leg: dict) -> list[str]:
	"""Full per-dataset section: heading, metrics, per-class recall."""
	rows = leg["rows"]
	lines = [
		f"### {leg['dataset']} — `{leg['split']}`",
		"",
		f"{len(leg['classes'])} classes · train {rows['train']:,} · test {rows['test']:,} "
		f"· features `{leg['feature_selection']}`",
		"",
	]
	lines += metrics_block(leg) + [""] + recall_block(leg) + [""]
	return lines


def build(legs: list[dict]) -> str:
	"""Assemble the whole README."""
	out = [
		"# RF / XGBoost multiclass baselines — the bar for the MCS arms",
		"",
		"**Generated** by `scripts/build_multiclass_baseline_table.py` from the",
		"`*.json` legs in this directory. Do NOT hand-edit — re-run the script.",
		"Measurements produced by `scripts/run_multiclass_baselines.py` on the same",
		"3-way splits and `top20` feature selection the WNN screening arms use.",
		"",
		"## Why this exists",
		"",
		"Reviewer A asked for multiclass; it had been deferred as future work. These",
		"are the no-box numbers a WNN multiclass arm has to be read against. **The bar",
		"is per-dataset macro-F1, not the binary F1** — binary looks near-solved on",
		"every dataset here while macro-F1 drops as far as 51 points below it.",
		"",
		"## At a glance",
		"",
		SUMMARY_COLS,
		"|" + "|".join(["-" * 32, "-" * 4, "-" * 13, "-" * 13, "-" * 14, "-" * 11, "-" * 12]) + "|",
	]
	comparable = [l for l in legs if leg_key(l) not in EXCLUDED]
	excluded = [l for l in legs if leg_key(l) in EXCLUDED]
	out += [summary_row(leg) for leg in comparable]
	out += ["", "## Per dataset", ""]
	for leg in comparable:
		out += leg_section(leg)
	if excluded:
		out += ["## Excluded protocols (measured, NOT comparable)", ""]
		for leg in excluded:
			out += [EXCLUDED[leg_key(leg)], ""] + leg_section(leg)
	return "\n".join(out) + "\n"


def main() -> None:
	legs = load_legs()
	if not legs:
		raise SystemExit(f"no baseline JSONs found in {BASE}")
	target = BASE / "README.md"
	target.write_text(build(legs))
	print(f"wrote {target} from {len(legs)} legs")


if __name__ == "__main__":
	main()
