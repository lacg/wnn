"""Generate a camera-ready draft markdown comparing r125+r124 (canonical-neto)
to r98 (paper baseline) and to RF/XGBoost baselines.

Pulls:
  - validation_summaries from DB for r98 (1156), r125 (1687), r124 (1686)
  - Per-class output files from /Users/lacg/wnn/analysis/per_class_flow*.md
  - Latest baseline log from /Users/lacg/wnn/logs/canonical_baselines_perclass_*.log

Writes:
  /Users/lacg/wnn/analysis/camera_ready_draft_<timestamp>.md

Usage:
    python scripts/draft_camera_ready_update.py

Designed to be invoked by auto_per_class_when_r124_done.py after both
per-class analyses complete.
"""

import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path("/Users/lacg/wnn/db/wnn.db")
ANALYSIS_DIR = Path("/Users/lacg/wnn/analysis")
LOGS_DIR = Path("/Users/lacg/wnn/logs")

R98_FLOW = 1156
R125_FLOW = 1687
R124_FLOW = 1686


def get_best_summaries(flow_id: int, threshold_mode: str = "train_cal") -> dict:
	"""Return per-genome-type best metrics for a flow at given threshold mode."""
	con = sqlite3.connect(str(DB_PATH))
	out = {}
	for metric in ["f1_macro", "fpr", "accuracy", "ce", "fitness"]:
		row = con.execute(
			"""
			SELECT bg.f1_macro, bg.fpr, bg.accuracy, bg.ce
			FROM best_genomes bg
			WHERE bg.flow_id = ? AND bg.metric = ? AND bg.threshold_mode = ?
			ORDER BY bg.rank ASC LIMIT 1
			""",
			(flow_id, metric, threshold_mode),
		).fetchone()
		if row:
			f1, fpr, acc, ce = row
			out[metric] = {"f1": f1, "fpr": fpr, "acc": acc, "ce": ce}
	con.close()
	return out


def get_flow_name(flow_id: int) -> str:
	con = sqlite3.connect(str(DB_PATH))
	row = con.execute("SELECT name FROM flows WHERE id = ?", (flow_id,)).fetchone()
	con.close()
	return row[0] if row else f"flow_{flow_id}"


def parse_per_class_md(path: Path) -> dict:
	"""Parse the per-class markdown file emitted by per_class_analysis.py.

	Expected format: a table with columns Class | Count | <metric> (recall%) ...
	Returns: {metric_name: {class_name: rate_pct}}
	"""
	if not path.exists():
		return {}
	text = path.read_text()
	# Find header line with metric column names
	m = re.search(r"^\|\s*Class\s*\|\s*Count\s*\|(.+)\|$", text, re.MULTILINE)
	if not m:
		return {}
	# Metric column names from header
	metric_cols = [c.strip().replace(" (recall%)", "") for c in m.group(1).split("|") if c.strip()]
	# Parse data rows
	results = {m: {} for m in metric_cols}
	for line in text.splitlines():
		if not line.startswith("| ") or "Class" in line or "---" in line:
			continue
		parts = [p.strip() for p in line.strip("|").split("|")]
		if len(parts) < 2 + len(metric_cols):
			continue
		cls = parts[0]
		# parts[1] = count, then metric values
		for i, m in enumerate(metric_cols):
			val_str = parts[2 + i]
			if val_str in ("—", "-", ""): continue
			try:
				results[m][cls] = float(val_str)
			except ValueError:
				pass
	return results


def parse_baseline_log(log_path: Path) -> dict:
	"""Parse the per-class baseline log to extract RF + XGB per-class rates."""
	if not log_path.exists():
		return {}
	text = log_path.read_text()
	results = {}
	current_model = None
	for line in text.splitlines():
		if "Training RF" in line:
			current_model = "RF"; results[current_model] = {"per_class": {}}
		elif "Training XGBoost" in line:
			current_model = "XGB"; results[current_model] = {"per_class": {}}
		elif current_model:
			# Aggregate metrics line: "    F1:  92.49%  |  FPR: 13.66%  |  Acc: 99.31%"
			m = re.match(r"\s+F1:\s+([\d.]+)%\s+\|\s+FPR:\s+([\d.]+)%\s+\|\s+Acc:\s+([\d.]+)%", line)
			if m:
				results[current_model]["f1"] = float(m.group(1))
				results[current_model]["fpr"] = float(m.group(2))
				results[current_model]["acc"] = float(m.group(3))
			# Per-class line: "      DDoS           : recall = 100.00%  (...)"
			m = re.match(r"\s+(\w[\w-]*)\s*:\s*\w+\s*=\s*([\d.]+)%", line)
			if m:
				results[current_model]["per_class"][m.group(1)] = float(m.group(2))
	return results


def find_latest(pattern: str, dir: Path) -> Path | None:
	matches = sorted(dir.glob(pattern), key=lambda p: p.stat().st_mtime)
	return matches[-1] if matches else None


def main():
	stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
	out_path = ANALYSIS_DIR / f"camera_ready_draft_{stamp}.md"
	ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

	# Pull WNN summaries for all 3 flows
	print(f"Reading WNN best-genome summaries from DB...")
	r98 = get_best_summaries(R98_FLOW)
	r125 = get_best_summaries(R125_FLOW)
	r124 = get_best_summaries(R124_FLOW)

	# Pull per-class results from latest analysis files
	r125_pc_path = find_latest(f"per_class_flow{R125_FLOW}_*.md", ANALYSIS_DIR)
	r124_pc_path = find_latest(f"per_class_flow{R124_FLOW}_*.md", ANALYSIS_DIR)
	r125_pc = parse_per_class_md(r125_pc_path) if r125_pc_path else {}
	r124_pc = parse_per_class_md(r124_pc_path) if r124_pc_path else {}

	# Pull baseline per-class
	bl_log = find_latest("canonical_baselines_perclass_*.log", LOGS_DIR)
	bl_pc = parse_baseline_log(bl_log) if bl_log else {}

	# Compose markdown
	lines = []
	lines.append(f"# Camera-ready draft update (generated {stamp})")
	lines.append("")
	lines.append("## Headline numbers comparison (train_cal threshold)")
	lines.append("")
	lines.append("| Flow | Best F1 | Best FPR | Best Acc |")
	lines.append("|---|---|---|---|")
	for fid, lbl, bs in [(R98_FLOW, "r98 (paper, 38.5M)", r98),
						 (R125_FLOW, "r125 (canonical, 45M)", r125),
						 (R124_FLOW, "r124 (canonical, 45M)", r124)]:
		if not bs: continue
		f1 = bs.get("f1_macro", {})
		fpr = bs.get("fpr", {})
		acc = bs.get("accuracy", {})
		def fmt(d): return f"{d['f1']*100:.2f}/{d['fpr']*100:.2f}/{d['acc']*100:.2f}" if d else "—"
		lines.append(f"| {lbl} | {fmt(f1)} | {fmt(fpr)} | {fmt(acc)} |")
	lines.append("")
	lines.append("_Format: F1% / FPR% / Acc%. Each cell is the genome optimal for that metric._")
	lines.append("")

	# Baselines on canonical-neto
	if bl_pc:
		lines.append("## Classical ML baselines on canonical-neto (45M)")
		lines.append("")
		lines.append("| Model | F1 | FPR | Acc |")
		lines.append("|---|---|---|---|")
		for m in ["RF", "XGB"]:
			if m in bl_pc:
				b = bl_pc[m]
				lines.append(f"| {m} | {b.get('f1', '—')}% | {b.get('fpr', '—')}% | {b.get('acc', '—')}% |")
		lines.append("")

	# Per-class table — merge baseline + r125 + r124
	all_classes = set()
	if bl_pc:
		for m in bl_pc.values(): all_classes.update(m.get("per_class", {}).keys())
	for r in [r125_pc, r124_pc]:
		for m in r.values(): all_classes.update(m.keys())
	# Order: Benign first
	if all_classes:
		ordered = (["Benign"] if "Benign" in all_classes else []) + sorted(c for c in all_classes if c != "Benign")
		lines.append("## Per-class breakdown")
		lines.append("")
		hdr_cols = []
		if bl_pc.get("RF"): hdr_cols.append("RF")
		if bl_pc.get("XGB"): hdr_cols.append("XGB")
		if r125_pc: hdr_cols.append("WNN r125 best_f1")
		if r124_pc: hdr_cols.append("WNN r124 best_f1")
		lines.append("| Class | " + " | ".join(hdr_cols) + " |")
		lines.append("|---|" + "|".join(["---" for _ in hdr_cols]) + "|")
		for cls in ordered:
			row = [cls]
			if bl_pc.get("RF"):
				v = bl_pc["RF"]["per_class"].get(cls)
				row.append(f"{v:.2f}%" if v is not None else "—")
			if bl_pc.get("XGB"):
				v = bl_pc["XGB"]["per_class"].get(cls)
				row.append(f"{v:.2f}%" if v is not None else "—")
			if r125_pc:
				v = next((m.get(cls) for m in r125_pc.values() if cls in m), None)
				row.append(f"{v:.2f}%" if v is not None else "—")
			if r124_pc:
				v = next((m.get(cls) for m in r124_pc.values() if cls in m), None)
				row.append(f"{v:.2f}%" if v is not None else "—")
			lines.append("| " + " | ".join(row) + " |")
		lines.append("")
		lines.append("_Benign row is FPR (false alarms); attack rows are recall (detection rate)._")
		lines.append("")

	# Provenance
	lines.append("---")
	lines.append("## Sources")
	lines.append(f"- DB: `{DB_PATH}` (best_genomes table for flows {R98_FLOW}, {R125_FLOW}, {R124_FLOW})")
	if r125_pc_path: lines.append(f"- r125 per-class: `{r125_pc_path}`")
	if r124_pc_path: lines.append(f"- r124 per-class: `{r124_pc_path}`")
	if bl_log: lines.append(f"- Baselines log: `{bl_log}`")

	out_path.write_text("\n".join(lines))
	print(f"✓ Wrote draft to {out_path}")
	print()
	print("\n".join(lines[:30]))  # preview


if __name__ == "__main__":
	main()
