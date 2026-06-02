"""Rank IDS arch/thermo/scheme configs on the ISO-FPR (dominating-genome) metric.

Why this exists (02/06/2026): on UNSW-random the calibrated thresholds
(train_cal/val_cal) drive every genome to ~100% attack recall, where the data
pins a fixed false-positive floor (~3,414 hard normals) → ALL configs collapse
to F1 93.5 / FPR 1.12 / Acc 98.92. So val_cal CANNOT discriminate architectures,
and fixed_0.5 discriminates but at a worse-than-deployable point. The only thing
that distinguishes archs is **how low an FPR they reach at ~full recall** — i.e.
the best F1 among held-out genomes at/below a target FPR. This tool ranks on that.

Metric per flow (held-out, GA phase):
  iso_fpr_F1  = max F1_macro among genomes with FPR <= --fpr-target (default 1%)
  iso_fpr_FPR = that genome's FPR
  min_FPR     = lowest FPR any genome reached (context)
  winner n×b  = that genome's neurons + per-neuron bit range (FPGA footprint)
  duration    = wall-clock

Usage:
  python scripts/compare_ids_archs.py [--pattern 'XDS-unsw-random%'] [--fpr-target 0.01]
"""

import argparse
import json
import sqlite3
import re

DB = "file:db/wnn.db?mode=ro"


def arch_bits(tiers_json):
	"""Return 'Nn b[lo-hi]' from a genome's tiers_json (heterogeneous per-neuron bits)."""
	if not tiers_json:
		return "—"
	try:
		t = json.loads(tiers_json)
		n = (t.get("neurons_per_cluster") or [t.get("neurons", "?")])[0]
		b = t.get("bits_per_neuron")
		if b:
			return f"{n}n b[{min(b)}-{max(b)}]"
		return f"{n}n"
	except Exception:
		m = re.search(r"neurons=(\d+)", tiers_json or "")
		return f"{m.group(1)}n" if m else "—"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pattern", default="XDS-unsw-random%", help="flow name LIKE pattern")
	ap.add_argument("--fpr-target", type=float, default=0.01, help="iso-FPR target (default 0.01 = 1%%)")
	args = ap.parse_args()
	db = sqlite3.connect(DB, uri=True)

	flows = db.execute(
		"SELECT id, name, config_json, started_at, completed_at FROM flows "
		"WHERE name LIKE ? AND completed_at IS NOT NULL ORDER BY id", (args.pattern,)).fetchall()

	rows = []
	for fid, name, cfg, started, completed in flows:
		p = json.loads(cfg).get("params", {})
		scheme = "Wa" if "-Wa-" in name else "Wb" if "-Wb-" in name else "Wc" if "-Wc-" in name else "?"
		thermo = p.get("ids_n_bits")
		arch = f"{p.get('max_neurons')}n{p.get('max_bits')}b"
		seed = p.get("seed")
		mins = db.execute(
			"SELECT CAST((julianday(?)-julianday(?))*24*60 AS INT)", (completed, started)).fetchone()[0]
		# iso-FPR genome: max F1 among GA-phase held-out genomes with FPR <= target
		iso = db.execute(
			"SELECT bg.f1_macro, bg.fpr, bg.genome_id FROM best_genomes bg "
			"JOIN experiments e ON e.id=bg.experiment_id "
			"WHERE bg.flow_id=? AND e.name LIKE 'GA%' AND bg.fpr <= ? AND bg.f1_macro > 0.6 "
			"ORDER BY bg.f1_macro DESC LIMIT 1", (fid, args.fpr_target)).fetchone()
		minfpr = db.execute(
			"SELECT MIN(bg.fpr) FROM best_genomes bg JOIN experiments e ON e.id=bg.experiment_id "
			"WHERE bg.flow_id=? AND e.name LIKE 'GA%' AND bg.f1_macro > 0.6", (fid,)).fetchone()[0]
		nb = "—"
		iso_f1 = iso_fpr = None
		if iso:
			iso_f1, iso_fpr, gid = iso
			tj = db.execute("SELECT tiers_json FROM genomes WHERE id=?", (gid,)).fetchone()
			nb = arch_bits(tj[0]) if tj else "—"
		rows.append({
			"id": fid, "scheme": scheme, "thermo": thermo, "arch": arch, "seed": seed,
			"iso_f1": iso_f1, "iso_fpr": iso_fpr, "min_fpr": minfpr, "nb": nb, "mins": mins})

	# Rank by iso-FPR F1 desc (the dominating-genome metric); Nones last.
	rows.sort(key=lambda r: (r["iso_f1"] is None, -(r["iso_f1"] or 0)))

	print(f"\nIDS arch comparison — iso-FPR @ ≤{args.fpr_target*100:.1f}% FPR (max F1 at deployable operating point)")
	print(f"pattern={args.pattern}  ({len(rows)} completed flows)\n")
	hdr = f"{'flow':>5} {'scheme':>6} {'thermo':>6} {'arch':>10} {'seed':>7} | {'iso-F1':>7} {'@FPR':>6} {'minFPR':>7} | {'winner n×b':>16} | {'dur':>5}"
	print(hdr); print("-" * len(hdr))
	for r in rows:
		f1 = f"{r['iso_f1']*100:.2f}" if r["iso_f1"] is not None else "  —  "
		fp = f"{r['iso_fpr']*100:.2f}" if r["iso_fpr"] is not None else "  — "
		mf = f"{r['min_fpr']*100:.2f}" if r["min_fpr"] is not None else "  — "
		print(f"{r['id']:>5} {r['scheme']:>6} {str(r['thermo'])+'b':>6} {r['arch']:>10} {str(r['seed']):>7} | "
		      f"{f1:>7} {fp:>6} {mf:>7} | {r['nb']:>16} | {r['mins']}m")


if __name__ == "__main__":
	main()
