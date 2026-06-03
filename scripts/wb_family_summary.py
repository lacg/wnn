#!/usr/bin/env python3
"""Wb-family duration + winner-genome + FPGA-deployability summary.

Aggregates the UNSW-random 500n34b OI Wb cohort (the active paper cohort) across
seeds, grouped by thermometer width. Reports duration mean±std, winner-genome
neuron count mean±std + bit-range, and a Z-7020 deployability verdict.

FPGA model (docs/paper_pareto_classes.md): Z-7020 = 762 KB total internal memory
(625 KB BRAM + 137 KB LUTRAM). A 5n×16b winner ≈ <4% → 99%+ headroom (~27 instances).
Deployability here = sparse-footprint proxy from winner neuron count × bit-width;
the 211×64b headline genome is the documented non-fit reference.
"""
import json, sqlite3, re, statistics as st

DB = "file:db/wnn.db?mode=ro"
FPR_TARGET = 0.01
# Canonical active cohort: UNSW-random, 500n34b, OI, Wb, exclude OLD/250n100b.
PATTERN = "XDS-unsw-random%Wb-C35-500n34b-OI-r%"


def winner_nb(tiers_json):
	"""(neurons:int, bit_lo:int, bit_hi:int) from a genome's tiers_json, or None."""
	if not tiers_json:
		return None
	try:
		t = json.loads(tiers_json)
		n = (t.get("neurons_per_cluster") or [t.get("neurons")])[0]
		b = t.get("bits_per_neuron")
		if n is None or not b:
			return None
		return int(n), int(min(b)), int(max(b))
	except Exception:
		return None


def main():
	db = sqlite3.connect(DB, uri=True)
	flows = db.execute(
		"SELECT id, name, config_json, started_at, completed_at FROM flows "
		"WHERE name LIKE ? AND status='completed' AND completed_at IS NOT NULL "
		"AND name NOT LIKE '%OLD%' ORDER BY id", (PATTERN,)).fetchall()

	by_width = {}  # thermo -> list of dict(seed, mins, n, blo, bhi, iso_f1)
	for fid, name, cfg, started, completed in flows:
		p = json.loads(cfg).get("params", {})
		thermo = p.get("ids_n_bits")
		seed = p.get("seed")
		mins = db.execute(
			"SELECT CAST((julianday(?)-julianday(?))*24*60 AS INT)", (completed, started)).fetchone()[0]
		iso = db.execute(
			"SELECT bg.f1_macro, bg.fpr, bg.accuracy, bg.genome_id FROM best_genomes bg "
			"JOIN experiments e ON e.id=bg.experiment_id "
			"WHERE bg.flow_id=? AND e.name LIKE 'GA%' AND bg.fpr <= ? AND bg.f1_macro > 0.6 "
			"ORDER BY bg.f1_macro DESC LIMIT 1", (fid, FPR_TARGET)).fetchone()
		nb = None
		iso_f1 = iso_fpr = iso_acc = None
		if iso:
			iso_f1, iso_fpr, iso_acc, gid = iso
			tj = db.execute("SELECT tiers_json FROM genomes WHERE id=?", (gid,)).fetchone()
			nb = winner_nb(tj[0]) if tj else None
		rec = {"fid": fid, "seed": seed, "mins": mins,
		       "iso_f1": iso_f1, "iso_fpr": iso_fpr, "iso_acc": iso_acc,
		       "n": nb[0] if nb else None, "blo": nb[1] if nb else None, "bhi": nb[2] if nb else None}
		by_width.setdefault(thermo, []).append(rec)

	def ms(vals, dec=0):
		vals = [v for v in vals if v is not None]
		if not vals:
			return "—"
		if len(vals) == 1:
			return f"{vals[0]:.{dec}f}"
		return f"{st.mean(vals):.{dec}f}±{st.pstdev(vals):.{dec}f}"

	print(f"\nWb family — UNSW-random 500n34b OI cohort (pattern={PATTERN})")
	print(f"grouped by thermometer width; mean±std (pop) across seeds; metrics = iso-FPR winner @≤{FPR_TARGET*100:.0f}% FPR\n")
	hdr = (f"{'thermo':>6} {'seeds':>5} | {'F1 %':>9} {'FPR %':>9} {'Acc %':>9} | "
	       f"{'dur min':>9} | {'neurons':>9} {'bit range':>11} | {'Z-7020':>8}")
	print(hdr); print("-"*len(hdr))
	allmins = []
	for thermo in sorted(by_width, key=lambda x: (x is None, x)):
		recs = by_width[thermo]
		allmins += [r["mins"] for r in recs if r["mins"] is not None]
		ns = [r["n"] for r in recs]
		blos = [r["blo"] for r in recs if r["blo"] is not None]
		bhis = [r["bhi"] for r in recs if r["bhi"] is not None]
		f1s = [r["iso_f1"]*100 for r in recs if r["iso_f1"] is not None]
		fprs = [r["iso_fpr"]*100 for r in recs if r["iso_fpr"] is not None]
		accs = [r["iso_acc"]*100 for r in recs if r["iso_acc"] is not None]
		nmax = max([n for n in ns if n is not None], default=0)
		brange = f"[{min(blos)}-{max(bhis)}]" if blos else "—"
		fit = "✓ ample" if (nmax and nmax <= 250 and (max(bhis) if bhis else 99) <= 34) else "review"
		print(f"{str(thermo):>6} {len(recs):>5} | {ms(f1s,1):>9} {ms(fprs,2):>9} {ms(accs,1):>9} | "
		      f"{ms([r['mins'] for r in recs]):>9} | {ms(ns):>9} {brange:>11} | {fit:>8}")
	print("-"*len(hdr))
	print(f"\nflows: " + ", ".join(f"{t}b×{len(by_width[t])}" for t in sorted(by_width, key=lambda x:(x is None,x))))
	print("metrics at the iso-FPR deployable operating point (max F1 among GA held-out genomes with FPR≤1%)")


if __name__ == "__main__":
	main()
