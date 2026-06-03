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
			"SELECT bg.f1_macro, bg.genome_id FROM best_genomes bg "
			"JOIN experiments e ON e.id=bg.experiment_id "
			"WHERE bg.flow_id=? AND e.name LIKE 'GA%' AND bg.fpr <= ? AND bg.f1_macro > 0.6 "
			"ORDER BY bg.f1_macro DESC LIMIT 1", (fid, FPR_TARGET)).fetchone()
		nb = None
		iso_f1 = None
		if iso:
			iso_f1, gid = iso
			tj = db.execute("SELECT tiers_json FROM genomes WHERE id=?", (gid,)).fetchone()
			nb = winner_nb(tj[0]) if tj else None
		rec = {"fid": fid, "seed": seed, "mins": mins, "iso_f1": iso_f1,
		       "n": nb[0] if nb else None, "blo": nb[1] if nb else None, "bhi": nb[2] if nb else None}
		by_width.setdefault(thermo, []).append(rec)

	def ms(vals):
		vals = [v for v in vals if v is not None]
		if not vals:
			return "—"
		if len(vals) == 1:
			return f"{vals[0]:.0f}"
		return f"{st.mean(vals):.0f}±{st.pstdev(vals):.0f}"

	print(f"\nWb family — UNSW-random 500n34b OI cohort (pattern={PATTERN})")
	print(f"grouped by thermometer width; mean±std (pop) across seeds; iso-FPR winner @≤{FPR_TARGET*100:.0f}% FPR\n")
	hdr = (f"{'thermo':>6} {'seeds':>5} | {'duration min':>14} | {'winner neurons':>15} "
	       f"{'bit range':>12} | {'iso-F1 %':>9} | {'Z-7020 fit':>11}")
	print(hdr); print("-"*len(hdr))
	allmins = []
	for thermo in sorted(by_width, key=lambda x: (x is None, x)):
		recs = by_width[thermo]
		allmins += [r["mins"] for r in recs if r["mins"] is not None]
		ns = [r["n"] for r in recs]
		blos = [r["blo"] for r in recs if r["blo"] is not None]
		bhis = [r["bhi"] for r in recs if r["bhi"] is not None]
		f1s = [r["iso_f1"]*100 for r in recs if r["iso_f1"] is not None]
		nmax = max([n for n in ns if n is not None], default=0)
		brange = f"[{min(blos)}-{max(bhis)}]" if blos else "—"
		# Deployability proxy: max winner ≤ ~250n and bit_hi ≤ ~34 fits Z-7020 w/ headroom.
		fit = "✓ ample" if (nmax and nmax <= 250 and (max(bhis) if bhis else 99) <= 34) else "review"
		print(f"{str(thermo):>6} {len(recs):>5} | {ms([r['mins'] for r in recs]):>14} | "
		      f"{ms(ns):>15} {brange:>12} | {ms(f1s):>9} | {fit:>11}")
	print("-"*len(hdr))
	if allmins:
		print(f"{'ALL':>6} {len(allmins):>5} | {ms(allmins):>14} | "
		      f"{'(per-width above)':>15} {'':>12} | {'':>9} | {'':>11}")
	print(f"\nflows: " + ", ".join(f"{t}b×{len(by_width[t])}" for t in sorted(by_width, key=lambda x:(x is None,x))))


if __name__ == "__main__":
	main()
