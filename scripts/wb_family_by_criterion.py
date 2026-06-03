#!/usr/bin/env python3
"""Wb-family per-criterion held-out breakdown — PROVENANCE-EXPLICIT.

For the UNSW-random 500n34b OI Wb cohort, for each thermometer width, show the
best_f1 / best_fpr / best_acc selected genomes: their architecture (neurons +
bit-range) and held-out F1/FPR/Acc, averaged mean±std across seeds.

SOURCE (audited): validation_summaries.validation_point='final' = the HELD-OUT 20%
re-evaluation, joined to the GA-phase experiment (phase_type='ga_neurons') and to
genomes.tiers_json via genome_hash. This is NOT the iterations table (K-fold
during-search fitness). Run with --show-sql to print the exact query.
"""
import json, sqlite3, re, statistics as st, sys

DB = "file:db/wnn.db?mode=ro"
PATTERN = "XDS-unsw-random%Wb-C35-500n34b-OI-r%"
CRITERIA = ["best_f1", "best_fpr", "best_acc"]


def bit_range(tiers_json):
	try:
		b = json.loads(tiers_json).get("bits_per_neuron")
		return (min(b), max(b)) if b else None
	except Exception:
		return None


def ms(vals, dec=0):
	vals = [v for v in vals if v is not None]
	if not vals:
		return "—"
	if len(vals) == 1:
		return f"{vals[0]:.{dec}f}"
	return f"{st.mean(vals):.{dec}f}±{st.pstdev(vals):.{dec}f}"


def main():
	if "--show-sql" in sys.argv:
		print("Source = validation_summaries (validation_point='final', held-out 20%)")
		print("Join   = experiments(phase_type='ga_neurons') + genomes ON genome_hash")
		print("NOT    = iterations (that table holds during-search K-fold fitness)\n")

	db = sqlite3.connect(DB, uri=True)
	flows = db.execute(
		"SELECT id, name, config_json FROM flows "
		"WHERE name LIKE ? AND status='completed' AND name NOT LIKE '%OLD%' ORDER BY id",
		(PATTERN,)).fetchall()

	# data[thermo][genome_type] -> list of dict
	data = {}
	for fid, name, cfg in flows:
		p = json.loads(cfg).get("params", {})
		thermo = p.get("ids_n_bits")
		rows = db.execute(
			"SELECT vs.genome_type, g.total_neurons, g.tiers_json, vs.f1_macro, vs.fpr, vs.accuracy "
			"FROM validation_summaries vs "
			"JOIN experiments e ON e.id = vs.experiment_id "
			"JOIN genomes g ON g.genome_hash = vs.genome_hash "
			"WHERE vs.flow_id=? AND vs.validation_point='final' AND e.phase_type='ga_neurons' "
			"AND vs.genome_type IN ('best_f1','best_fpr','best_acc')", (fid,)).fetchall()
		for gt, neurons, tiers, f1, fpr, acc in rows:
			br = bit_range(tiers)
			data.setdefault(thermo, {}).setdefault(gt, []).append(
				{"n": neurons, "blo": br[0] if br else None, "bhi": br[1] if br else None,
				 "f1": f1*100 if f1 is not None else None,
				 "fpr": fpr*100 if fpr is not None else None,
				 "acc": acc*100 if acc is not None else None})

	print(f"Wb family per-criterion — UNSW-random 500n34b OI (held-out final, GA phase)")
	print(f"pattern={PATTERN}; mean±std across seeds\n")
	hdr = (f"{'thermo':>6} {'criterion':>10} {'seeds':>5} | {'F1 %':>9} {'FPR %':>9} {'Acc %':>9} | "
	       f"{'neurons':>9} {'bit range':>11}")
	print(hdr); print("-"*len(hdr))
	for thermo in sorted(data, key=lambda x: (x is None, x)):
		for gt in CRITERIA:
			recs = data[thermo].get(gt, [])
			if not recs:
				continue
			blos = [r["blo"] for r in recs if r["blo"] is not None]
			bhis = [r["bhi"] for r in recs if r["bhi"] is not None]
			brange = f"[{min(blos)}-{max(bhis)}]" if blos else "—"
			print(f"{str(thermo):>6} {gt:>10} {len(recs):>5} | "
			      f"{ms([r['f1'] for r in recs],1):>9} {ms([r['fpr'] for r in recs],2):>9} "
			      f"{ms([r['acc'] for r in recs],1):>9} | "
			      f"{ms([r['n'] for r in recs]):>9} {brange:>11}")
		print("-"*len(hdr))


if __name__ == "__main__":
	main()
