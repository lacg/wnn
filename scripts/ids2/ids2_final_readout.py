"""READ-ONLY IDS-2 FINAL readout (26/09/2026): IDSXD + IDSXD2, cicids / ciciot (+ unswr quad provisional).

Pre-declared reporting column: GA-neurons phase, val_cal, held-out TEST (validation_point='final'),
each of the 5 genome types. Seeds parsed from the flow name; `-w64fix` / `-abi13` clones EXCLUDED
(targeted repair sets, reported separately).
"""
import json, re, sqlite3, statistics as st, sys
from collections import defaultdict

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]
GTS = ["best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"]
NAME_RE = re.compile(r"^(IDSXD2?)-(unswr|cicids|ciciot)-(qsr|quad)-(\d+)b-(.+?)-r(\d+)(-w64fix|-abi13)?$")
ORFOLD_FIX = "2026-08-30T02:27"
IMMIG_FIX = "2026-09-17T01:57"

SQL = """
SELECT f.id fid, f.name, f.started_at, e.phase_type, vs.genome_type gt, vs.genome_hash gh,
       vs.experiment_id eid, vs.threshold_metadata tm
FROM validation_summaries vs
JOIN flows f ON f.id = vs.flow_id
JOIN experiments e ON e.id = vs.experiment_id
WHERE (f.name LIKE 'IDSXD-%' OR f.name LIKE 'IDSXD2-%') AND f.name NOT LIKE 'IDSXD-unswt-%'
  AND f.status = 'completed' AND vs.validation_point = 'final'
"""
con = sqlite3.connect(DB, uri=True); con.row_factory = sqlite3.Row


def shape(gh, eid):
	r = con.execute("SELECT tiers_json FROM genomes WHERE experiment_id=? AND genome_hash=? LIMIT 1", (eid, gh)).fetchone()
	if not r:
		return None
	t = json.loads(r[0]); b = t.get("bits_per_neuron") or []
	return (len(b), st.mean(b) if b else 0, max(b) if b else 0)


D = defaultdict(dict)  # D[(ds,dec,arm,phase,gt)][seed] = {mode:(f1,fpr,acc)} + meta
EXTRA = []
for r in con.execute(SQL):
	m = NAME_RE.match(r["name"])
	if not m:
		print("UNMATCHED", r["name"]); continue
	coh, ds, dec, _, arm, seed, suf = m.groups()
	if r["gt"] not in GTS:
		continue
	ph = "GS" if r["phase_type"] == "grid_search" else "GA"
	tm = json.loads(r["tm"])
	vals = {k: (tm[k]["f1"] * 100, tm[k]["fpr"] * 100, tm[k]["acc"] * 100) for k in MODES if isinstance(tm.get(k), dict) and tm[k].get("f1") is not None}
	rec = dict(vals=vals, fid=r["fid"], start=r["started_at"], gh_full=r["gh"], eid=r["eid"], gh=r["gh"][:10])
	if suf:
		EXTRA.append((ds, dec, arm, int(seed), suf, ph, r["gt"], rec)); continue
	D[(ds, dec, arm, ph, r["gt"])][int(seed)] = rec


def shp(rec):
	if "shape" not in rec:
		rec["shape"] = shape(rec["gh_full"], rec["eid"])
	return rec["shape"]


def ms(xs):
	return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


def f(x):
	return f"{x[0]:6.2f}±{x[1]:4.2f}"


def agg(ds, dec, arm, ph, gt, mode, seeds=None):
	cell = D.get((ds, dec, arm, ph, gt), {})
	ss = sorted(s for s in cell if (seeds is None or s in seeds) and mode in cell[s]["vals"])
	if not ss:
		return None
	v = [cell[s]["vals"][mode] for s in ss]
	return ss, ms([x[0] for x in v]), ms([x[1] for x in v]), ms([x[2] for x in v])


def arms_of(ds, dec):
	return sorted({k[2] for k in D if k[0] == ds and k[1] == dec})


def era(ds, arm, seed, dec):
	rec = next(iter(D[(ds, dec, arm, "GA", "best_f1")].get(seed, {}).values()), None) if False else D[(ds, dec, arm, "GA", "best_f1")].get(seed)
	s = rec["start"] if rec else ""
	a = "pre" if (ds == "ciciot" and s < ORFOLD_FIX) else "post"
	b = "old-imm" if s < IMMIG_FIX else "new-imm"
	return f"{a}/{b}"


def arm_table(ds, dec, seeds=None, title=""):
	print(f"\n--- {ds} {dec} | GA val_cal per arm x genome_type {title} ---")
	print("arm      | gt           | n | seeds                    | F1           | FPR          | Acc")
	for arm in arms_of(ds, dec):
		for gt in GTS:
			a = agg(ds, dec, arm, "GA", gt, "val_cal", seeds)
			if a:
				print(f"{arm:<8} | {gt:<12} | {len(a[0])} | {','.join(str(s)[-2:] for s in a[0]):<24} | {f(a[1])} | {f(a[2])} | {f(a[3])}")


def per_seed(ds, dec, arm, gt):
	cell = D[(ds, dec, arm, "GA", gt)]
	out = []
	for s in sorted(cell):
		v = cell[s]["vals"]["val_cal"]; sh = shp(cell[s])
		out.append(f"  s{s} flow {cell[s]['fid']} [{era(ds, arm, s, dec)}] {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}  shape {sh[0]}n x mean {sh[1]:.1f}b (max {sh[2]}) hash {cell[s]['gh']}")
	return "\n".join(out)


def pareto(ds, dec, seeds=None, fmin=None):
	"""All arms x phases x gts x 7 modes; show non-dominated (F1 up, FPR down) columns."""
	cols = []
	for arm in arms_of(ds, dec):
		for ph in ("GS", "GA"):
			for gt in GTS:
				for md in MODES:
					a = agg(ds, dec, arm, ph, gt, md, seeds)
					if a and (seeds is None or len(a[0]) == len(seeds)):
						cols.append((arm, ph, gt, md, a))
	nd = [c for c in cols if not any(o[4][1][0] >= c[4][1][0] and o[4][2][0] <= c[4][2][0] and (o[4][1][0] > c[4][1][0] or o[4][2][0] < c[4][2][0]) for o in cols)]
	nd.sort(key=lambda c: -c[4][1][0])
	print(f"\n--- {ds} {dec} PARETO (mean F1 vs mean FPR) over {len(cols)} columns, seeds={seeds or 'all'} ---")
	for arm, ph, gt, md, a in nd:
		if fmin and a[1][0] < fmin:
			continue
		print(f"{arm:<8} {ph} {gt:<12} {md:<20} n={len(a[0])} F1 {f(a[1])} FPR {f(a[2])} Acc {f(a[3])}")


def rule7(ds, dec, arm):
	print(f"\n##### RULE-7 5-tables: IDSXD(+2)-{ds}-{dec}-{arm} (Grid vs GA, held-out TEST, mean±SD %) #####")
	for gt in GTS:
		hdr = []
		for ph, lab in (("GS", "Grid Search"), ("GA", "GA Neurons ")):
			sh = [x for x in (shp(r) for r in D[(ds, dec, arm, ph, gt)].values()) if x]
			n = ms([x[0] for x in sh]); b = ms([x[1] for x in sh])
			hdr.append(f"{lab} : {n[0]:.0f}±{n[1]:.0f} neurons | {b[0]:.1f}±{b[1]:.1f} bits (mean per-neuron)")
		ng = len(D[(ds, dec, arm, "GS", gt)]); na = len(D[(ds, dec, arm, "GA", gt)])
		print(f"\n{gt}  (runs: GS {ng} | GA {na})")
		print("\n".join(hdr))
		print("mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA")
		print("---------------------+--------------+--------------+--------------+--------------+--------------+-------------")
		for md in MODES:
			g = agg(ds, dec, arm, "GS", gt, md); a = agg(ds, dec, arm, "GA", gt, md)
			print(f"{md:<20} | {f(g[1])} | {f(a[1])} | {f(g[2])} | {f(a[2])} | {f(g[3])} | {f(a[3])}")


if __name__ == "__main__":
	what = sys.argv[1]
	if what == "arms":
		ds, dec = sys.argv[2], sys.argv[3]
		seeds = set(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else None
		arm_table(ds, dec, seeds, f"seeds={sorted(seeds) if seeds else 'all'}")
	elif what == "seeds":
		print(per_seed(*sys.argv[2:6]))
	elif what == "pareto":
		seeds = set(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] != "all" else None
		pareto(sys.argv[2], sys.argv[3], seeds, float(sys.argv[5]) if len(sys.argv) > 5 else None)
	elif what == "rule7":
		rule7(*sys.argv[2:5])
	elif what == "extra":
		for ds, dec, arm, seed, suf, ph, gt, rec in sorted(EXTRA, key=lambda x: (x[0], x[2], x[3], x[5], x[6])):
			if ph == "GA":
				v = rec["vals"]["val_cal"]; sh = shp(rec)
				print(f"{ds} {dec} {arm} s{seed}{suf} flow {rec['fid']} {ph} {gt:<12} val_cal {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f} shape {sh[0]}n x {sh[1]:.1f}b max {sh[2]}")
