"""READ-ONLY readout: best reporting column per dataset in the IDSXD sweep.

Column = (arm, mode, phase, genome_type, thr_mode); aggregated mean±SD over completed seeds
on HELD-OUT threshold_metadata (validation_point='final').
"""
import json
import re
import sqlite3
import statistics
from collections import defaultdict

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]
GENOMES = ["best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"]
NAME_RE = re.compile(r"^IDSXD2?-(unswr|cicids|ciciot)-(qsr|quad)-(\d+)b-([A-Za-z0-9]+-?[A-Z]*)-r(\d+)$")
FIX_UTC = "2026-08-30T02:27"

SQL = """
SELECT f.name, f.started_at, f.completed_at, e.phase_type, vs.genome_type, vs.threshold_metadata
FROM validation_summaries vs
JOIN flows f ON f.id = vs.flow_id
JOIN experiments e ON e.id = vs.experiment_id
WHERE (f.name LIKE 'IDSXD-%' OR f.name LIKE 'IDSXD2-%')
  AND f.name NOT LIKE 'IDSXD-unswt-%'
  AND f.status = 'completed'
  AND vs.validation_point = 'final'
"""

con = sqlite3.connect(DB, uri=True)
con.row_factory = sqlite3.Row
rows = list(con.execute(SQL))

# data[ds][(arm, mode, phase, gt, thr)] -> {seed: (f1, fpr, acc)}
data = defaultdict(lambda: defaultdict(dict))
era = defaultdict(lambda: defaultdict(dict))  # era[ds][arm][seed] = 'pre'/'post'
unmatched = set()
for r in rows:
	m = NAME_RE.match(r["name"])
	if not m:
		unmatched.add(r["name"])
		continue
	ds, mode, bits, arm, seed = m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
	phase = "GS" if r["phase_type"] == "grid_search" else "GA"
	gt = r["genome_type"]
	if gt not in GENOMES:
		continue
	era[ds][arm][seed] = "pre" if (r["started_at"] or "") < FIX_UTC else "post"
	tm = json.loads(r["threshold_metadata"])
	for thr in MODES:
		md = tm.get(thr, {})
		if not isinstance(md, dict) or md.get("f1") is None:
			continue
		data[ds][(arm, mode, phase, gt, thr)][seed] = (md["f1"] * 100, md["fpr"] * 100, md["acc"] * 100)

print("unmatched names:", sorted(unmatched))


def agg(vals):
	n = len(vals)
	mean = statistics.mean(vals)
	sd = statistics.stdev(vals) if n > 1 else 0.0
	return mean, sd


def fmt(mean, sd, n):
	return f"{mean:6.2f}±{sd:4.2f}" if n > 1 else f"{mean:6.2f}     "


for ds in ["unswr", "cicids", "ciciot"]:
	cols = []
	for key, seeds in data[ds].items():
		f1s = [v[0] for v in seeds.values()]
		fprs = [v[1] for v in seeds.values()]
		accs = [v[2] for v in seeds.values()]
		f1m, f1s_ = agg(f1s)
		fpm, fps_ = agg(fprs)
		acm, acs_ = agg(accs)
		arm = key[0]
		eras = era[ds][arm]
		npre = sum(1 for s in seeds if eras.get(s) == "pre")
		cols.append((key, len(seeds), f1m, f1s_, fpm, fps_, acm, acs_, npre))
	cols.sort(key=lambda c: -c[2])
	print(f"\n=== {ds} : {len(cols)} columns, held-out final threshold_metadata ===")
	print("era per arm/seed:", {a: dict(sorted(s.items())) for a, s in sorted(era[ds].items())})
	hdr = "rank | arm      | mode | ph | genome_type  | thr_mode             | n | pre | F1 mean±SD   | FPR mean±SD  | Acc mean±SD"
	print(hdr)
	print("-" * len(hdr))

	def line(i, c):
		key, n, f1m, f1s_, fpm, fps_, acm, acs_, npre = c
		arm, mode, ph, gt, thr = key
		return (f"{i:>4} | {arm:<8} | {mode:<4} | {ph:<2} | {gt:<12} | {thr:<20} | {n} | {npre:>3} | "
		        f"{fmt(f1m, f1s_, n)} | {fmt(fpm, fps_, n)} | {fmt(acm, acs_, n)}")

	for i, c in enumerate(cols[:10], 1):
		print(line(i, c))
	best = cols[0]
	thresh = best[2] - best[3]
	within = [c for c in cols if c[2] >= thresh]
	pareto = min(within, key=lambda c: c[4])
	print(f"  within-1SD-of-best set: {len(within)} columns (F1 >= {thresh:.2f})")
	print("  PARETO (min FPR within 1 SD of best F1):")
	print(line(cols.index(pareto) + 1, pareto))
	# era-clean ranking for ciciot: exclude columns with any pre-fix seed
	if ds == "ciciot":
		clean = [c for c in cols if c[8] == 0]
		print("  --- era-clean (0 pre-fix seeds) top 10 ---")
		for i, c in enumerate(clean[:10], 1):
			print(line(i, c))
		b = clean[0]
		w = [c for c in clean if c[2] >= b[2] - b[3]]
		p = min(w, key=lambda c: c[4])
		print(f"  era-clean within-1SD set: {len(w)}; PARETO:")
		print(line(clean.index(p) + 1, p))
	# Best FPR overall (any F1 >= 80) for reference
	lowfpr = min((c for c in cols if c[2] >= 80), key=lambda c: c[4])
	print("  lowest-FPR column with F1>=80 (reference):")
	print(line(cols.index(lowfpr) + 1, lowfpr))

# ---- supplementary views ----
print("\n\n######## SUPPLEMENTARY: per-arm best column (any phase/genome/thr) and val_cal-only top 5 ########")
for ds in ["unswr", "cicids", "ciciot"]:
	cols = []
	for key, seeds in data[ds].items():
		f1s = [v[0] for v in seeds.values()]; fprs = [v[1] for v in seeds.values()]; accs = [v[2] for v in seeds.values()]
		f1m, f1s_ = agg(f1s); fpm, fps_ = agg(fprs); acm, acs_ = agg(accs)
		npre = sum(1 for s in seeds if era[ds][key[0]].get(s) == "pre")
		cols.append((key, len(seeds), f1m, f1s_, fpm, fps_, acm, acs_, npre))
	cols.sort(key=lambda c: -c[2])
	best = cols[0]
	arms_within = sorted({c[0][0] for c in cols if c[2] >= best[2] - best[3]})
	print(f"\n=== {ds} — per-arm best column by mean F1 ===   (arms within 1 SD of global best: {arms_within})")
	hdr = "arm      | mode | ph | genome_type  | thr_mode             | n | pre | F1 mean±SD   | FPR mean±SD  | Acc mean±SD"
	print(hdr); print("-" * len(hdr))
	seen = set()
	for c in cols:
		arm = c[0][0]
		if arm in seen: continue
		seen.add(arm)
		key, n, f1m, f1s_, fpm, fps_, acm, acs_, npre = c
		print(f"{key[0]:<8} | {key[1]:<4} | {key[2]:<2} | {key[3]:<12} | {key[4]:<20} | {n} | {npre:>3} | {fmt(f1m,f1s_,n)} | {fmt(fpm,fps_,n)} | {fmt(acm,acs_,n)}")
	print(f"--- {ds} val_cal-only top 5 (Protocol v2 reporting mode) ---")
	vc = [c for c in cols if c[0][4] == "val_cal"]
	for c in vc[:5]:
		key, n, f1m, f1s_, fpm, fps_, acm, acs_, npre = c
		print(f"{key[0]:<8} | {key[1]:<4} | {key[2]:<2} | {key[3]:<12} | {key[4]:<20} | {n} | {npre:>3} | {fmt(f1m,f1s_,n)} | {fmt(fpm,fps_,n)} | {fmt(acm,acs_,n)}")
	b = vc[0]; w = [c for c in vc if c[2] >= b[2] - b[3]]; p = min(w, key=lambda c: c[4])
	key, n, f1m, f1s_, fpm, fps_, acm, acs_, npre = p
	print(f"val_cal PARETO (min FPR within 1SD): {key[0]:<8} | {key[1]:<4} | {key[2]:<2} | {key[3]:<12} | {key[4]:<20} | {n} | {npre:>3} | {fmt(f1m,f1s_,n)} | {fmt(fpm,fps_,n)} | {fmt(acm,acs_,n)}")

# ---- BEST INDIVIDUAL GENOME (best-of-N CEILING, not the claim) vs RF RAW comparator ----
print("\n\n######## BEST INDIVIDUAL GENOME per dataset (best-of-N CEILING) ########")
SQL_IND = """
SELECT f.id AS flow_id, f.name, f.started_at, e.phase_type, vs.genome_type, vs.genome_hash, vs.threshold_metadata
FROM validation_summaries vs
JOIN flows f ON f.id = vs.flow_id
JOIN experiments e ON e.id = vs.experiment_id
WHERE (f.name LIKE 'IDSXD-%' OR f.name LIKE 'IDSXD2-%')
  AND f.name NOT LIKE 'IDSXD-unswt-%'
  AND f.status = 'completed'
  AND vs.validation_point = 'final'
"""
# RF RAW comparator: logs/baselines_<ds>/rf_xgb_ada_random_3way_raw.log, rs=42, TEST partition, single run
RF_RAW = {
	"unswr":  {"fixed_05": (95.80, 0.31, 99.39), "val_cal": (95.82, 0.34, 99.38)},
	"cicids": {"fixed_05": (99.83, 0.07, 99.89), "val_cal": (99.85, 0.10, 99.90)},
	"ciciot": {"fixed_05": (95.53, 7.38, 97.84), "val_cal": (95.53, 7.38, 97.84)},
}
pts = defaultdict(list)          # pts[ds] = list of candidate dicts (one per genome-row x thr_mode)
genome_rows = defaultdict(int)   # distinct (flow, phase, genome_type) rows per ds
for r in con.execute(SQL_IND):
	m = NAME_RE.match(r["name"])
	if not m or r["genome_type"] not in GENOMES:
		continue
	ds, mode, bits, arm, seed = m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
	phase = "GS" if r["phase_type"] == "grid_search" else "GA"
	er = "pre" if (r["started_at"] or "") < FIX_UTC else "post"
	tm = json.loads(r["threshold_metadata"])
	genome_rows[ds] += 1
	vc = tm.get("val_cal", {})
	vc_triple = (vc["f1"] * 100, vc["fpr"] * 100, vc["acc"] * 100) if isinstance(vc, dict) and vc.get("f1") is not None else None
	for thr in MODES:
		md = tm.get(thr, {})
		if not isinstance(md, dict) or md.get("f1") is None:
			continue
		pts[ds].append(dict(flow_id=r["flow_id"], name=r["name"], seed=seed, arm=arm, mode=mode, phase=phase,
		                    gt=r["genome_type"], thr=thr, era=er, hash=r["genome_hash"][:10],
		                    f1=md["f1"] * 100, fpr=md["fpr"] * 100, acc=md["acc"] * 100, val_cal=vc_triple))

def show(ds, label, p):
	print(f"  {label:<22} | {p['name']} (flow {p['flow_id']}, seed {p['seed']}) | {p['arm']} {p['mode']} {p['phase']} "
	      f"{p['gt']} {p['thr']} | era={p['era']} | hash={p['hash']} | F1 {p['f1']:.2f} / FPR {p['fpr']:.2f} / Acc {p['acc']:.2f}")
	if p["thr"] != "val_cal" and p["val_cal"]:
		v = p["val_cal"]
		print(f"  {'  same genome @val_cal':<22} | F1 {v[0]:.2f} / FPR {v[1]:.2f} / Acc {v[2]:.2f}")

for ds in ["unswr", "cicids", "ciciot"]:
	cand = pts[ds]
	n_cand = len(cand)
	print(f"\n=== {ds}: max taken over {n_cand} candidates = {genome_rows[ds]} genome rows (flows x 2 phases x 5 genome_types) x 7 thr_modes ===")
	a = max(cand, key=lambda p: p["f1"])
	b = min((p for p in cand if p["f1"] > 90), key=lambda p: p["fpr"])
	c = max(cand, key=lambda p: p["acc"])
	show(ds, "(a) best F1", a); show(ds, "(b) best FPR | F1>90", b); show(ds, "(c) best Acc", c)
	if ds == "ciciot":
		clean = [p for p in cand if p["era"] == "post"]
		print("  --- era-clean (post-fix only) ---")
		show(ds, "(a) best F1 [post]", max(clean, key=lambda p: p["f1"]))
		show(ds, "(b) best FPR|F1>90 [post]", min((p for p in clean if p["f1"] > 90), key=lambda p: p["fpr"]))
		show(ds, "(c) best Acc [post]", max(clean, key=lambda p: p["acc"]))
	rf = RF_RAW[ds]
	print(f"\n  {ds} — WNN best-of-{n_cand} single genomes vs RF RAW single run (rs=42, TEST, random_3way)")
	hdr = "  source/config                                                  |    F1  |   FPR  |   Acc  | note"
	print(hdr); print("  " + "-" * (len(hdr) - 2))
	def row(label, f1, fpr, acc, note):
		print(f"  {label:<62} | {f1:6.2f} | {fpr:6.2f} | {acc:6.2f} | {note}")
	for label, p in (("WNN best-F1 genome", a), ("WNN best-FPR genome (F1>90)", b), ("WNN best-Acc genome", c)):
		lab = f"{label}: {p['arm']} {p['mode']} {p['phase']} {p['gt']} {p['thr']} s{p['seed']}"
		row(lab, p["f1"], p["fpr"], p["acc"], f"best-of-{n_cand} CEILING, era={p['era']}, flow {p['flow_id']}")
	row("RF raw fixed_05 (thr 0.50)", *rf["fixed_05"], "single deterministic run, n=1")
	row("RF raw val_cal", *rf["val_cal"], "single deterministic run, n=1")
