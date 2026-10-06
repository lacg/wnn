"""IDS-20 (IDSAGG) readout — zscore vs desirability aggregation, paired by seed.

Applies experiments/ids_aggregation_ab_rule.json (primary F1 / secondary FPR, Holm over
CICIDS + CIC-IoT, restricted to the IDSAGG- prefix) and prints the descriptive per-arm table,
the IDSAGGX- B05-AC extension (descriptive only), Pareto mining and the Rule-7 breakdown.
Read-only on the DB. Metric: held-out TEST val_cal, validation_summaries final rows.

Usage: python scripts/ids2/ids20_idsagg_readout.py <cmd>   (default: primary)
  diff | primary | arms [IDSAGG|IDSAGGX] [genome_type] | seeds [prefix] | b05vb15
  rule7 <cicids|ciciot> <zs|desir> | pareto | hdr
  IDSAGG = the pre-registered 40 flows; IDSAGGX = the B05-AC extension (descriptive only).
"""
import sqlite3, json, re, sys
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats
DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
MODES = ["train_cal","fixed_05","platt","beta","empirical","empirical_cumulative","val_cal"]
GT = ["best_f1","best_fpr","best_acc","best_ce","best_fitness"]
RE = re.compile(r"^(IDSAGGX?)-(cicids|ciciot)-quad-96b-(.+)-(zs|desir)-r(\d+)$")
con = sqlite3.connect(DB, uri=True)
flows = {}
for fid, name, st, en, status, cfg in con.execute("SELECT id,name,started_at,completed_at,status,config_json FROM flows WHERE name LIKE 'IDSAGG%' ORDER BY id"):
	m = RE.match(name)
	pre, ds, arm, agg, seed = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
	flows[fid] = dict(pre=pre, ds=ds, arm=arm, agg=agg, seed=seed, st=st, en=en, status=status, cfg=json.loads(cfg)["params"], name=name)
# --- config diff zs vs desir per (pre,ds,arm,seed)
pairs = defaultdict(dict)
for fid, f in flows.items(): pairs[(f["pre"], f["ds"], f["arm"], f["seed"])][f["agg"]] = fid
diffs = set()
for k, d in pairs.items():
	if "zs" in d and "desir" in d:
		a, b = flows[d["zs"]]["cfg"], flows[d["desir"]]["cfg"]
		diffs.add((k[0], k[1], k[2], tuple(sorted(x for x in set(a) | set(b) if a.get(x) != b.get(x)))))
# --- metrics
D = {}
GH = {}
for fid, eid, ph, gt, gh, tm in con.execute("""SELECT vs.flow_id,e.id,e.phase_type,vs.genome_type,vs.genome_hash,vs.threshold_metadata
	FROM validation_summaries vs JOIN experiments e ON e.id=vs.experiment_id JOIN flows f ON f.id=vs.flow_id
	WHERE f.name LIKE 'IDSAGG%' AND f.status='completed' AND vs.validation_point='final'"""):
	p = "GS" if ph == "grid_search" else "GA"
	t = json.loads(tm)
	GH[(fid, p, gt)] = (eid, gh)
	for md in MODES:
		v = t.get(md)
		if isinstance(v, dict) and v.get("f1") is not None:
			D[(fid, p, gt, md)] = (v["f1"]*100, v["fpr"]*100, v["acc"]*100)
def size(fid, p, gt):
	eid, gh = GH[(fid, p, gt)]
	r = con.execute("SELECT tiers_json,total_neurons FROM genomes WHERE experiment_id=? AND (genome_hash=? OR config_hash=?) LIMIT 1", (eid, gh, gh)).fetchone()
	t = json.loads(r[0]); b = t["bits_per_neuron"]
	return len(b), float(np.mean(b))
gens = {}
for fid, c in con.execute("""SELECT e.flow_id, COUNT(i.id) FROM experiments e JOIN flows f ON f.id=e.flow_id LEFT JOIN iterations i ON i.experiment_id=e.id
	WHERE f.name LIKE 'IDSAGG%' AND e.phase_type='ga_neurons' GROUP BY e.id"""): gens[fid] = c
def hrs(f):
	if not f["en"]: return None
	a = datetime.fromisoformat(f["st"]); b = datetime.fromisoformat(f["en"]); return (b-a).total_seconds()/3600
def fid_of(pre, ds, arm, agg, seed):
	return pairs.get((pre, ds, arm, seed), {}).get(agg)
def done(fid): return fid is not None and flows[fid]["status"] == "completed"
def ms(x): x = np.array(x); return f"{x.mean():.2f}±{x.std(ddof=1):.2f}" if len(x) > 1 else (f"{x[0]:.2f}" if len(x) else "n/a")
def ttest(d):
	d = np.array(d); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd/np.sqrt(n)
	t = m/se; p = 2*(1-stats.t.cdf(abs(t), n-1)); tc = stats.t.ppf(.975, n-1)
	return m, sd, t, p, m-tc*se, m+tc*se, int((d > 0).sum()), int((d < 0).sum())
def holm(ps):
	order = sorted(range(len(ps)), key=lambda i: ps[i]); k = len(ps); adj = [0]*k; run = 0
	for r, i in enumerate(order):
		run = max(run, min(1, (k-r)*ps[i])); adj[i] = run
	return adj
ARMS = {"cicids": ["Wa-CTRL", "CE20"], "ciciot": ["Wc-CTRL", "B15-AC"]}
SEEDS = [20414, 20415, 20416, 20417, 20418]
cmd = sys.argv[1] if len(sys.argv) > 1 else "primary"
if cmd == "diff":
	for x in sorted(diffs): print(x)
	print("counts", {s: sum(1 for f in flows.values() if f["status"] == s) for s in set(f["status"] for f in flows.values())})
if cmd == "primary":
	for idx, lab in [(0, "F1"), (1, "FPR")]:
		res = {}
		for ds in ARMS:
			per = []
			for s in SEEDS:
				dd = []
				for a in ARMS[ds]:
					z = D[(fid_of("IDSAGG", ds, a, "zs", s), "GA", "best_f1", "val_cal")][idx]
					de = D[(fid_of("IDSAGG", ds, a, "desir", s), "GA", "best_f1", "val_cal")][idx]
					dd.append(de - z)
				per.append(np.mean(dd))
			res[ds] = (per, ttest(per))
		adj = holm([res[ds][1][3] for ds in ARMS])
		print(f"== {lab}: mean over 2 arms of (desir - zs), GA best_f1 val_cal")
		for (ds, (per, (m, sd, t, p, lo, hi, np_, nn))), pa in zip(res.items(), adj):
			print(f"{ds:7s} per-seed {' '.join(f'{x:+.3f}' for x in per)} | mean {m:+.3f} sd {sd:.3f} CI95 [{lo:+.3f},{hi:+.3f}] t={t:+.2f} p={p:.4f} Holm={pa:.4f} signs +{np_}/-{nn}")
if cmd == "arms":
	pre = sys.argv[2] if len(sys.argv) > 2 else "IDSAGG"
	gt = sys.argv[3] if len(sys.argv) > 3 else "best_f1"
	arms = ARMS if pre == "IDSAGG" else {"ciciot": ["B05-AC"]}
	for ds in arms:
		for a in arms[ds]:
			for agg in ["zs", "desir"]:
				fs = [fid_of(pre, ds, a, agg, s) for s in SEEDS]; fs = [f for f in fs if done(f)]
				v = [D[(f, "GA", gt, "val_cal")] for f in fs]
				sz = [size(f, "GA", gt) for f in fs]
				print(f"{ds:6s} | {a:8s} | {agg:5s} | {len(fs)} | {ms([x[0] for x in v])} | {ms([x[1] for x in v])} | {ms([x[2] for x in v])} | {ms([x[0] for x in sz])}n x {ms([x[1] for x in sz])}b | gens {ms([gens[f] for f in fs])} | wall {ms([hrs(flows[f]) for f in fs])} h | seeds {','.join(str(flows[f]['seed']) for f in fs)}")
if cmd == "seeds":
	pre = sys.argv[2] if len(sys.argv) > 2 else "IDSAGG"
	arms = ARMS if pre == "IDSAGG" else {"ciciot": ["B05-AC"]}
	for ds in arms:
		for a in arms[ds]:
			for s in SEEDS:
				fz, fd = fid_of(pre, ds, a, "zs", s), fid_of(pre, ds, a, "desir", s)
				if not (done(fz) and done(fd)): continue
				z = D[(fz, "GA", "best_f1", "val_cal")]; d = D[(fd, "GA", "best_f1", "val_cal")]
				sz, sd_ = size(fz, "GA", "best_f1"), size(fd, "GA", "best_f1")
				print(f"{ds:6s} | {a:8s} | {s} | {z[0]:.2f} {d[0]:.2f} {d[0]-z[0]:+.3f} | {z[1]:.2f} {d[1]:.2f} {d[1]-z[1]:+.3f} | {z[2]:.2f} {d[2]:.2f} {d[2]-z[2]:+.3f} | {sz[0]}n/{sz[1]:.0f}b {sd_[0]}n/{sd_[1]:.0f}b | gens {gens[fz]} {gens[fd]} | {hrs(flows[fz]):.2f} {hrs(flows[fd]):.2f} h | {fz},{fd}")
if cmd == "b05vb15":
	for agg in ["zs", "desir"]:
		for s in SEEDS:
			f5, f15 = fid_of("IDSAGGX", "ciciot", "B05-AC", agg, s), fid_of("IDSAGG", "ciciot", "B15-AC", agg, s)
			if not (done(f5) and done(f15)): continue
			a, b = D[(f5, "GA", "best_f1", "val_cal")], D[(f15, "GA", "best_f1", "val_cal")]
			print(f"{agg:5s} | {s} | B05 {a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f} | B15 {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f} | d {a[0]-b[0]:+.3f} / {a[1]-b[1]:+.3f} / {a[2]-b[2]:+.3f}")
if cmd == "rule7":
	ds, agg = sys.argv[2], sys.argv[3]
	fs = [f for f, x in flows.items() if x["pre"] == "IDSAGG" and x["ds"] == ds and x["agg"] == agg and x["status"] == "completed"]
	for a in ARMS[ds]:
		af = [f for f in fs if flows[f]["arm"] == a]
		print(f"### {ds} {a} {agg}  (runs: {len(af)}/5)")
		for gt in GT:
			szg = [size(f, "GS", gt) for f in af]; sza = [size(f, "GA", gt) for f in af]
			print(f"{gt}  (runs: {len(af)}/5)")
			print(f"  Grid Search : {ms([x[0] for x in szg])} neurons | {ms([x[1] for x in szg])} bits")
			print(f"  GA Neurons  : {ms([x[0] for x in sza])} neurons | {ms([x[1] for x in sza])} bits")
			print(f"  {'mode':20s} | {'F1 Grid':11s} | {'F1 GA':11s} | {'FPR Grid':11s} | {'FPR GA':11s} | {'Acc Grid':11s} | {'Acc GA':11s}")
			print("  " + "-"*20 + "-+-" + "-+-".join(["-"*11]*6))
			for md in MODES:
				c = []
				for idx in range(3):
					for p in ["GS", "GA"]:
						c.append(ms([D[(f, p, gt, md)][idx] for f in af if (f, p, gt, md) in D]))
				print(f"  {md:20s} | " + " | ".join(f"{x:11s}" for x in c))
if cmd == "pareto":
	for ds in ARMS:
		rows = []
		for a in ARMS[ds]:
			for agg in ["zs", "desir"]:
				af = [f for f, x in flows.items() if x["pre"] == "IDSAGG" and x["ds"] == ds and x["arm"] == a and x["agg"] == agg]
				for gt in GT:
					for md in MODES:
						v = [D[(f, "GA", gt, md)] for f in af]
						rows.append((np.mean([x[0] for x in v]), np.mean([x[1] for x in v]), a, agg, gt, md, ms([x[0] for x in v]), ms([x[1] for x in v]), ms([x[2] for x in v])))
		nd = [r for r in rows if not any((o[0] >= r[0] and o[1] <= r[1]) and (o[0] > r[0] or o[1] < r[1]) for o in rows)]
		print(f"## {ds} non-dominated (GA, mean F1 up / mean FPR down) over 2 arms x 2 aggs x 5 gt x 7 modes = {len(rows)} cells")
		for r in sorted(nd, key=lambda r: -r[0]): print(f"  {r[2]:8s} {r[3]:5s} {r[4]:12s} {r[5]:20s} F1 {r[6]} FPR {r[7]} Acc {r[8]}")
if cmd == "hdr":
	for pre in ["IDSAGG", "IDSAGGX"]:
		for ds in ["cicids", "ciciot"]:
			fs = [f for f, x in flows.items() if x["pre"] == pre and x["ds"] == ds]
			if not fs: continue
			c = [f for f in fs if flows[f]["status"] == "completed"]; h = [hrs(flows[f]) for f in c]
			last = max(flows[f]["en"] for f in c)
			print(f"{pre} {ds}: {len(c)}/{len(fs)} completed | {sum(h):.1f} h total | {np.mean(h):.2f} h/run | last done {last} | statuses {sorted(set(flows[f]['status'] for f in fs))}")
