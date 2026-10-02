"""READ-ONLY IDS-8 FINAL readout (02/10/2026): IDSXD-unswt-quad-16b-* (flows 6237-6263).

UNSW-NB15 temporal_3way, Protocol v2 (thresholds fit on VAL, reported on TEST). Every number is
HELD-OUT, from validation_summaries.threshold_metadata at validation_point='final'.
NEVER iterations.best_f1.

Subcommands:
  audit                  status / experiments / start-time eras / min_bits census / generations
  arms [gt]              per-arm GA val_cal table (all 5 genome types, or one)
  armtable <gt>          per-arm GA val_cal + paired deltas vs B34-CTRL and Wb-CTRL (the readout table)
  stats <gt>             two-way residual sigma, MDD, paired t + Holm vs both controls
  seeds <arm> <gt>       per-seed GA val_cal rows with era tag and genome shape
  paired <gt> <ctrl>     per-seed GA val_cal deltas vs a control arm (F1/FPR/Acc)
  pareto [fmin]          non-dominated (mean F1 up, mean FPR down) over arms x GS/GA x 5 gt x 7 modes
  rule7 <arm>            CLAUDE.md Rule-7 5-tables (Grid vs GA, 7 modes, mean±SD %)
  refs                   reference cohorts (SP100-unswt-quad-16bWb, IDSZ-unswt-quad-16b) GA val_cal
"""
import json, re, sqlite3, statistics as st, sys
from collections import defaultdict

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]
GTS = ["best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"]
NAME_RE = re.compile(r"^IDSXD-unswt-quad-16b-(.+?)-r(\d+)$")
FLOW_LO, FLOW_HI = 6237, 6263
IMMIG_FIX = "2026-09-17T01:57"      # 129f8533 immigrant min_bits fix (IDS-17)
WNN1_RESTART = "2026-09-26T19:09"   # worker_swap to ABI 14 (counter-RNG offspring seeds, WNN-1)

con = sqlite3.connect(DB, uri=True); con.row_factory = sqlite3.Row


def ms(xs):
	return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


def f(x):
	return f"{x[0]:6.2f}±{x[1]:4.2f}"


def vals_of(tm_json):
	tm = json.loads(tm_json)
	return {k: (tm[k]["f1"] * 100, tm[k]["fpr"] * 100, tm[k]["acc"] * 100)
		for k in MODES if isinstance(tm.get(k), dict) and tm[k].get("f1") is not None}


def load(where, name_re):
	sql = f"""SELECT f.id fid, f.name, f.started_at, e.phase_type, vs.genome_type gt, vs.genome_hash gh,
		vs.experiment_id eid, vs.threshold_metadata tm
		FROM validation_summaries vs JOIN flows f ON f.id = vs.flow_id JOIN experiments e ON e.id = vs.experiment_id
		WHERE {where} AND f.status = 'completed' AND vs.validation_point = 'final'"""
	d = defaultdict(dict)
	for r in con.execute(sql):
		m = name_re.match(r["name"])
		if not m or r["gt"] not in GTS:
			continue
		arm, seed = m.group(1), int(m.group(2))
		ph = "GS" if r["phase_type"] == "grid_search" else "GA"
		d[(arm, ph, r["gt"])][seed] = dict(vals=vals_of(r["tm"]), fid=r["fid"], start=r["started_at"], gh=r["gh"], eid=r["eid"])
	return d


D = load(f"f.id BETWEEN {FLOW_LO} AND {FLOW_HI}", NAME_RE)


def bits_of(tiers_json):
	"""Per-neuron bits from either genome format: dict {bits_per_neuron} (validated winners) or a
	list of tier rows [{neurons, bits}] (tracked genomes; post-17/09 exact per tier)."""
	t = json.loads(tiers_json)
	if isinstance(t, dict):
		return t.get("bits_per_neuron") or []
	return [row["bits"] for row in t for _ in range(row["neurons"])]


def shape(rec):
	if "shape" not in rec:
		r = con.execute("SELECT tiers_json FROM genomes WHERE experiment_id=? AND genome_hash=? LIMIT 1", (rec["eid"], rec["gh"])).fetchone()
		b = bits_of(r[0]) if r else []
		rec["shape"] = (len(b), st.mean(b) if b else 0, min(b) if b else 0, max(b) if b else 0)
	return rec["shape"]


def agg(d, arm, ph, gt, mode, seeds=None):
	cell = d.get((arm, ph, gt), {})
	ss = sorted(s for s in cell if (seeds is None or s in seeds) and mode in cell[s]["vals"])
	if not ss:
		return None
	v = [cell[s]["vals"][mode] for s in ss]
	return ss, ms([x[0] for x in v]), ms([x[1] for x in v]), ms([x[2] for x in v])


def arms_of(d):
	return sorted({k[0] for k in d})


def era(start):
	return ("old-imm" if start < IMMIG_FIX else "new-imm") + "/" + ("wallclock" if start < WNN1_RESTART else "ctrRNG")


def audit():
	rows = con.execute(f"""SELECT f.id, f.name, f.status, f.started_at, f.completed_at, f.config_json,
		(SELECT COUNT(*) FROM experiments e WHERE e.flow_id=f.id) n_exp,
		(SELECT e.current_iteration FROM experiments e WHERE e.flow_id=f.id AND e.phase_type='ga_neurons') ga_gens
		FROM flows f WHERE f.id BETWEEN {FLOW_LO} AND {FLOW_HI} ORDER BY f.started_at""").fetchall()
	print("fid  | arm       | seed  | status    | n_exp | started (UTC)        | dur_min | era                | GA gens | bits band | genome bits census (all tracked genomes)")
	tot = 0.0
	for r in rows:
		p = json.loads(r["config_json"])["params"]
		m = NAME_RE.match(r["name"]); lo, hi = p["min_bits"], p["max_bits"]
		from datetime import datetime
		dur = (datetime.fromisoformat(r["completed_at"]) - datetime.fromisoformat(r["started_at"])).total_seconds() / 60; tot += dur
		bits = []
		for g in con.execute("SELECT tiers_json FROM genomes WHERE experiment_id IN (SELECT id FROM experiments WHERE flow_id=?)", (r["id"],)):
			bits += bits_of(g[0])
		bad = sum(1 for b in bits if b < lo or b > hi)
		print(f"{r['id']} | {m.group(1):<9} | {m.group(2)} | {r['status']:<9} | {r['n_exp']}     | {r['started_at'][:19]}  | {dur:7.1f} | {era(r['started_at']):<18} | {r['ga_gens']!s:>7} | {lo}-{hi}     | neurons={len(bits)} min={min(bits)} max={max(bits)} out_of_band={bad}")
	print(f"total {tot/60:.1f} h, mean {tot/len(rows):.1f} min/run, n={len(rows)}")


def arm_table(gts=GTS):
	print("arm       | gt           | n | F1 (GA val_cal) | FPR           | Acc           | GA shape n x bits (mean)")
	for arm in arms_of(D):
		for gt in gts:
			a = agg(D, arm, "GA", gt, "val_cal")
			if a:
				sh = [shape(r) for r in D[(arm, "GA", gt)].values()]
				print(f"{arm:<9} | {gt:<12} | {len(a[0])} | {f(a[1])}    | {f(a[2])}  | {f(a[3])}  | {st.mean(x[0] for x in sh):.0f}±{st.stdev(x[0] for x in sh):.0f} x {st.mean(x[1] for x in sh):.1f}")


def paired(gt, ctrl):
	c = D[(ctrl, "GA", gt)]
	print(f"--- paired by seed vs {ctrl}, GA {gt} val_cal (pp) ---")
	cv = [c[s]["vals"]["val_cal"] for s in sorted(c)]
	print(f"{ctrl} absolute: F1 {f(ms([v[0] for v in cv]))} FPR {f(ms([v[1] for v in cv]))} Acc {f(ms([v[2] for v in cv]))}")
	print("arm       | dF1 per seed (03/04/05)  | mean   sd    | dFPR per seed            | mean   sd    | dAcc mean")
	for arm in arms_of(D):
		if arm == ctrl:
			continue
		a = D[(arm, "GA", gt)]; ss = sorted(set(a) & set(c))
		d = [[a[s]["vals"]["val_cal"][i] - c[s]["vals"]["val_cal"][i] for s in ss] for i in range(3)]
		m = [ms(x) for x in d]
		print(f"{arm:<9} | {' '.join(f'{x:+6.2f}' for x in d[0])}    | {m[0][0]:+5.2f} {m[0][1]:4.2f}  | {' '.join(f'{x:+6.2f}' for x in d[1])}    | {m[1][0]:+5.2f} {m[1][1]:4.2f}  | {m[2][0]:+5.2f}")


def per_seed(arm, gt):
	for s, rec in sorted(D[(arm, "GA", gt)].items()):
		v = rec["vals"]["val_cal"]; sh = shape(rec)
		print(f"  {arm} s{s} flow {rec['fid']} [{era(rec['start'])}] {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f} shape {sh[0]}n x {sh[1]:.1f}b (min {sh[2]} max {sh[3]})")


def pareto(fmin=None):
	cols = []
	for arm in arms_of(D):
		for ph in ("GS", "GA"):
			for gt in GTS:
				for md in MODES:
					a = agg(D, arm, ph, gt, md)
					if a and len(a[0]) == 3:
						cols.append((arm, ph, gt, md, a))
	nd = [c for c in cols if not any(o[4][1][0] >= c[4][1][0] and o[4][2][0] <= c[4][2][0] and (o[4][1][0] > c[4][1][0] or o[4][2][0] < c[4][2][0]) for o in cols)]
	nd.sort(key=lambda c: -c[4][1][0])
	print(f"--- PARETO (mean F1 vs mean FPR) over {len(cols)} columns ---")
	for arm, ph, gt, md, a in nd:
		if fmin is None or a[1][0] >= fmin:
			print(f"{arm:<9} {ph} {gt:<12} {md:<20} n={len(a[0])} F1 {f(a[1])} FPR {f(a[2])} Acc {f(a[3])}")


def rule7(arm):
	print(f"\n##### RULE-7 5-tables: IDSXD-unswt-quad-16b-{arm} (Grid vs GA, held-out TEST, mean±SD %) #####")
	for gt in GTS:
		print(f"\n{gt}  (runs: GS {len(D[(arm, 'GS', gt)])} | GA {len(D[(arm, 'GA', gt)])})")
		for ph, lab in (("GS", "Grid Search"), ("GA", "GA Neurons ")):
			sh = [shape(r) for r in D[(arm, ph, gt)].values()]
			n = ms([x[0] for x in sh]); b = ms([x[1] for x in sh])
			print(f"{lab} : {n[0]:.0f}±{n[1]:.0f} neurons | {b[0]:.1f}±{b[1]:.1f} bits (mean per-neuron)")
		print("mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA")
		print("---------------------+--------------+--------------+--------------+--------------+--------------+-------------")
		for md in MODES:
			g = agg(D, arm, "GS", gt, md); a = agg(D, arm, "GA", gt, md)
			print(f"{md:<20} | {f(g[1])} | {f(a[1])} | {f(g[2])} | {f(a[2])} | {f(g[3])} | {f(a[3])}")


def refs():
	for label, where, rx in (
		("SP100-unswt-quad-16bWb (pre-19/08 code era)", "f.name LIKE 'SP100-unswt-quad-16bWb%'", re.compile(r"^SP100-unswt-quad-(16bWb)-.*?r?(\d+)$")),
		("IDSZ-unswt-quad-16b (zscore, seeds 20301-05)", "f.name LIKE 'IDSZ-unswt-quad-16b-%'", re.compile(r"^IDSZ-unswt-quad-16b-(.+?)-r(\d+)$")),
	):
		d = load(where, rx)
		print(f"\n--- {label} : GA val_cal ---")
		for arm in arms_of(d):
			for gt in ("best_f1", "best_fitness"):
				a = agg(d, arm, "GA", gt, "val_cal")
				if a:
					print(f"{arm:<10} {gt:<12} n={len(a[0]):<3} F1 {f(a[1])} FPR {f(a[2])} Acc {f(a[3])}")


if __name__ == "__main__":
	cmd = sys.argv[1]
	if cmd == "audit":
		audit()
	elif cmd == "arms":
		arm_table([sys.argv[2]] if len(sys.argv) > 2 else GTS)
	elif cmd == "paired":
		paired(sys.argv[2], sys.argv[3])
	elif cmd == "seeds":
		per_seed(sys.argv[2], sys.argv[3])
	elif cmd == "pareto":
		pareto(float(sys.argv[2]) if len(sys.argv) > 2 else None)
	elif cmd == "rule7":
		rule7(sys.argv[2])
	elif cmd == "refs":
		refs()


def stats(gt):
	"""Paired t (df=2) per arm vs each control, Holm over the 8 contrasts per control/metric;
	two-way (arm x seed) residual SD = seed noise after removing seed-block effects."""
	from scipy import stats as ss
	import numpy as np
	arms = arms_of(D); seeds = sorted(D[(arms[0], "GA", gt)])
	for i, met in enumerate(("F1", "FPR")):
		Y = np.array([[D[(a, "GA", gt)][s]["vals"]["val_cal"][i] for s in seeds] for a in arms])
		R = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
		sig = float(np.sqrt((R ** 2).sum() / ((len(arms) - 1) * (len(seeds) - 1))))
		raw_sd = float(np.sqrt(np.mean(Y.var(1, ddof=1))))
		tcrit = ss.t.ppf(0.975, (len(arms) - 1) * (len(seeds) - 1))
		print(f"[{gt} {met}] pooled within-arm SD {raw_sd:.3f} | two-way residual sigma {sig:.3f} (df {(len(arms)-1)*(len(seeds)-1)}) | paired-diff MDD(alpha .05, n=3) ~ {tcrit*sig*np.sqrt(2/len(seeds)):.2f} pp")
		for ctrl in ("B34-CTRL", "Wb-CTRL"):
			c = Y[arms.index(ctrl)]; res = []
			for a in arms:
				if a == ctrl:
					continue
				d = Y[arms.index(a)] - c
				t, p = ss.ttest_1samp(d, 0.0)
				res.append([a, d.mean(), int((d > 0).sum()), t, p])
			order = sorted(range(len(res)), key=lambda k: res[k][4]); m = len(res); run = 0.0
			for rank, k in enumerate(order):
				run = max(run, min(1.0, (m - rank) * res[k][4])); res[k].append(run)
			print(f"  vs {ctrl}: " + " | ".join(f"{a} {mu:+.2f} ({pos}/3+) p={p:.3f} holm={h:.2f}" for a, mu, pos, t, p, h in res))


if __name__ == "__main__" and sys.argv[1] == "stats":
	stats(sys.argv[2])


def arm_delta_table(gt):
	"""One row per arm: GA val_cal mean±SD + paired-by-seed deltas (mean±SD, seeds-in-favour) vs both controls."""
	print(f"GA {gt} val_cal, n=3 seeds each; delta = arm − control, paired by seed (pp); [k/3] = seeds where the arm is better")
	print("arm      | F1           | FPR          | Acc          | dF1 vs B34       | dFPR vs B34      | dF1 vs Wb        | dFPR vs Wb       | GA shape")
	print("---------+--------------+--------------+--------------+------------------+------------------+------------------+------------------+---------------")
	for arm in arms_of(D):
		a = agg(D, arm, "GA", gt, "val_cal"); cells = []
		for ctrl in ("B34-CTRL", "Wb-CTRL"):
			for i, better in ((0, 1), (1, -1)):
				if arm == ctrl:
					cells.append("       —        "); continue
				c = D[(ctrl, "GA", gt)]; x = D[(arm, "GA", gt)]
				d = [x[s]["vals"]["val_cal"][i] - c[s]["vals"]["val_cal"][i] for s in sorted(c)]
				m = ms(d); k = sum(1 for v in d if v * better > 0)
				cells.append(f"{m[0]:+5.2f}±{m[1]:4.2f} [{k}/3]")
		sh = [shape(r) for r in D[(arm, "GA", gt)].values()]
		print(f"{arm:<8} | {f(a[1])} | {f(a[2])} | {f(a[3])} | " + " | ".join(cells) + f" | {st.mean(x[0] for x in sh):.0f}n x {st.mean(x[1] for x in sh):.1f}b")


if __name__ == "__main__" and sys.argv[1] == "armtable":
	arm_delta_table(sys.argv[2])


def best_rows(mode="val_cal", f1_floor=90.0):
	"""mode='any' scans every genome type x all 7 threshold modes x Grid/GA (the paper's 'best found').
	Best SINGLE genome per arm (no mean/SD): highest F1, highest Acc, lowest FPR with F1 >= floor.
	One row = one (flow, phase, genome_type) held-out TEST result at threshold `mode` (VAL-calibrated).
	Best-of-N over 3 seeds x 2 phases x 5 genome types — report as 'best found', never as the expected result."""
	rows = []
	for (arm, ph, gt), cell in D.items():
		for seed, rec in cell.items():
			for md in (MODES if mode == "any" else [mode]):
				v = rec["vals"].get(md)
				if v:
					rows.append(dict(arm=arm, ph=ph, gt=gt, md=md, seed=seed, fid=rec["fid"], f1=v[0], fpr=v[1], acc=v[2], rec=rec))

	def line(tag, r):
		if r is None:
			return f"  {tag:<13} —"
		n, mb, lo, hi = shape(r["rec"])
		return (f"  {tag:<13} F1 {r['f1']:6.2f} | FPR {r['fpr']:5.2f} | Acc {r['acc']:6.2f} | {r['ph']} {r['gt']:<12} {r['md']:<20} "
			f"r{r['seed']} flow {r['fid']} | {n}n x {mb:.1f}b")

	def pick(rs):
		hi_f1 = [r for r in rs if r["f1"] >= f1_floor]
		return (max(rs, key=lambda r: (r["f1"], -r["fpr"])), max(rs, key=lambda r: (r["acc"], -r["fpr"])),
			min(hi_f1, key=lambda r: (r["fpr"], -r["f1"])) if hi_f1 else None)

	print(f"BEST SINGLE GENOMES — mode={mode}, held-out TEST; low-FPR row requires F1 >= {f1_floor}")
	for arm in arms_of(D) + ["ALL"]:
		rs = rows if arm == "ALL" else [r for r in rows if r["arm"] == arm]
		bf, ba, bl = pick(rs)
		print(f"{arm}  ({len(rs)} rows)")
		print(line("best F1", bf))
		print(line("best Acc", ba))
		print(line(f"min FPR@F1>={f1_floor:g}", bl))
		print(line("min FPR (any)", min(rs, key=lambda r: (r["fpr"], -r["f1"]))))


if __name__ == "__main__" and sys.argv[1] == "best":
	best_rows(sys.argv[2] if len(sys.argv) > 2 else "val_cal", float(sys.argv[3]) if len(sys.argv) > 3 else 90.0)
