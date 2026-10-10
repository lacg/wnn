#!/usr/bin/env python3
"""IDS-2 UNSW-random decision readout, applied literally from experiments/ids2_unswr_rule.json.

READ-ONLY (sqlite mode=ro). No completion assumptions: every arm uses whatever
COMPLETED flows exist; flows still queued/running for an arm are listed and get a
what-if bound. Re-run unchanged at 94/94.

Usage: python3 ids2_rule_readout.py [--label "EARLY LOOK 91/94 - not the decision"]
"""
import argparse, json, math, re, sqlite3, statistics as st
from collections import defaultdict
from datetime import datetime

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
RULE = "/Users/lacg/wnn/experiments/ids2_unswr_rule.json"
WEIGHTS = ["B05-AC", "B05-CE", "B10-AC", "B10-CE", "B15-AC", "B15-CE", "B34-CTRL", "CE20", "Wb-CTRL"]
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]
GTYPES = ["best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"]
PHASE_ABBR = {"grid_search": "Grid", "ga_neurons": "GA"}
INCUMBENT = ("Wb-CTRL", "QUAD")


def conn():
	return sqlite3.connect(DB, uri=True)


def seed_of(name):
	return int(re.search(r"-r(\d+)", name).group(1))


def hours(a, b):
	if not a or not b:
		return float("nan")
	fa = datetime.fromisoformat(a.replace("Z", "+00:00"))
	fb = datetime.fromisoformat(b.replace("Z", "+00:00"))
	return (fb - fa).total_seconds() / 3600


# ---------------------------------------------------------------- arm membership
def arm_flows(c, weight, mode):
	"""Return (completed_rows, pending_rows) per the rule's arm definition."""
	if mode == "QSR":
		pat = f"IDSXD-unswr-qsr-64b-{weight}-r%-abi13"
		rows = c.execute("select id,name,status,started_at,completed_at from flows where name like ?", (pat,)).fetchall()
		rows = [r for r in rows if re.fullmatch(rf"IDSXD-unswr-qsr-64b-{re.escape(weight)}-r\d+-abi13", r[1])]
	else:
		base = c.execute("select id,name,status,started_at,completed_at from flows where name like ?",
			(f"IDSXD-unswr-quad-64b-{weight}-r%",)).fetchall()
		orig = {seed_of(r[1]): r for r in base if re.fullmatch(rf"IDSXD-unswr-quad-64b-{re.escape(weight)}-r\d+", r[1])}
		fix = {seed_of(r[1]): r for r in base if re.fullmatch(rf"IDSXD-unswr-quad-64b-{re.escape(weight)}-r\d+-immfix", r[1])}
		rows = []
		for s in sorted(set(orig) | set(fix)):
			f = fix.get(s)
			# -immfix where one exists (and has completed); else the original
			rows.append(f if (f and f[2] == "completed") else (orig.get(s) or f))
	done = sorted([r for r in rows if r[2] == "completed"], key=lambda r: seed_of(r[1]))
	pend = [r for r in rows if r[2] in ("queued", "running")]
	return done, pend


# ---------------------------------------------------------------- cells
def flow_cells(c, fid):
	"""All held-out TEST cells: validation_point='final', both phases x genome types x 7 modes."""
	out = []
	q = """select e.phase_type, v.genome_type, v.threshold_metadata from validation_summaries v
		join experiments e on e.id=v.experiment_id where v.flow_id=? and v.validation_point='final'"""
	for ph, gt, tm in c.execute(q, (fid,)):
		if ph not in PHASE_ABBR or gt not in GTYPES or not tm:
			continue
		d = json.loads(tm)
		for m in MODES:
			x = d.get(m)
			if not isinstance(x, dict) or x.get("f1") is None:
				continue
			out.append(dict(phase=PHASE_ABBR[ph], gt=gt, mode=m,
				f1=100 * x["f1"], fpr=100 * x["fpr"], acc=100 * x["acc"]))
	return out


def run_scores(cells, floor):
	"""Per-run best-everywhere scores with the rule's tie-breaks."""
	bf1 = max(cells, key=lambda z: (z["f1"], -z["fpr"]))
	elig = [z for z in cells if z["f1"] >= floor]
	bfpr = min(elig, key=lambda z: (z["fpr"], -z["f1"])) if elig else None
	bacc = max(cells, key=lambda z: (z["acc"], -z["fpr"]))
	return dict(F1=bf1["f1"], FPR=(bfpr["fpr"] if bfpr else 100.0), Acc=bacc["acc"],
		cF1=bf1, cFPR=bfpr, cAcc=bacc, ncells=len(cells))


def loc(z):
	return "-" if z is None else f"{z['phase']}/{z['gt']}/{z['mode']}"


def ms(v):
	if not v:
		return "   n/a      "
	sd = st.stdev(v) if len(v) > 1 else float("nan")
	return f"{st.mean(v):6.2f}±{sd:4.2f}"


def pooled_sd(groups):
	num = sum((len(g) - 1) * st.variance(g) for g in groups if len(g) > 1)
	den = sum(len(g) - 1 for g in groups if len(g) > 1)
	return math.sqrt(num / den) if den else float("nan")


BETTER = {"F1": lambda a, b: a > b, "FPR": lambda a, b: a < b, "Acc": lambda a, b: a > b}


def decide(arms, metric):
	"""Winner by mean, then fallback (b) vs Wb-CTRL QUAD mean with pooled seed-SD."""
	means = {k: st.mean(v["scores"][metric]) for k, v in arms.items() if v["scores"][metric]}
	order = sorted(means, key=lambda k: means[k], reverse=(metric != "FPR"))
	psd = pooled_sd([v["scores"][metric] for v in arms.values() if v["scores"][metric]])
	win = order[0]
	inc = means.get(INCUMBENT)
	margin = (means[win] - inc) if metric != "FPR" else (inc - means[win])
	final = win if (win == INCUMBENT or margin >= psd) else INCUMBENT
	return dict(order=order, means=means, psd=psd, raw=win, inc=inc, margin=margin, final=final)


# ---------------------------------------------------------------- what-if for pending runs
def what_if(arms, key, metric, grid):
	"""For a pending run in arm `key`: which values x flip the metric's final winner?"""
	base = decide(arms, metric)["final"]
	flips = []
	for x in grid:
		a2 = {k: dict(scores={m: list(v["scores"][m]) for m in v["scores"]}) for k, v in arms.items()}
		a2[key]["scores"][metric].append(x)
		d = decide(a2, metric)
		if d["final"] != base:
			flips.append((x, d["final"]))
	return base, flips


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--label", default="")
	a = ap.parse_args()
	rule = json.load(open(RULE))
	floor = rule["criterion_a"]["f1_floor"]
	c = conn()
	L = a.label
	arms, pending, n_runs = {}, [], 0
	for w in WEIGHTS:
		for mm in ("QUAD", "QSR"):
			done, pend = arm_flows(c, w, mm)
			runs = []
			for fid, name, _, s0, s1 in done:
				cells = flow_cells(c, fid)
				if not cells:
					continue
				r = run_scores(cells, floor)
				r.update(fid=fid, name=name, seed=seed_of(name), h=hours(s0, s1))
				runs.append(r)
			n_runs += len(runs)
			arms[(w, mm)] = dict(runs=runs, scores={m: [r[m] for r in runs] for m in ("F1", "FPR", "Acc")})
			pending += [(w, mm, p[1], p[2]) for p in pend]
	print(f"== IDS-2 UNSW-random best-everywhere readout {L}")
	print(f"   ranked runs used: {n_runs}; pending (queued/running) ranked runs: {len(pending)}")
	for p in pending:
		print(f"   PENDING {p[0]:9s} {p[1]:4s} {p[2]} [{p[3]}]")

	# 1. per-arm table
	print(f"\n-- 1. Per-arm best-everywhere (held-out TEST, final; mean±SD %) {L}")
	print(f"{'arm':9s} {'mode':4s} {'n':>2s} | {'F1 (max)':>12s} | {'FPR@F1>=93':>12s} | {'Acc (max)':>12s} | {'h/run':>5s} | cells/run")
	for (w, mm), v in arms.items():
		s = v["scores"]
		hs = [r["h"] for r in v["runs"]]
		print(f"{w:9s} {mm:4s} {len(v['runs']):2d} | {ms(s['F1']):>12s} | {ms(s['FPR']):>12s} | {ms(s['Acc']):>12s} | "
			f"{(st.mean(hs) if hs else float('nan')):5.2f} | {','.join(str(r['ncells']) for r in v['runs'])}")

	# rankings + decision
	dec = {}
	for m in ("F1", "FPR", "Acc"):
		d = decide(arms, m)
		dec[m] = d
		print(f"\n-- Ranking {m} {L}")
		for i, k in enumerate(d["order"], 1):
			v = arms[k]["scores"][m]
			gap = d["means"][k] - d["means"][d["order"][0]]
			print(f"{i:2d}. {k[0]:9s} {k[1]:4s} n={len(v)}  {ms(v)}  gap-to-#1 {gap:+6.3f}  runs: {' '.join(f'{x:.3f}' for x in v)}")

	print(f"\n-- 2. Fallback (b) per metric {L}")
	print(f"{'metric':6s} | {'pooled SD':>9s} | {'Wb-CTRL QUAD':>12s} | {'raw winner':>15s} {'mean':>7s} | {'margin':>7s} | {'>=1 SD?':>7s} | final")
	for m, d in dec.items():
		ok = d["raw"] == INCUMBENT or d["margin"] >= d["psd"]
		print(f"{m:6s} | {d['psd']:9.3f} | {d['inc']:12.3f} | {d['raw'][0]+' '+d['raw'][1]:>15s} {d['means'][d['raw']]:7.3f} | "
			f"{d['margin']:+7.3f} | {('yes' if ok else 'NO'):>7s} | {d['final'][0]} {d['final'][1]}")

	print(f"\n-- 3. Decision {L}")
	finals = {m: d["final"] for m, d in dec.items()}
	if len(set(finals.values())) == 1:
		print(f"   SINGLE WINNER: {list(finals.values())[0]}")
	else:
		print("   SPLIT -> no automatic winner; escalate to Luiz. Winners with their other two metrics:")
		for m, k in finals.items():
			s = arms[k]["scores"]
			print(f"   {m:4s} winner {k[0]:9s} {k[1]:4s} n={len(s['F1'])}: F1 {ms(s['F1'])} | FPR {ms(s['FPR'])} | Acc {ms(s['Acc'])}")
	for m, d in dec.items():
		o = d["order"]
		print(f"   {m}: #1-#2 margin {abs(d['means'][o[0]]-d['means'][o[1]]):.3f} pp ({o[1][0]} {o[1][1]}); pooled SD {d['psd']:.3f}")

	# what-if for pending ranked runs
	for w, mm, name, _ in pending:
		k = (w, mm)
		print(f"\n   WHAT-IF pending {name}:")
		allv = {m: [x for v in arms.values() for x in v["scores"][m]] for m in ("F1", "FPR", "Acc")}
		for m in ("F1", "FPR", "Acc"):
			lo, hi = min(allv[m]), max(allv[m])
			if m == "FPR":
				grid = [round(lo - 0.5 + i * 0.001, 3) for i in range(int((hi - lo + 1.0) / 0.001) + 1)] + [100.0]
			else:
				grid = [round(lo - 0.5 + i * 0.001, 3) for i in range(int((hi - lo + 1.0) / 0.001) + 1)]
			base, flips = what_if(arms, k, m, grid)
			obs = arms[k]["scores"][m]
			if flips:
				xs = [f[0] for f in flips]
				print(f"   {m:4s}: base winner {base}; flips for x in [{min(xs):.3f}, {max(xs):.3f}] -> {sorted(set(f[1] for f in flips))}; "
					f"arm's observed runs {obs}; all-run range [{lo:.3f},{hi:.3f}]")
			else:
				print(f"   {m:4s}: base winner {base}; NO value in [{grid[0]:.3f},{grid[-1]:.3f}] flips it")

	# 4. winning-cell provenance
	print(f"\n-- 4. Winning-cell location per run (phase/genome/mode) {L}")
	print(f"{'arm':9s} {'mode':4s} {'seed':>5s} | {'F1':>6s} {'@':38s} | {'FPR':>6s} {'(F1)':>7s} {'@':38s} | {'Acc':>6s} {'@'}")
	for (w, mm), v in arms.items():
		for r in v["runs"]:
			cf = r["cFPR"]
			print(f"{w:9s} {mm:4s} {r['seed']:5d} | {r['F1']:6.3f} {loc(r['cF1']):38s} | {r['FPR']:6.3f} "
				f"{(cf['f1'] if cf else float('nan')):7.3f} {loc(cf):38s} | {r['Acc']:6.3f} {loc(r['cAcc'])}")
	tally = defaultdict(lambda: defaultdict(int))
	for v in arms.values():
		for r in v["runs"]:
			for m, cz in (("F1", r["cF1"]), ("FPR", r["cFPR"]), ("Acc", r["cAcc"])):
				if cz:
					tally[m][("phase", cz["phase"])] += 1
					tally[m][("gt", cz["gt"])] += 1
					tally[m][("mode", cz["mode"])] += 1
	print("   provenance tally over all ranked runs:")
	for m in ("F1", "FPR", "Acc"):
		print(f"   {m:4s}: " + "  ".join(f"{k[1]}={n}" for k, n in sorted(tally[m].items(), key=lambda kv: (kv[0][0], -kv[1]))))

	# 5. conventional val_cal table (GA Neurons phase, final)
	print(f"\n-- 5. Conventional: GA Neurons final, val_cal, best_f1 and best_fitness (mean±SD %) {L}")
	print(f"{'arm':9s} {'mode':4s} {'n':>2s} | {'best_f1 F1':>12s} {'FPR':>12s} {'Acc':>12s} | {'best_fitness F1':>15s} {'FPR':>12s} {'Acc':>12s}")
	for (w, mm), v in arms.items():
		acc = {g: defaultdict(list) for g in ("best_f1", "best_fitness")}
		for r in v["runs"]:
			for g in acc:
				tm = c.execute("""select v.threshold_metadata from validation_summaries v join experiments e on e.id=v.experiment_id
					where v.flow_id=? and v.validation_point='final' and e.phase_type='ga_neurons' and v.genome_type=?""", (r["fid"], g)).fetchone()
				if tm and tm[0]:
					x = json.loads(tm[0]).get("val_cal") or {}
					for k2 in ("f1", "fpr", "acc"):
						if x.get(k2) is not None:
							acc[g][k2].append(100 * x[k2])
		print(f"{w:9s} {mm:4s} {len(v['runs']):2d} | {ms(acc['best_f1']['f1']):>12s} {ms(acc['best_f1']['fpr']):>12s} {ms(acc['best_f1']['acc']):>12s} | "
			f"{ms(acc['best_fitness']['f1']):>15s} {ms(acc['best_fitness']['fpr']):>12s} {ms(acc['best_fitness']['acc']):>12s}")

	sp100(c, floor, L)


def sp100(c, floor, L):
	"""SP100 QUAD vs QSR-abi13 at Wb (alongside, NOT in the ranking)."""
	def grab(pat, rx):
		rows = [r for r in c.execute("select id,name,status from flows where name like ?", (pat,)) if re.fullmatch(rx, r[1])]
		done = [r for r in rows if r[2] == "completed"]
		pend = [r for r in rows if r[2] in ("queued", "running")]
		res = {}
		for fid, name, _ in done:
			cells = flow_cells(c, fid)
			if not cells:
				continue
			r = run_scores(cells, floor)
			vc = [z for z in cells if z["phase"] == "GA" and z["gt"] == "best_f1" and z["mode"] == "val_cal"]
			r["vc"] = vc[0] if vc else None
			res[seed_of(name)] = r
		return res, pend
	q, qp = grab("SP100-unswr-quad-64bWb-r%", r"SP100-unswr-quad-64bWb-r\d+")
	s, sp = grab("SP100-unswr-qsr-64bWb-r%-abi13", r"SP100-unswr-qsr-64bWb-r\d+-abi13")
	print(f"\n-- 6. SP100 Wb: QUAD vs QSR-abi13 (alongside, NOT ranked) {L}")
	print(f"   QUAD completed n={len(q)} (pending {len(qp)}); QSR-abi13 completed n={len(s)} (pending {len(sp)}: {', '.join(p[1]+'['+p[2]+']' for p in sp)})")
	paired = sorted(set(q) & set(s))
	def row(lbl, sel_q, sel_s):
		vq = [x for x in sel_q if x is not None]
		vs = [x for x in sel_s if x is not None]
		print(f"   {lbl:28s} QUAD {ms(vq)} (n={len(vq)}) | QSR {ms(vs)} (n={len(vs)})")
	for tag, qs, ss in (("ALL", list(q), list(s)), (f"PAIRED (n={len(paired)})", paired, paired)):
		print(f"   [{tag}]")
		row("val_cal GA best_f1  F1", [q[k]["vc"]["f1"] if q[k]["vc"] else None for k in qs], [s[k]["vc"]["f1"] if s[k]["vc"] else None for k in ss])
		row("val_cal GA best_f1  FPR", [q[k]["vc"]["fpr"] if q[k]["vc"] else None for k in qs], [s[k]["vc"]["fpr"] if s[k]["vc"] else None for k in ss])
		row("val_cal GA best_f1  Acc", [q[k]["vc"]["acc"] if q[k]["vc"] else None for k in qs], [s[k]["vc"]["acc"] if s[k]["vc"] else None for k in ss])
		row("best-everywhere F1", [q[k]["F1"] for k in qs], [s[k]["F1"] for k in ss])
		row("best-everywhere FPR@F1>=93", [q[k]["FPR"] for k in qs], [s[k]["FPR"] for k in ss])
		row("best-everywhere Acc", [q[k]["Acc"] for k in qs], [s[k]["Acc"] for k in ss])
	if paired:
		for m in ("F1", "FPR", "Acc"):
			d = [s[k][m] - q[k][m] for k in paired]
			wins = sum(1 for x in d if (x < 0 if m == "FPR" else x > 0))
			print(f"   paired delta QSR-QUAD best-everywhere {m:4s}: {st.mean(d):+.3f} ± {st.stdev(d) if len(d)>1 else float('nan'):.3f}  (QSR better {wins}/{len(d)})")
		for m, kk in (("F1", "f1"), ("FPR", "fpr"), ("Acc", "acc")):
			d = [s[k]["vc"][kk] - q[k]["vc"][kk] for k in paired if s[k]["vc"] and q[k]["vc"]]
			if d:
				wins = sum(1 for x in d if (x < 0 if m == "FPR" else x > 0))
				print(f"   paired delta QSR-QUAD val_cal GA best_f1 {m:4s}: {st.mean(d):+.3f} ± {st.stdev(d) if len(d)>1 else float('nan'):.3f}  (QSR better {wins}/{len(d)})")


if __name__ == "__main__":
	main()
