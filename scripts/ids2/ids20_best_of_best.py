import sys, io, contextlib
sys.argv=["x","none"]
src=open("/Users/lacg/wnn/scripts/ids2/ids20_idsagg_readout.py").read()
exec(compile(src,"readout","exec"))
FLOOR={"cicids":99.4,"ciciot":92.5}
SC={}
def sz(f,p,g):
	k=(f,p,g)
	if k not in SC: SC[k]=size(f,p,g)
	return SC[k]
def rows_for(ds, pres):
	R=[]
	for (f,p,g,md),v in D.items():
		x=flows[f]
		if x["ds"]!=ds or x["pre"] not in pres or x["status"]!="completed": continue
		R.append(dict(f=f,arm=x["arm"],agg=x["agg"],seed=x["seed"],p=p,g=g,md=md,F1=v[0],FPR=v[1],Acc=v[2]))
	return R
def line(r):
	n,b=sz(r["f"],r["p"],r["g"])
	return f"{r['f']} | {r['arm']:8s} | {r['agg']:5s} | {r['seed']} | {r['p']} | {r['g']:12s} | {r['md']:20s} | {r['F1']:6.3f} | {r['FPR']:6.3f} | {r['Acc']:6.3f} | {n}n x {b:.1f}b"
H="flow | arm      | agg   | seed  | ph | genome_type  | mode                 | F1     | FPR    | Acc    | shape"
def pareto(R):
	return [r for r in R if not any(o["F1"]>=r["F1"] and o["FPR"]<=r["FPR"] and (o["F1"]>r["F1"] or o["FPR"]<r["FPR"]) for o in R)]
def tally(lst,lab):
	from collections import Counter
	c=Counter(r["agg"] for r in lst); a=Counter(f"{r['arm']}-{r['agg']}" for r in lst)
	print(f"  TALLY {lab}: n={len(lst)} zs={c['zs']} desir={c['desir']} | per arm: {dict(sorted(a.items()))}")
for ds,pres,lab in [("cicids",["IDSAGG"],"CICIDS IDSAGG"),("ciciot",["IDSAGG"],"CIC-IoT IDSAGG"),("ciciot",["IDSAGG","IDSAGGX"],"CIC-IoT IDSAGG + PARTIAL IDSAGGX B05-AC")]:
	R=rows_for(ds,pres); fl=FLOOR[ds]
	print(f"\n######## {lab}: {len(R)} single-genome cells from {len(set(r['f'] for r in R))} flows")
	t10=sorted(R,key=lambda r:(-r["F1"],r["FPR"]))[:10]
	print("TOP-10 by F1"); print(H); [print(line(r)) for r in t10]
	print("TOP-5 by Acc"); print(H); [print(line(r)) for r in sorted(R,key=lambda r:(-r["Acc"],r["FPR"]))[:5]]
	lf=sorted([r for r in R if r["F1"]>=fl],key=lambda r:(r["FPR"],-r["F1"]))[:8]
	print(f"LOWEST FPR at F1>={fl} (n eligible={sum(r['F1']>=fl for r in R)})"); print(H); [print(line(r)) for r in lf]
	pf=sorted(pareto(R),key=lambda r:-r["F1"])
	print("PARETO (F1 up, FPR down)"); print(H); [print(line(r)) for r in pf]
	tally(t10,"top10 F1"); tally(pf,"pareto"); tally(lf,f"lowFPR@{fl} top8")
	# config-level
	print("CONFIG LEVEL: best anywhere vs 5-seed mean on that same cell (phase,gt,mode)")
	print("arm      | agg   | nflows | maxF1 cell -> best | mean±sd || maxAcc cell -> best | mean±sd || minFPR@floor cell -> best (F1) | mean FPR±sd (mean F1)")
	for arm in sorted(set(r["arm"] for r in R)):
		for agg in ["zs","desir"]:
			S=[r for r in R if r["arm"]==arm and r["agg"]==agg]
			if not S: continue
			nf=len(set(r["f"] for r in S))
			def cell(r): return [x for x in S if x["p"]==r["p"] and x["g"]==r["g"] and x["md"]==r["md"]]
			b=max(S,key=lambda r:(r["F1"],-r["FPR"])); c=cell(b)
			a=max(S,key=lambda r:(r["Acc"],-r["FPR"])); ca=cell(a)
			el=[r for r in S if r["F1"]>=fl]
			if el:
				m=min(el,key=lambda r:(r["FPR"],-r["F1"])); cm=cell(m)
				s3=f"{m['p']}/{m['g']}/{m['md']} -> {m['FPR']:.3f} ({m['F1']:.2f}) f{m['f']} s{m['seed']} | {ms([x['FPR'] for x in cm])} ({ms([x['F1'] for x in cm])})"
			else: s3="none at floor"
			print(f"{arm:8s} | {agg:5s} | {nf} | {b['p']}/{b['g']}/{b['md']} -> {b['F1']:.3f} f{b['f']} s{b['seed']} | {ms([x['F1'] for x in c])} || {a['p']}/{a['g']}/{a['md']} -> {a['Acc']:.3f} f{a['f']} | {ms([x['Acc'] for x in ca])} || {s3}")
