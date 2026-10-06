import sys
sys.argv=["x","none"]
exec(compile(open("/Users/lacg/wnn/scripts/ids2/ids20_idsagg_readout.py").read(),"readout","exec"))
def best_run(f):
	c=[(k,v) for k,v in D.items() if k[0]==f]
	k,v=max(c,key=lambda kv:(kv[1][0],-kv[1][1]))
	return k,v
def wt(f):
	h=hrs(flows[f]); return h
print("== distinct-flow top-10 by single-genome best F1 (one row per flow)")
for ds,pres in [("cicids",["IDSAGG"]),("ciciot",["IDSAGG","IDSAGGX"])]:
	fs=[f for f,x in flows.items() if x["ds"]==ds and x["pre"] in pres and x["status"]=="completed"]
	rs=sorted([(best_run(f),f) for f in fs],key=lambda r:-r[0][1][0])[:10]
	for (k,v),f in rs:
		x=flows[f]; n,b=size(f,k[1],k[2])
		print(f"{ds} | {f} | {x['pre']} | {x['arm']:8s} | {x['agg']:5s} | {x['seed']} | {k[1]} | {k[2]:12s} | {k[3]:20s} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {n}n x {b:.1f}b")
print("\n== paired per seed: GA best_f1 val_cal and GA best_fitness val_cal; best single genome per run; wall/gens/size")
for pre,ds,arm in [("IDSAGG","cicids","CE20"),("IDSAGG","cicids","Wa-CTRL"),("IDSAGG","ciciot","Wc-CTRL"),("IDSAGG","ciciot","B15-AC"),("IDSAGGX","ciciot","B05-AC")]:
	print(f"### {pre} {ds} {arm}")
	for gt in ["best_f1","best_fitness"]:
		dF=[];dP=[];dA=[]
		for s in SEEDS:
			fz,fd=fid_of(pre,ds,arm,"zs",s),fid_of(pre,ds,arm,"desir",s)
			if not(done(fz) and done(fd)):
				z=D.get((fz,"GA",gt,"val_cal")) if done(fz) else None
				print(f"  {gt:12s} {s} | UNPAIRED zs={'%.3f/%.3f/%.3f'%z if z else 'n/a'} (desir not completed)" if z else f"  {gt:12s} {s} | not completed"); continue
			z=D[(fz,"GA",gt,"val_cal")]; d=D[(fd,"GA",gt,"val_cal")]
			dF.append(d[0]-z[0]);dP.append(d[1]-z[1]);dA.append(d[2]-z[2])
			nz,bz=size(fz,"GA",gt); nd,bd=size(fd,"GA",gt)
			print(f"  {gt:12s} {s} | zs {z[0]:.3f}/{z[1]:.3f}/{z[2]:.3f} | desir {d[0]:.3f}/{d[1]:.3f}/{d[2]:.3f} | dF1 {d[0]-z[0]:+.3f} dFPR {d[1]-z[1]:+.3f} dAcc {d[2]-z[2]:+.3f} | {nz}n/{bz:.0f}b vs {nd}n/{bd:.0f}b")
		if len(dF)>1: print(f"  {gt:12s} MEAN d(desir-zs) n={len(dF)}: F1 {ms(dF)} (+{sum(x>0 for x in dF)}/-{sum(x<0 for x in dF)}) | FPR {ms(dP)} (+{sum(x>0 for x in dP)}/-{sum(x<0 for x in dP)}) | Acc {ms(dA)}")
	for agg in ["zs","desir"]:
		for s in SEEDS:
			f=fid_of(pre,ds,arm,agg,s)
			if not done(f): continue
			k,v=best_run(f); n,b=size(f,k[1],k[2]); gn,gb=size(f,"GA","best_f1")
			print(f"  run {f} {agg:5s} {s} | best-anywhere {k[1]}/{k[2]}/{k[3]} {v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f} {n}n/{b:.0f}b | GA gens {gens.get(f)} | wall {wt(f):.2f} h | GA best_f1 shape {gn}n x {gb:.1f}b")
	for agg in ["zs","desir"]:
		fs=[fid_of(pre,ds,arm,agg,s) for s in SEEDS]; fs=[f for f in fs if done(f)]
		print(f"  SUM {agg:5s} n={len(fs)} wall {ms([wt(f) for f in fs])} h (total {sum(wt(f) for f in fs):.1f} h) | gens {ms([gens[f] for f in fs])} | GA best_f1 neurons {ms([size(f,'GA','best_f1')[0] for f in fs])} bits {ms([size(f,'GA','best_f1')[1] for f in fs])} | GA best_f1 val_cal F1 {ms([D[(f,'GA','best_f1','val_cal')][0] for f in fs])} FPR {ms([D[(f,'GA','best_f1','val_cal')][1] for f in fs])} Acc {ms([D[(f,'GA','best_f1','val_cal')][2] for f in fs])} | best_fitness F1 {ms([D[(f,'GA','best_fitness','val_cal')][0] for f in fs])} FPR {ms([D[(f,'GA','best_fitness','val_cal')][1] for f in fs])}")
