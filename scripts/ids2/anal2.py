from anal import *
import itertools
print("\n\n=========== 35-column scans (GA phase, all genome types x 7 modes) ===========")
for ds,arms,seeds,A,B in [("cicids",arms_for("cicids",3),seeds_for("cicids",arms_for("cicids",3)),"CE20","Wa-CTRL"),
                          ("cicids",arms_for("cicids",3),seeds_for("cicids",arms_for("cicids",3)),"CE20","B15-AC"),
                          ("ciciot",arms_for("ciciot",5),seeds_for("ciciot",arms_for("ciciot",5)),"B15-AC","Wc-CTRL"),
                          ("ciciot",arms_for("ciciot",5),seeds_for("ciciot",arms_for("ciciot",5)),"B15-AC","B10-CE")]:
	winF=winP=dom=domd=0; bothsig=0; dF=[];dP=[]
	for gt in GT:
		for md in MODES:
			x=np.array([D[(ds,A,s,"GA",gt,md)][:2] for s in seeds]); y=np.array([D[(ds,B,s,"GA",gt,md)][:2] for s in seeds])
			mf=(x[:,0]-y[:,0]).mean(); mp=(x[:,1]-y[:,1]).mean(); dF.append(mf); dP.append(mp)
			winF+=mf>0; winP+=mp<0; dom+=(mf>0 and mp<0); domd+=(mf<0 and mp>0)
	print(f"{ds} {A} vs {B} n={len(seeds)}: F1 higher {winF}/35, FPR lower {winP}/35, A dominates {dom}/35, B dominates {domd}/35; median dF1={np.median(dF):+.3f} median dFPR={np.median(dP):+.3f}")
# arm rank-1 counts per column (by F1 and by FPR)
for ds,minn in [("cicids",3),("ciciot",5)]:
	arms=arms_for(ds,minn); seeds=seeds_for(ds,arms)
	c1=dict.fromkeys(arms,0); c2=dict.fromkeys(arms,0); nd=dict.fromkeys(arms,0)
	for gt in GT:
		for md in MODES:
			F=mat(ds,arms,seeds,gt,md,0).mean(1); P=mat(ds,arms,seeds,gt,md,1).mean(1)
			c1[arms[int(np.argmax(F))]]+=1; c2[arms[int(np.argmin(P))]]+=1
			for i in range(len(arms)):
				if not any((F[j]>=F[i] and P[j]<=P[i] and (F[j]>F[i] or P[j]<P[i])) for j in range(len(arms)) if j!=i): nd[arms[i]]+=1
	print(f"{ds}: top-F1 column count {c1}\n        top-FPR column count {c2}\n        non-dominated (F1,FPR) count {nd}")
# best-of-N null inflation: E[max of k means] under null with paired resid sd
for ds,minn,sd in [("cicids",3,0.064),("ciciot",5,0.18)]:
	arms=arms_for(ds,minn); n=len(seeds_for(ds,arms)); k=len(arms)
	rng=np.random.default_rng(0); z=rng.standard_normal((200000,k,n))*sd
	mx=z.mean(2); gap=np.sort(mx,1)
	print(f"{ds}: k={k} n={n} resid sd={sd}: null E[max arm mean - grand]={ (mx.max(1)-mx.mean(1)).mean():.3f}, E[top-2nd]={(gap[:,-1]-gap[:,-2]).mean():.3f}, 95th pct top-minus-ctrl={np.percentile(mx.max(1)-mx[:,0],95):.3f}")
# budget
print("\n=== GA generations (iteration rows, ga_neurons) ===")
for ds,minn in [("cicids",3),("ciciot",5)]:
	arms=arms_for(ds,minn); seeds=seeds_for(ds,arms)
	G=np.array([[gens[(ds,a,s)] for s in seeds] for a in arms],float)
	Fm=mat(ds,arms,seeds,"best_f1","val_cal",0); Pm=mat(ds,arms,seeds,"best_f1","val_cal",1)
	print(ds," ".join(f"{a}={G[i].mean():.0f}" for i,a in enumerate(arms)))
	r_arm=np.corrcoef(G.mean(1),Fm.mean(1))[0,1]
	Gr=G-G.mean(1,keepdims=True)-G.mean(0,keepdims=True)+G.mean(); Fr=Fm-Fm.mean(1,keepdims=True)-Fm.mean(0,keepdims=True)+Fm.mean()
	Pr=Pm-Pm.mean(1,keepdims=True)-Pm.mean(0,keepdims=True)+Pm.mean()
	print(f"   r_arm(gens,F1)={r_arm:+.2f}  r_arm(gens,FPR)={np.corrcoef(G.mean(1),Pm.mean(1))[0,1]:+.2f}  within-arm resid r(gens,F1)={np.corrcoef(Gr.ravel(),Fr.ravel())[0,1]:+.2f} r(gens,FPR)={np.corrcoef(Gr.ravel(),Pr.ravel())[0,1]:+.2f}")
# era interaction ciciot: B15-AC - CTRL by seed
arms=arms_for("ciciot",5); seeds=seeds_for("ciciot",arms)
for gt in ["best_f1","best_fitness"]:
	x=mat("ciciot",["B15-AC","Wc-CTRL"],seeds,gt,"val_cal",0); y=mat("ciciot",["B15-AC","Wc-CTRL"],seeds,gt,"val_cal",1)
	print("ciciot",gt,"per-seed dF1",np.round(x[0]-x[1],3),"dFPR",np.round(y[0]-y[1],3))
x=mat("cicids",["CE20","Wa-CTRL"],[20403,20404,20405],"best_f1","val_cal",0);y=mat("cicids",["CE20","Wa-CTRL"],[20403,20404,20405],"best_f1","val_cal",1)
print("cicids best_f1 per-seed dF1",np.round(x[0]-x[1],3),"dFPR",np.round(y[0]-y[1],3))
