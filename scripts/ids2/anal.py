from load import *
import numpy as np
from scipy import stats
D,meta,gens=load()
CTRL={"cicids":"Wa-CTRL","ciciot":"Wc-CTRL"}
def arms_for(ds,minn):
	arms=sorted({k[1] for k in meta if k[0]==ds})
	return [a for a in arms if len([1 for k in meta if k[0]==ds and k[1]==a])>=minn]
def seeds_for(ds,arms):
	ss=None
	for a in arms:
		s={k[2] for k in meta if k[0]==ds and k[1]==a}
		ss=s if ss is None else ss&s
	return sorted(ss)
def mat(ds,arms,seeds,gt,md,idx,ph="GA"):
	return np.array([[D[(ds,a,s,ph,gt,md)][idx] for s in seeds] for a in arms])
def omnibus(M):
	# two-way additive (arm + seed), F test for arm
	a,n=M.shape; gm=M.mean()
	ssa=n*((M.mean(1)-gm)**2).sum(); sss=a*((M.mean(0)-gm)**2).sum()
	sst=((M-gm)**2).sum(); sse=sst-ssa-sss; dfa=a-1; dfe=(a-1)*(n-1)
	F=(ssa/dfa)/(sse/dfe); return F,1-stats.f.cdf(F,dfa,dfe),np.sqrt(sse/dfe),np.sqrt(sss/(n-1)/a)
def paired(x,y):
	d=x-y; n=len(d); m=d.mean(); sd=d.std(ddof=1); se=sd/np.sqrt(n); tc=stats.t.ppf(.975,n-1)
	t=m/se if se>0 else np.inf; p=2*(1-stats.t.cdf(abs(t),n-1))
	return m,sd,m-tc*se,m+tc*se,p,int((d>0).sum()),int((d<0).sum())
for ds,minn in [("cicids",3),("ciciot",5),("ciciot",3)]:
	arms=arms_for(ds,minn); seeds=seeds_for(ds,arms)
	if minn==3 and ds=="ciciot":
		seeds=[20405]; # not useful
		arms=[a for a in arms_for(ds,3)]; seeds=seeds_for(ds,arms)
	print(f"\n##### {ds} arms={arms} seeds={seeds}")
	for gt in ["best_fitness","best_f1","best_fpr"]:
		for idx,lab in [(0,"F1"),(1,"FPR")]:
			M=mat(ds,arms,seeds,gt,"val_cal",idx)
			F,p,sres,_=omnibus(M)
			mu=M.mean(1); order=np.argsort(-mu if idx==0 else mu)
			print(f" GA {gt} val_cal {lab}: omnibus F={F:.2f} p={p:.3f} resid(paired)SD={sres:.3f}  "+
			      " ".join(f"{arms[i]}={mu[i]:.2f}±{M[i].std(ddof=1):.2f}" for i in order))
			c=arms.index(CTRL[ds]); top=order[0]; ru=order[1]
			for nm,j in [("top-vs-CTRL",c),("top-vs-runnerup",ru)]:
				if j==top: print(f"    {nm}: top IS control"); continue
				m,sd,lo,hi,pp,npos,nneg=paired(M[top],M[j])
				k=len(arms)-1
				print(f"    {nm} {arms[top]}-{arms[j]}: d={m:+.3f} sd={sd:.3f} CI95[{lo:+.3f},{hi:+.3f}] p={pp:.3f} (Bonf x{k}: {min(1,pp*k):.3f}) signs +{npos}/-{nneg}")
