import json, numpy as np
OUT="/Users/lacg/wnn/logs/ids/ids77"
lab=np.load(f"{OUT}/val_labels.npz"); yv=lab["y_multi"]; yb=lab["y_bin"]
meta=json.load(open(f"{OUT}/meta.json")); names=meta["names"]
recs=[json.loads(l) for l in open(f"{OUT}/wnn_val_diag.jsonl")]
def f1m(tp,fp,fn,tn):
	p1=tp/max(tp+fp,1); r1=tp/max(tp+fn,1); a=2*p1*r1/max(p1+r1,1e-12)
	p0=tn/max(tn+fn,1); r0=tn/max(tn+fp,1); b=2*p0*r0/max(p0+r0,1e-12); return (a+b)/2
for r in recs:
	z=np.load(f"{OUT}/val_{r['flow']}_{r['phase']}.npz"); m=z["margin"]; sc=z["scores"]; tau=r["tau"]
	pred=m>=tau
	fn=(yb==1)&~pred; fp=(yb==0)&pred
	u=np.unique(m)
	# sweep all distinct thresholds: max F1 on VAL and F1 at matching FPR
	order=np.argsort(-m); ms=m[order]; ys=yb[order]
	tpc=np.cumsum(ys==1); fpc=np.cumsum(ys==0); P=(yb==1).sum(); N=(yb==0).sum()
	last=np.r_[ms[1:]!=ms[:-1],True]
	tp=tpc[last]; fpv=fpc[last]
	f1s=np.array([f1m(t,f,P-t,N-f) for t,f in zip(tp,fpv)])
	i=int(np.argmax(f1s))
	print(f"\n== {r['flow']} {r['phase']} n={r['n_neurons']} valF1={r['val_f1']:.3f} FPR={r['val_fpr']:.3f} tau={tau:.5f} distinct_margins={len(u)}  sweep-max F1={f1s[i]*100:.3f} at FPR={fpv[i]/N*100:.3f} recall={tp[i]/P*100:.2f}")
	print(f"   scores: c0 (normal) min/med/max {sc[:,0].min():.3f}/{np.median(sc[:,0]):.3f}/{sc[:,0].max():.3f}  c1 {sc[:,1].min():.3f}/{np.median(sc[:,1]):.3f}/{sc[:,1].max():.3f}")
	# FN anatomy
	s1=sc[:,1]; s0=sc[:,0]
	for nm,mask in (("FN",fn),("TP",(yb==1)&pred),("FP",fp),("TN",(yb==0)&~pred)):
		print(f"   {nm:2s} n={mask.sum():6d}  attack-cluster score q10/50/90 = {np.quantile(s1[mask],[.1,.5,.9]).round(3)}  normal-cluster = {np.quantile(s0[mask],[.1,.5,.9]).round(3)}  margin q10/50/90 = {np.quantile(m[mask],[.1,.5,.9]).round(4)}")
	# FN by how far below tau
	d=m[fn]
	print("   FN margin bins: |m-tau|<0.01:",int((tau-d<0.01).sum())," <0.05:",int((tau-d<0.05).sum())," <0.1:",int((tau-d<0.1).sum()), " total",int(fn.sum()))
	# rows both clusters ~ silent
	sil=(s0<0.05)&(s1<0.05)
	print(f"   both-cluster-silent (<5% firing) rows: attack {int((sil&(yb==1)).sum())} benign {int((sil&(yb==0)).sum())}; FN silent {int((sil&fn).sum())}")
	cat=r["cat"]
	print("   per-cat recall:", {k:f"{v[0]}/{v[1]}" for k,v in cat.items()})
