import json, glob, numpy as np
OUT="/Users/lacg/wnn/logs/ids/ids77"
lab=np.load(f"{OUT}/val_labels.npz"); yb=lab["y_bin"]; P=(yb==1).sum(); N=(yb==0).sum()
def curve(s):
	o=np.argsort(-s, kind="stable"); ss=s[o]; ys=yb[o]
	tp=np.cumsum(ys==1); fp=np.cumsum(ys==0); last=np.r_[ss[1:]!=ss[:-1],True]
	return tp[last]/P, fp[last]/N, tp[last], fp[last]
def f1m(tp,fp):
	fn=P-tp; tn=N-fp
	a=2*tp/np.maximum(2*tp+fp+fn,1); b=2*tn/np.maximum(2*tn+fn+fp,1); return (a+b)/2
def at_fpr(s, target):
	r,f,tp,fp=curve(s); ok=f<=target; i=np.where(ok)[0][-1]; return r[i]*100, f[i]*100, f1m(tp[i],fp[i])*100
srcs={}
for fl in (7591,7592,7593):
	for ph in ("ga_neurons","grid_search"):
		srcs[f"wnn_{fl}_{ph[:4]}"]=np.load(f"{OUT}/val_{fl}_{ph}.npz")["margin"]
for p in sorted(glob.glob(f"{OUT}/pv_*.npy")):
	srcs[p.split("/")[-1][3:-4]]=np.load(p).astype(np.float64)
print("source               | rec@FPR<=0.20 F1 | rec@FPR<=0.25 F1 | rec@FPR<=0.30 F1 | rec@FPR<=0.40 F1 | maxF1 (sweep) @FPR rec")
for k,s in srcs.items():
	cells=[]
	for t in (0.002,0.0025,0.003,0.004):
		r,f,F=at_fpr(s,t); cells.append(f"{r:5.2f} {F:6.3f}")
	r,f,tp,fp=curve(s); F=f1m(tp,fp); i=int(np.argmax(F))
	print(f"{k:20s} | "+" | ".join(cells)+f" | {F[i]*100:6.3f} @{f[i]*100:5.3f} {r[i]*100:5.2f}")
