import numpy as np, itertools, json
OUT="/Users/lacg/wnn/logs/ids/ids77"
lab=np.load(f"{OUT}/val_labels.npz"); yb=lab["y_bin"]; yv=lab["y_multi"]; P=(yb==1).sum(); N=(yb==0).sum()
names=json.load(open(f"{OUT}/meta.json"))["names"]; FUZ=names.index("Fuzzers")
def rd(sc):
	m=sc[:,1]-sc[:,0]; o=np.argsort(-m,kind="stable"); ms=m[o]; ys=yb[o]
	tp=np.cumsum(ys==1); fp=np.cumsum(ys==0); last=np.r_[ms[1:]!=ms[:-1],True]; tp,fp=tp[last],fp[last]; fn=P-tp; tn=N-fp
	f1=(2*tp/np.maximum(2*tp+fp+fn,1)+2*tn/np.maximum(2*tn+fn+fp,1))/2; i=int(np.argmax(f1)); pred=m>=ms[last][i]
	return f"F1 {f1[i]*100:.3f} FPR {fp[i]/N*100:.3f} rec {tp[i]/P*100:.2f} fuz {pred[yv==FUZ].mean()*100:.2f}"
S={f:np.load(f"{OUT}/val_{f}_ga_neurons.npz")["scores"] for f in (7591,7592,7593)}
G={f:np.load(f"{OUT}/val_{f}_grid_search.npz")["scores"] for f in (7591,7592,7593)}
for f in S: print("single GA",f,rd(S[f]))
for a,b in itertools.combinations(S,2): print("pair GA",a,b,rd((S[a]+S[b])/2))
print("triple GA",rd(sum(S.values())/3))
print("triple GA+3 grid",rd((sum(S.values())+sum(G.values()))/6))
