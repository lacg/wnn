import re, sys, numpy as np, glob
sys.path.insert(0,'/Users/lacg/wnn/logs/controller/ctrl70')
D="/Users/lacg/wnn/logs/controller/ctrl48/"
P="C-S2-d_wposx2_%s_rf_c200_dc_biasin_ms01_P_both_L1_yld_%s_full_s%d"
def units(tag):
	txt=open(D+tag+".out",errors="replace").read()
	hd=re.findall(r"\[stage-select\] HEADLINE stage=(\S+) genome=(\S+)",txt)[-1]
	has_h="HELD-OUT REPORT [HEADLINE-" in txt
	want=(lambda l:l.startswith("HEADLINE-")) if has_h else (lambda l:l==hd[0])
	out={};cur=None
	for line in txt.splitlines():
		m=re.search(r"HELD-OUT REPORT \[([^\]]+)\].*FRESH seed (\d+)",line)
		if m:
			cur=int(m.group(2)) if want(m.group(1)) and 99990101<=int(m.group(2))<=99990135 else None;continue
		if cur and "RESULT" in line:
			g=lambda k:float(re.search(k+r"=([\d.]+)",line).group(1))
			out[cur]=[g("stable"),g("err"),g("steady"),g("alt"),g("pos"),g("pos_hold"),g("dz_hold"),g("jerk"),g("effort"),g("dx"),g("dy")];cur=None
	return np.array([out[u] for u in sorted(out)])
r=np.load("/Users/lacg/wnn/logs/controller/ctrl73/data/n0g3gz_s1.npz")["units"]
asym=r[:,26:30]; c1=asym@[1,-1,1,-1]
print("c1 mean %+.3f sd %.3f"%(c1.mean(),c1.std()))
arms=[("G+T","a62","n0g3gz"),("l2b","a63","gt_l2b"),("l2bl3","a63","gt_l2bl3"),("l2a","a63","gt_l2a"),("l3","a63","gt_l3"),("l2al3","a63","gt_l2al3"),("l1","a63","gt_l1"),("l1l2b","a63","gt_l1l2b")]
A={}
for n,a,t in arms:
	A[n]=np.stack([units(P%(a,t,s)) for s in (1,2,3)])  # seed x unit x metric
cols=["stable","err","steady","alt","pos","pos_hold","dz_hold","jerk","effort","dx","dy"]
print("arm     "+" ".join(f"{c:>8s}" for c in cols))
for n in A:
	print(f"{n:7s} "+" ".join(f"{A[n][:,:,i].mean():8.4f}" for i in range(len(cols))))
g=A["G+T"]
print("\nper-unit (seed-mean) stats: err min/med/max/p90 | steady min/med/max | err slope on |c1|, r | steady slope on |c1|, r | cross-seed unit r (err) s1s2,s1s3,s2s3 | steady same")
for n in A:
	e=A[n][:,:,1].mean(0); s=A[n][:,:,2].mean(0)
	ke=np.polyfit(abs(c1),e,1)[0]; ks=np.polyfit(abs(c1),s,1)[0]
	cr=lambda i:[np.corrcoef(A[n][a,:,i],A[n][b,:,i])[0,1] for a,b in((0,1),(0,2),(1,2))]
	print(f"{n:7s} err {e.min():.3f}/{np.median(e):.3f}/{e.max():.3f} p90 {np.percentile(e,90):.3f} | st {s.min():.3f}/{np.median(s):.3f}/{s.max():.3f} | e~|c1| {ke:+.3f} r{np.corrcoef(abs(c1),e)[0,1]:+.2f} | s~|c1| {ks:+.3f} r{np.corrcoef(abs(c1),s)[0,1]:+.2f} | "+",".join("%.2f"%x for x in cr(1))+" | "+",".join("%.2f"%x for x in cr(2)))
print("\nvs G+T paired per unit (seed-mean): units lower err / steady / pos / pos_hold of 35; r(unit err arm, unit err G+T)")
for n in A:
	if n=="G+T":continue
	f=lambda i:int((A[n][:,:,i].mean(0)<g[:,:,i].mean(0)-1e-9).sum())
	print(f"{n:7s} err {f(1)}/35 steady {f(2)}/35 pos {f(4)}/35 pos_hold {f(5)}/35 alt {f(3)}/35 | r_err {np.corrcoef(A[n][:,:,1].mean(0),g[:,:,1].mean(0))[0,1]:.2f} r_steady {np.corrcoef(A[n][:,:,2].mean(0),g[:,:,2].mean(0))[0,1]:.2f}")
print("\nper seed: units lower than G+T same seed (err/steady)")
for n in ("l2b","l2bl3","l3","l2a"):
	print(n," ".join(f"s{s+1}: {int((A[n][s,:,1]<g[s,:,1]).sum())}/{int((A[n][s,:,2]<g[s,:,2]).sum())}" for s in range(3)))
us=sorted(int(u) for u in r[:,0])
print("\nworst 5 units by seed-mean steady and err:")
for n in ("G+T","l2b","l2bl3"):
	e=A[n][:,:,1].mean(0); s=A[n][:,:,2].mean(0)
	print(n,"steady:",[(us[i]%1000,round(s[i],3),round(c1[i],2)) for i in np.argsort(-s)[:5]],"err:",[(us[i]%1000,round(e[i],3),round(c1[i],2)) for i in np.argsort(-e)[:5]])
print("\nunits with stable<100:")
for n in A:
	for s in range(3):
		b=[(us[i]%1000,A[n][s,i,0]) for i in range(35) if A[n][s,i,0]<100]
		if b: print(n,"s%d"%(s+1),b)
