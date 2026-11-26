import numpy as np, glob
D=np.degrees
def L(p): return np.load(p)["units"]
G={s:L(f"/Users/lacg/wnn/logs/controller/ctrl73/data/n0g3gz_s{s}.npz") for s in (1,2,3)}
A={"G+T":G}
for n in ("l2b","l2bl3"): A[n]={s:L(f"/tmp/ctrl77gap/data/{n}_s{s}.npz") for s in (1,2,3)}
A["l2a(s1)"]={1:L("/tmp/ctrl77gap/data/l2a_s1.npz")}; A["l1(s1)"]={1:L("/tmp/ctrl77gap/data/l1_s1.npz")}; A["G+T(s1)"]={1:G[1]}
c1=G[1][:,26:30]@[1,-1,1,-1]
def v(u):
	m=u.mean(0);a=m[12:26]
	d=dict(stable=100*m[1],fails=int(round(((1-u[:,1])*100).sum())),err=m[2],steady=m[3],mid=m[4],early=(m[2]-.65*m[4]-.2*m[3])/.15,pos=m[5],alt=m[6],ph=m[7],jerk=m[8],
	 er=D(a[0]),ep=D(a[1]),ey=D(a[2]),hr=D(a[4]),hp=D(a[5]),hy=D(a[6]),sy=D(np.abs(u[:,22]).mean()))
	for x in "rpy": d["pre_"+x]=(d["e"+x]-.2*d["h"+x])/.8
	d["off_slope"]=np.polyfit(c1,D(u[:,22]),1)[0]; d["off_r"]=np.corrcoef(c1,D(u[:,22]))[0,1]
	d["err_c1"]=np.polyfit(abs(c1),u[:,2],1)[0]
	return d
keys=["stable","fails","err","er","ep","ey","pre_r","pre_p","pre_y","early","mid","steady","hr","hp","hy","sy","off_slope","off_r","err_c1","pos","alt","ph","jerk"]
print(f"{'':10s}"+" ".join(f"{k:>8s}" for k in keys))
for n in A:
	R=[v(A[n][s]) for s in sorted(A[n])]
	for s,r in zip(sorted(A[n]),R): print(f"{n[:6]:6s} s{s}  "+" ".join(f"{r[k]:8.4f}" for k in keys))
	if len(R)>1: print(f"{n:6s} MEAN"+" ".join(f"{np.mean([r[k] for r in R]):8.4f}" for k in keys))
# seed-mean per-unit offsets
for n in ("G+T","l2b","l2bl3"):
	U=np.stack([A[n][s] for s in (1,2,3)]).mean(0)
	print(n,"seed-mean signed hold yaw slope on c1 %.3f r %.2f |mean offset| %.3f ; err~|c1| %.3f"%(np.polyfit(c1,D(U[:,22]),1)[0],np.corrcoef(c1,D(U[:,22]))[0,1],abs(D(U[:,22])).mean(),np.polyfit(abs(c1),U[:,2],1)[0]))
# episode distribution
for n in ("l2b","l2bl3"):
	for s in (1,2,3):
		e=np.load(f"/tmp/ctrl77gap/data/{n}_s{s}.npz")["eps"][:,2]; g=np.load(f"/Users/lacg/wnn/logs/controller/ctrl73/data/n0g3gz_s{s}.npz")["eps"][:,2]
		print(n,s,"eps p50/p90/p99/max %.2f/%.2f/%.2f/%.2f | G+T %.2f/%.2f/%.2f/%.2f | arm lower on %d/3500 eps; eps>4: %d vs %d; eps>3: %d vs %d"%(*np.percentile(e,[50,90,99]),e.max(),*np.percentile(g,[50,90,99]),g.max(),(e<g).sum(),(e>4).sum(),(g>4).sum(),(e>3).sum(),(g>3).sum()))
		E=np.load(f"/tmp/ctrl77gap/data/{n}_s{s}.npz")["eps"]
		for (u,k) in ((99990123,16),(99990112,86),(99990123,33),(99990118,1)):
			i=np.where((E[:,0]==u)&(E[:,1]==k))[0][0]; print("    u%d ep%d arm %.2f G+T %.2f"%(u%1000,k,e[i],g[i]),end="")
		print()
