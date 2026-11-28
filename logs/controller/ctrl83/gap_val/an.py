import numpy as np, sys
PFX = sys.argv[1]
np.set_printoptions(precision=3, suppress=True)
Us, Es, Rs = [], [], []
for s in (1, 2, 3):
	z = np.load(PFX.format(s=s)); U, E, R = z["units"], z["eps"], z["riv"]
	Us.append(np.c_[np.full(len(U), s), U]); Es.append(np.c_[np.full(len(E), s), E]); Rs.append(R)
	nf = int(round(((1 - U[:, 1]) * 100).sum())); Cr = R[R[:, 1] == 0]; Ar = R[R[:, 1] == 1]
	print(f"s{s} WNN fails {nf} err {U[:,2].mean():.3f} steady {U[:,3].mean():.3f} | whole r/p/y {U[:,4:7].mean(0)} hold r/p/y {U[:,8:11].mean(0)}"
	      f" | C fails {int(round(((1-Cr[:,2])*100).sum()))} err {Cr[:,3].mean():.3f} steady {Cr[:,4].mean():.3f} | A err {Ar[:,3].mean():.3f}")
U = np.vstack(Us); E = np.vstack(Es); R = np.vstack(Rs)
a = U[:, 13:17]; ab = a.mean(1, keepdims=True); an = a / ab
roll = an[:, 3] - an[:, 1]; pitch = an[:, 2] - an[:, 0]; yaw = an[:, 0] - an[:, 1] + an[:, 2] - an[:, 3]
tilt = np.hypot(roll, pitch); amin = a.min(1)
C = R[R[:, 1] == 0]; Cerr = C[:, 3]; Cst = C[:, 2]
werr = U[:, 3]; wf = (1 - U[:, 2]) * 100; cf = (1 - Cst) * 100
# capped means per unit
cap = lambda x: np.minimum(x, 8.0)
uid = E[:, 0] * 1e10 + E[:, 1]
d_ep = cap(E[:, 3]) - cap(E[:, 7])
units = np.unique(U[:, 0] * 1e10 + U[:, 1])
d_unit = np.array([d_ep[uid == k].mean() for k in units])
print("\nper-unit capped err: WNN", np.mean([cap(E[uid==k,3]).mean() for k in units]).round(3), "C", np.mean([cap(E[uid==k,7]).mean() for k in units]).round(3))
def corr(x, y): return np.corrcoef(x, y)[0, 1]
print("\nper-unit corr with d(WNN-C capped err), n=60:")
for nm, x in [("|roll imb|", abs(roll)), ("|pitch imb|", abs(pitch)), ("|tilt imb|", tilt), ("|yaw imb|", abs(yaw)), ("min motor", amin), ("mean motor", ab[:, 0])]:
	print(f"  {nm:12s} r={corr(x, d_unit):+.2f}   WNNfails r={corr(x, wf):+.2f}  Cfails r={corr(x, cf):+.2f}")
o = np.argsort(-wf)
print("\nworst units: seed unit WNNfails Cfails WNNerr Cerr tilt yaw min mean | WNN hold r/p/y")
for i in o[:12]:
	print(f"  s{int(U[i,0])} {int(U[i,1])} {wf[i]:5.0f} {cf[i]:5.0f} {U[i,3]:6.2f} {Cerr[i]:6.2f} {tilt[i]:.3f} {yaw[i]:+.3f} {amin[i]:.3f} {ab[i,0]:.3f} | {U[i,9:12]}")
tot_w, tot_c = wf.sum(), cf.sum()
print(f"\nfails total WNN {tot_w:.0f} C {tot_c:.0f}; top-5 units carry WNN {wf[o[:5]].sum():.0f}, top-10 {wf[o[:10]].sum():.0f}")
# episode-level
wfail = (E[:, 3] >= 5.0) | (E[:, 3] >= 8.0)
mass, lag = E[:, 4], E[:, 6]
print("\nepisode corr (n=%d) with d capped err: mass r=%+.3f lag r=%+.3f cf r=%+.3f" % (len(E), corr(mass, d_ep), corr(lag, d_ep), corr(E[:,5], d_ep)))
# within-unit corr (remove unit effect)
def demean(x): 
	y = x.copy()
	for k in units: m = uid == k; y[m] -= x[m].mean()
	return y
print("within-unit corr: mass r=%+.3f lag r=%+.3f  (WNN err alone: mass %+.3f lag %+.3f | C err: mass %+.3f lag %+.3f)" % (
	corr(demean(mass), demean(d_ep)), corr(demean(lag), demean(d_ep)), corr(demean(mass), demean(cap(E[:,3]))), corr(demean(lag), demean(cap(E[:,3]))),
	corr(demean(mass), demean(cap(E[:,7]))), corr(demean(lag), demean(cap(E[:,7])))))
mn = np.median(mass)
for nm, x, edges in [("mass/nominal", mass, None), ("lag s", lag, [0.15, 0.1875, 0.225, 0.2625, 0.30001])]:
	if edges is None: edges = np.quantile(x, [0, .25, .5, .75, 1.0]); edges[-1] += 1e-9
	print(f"\n{nm} quartiles: lo-hi | WNN err | C err | d | WNN fail% | C fail%")
	for lo, hi in zip(edges[:-1], edges[1:]):
		m = (x >= lo) & (x < hi)
		print(f"  {lo:.3f}-{hi:.3f} {cap(E[m,3]).mean():6.3f} {cap(E[m,7]).mean():6.3f} {d_ep[m].mean():+6.3f} {100*(E[m,3]>=8).mean():5.2f} {100*(1-E[m,8]).mean():5.2f}")
# fails excluding bad units
bad = np.isin(uid // 1e10 * 1e10 , [0])
print("\nlag min/max", lag.min(), lag.max(), np.percentile(lag,[10,50,90]))
print("cf min/max", E[:,5].min(), E[:,5].max())
for lo, hi in zip(np.quantile(E[:,5],[0,.25,.5,.75]), np.quantile(E[:,5],[.25,.5,.75,1])+[0,0,0,1e-9]):
	m = (E[:,5]>=lo)&(E[:,5]<hi)
	print(f"  cf {lo:+.3f}..{hi:+.3f} WNN {cap(E[m,3]).mean():.3f} C {cap(E[m,7]).mean():.3f} d {d_ep[m].mean():+.3f}")
good = np.array([wf[i] < 10 for i in range(len(U))])
gk = units[good]
m = np.isin(uid, gk)
print(f"\nexcluding units with WNN fails>=10 ({(~good).sum()} units): WNN capped err {cap(E[m,3]).mean():.3f} C {cap(E[m,7]).mean():.3f} d {d_ep[m].mean():+.3f}; WNN steady {U[good,4].mean():.3f}")
print("all units: d", d_ep.mean().round(3), " share of total d from bad units:", (d_ep[~m].sum()/d_ep.sum()).round(3))
# per-axis in good units
print("good-unit WNN whole r/p/y", U[good,5:8].mean(0), "hold r/p/y", U[good,9:12].mean(0), "steady", U[good,4].mean())
# signed relation: is hold error aligned with imbalance direction in good units?
print("good units corr |pitch imb| vs WNN hold pitch r=%.2f ; |roll imb| vs hold roll r=%.2f; |yaw imb| vs hold yaw r=%.2f" % (
	corr(abs(pitch[good]), U[good,10]), corr(abs(roll[good]), U[good,9]), corr(abs(yaw[good]), U[good,11])))
