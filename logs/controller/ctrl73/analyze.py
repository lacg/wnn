"""CTRL-73: gap table + paired per-axis deltas + per-unit + fails, G+T vs N0/G/T vs MPCOF (TEST, CPU re-scores).
wnn units cols: 0 unit,1 stable,2 err,3 steady,4 mid(0.3-1.6 s),5 pos,6 alt,7 pos_hold,8 jerk,9 rise,10 settle,11 itae,
12..25 axes [err r,p,y,|v| | hold r,p,y,|v| | signed hold r,p,y | fed-yaw drift | n | n_hold], 26..29 asym.
rival cols: 0 unit,1 stable,2 err,3 steady,4 pos,5 alt,6 pos_hold, 7..20 axes, 21..24 asym."""
import numpy as np, glob, math
D = math.degrees
R = np.load("/tmp/ctrl73/rival.npz"); C = R["C"]; Ar = R["A"]
ARMS = [("N0", "n0"), ("G", "n0g3"), ("T", "n0gz"), ("G+T", "n0g3gz")]
def load(p):
	return {int(f.split("_s")[-1][:-4]): np.load(f) for f in glob.glob(f"/tmp/ctrl73/{p}_s*.npz")}
W = {k: load(p) for k, p in ARMS}
def fails(u): return int(round(((1 - u[:, 1]) * 100).sum()))
def row(U):  # U: seeds x 35 x cols
	m = U.mean(axis=(0, 1)); a = m[12:26]
	e = [D(a[i]) for i in range(3)]; h = [D(a[4 + i]) for i in range(3)]; pre = [(e[i] - .2 * h[i]) / .8 for i in range(3)]
	early = (m[2] - .65 * m[4] - .2 * m[3]) / .15
	return m, e, h, pre, early
print("== gap table (TEST, CPU re-score, seed-mean over 35 units x 100 eps)")
print("arm     n  stable fails  err   | err r/p/y            | pre-hold r/p/y       | steady | hold r/p/y           | early(0-0.3) mid(0.3-1.6) | pos    alt    pos_hold | jerk")
for k, _ in ARMS:
	U = np.stack([W[k][s]["units"] for s in sorted(W[k])])
	m, e, h, pre, early = row(U); nf = sum(fails(W[k][s]["units"]) for s in W[k])
	print(f"{k:6s} {len(W[k]):2d} {100*m[1]:6.2f} {nf:3d}/{100*35*len(W[k])} {m[2]:.3f} | " + " / ".join(f"{x:.3f}" for x in e) + " | " + " / ".join(f"{x:.3f}" for x in pre)
	      + f" | {m[3]:.3f}  | " + " / ".join(f"{x:.3f}" for x in h) + f" | {early:.3f} {m[4]:.3f} | {m[5]:.4f} {m[6]:.4f} {m[7]:.4f} | {m[8]:.4g}")
for lab, X in (("MPCOF-C", C), ("MPCOF-A", Ar)):
	m = X.mean(0); e = [D(m[7 + i]) for i in range(3)]; h = [D(m[11 + i]) for i in range(3)]; pre = [(e[i] - .2 * h[i]) / .8 for i in range(3)]
	print(f"{lab:8s}  {100*m[1]:6.2f}   0     {m[2]:.3f} | " + " / ".join(f"{x:.3f}" for x in e) + " | " + " / ".join(f"{x:.3f}" for x in pre)
	      + f" | {m[3]:.3f}  | " + " / ".join(f"{x:.3f}" for x in h) + f" |  n/a  | {m[4]:.4f} {m[5]:.4f} {m[6]:.4f}")
print("\n== per seed (CPU): seed | err steady pos alt pos_hold fails | err r/p/y | hold r/p/y | |signed hold yaw| | mid early")
PS = {}
for k, _ in ARMS:
	print(k)
	for s in sorted(W[k]):
		u = W[k][s]["units"]; m = u.mean(0); a = m[12:26]
		early = (m[2] - .65 * m[4] - .2 * m[3]) / .15
		v = dict(err=m[2], steady=m[3], pos=m[5], alt=m[6], ph=m[7], fails=fails(u), er=D(a[0]), ep=D(a[1]), ey=D(a[2]), hr=D(a[4]), hp=D(a[5]), hy=D(a[6]),
		         sy=D(np.abs(u[:, 22]).mean()), mid=m[4], early=early, jerk=m[8], pre_y=(D(a[2]) - .2 * D(a[6])) / .8, pre_r=(D(a[0]) - .2*D(a[4]))/.8, pre_p=(D(a[1]) - .2*D(a[5]))/.8)
		PS[(k, s)] = v
		print(f"  s{s:<2d} {v['err']:.3f} {v['steady']:.3f} {v['pos']:.4f} {v['alt']:.4f} {v['ph']:.4f} {v['fails']} | {v['er']:.3f} {v['ep']:.3f} {v['ey']:.3f} | {v['hr']:.3f} {v['hp']:.3f} {v['hy']:.3f} | {v['sy']:.3f} | {v['mid']:.3f} {v['early']:.3f}")
print("\n== paired vs N0 (same seed): mean delta, arm lower on k/n")
keys = ["err", "steady", "pos", "alt", "ph", "er", "ep", "ey", "pre_r", "pre_p", "pre_y", "hr", "hp", "hy", "sy", "mid", "early", "jerk"]
for k in ("G", "T", "G+T"):
	ss = sorted(set(s for (a, s) in PS if a == k) & set(s for (a, s) in PS if a == "N0"))
	print(f"{k} vs N0 (n={len(ss)}): " + " | ".join(f"{q} {np.mean([PS[(k,s)][q]-PS[('N0',s)][q] for s in ss]):+.4f} {sum(PS[(k,s)][q]<PS[('N0',s)][q] for s in ss)}/{len(ss)}" for q in keys))
for k in ("G", "T"):
	ss = sorted(set(s for (a, s) in PS if a == k))
	print(f"G+T vs {k}: " + " | ".join(f"{q} {np.mean([PS[('G+T',s)][q]-PS[(k,s)][q] for s in ss]):+.4f} {sum(PS[('G+T',s)][q]<PS[(k,s)][q] for s in ss)}/{len(ss)}" for q in keys[:8] + ['hy', 'sy']))
np.save("/tmp/ctrl73/ps.npy", np.array([PS], dtype=object))
