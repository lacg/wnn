"""CTRL-73 VAL diagnosis: implied yaw-bank delta per action in the hold, d_k = sp_k - 0.7*sp_{k-1} (delta law, --delta-leak 0.7,
action every 10 ticks). If the WNN holds a wear trim through the leaky accumulator, d_k ~ 0.3*trim every action."""
import numpy as np
for lab, f in {"N0 s1": "/tmp/ctrl70/trace_n0.npz", "G+T s1": "/tmp/ctrl73/trace_gt.npz"}.items():
	T = np.load(f)["trace"].astype(float); key = T[:,0]*1000+T[:,1]
	D, S, U = [], [], []
	for k in np.unique(key):
		s = key == k; t = T[s,2]; sp = T[s,19]
		a = sp[(t % 10) == 9] if False else sp[::10]
		tt = t[::10]
		d = a[1:] - 0.7*a[:-1]; h = tt[1:] >= 1600
		D.append(d[h]); S.append(a[1:][h]); U.append(np.full(h.sum(), T[s,0][0]))
	D, S, U = map(np.concatenate, (D, S, U))
	print(f"== {lab} hold actions: |held sp_r| mean {abs(S).mean():.3f} | implied delta: mean {D.mean():+.4f} |d| mean {abs(D).mean():.4f}; "
	      f"share with |d| < 0.01 rad/s {100*(abs(D)<0.01).mean():.0f}%; sign(d)==sign(sp) {100*(np.sign(D)==np.sign(S)).mean():.0f}%")
	for u in np.unique(U):
		m = U == u
		print(f"   val unit {int(u)}: mean sp {S[m].mean():+.3f}, mean d {D[m].mean():+.4f} (0.3*mean sp = {0.3*S[m].mean():+.4f}), std d {D[m].std():.4f}")
