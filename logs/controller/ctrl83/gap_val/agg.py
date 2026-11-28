import numpy as np, sys, os
cap = lambda x: np.minimum(x, 8.0)
for c in sys.argv[1:]:
	out = []
	for s in (1, 2, 3):
		f = f"/tmp/ctrl83val/{c}_s{s}.npz" if c not in ("v1", "zhang") else {"v1": f"/tmp/ctrl83val/v1_s{s}_riv.npz", "zhang": f"/tmp/ctrl83val/zhang/s{s}_riv.npz"}[c]
		if not os.path.exists(f): continue
		z = np.load(f); U, E, R = z["units"], z["eps"], z["riv"]; Cr = R[R[:, 1] == 0]
		out.append([int(round(((1-U[:,1])*100).sum())), U[:,2].mean(), U[:,3].mean(), *U[:,8:11].mean(0), int(round(((1-Cr[:,2])*100).sum())), Cr[:,3].mean(), Cr[:,4].mean()])
	o = np.array(out)
	if o.ndim < 2: continue
	print(f"{c:7s} seeds {len(o)} | WNN fails {o[:,0].astype(int)} err {o[:,1].mean():.3f} steady {o[:,2].mean():.3f} hold r/p/y {o[:,3:6].mean(0).round(3)} | C fails {o[:,6].astype(int)} err {o[:,7].mean():.3f} steady {o[:,8].mean():.3f} | d err {o[:,1].mean()-o[:,7].mean():+.3f} d steady {o[:,2].mean()-o[:,8].mean():+.3f}")
