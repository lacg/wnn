"""CTRL-73 (TEST): where does G+T's +alt/+pos_hold come from? per-unit paired delta vs N0 (seed-mean), against |c1| and hold yaw offset."""
import numpy as np, glob
D = np.degrees
C = np.load("/tmp/ctrl73/rival.npz")["C"]; c1 = C[:, 21:25] @ [1, -1, 1, -1]
M = lambda p: np.stack([np.load(f)["units"] for f in glob.glob(f"/tmp/ctrl73/{p}_s*.npz")]).mean(0)
N, G = M("n0"), M("n0g3gz")
for name, j in (("alt", 6), ("pos", 5), ("pos_hold", 7)):
	d = G[:, j] - N[:, j]
	print(f"{name}: mean delta {1000*d.mean():+.2f} mm, units G+T higher {int((d>0).sum())}/35; r(|c1|, d) {np.corrcoef(abs(c1), d)[0,1]:+.2f}; r(d, delta |signed hold yaw|) {np.corrcoef(d, D(abs(G[:,22]))-D(abs(N[:,22])))[0,1]:+.2f}; r(d, delta hold roll) {np.corrcoef(d, D(G[:,16])-D(N[:,16]))[0,1]:+.2f}")
