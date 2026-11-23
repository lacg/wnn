"""CTRL-71 no-flight pre-check (VAL diagnosis only, never TEST, not a result): the gyro_z ladder phased_ga
actually fits for the flown N0 s1 argv, flag off vs --gyro-z-ladder dense, applied to the CTRL-70 VAL traces
(/tmp/ctrl70/trace_{n0,y3}.npz = 3 val units x 100 episodes of the N0 s1 / Y3 s1 winners).
Prints, per time window: share of ticks beyond the outer cut (the prereg gate: hold <= 5 %, every transient
window <= 25 %), and the share of the yaw-rate variance carried by the 9-bin gyro_z code (eta^2; the yaw rate is
the centred +-5-tick finite difference of the traced true yaw angle).
Run: PYTHONPATH=<tree>/src python precheck_val.py <tree>/tests"""
import math, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import controller_ctrl71_gyro_z_ladder as t71

W = [(0, 150), (150, 300), (300, 600), (600, 1000), (1000, 1600), (1600, 2000)]


def ladder(extra):
	d = t71.dump(extra)
	i = d["names"].index("gyro_z")
	return np.array([float(x) for x in d["thresholds"][i * d["bpf"]:(i + 1) * d["bpf"]]]), d["token"]


def yaw_rate(T):
	w, x, y, z = T[:, 6], T[:, 7], T[:, 8], T[:, 9]
	yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
	r = np.full(len(yaw), np.nan)
	r[5:-5] = (yaw[10:] - yaw[:-10]) / 0.010
	r[(T[:, 2] < 5) | (T[:, 2] >= 1995)] = np.nan   # no difference across an episode boundary
	return r


def eta2(code, y):
	ok = ~np.isnan(y); code, y = code[ok], y[ok]
	tot = y.var()
	between = sum((code == c).mean() * (y[code == c].mean() - y.mean()) ** 2 for c in np.unique(code))
	return between / tot if tot > 0 else float("nan")


lads = {"flag off (flown N0)": ladder([]), "--gyro-z-ladder dense": ladder(["--gyro-z-ladder", "dense"])}
for lab in ("n0", "y3"):
	T = np.load(f"/tmp/ctrl70/trace_{lab}.npz")["trace"].astype(float)
	t, gz, r = T[:, 2], T[:, 5], yaw_rate(T)
	for name, (thr, tok) in lads.items():
		code = np.searchsorted(thr, gz)
		sat = (gz < thr[0]) | (gz > thr[-1])
		print(f"{lab} | {name} [{tok}] cuts {np.round(thr, 4).tolist()}")
		print("   beyond outer cut %: " + " | ".join(f"{a/1000:.2f}-{b/1000:.2f}s {100 * sat[(t >= a) & (t < b)].mean():5.1f}" for a, b in W))
		print("   yaw-rate var carried: " + " | ".join(f"{eta2(code[(t >= a) & (t < b)], r[(t >= a) & (t < b)]):.3f}" for a, b in W)
		      + f" | all {eta2(code, r):.3f}")
