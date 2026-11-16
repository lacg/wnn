"""Aggregate hd_s{1..5}_{val,test}[_swap].npz (heading_diag.py) + yf_* teacher rows: drift vs yaw-law split
and the swap replay (existing genomes, NOT retrained). Mean over 5 seeds of per-seed unit means; deg."""
import numpy as np
D = "/Users/lacg/wnn/logs/controller/ctrl48/yaw_floor/"
dg = np.degrees
for part in ("val", "test"):
	tag = "TUNING" if part == "val" else "REPORT"
	print(f"\n=== {part.upper()} ({tag}) — s1-s5, hold window, deg (WORLD Euler yaw for heading terms; |v_z| = BODY rotvec) ===")
	yf = [np.load(f"{D}yf_s{s}_{part}.npz") for s in range(1, 6)]
	for r, nm in (("oracle", "MPCOF oracle"), ("est", "MPCOF est")):
		m = np.mean([z[r].mean(0) for z in yf], 0)
		print(f"{nm:<22} steady {m[2]:.3f}  |v_z| {m[8]:.3f}  drift {m[10]:.3f}")
	out = {}
	for sw in ("", "_swap"):
		Z = [np.load(f"{D}hd_s{s}_{part}{sw}.npz") for s in range(1, 6)]
		R = np.array([z["rows"].mean(0) for z in Z]); M = np.array([z["meta"][:, 1:].mean(0) for z in Z])
		fails = sum(int(round(((1 - z["meta"][:, 1]) * 100).sum())) for z in Z)
		m = R.mean(0); out[sw] = (R, M)
		e_true, e_est, d_sq, cross = (m[8], m[9], m[10], m[11])
		lab = "C (dead-reckon, as flown)" if not sw else "C SWAP (Mahony yaw; existing genomes, NOT retrained)"
		print(f"{lab}\n  scorer: stable {100*M[:,0].mean():.2f}% fails {fails}  err {M[:,1].mean():.3f}  steady {M[:,2].mean():.3f}"
		      f"   | roll {dg(m[14]):.3f} pitch {dg(m[15]):.3f} |v_z| {dg(m[0]):.3f} (signed {dg(m[1]):+.3f})")
		print(f"  heading: drift |y_true-y_used| {dg(m[2]):.3f}   law |y_used - 0| {dg(m[6]):.3f}   "
		      f"[dead-reckon drift {dg(m[2] if not sw else np.nan):.3f} = kinematic {dg(m[4]):.3f} (perfect gyro) + sensor {dg(m[5]):.3f}; own Mahony drift {dg(m[3]):.3f}]")
		print(f"  energy of true yaw E[y²] {dg(dg(e_true)):.4f} deg² = law E[h²] {100*e_est/e_true:.0f}% + drift E[d²] {100*d_sq/e_true:.0f}% + cross {100*cross/e_true:+.0f}%")
	(R0, M0), (R1, M1) = out[""], out["_swap"]
	ds = M1[:, 2] - M0[:, 2]; dv = dg(R1[:, 0] - R0[:, 0]); de = M1[:, 1] - M0[:, 1]
	print(f"  SWAP − AS-FLOWN per seed: steady {np.round(ds,3)} mean {ds.mean():+.3f}±{ds.std(ddof=1):.3f} | |v_z| {np.round(dv,3)} mean {dv.mean():+.3f} | err mean {de.mean():+.3f}")
