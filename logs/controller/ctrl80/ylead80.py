"""CTRL-80 VAL pre-check (diagnosis only, never a result): does yaw_lead = yaw_err - tau * gyro_z carry
information the student's existing yaw inputs (yaw_err code x gyro_z code) do not? trace80.py npz files.
Usage: ylead80.py [label=path ...] ; env TAU (default 0.10 s), CUTS (deg, default the shipped ladder)."""
import os, sys
import numpy as np
TR = {"G+T s1": "/tmp/ctrl80/trace_gt_s1.npz"}
if len(sys.argv) > 1:
	TR = dict(a.split("=") for a in sys.argv[1:])
TAU = float(os.environ.get("TAU", "0.10"))
CUTS = [float(c) for c in os.environ.get("CUTS", "0.40,0.80,1.35,2.25").split(",")]   # deg, symmetric pairs
H = int(round(TAU * 1000))


def code(v, cuts):
	return np.searchsorted(np.asarray(cuts), v, side="right")


def cv_acc(keys, y, grp):
	"""Majority-vote table predictor of binary y from an integer key, leave-one-unit-out."""
	hit = 0
	for g in np.unique(grp):
		tr, te = grp != g, grp == g
		k = int(keys.max()) + 1
		pos = np.bincount(keys[tr], weights=y[tr], minlength=k); tot = np.bincount(keys[tr], minlength=k)
		pred = np.where(tot > 0, pos / np.maximum(tot, 1), y[tr].mean()) >= 0.5
		hit += (pred[keys[te]] == (y[te] > 0.5)).sum()
	return 100.0 * hit / len(y)


def cv_rmse(keys, y, grp):
	"""Cell-mean regression of y on an integer key, leave-one-unit-out RMSE."""
	se = 0.0
	for g in np.unique(grp):
		tr, te = grp != g, grp == g
		k = int(keys.max()) + 1
		s = np.bincount(keys[tr], weights=y[tr], minlength=k); tot = np.bincount(keys[tr], minlength=k)
		pred = np.where(tot > 0, s / np.maximum(tot, 1), y[tr].mean())
		se += ((pred[keys[te]] - y[te]) ** 2).sum()
	return float(np.sqrt(se / len(y)))


for lab, path in TR.items():
	Z = np.load(path); T = Z["trace"].astype(float); F = Z["feat"].astype(float); names = list(Z["fnames"]); TH = Z["thr"].astype(float)
	t = T[:, 2]; key = T[:, 0] * 1000 + T[:, 1]; n_ep = len(np.unique(key)); L = len(t) // n_ep
	R = lambda v: v.reshape(n_ep, L)
	ye, gz = R(F[:, names.index("yaw_err")]), R(F[:, names.index("gyro_z")]); unit = R(T[:, 0])
	ye_thr, gz_thr = TH[names.index("yaw_err")], TH[names.index("gyro_z")]
	lead_cuts = np.radians(sorted([-c for c in CUTS] + CUTS))
	print(f"== {lab}  tau {TAU:.2f} s, lead cuts deg +-{CUTS}, {n_ep} VAL episodes x {L} ticks")
	# --- 1. sign convention: yaw_err(t+tau) - yaw_err(t) regressed on gyro_z(t), decision ticks 0-0.6 s
	dcols = np.arange(0, 600, 10)
	y0, g0, y1 = ye[:, dcols].ravel(), gz[:, dcols].ravel(), ye[:, dcols + H].ravel()
	slope = float(np.dot(g0, y1 - y0) / np.dot(g0, g0))
	rm = lambda p: float(np.degrees(np.sqrt(np.mean((p - y1) ** 2))))
	print(f"   sign: d(yaw_err over {H} ms) = {slope:+.4f} * gyro_z  (ideal -tau = {-TAU:+.4f}); RMSE deg predicting yaw_err(t+{H} ms), 0-0.6 s: "
	      f"yaw_err {rm(y0):.3f} | yaw_err - tau*gz {rm(y0 - TAU * g0):.3f} | yaw_err + tau*gz {rm(y0 + TAU * g0):.3f}")
	lead = ye - TAU * gz
	# --- 2. distribution of yaw_lead (deg) and ladder occupancy
	for nm, cols in (("0-0.6 s", np.arange(0, 600, 10)), ("0.6-1.6 s", np.arange(600, 1600, 10)), ("hold 1.6-2.0 s", np.arange(1600, L, 10))):
		v = np.degrees(lead[:, cols].ravel()); a = abs(v); e = np.degrees(ye[:, cols].ravel())
		occ = np.bincount(code(np.radians(v), lead_cuts), minlength=9) * 100.0 / len(v)
		print(f"   {nm:14s} |lead| deg p20/40/50/60/80/90/95/99: " + " ".join(f"{np.percentile(a, p):.2f}" for p in (20, 40, 50, 60, 80, 90, 95, 99))
		      + f" | sd(lead - yaw_err) {np.std(v - e):.3f} | lead code occupancy % " + " ".join(f"{o:.1f}" for o in occ)
		      + f" | outer {occ[0] + occ[8]:.1f}")
	# --- 3. what the codes can tell apart, decision ticks 0-0.6 s
	yc, gc, lc = code(y0, ye_thr), code(g0, gz_thr), code(y0 - TAU * g0, lead_cuts)
	grp = unit[:, dcols].ravel(); c = np.sign(y0); close = c * g0; aye = np.degrees(abs(y0))
	fut = c * y1                                     # yaw_err tau later, in the current error's direction: < 0 = crossed (overshoot side)
	ypk = np.array([[(np.sign(ye[i, j]) * ye[i, j:j + 300]).min() for j in dcols] for i in range(n_ep)]).ravel()   # worst crossing within 300 ms
	crossed, over = (fut < 0).astype(float), (ypk < -np.radians(0.5)).astype(float)
	K = {"yaw_err code": yc, "yaw_err x gyro_z codes (the student today)": yc * 9 + gc, "lead code": lc,
	     "yaw_err x gyro_z x lead codes (the arm)": (yc * 9 + gc) * 9 + lc}
	sel = {"all decision ticks 0-0.6 s": np.ones(len(y0), bool),
	       "cell |yaw_err| [1.23,2.62) x closing [0.15,0.45)": (aye >= 1.23) & (aye < 2.62) & (close >= 0.15) & (close < 0.45),
	       "cell |yaw_err| [0.25,1.23) x closing [0.15,0.45)": (aye >= 0.25) & (aye < 1.23) & (close >= 0.15) & (close < 0.45),
	       "cell |yaw_err| [0.25,1.23) x closing [0.05,0.15)": (aye >= 0.25) & (aye < 1.23) & (close >= 0.05) & (close < 0.15)}
	for sn, m in sel.items():
		print(f"   -- {sn} (n {int(m.sum())}): crossed zero {H} ms later {100*crossed[m].mean():.1f} % ; overshoot > 0.5 deg within 300 ms {100*over[m].mean():.1f} %")
		for kn, k in K.items():
			print(f"      {kn:44s}: distinct codes {len(np.unique(k[m])):3d} | acc crossed {cv_acc(k[m], crossed[m], grp[m]):.1f} % | acc overshoot {cv_acc(k[m], over[m], grp[m]):.1f} %"
			      f" | RMSE deg yaw_err(t+{H} ms) {np.degrees(cv_rmse(k[m], fut[m], grp[m])):.3f}")
	# --- 4. the flat cell: time-to-go terciles vs the codes
	m = sel["cell |yaw_err| [1.23,2.62) x closing [0.15,0.45)"]
	ttg = np.radians(aye[m]) / close[m]; q = np.percentile(ttg, [0, 33, 67, 100]); sl = (c * (y0 - TAU * g0))[m]
	slc = code(sl, lead_cuts)
	print("   -- flat cell, by time-to-go tercile: median ttg ms | (yaw_err, gyro_z) joint codes used | signed lead deg p10/50/90 | lead code occupancy % (code of sign-normalised lead, 0..8)")
	for j in range(3):
		mm = (ttg >= q[j]) & (ttg <= q[j + 1]); jc = (yc * 9 + gc)[m][mm]
		print(f"      ttg {1000*np.median(ttg[mm]):4.0f} ms | joint codes {len(np.unique(jc))} (top share {100*np.bincount(jc).max()/mm.sum():.0f} %) | "
		      + "/".join(f"{np.percentile(np.degrees(sl[mm]), p):+.2f}" for p in (10, 50, 90)) + " | "
		      + " ".join(f"{o:.0f}" for o in np.bincount(slc[mm], minlength=9) * 100.0 / mm.sum()))
