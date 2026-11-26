"""CTRL-80 VAL diagnosis (never a result): where the pre-hold yaw error lives (rise vs overshoot/tail) and what the
yaw bank asks for across the (yaw_err code x closing-rate code) plane. G+T s1 vs l3 s1, trace80.py npz files."""
import sys
import numpy as np
TR = {"G+T s1": "/tmp/ctrl80/trace_gt_s1.npz", "l3 s1": "/tmp/ctrl80/trace_l3_s1.npz"}
if len(sys.argv) > 1:
	TR = dict(a.split("=") for a in sys.argv[1:])
def yaw(q): w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]; return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
YE = [(0, 0.25), (0.25, 1.23), (1.23, 2.62), (2.62, 4.5), (4.5, 99)]
GZ = [(-9, -0.15), (-0.15, -0.05), (-0.05, 0.05), (0.05, 0.15), (0.15, 0.45), (0.45, 9)]   # closing rate (+ = yaw error shrinking), rad/s
for lab, path in TR.items():
	Z = np.load(path); T = Z["trace"].astype(float); F = Z["feat"].astype(float); names = list(Z["fnames"])
	lv, dmy, leak, rng = Z["meta"]
	t = T[:, 2]; key = T[:, 0] * 1000 + T[:, 1]; yw = yaw(T[:, 6:10]); spr = T[:, 19]; gz = T[:, 5]
	n_ep = len(np.unique(key)); Y = yw.reshape(n_ep, -1); S = spr.reshape(n_ep, -1); G = gz.reshape(n_ep, -1)
	assert (t.reshape(n_ep, -1)[:, 0] == 0).all() and Y.shape[1] == 2000
	s0 = np.sign(Y[:, 0])[:, None]; Yn = Y * s0            # normalised: starts positive
	# --- rise vs overshoot/tail split of the pre-hold |yaw| integral
	pre = Y[:, :1600]; tx = np.array([np.argmax(r < 0.25) if (r < 0.25).any() else 1600 for r in (Yn[:, :1600])])   # first time within 0.25 deg of target
	rise = np.array([abs(pre[i, :tx[i]]).sum() for i in range(n_ep)]) / 1000; tail = np.array([abs(pre[i, tx[i]:]).sum() for i in range(n_ep)]) / 1000
	ov = np.array([max(0.0, -Yn[i, tx[i]:1600].min()) if tx[i] < 1600 else 0.0 for i in range(n_ep)])
	y0 = abs(Y[:, 0])
	print(f"== {lab}: pre-hold mean |yaw| {abs(pre).mean():.3f} deg = rise {rise.sum()/n_ep/1.6:.3f} (until first within 0.25 deg; mean t {tx.mean()/1000:.3f} s) + after {tail.sum()/n_ep/1.6:.3f} (overshoot + tail)")
	YB = [(0, 1.23), (1.23, 2.62), (2.62, 3.5), (3.5, 4.5), (4.5, 5.1)]
	g = lambda v, f="{:.3f}": " | ".join(f"[{a},{b}) " + f.format(v[(y0 >= a) & (y0 < b)].mean()) for a, b in YB)
	print("   by |yaw0|: rise part deg  : " + g(rise / 1.6)); print("   by |yaw0|: after part deg : " + g(tail / 1.6))
	print("   by |yaw0|: overshoot deg  : " + g(ov, "{:.2f}")); print("   by |yaw0|: rise time s    : " + g(tx / 1000))
	# ideal-rise reference: the same episodes if yaw simply stopped at the first arrival (no overshoot, no tail)
	win = [(0, 150), (150, 300), (300, 600), (600, 1000), (1000, 1600)]
	print("   mean |yaw| by window 0-.15/.15-.3/.3-.6/.6-1/1-1.6: " + " / ".join(f"{abs(Y[:, a:b]).mean():.2f}" for a, b in win)
	      + " ; share of the pre-hold integral %: " + " / ".join(f"{100*abs(Y[:, a:b]).sum()/abs(pre).sum():.0f}" for a, b in win))
	# --- inner-loop tracking: lag of gyro_z behind the held yaw-rate setpoint (0-0.6 s), by cross-correlation
	a_, b_ = S[:, :600] - S[:, :600].mean(1, keepdims=True), G[:, :700] - G[:, :700].mean(1, keepdims=True)
	lags = range(0, 200, 10); cc = [np.mean([np.corrcoef(a_[i, :500], b_[i, L:L + 500])[0, 1] for i in range(0, n_ep, 3)]) for L in lags]
	print(f"   setpoint -> gyro_z lag (0-0.6 s): best {list(lags)[int(np.argmax(cc))]} ms (r {max(cc):.2f}); r at 0 ms {cc[0]:.2f}")
	# --- the (yaw_err, closing rate) plane on decision ticks, 0-0.6 s
	dec = (t % 10 == 0) & (t < 600); i = np.where(dec)[0]
	ye = np.degrees(F[i, names.index("yaw_err")]); c = np.sign(ye)             # feature = -true yaw: corrective direction has the feature's sign
	close = c * gz[i]; csp = c * spr[i]; prev = np.where(t[i] == 0, 0.0, spr[np.maximum(i - 10, 0)]); cdl = c * (spr[i] - leak * prev); aye = abs(ye)
	print("   decision ticks 0-0.6 s: rows |yaw_err| deg, cols closing rate rad/s " + " ".join(f"[{a},{b})" for a, b in GZ))
	for nm, v, f in (("occupancy %", None, "{:5.1f}"), ("mean corrective sp_r", csp, "{:+.2f}"), ("mean corrective delta", cdl, "{:+.3f}")):
		print(f"     {nm}")
		for x, y in YE:
			cells = [(aye >= x) & (aye < y) & (close >= a) & (close < b) for a, b in GZ]
			print(f"       [{x},{y}): " + " ".join((f.format(100 * m.sum() / len(i)) if v is None else (f.format(v[m].mean()) if m.sum() >= 30 else "   . ")) for m in cells))
	# --- inside the widest braking cells: is the command flat across the cell (input quantisation) ?
	for (x, y), (a, b) in (((1.23, 2.62), (0.15, 0.45)), ((0.25, 1.23), (0.15, 0.45)), ((0.25, 1.23), (0.05, 0.15))):
		m = (aye >= x) & (aye < y) & (close >= a) & (close < b)
		if m.sum() < 200: continue
		ttg = np.radians(aye[m]) / close[m]   # time-to-go at the current rate, s
		q = np.percentile(ttg, [0, 33, 67, 100])
		out = []
		for j in range(3):
			mm = (ttg >= q[j]) & (ttg <= q[j + 1])
			out.append(f"ttg {1000*np.median(ttg[mm]):.0f} ms: sp {csp[m][mm].mean():+.3f} delta {cdl[m][mm].mean():+.3f}")
		print(f"     inside cell |yaw_err| [{x},{y}) x closing [{a},{b}) (n {int(m.sum())}), split by time-to-go terciles: " + " | ".join(out))
