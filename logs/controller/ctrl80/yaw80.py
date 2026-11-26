"""CTRL-80 VAL diagnosis (never a result): the yaw transient of G+T s1 vs l3 s1 on the same 3 VAL units x 100 episodes.
Reads the trace80.py npz files. All commands are read on DECISION ticks (t % 10 == 0, action_repeat 10).
yaw law: sp_k = clamp(0.7 * sp_{k-1} + delta_k, +-2.0 rad/s), delta_k = sign(t) |t|^3 * 1.0 rad/s, t on a 1/32 lattice (64 cells)."""
import sys
import numpy as np
TR = {"G+T s1": "/tmp/ctrl80/trace_gt_s1.npz", "l3 s1": "/tmp/ctrl80/trace_l3_s1.npz"}
if len(sys.argv) > 1:
	TR = dict(a.split("=") for a in sys.argv[1:])
BINS = [(0, 1.23), (1.23, 2.62), (2.62, 4.5), (4.5, 99)]
WIN = [(0, 150), (150, 300), (300, 600), (600, 1600), (1600, 2000)]
def yaw(q): w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]; return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
fmt = lambda xs, f="{:.3f}": " | ".join(f.format(x) for x in xs)
for lab, path in TR.items():
	Z = np.load(path); T = Z["trace"].astype(float); F = Z["feat"].astype(float); names = list(Z["fnames"]); TH = Z["thr"]
	lv, dmy, leak, rng = Z["meta"]; dmax = dmy * rng
	t = T[:, 2]; key = T[:, 0] * 1000 + T[:, 1]; yw = yaw(T[:, 6:10]); spr = T[:, 19]; gz = T[:, 5]
	ye = np.degrees(F[:, names.index("yaw_err")]); cuts = np.degrees(TH[names.index("yaw_err")])
	dec = (t % 10 == 0)
	# implied per-action delta on decision ticks: sp_k - leak * sp_{k-1} (sp_{-1} = 0 at the episode start)
	i = np.where(dec)[0]; sp_k = spr[i]; sp_p = np.where(t[i] == 0, 0.0, spr[np.maximum(i - 10, 0)])
	dl = sp_k - leak * sp_p; clamp = abs(sp_k) >= rng - 1e-6
	tt = np.cbrt(dl / dmax) * (lv / 2)   # lattice index (should be integer where the +-2.0 clamp is not active)
	lat = np.abs(tt - np.round(tt))[~clamp]
	sgn = np.sign(np.corrcoef(ye[i], yw[i])[0, 1])   # sign of the yaw_err feature relative to true yaw
	cs = -np.sign(ye[i]) * np.sign(sgn); cdl = cs * dl * 1.0; csp = cs * sp_k; ti = t[i]; aye = abs(ye[i])
	print(f"== {lab}  (yaw_err cuts deg: {fmt(cuts, '{:+.2f}')}; yaw bank {int(lv)} cells, delta max {dmax:.2f} rad/s per action, leak {leak}, range +-{rng})")
	print(f"  lattice check: |t*32 - int| median {np.median(lat):.4f}, p99 {np.percentile(lat, 99):.4f}; corr(yaw_err feature, true yaw) {np.corrcoef(ye, yw)[0,1]:+.3f}; mean |feature - true| {abs(abs(ye) - abs(yw)).mean():.3f} deg")
	for wlab, (a, b) in (("0-0.6 s", (0, 600)), ("0-1.6 s (pre-hold)", (0, 1600))):
		w = (ti >= a) & (ti < b)
		print(f"  -- decision ticks {wlab}, by |yaw_err input| deg: " + " | ".join(f"[{x},{y})" for x, y in BINS))
		sel = [w & (aye >= x) & (aye < y) for x, y in BINS]
		print("     occupancy %                 : " + fmt([100 * m.sum() / w.sum() for m in sel], "{:.1f}"))
		print("     mean corrective sp_r rad/s  : " + fmt([csp[m].mean() if m.sum() else np.nan for m in sel]))
		print("     mean corrective delta rad/s : " + fmt([cdl[m].mean() if m.sum() else np.nan for m in sel]))
		print("     p90 / max corrective delta  : " + " | ".join(f"{np.percentile(cdl[m],90):.3f} / {cdl[m].max():.3f}" if m.sum() else "nan" for m in sel))
		print("     delta saturated (|t|=1) %   : " + fmt([100 * (abs(dl[m]) >= dmax - 1e-4).mean() if m.sum() else np.nan for m in sel], "{:.2f}"))
		print("     |delta| >= 0.42 (|t|>=.75) %: " + fmt([100 * (abs(dl[m]) >= 0.42).mean() if m.sum() else np.nan for m in sel], "{:.2f}"))
		print("     sp_r at +-2.0 clamp %       : " + fmt([100 * clamp[m].mean() if m.sum() else np.nan for m in sel], "{:.2f}"))
	# first action of the episode (t = 0): the cleanest read of what the memory asks for a given yaw error
	f0 = ti == 0
	print("  -- FIRST action (t=0), by |yaw_err input|: n / mean corrective delta / p90 / mean |t| code (of 32)")
	print("     " + " | ".join(f"[{x},{y}) n{int((f0&(aye>=x)&(aye<y)).sum())} {cdl[f0&(aye>=x)&(aye<y)].mean():.3f} / {np.percentile(cdl[f0&(aye>=x)&(aye<y)],90):.3f} / {abs(tt[f0&(aye>=x)&(aye<y)]).mean():.1f}" for x, y in BINS if (f0 & (aye >= x) & (aye < y)).sum()))
	# input code occupancy (9 codes) per window, all ticks
	code = np.searchsorted(cuts, ye)   # 0..8
	print("  -- yaw_err input code occupancy % (code 0 = below the lowest cut ... 8 = above the highest), all ticks")
	for a, b in WIN:
		w = (t >= a) & (t < b)
		print(f"     {a/1000:.2f}-{b/1000:.2f} s: " + " ".join(f"{100*(code[w]==c).mean():5.1f}" for c in range(9)) + f"   | outer codes (0+8) {100*((code[w]==0)|(code[w]==8)).mean():.1f}")
	# per-episode settle + rate tracking
	ks = np.unique(key); y0, t05, t1s, pk, pks = [], [], [], [], []
	for k in ks:
		m = key == k; y = yw[m]; ta = t[m]
		y0.append(abs(y[0])); j = np.where(abs(y) < 0.5)[0]; t05.append(j[0] / 1000 if len(j) else 2.0)
		j = np.where(abs(y) >= 1.0)[0]; t1s.append((j[-1] + 1) / 1000 if len(j) else 0.0)
		c = -np.sign(y[0]); e = ta < 600; pk.append((c * gz[m][e]).max()); pks.append((c * spr[m][e]).max())
	y0, t05, t1s, pk, pks = map(np.array, (y0, t05, t1s, pk, pks))
	YB = [(0, 1.23), (1.23, 2.62), (2.62, 3.5), (3.5, 4.5), (4.5, 5.1)]
	g = lambda v: " | ".join(f"[{a},{b}) {v[(y0>=a)&(y0<b)].mean():.3f}" for a, b in YB if ((y0 >= a) & (y0 < b)).sum())
	print("  -- per episode by |yaw0| deg (n: " + " ".join(str(int(((y0 >= a) & (y0 < b)).sum())) for a, b in YB) + ")")
	print("     time to first |yaw|<0.5 deg, s : " + g(t05))
	print("     last time |yaw|>=1.0 deg, s    : " + g(t1s))
	print("     peak corrective sp_r, 0-0.6 s  : " + g(pks))
	print("     peak corrective gyro_z, 0-0.6 s: " + g(pk))
	# mean time course for |yaw0| >= 3.5 deg
	big = np.isin(key, ks[y0 >= 3.5])
	first = {k: np.sign(yw[key == k][0]) for k in ks[y0 >= 3.5]}
	sg = np.array([first.get(k, 0.0) for k in key])
	print(f"  -- mean time course, episodes with |yaw0| >= 3.5 deg (n {len(first)}): t ms -> |yaw| deg / corrective sp_r / corrective gyro_z rad/s")
	print("     " + " | ".join(f"{tm}: {(sg[big&(t==tm)]*yw[big&(t==tm)]).mean():.2f} / {(-sg[big&(t==tm)]*spr[big&(t==tm)]).mean():.3f} / {(-sg[big&(t==tm)]*gz[big&(t==tm)]).mean():.3f}" for tm in (0, 20, 50, 100, 150, 200, 300, 400, 600, 1000)))
	pre = t < 1600
	print(f"  -- mean |yaw| deg: pre-hold {abs(yw[pre]).mean():.3f}, hold {abs(yw[~pre]).mean():.3f}, whole {abs(yw).mean():.3f}")
