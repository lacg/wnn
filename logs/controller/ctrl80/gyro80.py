"""CTRL-80 / CTRL-75 VAL diagnosis (never a result): L2 pre-check (a) re-run on the l2b s1 winner vs G+T s1.
Each trace is read through ITS OWN fitted ladder (the student's raw gyro_x / gyro_y / roll_err / pitch_err inputs)."""
import sys
import numpy as np
TR = {"G+T s1": "/tmp/ctrl80/trace_gt_s1.npz", "l2b s1": "/tmp/ctrl80/trace_l2b_s1.npz"}
if len(sys.argv) > 1:
	TR = dict(a.split("=") for a in sys.argv[1:])
WIN = [(0, 150), (150, 300), (300, 600), (600, 1000), (1000, 1600), (1600, 2000)]
WL = "0-.15 / .15-.3 / .3-.6 / .6-1 / 1-1.6 / hold"
row = lambda v, f="{:5.1f}": " / ".join(f.format(x) for x in v)
for lab, path in TR.items():
	Z = np.load(path); T = Z["trace"].astype(float); F = Z["feat"].astype(float); names = list(Z["fnames"]); TH = Z["thr"]
	t = T[:, 2]; W = [(t >= a) & (t < b) for a, b in WIN]
	print(f"== {lab}   windows s: {WL}")
	for ax in ("gyro_x", "gyro_y"):
		v = F[:, names.index(ax)]; cuts = TH[names.index(ax)]; code = np.searchsorted(cuts, v); a = abs(v)
		print(f"  {ax} cuts rad/s: " + " ".join(f"{c:+.4f}" for c in cuts))
		print(f"    beyond the OUTER cut (codes 0+8) %   : " + row([100 * ((code[w] == 0) | (code[w] == 8)).mean() for w in W]))
		print(f"    in the next-to-outer codes (1+7) %   : " + row([100 * ((code[w] == 1) | (code[w] == 7)).mean() for w in W]))
		print(f"    |{ax}| > 0.026 rad/s %             : " + row([100 * (a[w] > 0.026).mean() for w in W]))
		print(f"    |{ax}| > 0.069 / > 0.15 / > 0.30 %  : " + " ; ".join(row([100 * (a[w] > c).mean() for w in W]) for c in (0.069, 0.15, 0.30)))
		print(f"    |{ax}| p50 / p90 / p99 rad/s        : " + " ; ".join(row([np.percentile(a[w], q) for w in W], "{:.3f}") for q in (50, 90, 99)))
		print(f"    hold code occupancy % (codes 0..8)   : " + " ".join(f"{100*(code[W[5]]==c).mean():.1f}" for c in range(9)))
	for ax in ("roll_err", "pitch_err"):
		v = np.degrees(F[:, names.index(ax)]); cuts = np.degrees(TH[names.index(ax)]); code = np.searchsorted(cuts, v)
		print(f"  {ax} cuts deg: " + " ".join(f"{c:+.2f}" for c in cuts))
		print(f"    mean |{ax}| deg                   : " + row([abs(v[w]).mean() for w in W], "{:.3f}") + f" ; pre-hold {abs(v[t < 1600]).mean():.3f}, hold {abs(v[W[5]]).mean():.3f}")
		print(f"    share of the pre-hold integral %     : " + row([100 * abs(v[w]).sum() / abs(v[t < 1600]).sum() for w in W[:5]], "{:.0f}"))
		print(f"    beyond the OUTER cut (codes 0+8) %   : " + row([100 * ((code[w] == 0) | (code[w] == 8)).mean() for w in W]))
		print(f"    in the next-to-outer codes (1+7) %   : " + row([100 * ((code[w] == 1) | (code[w] == 7)).mean() for w in W]))
