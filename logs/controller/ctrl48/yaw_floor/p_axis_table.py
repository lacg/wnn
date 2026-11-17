"""17/11 (Luiz): block 10b P (--alt-integral pole3) vs 10b C by axis, through the production CPU scorer
(ABI 56 axis_out; L3 flown natively). Paired by seed; deg; BODY rotvec axes. VAL = tuning, TEST = report."""
import numpy as np
D = "/Users/lacg/wnn/logs/controller/ctrl48/yaw_floor/p_axis/"
dg = np.degrees
def seedmeans(arm, part, seeds):
	out = []
	for s in seeds:
		r = np.load(f"{D}{arm}_s{s}_{part}.npz")["rows"]
		m = r.mean(0)
		out.append([m[1], m[2], m[3], dg(m[4]), dg(m[5]), dg(m[6]), dg(m[8]), dg(m[9]), dg(m[10]), dg(m[15])])
	return np.array(out)
H = "stable  err   steady | err roll pitch  yaw | steady roll pitch  yaw | head drift"
for part in ("val", "test"):
	for seeds, lab in (((1, 2, 3, 4), "s1-s4 (paired, asked)"), ((1, 2, 3, 4, 5), "s1-s5")):
		print(f"\n=== {part.upper()} ({'TUNING' if part == 'val' else 'REPORT'}) {lab} ===\n{'':<10}{H}")
		P, C = seedmeans("P", part, seeds), seedmeans("C", part, seeds)
		for nm, X in (("10b C", C), ("10b P", P)):
			m = X.mean(0)
			print(f"{nm:<10}{100*m[0]:6.2f} {m[1]:.3f} {m[2]:.3f}  |   {m[3]:.3f} {m[4]:.3f} {m[5]:.3f} |   {m[6]:.3f} {m[7]:.3f} {m[8]:.3f} |  {m[9]:.3f}")
		d = P - C; m = d.mean(0)
		print(f"{'P - C':<10}{100*m[0]:+6.2f} {m[1]:+.3f} {m[2]:+.3f}  |  {m[3]:+.3f} {m[4]:+.3f} {m[5]:+.3f} |  {m[6]:+.3f} {m[7]:+.3f} {m[8]:+.3f} | {m[9]:+.3f}")
		for k, nm in ((1, "err"), (3, "err roll"), (4, "err pitch"), (5, "err yaw"), (2, "steady"), (8, "steady yaw")):
			print(f"   {nm:<11} per seed {np.round(d[:, k], 3)}  (P worse in {int((d[:, k] > 0).sum())}/{len(seeds)})")
