"""Aggregate s1miss/*.npz (dmy_axis.py). deg / m; per-axis = BODY rotvec; drift = |heading used - true yaw| (hold)."""
import glob, os
import numpy as np
D = "/Users/lacg/wnn/logs/controller/ctrl48/dmax_yaw/s1miss/"
dg = np.degrees


def row(name):
	z = np.load(D + name + ".npz")["rows"]; m = z.mean(0)
	return dict(stable=100 * m[1], err=m[2], steady=m[3], alt=m[4], pos=m[5], dzh=m[6],
	            er=dg(m[7]), ep=dg(m[8]), ey=dg(m[9]), sr=dg(m[11]), sp=dg(m[12]), sy=dg(m[13]), drift=dg(m[18]))


H = f"{'':<34}{'stable':>7}{'err':>7}{'steady':>7} | {'st:roll':>7}{'pitch':>6}{'yaw':>6} | {'err:roll':>8}{'pitch':>6}{'yaw':>6} | {'drift':>6} | {'alt':>6}{'pos':>6}{'dz_h':>6}"
def line(lab, r):
	return (f"{lab:<34}{r['stable']:7.2f}{r['err']:7.3f}{r['steady']:7.3f} | {r['sr']:7.3f}{r['sp']:6.3f}{r['sy']:6.3f} | {r['er']:8.3f}{r['ep']:6.3f}{r['ey']:6.3f} |"
	        f" {r['drift']:6.3f} | {r['alt']:6.3f}{r['pos']:6.3f}{r['dzh']:6.3f}")

for p in ("val", "test"):
	print(f"\n=== Q1/Q3/Q4 headlines, {p.upper()} ({'TUNING' if p == 'val' else 'REPORT'}), 20/35 units x 100 eps ===\n{H}")
	rs = {}
	for lab, n in (("M s1 NEURONS#0 (flown, yaw 0.6)", f"dmyM_full_s1_HEADLINE_asflown_{p}"),
	               ("M s1 NEURONS#0 decoded yaw 0.5", f"dmyM_full_s1_HEADLINE_0.5_{p}"),
	               ("Y s1 GRID#2 (flown, trained 0.5)", f"dmyY_full_s1_WINNER_asflown_{p}"),
	               ("Y s1 GRID#2 decoded yaw 0.6", f"dmyY_full_s1_WINNER_none_{p}"),
	               ("D s1 NEURONS#0 (flown, all 0.5)", f"dmD_full_s1_HEADLINE_asflown_{p}")):
		rs[lab] = row(n); print(line(lab, rs[lab]))
	m = rs["M s1 NEURONS#0 (flown, yaw 0.6)"]
	for lab in list(rs)[1:]:
		r = rs[lab]
		print(f"  {lab[:28]:<28} - M: steady {r['steady']-m['steady']:+.3f} (r {r['sr']-m['sr']:+.3f} p {r['sp']-m['sp']:+.3f} y {r['sy']-m['sy']:+.3f})"
		      f" | err {r['err']-m['err']:+.3f} (r {r['er']-m['er']:+.3f} p {r['ep']-m['ep']:+.3f} y {r['ey']-m['ey']:+.3f}) | dz_hold {r['dzh']-m['dzh']:+.4f}")

print(f"\n=== Q2 stage candidates, VAL (TUNING): each genome as flown and at the other yaw gain (decode only) ===")
print(f"{'genome':<12}{'Y flown (0.5)':>22}{'Y at yaw 0.6':>22}{'lever':>8} || {'M flown (0.6)':>22}{'M at yaw 0.5':>22}{'lever':>8}")
print(f"{'':<12}{'steady  yaw   err':>22}{'steady  yaw   err':>22}{'dSt':>8} || {'steady  yaw   err':>22}{'steady  yaw   err':>22}{'dSt':>8}")
for st in ("GRID", "NEURONS", "MEMORY"):
	for i in range(3):
		g = f"{st}-{i}"
		c = lambda r: f"{r['steady']:.3f} {r['sy']:.3f} {r['err']:.3f}"
		if g == "GRID-2":  # the stage checkpoint lost GRID#2's cells (cells=None) in BOTH runs
			yf, yn = row("dmyY_full_s1_WINNER_asflown_val"), row("dmyY_full_s1_WINNER_none_val")
			print(f"{'GRID#2':<12}{c(yf):>22}{c(yn):>22}{yf['steady']-yn['steady']:+8.3f} || {'ckpt has no cells (n/a)':>52}   [Y = HEADLINE, from the winner file]")
			continue
		yf, yn = row(f"dmyY_full_s1_{g}_asflown_val"), row(f"dmyY_full_s1_{g}_none_val")
		mf, m5 = row(f"dmyM_full_s1_{g}_asflown_val"), row(f"dmyM_full_s1_{g}_0.5_val")
		print(f"{st+'#'+str(i):<12}{c(yf):>22}{c(yn):>22}{yf['steady']-yn['steady']:+8.3f} || {c(mf):>22}{c(m5):>22}{m5['steady']-mf['steady']:+8.3f}")
