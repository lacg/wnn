"""17/11 (Luiz): the yaw floor on the ADOPTED base (yfeM = both_L1 + --zrank-pos-floor auto + --obs-yaw-from-att-est,
ABI 55) vs its same-build control yfeC, with 10b C (ABI 52) for reference. Same units/episodes (CRN asserted in
yaw_floor.py). Mean over s1-s5 of per-seed unit means; deg; per-axis = BODY rotvec of q_ref^-1 q_true; heading terms
in WORLD Euler yaw. VAL = tuning (diagnosis), TEST = report."""
import numpy as np
D = "/Users/lacg/wnn/logs/controller/ctrl48/yaw_floor/"
dg = np.degrees; KTH = 9.586
def yf(arm, s, p): return np.load(f"{D}yfe/yf_{arm}_s{s}_{p}.npz") if arm in "CM" else np.load(f"{D}yf_s{s}_{p}.npz")
def hd(arm, s, p): return np.load(f"{D}yfe/hd_{arm}_s{s}_{p}.npz") if arm in "CM" else np.load(f"{D}hd_s{s}_{p}.npz")
for part in ("val", "test"):
	print(f"\n=== {part.upper()} ({'TUNING' if part == 'val' else 'REPORT'}) — s1-s5 ===")
	print(f"{'row':<18}{'stable%':>8}{'fails':>6} | {'steady:roll':>11}{'pitch':>7}{'yaw':>7}{'total':>7} | {'err:roll':>9}{'pitch':>7}{'yaw':>7}{'total':>7}")
	rows = {}
	for key, arm, field in (("MPCOF oracle", "C", "oracle"), ("MPCOF est", "C", "est"), ("10b C (a52)", "x", "C"), ("yfe C", "C", "C"), ("yfe M (new base)", "M", "C")):
		Z = [yf(arm, s, part) for s in range(1, 6)]
		per = np.array([z[field].mean(0) for z in Z]); m = per.mean(0)
		fails = sum(int(round(((1 - z[field][:, 0]) * z[field][:, 11]).sum())) for z in Z)
		rows[key] = per
		print(f"{key:<18}{100*m[0]:>8.2f}{fails:>6} | {m[6]:>11.3f}{m[7]:>7.3f}{m[8]:>7.3f}{m[2]:>7.3f} | {m[3]:>9.3f}{m[4]:>7.3f}{m[5]:>7.3f}{m[1]:>7.3f}")
	o = rows["MPCOF oracle"][:, 8]; e = rows["MPCOF est"][:, 8]
	for k in ("10b C (a52)", "yfe C", "yfe M (new base)"):
		g = rows[k][:, 8] - e
		print(f"  yaw steady gap {k:<17} over est-fed MPCOF: {g.mean():+.3f}±{g.std(ddof=1):.3f}  per seed {np.round(g, 3)}")
	d = rows["yfe M (new base)"][:, 8] - rows["yfe C"][:, 8]; dt = rows["yfe M (new base)"][:, 2] - rows["yfe C"][:, 2]
	de = rows["yfe M (new base)"][:, 1] - rows["yfe C"][:, 1]
	print(f"  M - C: yaw steady {d.mean():+.3f} {np.round(d,3)} | total steady {dt.mean():+.3f} {np.round(dt,3)} | err {de.mean():+.3f}")
	print(f"  floor (oracle) yaw steady {o.mean():.3f}; est-oracle {np.mean(e-o):+.3f}")
	print(f"  heading split (hold; drift = |true - heading used|, law = |heading used - 0|; energy = share of E[true yaw^2])")
	for k, arm in (("10b C (a52)", "x"), ("yfe C", "C"), ("yfe M (new base)", "M")):
		R = np.array([hd(arm, s, part)["rows"].mean(0) for s in range(1, 6)]); m = R.mean(0)
		print(f"  {k:<17} |v_z| {dg(m[0]):.3f}  drift {dg(m[2]):.3f} (kinematic {dg(m[4]):.3f}, own-Mahony {dg(m[3]):.3f})  law {dg(m[6]):.3f}"
		      f"  energy law {100*m[9]/m[8]:.0f}% drift {100*m[10]/m[8]:.0f}% cross {100*m[11]/m[8]:+.0f}%"
		      f"  | offset-implied bias K_th*|yaw| true {KTH*m[0]:.4f} / heading {KTH*m[6]:.4f} rad/s (offset {dg(m[6]):.3f} deg)")
