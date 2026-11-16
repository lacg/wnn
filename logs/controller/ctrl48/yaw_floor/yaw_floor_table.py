"""Aggregate yf_s{1..5}_{val,test}.npz: per-axis table (mean over seeds of the per-seed unit mean)
and the three-way yaw decomposition. Partition stated on every block (VAL = tuning, TEST = report)."""
import numpy as np, sys
D = "/Users/lacg/wnn/logs/controller/ctrl48/yaw_floor/"
ROW = ("stable", "err", "steady", "err_roll", "err_pitch", "err_yaw", "hold_roll", "hold_pitch", "hold_yaw",
       "signed_yaw", "fed_yaw_drift", "episodes")
I = {k: i for i, k in enumerate(ROW)}
for part in ("val", "test"):
	Z = [np.load(f"{D}yf_s{s}_{part}.npz") for s in range(1, 6)]
	nu = sum(len(z["unit"]) for z in Z)
	print(f"\n=== {part.upper()} ({'TUNING' if part=='val' else 'REPORT'}) units, s1-s5, {nu} units x 100 eps; deg, body frame ===")
	print(f"{'row':<14}{'stable%':>8}{'fails':>6} | {'steady:roll':>11}{'pitch':>7}{'yaw':>7}{'total':>7} | {'err:roll':>9}{'pitch':>7}{'yaw':>7}{'total':>7} | {'sgn yaw':>8}{'fed drift':>10}")
	agg = {}
	for r in ("oracle", "est", "C"):
		per = np.array([z[r].mean(0) for z in Z])  # per-seed means
		fails = sum(int(round(((1 - z[r][:, 0]) * z[r][:, 11]).sum())) for z in Z)
		m = per.mean(0); sd = per.std(0, ddof=1); agg[r] = (m, sd, per)
		print(f"{'MPCOF '+r if r!='C' else 'C (WNN)':<14}{100*m[0]:>8.2f}{fails:>6} | {m[6]:>11.3f}{m[7]:>7.3f}{m[8]:>7.3f}{m[2]:>7.3f} | {m[3]:>9.3f}{m[4]:>7.3f}{m[5]:>7.3f}{m[1]:>7.3f} | {m[9]:>8.3f}{m[10]:>10.3f}")
	for k, nm in ((8, "steady yaw"), (5, "err yaw"), (2, "steady total"), (1, "err total")):
		o, e, c = (agg[r][2][:, k] for r in ("oracle", "est", "C"))
		print(f"  {nm:<13} floor(oracle) {o.mean():.3f}±{o.std(ddof=1):.3f} | est−oracle {np.mean(e-o):+.3f}±{np.std(e-o,ddof=1):.3f} | C−est {np.mean(c-e):+.3f}±{np.std(c-e,ddof=1):.3f}  per-seed C−est {np.round(c-e,3)}")
