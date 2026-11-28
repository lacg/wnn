"""CTRL-83/87 stress readout in the standard orientation (Luiz 28/11): per cell, ONE row per controller —
MPCOF eq-info (row B) on top, then every WNN arm (s1|s2|s3 + mean μ / total Σ), then the rivals (MPCOF est-fed,
PID firmware, LQI; deterministic, one value). Usage: stress_table.py [arm=prefix ...]  (default l2b, l2bnm, l2bnmw)."""
import numpy as np, os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import report as R
D = os.path.join(HERE, "data")
W = [a.split("=") for a in sys.argv[1:]] or [("WNN l2b", "wnn"), ("WNN l2bnm", "l2bnm"), ("WNN l2bnmw", "l2bnmw")]
CELLS = [("nom", "in range"), ("wear30", "wear ±30%"), ("lag30", "lag 0.30 s"), ("mass25", "mass spread ±25%"), ("lag10", "lag 0.10 s"), ("noise2", "noise ×2")]
RIV = [("mpcofA", "MPCOF est-fed"), ("pid", "PID firmware"), ("lqi", "LQI")]
print("WNN rows: seeds s1|s2|s3 then the arm's mean μ / total Σ (f = flying, – = not flown) · rival rows: one deterministic controller, one value")
print(f"{'cell / controller':32s} | {'stable %':27s} | {'err °':22s} | {'steady °':22s} | fails /3500")
for c, lab in CELLS:
	print("-" * 128)
	B = json.load(open(os.path.join(D, f"mpcofB__{c}.json")))["rows"]["B"]
	print(f"{lab + ' · MPCOF eq-info':32s} | {B['stable']:<27.2f} | {B['err']:<22.3f} | {B['steady']:<22.3f} | {B['fails']}")
	for n, p in W:
		v = []
		for s in (1, 2, 3):
			f = os.path.join(D, f"{p}_s{s}__{c}.npz")
			v.append(R.row(np.load(f)["units"], np.load(f)["eps"]) if os.path.exists(f) else None)
		d = [x for x in v if x is not None]
		if not d:
			continue
		col = lambda i, fmt: "|".join(fmt.format(x[i]) if x is not None else "f" for x in v)
		m = lambda i: np.mean([x[i] for x in d])
		print(f"{lab + ' · ' + n:32s} | {col(0, '{:.2f}') + '  μ%.2f' % (np.floor(100 * m(0)) / 100):27s} | {col(1, '{:.2f}') + '  μ%.3f' % m(1):22s} | {col(2, '{:.2f}') + '  μ%.3f' % m(2):22s} | {col(4, '{:.0f}') + '  Σ%d' % sum(x[4] for x in d)}")
	for k, n in RIV:
		z = np.load(os.path.join(D, f"{k}__{c}.npz")); r = R.row(z["units"], z["eps"])
		print(f"{lab + ' · ' + n:32s} | {r[0]:<27.2f} | {r[1]:<22.3f} | {r[2]:<22.3f} | {r[4]}")
