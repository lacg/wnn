"""CTRL-83/87 stress readout in the standard orientation (Luiz 28/11): per cell, ONE row per controller —
MPCOF eq-info (row B) on top, then every WNN arm (seeds s1|s2|s3.. + mean μ / total Σ), then the rivals (MPCOF
est-fed, PID firmware, LQI; deterministic, one value).

Default (prereg Amendments 2+3, Luiz 28/11): ONLY the joint-OOD cell `all` (Zhang rule since Amdt 3; the
Amendment-2 cell is `all_v1`, --diagnostic) — THE robustness number — plus the
in-range TEST row `nom` for reference. The one-factor cells are diagnostic-only: --diagnostic prints them too.

  stress_table.py [--diagnostic] [arm=prefix ...]   (default arms: l2b=wnn, l2bnm, l2bnmw)
"""
import numpy as np, os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import report as R
D = os.path.join(HERE, "data")
ARGS = [a for a in sys.argv[1:] if a != "--diagnostic"]
DIAG = "--diagnostic" in sys.argv[1:]
W = [a.split("=") for a in ARGS] or [("WNN l2b", "wnn"), ("WNN l2bnm", "l2bnm"), ("WNN l2bnmw", "l2bnmw"),
                                     ("WNN l2bnmwz", "l2bnmwz")]
LABEL = {"nom": "in range (TEST)", "all": "ALL joint-OOD (Zhang)", "all_v1": "all_v1 (Amdt 2)", "all_a69": "all_a69 (Amdt 3, ABI 69)","allz90": "allz90 (CTRL-90 OOD)", "wear30": "wear ±30%", "lag30": "lag 0.30 s",
         "mass25": "mass spread ±25%", "lag10": "lag 0.10 s", "noise2": "noise ×2"}
CELLS = [("all", LABEL["all"]), ("nom", LABEL["nom"])]
if DIAG:
	CELLS += [(c, LABEL[c] + " (diag)") for c in ("allz90", "all_a69", "all_v1", "wear30", "mass25", "lag30", "lag10", "noise2")]
RIV = [("mpcofA", "MPCOF est-fed"), ("pid", "PID firmware"), ("lqi", "LQI")]
SEEDS = (1, 2, 3, 4, 5)


def wnn_rows(prefix, c):
	"""Per seed the report row (None = not flown/scored); trailing never-flown seeds are dropped."""
	v = []
	for s in SEEDS:
		f = os.path.join(D, f"{prefix}_s{s}__{c}.npz")
		v.append(R.row(np.load(f)["units"], np.load(f)["eps"]) if os.path.exists(f) else None)
	while v and v[-1] is None and len(v) > 3:
		v.pop()
	return v


print("cell `all` (Amdts 3+4, Zhang x0.8/x1.2 on l2b's ranges, ABI 70) = every episode draws wear factor U(0.64, 1.44)"
      " per motor + mass ±38% + plant lag T_e ~ U(0.16, 0.24) s + sensor noise ×1.2 + inertia ±20% + latency 8-12 ms at"
      " once; every model stays at the nominal 0.20 s · TEST, 35 CF2.1 units × 100 eps · older cells = all_a69 (no"
      " inertia/latency), all_v1 (--diagnostic)")
print("WNN rows: seeds s1|s2|s3.. then the arm's mean μ / total Σ over the seeds scored (f = flying/not scored) ·"
      " rival rows: one deterministic controller, one value")
print(f"{'cell / controller':34s} | {'stable %':44s} | {'err °':33s} | {'steady °':33s} | fails /3500")
for c, lab in CELLS:
	print("-" * 166)
	bj = os.path.join(D, f"mpcofB__{c}.json")
	if os.path.exists(bj):
		B = json.load(open(bj))["rows"]["B"]
		print(f"{lab + ' · MPCOF eq-info':34s} | {B['stable']:<44.2f} | {B['err']:<33.3f} | {B['steady']:<33.3f} | {B['fails']}")
	else:
		print(f"{lab + ' · MPCOF eq-info':34s} | not scored")
	for n, p in W:
		v = wnn_rows(p, c)
		d = [x for x in v if x is not None]
		if not d:
			continue
		col = lambda i, fmt: "|".join(fmt.format(x[i]) if x is not None else "f" for x in v)
		m = lambda i: np.mean([x[i] for x in d])
		print(f"{lab + ' · ' + n:34s} | {col(0, '{:.2f}') + '  μ%.2f' % (np.floor(100 * m(0)) / 100):44s} | "
		      f"{col(1, '{:.2f}') + '  μ%.3f' % m(1):33s} | {col(2, '{:.2f}') + '  μ%.3f' % m(2):33s} | "
		      f"{col(4, '{:.0f}') + '  Σ%d' % sum(x[4] for x in d)}")
	for k, n in RIV:
		f = os.path.join(D, f"{k}__{c}.npz")
		if not os.path.exists(f):
			print(f"{lab + ' · ' + n:34s} | not scored")
			continue
		z = np.load(f); r = R.row(z["units"], z["eps"])
		print(f"{lab + ' · ' + n:34s} | {r[0]:<44.2f} | {r[1]:<33.3f} | {r[2]:<33.3f} | {r[4]}")
