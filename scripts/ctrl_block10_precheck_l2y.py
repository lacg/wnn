"""docs/ctrl48_block10_levers_spec.md §L2y — VAL-only OFFLINE pre-check of the yaw
analogue of L1, on the block-9 student traces (scripts/ctrl_gap_student_trace.py with the
EXTRA_FEATURES columns: /tmp/ctrl_gap/b10_s*.npz), the teacher's labels along those flights
(label_replay_tests, WNN_GAP_LABEL_DUMP: /tmp/ctrl_gap/both_s*.labels.f32) and the students'
own ladders (/tmp/ctrl_gap/ladders_s*.npz). Nothing is flown or trained.

L2y input (BODY yaw = WORLD yaw at these tilts; rad):
    yaw_err_y = yaw_err − KAPPA · cresid_z,    KAPPA = (K_I · L) / K_θ
K_θ, K_I = the MPCOF>rate teacher's own unconstrained yaw gains on θ and on the rate
integrator (ctbr_axis_mpc.rs, gain row of the outer-step law, cf21bl, n_o 10, lag-free —
the lag moves kappa by < 0.2 %),
L = the yaw rate loop's i_limit (cresid_z = (I − I_ss)/L). Derived, not fitted.

"Carried" = between-bin share of variance (the best possible table on that input alone).
The decisive read is the RESIDUAL test: per episode, the mean of (label − E[label | bin of
the input]) regressed on the episode's TRUE yaw offset. The missed proportional term shows
up as a slope of about −K_θ; an input that carries it leaves a slope near 0.
"""
import numpy as np

# yaw_trim_err.rs YawTrimErr::from_spec on cf21_brushless at n_o = 10 (lag-free plan):
K_THETA, K_I, I_LIMIT = -9.5859, -0.137230, 2.909463863074547
KAPPA = K_I * I_LIMIT / K_THETA           # rad of yaw per unit of cresid_z
DEG = np.radians
YAW_INNER = DEG(np.array([0.25, 0.75, 1.50]))   # L2y dense ladder, fixed inner cuts (spec)
NEAR_ZERO = DEG(np.array([0.10, 0.25, 0.60]))   # L1-shaped alternative, for comparison


def dense(inner, outer):
	m = np.concatenate([inner, np.atleast_1d(outer)])
	return np.concatenate([-m[::-1], m])


def bins(x, cuts):
	return np.searchsorted(np.sort(cuts), x)


def table(y, b):
	"""E[y | bin] per sample (bins with < 5 samples pooled)."""
	_, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
	inv = np.where(cnt[inv] < 5, len(cnt), inv)
	m = np.bincount(inv, y) / np.maximum(np.bincount(inv), 1)
	return m[inv]


def carried(y, *codes):
	key = np.zeros(len(y), dtype=np.int64)
	for c in codes:
		key = key * 16 + c
	return 1.0 - ((y - table(y, key)) ** 2).mean() / y.var()


def additive(y, *codes):
	X = np.concatenate([np.eye(16)[c][:, 1:] for c in codes] + [np.ones((len(y), 1))], axis=1)
	c, *_ = np.linalg.lstsq(X, y, rcond=None)
	return 1.0 - ((y - X @ c) ** 2).mean() / y.var()


def slope_r2(y, x):
	k, b = np.polyfit(x, y, 1)
	r = y - (k * x + b)
	return k, 1.0 - r.var() / y.var()


def one(s):
	z = np.load(f"/tmp/ctrl_gap/b10_s{s}.npz"); R = z["rows"].astype(np.float64); ex = list(z["extra"])
	col = lambda A, n: A[:, 19 + ex.index(n)]
	lad = np.load(f"/tmp/ctrl_gap/ladders_s{s}.npz"); names = list(lad["names"]); thr = lad["thr"]
	cut = lambda n: thr[names.index(n)]
	L = np.fromfile(f"/tmp/ctrl_gap/both_s{s}.labels.f32", dtype="<f4").reshape(-1, 11).astype(np.float64)
	key = lambda a: (a[:, 0].astype(np.int64) * 1000 + a[:, 1].astype(np.int64)) * 10000 + a[:, 2].astype(np.int64)
	idx = {k: i for i, k in enumerate(key(R))}
	H = R[np.array([idx[k] for k in key(L)])]          # hold decision ticks with a label
	y = L[:, 6]                                         # the r (yaw-rate) bank label, rad/s
	ye, yi, cz = col(H, "yaw_err"), col(H, "yaw_err_i"), col(H, "cresid_z")
	x = ye - KAPPA * cz
	xall = col(R, "yaw_err") - KAPPA * col(R, "cresid_z")
	c_q = np.quantile(xall, np.arange(1, 9) / 9)
	# evaluator._dense_thresholds' outer pair (bpf 8, no outer_quantile ⇒ q0 = 1/9): the
	# |x| quantile at mass 1 - 2*q0 of the samples beyond the last fixed cut.
	outer = np.quantile(np.abs(xall)[np.abs(xall) > YAW_INNER[-1]], 1.0 - 2.0 / 9.0)
	c_d, c_z = dense(YAW_INNER, outer), dense(NEAR_ZERO, outer)
	B = {"yaw_err": bins(ye, cut("yaw_err")), "yaw_err_i": bins(yi, cut("yaw_err_i")), "cresid_z": bins(cz, cut("cresid_z")),
	     "L2y dense": bins(x, c_d), "L2y near-zero": bins(x, c_z), "L2y quantile": bins(x, c_q)}
	o = {}
	for n in ("yaw_err", "yaw_err_i", "cresid_z", "L2y dense", "L2y near-zero", "L2y quantile"):
		o[f"hold label carried by {n} (9 bins)"] = carried(y, B[n])
	o["hold label carried by yaw_err + cresid_z, additive (2 inputs)"] = additive(y, B["yaw_err"], B["cresid_z"])
	o["hold label carried by yaw_err x cresid_z (81 cells)"] = carried(y, B["yaw_err"], B["cresid_z"])
	o["hold label carried by L2y, unquantised (linear)"] = np.corrcoef(x, y)[0, 1] ** 2
	# per episode (the traced episodes that have labels)
	ek = H[:, 0] * 1000 + H[:, 1]; _, inv = np.unique(ek, return_inverse=True)
	em = lambda q: np.bincount(inv, q) / np.bincount(inv)
	off = em(H[:, 6])                                   # TRUE yaw, target 0: the frozen offset (rad)
	for n in ("cresid_z", "yaw_err", "L2y dense", "L2y quantile"):
		resid = em(y - table(y, B[n]))
		k, r2 = slope_r2(resid, off)
		o[f"residual after a table on {n}: slope on yaw offset (rad/s per rad)"] = k
		o[f"   ... R2"] = r2 if n == "cresid_z" else o.get("   ... R2", r2)
		o[f"residual after a table on {n}: R2 on yaw offset"] = r2
		o[f"episode-mean label left inside one {n} bin, as yaw deg (/|K_theta|)"] = float(np.degrees(np.abs(resid).mean() / abs(K_THETA)))
	o.pop("   ... R2")
	# the TRUE offset, all episodes (as the L2 table did)
	ek = R[:, 0] * 1000 + R[:, 1]; _, inv = np.unique(ek, return_inverse=True)
	em = lambda q: np.bincount(inv, q) / np.bincount(inv)
	off = em(R[:, 6]); xe = em(col(R, "yaw_err") - KAPPA * col(R, "cresid_z"))
	for n, v, c in (("yaw_err", em(col(R, "yaw_err")), cut("yaw_err")), ("L2y dense", xe, c_d)):
		b = bins(v, c)
		o[f"per-episode TRUE yaw offset carried by {n}"] = carried(off, b)
		o[f"mean |yaw offset| left inside one {n} bin, deg"] = float(np.degrees(np.abs(off - table(off, b)).mean()))
	o["mean |yaw offset| deg"] = float(np.degrees(np.abs(off).mean()))
	o["L2y dense ladder outer cut, deg"] = float(np.degrees(outer))
	return o


if __name__ == "__main__":
	print(f"KAPPA = {KAPPA:.5f} rad per cresid unit ({np.degrees(KAPPA):.3f} deg)")
	res = [one(s) for s in (1, 2, 3)]
	for k in res[0]:
		print(f"{k:84s} | " + " | ".join(f"{r[k]:+.3f}" for r in res))
