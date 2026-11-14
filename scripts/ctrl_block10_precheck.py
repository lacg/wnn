"""docs/ctrl48_block10_levers_spec.md — VAL-only OFFLINE pre-checks for L1 and L2, on the
block-9 student traces (scripts/ctrl_gap_student_trace.py), the labels the teacher gives
along those flights (label_replay_tests, WNN_GAP_LABEL_DUMP) and the students' own
ladders (scripts/ctrl_gap_ladders.py). Nothing is flown or trained here.

Question: how much of the HOLD label, and of the frozen per-episode offset, can ONE
quantised input carry — today's inputs on today's ladders vs the block-10 inputs?
"Carried" = between-bin share of variance (the R² of the best possible table on that
input alone), so it is an upper bound for a table and a like-for-like comparison.
"""
import sys
import numpy as np

DEG = np.radians
L1_INNER = DEG(np.array([0.08, 0.20, 0.45]))
YAW_INNER = DEG(np.array([0.10, 0.25]))


def dense(inner, outer):
	"""Symmetric pairs, no cut at 0: fixed inner cuts then the given outer cuts."""
	m = np.concatenate([inner, np.atleast_1d(outer)])
	return np.concatenate([-m[::-1], m])


def bins(x, cuts):
	return np.searchsorted(np.sort(cuts), x)


def carried(y, *codes):
	"""Between-cell share of var(y) for the joint code (cells with < 5 samples pooled)."""
	key = np.zeros(len(y), dtype=np.int64)
	for c in codes:
		key = key * 16 + c
	_, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
	inv = np.where(cnt[inv] < 5, len(cnt), inv)
	m = np.bincount(inv, y) / np.maximum(np.bincount(inv), 1)
	return 1.0 - ((y - m[inv]) ** 2).mean() / y.var()


def additive(y, *codes):
	X = np.concatenate([np.eye(16)[c][:, 1:] for c in codes] + [np.ones((len(y), 1))], axis=1)
	c, *_ = np.linalg.lstsq(X, y, rcond=None)
	return 1.0 - ((y - X @ c) ** 2).mean() / y.var()


def one(s):
	z = np.load(f"/tmp/ctrl_gap/both_s{s}.npz"); R = z["rows"].astype(np.float64)
	lad = np.load(f"/tmp/ctrl_gap/ladders_s{s}.npz"); names = list(lad["names"]); thr = lad["thr"]
	cut = lambda n: thr[names.index(n)]
	L = np.fromfile(f"/tmp/ctrl_gap/both_s{s}.labels.f32", dtype="<f4").reshape(-1, 11).astype(np.float64)
	key = lambda a: (a[:, 0].astype(np.int64) * 1000 + a[:, 1].astype(np.int64)) * 10000 + a[:, 2].astype(np.int64)
	idx = {k: i for i, k in enumerate(key(R))}
	sel = np.array([idx[k] for k in key(L)])
	H = R[sel]                                    # hold decision ticks that have a label
	out = {}
	for ax, (ie, iee, iref, iabp, ipos, ivel, sgn, ilab) in {"roll": (4, 7, 10, 17, 13, 15, -1.0, 4), "pitch": (5, 8, 11, 18, 12, 14, -1.0, 5)}.items():
		def inputs(A):
			est = A[:, iee] + A[:, iref]                      # estimated angle
			old = -est                                        # today: target(0) - est
			new = A[:, iref] + A[:, iabp] - est               # L1: (ref + abp) - est
			return old, new, A[:, iabp], -A[:, ipos], A[:, ivel]
		old, new, abp, e, v = inputs(H)
		c_old, c_abp, c_e, c_v = cut(f"{ax}_err"), cut(f"abp_{ax[0]}"), cut("e_y" if ax == "roll" else "e_x"), cut("v_y" if ax == "roll" else "v_x")
		c_new = dense(L1_INNER, np.quantile(np.abs(new), 0.975))
		y = L[:, ilab]
		out[f"{ax}: hold label carried by today's {ax}_err (9 bins)"] = carried(y, bins(old, c_old))
		out[f"{ax}: ... by {ax}_err x abp (81 cells)"] = carried(y, bins(old, c_old), bins(abp, c_abp))
		out[f"{ax}: ... additive in {ax}_err, abp, e, v (4 inputs)"] = additive(y, bins(old, c_old), bins(abp, c_abp), bins(e, c_e), bins(v, c_v))
		out[f"{ax}: ... by the L1 input, dense ladder (9 bins)"] = carried(y, bins(new, c_new))
		out[f"{ax}: ... by the L1 input on today's quantile-style cuts"] = carried(y, bins(new, c_old))
		out[f"{ax}: ... by the L1 input, unquantised (linear)"] = np.corrcoef(new, y)[0, 1] ** 2
		# the frozen offset: per-episode mean TRUE error, all hold samples of all episodes
		old, new, abp, e, v = inputs(R)
		ek = R[:, 0] * 1000 + R[:, 1]; _, inv = np.unique(ek, return_inverse=True)
		em = lambda q: (np.bincount(inv, q) / np.bincount(inv))
		off = em(R[:, ie])
		out[f"{ax}: per-episode TRUE offset carried by today's {ax}_err bin"] = carried(off, bins(em(old), c_old))
		out[f"{ax}: ... by today's 4 inputs, additive"] = additive(off, bins(em(old), c_old), bins(em(abp), c_abp), bins(em(e), c_e), bins(em(v), c_v))
		out[f"{ax}: ... by the L1 input, dense ladder"] = carried(off, bins(em(new), c_new))
		b = bins(em(new), c_new)
		out[f"{ax}: episodes whose offset is in the dense ladder's dead band (|x| < 0.08 deg)"] = float((b == 4).mean())
		out[f"{ax}: mean |offset| left inside one dense bin, deg"] = float(np.degrees(np.abs(off - (np.bincount(b, off, minlength=9) / np.maximum(np.bincount(b, minlength=9), 1))[b]).mean()))
		out[f"{ax}: mean |offset| deg"] = float(np.degrees(np.abs(off).mean()))
	# yaw: today's centred ladder vs dense
	yaw_in = -H[:, 9]; y = L[:, 6]
	c_old = cut("yaw_err"); mags = np.sort(np.abs(c_old))[::-1]
	c_new = dense(YAW_INNER, np.sort(np.unique(mags))[-2:])
	out["yaw: hold label carried by today's yaw_err (centred ladder)"] = carried(y, bins(yaw_in, c_old))
	out["yaw: ... by yaw_err on the dense ladder"] = carried(y, bins(yaw_in, c_new))
	ek = R[:, 0] * 1000 + R[:, 1]; _, inv = np.unique(ek, return_inverse=True)
	em = lambda q: (np.bincount(inv, q) / np.bincount(inv))
	off = em(R[:, 6]); yin = em(-R[:, 9])
	for nm, c in (("today's ladder", c_old), ("the dense ladder", c_new)):
		b = bins(yin, c)
		out[f"yaw: per-episode TRUE offset carried by yaw_err on {nm}"] = carried(off, b)
		out[f"yaw: mean |offset| left inside one bin on {nm}, deg"] = float(np.degrees(np.abs(off - (np.bincount(b, off, minlength=9) / np.maximum(np.bincount(b, minlength=9), 1))[b]).mean()))
	out["yaw: mean |offset| deg"] = float(np.degrees(np.abs(off).mean()))
	return out


res = [one(s) for s in (1, 2, 3)]
for k in res[0]:
	print(f"{k:78s} | " + " | ".join(f"{r[k]:.3f}" for r in res))
