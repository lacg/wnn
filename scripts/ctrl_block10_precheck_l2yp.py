"""docs/ctrl48_block10_levers_spec.md §L2y' — VAL-only OFFLINE pre-check of L2y' (ABI 52,
--obs-yaw-trim-err-p): the proportional yaw input is the L2y value x = yaw_err − κ·cresid_z
(so its label share is round 2's pre-check, unchanged), while yaw_err_i keeps integrating the
RAW heading error. What must hold: the integral keeps its pre-L2y meaning — same scale, same
sign of correlation with the true per-episode yaw offset — unlike L2y's integral.

Data (VAL units only): the block-9 student traces (/tmp/ctrl_gap/b10_s*.npz, raw integral
recorded) with the teacher's labels, and the block-10 s1 VAL replays of the L1 (A) and
L1+L2y headline genomes (/tmp/ctrl_s1diag/*_trace.npz, scripts/ctrl_gap_student_trace.py).
"""
import numpy as np
import ctrl_block10_precheck_l2y as P

D = np.degrees


def ep(R):
	ek = R[:, 0] * 1000 + R[:, 1]
	_, inv = np.unique(ek, return_inverse=True)
	return inv, (lambda q: np.bincount(inv, q) / np.bincount(inv))


def integral_meaning(path, ex_key="extra"):
	z = np.load(path); R = z["rows"].astype(np.float64); ex = list(z[ex_key])
	col = lambda n: R[:, 19 + ex.index(n)]
	_, em = ep(R)
	yi, yaw = em(col("yaw_err_i")), em(R[:, 6])
	return np.abs(yi).mean(), np.corrcoef(yi, yaw)[0, 1]


def label_view(s):
	z = np.load(f"/tmp/ctrl_gap/b10_s{s}.npz"); R = z["rows"].astype(np.float64); ex = list(z["extra"])
	col = lambda A, n: A[:, 19 + ex.index(n)]
	lad = np.load(f"/tmp/ctrl_gap/ladders_s{s}.npz"); names = list(lad["names"]); thr = lad["thr"]
	cut = lambda n: thr[names.index(n)]
	L = np.fromfile(f"/tmp/ctrl_gap/both_s{s}.labels.f32", dtype="<f4").reshape(-1, 11).astype(np.float64)
	key = lambda a: (a[:, 0].astype(np.int64) * 1000 + a[:, 1].astype(np.int64)) * 10000 + a[:, 2].astype(np.int64)
	idx = {k: i for i, k in enumerate(key(R))}
	H = R[np.array([idx[k] for k in key(L)])]
	y = L[:, 6]
	x = col(H, "yaw_err") - P.KAPPA * col(H, "cresid_z")
	xall = col(R, "yaw_err") - P.KAPPA * col(R, "cresid_z")
	outer = np.quantile(np.abs(xall)[np.abs(xall) > P.YAW_INNER[-1]], 1.0 - 2.0 / 9.0)
	bx = P.bins(x, P.dense(P.YAW_INNER, outer))
	bi = P.bins(col(H, "yaw_err_i"), cut("yaw_err_i"))
	be = P.bins(col(H, "yaw_err"), cut("yaw_err"))
	return {"x alone (dense)": P.carried(y, bx), "x + yaw_err_i raw (additive)": P.additive(y, bx, bi),
	        "today: yaw_err + yaw_err_i (additive)": P.additive(y, be, bi)}


if __name__ == "__main__":
	print("YAW integral input yaw_err_i: per-episode |mean| and corr(ep-mean yaw_err_i, ep-mean TRUE yaw offset), VAL hold")
	for s in (1, 2, 3):
		m, c = integral_meaning(f"/tmp/ctrl_gap/b10_s{s}.npz")
		print(f"  block-9 both s{s} (raw integral = what L2y' feeds) : |mean| {m:6.2f}  corr {c:+.2f}")
	for name, lab in (("L1_", "block-10 L1 (A) s1 headline, raw integral"), ("L2y_", "block-10 L1+L2y s1 headline, SHIFTED integral")):
		m, c = integral_meaning(f"/tmp/ctrl_s1diag/{name}trace.npz")
		print(f"  {lab:48s}: |mean| {m:6.2f}  corr {c:+.2f}")
	print("\nYAW hold label carried (block-9 traces, VAL; s1 | s2 | s3)")
	res = [label_view(s) for s in (1, 2, 3)]
	for k in res[0]:
		print(f"  {k:40s} | " + " | ".join(f"{r[k]:.3f}" for r in res))
