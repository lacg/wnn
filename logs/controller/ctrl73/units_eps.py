"""CTRL-73 (TEST): per-unit gap, wear (c1) regressions, paired-episode gap vs rival C on the 6 rival_eps units,
|yaw0| bins, tail shares, and the fails table. Same data layout as CTRL-70."""
import numpy as np, glob, math
D = np.degrees
C = np.load("/tmp/ctrl73/rival.npz")["C"]; R = np.load("/tmp/ctrl73/rival_eps.npy"); I = np.load("/tmp/ctrl73/ics.npy")
ARMS = [("N0", "n0"), ("G", "n0g3"), ("T", "n0gz"), ("G+T", "n0g3gz")]
def L(p): return [np.load(f) for f in sorted(glob.glob(f"/tmp/ctrl73/{p}_s*.npz"), key=lambda f: int(f.split("_s")[-1][:-4]))]
c1 = C[:, 21:25] @ [1, -1, 1, -1]
key = {(int(u), int(k)): i for i, (u, k) in enumerate(I[:, :2])}
y0 = abs(I[:, 4]); rv = {(int(u), int(k)): e for u, k, e, s in R}
print("== per unit (seed-mean) vs rival C")
for lab, p in ARMS:
	X = np.stack([r["units"] for r in L(p)]); m = X.mean(0)
	g = m[:, 2] - C[:, 2]; gs = m[:, 3] - C[:, 3]; sg = D(m[:, 22])
	k, b = np.polyfit(c1, sg, 1)
	print(f"{lab:4s} err gap/unit min {g.min():.3f} med {np.median(g):.3f} max {g.max():.3f} worse {int((g>0).sum())}/35 | steady gap min {gs.min():+.3f} med {np.median(gs):+.3f} max {gs.max():+.3f} worse {int((gs>0).sum())}/35"
	      f" | hold yaw signed vs c1: slope {k:+.3f} r {np.corrcoef(c1, sg)[0,1]:+.2f}; |seed-mean offset| {np.abs(sg).mean():.3f}; r(|c1|, hold|yaw|) {np.corrcoef(abs(c1), D(m[:,18]))[0,1]:+.2f}"
	      f" | r(|c1|, steady gap) {np.corrcoef(abs(c1), gs)[0,1]:+.2f} r(|c1|, err gap) {np.corrcoef(abs(c1), g)[0,1]:+.2f}")
	if lab == "G+T":
		o = np.argsort(gs)[::-1][:6]
		print("   G+T worst steady-gap units: " + " ".join(f"u{int(C[i,0])%1000}(c1 {c1[i]:+.2f}, {m[i,3]:.3f} vs C {C[i,3]:.3f}, hold y {D(m[i,18]):.3f})" for i in o))
		o = np.argsort(g)[::-1][:6]
		print("   G+T worst err-gap units:    " + " ".join(f"u{int(C[i,0])%1000}(c1 {c1[i]:+.2f}, {m[i,2]:.3f} vs C {C[i,2]:.3f})" for i in o))
print("\n== paired episodes on the 6 rival-scored units (seed-mean WNN vs C, same episodes)")
for lab, p in ARMS:
	W = np.stack([r["eps"] for r in L(p)]); e = W[:, :, 2].mean(0)
	m = {(int(u), int(k)): v for (u, k), v in zip(W[0, :, :2], e)}
	we = np.array([m[(int(u), int(k))] for u, k, _, _ in R]); ce = R[:, 2]; d = we - ce
	yy = np.array([y0[key[(int(u), int(k))]] for u, k, _, _ in R])
	srt = np.sort(d)[::-1]
	allE = W[:, :, 2].ravel(); q = np.percentile(allE, [50, 90, 99])
	k3500, _ = np.polyfit(y0[[key[(int(u), int(k))] for u, k in W[0, :, :2]]], e, 1)
	print(f"{lab:4s} gap mean {d.mean():.3f} median {np.median(d):.3f} worse {int((d>0).sum())}/600 | top1%/10% share {srt[:6].sum()/d.sum():.2f}/{srt[:60].sum()/d.sum():.2f}"
	      f" | pooled p50/p90/p99 {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f} max {allE.max():.2f} | slope err/|yaw0| {k3500:+.3f} | r(|yaw0|, gap) {np.corrcoef(yy, d)[0,1]:+.2f}")
	edges = [0, 1, 2, 3, 4, 5.01]
	print("      |yaw0| bins gap: " + " | ".join(f"[{a},{b}) {d[(yy>=a)&(yy<b)].mean():.2f}" for a, b in zip(edges[:-1], edges[1:])) + f"   (C slope {np.polyfit(yy, ce, 1)[0]:+.3f})")
print("\n== fails (err > 5 deg), per arm")
for lab, p in ARMS:
	for s, r in enumerate(L(p), 1):
		for f in r["fails"]:
			u, k, e = int(f[0]), int(f[1]), f[2]
			print(f"{lab:4s} s{s:<2d} u{u%1000} ep{k:<3d} err {e:.2f} | whole r/p/y {f[5]:.2f}/{f[6]:.2f}/{f[7]:.2f} | mid {f[13]:.2f} | hold |v| {f[12]:.2f} | hold r/p/y {f[9]:.2f}/{f[10]:.2f}/{f[11]:.2f}")
print("\n== err / steady / hold-yaw vs |c1| (per-unit slope, deg per unit |c1|)")
for lab, X in [(l, np.stack([r["units"] for r in L(p)]).mean(0)) for l, p in ARMS] + [("C", None)]:
	e, s, hy = (C[:, 2], C[:, 3], D(C[:, 13])) if X is None else (X[:, 2], X[:, 3], D(X[:, 18]))
	print(f"{lab:4s} err {np.polyfit(abs(c1), e, 1)[0]:+.3f} (r {np.corrcoef(abs(c1), e)[0,1]:+.2f}) | steady {np.polyfit(abs(c1), s, 1)[0]:+.3f} (r {np.corrcoef(abs(c1), s)[0,1]:+.2f}) | hold|yaw| {np.polyfit(abs(c1), hy, 1)[0]:+.3f}")
