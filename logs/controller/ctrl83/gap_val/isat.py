import numpy as np, sys
cap = lambda x: np.minimum(x, 8.0)
ICAP = 0.06994220433427308 * 0.5811946409141117   # rate-PID ki * i_limit (N), roll=pitch axes
for pfx in sys.argv[1:]:
	rows = []
	for s in (1, 2, 3):
		z = np.load(pfx.format(s=s)); U, E = z["units"], z["eps"]
		for u in U:
			a = u[12:16]; ab = a.mean()
			m = E[:, 0] == u[0]; mass = E[m, 3]
			th = mass * 9.81 / 4.0                       # per-motor hover thrust (N)
			# thrust-balance differential the roll/pitch rate PID must supply: u = th*(a_hi-a_lo)/(a_hi+a_lo) per pair
			ur = th * abs(a[3] - a[1]) / (a[3] + a[1]); up = th * abs(a[2] - a[0]) / (a[2] + a[0])
			ureq = np.maximum(ur, up)
			for i in np.where(m)[0]:
				pass
			rows.append(np.c_[ureq / ICAP, E[m, 2], 1 - E[m, 7], cap(E[m, 6])])
	R = np.vstack(rows)
	print(pfx, "n", len(R), "rate-I saturation ratio p50/p90/max", np.percentile(R[:, 0], [50, 90]).round(2), R[:, 0].max().round(2))
	for lo, hi in [(0, .5), (.5, .8), (.8, 1.0), (1.0, 1.2), (1.2, 9)]:
		k = (R[:, 0] >= lo) & (R[:, 0] < hi)
		if k.sum():
			print(f"  u_req/I_cap {lo:.1f}-{hi:.1f}: n {k.sum():5d}  WNN fail% {100*(R[k,1]>=8).mean():5.1f}  C fail% {100*R[k,2].mean():5.1f}  WNN err {cap(R[k,1]).mean():.2f} C err {R[k,3].mean():.2f}")
