"""CTRL-82 VAL: does the OUTER loop (WNN p/q banks) ask for less recovery at large tilt error? Hard eps, decision ticks
(t % 10 == 0, t < 1.6 s); corrective sign = toward reducing the dominant-axis error. Also the p/q-bank TRUE-cell evidence
(BINARY antagonist decode: E half pushes +, I half pushes -) from the cells re-trace (same hard episodes)."""
exec(open("/Users/lacg/wnn/logs/controller/ctrl82/an82.py").read().split("t = np.arange(2000)")[0])
hk = np.load("/tmp/ctrl82/hk.npy")
CL = {"G+T": "/tmp/ctrl82/cl_gt.npz", "lk90": "/tmp/ctrl82/cl_lk90.npz", "l2b": "/tmp/ctrl82/cl_l2b.npz"}
BIN = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 99)]
print("VAL hard episodes (47), decision ticks t<1.6 s. Per |axis err| bin deg: " + " | ".join(f"[{a},{b})" for a, b in BIN))
for k, d in D.items():
	m = hk; T_ = d["T"][m][:, :1600:10]; re = d["re"][m][:, :1600:10]; pe = d["pe"][m][:, :1600:10]
	out = {}
	for ax, e, spc, gyc in (("roll", re, 17, 3), ("pitch", pe, 18, 4)):
		s = -np.sign(e); sp = T_[..., spc]; prev = np.concatenate([np.zeros_like(sp[:, :1]), sp[:, :-1]], 1)
		dl = sp - 0.7 * prev; g = T_[..., gyc]; a = abs(e)
		sgn = np.sign(np.corrcoef((s * sp).ravel(), a.ravel())[0, 1])
		out[ax] = (a, s * sp, s * dl, s * g)
	print(f"== {k}")
	for ax, (a, csp, cdl, cg) in out.items():
		cells = [(a >= x) & (a < y) for x, y in BIN]
		print(f"   {ax:5s} n          : " + " ".join(f"{int(c.sum()):7d}" for c in cells))
		print(f"   {ax:5s} corr sp    : " + " ".join(f"{csp[c].mean():+7.3f}" if c.sum() else "     . " for c in cells))
		print(f"   {ax:5s} corr delta : " + " ".join(f"{cdl[c].mean():+7.3f}" if c.sum() else "     . " for c in cells))
		print(f"   {ax:5s} corr gyro  : " + " ".join(f"{cg[c].mean():+7.3f}" if c.sum() else "     . " for c in cells))
	a, csp, cdl, cg = out["roll"]; st = (a > 3) & (cg < 0.1)
	print(f"   roll stall (|err|>3 deg and corrective gyro_x < 0.1 rad/s): {int(st.sum())} decision ticks = {100*st.sum()/max((a>3).sum(),1):.1f}% of |err|>3 ticks")
	# TRUE-cell evidence in the p bank on the same ticks (cells trace = same 47 hard eps, same order as sel_hard)
	Z = np.load(CL[k]); C = Z["cells"].reshape(47, 200, -1); nb = C.shape[2] // 4; P_ = C[..., nb:2 * nb] == 1
	E, Ih = P_[..., :nb // 2].sum(-1), P_[..., nb // 2:].sum(-1)
	keys_cl = [(int(u), int(e)) for u, e in Z["gate"][:, :2]]; hard_keys = [kk for kk, h in zip(d["key"], hk) if h]
	idx = [keys_cl.index(kk) for kk in hard_keys]; E, Ih = E[idx][:, :160], Ih[idx][:, :160]
	tot = (E + Ih) / nb * 100
	print(f"   p-bank TRUE cells % of bank (E+I) by |roll err| bin: " + " ".join(f"{tot[c].mean():6.1f}" if c.sum() else "    . " for c in [(a >= x) & (a < y) for x, y in BIN]))
