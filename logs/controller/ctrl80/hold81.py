"""CTRL-80 follow-up VAL diagnosis (never a result): the hold cost of the yaw lead input.
Reads trace81.py npz files (10 VAL units x 100 episodes x 2000 ticks per seed), l2b vs l2bylead.
Usage: hold81.py label=path [label=path ...]
Hold = t >= 1.6 s. Decision ticks = t % 10 == 0 (action_repeat 10). The lead is RECOMPUTED for every arm from the student's
own recorded yaw_err and gyro_z features (yaw_err - 0.10 * gyro_z) and, for the lead arm, asserted equal to the recorded input.
Candidate pre-checks (section C) re-code the same traces through alternative ladders / gates / filters; they are closed-loop
under the FLOWN policy (the distribution moves once a student acts on the new input)."""
import sys
import numpy as np

TAU = 0.10
CUTS = np.radians([-2.25, -1.35, -0.80, -0.40, 0.40, 0.80, 1.35, 2.25])
HOLD = 1600


def yaw(q):
	w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
	return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def body_rate_z(q):
	"""TRUE body z rate (rad/s) from consecutive q_true (1 ms): 2 * vec_z(q_k^-1 q_k+1) / dt; last tick repeated.
	The trace's gyro columns ARE the student's measured gyro_z (bit-identical to the feature), so this is the noise-free twin."""
	w1, x1, y1, z1 = q[:, :-1, 0], -q[:, :-1, 1], -q[:, :-1, 2], -q[:, :-1, 3]
	w2, x2, y2, z2 = q[:, 1:, 0], q[:, 1:, 1], q[:, 1:, 2], q[:, 1:, 3]
	dw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; dz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
	wz = 2 * dz * np.sign(dw) / 1e-3
	return np.concatenate([wz, wz[:, -1:]], axis=1)


def code(v, cuts):
	return np.searchsorted(np.asarray(cuts), v, side="right")


def toggles_per_s(c):
	"""Code changes between consecutive decision ticks, per second of hold, mean over episodes. c (n_ep, n_dec)."""
	return float((np.diff(c, axis=1) != 0).sum(1).mean() / ((c.shape[1] - 1) * 0.010))


def zc_hz(y):
	"""Oscillation frequency from zero crossings about the episode mean. y (n_ep, n_tick) at 1 kHz."""
	d = y - y.mean(1, keepdims=True)
	return float(((np.diff(np.sign(d), axis=1) != 0).sum(1) / (y.shape[1] / 1000) / 2).mean())


def fft_peak_hz(y):
	d = y - y.mean(1, keepdims=True); d = d * np.hanning(d.shape[1])
	P = (abs(np.fft.rfft(d, axis=1)) ** 2).mean(0); f = np.fft.rfftfreq(d.shape[1], 1e-3)
	k = 1 + int(np.argmax(P[1:])); return float(f[k])


def wcorr(a, b):
	"""Within-episode correlation (both demeaned per episode), pooled. a, b (n_ep, n)."""
	a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
	return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def load(path):
	Z = np.load(path); T = Z["trace"].astype(float); F = Z["feat"].astype(float); names = list(Z["fnames"]); TH = Z["thr"].astype(float)
	n_ep = len(np.unique(T[:, 0] * 1000 + T[:, 1])); L = len(T) // n_ep
	R = lambda v: v.reshape(n_ep, L)
	d = dict(t=R(T[:, 2]), unit=R(T[:, 0]), yw=R(yaw(T[:, 6:10])), spr=R(T[:, 19]), gz_true=R(T[:, 5]),
	         ye=R(F[:, names.index("yaw_err")]), gz=R(F[:, names.index("gyro_z")]), yi=R(F[:, names.index("yaw_err_i")]),
	         ye_thr=TH[names.index("yaw_err")], gz_thr=TH[names.index("gyro_z")], yi_thr=TH[names.index("yaw_err_i")], meta=Z["meta"], n_ep=n_ep, L=L)
	d["wz"] = body_rate_z(T[:, 6:10].reshape(n_ep, L, 4))
	d["lead"] = d["ye"] - TAU * d["gz"]
	d["has_lead"] = "yaw_lead" in names
	if d["has_lead"]:
		rec = R(F[:, names.index("yaw_lead")])
		assert np.abs(rec - d["lead"]).max() < 1e-6, np.abs(rec - d["lead"]).max()
		assert np.allclose(TH[names.index("yaw_lead")], CUTS, atol=1e-6), TH[names.index("yaw_lead")]
	assert (d["t"][:, 0] == 0).all() and L == 2000
	return d


def section_a(lab, d):
	"""Mechanism: hold occupancy, toggles, setpoint, oscillation, correlations, offset."""
	h = slice(HOLD, d["L"]); dc = np.arange(HOLD, d["L"], 10)
	ye, gz, lead, spr, yw = d["ye"], d["gz"], d["lead"], d["spr"], d["yw"]
	lc = code(lead[:, dc], CUTS); yc = code(ye[:, dc], d["ye_thr"]); gc = code(gz[:, dc], d["gz_thr"])
	noise = gz - d["wz"]
	lc_true = code((ye - TAU * d["wz"])[:, dc], CUTS)
	occ = np.bincount(lc.ravel(), minlength=9) * 100.0 / lc.size
	em = spr[:, h].mean(1); es = spr[:, h].std(1)
	off = yw[:, h].mean(1); units = d["unit"][:, 0]
	um = np.array([off[units == u].mean() for u in np.unique(units)])
	spd = spr[:, dc]; lcd = lc - 4.0
	r = dict(
		occ=" ".join(f"{o:.1f}" for o in occ), centre=occ[4],
		tog_lead=toggles_per_s(lc), tog_lead_truegz=toggles_per_s(lc_true), tog_ye=toggles_per_s(yc), tog_gz=toggles_per_s(gc),
		tog_addr_base=toggles_per_s(yc * 9 + gc), tog_addr_lead=toggles_per_s((yc * 9 + gc) * 9 + lc),
		sd_noise=np.degrees(TAU * noise[:, h].std()), sd_truegz=np.degrees(TAU * d["wz"][:, h].std()),
		sd_lead_minus_ye=np.degrees((lead - ye)[:, h].std()),
		sp_sd=es.mean(), sp_trim=abs(em).mean(), dsp=np.median(abs(np.diff(spd, axis=1))[abs(np.diff(spd, axis=1)) > 1e-6]),
		yaw_abs=abs(yw[:, h]).mean(), yaw_sd=yw[:, h].std(1).mean(), off_abs=abs(off).mean(), off_unit=abs(um).mean(),
		hz_zc=zc_hz(yw[:, h]), hz_fft=fft_peak_hz(yw[:, HOLD - 400:]),
		c_sp_ye=wcorr(spd, ye[:, dc]), c_sp_leadc=wcorr(spd, lcd), c_sp_lead=wcorr(spd, lead[:, dc]), c_sp_gz=wcorr(spd, -gz[:, dc]),
		c_sp_noise=wcorr(spd, -noise[:, dc]), c_sp_yi=wcorr(spd, d["yi"][:, dc]),
		trim_vs_off=float(np.corrcoef(em, off)[0, 1]),
		off_per_trim_ep=float(np.dot(em, off) / np.dot(em, em)),
		off_per_trim_unit=float(np.dot(np.array([em[units == u].mean() for u in np.unique(units)]), um)
		                        / np.sum(np.array([em[units == u].mean() for u in np.unique(units)]) ** 2)),
		trim_unit=float(np.mean([abs(em[units == u].mean()) for u in np.unique(units)])))
	return r


def section_b(lab, d):
	"""Transient side (pre-hold) for the same traces, so the candidate trade-off is visible."""
	yw = d["yw"]; return dict(pre=abs(yw[:, :HOLD]).mean(), w06=abs(yw[:, :600]).mean(), whole=abs(yw).mean())


def lowpass(x, fc):
	"""First-order IIR at fc Hz on the 1 kHz tick stream (a fixed causal filter; what a firmware LPF would do)."""
	a = 1.0 - np.exp(-2 * np.pi * fc * 1e-3); y = np.empty_like(x); y[:, 0] = x[:, 0]
	for k in range(1, x.shape[1]):
		y[:, k] = y[:, k - 1] + a * (x[:, k] - y[:, k - 1])
	return y


def candidate_codes(d):
	"""{name: (lead-like value per tick, cuts)} for the candidate pre-checks. Same student features only."""
	ye, gz = d["ye"], d["gz"]; cut = lambda *c: np.radians(sorted([-x for x in c] + list(c)))
	out = {"shipped tau .10, +-.40": (ye - TAU * gz, CUTS)}
	out["(a) inner cut +-0.80 (.8/1.35/2.25/3.2)"] = (ye - TAU * gz, cut(0.80, 1.35, 2.25, 3.20))
	out["(a) inner cut +-0.60 (.6/1.0/1.6/2.4)"] = (ye - TAU * gz, cut(0.60, 1.00, 1.60, 2.40))
	gate = (abs(ye) < np.radians(0.40)) & (abs(gz) < 0.05)
	out["(b) gate |ye|<.40 & |gz|<.05 -> ye"] = (np.where(gate, ye, ye - TAU * gz), CUTS)
	gate2 = abs(ye) < np.radians(0.80)
	out["(b') gate |ye|<.80 -> ye"] = (np.where(gate2, ye, ye - TAU * gz), CUTS)
	for fc in (20.0, 10.0):
		out[f"(c) gz LPF {fc:.0f} Hz"] = (ye - TAU * lowpass(gz, fc), CUTS)
	out["(d) tau .05"] = (ye - 0.05 * gz, CUTS)
	out["(d) tau .05, cuts x0.6"] = (ye - 0.05 * gz, CUTS * 0.6)
	# (f) the lead scaled-shrink: deadband on the RATE term only (rate term zero when |gz| small)
	gzd = np.sign(gz) * np.maximum(abs(gz) - 0.05, 0.0)
	out["(f) rate deadband |gz|-.05"] = (ye - TAU * gzd, CUTS)
	return out


def section_c(d):
	"""Per candidate: hold toggles/s + centre-code occupancy (the hold cost proxy) and the braking-cell information
	(acc of the sign of yaw_err 100 ms ahead in the two flat braking cells, table predictor leave-one-unit-out)."""
	dc_h = np.arange(HOLD, d["L"], 10); dc_t = np.arange(0, 600, 10)
	ye, gz = d["ye"], d["gz"]; grp = d["unit"][:, dc_t].ravel()
	y0, g0, y1 = ye[:, dc_t].ravel(), gz[:, dc_t].ravel(), ye[:, dc_t + 100].ravel()
	c = np.sign(y0); close = c * g0; aye = np.degrees(abs(y0)); crossed = (c * y1 < 0).astype(float)
	cells = [((aye >= 1.23) & (aye < 2.62) & (close >= 0.15) & (close < 0.45)), ((aye >= 0.25) & (aye < 1.23) & (close >= 0.15) & (close < 0.45))]
	yc, gc = code(y0, d["ye_thr"]), code(g0, d["gz_thr"])
	base = [cv_acc(yc[m] * 9 + gc[m], crossed[m], grp[m]) for m in cells]
	rows = []
	for name, (v, cuts) in candidate_codes(d).items():
		lc_h = code(v[:, dc_h], cuts); lc_t = code(v[:, dc_t].ravel(), cuts)
		occ_c = 100.0 * (lc_h == 4).mean()
		acc = [cv_acc((yc[m] * 9 + gc[m]) * 9 + lc_t[m], crossed[m], grp[m]) for m in cells]
		rows.append((name, toggles_per_s(lc_h), occ_c, acc[0], acc[1]))
	# (e) which yaw-family inputs the yaw bank keeps: key sets, transient information + hold command representability
	lc = code(y0 - TAU * g0, CUTS); allm = np.ones(len(y0), bool)
	yh, gh = ye[:, dc_h], gz[:, dc_h]; grp_h = d["unit"][:, dc_h].ravel()
	ych, gch, lch = code(yh.ravel(), d["ye_thr"]), code(gh.ravel(), d["gz_thr"]), code((yh - TAU * gh).ravel(), CUTS)
	ich = code(d["yi"][:, dc_h].ravel(), d["yi_thr"]); sph = d["spr"][:, dc_h].ravel(); hyw = d["yw"][:, dc_h].ravel()
	KS = {"ye x gz (l2b)": (yc * 9 + gc, ych * 9 + gch), "ye x gz x lead (flown arm)": ((yc * 9 + gc) * 9 + lc, (ych * 9 + gch) * 9 + lch),
	      "gz x lead (drop yaw_err)": (gc * 9 + lc, gch * 9 + lch), "ye x lead (drop gyro_z)": (yc * 9 + lc, ych * 9 + lch)}
	keys = []
	for kn, (kt, kh) in KS.items():
		kh2 = kh * 9 + ich
		keys.append((kn, cv_acc(kt[cells[0]], crossed[cells[0]], grp[cells[0]]), cv_acc(kt[cells[1]], crossed[cells[1]], grp[cells[1]]),
		             cv_acc(kt, crossed, grp), np.degrees(cv_rmse(kt, y1, grp)), cv_rmse(kh2, sph, grp_h), cv_rmse(kh2, hyw, grp_h), toggles_per_s(kh.reshape(yh.shape))))
	return base, rows, keys


def cv_rmse(keys, y, grp):
	se = 0.0
	for g in np.unique(grp):
		tr, te = grp != g, grp == g; k = int(keys.max()) + 1
		s_ = np.bincount(keys[tr], weights=y[tr], minlength=k); tot = np.bincount(keys[tr], minlength=k)
		pred = np.where(tot > 0, s_ / np.maximum(tot, 1), y[tr].mean()); se += ((pred[keys[te]] - y[te]) ** 2).sum()
	return float(np.sqrt(se / len(y)))


def cv_acc(keys, y, grp):
	hit = 0
	for g in np.unique(grp):
		tr, te = grp != g, grp == g; k = int(keys.max()) + 1
		pos = np.bincount(keys[tr], weights=y[tr], minlength=k); tot = np.bincount(keys[tr], minlength=k)
		pred = np.where(tot > 0, pos / np.maximum(tot, 1), y[tr].mean()) >= 0.5
		hit += (pred[keys[te]] == (y[te] > 0.5)).sum()
	return 100.0 * hit / len(y)


TR = dict(a.split("=") for a in sys.argv[1:])
A, B, C = {}, {}, {}
for lab, path in TR.items():
	d = load(path); A[lab] = section_a(lab, d); B[lab] = section_b(lab, d); C[lab] = section_c(d)
	print(f"loaded {lab}: {d['n_ep']} VAL episodes, yaw levels {int(d['meta'][0])}, lead input {d['has_lead']}", flush=True)
labs = list(TR)
ROWS = [("lead code occupancy % (codes 0..8)", "occ", "{}"), ("  centre code (|lead|<0.40) %", "centre", "{:.1f}"),
        ("lead-code toggles /s (recomputed, measured gz)", "tog_lead", "{:.1f}"), ("  same with TRUE gz (noise-free)", "tog_lead_truegz", "{:.1f}"),
        ("yaw_err code toggles /s", "tog_ye", "{:.1f}"), ("gyro_z code toggles /s", "tog_gz", "{:.1f}"),
        ("yaw address toggles /s (ye x gz)", "tog_addr_base", "{:.1f}"), ("yaw address toggles /s (ye x gz x lead)", "tog_addr_lead", "{:.1f}"),
        ("sd(tau * gz noise) deg", "sd_noise", "{:.3f}"), ("sd(tau * true gz) deg", "sd_truegz", "{:.3f}"), ("sd(lead - yaw_err) deg", "sd_lead_minus_ye", "{:.3f}"),
        ("held sp_r within-ep SD rad/s", "sp_sd", "{:.4f}"), ("per-ep trim |mean sp_r| rad/s", "sp_trim", "{:.4f}"), ("median nonzero |d sp_r| rad/s", "dsp", "{:.4f}"),
        ("hold |yaw| deg", "yaw_abs", "{:.3f}"), ("hold yaw within-ep SD deg", "yaw_sd", "{:.3f}"),
        ("per-ep |mean hold yaw| (offset) deg", "off_abs", "{:.3f}"), ("per-UNIT |mean hold yaw| deg", "off_unit", "{:.3f}"),
        ("hold yaw osc Hz (zero crossings)", "hz_zc", "{:.2f}"), ("hold yaw FFT peak Hz (1.2-2 s)", "hz_fft", "{:.2f}"),
        ("wcorr(sp_r, yaw_err)", "c_sp_ye", "{:+.3f}"), ("wcorr(sp_r, lead code)", "c_sp_leadc", "{:+.3f}"), ("wcorr(sp_r, lead value)", "c_sp_lead", "{:+.3f}"),
        ("wcorr(sp_r, -gyro_z)", "c_sp_gz", "{:+.3f}"), ("wcorr(sp_r, -gz noise)", "c_sp_noise", "{:+.3f}"), ("wcorr(sp_r, yaw_err_i)", "c_sp_yi", "{:+.3f}"),
        ("corr(per-ep trim, per-ep offset)", "trim_vs_off", "{:+.3f}"),
        ("offset per trim, per-ep fit deg/(rad/s)", "off_per_trim_ep", "{:+.3f}"), ("offset per trim, per-UNIT fit deg/(rad/s)", "off_per_trim_unit", "{:+.3f}"),
        ("per-UNIT |mean sp_r| (trim) rad/s", "trim_unit", "{:.4f}")]
print("\n== A. HOLD (t >= 1.6 s), VAL, decision ticks for codes / setpoint")
print(f"{'':48s}" + "".join(f"{l:>22s}" for l in labs))
for nm, k, f in ROWS:
	print(f"{nm:48s}" + "".join(f"{f.format(A[l][k]):>22s}" for l in labs))
print("\n== B. TRANSIENT, VAL: mean |yaw| deg 0-0.6 s / pre-hold 0-1.6 s / whole")
for l in labs:
	print(f"   {l:20s} {B[l]['w06']:.3f} / {B[l]['pre']:.3f} / {B[l]['whole']:.3f}")
print("\n== C. CANDIDATE PRE-CHECKS (re-coding the flown traces): hold toggles/s | hold centre-code % | braking-cell acc of sign(yaw_err +100 ms), cell1 / cell2 (today's codes in brackets)")
for l in labs:
	base, rows, _ = C[l]
	print(f"-- {l}  (today ye x gz: cell1 {base[0]:.1f} %, cell2 {base[1]:.1f} %)")
	for name, tg, oc, a1, a2 in rows:
		print(f"   {name:42s} {tg:6.1f} /s | centre {oc:5.1f} % | {a1:5.1f} / {a2:5.1f} %")
print("\n== D. KEY SETS for candidate (e) (which yaw-family inputs the yaw bank addresses). Transient 0-0.6 s: acc sign(yaw_err +100 ms) cell1 / cell2 / all ticks %, RMSE yaw_err(+100 ms) deg;"
      " hold: RMSE of the flown sp_r rad/s and of hold yaw deg from key x yaw_err_i code (leave-one-unit-out)")
for l in labs:
	print(f"-- {l}")
	for kn, a1, a2, aa, rm, rs, ry, tg in C[l][2]:
		print(f"   {kn:30s} {a1:5.1f} / {a2:5.1f} / {aa:5.1f} % | {rm:.3f} deg | hold sp_r {rs:.4f} | hold yaw {ry:.3f} | hold address toggles {tg:.1f} /s")
