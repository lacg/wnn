"""CTRL-69 analysis (prereg: docs/ctrl69_rate_err_precheck.md). Pure statistics over the
recorded streams — no plant/controller logic.
Usage: python analyze.py fit.npz val.npz [--prev-label]
--prev-label (POST-HOC, not in the prereg): the remainder's ridge also gets the thermometer bits
of the PREVIOUS decision's label — removes the "copy the teacher's previous setpoint" channel
that e = sp_prev − gyro carries under teacher-driven rollouts."""
import sys

import numpy as np

AX = ["roll", "pitch", "yaw"]
TAUS_MS = [20, 50, 100, 200, 500]
VARIANTS = ["E0", "Ewin"] + [f"E{t}" for t in TAUS_MS]
X_CUTS, Y_BINS = 8, 16


def e_variants(phys, action_repeat, dt):
	"""Per decision record, per axis: E0, Ewin, E<tau> from the flown loop's per-update e.
	A rate update is visible on the row after it (phase 1); decisions are phase-0 rows."""
	dec = phys[:, 6] > 0.5
	n_dec = int(dec.sum())
	out = {v: np.zeros((n_dec, 3), np.float64) for v in VARIANTS}
	dt_r = 2.0 * dt  # 500 Hz rate loop at 1 kHz (decimation 2)
	alphas = {t: dt_r / (t * 1e-3 + dt_r) for t in TAUS_MS}
	k = -1
	prev_ep = None
	lp = {t: np.zeros(3) for t in TAUS_MS}
	win = []
	for row in phys:
		ep = row[8]
		if ep != prev_ep:
			prev_ep = ep
			lp = {t: np.zeros(3) for t in TAUS_MS}
			win = []
		e = row[0:3].astype(np.float64)
		if row[7] == 1.0:  # an update just happened: a new e sample
			for t in TAUS_MS:
				lp[t] += alphas[t] * (e - lp[t])
			win.append(e)
		if row[6] > 0.5:
			k += 1
			out["E0"][k] = e
			last = win[-(action_repeat // 2):] if win else [np.zeros(3)]
			out["Ewin"][k] = np.mean(last, axis=0)
			for t in TAUS_MS:
				out[f"E{t}"][k] = lp[t]
			win = []
	assert k + 1 == n_dec
	return out


def cuts(x, n):
	q = np.quantile(x, np.arange(1, n + 1) / (n + 1))
	return np.unique(q)


def level(x, c):
	return np.searchsorted(c, x, side="left")  # thermometer level = #cuts below x


def ce_bits(y, x, ny, nx, prior_y=None, joint=None):
	"""log-loss (bits) of y under p(y|x) given as a (nx, ny) prob table."""
	return -np.mean(np.log2(joint[x, y]))


def fit_cond(yf, xf, ny, nx):
	cnt = np.zeros((nx, ny))
	np.add.at(cnt, (xf, yf), 1.0)
	cnt += 0.5
	return cnt / cnt.sum(1, keepdims=True)


def held_out_ig(yf, xf, yv, xv, ny, nx):
	marg = fit_cond(yf, np.zeros_like(xf), ny, 1)
	cond = fit_cond(yf, xf, ny, nx)
	h = -np.mean(np.log2(marg[0, yv]))
	ce = -np.mean(np.log2(cond[xv, yv]))
	return h - ce, h


def plugin_mi(y, x, ny, nx):
	j = np.zeros((nx, ny))
	np.add.at(j, (x, y), 1.0)
	j /= j.sum()
	px, py = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
	nz = j > 0
	return float((j[nz] * np.log2(j[nz] / (px @ py)[nz])).sum())


def spearman(a, b):
	ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
	return float(np.corrcoef(ra, rb)[0, 1])


def thermo_bits(F, cut_list):
	return np.concatenate([(F[:, [i]] > c[None, :]).astype(np.float64) for i, c in enumerate(cut_list)], axis=1)


def prev_label(d):
	"""Previous decision's label p/q/r within the episode (0 at an episode's first record)."""
	lab = d["labels"][:, 1:4].astype(np.float64)
	ep = episode_ids(d)
	prev = np.zeros_like(lab)
	prev[1:] = lab[:-1]
	prev[np.r_[True, ep[1:] != ep[:-1]]] = 0.0
	return prev


def remainder(fit, val, with_prev=False):
	"""label − ridge(thermometer bits of the N0 feature vector), fitted on FIT."""
	if with_prev:
		fit, val = dict(fit), dict(val)
		fit["feats"] = np.hstack([fit["feats"], prev_label(fit)])
		val["feats"] = np.hstack([val["feats"], prev_label(val)])
		fit["names"] = np.concatenate([fit["names"], ["prev_p", "prev_q", "prev_r"]])
	names = list(fit["names"])
	drop = {n for n in names if n.startswith("rate_i_") or n.startswith("cresid_")}
	keep = [i for i, n in enumerate(names) if n not in drop and np.ptp(fit["feats"][:, i]) > 0]
	cl = [cuts(fit["feats"][:, i], X_CUTS) for i in keep]
	Xf = thermo_bits(fit["feats"][:, keep], cl)
	Xv = thermo_bits(val["feats"][:, keep], cl)
	Xf = np.hstack([Xf, np.ones((len(Xf), 1))])
	Xv = np.hstack([Xv, np.ones((len(Xv), 1))])
	A = Xf.T @ Xf + 1e-3 * len(Xf) * np.eye(Xf.shape[1])
	W = np.linalg.solve(A, Xf.T @ fit["labels"][:, 1:4].astype(np.float64))
	rf = fit["labels"][:, 1:4] - Xf @ W
	rv = val["labels"][:, 1:4] - Xv @ W
	r2 = 1 - rv.var(0) / val["labels"][:, 1:4].var(0)
	return rf, rv, r2, [names[i] for i in keep]


def xvars(d):
	names = list(d["names"])
	ev = e_variants(d["phys"], int(d["action_repeat"]), float(d["dt"]))
	# R = the shadow residual block (bit-exact vs the cresid feature where both exist)
	xs = {"R": d["shadow_resid"].astype(np.float64),
	      # the flown loop's I/i_limit at the decision (== the rate_i feature when on: checked 0.0 diff)
	      "I": d["phys"][d["phys"][:, 6] > 0.5, 3:6].astype(np.float64)}
	xs.update(ev)
	return xs


def episode_ids(d):
	ph = d["phys"]
	return ph[ph[:, 6] > 0.5, 8].astype(int)


def main():
	fit, val = (dict(np.load(p, allow_pickle=False)) for p in sys.argv[1:3])
	xf, xv = xvars(fit), xvars(val)
	with_prev = "--prev-label" in sys.argv
	rf, rv, r2, kept = remainder(fit, val, with_prev)
	if with_prev:
		print("POST-HOC: remainder ridge includes the previous label (copycat control)")
	targets = {
		"(1) R": (xf["R"], xv["R"]),
		"(2a) label": (fit["labels"][:, 1:4].astype(np.float64), val["labels"][:, 1:4].astype(np.float64)),
		"(2b) remainder [PRIMARY]": (rf, rv),
	}
	print(f"FIT {len(fit['feats'])} records / VAL {len(val['feats'])} records; N0 ridge inputs {len(kept)} features; "
	      f"ridge VAL R^2 of label p/q/r = {np.round(r2, 3).tolist()}")
	xnames = ["R", "I"] + VARIANTS
	ig = {}
	for tname, (yfit, yval) in targets.items():
		print(f"\n### target {tname} — held-out IG bits (plug-in MI VAL bits | Spearman rho VAL)")
		print("| x | " + " | ".join(AX) + " |")
		print("|---|" + "---|" * 3)
		for xn in xnames:
			cells = []
			for a in range(3):
				yc = cuts(yfit[:, a], Y_BINS - 1)
				yf_, yv_ = level(yfit[:, a], yc), level(yval[:, a], yc)
				xc = cuts(xf[xn][:, a], X_CUTS)
				xf_, xv_ = level(xf[xn][:, a], xc), level(xv[xn][:, a], xc)
				ny, nx = len(yc) + 1, len(xc) + 1
				g, h = held_out_ig(yf_, xf_, yv_, xv_, ny, nx)
				ig[(tname, xn, a)] = g
				cells.append(f"{g:.4f} ({plugin_mi(yv_, xv_, ny, nx):.4f} / {spearman(xv[xn][:, a], yval[:, a]):+.2f})")
			print(f"| {xn} | " + " | ".join(cells) + " |")
		# secondary: joint I x E<tau>
		for t in TAUS_MS:
			cells = []
			for a in range(3):
				yc = cuts(yfit[:, a], Y_BINS - 1)
				yf_, yv_ = level(yfit[:, a], yc), level(yval[:, a], yc)
				ci, ce = cuts(xf["I"][:, a], X_CUTS), cuts(xf[f"E{t}"][:, a], X_CUTS)
				nx = (len(ci) + 1) * (len(ce) + 1)
				jf = level(xf["I"][:, a], ci) * (len(ce) + 1) + level(xf[f"E{t}"][:, a], ce)
				jv = level(xv["I"][:, a], ci) * (len(ce) + 1) + level(xv[f"E{t}"][:, a], ce)
				g, _ = held_out_ig(yf_, jf, yv_, jv, len(yc) + 1, nx)
				ig[(tname, f"IxE{t}", a)] = g
				cells.append(f"{g:.4f}")
			print(f"| I×E{t} (secondary) | " + " | ".join(cells) + " |")
	P = "(2b) remainder [PRIMARY]"
	print("\n### recovery IG(2b; x) / IG(2b; R), and beats-I flag")
	print("| x | " + " | ".join(AX) + " | PASS? |")
	print("|---|" + "---|" * 4)
	passing = []
	for xn in VARIANTS + [f"IxE{t}" for t in TAUS_MS]:
		cells, ok = [], True
		for a in range(3):
			base, g, gi = ig[(P, "R", a)], ig[(P, xn, a)], ig[(P, "I", a)]
			beats = g > gi
			if base < 0.005:
				cells.append(f"N/A ({'>' if beats else '<='}I)")
				ok &= beats
			else:
				rec = g / base
				cells.append(f"{rec:.2f} ({'>' if beats else '<='}I)")
				ok &= (rec >= 0.5) and beats
		secondary = xn.startswith("IxE")
		if ok and not secondary:
			passing.append(xn)
		print(f"| {xn}{' (secondary, not in rule)' if secondary else ''} | " + " | ".join(cells) + f" | {'yes' if ok else 'no'} |")
	print(f"\nI recovery: " + ", ".join(f"{AX[a]} {ig[(P, 'I', a)] / max(ig[(P, 'R', a)], 1e-12):.2f}" for a in range(3)))
	print(f"VERDICT (prereg rule): {'PASS via ' + ','.join(passing) if passing else 'FAIL'}")
	# episode-block bootstrap of the recovery for every variant (VAL resampled, FIT tables fixed)
	ep = episode_ids(val)
	ueps = np.unique(ep)
	rng = np.random.default_rng(69)
	idx_by_ep = {e: np.flatnonzero(ep == e) for e in ueps}
	print("\n### episode-block bootstrap (200x over VAL episodes) of recovery on (2b): median [2.5%, 97.5%]")
	print("| x | " + " | ".join(AX) + " |")
	print("|---|" + "---|" * 3)
	prep = {}
	for xn in ["R", "I"] + VARIANTS:
		for a in range(3):
			yc = cuts(rf[:, a], Y_BINS - 1)
			yf_, yv_ = level(rf[:, a], yc), level(rv[:, a], yc)
			xc = cuts(xf[xn][:, a], X_CUTS)
			ny, nx = len(yc) + 1, len(xc) + 1
			marg = fit_cond(yf_, np.zeros_like(yf_), ny, 1)[0]
			cond = fit_cond(yf_, level(xf[xn][:, a], xc), ny, nx)
			xv_ = level(xv[xn][:, a], xc)
			prep[(xn, a)] = -np.log2(marg[yv_]) + np.log2(cond[xv_, yv_])  # per-record IG contribution
	boots = []
	for _ in range(200):
		sel = np.concatenate([idx_by_ep[e] for e in rng.choice(ueps, len(ueps))])
		boots.append({k: v[sel].mean() for k, v in prep.items()})
	for xn in ["I"] + VARIANTS:
		cells = []
		for a in range(3):
			r = np.array([b[(xn, a)] / b[("R", a)] for b in boots])
			cells.append(f"{np.median(r):.2f} [{np.quantile(r, .025):.2f}, {np.quantile(r, .975):.2f}]")
		print(f"| {xn} | " + " | ".join(cells) + " |")


main()
