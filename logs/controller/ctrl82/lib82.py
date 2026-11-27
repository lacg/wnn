"""CTRL-82 VAL analysis helpers (mechanism only): load a trace82 npz as (episode, tick) arrays, replay the firmware rate
stage + mixer in numpy and GATE it against the traced pwm, so commanded per-axis forces and mixer clamps are known."""
import numpy as np
TAGS = {"G+T": "/tmp/ctrl82/tr_gt.npz", "lk90": "/tmp/ctrl82/tr_lk90.npz", "l2b": "/tmp/ctrl82/tr_l2b.npz"}
HARD = "/tmp/ctrl82/sel_hard.npy"


def euler(q):
	w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
	r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
	p = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
	yw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
	return r, p, yw


def lpf_coeffs(fs, fc):
	fr = fs / fc; ohm = np.tan(np.pi / fr); c45 = np.cos(np.pi / 4); c = 1 + 2 * c45 * ohm + ohm * ohm
	b0 = ohm * ohm / c
	return b0, 2 * b0, b0, 2 * (ohm * ohm - 1) / c, (1 - 2 * c45 * ohm + ohm * ohm) / c


def load(path):
	Z = np.load(path); T = Z["trace"].astype(np.float64); n = 2000
	ne = len(T) // n; assert ne * n == len(T) and (T[:, 2].reshape(ne, n) == np.arange(n)).all()
	S = lambda a: a.reshape(ne, n, *a.shape[1:])
	d = dict(T=S(T), F=S(Z["feat"].astype(np.float64)), tref=S(Z["tref"].astype(np.float64)), fnames=list(Z["fnames"]),
	         inner=Z["inner"], key=[(int(u), int(k)) for u, k in Z["gate"][:, :2]], gate=Z["gate"])
	d["unit"] = np.array([k[0] for k in d["key"]]); d["ep"] = np.array([k[1] for k in d["key"]])
	return d


def inner_replay(d, fs=500.0, fc=30.0, dec=2):
	"""numpy firmware rate stage over all episodes at once; returns dict of per-tick arrays (episode, tick)."""
	T = d["T"]; ol, hn, kt, hp = d["inner"][:4]; G = d["inner"][4:].reshape(3, 4); dt = 1.0 / fs
	ne, n = T.shape[:2]; b0, b1, b2, a1, a2 = lpf_coeffs(fs, fc)
	integ = np.zeros((ne, 3)); prev = np.zeros((ne, 3)); d1 = np.zeros((ne, 3)); d2 = np.zeros((ne, 3)); held = np.zeros((ne, 3))
	raw = np.zeros((ne, n, 3)); cmd = np.zeros((ne, n, 3))
	for t in range(n):
		if t % dec == 0:
			sp = T[:, t, 17:20]; gy = T[:, t, 3:6]; err = sp - gy
			integ = np.clip(integ + err * dt, -G[:, 3], G[:, 3])
			der = np.zeros((ne, 3)) if t == 0 else -(gy - prev) / dt
			x0 = der - d1 * a1 - d2 * a2; out = x0 * b0 + d1 * b1 + d2 * b2; d2, d1 = d1, x0
			prev = gy
			u = G[:, 0] * err + G[:, 1] * integ + G[:, 2] * out
			raw_t = u; held = np.clip(u, -ol, ol)
		raw[:, t] = raw_t; cmd[:, t] = held
	r, p, y = cmd[..., 0] / 2, cmd[..., 1] / 2, cmd[..., 2]
	tm = np.stack([hn - p + y, hn - r - y, hn + p + y, hn + r - y], -1)            # mixer-frame motor thrusts, unclamped
	base = np.sqrt(np.clip(tm, 0, kt) / kt)
	cd = T[..., 24].astype(np.float32).astype(np.float64) - hp + T[..., 16]
	pwm = np.where(cd[..., None] == 0.0, base, np.clip(base + cd[..., None], 0, 1))
	gate = np.abs(pwm - T[..., 20:24]).max()
	te = kt * np.clip(T[..., 20:24] - cd[..., None], 0, None) ** 2                 # delivered, mixer frame
	floor = (tm <= 0) | (base + cd[..., None] <= 0)
	ceil = (tm >= kt) | (base + cd[..., None] >= 1)
	ty0 = np.stack([hn - p, hn - r, hn + p, hn + r], -1)                            # same command with yaw removed
	sat_noyaw = ((ty0 <= 0) | (ty0 >= kt)).any(-1)
	return dict(raw=raw, cmd=cmd, tm=tm, te=te, cd=cd, floor=floor, ceil=ceil, sat=(floor | ceil).any(-1), sat_noyaw=sat_noyaw,
	            olsat=np.abs(raw) > ol, gate=gate, r_del=te[..., 3] - te[..., 1], p_del=te[..., 2] - te[..., 0],
	            y_del=(te[..., 0] + te[..., 2] - te[..., 1] - te[..., 3]) / 4)
