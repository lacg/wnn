"""CTRL-82 VAL mechanism analysis (never a result). Matched VAL episodes (same unit/ep) for G+T s1, lk90 s1, l2b s1.
Inner rate loop + mixer replayed in numpy and gated against the traced pwm (lib82.inner_replay)."""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl82")
from lib82 import *
P = {int(k): v for k, v in json.load(open("/tmp/ctrl82/params.json")).items()}
H = set(map(tuple, np.load(HARD)))
D = {k: load(v) for k, v in TAGS.items()}
EX = 3.0   # excursion tick: |roll or pitch err vs tilt ref| > EX deg (true attitude), t < 1600 ms


def lag(x, alpha):
	y = np.empty_like(x); y[:, 0] = x[:, 0]
	for t in range(1, x.shape[1]):
		y[:, t] = y[:, t - 1] + alpha * (x[:, t] - y[:, t - 1])
	return y


def torques(d, I, pwm):
	"""Physical body torque (sim body_torque_asym) of a pwm stream after the sim's motor lag, per episode's unit."""
	asym = np.array([P[u]["dist_motor_asym"] for u in d["unit"]])[:, None, :]
	p0 = P[d["unit"][0]]; L, kd, kt = p0["arm_length"], p0["k_drag"], p0["k_thrust"]; alpha = 4 * 0.001 / max(p0["motor_lag_s"], 0.004)
	pl = np.stack([lag(pwm[..., m], alpha) for m in range(4)], -1)
	t = kt * asym * pl ** 2
	return np.stack([L * (t[..., 3] - t[..., 1]), L * (t[..., 2] - t[..., 0]), kd * (t[..., 0] - t[..., 1] + t[..., 2] - t[..., 3])], -1)


def prep(d):
	I = inner_replay(d); assert I["gate"] < 1e-6, I["gate"]; d["I"] = I
	r, p, yw = euler(d["T"][..., 6:10]); d["phi"], d["th"] = r, p
	d["re"] = np.degrees(r - d["tref"][..., 0]); d["pe"] = np.degrees(p - d["tref"][..., 1]); d["yw"] = np.degrees(yw)
	ol, hn, kt, hp = d["inner"][:4]
	cmd = I["cmd"]; r_, p_ = cmd[..., 0] / 2, cmd[..., 1] / 2
	tm0 = np.stack([hn - p_, hn - r_, hn + p_, hn + r_], -1); base0 = np.sqrt(np.clip(tm0, 0, kt) / kt); cd = I["cd"][..., None]
	pwm0 = np.where(cd == 0.0, base0, np.clip(base0 + cd, 0, 1))      # same command, yaw force removed
	d["tau"] = torques(d, I, d["T"][..., 20:24]); d["tau0"] = torques(d, I, pwm0); d["xt"] = d["tau"] - d["tau0"]
	gx, gy, gz = (d["T"][..., 3 + i] for i in range(3))
	d["kin_r"] = np.degrees(gz * np.cos(r) * np.tan(p))                 # yaw-rate term of the Euler roll rate, deg/s
	d["kin_p"] = np.degrees(-gz * np.sin(r))                            # yaw-rate term of the Euler pitch rate, deg/s


for d in D.values():
	prep(d)
t = np.arange(2000)[None, :]
hk = np.array([k in H for k in D["G+T"]["key"]]); assert all(D[k]["key"] == D["G+T"]["key"] for k in D)
f = lambda v: f"{v:8.3f}"
print("VAL ONLY. episodes: hard", hk.sum(), "random", (~hk).sum(), "| inner-loop replay gate max |pwm diff|:", " ".join(f"{k} {d['I']['gate']:.1e}" for k, d in D.items()))
print("\n== A. per-episode mean |err| by axis (deg, whole 2 s), VAL")
print("   set      arm  :   roll    pitch      yaw  | pre-hold roll/pitch/yaw | hold roll/pitch/yaw")
for lab, m in (("hard", hk), ("random", ~hk)):
	for k, d in D.items():
		a = [abs(d[x][m]).mean() for x in ("re", "pe", "yw")]; pre = [abs(d[x][m][:, :1600]).mean() for x in ("re", "pe", "yw")]; ho = [abs(d[x][m][:, 1600:]).mean() for x in ("re", "pe", "yw")]
		print(f"   {lab:7s} {k:5s}:" + "".join(f(v) for v in a) + "  |" + "".join(f(v) for v in pre) + " |" + "".join(f(v) for v in ho))
print(f"\n== B. on EXCURSION ticks (hard eps, |roll|or|pitch err| > {EX} deg, t<1.6 s), each arm on its own excursion ticks; VAL")
rows = []
for k, d in D.items():
	I = d["I"]; ex = hk[:, None] & (t < 1600) & ((abs(d["re"]) > EX) | (abs(d["pe"]) > EX))
	rx = abs(d["re"]) >= abs(d["pe"])         # dominant axis of the excursion tick
	er = np.where(rx, d["re"], d["pe"]); ax = np.where(rx, 0, 1)
	cmdax = np.where(rx, I["cmd"][..., 0], I["cmd"][..., 1]); delax = np.where(rx, I["r_del"], I["p_del"])
	tau = np.where(rx, d["tau"][..., 0], d["tau"][..., 1]); xt = np.where(rx, d["xt"][..., 0], d["xt"][..., 1])
	kin = np.where(rx, d["kin_r"], d["kin_p"]); s = -np.sign(er)    # corrective direction
	gzx = abs(d["T"][..., 5]); y = abs(I["cmd"][..., 2]); hn = d["inner"][1]
	rows.append((k, ex.sum(), 100 * I["sat"][ex].mean(), 100 * (I["sat"] & ~I["sat_noyaw"])[ex].mean(), 100 * I["olsat"][..., :2].any(-1)[ex].mean(),
	             abs(delax[ex]).sum() / abs(cmdax[ex]).sum(), 100 * y[ex].mean() / hn, gzx[ex].mean(),
	             abs(xt[ex]).mean() / abs(tau[ex]).mean() * 100, (s * xt)[ex].mean() / abs(tau[ex]).mean() * 100,
	             abs(kin[ex]).mean(), (-s * kin)[ex].mean()))
hdr = ["ticks", "motor-sat %", "yaw-caused sat %", "rate out-limit %", "deliv/cmd axis", "|yaw force|/hover %", "|gyro_z| rad/s",
       "|yaw->axis torque| % of axis torque", "signed (+ = helps) %", "|yaw-kinematic axis rate| deg/s", "signed (+ = grows err) deg/s"]
print("   " + " | ".join(f"{r[0]:>9s}" for r in rows) + "   (columns = arms)")
for j, h in enumerate(hdr, 1):
	print(f"   {h:38s}" + " ".join(f"{r[j]:10.3f}" if isinstance(r[j], float) else f"{r[j]:10d}" for r in rows))
print("\n== C. same quantities on HOLD ticks (t >= 1.6 s, all 167 eps) and on normal pre-hold ticks (random eps, t<1.6 s, not excursion); VAL")
for lab, sel in (("hold", lambda d: t >= 1600), ("normal pre-hold", lambda d: (~hk)[:, None] & (t < 1600) & (abs(d["re"]) <= EX) & (abs(d["pe"]) <= EX))):
	print(f"   -- {lab}")
	for k, d in D.items():
		m = np.broadcast_to(sel(d), d["re"].shape); I = d["I"]
		print(f"   {k:5s} motor-sat % {100*I['sat'][m].mean():.4f} | |yaw force|/hover % {100*abs(I['cmd'][...,2])[m].mean()/d['inner'][1]:.2f} | |gyro_z| {abs(d['T'][...,5])[m].mean():.3f}"
		      f" | |yaw->roll torque|/|roll torque| % {100*abs(d['xt'][...,0])[m].mean()/abs(d['tau'][...,0])[m].mean():.1f} | |kin roll rate| {abs(d['kin_r'])[m].mean():.3f} deg/s")
np.save("/tmp/ctrl82/hk.npy", hk)
