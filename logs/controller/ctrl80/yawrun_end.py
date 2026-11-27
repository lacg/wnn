"""CTRL-80 yaw-runaway count — a TEST REPORT metric (35 CF2.1 units x 100 episodes).
NEVER use it to select, tune or design anything: it reads the TEST units, which are touched once, for the report.

Usage: yawrun_end.py fly <tag> <out.npz> [units]    per-tick replay of the published winner -> per-episode npz
       yawrun_end.py table <arm_tag> <arm.npz> <control_tag> <control.npz>    the fixed-format table
  units  "test" (default: the run's --test-seeds, asserted to be 99990101..99990135) | "val" | comma list of seeds.

The flight is ctrl82/trace82.py's fly_episode (= ctrl_gap_student_trace.replay's loop) on the EXACT kwargs the
production scorer (ControllerEvaluator.score_genomes -> score_controllers_cpu) was called with, captured per unit.
GATE per episode: the replay's mean err must equal the Rust scorer's own episode err (the captured call re-issued
with episode_errs=True; a diverged episode is the scorer's 8.0 deg cap and must also have ended early in the replay);
the `table` step additionally asserts the per-episode err against the cached TEST re-score (ctrl80/data/<tag>.npz
`eps[:,2]`, from ctrl73/wnn_units.py), so the traced rollout is proven to be the scored rollout.

yaw error = |z| of the rotation vector Log(q_ref^-1 * q_true) after each plant step, q_ref = the scorer's tilt
reference (yaw 0) — the same body-frame yaw component as the re-score's "whole yaw" column (axis_split.rs).
Per episode: max over its ticks. Percentiles are numpy inverted_cdf (the ceil(n*p)-th order statistic, the
codebase's tail definition): p99.9 of 3500 = the 4th worst episode.
yaw growth = max_t |yaw err(t)| - |yaw err(0)|, floored at 0 (deg); yaw err(0) = the first recorded tick (after the
first 1 ms plant step). The per-episode max |yaw err| alone is ~ the start yaw (corr 0.96-0.97): context only.
npz: eps rows [unit, ep, replay err deg, scorer err deg, max |yaw err| deg, diverged, ticks flown,
     |yaw err| after the first step deg], tag."""
import math, os, re, sys, time
import numpy as np

CAP = 8.0
P_DEF = (50, 99, 99.9)


def pct(v, p):
	return float(np.percentile(v, p, method="inverted_cdf"))


# ----------------------------------------------------------------------------------------------- fly (per unit)
_G = {}


def _setup(tag):
	"""Per process: load the run args + winner, patch the scorer entry point to capture its call."""
	os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
	sys.path.insert(0, "/Users/lacg/wnn/scripts")
	import ctrl_gap_student_trace as T
	import wnn.control._accel as A
	import ram_controller as RC
	import wnn.control.phased_ga as PG
	from wnn.control.evaluator import ControllerEvaluator
	txt, a, ec = T.run_args(tag)
	pay = PG._ctl_load(T.D + tag + "_winner.yaml.gz")
	cap = {"calls": []}; real = A.score_controllers_cpu
	def wrap(controllers, *aa, **kk):
		cap["calls"].append((controllers, aa, kk)); return real(controllers, *aa, **kk)
	A.score_controllers_cpu = wrap
	_G.update(T=T, RC=RC, PG=PG, CE=ControllerEvaluator, a=a, ec=ec, spec=pay["spec"], genome=pay["best_genome"], cap=cap, real=real)


def _scorer_ep_errs(ctrls, aa, kk):
	"""The Rust scorer's own per-episode err (deg, diverged -> cap) on the captured call."""
	kk = dict(kk); kk.update(episode_errs=True, tail_diverged_err_deg=CAP)
	row = _G["real"](ctrls, *aa, **kk)[0]
	n = int(row[20]); return list(row[21:21 + n])


def _yaw_abs(Q, R):
	"""|z| of Log(q_ref^-1 * q) per tick (deg); Q, R (n, 4) wxyz. axis_split.rs::rotvec_rel, vectorised."""
	w1, x1, y1, z1 = R[:, 0], -R[:, 1], -R[:, 2], -R[:, 3]
	w2, x2, y2, z2 = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
	dw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
	dx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
	dy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
	dz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
	s = np.where(dw < 0, -1.0, 1.0); dw, dz = dw * s, dz * s
	n = np.sqrt(dx * dx + dy * dy + dz * dz)
	a = np.where(n < 1e-12, 0.0, 2.0 * np.arctan2(n, dw) / np.maximum(n, 1e-300))
	return np.degrees(np.abs(a * dz))


def _fly_episode(c, sim, k, q0, w0, ep):
	"""ctrl82/trace82.py fly_episode, recording only the post-step attitude vs the tilt reference."""
	RC, T = _G["RC"], _G["T"]
	gvt, kt = k["gravity"], k["k_thrust"]
	om, ze, mt = k.get("pos_omega", 1.0), k.get("pos_zeta", 1.0), k.get("pos_max_tilt_rad", 0.5236)
	talt = k["target_altitude"]; tgt = list(k.get("target", [0.0, 0.0, 0.0]))
	def tilt_ref(x, y, vx, vy):
		ad = lambda e, v: om * om * e - 2 * ze * om * v
		return max(-mt, min(mt, -ad(-y, vy) / gvt)), max(-mt, min(mt, ad(-x, vx) / gvt))
	q = list(q0[ep * 4:ep * 4 + 4]); w = list(w0[ep * 3:ep * 3 + 3])
	c.reset(RC.yaw_from_quat(q)); c.seed_att_estimator(q)
	c.set_decode_seed(RC.disturbance_episode_seed(k["dist_seed"], ep))
	sim.reset(q, w)
	sim.set_translation(k["ep_mass"][ep]); sim.set_vertical_state(k["init_z"][ep], k["init_vz"][ep])
	sim.set_horizontal_state(k["init_x"][ep], k["init_y"][ep], 0.0, 0.0)
	# stage1.rs collective_pwm, in f32 like the scorer (trace82's f64 version is 1 ulp off on some episodes,
	# which flips a quantised decision and moves ~0.3% of episode errs by 0.01-0.09 deg).
	f = np.float32
	hover = np.sqrt(f(k["ep_mass"][ep]) * f(gvt) / (f(4.0) * f(kt)))
	coll = float(np.clip(hover * (f(1.0) + f(k["ep_collective_frac"][ep])), f(0.0), f(1.0)))
	c.set_collective_anchor(coll)
	if k["dist_enabled"]:
		sim.set_disturbance(list(k["dist_tau_bias"]), k["dist_gust_sigma"], k["dist_gust_tau_c"], list(k["dist_motor_asym"]),
		                    k["dist_gyro_sigma"], k["dist_gyro_bias_walk"], k["dist_accel_sigma"],
		                    RC.disturbance_episode_seed(k["dist_seed"], ep), k["dist_dropout_prob"],
		                    k["dist_dropout_len_steps"], k["dist_obs_delay_steps"], k["dist_torque_scale_jitter"])
	ftalt = f(talt)   # cpu_score.rs: s1.target_altitude - sim.altitude_rs(), f32
	Q, Rf = [], []; es = 0.0; n = 0
	for t in range(k["_steps"]):
		if sim.is_unstable():
			break
		gy, ac = sim.read_imu()
		c.set_vertical_obs(coll, float(ftalt - f(sim.altitude)), sim.vertical_velocity)
		x, y = sim.position_xy(); vx, vy = sim.velocity_xy()
		c.set_horizontal_obs(-x, -y, vx, vy)
		pwm = c.step(gy, ac, tgt)
		sim.step(list(pwm))
		x, y = sim.position_xy(); vx, vy = sim.velocity_xy()
		rq = T.rp_to_quat(*tilt_ref(x, y, vx, vy))
		es += sim.attitude_error(rq); n += 1
		Q.append(sim.quaternion); Rf.append(rq)
	ya = _yaw_abs(np.asarray(Q, float), np.asarray(Rf, float)) if n else np.zeros(1)
	return math.degrees(es / max(n, 1)), float(ya.max()), float(ya[0]), n


def fly_unit(job):
	"""One unit: score it with the production scorer (captures the call), re-issue for the per-episode errs, replay."""
	tag, unit, thr = job
	if not _G:
		_setup(tag)
	g = _G; a = g["a"]; RC = g["RC"]
	ev = g["CE"](g["spec"], num_eval_episodes=a.test_episodes, seed=unit, episode_config=g["ec"], thresholds=thr,
	             rg_config=g["PG"]._rg_config(a, g["ec"], unit), max_train_workers=1, num_eval_folds=a.num_eval_folds,
	             max_eval_workers_gpu=False)
	g["cap"]["calls"] = []
	ev.score_genomes([g["genome"]])
	assert len(g["cap"]["calls"]) == 1, len(g["cap"]["calls"])
	ctrls, aa, kk = g["cap"]["calls"][0]
	ref = _scorer_ep_errs(ctrls, aa, kk)
	c = ctrls[0]; k = {**{"dt": 0.001, "dist_seed": 0}, **kk}; k["_steps"] = aa[3]
	assert not k.get("alt_integral", 0), "alt_integral on: this replay feeds alt_err raw"
	sim = RC.AttitudeSim(dt=k.get("dt", 0.001), arm_length=k["arm_length"], k_thrust=k["k_thrust"], k_drag=k["k_drag"],
	                     inertia=list(k["inertia"]), gravity=k["gravity"])
	if k.get("motor_lag_s", 0.0) > 0:
		sim.set_motor_lag(k["motor_lag_s"])
	c.set_observer_motor_lag(k.get("motor_lag_s", 0.0))
	if k.get("imu_specific_force"):
		sim.set_imu_specific_force(True)
	if k.get("rotor_drag", 0.0) != 0.0:
		sim.set_rotor_drag(k["rotor_drag"])
	rows, bad = [], []
	for ep in range(aa[2]):
		e, ym, y0, n = _fly_episode(c, sim, k, aa[0], aa[1], ep)
		div = n < aa[3]
		ok = abs(e - ref[ep]) < 0.01 or (div and ref[ep] == CAP)
		if not ok:
			bad.append(f"u{unit} ep{ep}: replay {e:.4f} vs scorer {ref[ep]:.4f} (ticks {n}/{aa[3]})")
		rows.append([unit, ep, CAP if div else e, ref[ep], ym, float(div), n, y0])
	return rows, bad


def resolve_units(a, which):
	if which == "test":
		u = [int(x) for x in a.test_seeds]
		assert u == list(range(99990101, 99990136)), f"run's --test-seeds are not the 35 TEST units: {u[:3]}..{u[-1:]}"
		return u
	if which == "val":
		sys.path.insert(0, "/Users/lacg/wnn/scripts")
		import ctrl_gap_student_trace as T
		return [int(x) for x in T.val_units(a._tag)]
	return [int(x) for x in which.split(",")]


def fly(tag, out, which="test"):
	t0 = time.time()
	_setup(tag); a = _G["a"]; a._tag = tag
	from wnn.seeds import resolve_seed_set
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	units = resolve_units(a, which)
	# Thresholds exactly as ctrl73/wnn_units.py (the TEST re-score): fitted once, on the first TEST unit.
	thr = _G["PG"]._test_thresholds(a, _G["ec"], _G["spec"], [int(x) for x in a.test_seeds][0], seeds.train, use_score=True)
	print(f"[yawrun] {tag[-40:]}: thresholds {time.time() - t0:.0f}s, {len(units)} units", flush=True)
	nw = int(os.environ.get("WORKERS", "1"))
	jobs = [(tag, u, thr) for u in units]
	if nw > 1:
		import multiprocessing as mp
		with mp.get_context("spawn").Pool(nw) as pool:
			res = pool.map(fly_unit, jobs, chunksize=1)
	else:
		res = [fly_unit(j) for j in jobs]
	R = [r for rows, _ in res for r in rows]; bad = [b for _, bs in res for b in bs]
	if bad:
		print(f"YAWRUN FAIL: {len(bad)} episodes: the replay is NOT the scored rollout:\n  " + "\n  ".join(bad[:20]), file=sys.stderr)
		sys.exit(2)
	np.savez(out, eps=np.array(R), tag=tag, units=np.array(units))
	print(f"[yawrun] {tag[-40:]}: {len(R)} episodes gated OK vs the Rust scorer, {time.time() - t0:.0f}s ({nw} workers)", flush=True)


# ----------------------------------------------------------------------------------------------- table
def gate_rescore(tag, Y, rescore_path):
	"""The traced per-episode err must be the cached TEST re-score's eps[:,2] (same unit, same episode)."""
	Z = np.load(rescore_path)
	if str(Z["tag"]) != tag:
		sys.exit(f"YAWRUN FAIL: re-score npz {rescore_path} holds tag {Z['tag']}, not {tag}")
	ref = {(int(u), int(k)): e for u, k, e in Z["eps"]}
	E = Y["eps"]
	if len(E) != len(ref):
		sys.exit(f"YAWRUN FAIL: {tag}: {len(E)} traced episodes vs {len(ref)} re-scored")
	d = np.array([abs(r[2] - ref[(int(r[0]), int(r[1]))]) for r in E])
	if d.max() >= 0.01:
		i = int(d.argmax()); r = E[i]
		print(f"YAWRUN FAIL: {tag}: traced err != re-score eps[:,2] on {(d >= 0.01).sum()} episodes; worst u{int(r[0])} ep{int(r[1])}"
		      f" traced {r[2]:.4f} vs re-score {ref[(int(r[0]), int(r[1]))]:.4f}", file=sys.stderr)
		sys.exit(2)
	return float(d.max())


def worst5(E, v):
	o = np.argsort(-v, kind="stable")[:5]
	return [(int(E[i, 0]) % 1000, int(E[i, 1]), float(v[i])) for i in o]


def stats(Y):
	"""growth = max_t |yaw(t)| - |yaw(0)|, floored at 0 (deg); yaw(0) = column 7, the first recorded tick."""
	E = Y["eps"]; ym = E[:, 4]
	if E.shape[1] < 8:
		sys.exit("YAWRUN FAIL: cache predates the start-yaw column 7; delete data/yawrun_<tag>.npz and re-fly")
	gr = np.maximum(ym - E[:, 7], 0.0)
	return dict(n=len(E), errp=pct(E[:, 2], 99.9),
	            **{f"g{p}": pct(gr, p) for p in P_DEF}, gmax=float(gr.max()), gt1=int((gr > 1).sum()), gt2=int((gr > 2).sum()),
	            **{f"y{p}": pct(ym, p) for p in P_DEF}, ymax=float(ym.max()), y5=int((ym > 5).sum()), y10=int((ym > 10).sum()),
	            wg=worst5(E, gr), wy=worst5(E, ym))


def print_worst(lab, sc, sa, k):
	print(f"{lab:64s}{'control':>20s}  {'arm':>20s}")
	f = lambda w: f"u{w[0]} ep{w[1]:<3d} {w[2]:7.2f}"
	for i in range(5):
		print(f"{'  #%d' % (i + 1):64s}{f(sc[k][i]):>20s}  {f(sa[k][i]):>20s}")


def table(atag, anpz, arescore, ctag, cnpz, crescore):
	A, C = np.load(anpz), np.load(cnpz)
	for t, z in ((atag, A), (ctag, C)):
		if str(z["tag"]) != t:
			sys.exit(f"YAWRUN FAIL: cache holds {z['tag']}, not {t}")
	da, dc = gate_rescore(atag, A, arescore), gate_rescore(ctag, C, crescore)
	sa, sc = stats(A), stats(C)
	short = lambda t: re.sub(r"^.*_yld_", "", t)
	n = sc["n"]; ctx = " (≈ start yaw; context only)"
	print(f"YAWRUN {short(atag)} vs control {short(ctag)}  (TEST REPORT ONLY, {len(A['units'])} units x {n // len(C['units'])} eps;"
	      f" per-episode err == re-score, max |d| {max(da, dc):.4f} deg)")
	print(f"{'row':64s}{'control':>10s}{'arm':>10s}")
	rows = [("whole err p99.9 deg", "errp", 3)] + \
	       [(f"yaw growth p{p:g} deg", f"g{p}", 3) for p in P_DEF] + [("yaw growth max deg", "gmax", 3),
	        (f"eps yaw growth > 1 deg (of {n})", "gt1", 0), (f"eps yaw growth > 2 deg (of {n})", "gt2", 0)]
	for lab, k, nd in rows:
		print(f"{lab:64s}{sc[k]:>10.{nd}f}{sa[k]:>10.{nd}f}")
	print_worst("worst 5 by yaw growth (u ep deg)", sc, sa, "wg")
	rows = [(f"max|yaw err| p{p:g} deg" + ctx, f"y{p}", 3) for p in P_DEF] + [("max|yaw err| max deg" + ctx, "ymax", 3),
	        (f"eps max|yaw err| > 5 deg (of {n})" + ctx, "y5", 0), (f"eps max|yaw err| > 10 deg (of {n})" + ctx, "y10", 0)]
	for lab, k, nd in rows:
		print(f"{lab:64s}{sc[k]:>10.{nd}f}{sa[k]:>10.{nd}f}")
	print_worst("worst 5 by max|yaw err| (u ep deg)" + ctx, sc, sa, "wy")


if __name__ == "__main__":
	if sys.argv[1] == "fly":
		fly(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "test")
	elif sys.argv[1] == "table":
		table(*sys.argv[2:8])
	else:
		sys.exit(__doc__)
