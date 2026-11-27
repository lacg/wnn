"""CTRL-82 VAL pass 2 (mechanism only, never a result): per-tick trace of SELECTED (val unit, episode) pairs of a run's
HEADLINE genome. Same flight as ctrl_gap_student_trace.replay (copied loop, restricted to the selected episodes; each
episode is reset from scratch, so a subset flight equals the full flight) — gated per episode on the Rust scorer's
episode err from eps82.py.
Usage: trace82.py <tag> <eps82.npz> <select.npy (rows unit, ep)> <out.npz>
  trace  ctrl_gap_student_trace layout (26 cols): 0 unit_idx,1 ep,2 t,3-5 gyro,6-9 q_true,10 x,11 y,12 vx,13 vy,14 alt,
         15 vz,16-19 held CTBR sp [coll_dev,p,q,r],20-23 pwm (inner-loop output, motor order front,right,rear,left),24 anchor,25 talt
  feat   raw student features `fnames` per row;  tref  per row [roll_ref, pitch_ref] of the scorer's tilt reference
  inner  [out_limit_n, hover_n, k_thrust, hover_pwm, roll kp,ki,kd,ilim, pitch .., yaw ..]"""
import math, os, sys
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T
tag, epsf, self_, out = sys.argv[1:5]
CELLS = os.environ.get("CELLS") == "1"; CL = []   # raw output cells on decision ticks (t % 10 == 0)
SUMMARY = len(sys.argv) > 5 and sys.argv[5] == "summary"   # per-episode summaries only (all episodes of the selected units)
import wnn.control._accel as A
import ram_controller as RC
import wnn.control.phased_ga as PG
from wnn.control.evaluator import ControllerEvaluator
from wnn.control.feature_layout import feature_names
from wnn.seeds import resolve_seed_set
WANT = ("gyro_x", "gyro_y", "gyro_z", "roll_err", "pitch_err", "yaw_err", "yaw_err_i")
txt, a, ec = T.run_args(tag)
seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
pay = PG._ctl_load(T.D + tag + "_winner.yaml.gz"); spec, genome = pay["spec"], pay["best_genome"]
EP = np.load(epsf); ref = {(int(u), int(k)): e for u, k, e in EP["eps"]}
SEL = np.load(self_).astype(int); units = sorted(set(SEL[:, 0].tolist()))
assert not any(99990101 <= u <= 99990400 for u in units), "TEST units refused"
names = list(feature_names(spec)); idx = [names.index(n) for n in WANT]
cap = {}; real = A.score_controllers_cpu
def wrap(controllers, *aa, **kk):
	cap.setdefault("calls", []).append((controllers[0], aa, kk)); return real(controllers, *aa, **kk)
A.score_controllers_cpu = wrap
thr = PG._test_thresholds(a, ec, spec, int(EP["units"][0]), seeds.train, use_score=True)


def fly_episode(c, sim, k, q0, w0, ep, ui, coll_hover):
	"""One episode, ctrl_gap_student_trace.replay's loop body; returns (rows, feats, tref, mean err deg)."""
	gvt, kt = k["gravity"], k["k_thrust"]
	om, ze, mt = k.get("pos_omega", 1.0), k.get("pos_zeta", 1.0), k.get("pos_max_tilt_rad", 0.5236)
	talt = k["target_altitude"]
	def tilt_ref(x, y, vx, vy):
		ad = lambda e, v: om * om * e - 2 * ze * om * v
		return max(-mt, min(mt, -ad(-y, vy) / gvt)), max(-mt, min(mt, ad(-x, vx) / gvt))
	q = list(q0[ep * 4:ep * 4 + 4]); w = list(w0[ep * 3:ep * 3 + 3])
	c.reset(RC.yaw_from_quat(q)); c.seed_att_estimator(q)
	c.set_decode_seed(RC.disturbance_episode_seed(k["dist_seed"], ep))
	sim.reset(q, w)
	sim.set_translation(k["ep_mass"][ep]); sim.set_vertical_state(k["init_z"][ep], k["init_vz"][ep])
	sim.set_horizontal_state(k["init_x"][ep], k["init_y"][ep], 0.0, 0.0)
	hover = math.sqrt(k["ep_mass"][ep] * gvt / (4 * kt))
	coll = min(1.0, max(0.0, hover * (1 + k["ep_collective_frac"][ep])))
	c.set_collective_anchor(coll)
	if k["dist_enabled"]:
		sim.set_disturbance(list(k["dist_tau_bias"]), k["dist_gust_sigma"], k["dist_gust_tau_c"], list(k["dist_motor_asym"]),
		                    k["dist_gyro_sigma"], k["dist_gyro_bias_walk"], k["dist_accel_sigma"],
		                    RC.disturbance_episode_seed(k["dist_seed"], ep), k["dist_dropout_prob"],
		                    k["dist_dropout_len_steps"], k["dist_obs_delay_steps"], k["dist_torque_scale_jitter"])
	rows, feats, trf = [], [], []; es = 0.0; n = 0
	for t in range(k["_steps"]):
		if sim.is_unstable():
			break
		gy, ac = sim.read_imu()
		c.set_vertical_obs(coll, talt - sim.altitude, sim.vertical_velocity)
		x, y = sim.position_xy(); vx, vy = sim.velocity_xy()
		c.set_horizontal_obs(-x, -y, vx, vy)
		rr0, rp0 = tilt_ref(x, y, vx, vy)
		pre = [ui, ep, t, *gy, *sim.quaternion, x, y, vx, vy, sim.altitude, sim.vertical_velocity]
		pwm = c.step(gy, ac, list(k.get("target", [0.0, 0.0, 0.0])))
		sp = c.ctbr_held_setpoint
		rows.append(pre + list(sp() if callable(sp) else sp) + list(pwm) + [coll, talt])
		fv = c.get_last_feature_vector(); feats.append([fv[i] for i in idx]); trf.append([rr0, rp0])
		if CELLS and t % 10 == 0: CL.append(list(c.get_last_output_cells()))
		sim.step(list(pwm))
		x, y = sim.position_xy(); vx, vy = sim.velocity_xy()
		rr, rp = tilt_ref(x, y, vx, vy)
		es += sim.attitude_error(T.rp_to_quat(rr, rp)); n += 1
	return rows, feats, trf, math.degrees(es / max(n, 1))


def summarize(X, Fe, tr, unit, ep, e, r):
	"""[unit, ep, replay err, scorer err, roll0, pitch0, yaw0 (deg), |w0|, then per window (pre-hold, hold): mean |roll e|,
	|pitch e|, |yaw| (deg); max |roll e|, max |pitch e|, t of max tilt err (ms), mean |sp_r|, mean |dsp_r| per decision,
	mean |gyro_z|, max |gyro_xy| first 150 ms]"""
	q = X[:, 6:10]; w, x, y, z = q.T
	rl = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) - tr[:, 0])
	pt = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)) - tr[:, 1])
	yw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
	n = len(X); pre = slice(0, min(1600, n)); ho = slice(1600, n)
	m = lambda v, s: float(np.abs(v[s]).mean()) if len(v[s]) else float("nan")
	tilt = np.hypot(rl, pt); spr = X[::10, 19]
	return [unit, ep, e, r, rl[0], pt[0], yw[0], float(np.linalg.norm(X[0, 3:6])),
	        m(rl, pre), m(pt, pre), m(yw, pre), m(rl, ho), m(pt, ho), m(yw, ho),
	        float(np.abs(rl).max()), float(np.abs(pt).max()), float(np.argmax(tilt)), float(np.abs(X[:, 19]).mean()),
	        float(np.abs(np.diff(spr)).mean()), float(np.abs(X[:, 5]).mean()), float(np.hypot(X[:150, 3], X[:150, 4]).max()), n]


R, F, TR, GATE, inner = [], [], [], [], None
for ui, unit in enumerate(units):
	ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
	                         rg_config=PG._rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds, max_eval_workers_gpu=False)
	cap["calls"] = []
	ev.score_genomes([genome])
	c, aa, kk = cap["calls"][0]; k = {**{"dt": 0.001, "dist_seed": 0}, **kk}; k["_steps"] = aa[3]
	sim = RC.AttitudeSim(dt=k.get("dt", 0.001), arm_length=k["arm_length"], k_thrust=k["k_thrust"], k_drag=k["k_drag"],
	                     inertia=list(k["inertia"]), gravity=k["gravity"])
	if k.get("motor_lag_s", 0.0) > 0:
		sim.set_motor_lag(k["motor_lag_s"])
	c.set_observer_motor_lag(k.get("motor_lag_s", 0.0))
	if k.get("imu_specific_force"):
		sim.set_imu_specific_force(True)
	if k.get("rotor_drag", 0.0) != 0.0:
		sim.set_rotor_drag(k["rotor_drag"])
	assert not k.get("alt_integral", 0)
	if inner is None:
		s = c.ctbr_spec; hp = s.hover_pwm() if callable(s.hover_pwm) else s.hover_pwm
		inner = [s.out_limit_n, s.hover_n, s.k_thrust, hp, *s.rate_gains]
	for ep in (range(aa[2]) if SUMMARY else SEL[SEL[:, 0] == unit, 1]):
		rows, feats, trf, e = fly_episode(c, sim, k, aa[0], aa[1], int(ep), unit, None)
		r = ref[(unit, int(ep))]
		GATE.append([unit, ep, e, r])
		ok = abs(e - r) < 0.01 or r >= 8.0
		if not ok or not SUMMARY: print(f"VAL u{unit} ep{ep}: replay err {e:.3f} | scorer ep err {r:.3f} {'OK' if ok else 'MISMATCH'}", flush=True)
		if SUMMARY:
			R.append(summarize(np.array(rows), np.array(feats), np.array(trf), unit, int(ep), e, r))
		else:
			R += rows; F += feats; TR += trf
if SUMMARY:
	np.savez(out, summ=np.array(R), gate=np.array(GATE)); print("saved", out, len(R), flush=True); sys.exit(0)
np.savez(out, cells=np.array(CL, dtype=np.uint8), trace=np.array(R, dtype=np.float32), feat=np.array(F, dtype=np.float32), tref=np.array(TR, dtype=np.float32),
         fnames=np.array(WANT), inner=np.array(inner), gate=np.array(GATE), allfeat=np.array(names))
print("saved", out, len(R), inner, flush=True)
