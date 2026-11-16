"""Lever 1 (Luiz 16/11): the HEADING ESTIMATE. Same units / episodes / C headline genome as
yaw_floor.py (scorer-gated replay of axis_trace.replay's loop, here with the heading channels).

Per step, at FEED time (the IMU sample the controller reads; sim state before sim.step), yaw in
WORLD Euler (z-y-x) radians:
  y_true    Euler yaw of the true quaternion
  y_dr      the student's heading = -yaw_err feature (target yaw 0, trim off: asserted) —
            WnnController.yaw_heading, the gyro_z*dt dead-reckon
  y_mah     Euler yaw of the student's OWN Mahony quaternion (c.att_est_quat(); obs_att_estimator)
  y_kin     integral of the TRUE body rate omega_z * dt from the true initial yaw: what the
            dead-reckon would read with a PERFECT gyro (isolates kinematics from sensor error)
  v_z       body-frame yaw component of Log(q_ref^-1 q_true) after the step (the scorer's split)
Hold window = t >= ceil(0.8 steps). Per-episode hold means, then the mean over episodes.
Optional --swap: fly with WnnController.set_yaw_from_att_est(True) (ABI 54) — EXISTING GENOMES,
NOT RETRAINED."""
import argparse, math, os, re, sys, time
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl48/l2yp_investigation")
import numpy as np
import ctrl_gap_student_trace as T
import axis_trace as AX

COLS = ("hold_vz_abs", "hold_vz_signed", "d_dr_abs", "d_mah_abs", "d_kin_abs", "d_noise_abs", "est_dr_abs",
        "est_mah_abs", "e_true_sq", "e_est_sq", "d_sq", "cross", "err", "steady", "hold_vx", "hold_vy")


def wrap(a):
	return (a + math.pi) % (2 * math.pi) - math.pi


def replay(RC, c, q0, w0, n_eps, steps, k, yi, swap):
	"""axis_trace.replay's loop verbatim (same order of calls), plus the heading channels."""
	sim = RC.AttitudeSim(dt=k.get("dt", 0.001), arm_length=k["arm_length"], k_thrust=k["k_thrust"], k_drag=k["k_drag"],
	                     inertia=list(k["inertia"]), gravity=k["gravity"])
	if k.get("motor_lag_s", 0.0) > 0: sim.set_motor_lag(k["motor_lag_s"])
	c.set_observer_motor_lag(k.get("motor_lag_s", 0.0))
	if k.get("imu_specific_force"): sim.set_imu_specific_force(True)
	if k.get("rotor_drag", 0.0) != 0.0: sim.set_rotor_drag(k["rotor_drag"])
	if swap: c.set_yaw_from_att_est(True)
	dt = k.get("dt", 0.001)
	gvt, kt = k["gravity"], k["k_thrust"]
	om, ze, mt = k.get("pos_omega", 1.0), k.get("pos_zeta", 1.0), k.get("pos_max_tilt_rad", 0.5236)
	talt = k["target_altitude"]
	def tilt_ref(x, y, vx, vy):
		ad = lambda e, v: om*om*e - 2*ze*om*v
		return max(-mt, min(mt, -ad(-y, vy)/gvt)), max(-mt, min(mt, ad(-x, vx)/gvt))
	tail = math.ceil(steps*0.8)
	E = []
	for ep in range(n_eps):
		q = list(q0[ep*4:ep*4+4]); w = list(w0[ep*3:ep*3+3])
		c.reset(RC.yaw_from_quat(q)); c.seed_att_estimator(q)
		c.set_decode_seed(RC.disturbance_episode_seed(k["dist_seed"], ep))
		sim.reset(q, w)
		sim.set_translation(k["ep_mass"][ep]); sim.set_vertical_state(k["init_z"][ep], k["init_vz"][ep])
		sim.set_horizontal_state(k["init_x"][ep], k["init_y"][ep], 0.0, 0.0)
		hover = math.sqrt(k["ep_mass"][ep]*gvt/(4*kt)); coll = min(1.0, max(0.0, hover*(1+k["ep_collective_frac"][ep])))
		c.set_collective_anchor(coll)
		sim.set_disturbance(list(k["dist_tau_bias"]), k["dist_gust_sigma"], k["dist_gust_tau_c"], list(k["dist_motor_asym"]),
		                    k["dist_gyro_sigma"], k["dist_gyro_bias_walk"], k["dist_accel_sigma"],
		                    RC.disturbance_episode_seed(k["dist_seed"], ep), k["dist_dropout_prob"],
		                    k["dist_dropout_len_steps"], k["dist_obs_delay_steps"], k["dist_torque_scale_jitter"])
		y_kin = T.euler(q)[2]
		H = []; se = sh = 0.0; n = nh = 0
		for t in range(steps):
			if sim.is_unstable(): break
			gy, ac = sim.read_imu()
			qt = sim.quaternion; wt = sim.angular_velocity
			c.set_vertical_obs(coll, talt - sim.altitude, sim.vertical_velocity)
			x, y = sim.position_xy(); vx, vy = sim.velocity_xy()
			c.set_horizontal_obs(-x, -y, vx, vy)
			pwm = c.step(gy, ac, list(k.get("target", [0.0, 0.0, 0.0])))
			if t >= tail:
				fv = c.get_last_feature_vector()
				y_true = T.euler(qt)[2]
				y_dr = -fv[yi]
				y_mah = T.euler(c.att_est_quat())[2]
			sim.step(list(pwm))
			x, y = sim.position_xy(); vx, vy = sim.velocity_xy()
			rr, rp = tilt_ref(x, y, vx, vy)
			qr = T.rp_to_quat(rr, rp)
			err = sim.attitude_error(qr); se += err; n += 1
			if t >= tail:
				v = AX.rotvec(qr, sim.quaternion)
				d_dr = wrap(y_true - y_dr); d_mah = wrap(y_true - y_mah); d_kin = wrap(y_true - y_kin)
				H.append([abs(v[2]), v[2], abs(d_dr), abs(d_mah), abs(d_kin), abs(wrap(y_kin - y_dr)), abs(y_dr), abs(y_mah),
				          y_true**2, y_dr**2, d_dr**2, 2*y_dr*d_dr, abs(v[0]), abs(v[1])])
				sh += err; nh += 1
			y_kin += wt[2] * dt
		h = np.mean(H, 0)
		E.append([*h[:12], se/max(n, 1), sh/max(nh, 1), h[12], h[13]])
	return np.array(E)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
	ap.add_argument("--units", choices=["val", "test"], default="val")
	ap.add_argument("--max-units", type=int, default=100)
	ap.add_argument("--swap", action="store_true", help="yaw features from the student's own Mahony (existing genomes, NOT retrained)")
	o = ap.parse_args()
	import wnn.control._accel as A
	import ram_controller as RC
	from wnn.control.phased_ga import _test_thresholds, _rg_config
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.seeds import resolve_seed_set
	from probe_episode_failures import _load_stage_genome
	from wnn.control.feature_layout import feature_names
	txt, a, ec = T.run_args(o.tag)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	units = (T.val_units(o.tag) if o.units == "val" else [int(u) for u in a.test_seeds])[:o.max_units]
	g = re.findall(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", txt)[-1]
	lab, idx = g.split("#")
	_, spec, genome = _load_stage_genome(T.D + "ckpt_" + o.tag, T.STAGE[lab], int(idx))
	assert spec.obs_yaw_err and spec.obs_att_estimator and not spec.obs_yaw_trim_err and not spec.obs_yaw_trim_err_p
	yi = feature_names(spec).index("yaw_err")
	cap = {}; real = A.score_controllers_cpu
	def wrap_(controllers, *aa, **kk):
		cap.setdefault("calls", []).append((controllers[0], aa, kk)); return real(controllers, *aa, **kk)
	A.score_controllers_cpu = wrap_
	R, S = [], []  # S: [unit, stable, err_deg, steady_deg] from the scorer (swapped scorer under --swap)
	for unit in units:
		t0 = time.time()
		thr = _test_thresholds(a, ec, spec, unit, seeds.train, use_score=True)
		ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
		                         rg_config=_rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds,
		                         max_eval_workers_gpu=False)
		cap["calls"] = []
		m = ev.score_genomes([genome])[0]
		c, aa, k_raw = cap["calls"][0]; k = {**{"dt": 0.001, "dist_seed": 0}, **k_raw}
		assert k["dist_enabled"]
		ref = (m.stable_rate, m.mean_attitude_error_deg, m.mean_steady_error_deg)
		if o.swap:  # the PRODUCTION CPU scorer flies the swapped controller (existing genome, not retrained)
			c.set_yaw_from_att_est(True)
			r = real([c], *aa, **k_raw)[0]
			ref = (r[2], math.degrees(r[1]), math.degrees(r[5]))
		e = replay(RC, c, aa[0], aa[1], aa[2], aa[3], k, yi, o.swap)
		assert abs(math.degrees(e[:, 13].mean()) - ref[2]) < 0.02, (unit, math.degrees(e[:, 13].mean()), ref)
		R.append(e.mean(0)); S.append([unit, *ref])
		d = np.degrees(e.mean(0))
		print(f"[heading] {unit}{' SWAP' if o.swap else ''}: steady {d[13]:.3f} |v_z| {d[0]:.3f} | drift dr {d[2]:.3f} mah {d[3]:.3f} "
		      f"kin {d[4]:.3f} noise {d[5]:.3f} ({time.time()-t0:.0f}s)", flush=True)
	np.savez(o.out, rows=np.array(R), meta=np.array(S), cols=np.array(COLS), genome=g, partition=o.units, swap=o.swap)
	print(f"[heading] done -> {o.out}", flush=True)


if __name__ == "__main__":
	main()
