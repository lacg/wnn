"""docs/ctrl_gap_after_block9.md §B: re-fly a run's HEADLINE genome on VAL units and
record what the decomposition needs. Read-only; scores through the INSTALLED wheel
(the scorer is never reimplemented — this is the block-6 probe, /tmp/steady_block6/probe.py,
pointed at val units and extended with a per-tick trace).

Per unit: the production scorer scores the genome (ControllerEvaluator.score_genomes,
CPU), its exact kwargs are captured, and the same episodes are replayed step by step in
cpu_score::rollout_one's order. GATE: the replay's steady must reproduce the scorer's.

Outputs (npz + raw f32):
  rows    hold-window (t >= 0.8*steps) samples, every episode:
          [unit, ep, t, err, e_true_r, e_true_p, yaw_true, e_est_r, e_est_p, yaw_est,
           roll_ref, pitch_ref, x, y, vx, vy, dz, abp_r, abp_p]
  eps     per episode: [unit, ep, steady, mean_err, steps_done, pos3d_mean, pos3d_fifths x5,
          abs dx, dy, dz means, hold abs dx, dy, dz]
  trace   (--trace-episodes N per unit) one f32 record per tick, for the Rust
          label replay (label_replay_tests.rs): [unit_idx, ep, t, gyro x3, q_true x4, x, y,
          vx, vy, alt, vz, sp x4 (the student's held CTBR setpoint), pwm x4 (applied),
          collective anchor, target altitude]

VAL ONLY: units come from --units-from's stage-select val units; any seed in
99990101..99990400 is refused.
"""
import argparse, math, os, re, shlex, sys
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np

D = "/Users/lacg/wnn/logs/controller/ctrl48/"
STAGE = {"GRID": "stage0_grid", "NEURONS": "stage1_neurons", "MEMORY": "stage4_memory"}
TRACE_W = 26
EXTRA_FEATURES = ("roll_err", "pitch_err", "roll_i", "pitch_i", "yaw_err", "yaw_err_i", "cresid_x", "cresid_y", "cresid_z", "gyro_z")


def run_args(tag):
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args
	txt = open(D + tag + ".out", errors="replace").read()
	argv = [l for l in txt.splitlines() if l.startswith("[argv]")][0][len("[argv]"):]
	a = build_arg_parser().parse_args(shlex.split(argv))
	return txt, a, episode_config_from_args(a)


def val_units(tag):
	from wnn.control.val_units import checked_val_unit_seeds
	from wnn.seeds import resolve_seed_set
	_, a, _ = run_args(tag)
	s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	units = checked_val_unit_seeds(a, s)
	assert not any(99990101 <= u <= 99990400 for u in units), "report-only units refused"
	return units


def euler(q):
	w, x, y, z = q
	return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
	        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
	        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rp_to_quat(roll, pitch):
	cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
	return [cr * cp, sr * cp, cr * sp, -sr * sp]


def replay(RC, c, q0, w0, n_eps, steps, k, ui, n_trace, extra_idx=()):
	sim = RC.AttitudeSim(dt=k.get("dt", 0.001), arm_length=k["arm_length"], k_thrust=k["k_thrust"], k_drag=k["k_drag"],
	                     inertia=list(k["inertia"]), gravity=k["gravity"])
	if k.get("motor_lag_s", 0.0) > 0:
		sim.set_motor_lag(k["motor_lag_s"])
	c.set_observer_motor_lag(k.get("motor_lag_s", 0.0))
	if k.get("imu_specific_force"):
		sim.set_imu_specific_force(True)
	if k.get("rotor_drag", 0.0) != 0.0:
		sim.set_rotor_drag(k["rotor_drag"])
	gvt, kt = k["gravity"], k["k_thrust"]
	om, ze, mt = k.get("pos_omega", 1.0), k.get("pos_zeta", 1.0), k.get("pos_max_tilt_rad", 0.5236)
	talt = k["target_altitude"]

	def tilt_ref(x, y, vx, vy):
		ad = lambda e, v: om * om * e - 2 * ze * om * v
		return max(-mt, min(mt, -ad(-y, vy) / gvt)), max(-mt, min(mt, ad(-x, vx) / gvt))

	tail = math.ceil(steps * 0.8)
	rows, eps = [], []
	trace = np.zeros((n_trace * steps, TRACE_W), dtype=np.float32)
	nt = 0
	for ep in range(n_eps):
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
		ts = es = 0.0; tc = ec_ = 0
		p5 = [0.0] * 5; n5 = [0] * 5; ax = [0.0] * 6
		for t in range(steps):
			if sim.is_unstable():
				break
			gy, ac = sim.read_imu()
			c.set_vertical_obs(coll, talt - sim.altitude, sim.vertical_velocity)
			x, y = sim.position_xy(); vx, vy = sim.velocity_xy()
			c.set_horizontal_obs(-x, -y, vx, vy)
			if ep < n_trace:
				qt0 = sim.quaternion
				pre = [ui, ep, t, *gy, *qt0, x, y, vx, vy, sim.altitude, sim.vertical_velocity]
			pwm = c.step(gy, ac, list(k.get("target", [0.0, 0.0, 0.0])))
			if ep < n_trace:
				sp = c.ctbr_held_setpoint
				trace[nt] = pre + list(sp() if callable(sp) else sp) + list(pwm) + [coll, talt]
				nt += 1
			qe = c.att_est_quat()
			fv = c.get_last_feature_vector() if t >= tail else None
			sim.step(list(pwm))
			x, y = sim.position_xy(); vx, vy = sim.velocity_xy()
			rr, rp = tilt_ref(x, y, vx, vy)
			err = sim.attitude_error(rp_to_quat(rr, rp))
			dz = talt - sim.altitude
			es += err; ec_ += 1
			f = min(4, 5 * t // steps)
			p5[f] += math.sqrt(x * x + y * y + dz * dz); n5[f] += 1
			ax[0] += abs(x); ax[1] += abs(y); ax[2] += abs(dz)
			if t >= tail:
				ts += err; tc += 1
				ax[3] += abs(x); ax[4] += abs(y); ax[5] += abs(dz)
				et = euler(sim.quaternion); ee = euler(qe)
				rows.append([ui, ep, t, err, et[0] - rr, et[1] - rp, et[2], ee[0] - rr, ee[1] - rp, ee[2], rr, rp, x, y, vx, vy, dz,
				             fv[-2], fv[-1], *[fv[i] for i in extra_idx]])
		n = max(ec_, 1)
		eps.append([ui, ep, ts / tc if tc else float("nan"), es / n, ec_, sum(p5) / n,
		            *[p5[i] / n5[i] if n5[i] else float("nan") for i in range(5)],
		            ax[0] / n, ax[1] / n, ax[2] / n, *[ax[3 + i] / tc if tc else float("nan") for i in range(3)]])
	return np.array(rows), np.array(eps), trace[:nt]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tag", required=True)
	ap.add_argument("--units-from", required=True, help="run tag whose stage-select VAL units are flown")
	ap.add_argument("--out", required=True)
	ap.add_argument("--trace-episodes", type=int, default=20)
	args = ap.parse_args()
	import wnn.control._accel as A
	import ram_controller as RC
	from wnn.control.phased_ga import _test_thresholds, _rg_config
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.seeds import resolve_seed_set
	from probe_episode_failures import _load_stage_genome
	txt, a, ec = run_args(args.tag)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	g = re.findall(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", txt)[-1]
	lab, idx = g.split("#")
	_, spec, genome = _load_stage_genome(D + "ckpt_" + args.tag, STAGE[lab], int(idx))
	units = val_units(args.units_from)
	print(f"[trace] {args.tag} headline {g} out_neurons={genome.output_neurons} bits={genome.output_bits_per_neuron} "
	      f"on {len(units)} VAL units {units[0]}..{units[-1]}", flush=True)
	# Block-10 pre-check: the student's own input VALUES in the hold (appended to rows after
	# abp_r, abp_p, in this order), read from the feature vector by name.
	from wnn.control.feature_layout import feature_names
	names = feature_names(spec)
	extra = [n for n in EXTRA_FEATURES if n in names]
	extra_idx = tuple(names.index(n) for n in extra)
	cap = {}
	real = A.score_controllers_cpu

	def wrap(controllers, *aa, **kk):
		cap.setdefault("calls", []).append((controllers[0], aa, kk))
		return real(controllers, *aa, **kk)
	A.score_controllers_cpu = wrap
	R, E, T, S = [], [], [], []
	for ui, unit in enumerate(units):
		thr = _test_thresholds(a, ec, spec, unit, seeds.train, use_score=True)
		ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
		                         rg_config=_rg_config(a, ec, unit), max_train_workers=1,
		                         num_eval_folds=a.num_eval_folds, max_eval_workers_gpu=False)
		cap["calls"] = []
		m = ev.score_genomes([genome])[0]
		assert len(cap["calls"]) == 1, len(cap["calls"])
		c, aa, k = cap["calls"][0]
		k = {**{"dt": 0.001, "dist_seed": 0}, **k}
		rows, eps, tr = replay(RC, c, aa[0], aa[1], aa[2], aa[3], k, ui, args.trace_episodes, extra_idx)
		rs = math.degrees(float(np.nanmean(eps[:, 2])))
		assert abs(rs - m.mean_steady_error_deg) < 0.005, (unit, rs, m.mean_steady_error_deg)
		R.append(rows); E.append(eps); T.append(tr)
		S.append([unit, m.acc, m.mean_attitude_error_deg, m.mean_steady_error_deg])
		print(f"[trace] unit {unit}: scorer stable {m.acc*100:.1f}% err {m.mean_attitude_error_deg:.3f} steady "
		      f"{m.mean_steady_error_deg:.4f} | replay steady {rs:.4f}", flush=True)
	np.savez(args.out + ".npz", rows=np.vstack(R).astype(np.float32), eps=np.vstack(E), scorer=np.array(S),
	         units=np.array(units), out_neurons=genome.output_neurons, extra=np.array(extra))
	np.vstack(T).astype("<f4").tofile(args.out + ".trace.f32")
	S = np.array(S)
	print(f"[trace] VAL mean: stable {S[:,1].mean()*100:.2f}% err {S[:,2].mean():.3f} steady {S[:,3].mean():.3f}", flush=True)


if __name__ == "__main__":
	main()
