"""CTRL-83 stress panel: score ONE (controller x cell) on the 35 TEST units 99990101..135.

  score_cell.py <controller> <cell> <out.npz> [--rival-tail]
    controller: wnn_s1..wnn_s5 | mpcofA | mpcofC | pid | lqi | <EXTRA_WNN arm>_sN (cells.py)
    cell      : a cells.CELLS id

REPORT-ONLY. No training, no selection: the WNN is the published winner
(ctrl48/<tag>_winner.yaml.gz) with its thermometer thresholds fitted on the NOMINAL plant
(the train-seed fit the run itself used); only the plant the episodes fly is stressed.

Scorers (never reimplemented):
  WNN    ControllerEvaluator.score_genomes -> score_controllers_cpu (CPU), the call captured and
         re-issued with episode_errs=True (ctrl73/wnn_units.py's method).
  rivals classical_baseline.teacher_metrics -> score_classical_baseline, the call captured;
         --rival-tail re-issues it on every prefix 1..100 and differences the means to get the
         per-episode error (the scorer has no episode_errs output).

Gates (exit 2 on failure, nothing written):
  G1 re-issued WNN row == scored metrics (as wnn_units.py).
  G2 PLANT IDENTITY: per unit, every plant/episode kwarg the WNN call receives equals the one
     the rival call receives on the same stressed config (ICs, weather seed, motor wear, mass
     draws, lag, noise, drag, IMU, airframe — and, cell `all`, the per-episode lag-draw range:
     with the weather seed and episode count already equal, that fixes T_e per episode, since
     both calls draw it with the same Rust function). Run in every WNN job.
  G3 the cell's stress actually reached the call (the stressed kwarg differs from nominal).
  G4 cell nom only: WNN reproduces its logged END headline (ctrl80/rescore_end.py tolerances);
     rivals reproduce the per-unit rival lines the run printed ("vs PID[est]",
     "vs MPCOF>rate(h10)[est]"; 0.01 deg print precision).
"""
import argparse, json, math, os, re, sys, time
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/lacg/wnn/scripts")
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl80")
import inspect
import numpy as np
import cells as C

N_MET = 20
CAP_DEG = 8.0       # per-episode err cap for the tail (episode_tail.rs gate limit); applied to BOTH sides at report
                    # time — the WNN scorer caps only episodes it flags diverged, so raw values are cached
UNITS = list(range(99990101, 99990136))
# WNN kwarg -> rival kwarg, for the plant identity gate (G2).
PLANT_MAP = {
	"q0": "init_qs", "omega0": "init_omegas", "steps": "steps", "dt": "af_dt",
	"arm_length": "af_arm_length", "k_thrust": "af_k_thrust", "k_drag": "af_k_drag",
	"inertia": "af_inertia", "gravity": "af_gravity", "motor_lag_s": "motor_lag_s",
	"motor_lag_down_s": "motor_lag_down_s", "pwm_min": "pwm_min",
	**{k: k for k in ("dist_enabled", "dist_tau_bias", "dist_gust_sigma", "dist_gust_tau_c", "dist_motor_asym",
	                  "dist_gyro_sigma", "dist_gyro_bias_walk", "dist_accel_sigma", "dist_seed", "dist_dropout_prob",
	                  "dist_dropout_len_steps", "dist_obs_delay_steps", "dist_torque_scale_jitter",
	                  "translation", "imu_specific_force", "rotor_drag", "att_err_tilt_ref", "pos_zeta",
	                  "pos_max_tilt_rad", "alt_integral",
	                  # ABI 66 (CTRL-83 D1/D3): the design lag and the payload step reach both sides.
	                  "model_motor_lag_s", "payload_step_s", "payload_mass_scale",
	                  # ABI 69 (cell `all`): the per-episode plant-lag range. Same range + same dist_seed +
	                  # same episode count (init_qs) on both calls => the identical T_e per episode (one Rust
	                  # draw function; pinned by ctrl83_all_cell_tests point-range == fixed-plant identities).
	                  "train_motor_lag_range")},
	"target_altitude": "s1_target_altitude", "init_z": "s1_init_z", "init_vz": "s1_init_vz",
	"ep_mass": "s1_mass", "ep_collective_frac": "s1_collective_frac", "init_x": "s2_init_x",
	"init_y": "s2_init_y", "pos_omega": "pos_omega_n",
}
# Which WNN kwarg each axis must move (G3).
AXIS_KW = {"wear": ("dist_motor_asym",), "mass": ("ep_mass",), "lag": ("motor_lag_s",), "noise": ("dist_gyro_sigma",),
           "payload": ("payload_step_s",),
           # cell `all`: every stressed parameter must reach the call at once.
           "joint": ("dist_motor_asym", "ep_mass", "dist_gyro_sigma", "train_motor_lag_range")}


def die(msg):
	print(f"CTRL83 GATE FAIL: {msg}", file=sys.stderr)
	sys.exit(2)


def bound(fn, args, kwargs):
	"""Every argument of a captured call by name, scorer defaults filled in."""
	b = inspect.signature(fn).bind(*args, **kwargs)
	b.apply_defaults()
	return dict(b.arguments)


def as_arr(v):
	if v is None:
		return None
	if isinstance(v, str):
		return v
	return np.asarray(v, dtype=float).ravel()


def plant_identity(wnn_kw, riv_kw, unit):
	bad = []
	for wk, rk in PLANT_MAP.items():
		a, b = as_arr(wnn_kw.get(wk)), as_arr(riv_kw.get(rk))
		if a is None and b is None:
			continue
		if a is None or b is None or a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-9):
			bad.append(f"{wk}/{rk}: wnn {None if a is None else a[:6]} vs rival {None if b is None else b[:6]}")
	if bad:
		die(f"unit {unit}: WNN and rival do NOT fly the same plant: " + "; ".join(bad))


def load_run(controller):
	import ctrl_gap_student_trace as T
	from wnn.control import phased_ga as P
	tag = C.wnn_tag(controller if C.is_wnn(controller) else "wnn_s1")
	txt, a, ec0 = T.run_args(tag)
	P._resolve_ctbr(a, ec0)
	if [int(u) for u in a.test_seeds] != UNITS:
		die(f"{tag}: test seeds are not the 35 TEST units")
	return tag, txt, a, ec0


class Capture:
	"""Wrap one wnn.control._accel scorer; keep every call."""
	def __init__(self, name):
		import wnn.control._accel as A
		self.A, self.name, self.real, self.calls = A, name, getattr(A, name), []
		setattr(A, name, self)

	def __call__(self, *aa, **kk):
		self.calls.append((aa, kk))
		return self.real(*aa, **kk)


def rival_call(a, ec, unit, rv):
	"""(metrics dict, captured (args, kwargs)) of one rival row on one unit."""
	from wnn.control import classical_baseline as CB
	_, tid, hold, est, label = rv
	hold = int(a.action_repeat) if hold is None else hold
	cap = Capture("score_classical_baseline")
	try:
		draw = CB.HoldoutDraw(seed=unit, episodes=a.test_episodes, steps=ec.steps_per_episode, eval_folds=a.num_eval_folds)
		r = CB.teacher_metrics(ec, draw, CB.TeacherFeed(use_estimator=est), tid, hold, label)
	finally:
		setattr(cap.A, cap.name, cap.real)
	if len(cap.calls) != 1:
		die(f"rival {rv[0]} made {len(cap.calls)} scorer calls, expected 1")
	return r, cap.calls[0], cap.real


def rival_episodes(real, call, n):
	"""Per-episode (err deg, stable) by prefix differencing: e_k = k*m_k - (k-1)*m_(k-1)."""
	aa, kk = call
	q0, w0 = list(aa[1]), list(aa[2])
	per_ep = [k for k, v in kk.items() if isinstance(v, (list, tuple)) and len(v) == n]
	errs, stab, prev_e, prev_s = [], [], 0.0, 0.0
	for k in range(1, n + 1):
		kw = {**kk, **{key: list(kk[key])[:k] for key in per_ep}}
		r = real(aa[0], q0[:4 * k], w0[:3 * k], *aa[3:], **kw)
		se, ss = r[1] * k, r[0] * k
		errs.append(se - prev_e); stab.append(ss - prev_s)
		prev_e, prev_s = se, ss
	return np.array(errs), np.array(stab)


def score_rival(controller, cell, tail):
	rv = [r for r in C.RIVALS if r[0] == controller][0]
	tag, txt, a, ec0 = load_run("wnn_s1")
	ec = C.apply(cell, ec0)
	U, EPS = [], []
	for unit in UNITS:
		r, call, real = rival_call(a, ec, unit, rv)
		n = a.test_episodes
		st = float(r["stable_rate"])
		U.append([unit, st, r["mean_attitude_error_deg"], r["mean_steady_error_deg"], r["mean_position_error_m"],
		          r["mean_altitude_error_m"], r["mean_position_hold_m"], round((1 - st) * n)])
		if tail:
			e, s = rival_episodes(real, call, n)
			if abs(e.mean() - r["mean_attitude_error_deg"]) > 1e-6:
				die(f"{controller} u{unit}: prefix-differenced episodes do not average to the row")
			for i, (x, y) in enumerate(zip(e, s)):
				EPS.append([unit, i, float(x), round(float(y))])  # raw; the report caps both sides
	return tag, a, np.array(U), np.array(EPS) if EPS else np.zeros((0, 4))


def score_wnn(controller, cell):
	import wnn.control._accel as A
	from wnn.control.phased_ga import _test_thresholds, _rg_config, _ctl_load
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.seeds import resolve_seed_set
	tag, txt, a, ec0 = load_run(controller)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	pay = _ctl_load(f"/Users/lacg/wnn/logs/controller/ctrl48/{tag}_winner.yaml.gz")
	spec, genome = pay["spec"], pay["best_genome"]
	if genome.cells is None:
		die(f"{tag}: winner has no cells")
	# The address function is part of the published controller: fitted on the NOMINAL plant.
	thr = _test_thresholds(a, ec0, spec, UNITS[0], seeds.train, use_score=True)
	ec = C.apply(cell, ec0)
	axis = C.CELL[cell][1]
	real = A.score_controllers_cpu
	U, EPS = [], []
	for unit in UNITS:
		cap = Capture("score_controllers_cpu")
		try:
			ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
			                         rg_config=_rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds,
			                         max_eval_workers_gpu=False)
			m = ev.score_genomes([genome])[0]
		finally:
			A.score_controllers_cpu = real
		if len(cap.calls) != 1:
			die(f"{tag} u{unit}: {len(cap.calls)} scorer calls, expected 1")
		aa, kk = cap.calls[0]
		ax = []
		row = real(*aa, **{**kk, "axis_out": ax, "episode_errs": True, "tail_diverged_err_deg": CAP_DEG})[0]
		ne = int(row[N_MET]); ep = list(row[N_MET + 1:N_MET + 1 + ne])
		if abs(math.degrees(row[1]) - m.mean_attitude_error_deg) > 1e-9 or abs(math.degrees(row[5]) - m.mean_steady_error_deg) > 1e-9:
			die(f"{tag} u{unit}: G1 re-issued row != scored metrics")
		# G2: the rivals fly this plant too (cheap: one MPCOF est-fed call on the same config).
		wkw = bound(real, aa, kk)
		_, rcall, rreal = rival_call(a, ec, unit, C.RIVALS[0])
		plant_identity(wkw, bound(rreal, *rcall), unit)
		# G3: the stress reached the call.
		if axis in AXIS_KW and unit == UNITS[0]:
			nom_kw = _nominal_kw(a, ec0, unit)
			for kw in AXIS_KW[axis]:
				got, nom = as_arr(wkw.get(kw)), as_arr(nom_kw.get(PLANT_MAP[kw]))
				if got is None or (nom is not None and got.shape == nom.shape and np.allclose(got, nom)):
					die(f"{cell}: {kw} identical to nominal — the stress did not reach the WNN scorer")
			# D1 (ABI 66): a lag cell holds every model at the nominal design lag.
			if axis in ("lag", "joint") and not np.allclose(as_arr(wkw.get("model_motor_lag_s")), C.NOMINAL["lag_T"]):
				die(f"{cell}: model_motor_lag_s {wkw.get('model_motor_lag_s')} is not the nominal design lag")
			# Cell `all`: the plant's BASE lag (= the WNN twin) is the nominal one; only the per-episode draw moves.
			if axis == "joint":
				if not np.allclose(as_arr(wkw.get("motor_lag_s")), C.NOMINAL["lag_T"]):
					die(f"{cell}: motor_lag_s {wkw.get('motor_lag_s')} is not the nominal lag the twin keeps")
				if not np.allclose(as_arr(wkw.get("train_motor_lag_range")), C.CELL_LAG_RANGE[cell]):
					die(f"{cell}: train_motor_lag_range {wkw.get('train_motor_lag_range')} != {C.CELL_LAG_RANGE[cell]}")
		st = float(row[2])
		U.append([unit, st, math.degrees(row[1]), math.degrees(row[5]), m.mean_position_error_m, m.mean_altitude_error_m,
		          m.mean_position_hold_m, round((1 - st) * a.test_episodes)])
		for i, e in enumerate(ep):
			EPS.append([unit, i, float(e), -1])
	return tag, a, np.array(U), np.array(EPS)


def _nominal_kw(a, ec0, unit):
	_, rcall, rreal = rival_call(a, ec0, unit, C.RIVALS[0])
	return bound(rreal, *rcall)


def gate_nominal(controller, tag, txt, U):
	m = U.mean(0)
	if C.is_wnn(controller):
		from rescore_end import logged_end, TOL
		end = logged_end(tag)
		v = dict(stable=100 * m[1], err=m[2], steady=m[3], pos=m[4])
		bad = [f"{k} {v[k]:.4f} vs logged {end[k]}" for k in TOL if abs(v[k] - end[k]) > TOL[k]]
		if int(U[:, 7].sum()) != end["fails"]:
			bad.append(f"fails {int(U[:, 7].sum())} vs logged {end['fails']}")
		if bad:
			die(f"{tag}: nominal re-score does not reproduce the logged END: " + "; ".join(bad))
		return f"G4 ok: reproduces logged END stable {end['stable']} err {end['err']} steady {end['steady']} pos {end['pos']} fails {end['fails']}"
	printed = {"pid": "PID[est]", "mpcofA": "MPCOF>rate(h10)[est]"}.get(controller)
	if printed is None:
		return "G4 n/a: the run prints no per-unit line for this rival"
	sys.path.insert(0, "/Users/lacg/wnn/scripts")
	from ctrl61_rival_rows import unit_rows
	rows = unit_rows(f"/Users/lacg/wnn/logs/controller/ctrl48/{tag}.out")[printed]
	worst, worst_st = 0.0, 0.0
	for u in U:
		p = rows[int(u[0])]
		worst_st = max(worst_st, abs(100 * u[1] - p[0]))
		worst = max(worst, abs(u[2] - p[1]), abs(u[3] - p[2]), abs(u[4] - p[3]))
	if worst > 0.0051 or worst_st > 0.051:
		die(f"{controller}: nominal does not reproduce the run's printed per-unit '{printed}' lines (max dev {worst:.4f})")
	return f"G4 ok: 35/35 units reproduce the run's printed '{printed}' lines (max dev {worst:.4f}, print precision 0.005)"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("controller", choices=C.ALL_CONTROLLERS)
	ap.add_argument("cell", choices=list(C.CELL))
	ap.add_argument("out")
	ap.add_argument("--rival-tail", action="store_true")
	o = ap.parse_args()
	import wnn.control._accel as A
	t0 = time.time()
	if C.is_wnn(o.controller):
		tag, a, U, EPS = score_wnn(o.controller, o.cell)
	else:
		tag, a, U, EPS = score_rival(o.controller, o.cell, o.rival_tail)
	txt = open(f"/Users/lacg/wnn/logs/controller/ctrl48/{tag}.out", errors="replace").read()
	g4 = gate_nominal(o.controller, tag, txt, U) if o.cell == "nom" else "G4 n/a (stressed cell)"
	abi = int(__import__("ram_controller").ABI_VERSION)
	m = U.mean(0)
	fails = int(U[:, 7].sum())
	tail = (f"p99.9(cap {CAP_DEG:g}) {np.percentile(np.minimum(EPS[:, 2], CAP_DEG), 99.9, method='inverted_cdf'):.3f}"
	        if len(EPS) else "p99.9 n/a")
	meta = dict(controller=o.controller, cell=o.cell, cell_def=C.CELL[o.cell][2], tag=tag, abi=abi,
	            seconds=round(time.time() - t0, 1), gate=g4, units=len(U), episodes=int(a.test_episodes))
	tmp = o.out + ".tmp.npz"
	np.savez(tmp, units=U, eps=EPS, meta=json.dumps(meta))
	os.replace(tmp, o.out)
	print(f"[ctrl83] {o.controller:8s} {o.cell:7s} stable {100*m[1]:.2f} err {m[2]:.3f} steady {m[3]:.3f} pos {m[4]:.4f} "
	      f"fails {fails}/{len(U)*a.test_episodes} worst-unit err {U[:, 2].max():.3f} {tail} | ABI {abi} | {meta['seconds']}s | {g4}", flush=True)


if __name__ == "__main__":
	main()
