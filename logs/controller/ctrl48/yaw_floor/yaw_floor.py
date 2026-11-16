"""CTRL-48 proposal #3 (Luiz 16/11/2026): the YAW FLOOR. Per-axis (roll / pitch / yaw, BODY frame,
v = Log(q_ref^-1 q_true) against the scorer's own G5 tilt reference) err and steady, plus stable/fails,
for three rows on the SAME episodes (CRN, asserted):

  MPCOF oracle : MPCOF>rate(h<ar>) cascade rival fed the TRUE quaternion  (control-law floor)
  MPCOF est    : the same rival fed the Mahony estimate (dead-reckoned yaw) — the 10b comparator row
  C            : arm C's stage-select HEADLINE genome (VAL replay through the production CPU scorer)

Teacher rows: ABI-53 score_classical_baseline(axis_out=) via classical_baseline.teacher_metrics(axes=True)
— measured inside the Rust scorer. C row: l2yp_investigation/axis_trace.replay (Rust sim/controller
stepped in the scorer's order, gated: replay steady == scorer steady to 0.01 deg).

--units val (default): the run's 20 stage-select VAL units (TUNING partition).
--units test: the run's --test-seeds (REPORT partition) — descriptive, one-shot, never for tuning.
Output: one npz per (tag, partition) with per-unit rows; aggregation is yaw_floor_table.py."""
import argparse, math, os, re, sys, shlex, time
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl48/l2yp_investigation")
import numpy as np
import ctrl_gap_student_trace as T
import axis_trace as AX

ROW = ("stable", "err", "steady", "err_roll", "err_pitch", "err_yaw", "hold_roll", "hold_pitch", "hold_yaw",
       "signed_yaw", "fed_yaw_drift", "episodes")


def teacher_ec(tag):
	from wnn.control import phased_ga as P
	txt = open(T.D + tag + ".out", errors="replace").read()
	argv = shlex.split([l for l in txt.splitlines() if l.startswith("[argv]")][0][len("[argv]"):])
	a = P.build_arg_parser().parse_args(argv); ec = P.episode_config_from_args(a); P._resolve_ctbr(a, ec)
	return a, ec


def teacher_row(CB, ec, a, unit, use_est):
	draw = CB.HoldoutDraw(seed=unit, episodes=a.test_episodes, steps=ec.steps_per_episode, eval_folds=a.num_eval_folds)
	r = CB.teacher_metrics(ec, draw, CB.TeacherFeed(use_estimator=use_est), 4, int(a.action_repeat), "MPCOF>rate", axes=True)
	x = r["axes"]; d = math.degrees
	return [r["stable_rate"], r["mean_attitude_error_deg"], r["mean_steady_error_deg"],
	        d(x["err_roll"]), d(x["err_pitch"]), d(x["err_yaw"]), d(x["hold_roll"]), d(x["hold_pitch"]), d(x["hold_yaw"]),
	        d(x["signed_yaw"]), d(x["fed_yaw_drift"]), x["episodes_all"]], draw


def crn_check(CB, ec_t, draw, aa, k):
	"""The WNN scorer call and the teacher draw must fly the same episodes."""
	q0, w0, f = CB._episode_fields(ec_t, draw)
	s1 = CB._stage1_fields(ec_t, draw)
	assert np.array_equal(np.asarray(q0, np.float32), np.asarray(aa[0], np.float32)), "q0 differs"
	assert np.array_equal(np.asarray(w0, np.float32), np.asarray(aa[1], np.float32)), "w0 differs"
	assert int(f["dist_seed"]) == int(k["dist_seed"]), (f["dist_seed"], k["dist_seed"])
	assert list(f["dist_motor_asym"]) == [float(v) for v in k["dist_motor_asym"]]
	for key_t, key_w in (("s2_init_x", "init_x"), ("s2_init_y", "init_y"), ("s1_init_z", "init_z")):
		assert np.allclose(np.asarray(s1[key_t], np.float32), np.asarray(k[key_w], np.float32), atol=0, rtol=0), key_t


def wnn_row(RC, c, aa, k, ui):
	ep, hr = AX.replay(RC, c, aa[0], aa[1], aa[2], aa[3], k, ui, [])
	d = math.degrees
	assert np.all(ep[:, 2] == aa[3]), "a WNN episode diverged before the hold window — hold means need nh accounting"
	sy = np.array([hr[hr[:, 1] == e, 5].mean() for e in np.unique(hr[:, 1])])
	return [d(ep[:, 3].mean()), d(ep[:, 11].mean()), d(ep[:, 4].mean()), d(ep[:, 5].mean()), d(ep[:, 6].mean()),
	        d(ep[:, 12].mean()), d(ep[:, 13].mean()), d(ep[:, 14].mean()), d(sy.mean())]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
	ap.add_argument("--units", choices=["val", "test"], default="val")
	ap.add_argument("--max-units", type=int, default=100)
	ap.add_argument("--teachers-only", action="store_true")
	ap.add_argument("--gate-deg", type=float, default=0.01, help="replay-vs-scorer steady gate (deg)")
	o = ap.parse_args()
	import wnn.control._accel as A
	import ram_controller as RC
	from wnn.control import classical_baseline as CB
	from wnn.control.phased_ga import _test_thresholds, _rg_config
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.seeds import resolve_seed_set
	from probe_episode_failures import _load_stage_genome
	assert RC.ABI_VERSION >= 53, RC.ABI_VERSION
	txt, a, ec = T.run_args(o.tag)
	a_t, ec_t = teacher_ec(o.tag)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	units = (T.val_units(o.tag) if o.units == "val" else [int(u) for u in a.test_seeds])[:o.max_units]
	g = re.findall(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", txt)[-1]
	lab, idx = g.split("#")
	_, spec, genome = _load_stage_genome(T.D + "ckpt_" + o.tag, T.STAGE[lab], int(idx))
	print(f"[yaw-floor] {o.tag} {o.units.upper()} units={len(units)} C genome={g}", flush=True)
	cap = {}; real = A.score_controllers_cpu
	def wrap(controllers, *aa, **kk):
		cap.setdefault("calls", []).append((controllers[0], aa, kk)); return real(controllers, *aa, **kk)
	A.score_controllers_cpu = wrap
	R = {"oracle": [], "est": [], "C": [], "unit": []}
	for ui, unit in enumerate(units):
		t0 = time.time()
		orc, draw = teacher_row(CB, ec_t, a_t, unit, False)
		est, _ = teacher_row(CB, ec_t, a_t, unit, True)
		R["oracle"].append(orc); R["est"].append(est); R["unit"].append(unit)
		if not o.teachers_only:
			thr = _test_thresholds(a, ec, spec, unit, seeds.train, use_score=True)
			ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
			                         rg_config=_rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds,
			                         max_eval_workers_gpu=False)
			cap["calls"] = []
			m = ev.score_genomes([genome])[0]
			c, aa, k = cap["calls"][0]; k = {**{"dt": 0.001, "dist_seed": 0}, **k}
			crn_check(CB, ec_t, draw, aa, k)
			w = wnn_row(RC, c, aa, k, ui)
			assert abs(w[1] - m.mean_steady_error_deg) < o.gate_deg, (unit, w[1], m.mean_steady_error_deg)
			R["C"].append([m.stable_rate, m.mean_attitude_error_deg, m.mean_steady_error_deg, *w[2:], float("nan"), float(a.test_episodes)])
			assert len(R["C"][-1]) == len(ROW) == len(orc)
		print(f"[yaw-floor] unit {unit}: oracle steady {orc[2]:.3f} yaw {orc[8]:.3f} | est steady {est[2]:.3f} yaw {est[8]:.3f}"
		      + (f" | C steady {R['C'][-1][2]:.3f} yaw {R['C'][-1][8]:.3f}" if not o.teachers_only else "")
		      + f"  ({time.time()-t0:.0f}s)", flush=True)
	np.savez(o.out, unit=np.array(R["unit"]), oracle=np.array(R["oracle"]), est=np.array(R["est"]),
	         C=np.array(R["C"]), genome=g, partition=o.units, tag=o.tag)
	print(f"[yaw-floor] done -> {o.out}", flush=True)


if __name__ == "__main__":
	main()
