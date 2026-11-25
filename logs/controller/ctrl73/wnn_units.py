"""CTRL-70: per-unit / per-axis / per-episode decomposition of a published winner on the 35 TEST units.
Production scorer only (ControllerEvaluator.score_genomes -> score_controllers_cpu), exact call captured and
re-issued with axis_out + episode_errs. Gate: re-issued row == scored metrics; mean over units == run's HEADLINE line.
For each FAILED episode (err>5 or diverged), the call is re-issued on the first k+1 / k episodes (prefix) and
differenced -> that episode's own per-axis err/hold + mid window (gated: prefix ep errs == full-call ep errs)."""
import math, os, re, sys, time
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T

tag, out = sys.argv[1], sys.argv[2]
import wnn.control._accel as A
from wnn.control.phased_ga import _test_thresholds, _rg_config, _ctl_load
from wnn.control.evaluator import ControllerEvaluator, disturbance_stream, fold_pool_seed
from wnn.seeds import resolve_seed_set
txt, a, ec = T.run_args(tag)
seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
units = [int(u) for u in a.test_seeds]
head = re.findall(r"\[stage-select\] HEADLINE held-out: stable=([\d.]+)%\s+err=([\d.]+)°\s+steady=([\d.]+)°", txt)[-1]
pay = _ctl_load(T.D + tag + "_winner.yaml.gz")
spec, genome = pay["spec"], pay["best_genome"]
assert genome.cells is not None
t0 = time.time()
thr = _test_thresholds(a, ec, spec, units[0], seeds.train, use_score=True)

import dataclasses as _dc
_KO = os.environ.get("KO", "")
def knockout(ec):
	if not _KO: return ec
	d = ec.disturbance
	ch = {"asym": dict(motor_asym_mag=0.0), "jitter": dict(torque_scale_jitter=0.0),
	      "sensor": dict(gyro_sigma=0.0, gyro_bias_walk=0.0, accel_sigma=0.0)}[_KO]
	return _dc.replace(ec, disturbance=_dc.replace(d, **ch))

ec = knockout(ec)
print(f"[{tag[-30:]}] KO={_KO} thresholds {time.time()-t0:.0f}s", flush=True)
cap = {}; real = A.score_controllers_cpu
def wrap(controllers, *aa, **kk):
	cap.setdefault("calls", []).append((controllers, aa, kk)); return real(controllers, *aa, **kk)
A.score_controllers_cpu = wrap
N_MET = 20

def issue(ctrls, aa, kk, n=None):
	"""Re-issue the captured call, optionally on the first n episodes (prefix). Returns (row, axes14, ep_errs)."""
	aa = list(aa); kk = dict(kk)
	N = aa[2]
	if n is not None:
		aa[0] = list(aa[0])[:4 * n]; aa[1] = list(aa[1])[:3 * n]; aa[2] = n
		for k, v in list(kk.items()):
			if isinstance(v, (list, tuple)) and len(v) == N and k not in ("dist_tau_bias", "dist_motor_asym", "inertia", "target"):
				kk[k] = list(v)[:n]
	ax = []
	kk.update(axis_out=ax, episode_errs=True, tail_diverged_err_deg=8.0)
	row = real(ctrls, *aa, **kk)[0]
	ne = int(row[N_MET]); ep = list(row[N_MET + 1:N_MET + 1 + ne])
	return list(row[:N_MET]), ax, ep

U, EPS, FAILS = [], [], []
for unit in units:
	ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
	                         rg_config=_rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds,
	                         max_eval_workers_gpu=False)
	cap["calls"] = []
	m = ev.score_genomes([genome])[0]
	assert len(cap["calls"]) == 1, len(cap["calls"])
	ctrls, aa, kk = cap["calls"][0]
	row, ax, ep = issue(ctrls, aa, kk)
	assert abs(math.degrees(row[1]) - m.mean_attitude_error_deg) < 1e-9 and abs(math.degrees(row[5]) - m.mean_steady_error_deg) < 1e-9
	assert abs(math.degrees(ax[3]) - m.mean_attitude_error_deg) < 0.01 and abs(math.degrees(ax[7]) - m.mean_steady_error_deg) < 0.01
	pool = fold_pool_seed(unit, 0) if a.num_eval_folds > 1 else unit
	_, asym = disturbance_stream(ec.disturbance, pool)
	U.append([unit, row[2], *[math.degrees(row[i]) for i in (1, 5, 19)], m.mean_position_error_m, m.mean_altitude_error_m,
	          m.mean_position_hold_m, row[3], row[6], row[7], row[9], *ax, *asym])
	for i, e in enumerate(ep):
		EPS.append([unit, i, e])
	bad = [i for i, e in enumerate(ep) if e > 5.0]
	for k in bad:
		r1, a1, e1 = issue(ctrls, aa, kk, k + 1)
		assert np.allclose(e1, ep[:k + 1], atol=1e-9), "prefix call does not reproduce the episodes"
		if k > 0:
			r0, a0, e0 = issue(ctrls, aa, kk, k)
		else:
			r0, a0 = [0.0] * N_MET, [0.0] * 14
		n1, n0 = a1[12], a0[12] if k > 0 else 0.0
		h1, h0 = a1[13], (a0[13] if k > 0 else 0.0)
		own_err = [(a1[j] * n1 - a0[j] * n0) for j in range(4)]
		reached = h1 - h0
		own_hold = [(a1[j] * h1 - a0[j] * h0) / reached if reached > 0 else float("nan") for j in range(4, 8)]
		own_mid = (r1[19] * (k + 1) - r0[19] * k)
		own_stable = (r1[2] * (k + 1) - r0[2] * k)
		FAILS.append([unit, k, ep[k], reached, own_stable, *[math.degrees(x) for x in own_err], *[math.degrees(x) for x in own_hold], math.degrees(own_mid)])
		print(f"  FAIL u{unit} ep{k}: err {ep[k]:.2f} reached_hold {reached:.0f} stable {own_stable:.0f} | whole r/p/y/|v| "
		      + "/".join(f"{math.degrees(x):.2f}" for x in own_err) + " | hold r/p/y/|v| " + "/".join(f"{math.degrees(x):.2f}" for x in own_hold)
		      + f" | mid {math.degrees(own_mid):.2f}", flush=True)
r = np.array(U); mm = r.mean(0)
nf = int(round(sum((1 - x) * a.test_episodes for x in r[:, 1])))
print(f"[{tag}] {time.time()-t0:.0f}s stable {100*mm[1]:.2f} err {mm[2]:.3f} steady {mm[3]:.3f} mid {mm[4]:.3f} pos {mm[5]:.4f} fails {nf}"
      f" | logged HEADLINE {head}", flush=True)
np.savez(out, units=r, eps=np.array(EPS), fails=np.array(FAILS) if FAILS else np.zeros((0, 17)), tag=tag, head=np.array(head, dtype=float))
