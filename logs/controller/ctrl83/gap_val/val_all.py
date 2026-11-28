"""VAL x cell `all` decomposition (diagnosis only; never TEST). usage: val_all.py <wnn_id e.g. l2bnmw_s1> <out.npz> [--rivals]"""
import math, os, sys, subprocess, time
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl83"); sys.path.insert(0, "/Users/lacg/wnn/scripts")
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl80")
import numpy as np
import cells as C, score_cell as S
import ram_controller as RC
CELL = os.environ.get("CELL", "all")
LAGR = (C.ALL_LEVELS_V1 if CELL == "all_v1" else C.ALL_LEVELS)["lag_range"]
LAGBIN = "/Volumes/20260401-WDBlack-SN850X-2TB/cargo-target/release/lagdraw"
TEST = set(range(99990101, 99990401))

def lags(dist_seed, n):
	es = [str(RC.disturbance_episode_seed(int(dist_seed), e)) for e in range(n)]
	out = subprocess.run([LAGBIN, *map(str, LAGR), *es], capture_output=True, text=True, check=True).stdout
	return [float(x) for x in out.split()]

def main():
	cid, out = sys.argv[1], sys.argv[2]; rivals = "--rivals" in sys.argv
	import wnn.control._accel as A
	from wnn.control.phased_ga import _test_thresholds, _rg_config, _ctl_load
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control.val_units import checked_val_unit_seeds
	from wnn.seeds import resolve_seed_set
	tag, txt, a, ec0 = S.load_run(cid)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	units = checked_val_unit_seeds(a, seeds)
	assert not (set(units) & TEST), "VAL only"
	pay = _ctl_load(f"/Users/lacg/wnn/logs/controller/ctrl48/{tag}_winner.yaml.gz")
	spec, genome = pay["spec"], pay["best_genome"]
	thr = _test_thresholds(a, ec0, spec, S.UNITS[0], seeds.train, use_score=True)  # nominal fit, as score_cell
	ec = C.apply(CELL, ec0)
	real = A.score_controllers_cpu
	U, E, R = [], [], []
	for unit in units:
		t0 = time.time()
		cap = S.Capture("score_controllers_cpu")
		try:
			ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
			                         rg_config=_rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds,
			                         max_eval_workers_gpu=False)
			m = ev.score_genomes([genome])[0]
		finally:
			A.score_controllers_cpu = real
		aa, kk = cap.calls[0]
		ax = []
		row = real(*aa, **{**kk, "axis_out": ax, "episode_errs": True, "tail_diverged_err_deg": S.CAP_DEG})[0]
		ne = int(row[S.N_MET]); ep = list(row[S.N_MET + 1:S.N_MET + 1 + ne])
		assert abs(math.degrees(row[1]) - m.mean_attitude_error_deg) < 1e-9
		wkw = S.bound(real, aa, kk)
		_, rcall, rreal = S.rival_call(a, ec, unit, C.RIVALS[0])
		S.plant_identity(wkw, S.bound(rreal, *rcall), unit)          # G2 on VAL too
		asym = list(np.asarray(wkw["dist_motor_asym"], float))
		mass = list(np.asarray(wkw["ep_mass"], float)); cf = list(np.asarray(wkw["ep_collective_frac"], float))
		L = lags(wkw["dist_seed"], ne)
		U.append([unit, row[2], math.degrees(row[1]), math.degrees(row[5]), *[math.degrees(x) for x in ax[:8]], *asym])
		rv = {}
		if rivals:
			for rid in ("mpcofC", "mpcofA"):
				rr = [r for r in C.RIVALS if r[0] == rid][0]
				r, call, rre = S.rival_call(a, ec, unit, rr)
				e, s = S.rival_episodes(rre, call, ne) if rid == "mpcofC" else (np.full(ne, np.nan), np.full(ne, np.nan))
				rv[rid] = (r, e, s)
				R.append([unit, ["mpcofC", "mpcofA"].index(rid), r["stable_rate"], r["mean_attitude_error_deg"], r["mean_steady_error_deg"]])
		for i in range(ne):
			ce = rv["mpcofC"][1][i] if rivals else np.nan
			cs = rv["mpcofC"][2][i] if rivals else np.nan
			E.append([unit, i, ep[i], mass[i], cf[i], L[i], ce, cs])
		print(f"u{unit} stable {100*row[2]:.1f} err {math.degrees(row[1]):.3f} steady {math.degrees(row[5]):.3f} "
		      + (f"| C err {rv['mpcofC'][0]['mean_attitude_error_deg']:.3f} A err {rv['mpcofA'][0]['mean_attitude_error_deg']:.3f} " if rivals else "")
		      + f"asym {np.round(asym,3)} {time.time()-t0:.0f}s", flush=True)
	np.savez(out, units=np.array(U), eps=np.array(E), riv=np.array(R) if R else np.zeros((0, 5)), tag=tag)

main()
