"""CTRL-82 VAL pass 1 (mechanism only, never a result): per-episode whole err (deg) of a run's HEADLINE genome on
its first N VAL units, production CPU scorer re-issued with episode_errs=True (ctrl73/wnn_units.py pattern).
Usage: eps82.py <tag> <out.npz> <n_val_units>   -> eps rows [unit, ep, err_deg]; feature names; val units."""
import math, os, sys
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T
tag, out, nunits = sys.argv[1], sys.argv[2], int(sys.argv[3])
import wnn.control._accel as A
import wnn.control.phased_ga as PG
from wnn.control.evaluator import ControllerEvaluator
from wnn.control.feature_layout import feature_names
from wnn.seeds import resolve_seed_set
txt, a, ec = T.run_args(tag)
seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
pay = PG._ctl_load(T.D + tag + "_winner.yaml.gz"); spec, genome = pay["spec"], pay["best_genome"]
from wnn.control.val_units import checked_val_unit_seeds
a.val_units = nunits
units = checked_val_unit_seeds(a, seeds)   # extends the SAME val list past 20; disjointness from TEST/train checked
assert not any(99990101 <= u <= 99990400 for u in units)
cap = {}; real = A.score_controllers_cpu
def wrap(controllers, *aa, **kk):
	cap.setdefault("calls", []).append((controllers, aa, kk)); return real(controllers, *aa, **kk)
A.score_controllers_cpu = wrap
thr = PG._test_thresholds(a, ec, spec, units[0], seeds.train, use_score=True)
N_MET = 20; EPS = []
for unit in units:
	ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
	                         rg_config=PG._rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds, max_eval_workers_gpu=False)
	cap["calls"] = []
	m = ev.score_genomes([genome])[0]
	ctrls, aa, kk = cap["calls"][0]; kk = dict(kk); ax = []
	kk.update(axis_out=ax, episode_errs=True, tail_diverged_err_deg=8.0)
	row = real(ctrls, *aa, **kk)[0]
	assert abs(math.degrees(row[1]) - m.mean_attitude_error_deg) < 1e-9
	ne = int(row[N_MET]); ep = list(row[N_MET + 1:N_MET + 1 + ne])
	EPS += [[unit, i, e] for i, e in enumerate(ep)]
	print(f"VAL unit {unit}: err {m.mean_attitude_error_deg:.3f} stable {m.stability_rate if hasattr(m,'stability_rate') else float('nan')} max ep err {max(ep):.2f}", flush=True)
np.savez(out, eps=np.array(EPS), units=np.array(units), fnames=np.array(list(feature_names(spec))))
print("saved", out, flush=True)
