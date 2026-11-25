"""VAL-unit per-tick trace (diagnosis only, never reported as a result): ctrl_gap_student_trace.replay on the first
N val units of a run's HEADLINE genome (winner file), gated on the scorer's steady."""
import math, os, re, sys
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T
tag, out, nunits = sys.argv[1], sys.argv[2], int(sys.argv[3])
import wnn.control._accel as A
import ram_controller as RC
from wnn.control.phased_ga import _test_thresholds, _rg_config, _ctl_load
from wnn.control.evaluator import ControllerEvaluator
from wnn.seeds import resolve_seed_set
txt, a, ec = T.run_args(tag)
seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
pay = _ctl_load(T.D + tag + "_winner.yaml.gz"); spec, genome = pay["spec"], pay["best_genome"]
units = T.val_units(tag)[:nunits]
cap = {}; real = A.score_controllers_cpu
def wrap(controllers, *aa, **kk):
	cap.setdefault("calls", []).append((controllers[0], aa, kk)); return real(controllers, *aa, **kk)
A.score_controllers_cpu = wrap
thr = _test_thresholds(a, ec, spec, units[0], seeds.train, use_score=True)
Tr, E = [], []
for ui, unit in enumerate(units):
	ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
	                         rg_config=_rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds, max_eval_workers_gpu=False)
	cap["calls"] = []
	m = ev.score_genomes([genome])[0]
	c, aa, k = cap["calls"][0]; k = {**{"dt": 0.001, "dist_seed": 0}, **k}
	rows, eps, tr = T.replay(RC, c, aa[0], aa[1], aa[2], aa[3], k, ui, aa[2])
	rs = math.degrees(float(np.nanmean(eps[:, 2])))
	print(f"unit {unit}: scorer err {m.mean_attitude_error_deg:.3f} steady {m.mean_steady_error_deg:.4f} | replay steady {rs:.4f} err {math.degrees(np.mean(eps[:,3])):.3f}", flush=True)
	assert abs(rs - m.mean_steady_error_deg) < 0.005
	Tr.append(tr); E.append(eps)
np.savez(out, trace=np.vstack(Tr), eps=np.vstack(E), units=np.array(units))
