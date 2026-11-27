"""CTRL-82 VAL: the plant constants the scorer flew each selected VAL unit with (motor asym, arm, k_drag, inertia, lag,
torque-jitter) — read from the captured production scorer call; plus the per-episode torque scale is NOT captured
(applies to the whole torque, so a ratio of torques is unaffected)."""
import json, os, sys
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T
tag, out = sys.argv[1], sys.argv[2]
import wnn.control._accel as A
import wnn.control.phased_ga as PG
from wnn.control.evaluator import ControllerEvaluator
from wnn.seeds import resolve_seed_set
txt, a, ec = T.run_args(tag)
seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
pay = PG._ctl_load(T.D + tag + "_winner.yaml.gz"); spec, genome = pay["spec"], pay["best_genome"]
units = sorted(set(np.load("/tmp/ctrl82/sel_all.npy")[:, 0].tolist()))
cap = {}; real = A.score_controllers_cpu
def wrap(controllers, *aa, **kk):
	cap.setdefault("calls", []).append((controllers[0], aa, kk)); return real(controllers, *aa, **kk)
A.score_controllers_cpu = wrap
thr = PG._test_thresholds(a, ec, spec, units[0], seeds.train, use_score=True)
P = {}
for unit in units:
	ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
	                         rg_config=PG._rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds, max_eval_workers_gpu=False)
	cap["calls"] = []; ev.score_genomes([genome]); k = cap["calls"][0][2]
	P[unit] = {n: (list(k[n]) if isinstance(k[n], (list, tuple)) else k[n]) for n in
	           ("dist_motor_asym", "arm_length", "k_drag", "inertia", "motor_lag_s", "dist_torque_scale_jitter", "k_thrust", "dist_tau_bias") if n in k}
json.dump(P, open(out, "w"), indent=0); print("saved", out, P[units[0]])
