"""CTRL-73: does the rival (MPCOF>rate, row C, oracle-fed) change under the G+T argv? Re-score 5 TEST units with the
G+T s1 run's args through classical_baseline.teacher_metrics(axes=True) and compare with the CTRL-70 rival.npz (N0 args)."""
import sys, numpy as np
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import ctrl_gap_student_trace as T
from wnn.control import phased_ga as P
from wnn.control import classical_baseline as CB
tag = sys.argv[1]
txt, a, ec = T.run_args(tag); P._resolve_ctbr(a, ec)
R = np.load("/tmp/ctrl73/rival.npz")["C"]
for i, unit in enumerate([int(u) for u in a.test_seeds][:5]):
	draw = CB.HoldoutDraw(seed=unit, episodes=a.test_episodes, steps=ec.steps_per_episode, eval_folds=a.num_eval_folds)
	r = CB.teacher_metrics(ec, draw, CB.TeacherFeed(use_estimator=False), 4, int(a.action_repeat), "MPCOF>rate", axes=True)
	print(unit, f"G+T-args err {r['mean_attitude_error_deg']:.6f} steady {r['mean_steady_error_deg']:.6f} | CTRL-70 (N0 args) err {R[i,2]:.6f} steady {R[i,3]:.6f}", flush=True)
