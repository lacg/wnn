"""Parity probe for ABI 53: the MPCOF>rate(h<ar>) est + oracle rows (and PID[est]) of one run on
one unit, exact floats as JSON. Run once on the ABI-52 wheel (HEAD python tree) and once on
ABI 53 (axis_out off AND on) — all three must be bit-identical."""
import json, sys, shlex
log, unit, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
axes = len(sys.argv) > 4 and sys.argv[4] == "axes"
from wnn.control import phased_ga as P
from wnn.control import classical_baseline as CB
txt = open(log, errors="replace").read()
argv = shlex.split([l for l in txt.splitlines() if l.startswith("[argv]")][0][len("[argv]"):])
a = P.build_arg_parser().parse_args(argv); ec = P.episode_config_from_args(a); P._resolve_ctbr(a, ec)
draw = CB.HoldoutDraw(seed=unit, episodes=a.test_episodes, steps=ec.steps_per_episode, eval_folds=a.num_eval_folds)
rows = {}
for feed in (True, False):
	kw = {"axes": True} if axes else {}
	r = CB.teacher_metrics(ec, draw, CB.TeacherFeed(use_estimator=feed), 4, int(a.action_repeat), "MPCOF>rate", **kw)
	rows[r["label"]] = {k: v for k, v in r.items() if k != "axes"}
	if axes: rows[r["label"] + ":axes"] = r["axes"]
r = CB.pid_metrics(ec, draw, CB.TeacherFeed(use_estimator=True)); rows[r["label"]] = r
json.dump(rows, open(out, "w"), indent=1, default=repr)
for k, v in rows.items():
	if ":axes" not in k: print(k, f"stable={v['stable_rate']} err={v['mean_attitude_error_deg']!r} steady={v['mean_steady_error_deg']!r}")
