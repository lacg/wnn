"""docs/ctrl_dres_failures.md ideas #3 and #4 (controller ABI 48), default OFF.

  #3 --select-tail {off,near_fail} / --select-tail-deg X  (stage-select only)
  #4 --train-xy-offset M / --train-init-vxy V             (DAgger trainer only)

Sections:
  A  unit: tail ranking (average ranks, rank-sum winner, tie invariance), tail_count
     on per-episode rows, flag validation, provenance tokens (no wheel needed).
  B  pin: flags OFF == the pre-change tree on the ABI-47 wheel — thresholds, features,
     training fingerprint and scorer rows (the controller_ctrl48_abi44_pin pattern),
     plus the stage-select tail flag leaving the same four untouched.
  C  flags ON change only their own place:
       #4 training cells move; thresholds, the scorer's episode draws and the scorer's
          rows for a fixed genome do not; the calibration/recorder config is untouched;
       #3 the per-episode sink is armed on the stage-select VAL calls only, its rows
          reproduce the aggregate the scorer returns, and the headline obeys the rank
          sum while the fitness values are unchanged.

Run (ABI-48 wheel first on the path; base = the pre-change tree + the ABI-47 wheel):
  PYTHONPATH=/tmp/ctrl_dres_site:src/wnn:src \\
  DRES_BASE_SRC=/tmp/ctrl_dres_base/src DRES_BASE_SITE=/tmp/ctrl_fix_site \\
    python tests/controller_dres_ideas_flags.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
BASE_SRC = os.environ.get("DRES_BASE_SRC", "/tmp/ctrl_dres_base/src")
BASE_SITE = os.environ.get("DRES_BASE_SITE", "/tmp/ctrl_fix_site")
sys.path.insert(0, HERE)
from controller_ctrl48_abi44_pin import RECIPE, _features, _grid, _metrics  # noqa: E402

TRAIN_ON = ["--train-xy-offset", "0.8", "--train-init-vxy", "0.4"]
TAIL_ON = ["--select-tail", "near_fail", "--select-tail-deg", "3.0"]


# ---------------------------------------------------------------- A: unit
def test_average_ranks_and_rank_sum():
	from wnn.control.select_tail import average_ranks, tail_rank_winner
	assert average_ranks([0, 0, 1, 0, 3]) == [2.0, 2.0, 4.0, 2.0, 5.0]
	assert average_ranks([0.3, 0.1, 0.2]) == [3.0, 1.0, 2.0]
	labels = ["M#0", "M#1", "N#0", "N#1", "C#0"]
	fits = {"M#0": 1.0, "M#1": 1.1, "N#0": 1.2, "N#1": 5.0, "C#0": 9.0}
	# Every tail equal => exactly today's winner (rank(tail) is a constant).
	for t in (0, 3):
		assert tail_rank_winner(labels, fits, {l: t for l in labels})[0] == "M#0"
	# One near-miss on the fit leader among near-tied clean candidates: the clean one wins.
	tails = {"M#0": 1, "M#1": 0, "N#0": 0, "N#1": 0, "C#0": 0}
	win, comb = tail_rank_winner(labels, fits, tails)
	assert win == "M#1" and comb["M#0"] == 1 + 5 and comb["M#1"] == 2 + 2.5, comb
	# A clearly better fit survives one near-miss when the clean field is far behind.
	tails2 = {"M#0": 1, "M#1": 2, "N#0": 2, "N#1": 0, "C#0": 0}
	assert tail_rank_winner(labels, fits, tails2)[0] == "M#0"
	# Ties on the rank sum go to the lower fit.
	assert tail_rank_winner(["a", "b"], {"a": 2.0, "b": 1.0}, {"a": 0, "b": 1})[0] == "b"
	print("✓ A1 average ranks + rank-sum winner (tie invariance, demotion, survival, tie-break)")


def test_tail_count_rows():
	from wnn.control.select_tail import tail_count
	r = math.radians
	# [gi, ep, stable, diverged, mean_err_rad, ...]
	rows = [[0, 0, 1, 0, r(1.0)], [0, 1, 1, 0, r(4.5)], [0, 2, 0, 0, r(6.0)],
	        [0, 3, 0, 1, r(0.5)], [0, 4, 0, 0, float("nan")], [1, 0, 0, 1, r(9.0)]]
	assert tail_count([{"episodes": rows}], 4.0, "t") == 4      # 4.5, 6.0, diverged, nan; genome 1 ignored
	assert tail_count([{"episodes": rows[:2]}, {"episodes": rows[2:3]}], 5.0, "t") == 1
	try:
		tail_count([], 4.0, "t")
	except RuntimeError as e:
		assert "GPU-scorer only" in str(e)
	else:
		raise AssertionError("an empty sink must refuse, not read as a clean candidate")
	print("✓ A2 tail_count: diverged / > X / non-finite counted, genome 0 only, empty sink refused")


def _refuses(fn, *a) -> bool:
	try:
		fn(*a)
	except SystemExit:
		return True
	return False


def test_validation_and_tokens():
	from wnn.control.select_tail import select_tail_token, validate_select_tail
	from wnn.control.train_ics import train_ic_token, validate_train_ics
	ns = SimpleNamespace
	for deg in (0.0, -1.0, 5.0, 7.0, float("nan")):
		assert _refuses(validate_select_tail, ns(select_tail="near_fail", select_tail_deg=deg)), deg
	assert _refuses(validate_select_tail, ns(select_tail="p99", select_tail_deg=4.0))
	assert not _refuses(validate_select_tail, ns(select_tail="near_fail", select_tail_deg=4.0))
	assert select_tail_token(ns(select_tail="off", select_tail_deg=4.0)) == "off"
	assert select_tail_token(ns(select_tail="near_fail", select_tail_deg=3.5)) == "near_fail:3.5"
	h = ns(translation=True, max_initial_xy_offset_m=0.5)
	flat = ns(translation=True, max_initial_xy_offset_m=0.0)
	for xy, v, ec in ((0.0, 0.0, h), (-0.2, 0.0, h), (None, -0.1, h), (None, float("inf"), h),
	                  (0.8, 0.0, flat), (None, 0.3, flat)):
		assert _refuses(validate_train_ics, ns(train_xy_offset=xy, train_init_vxy=v), ec), (xy, v)
	assert not _refuses(validate_train_ics, ns(train_xy_offset=None, train_init_vxy=0.0), flat)
	assert not _refuses(validate_train_ics, ns(train_xy_offset=0.8, train_init_vxy=0.4), h)
	assert train_ic_token(ns(train_xy_offset=None, train_init_vxy=0.0)) == "off"
	assert train_ic_token(ns(train_xy_offset=0.8, train_init_vxy=0.4)) == "xy:0.8,vxy:0.4"
	assert train_ic_token(ns(train_xy_offset=None, train_init_vxy=0.3)) == "xy:scorer,vxy:0.3"
	print("✓ A3 validation refuses bad bounds / missing horizontal channel; tokens")


# ---------------------------------------------------------------- B: OFF pin
def dump_pin(extra: list) -> dict:
	gs = _grid(RECIPE + extra)
	g = gs._random_genome(gs._enumerate_points()[0])
	feats = _features(g, gs)
	gs._evaluator._evaluate_core([g], write_back=True, return_stats=True)
	digest = list(g.cells.fingerprint()) if g.cells is not None else None
	rows = [_metrics(m) for m in gs._evaluator.score_genomes([g])]
	return {"thresholds": [float(t) for t in gs.thresholds], "features": feats,
	        "train_digest": digest, "scorer_rows": rows}


def _run(mode: str, extra: list, pythonpath: str) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([PY, os.path.abspath(__file__), "--dump", mode, json.dumps(extra)], env=env,
	                   capture_output=True, text=True, cwd=HERE)
	if r.returncode != 0:
		raise RuntimeError(f"{mode} {extra} @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def test_flags_off_pin():
	base = _run("pin", [], f"{BASE_SITE}:{BASE_SRC}/wnn:{BASE_SRC}")
	for extra in ([], TAIL_ON):
		new = _run("pin", extra, os.environ["PYTHONPATH"])
		for key in base:
			assert base[key] == new[key], f"{extra}: {key} moved vs the pre-change tree (ABI 47)"
	print(f"✓ B  flags off (and --select-tail on) == pre-change tree on ABI 47: thresholds "
	      f"({len(base['thresholds'])}), features, train digest {base['train_digest']}, scorer rows")


# ---------------------------------------------------------------- C: ON scope
def dump_scope() -> dict:
	"""#4 in one process: the OFF and ON evaluators train the same genome; the ON
	trainer's cells differ, while thresholds, scorer draws and scorer rows of the
	OFF-trained genome are identical under both evaluators."""
	import copy
	from wnn.control.evaluator import reference_packed_config
	from wnn.control.training import sample_horizontal_ics_flat
	off, on = _grid(RECIPE), _grid(RECIPE + TRAIN_ON)
	g = off._random_genome(off._enumerate_points()[0])
	g_on = copy.deepcopy(g)
	off._evaluator._evaluate_core([g], write_back=True, return_stats=True)
	on._evaluator._evaluate_core([g_on], write_back=True, return_stats=True)
	rows = [[_metrics(m) for m in gs._evaluator.score_genomes([g])] for gs in (off, on)]
	draws = [list(map(list, sample_horizontal_ics_flat(99990101, 50, gs._ec))) for gs in (off, on)]
	ref = [reference_packed_config(gs._ec) for gs in (off, on)]
	rg = on._evaluator.rg_config
	return {"digest_off": list(g.cells.fingerprint()), "digest_on": list(g_on.cells.fingerprint()),
	        "thresholds_equal": list(off.thresholds) == list(on.thresholds),
	        "rows_equal": rows[0] == rows[1], "draws_equal": draws[0] == draws[1],
	        "ref_xy": [c.xy_offset for c in ref], "ref_vxy": [c.init_vxy for c in ref],
	        "rg": [rg.train_xy_offset, rg.train_init_vxy]}


def test_train_flags_scope():
	d = _run("scope", [], os.environ["PYTHONPATH"])
	assert d["digest_off"] != d["digest_on"], "the trainer flags did not reach the DAgger rollouts"
	assert d["thresholds_equal"], "thresholds moved — calibration must keep --xy-offset at rest"
	assert d["draws_equal"] and d["rows_equal"], "the scorer's episodes moved"
	assert d["ref_xy"] == [0.5, 0.5] and d["ref_vxy"] == [0.0, 0.0], d
	assert d["rg"] == [0.8, 0.4], d["rg"]
	print(f"✓ C1 #4: trainer cells move ({d['digest_off']} -> {d['digest_on']}); thresholds, scorer "
	      f"draws, scorer rows and the calibration/recorder config (xy 0.5, vxy 0) unchanged")


def test_each_train_flag_alone():
	"""Each #4 flag on its own reaches the Rust trainer (vxy needs the ABI-48 kwarg) and
	leaves the thresholds and the feature encoding alone."""
	pp = os.environ["PYTHONPATH"]
	base = _run("pin", [], pp)
	for extra in (["--train-init-vxy", "0.4"], ["--train-xy-offset", "0.8"]):
		d = _run("pin", extra, pp)
		assert d["train_digest"] != base["train_digest"], f"{extra} did not reach the trainer"
		assert d["thresholds"] == base["thresholds"] and d["features"] == base["features"], extra
	print("✓ C1b #4: --train-init-vxy alone and --train-xy-offset alone each move the trainer cells; "
	      "thresholds + features unchanged")


def dump_sink() -> dict:
	"""#3 on the GPU: the sink-armed _holdout_report returns the same metrics as the
	unarmed call, and its per-episode rows reproduce the aggregate."""
	from wnn.control import phased_ga as pg
	gs = _grid(RECIPE + TAIL_ON + ["--report-episodes", "40"])
	g = gs._random_genome(gs._enumerate_points()[0])
	args, ec = gs._args, gs._ec
	sink: list = []
	m0 = pg._holdout_report(args, ec, gs._probe_spec, g, None, 99990201, 31337002, "VAL-off")
	m1 = pg._holdout_report(args, ec, gs._probe_spec, g, None, 99990201, 31337002, "VAL-on",
	                        episode_sink=sink)
	eps = [r for e in sink for r in e["episodes"]]
	return {"same": _metrics(m0) == _metrics(m1), "n": len(eps),
	        "stable": sum(r[2] for r in eps) / max(len(eps), 1), "agg_stable": m1.stable_rate,
	        "err": math.degrees(sum(r[4] for r in eps) / max(len(eps), 1)),
	        "agg_err": m1.mean_attitude_error_deg}


def test_sink_on_gpu():
	d = _run("sink", [], os.environ["PYTHONPATH"])
	assert d["same"], "arming the sink moved the scorer's metrics"
	assert d["n"] == 40, f"expected 40 per-episode rows, got {d['n']}"
	assert abs(d["stable"] - d["agg_stable"]) < 1e-9 and abs(d["err"] - d["agg_err"]) < 1e-6, d
	print(f"✓ C2 #3: sink rows ({d['n']}) reproduce the aggregate (stable {d['agg_stable']:.3f}, "
	      f"err {d['agg_err']:.3f}°); metrics identical with and without the sink")


def test_stage_select_rank_sum():
	"""_select_headline_stage on fake candidates with a stub _holdout_report: off arms no
	sink and picks the fitness winner; near_fail sums ranks; fitness values unchanged."""
	from wnn.control import phased_ga as pg
	means = {"MEMORY": (0.99, 1.00), "NEURONS": (0.99, 1.01), "GRID": (0.95, 3.0)}
	tails = {"MEMORY": 2, "NEURONS": 0, "GRID": 0}
	seen = []

	def stub(args, ec, spec, genome, fp, vs, ts, stage_label="final", episode_sink=None):
		seen.append(episode_sink is not None)
		st, err = means[genome]
		if episode_sink is not None and vs == 7000:
			bad = [[0, i, 0, 0, math.radians(4.5)] for i in range(tails[genome])]
			episode_sink.append({"episodes": bad + [[0, 99, 1, 0, math.radians(1.0)]]})
		elif episode_sink is not None:
			episode_sink.append({"episodes": [[0, 0, 1, 0, math.radians(1.0)]]})
		return SimpleNamespace(reward=-err, stable_rate=st, acc=st, mean_attitude_error_deg=err,
		                       mean_steady_error_deg=None, motor_jerk_mean=None,
		                       mono_violations_total=None, mean_effort=None,
		                       mean_altitude_error_m=None, mean_position_error_m=None,
		                       mean_dx_m=None, mean_dy_m=None)

	real = pg._holdout_report
	pg._holdout_report = stub
	try:
		heads = [_select(pg, mode) for mode in ("off", "near_fail")]
	finally:
		pg._holdout_report = real
	n_calls = len(seen) // 2
	assert not any(seen[:n_calls]) and all(seen[n_calls:]), "sink armed outside near_fail VAL calls"
	assert heads[0].key == "MEMORY" and heads[1].key == "NEURONS", [h.key for h in heads]
	print("✓ C3 #3: off arms no sink (headline MEMORY = best fit); near_fail arms it on every VAL "
	      "call and the rank sum moves the headline to the clean NEURONS")


def _select(pg, mode: str):
	args = pg.build_arg_parser().parse_args(
		["--select-tail", mode, "--fit-weight-err-sq", "1", "--fit-weight-stable", "1",
		 "--stage-select-top-k", "1"])
	res = lambda name: SimpleNamespace(final_population=[name], best_genome=name)
	entries = [(n, None, res(n)) for n in ("GRID", "NEURONS", "MEMORY")]
	holdouts = {n: SimpleNamespace(acc=0.99, mean_attitude_error_deg=1.0) for n in ("GRID", "NEURONS", "MEMORY")}
	pg._heldout_row_str = lambda _ho: ""
	return pg._select_headline_stage(args, None, SimpleNamespace(val=7000, train=1), entries, holdouts)


if __name__ == "__main__":
	if len(sys.argv) >= 3 and sys.argv[1] == "--dump":
		import contextlib, io
		fn = {"pin": lambda: dump_pin(json.loads(sys.argv[3])), "scope": dump_scope, "sink": dump_sink}
		with contextlib.redirect_stdout(io.StringIO()):
			d = fn[sys.argv[2]]()
		print(json.dumps(d))
	else:
		test_average_ranks_and_rank_sum()
		test_tail_count_rows()
		test_validation_and_tokens()
		test_stage_select_rank_sum()
		test_flags_off_pin()
		test_train_flags_scope()
		test_each_train_flag_alone()
		test_sink_on_gpu()
		print("\nideas #3/#4 flag tests passed.")
