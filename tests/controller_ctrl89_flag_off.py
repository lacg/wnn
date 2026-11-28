"""CTRL-89 (ABI 70): plant INERTIA scaling + CONTROL LATENCY — the cf21 FLAG-OFF pin vs the
PREVIOUS build (ABI 68 when written; ABI 69 since the r69 rebase) on the flown l2bnm recipe (the CTRL-86 harness: G+T / l2b argv + --anchor-mass nominal, the same
shrink, the same dump), plus the stress-plant and training-flag plumbing through the wheel:
  previous: $CTRL89_BASE_PYTHONPATH (the previous src tree + its wheel first, or the venv's);
            $CTRL89_BASE_ABI pins which ABI that must be (optional; else any ABI below this tree's);
  this:     this tree with its own (built, not installed) wheel first on PYTHONPATH — its ABI is
            this tree's _accel.EXPECTED_ABI, never a number hard-coded here.

  1. flag off == the previous build: feature names, fitted thresholds, calibration token, resolved ctbr, every
     parsed argument (minus the two new keys, which must resolve to None), provenance, 40 steps of
     features, the Lamarckian write-back digest after one short DAgger and the GA-search scorer rows
     (default device: Metal unless WNN_CONTROLLER_GPU_EVAL=0).
  2. the zeros spelled out (--train-inertia-jitter 0 --train-latency-range 0 0, CPU scorer): every
     behaviour key equal to flag off; provenance records the spelling and nothing else.
  3. the training spread (--train-inertia-jitter 0.30 --train-latency-range 2 12) moves the
     calibration, the trainer digest and the search row; CPU and Metal search rows agree.
  4. the STRESS plant (EpisodeConfig.inertia_jitter / control_latency_ms, what a stress cell sets):
     reaches the WNN report scorer AND the rival (classical_baseline.teacher_metrics) with the SAME
     kwargs (the score_cell.py plant-identity gate's view), moves both rows, and never reaches the
     trainer (RewardGatedConfigPacked keeps train_* at off).

Run: CTRL89_BASE_PYTHONPATH=/tmp/ctrl87_site:/tmp/ctrl89_base/src \\
     PYTHONPATH=/tmp/ctrl89_site:src python tests/controller_ctrl89_flag_off.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controller_ctrl86_flag_off as H  # noqa: E402  the CTRL-86 harness (argv, shrink, dump)

L2BNM = ["--anchor-mass", "nominal"]
VARIANTS = {"off": L2BNM, "zeros": L2BNM + ["--train-inertia-jitter", "0", "--train-latency-range", "0", "0"],
            "spread": L2BNM + ["--train-inertia-jitter", "0.30", "--train-latency-range", "2", "12"]}
NEW_KEYS = ("anchor_mass", "train_motor_asym_mag", "train_motor_lag_range", "train_inertia_jitter", "train_latency_range",
            "train_mass_jitter", "train_noise_scale", "train_motor_asym_range")  # + the ABI-70 training ranges
STRESS = {"inertia_jitter": 0.20, "control_latency_ms": (8.0, 12.0)}  # the proposed golden-cell values


def dump(variant: str) -> dict:
	H.VARIANTS = {variant: VARIANTS[variant]}
	H.NEW_KEYS = H.NEW_KEYS + NEW_KEYS
	d = H.dump(variant)
	args, ec = H._args(VARIANTS[variant])
	d["_new"] = {k: repr(getattr(args, k, None)) for k in ("train_inertia_jitter", "train_latency_range")}
	d["_jit"] = repr((getattr(ec, "train_inertia_jitter", None), getattr(ec, "train_latency_ms", None)))
	try:
		from wnn.control.evaluator import train_jitter_token
		if train_jitter_token(ec):
			d["prov"] = {**d["prov"], "train_jitter": train_jitter_token(ec)}
	except ImportError:  # a pre-70 tree has no train_jitter_token
		pass
	return d


def _dump_in(pythonpath: str, variant: str, cpu: bool = False) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	if cpu:
		env["WNN_CONTROLLER_GPU_EVAL"] = "0"
	r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", variant], env=env,
	                   capture_output=True, text=True, cwd="/tmp")
	if r.returncode != 0:
		raise RuntimeError(f"{variant} @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def previous_build_pair(old: dict, new: dict, env_key: str) -> tuple:
	"""Flag-off identity is measured against the PREVIOUS build, not a hard-coded ABI: the base
	dump must come from an older ABI than this tree's own EXPECTED_ABI (each tree's _accel
	asserts its wheel, so a stale wheel fails inside the dump), from a different source tree,
	and — when $<env_key> is set — from exactly that ABI."""
	o, n = int(old["_abi"]), int(new["_abi"])
	assert o < n, f"base ABI {o} is not older than this tree's ABI {n}"
	if os.environ.get(env_key):
		assert o == int(os.environ[env_key]), f"base ABI {o} != ${env_key}={os.environ[env_key]}"
	assert old["_src"] != new["_src"], "both dumps imported the same source tree"
	return o, n


def test_flag_off_is_previous_build() -> dict:
	base_pp = os.environ.get("CTRL89_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL89_BASE_PYTHONPATH to the previous build's src tree (+ wheel)")
	new_pp = os.environ.get("PYTHONPATH", H._here())
	old, new = _dump_in(base_pp, "off"), _dump_in(new_pp, "off")
	o, n = previous_build_pair(old, new, "CTRL89_BASE_ABI")
	for key in old:
		if not key.startswith("_"):
			assert old[key] == new[key], f"off: {key} moved between ABI {o} and ABI {n}"
	assert new["_jit"] == "(None, None)", new["_jit"]
	assert new["_new"] == {"train_inertia_jitter": "None", "train_latency_range": "None"}, new["_new"]
	print(f"✓ flag off (l2bnm) == ABI {o} (this tree ABI {n}): names ({len(new['names'])}), thresholds ({len(new['thresholds'])}), token, "
	      f"ctbr, args ({len(new['args'])}), provenance, features ({len(new['features'])} steps), train digest "
	      f"{new['train_digest']}, scorer rows {new['scorer_rows'][0]['mean_attitude_error_deg']}; both new flags None")
	return new


def test_zeros_spelled_out():
	pp = os.environ.get("PYTHONPATH", H._here())
	off, spelled = _dump_in(pp, "off", cpu=True), _dump_in(pp, "zeros", cpu=True)
	for key in H.BEHAVIOUR:
		if key != "prov":
			assert spelled[key] == off[key], f"the zeros spelled out moved {key}"
	assert spelled["prov"] == {**off["prov"], "train_jitter": "inertia:0+latency:0,0ms"}, spelled["prov"]
	print("✓ --train-inertia-jitter 0 --train-latency-range 0 0 (CPU scorer): every behaviour key equal to flag "
	      "off; provenance adds train_jitter=inertia:0+latency:0,0ms only")


def test_spread_reaches_every_host(off: dict):
	pp = os.environ.get("PYTHONPATH", H._here())
	gpu, cpu = _dump_in(pp, "spread"), _dump_in(pp, "spread", cpu=True)
	for key in ("thresholds", "train_digest", "scorer_rows"):
		assert gpu[key] != off[key], f"the jitter did not move {key}"
	assert gpu["thresholds"] == cpu["thresholds"] and gpu["train_digest"] == cpu["train_digest"]
	for k in ("mean_attitude_error_deg", "stable_rate", "mean_altitude_error_m"):
		g, c = float(gpu["scorer_rows"][0][k]), float(cpu["scorer_rows"][0][k])
		assert abs(g - c) <= max(1e-3, 2e-2 * abs(c)), f"search row {k}: gpu {g} cpu {c}"
	assert gpu["prov"]["train_jitter"] == "inertia:0.3+latency:2,12ms", gpu["prov"]
	print(f"✓ spread (inertia ±0.30, latency 2-12 ms): thresholds, train digest and search row move; search err "
	      f"gpu {float(gpu['scorer_rows'][0]['mean_attitude_error_deg']):.4f} cpu "
	      f"{float(cpu['scorer_rows'][0]['mean_attitude_error_deg']):.4f} (off {float(off['scorer_rows'][0]['mean_attitude_error_deg']):.4f})")


def test_stress_plant_reaches_wnn_and_rival_identically():
	"""What a stress cell does: dataclasses.replace(ec, **STRESS) on the report config."""
	import dataclasses
	import wnn.control._accel as A
	from wnn.control import classical_baseline as CB
	from wnn.control.evaluator import _train_asym_kwargs
	os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
	args, ec0 = H._args(VARIANTS["off"])
	ec = dataclasses.replace(ec0, **STRESS)
	assert ec.stress_kwargs() == {**ec0.stress_kwargs(), **STRESS}, ec.stress_kwargs()
	assert ec0.stress_kwargs() == {} or "inertia_jitter" not in ec0.stress_kwargs()
	tr = _train_asym_kwargs(ec)
	assert "train_inertia_jitter" not in tr and "train_latency_ms" not in tr, "the stress plant reached the trainer"

	calls = {}

	def capture(name):
		real = getattr(A, name)

		def f(*aa, **kk):
			calls[name] = kk
			return real(*aa, **kk)
		setattr(A, name, f)
		return real

	real = capture("score_classical_baseline")
	try:
		draw = CB.HoldoutDraw(seed=99990101, episodes=4, steps=ec.steps_per_episode, eval_folds=1)
		rows = [CB.teacher_metrics(e, draw, CB.TeacherFeed(use_estimator=True), 4, int(args.action_repeat), "mpcof")
		        for e in (ec0, ec)]
	finally:
		A.score_classical_baseline = real
	rk = calls["score_classical_baseline"]
	assert rk["inertia_jitter"] == STRESS["inertia_jitter"] and tuple(rk["control_latency_ms"]) == STRESS["control_latency_ms"]
	assert rows[0]["mean_attitude_error_deg"] != rows[1]["mean_attitude_error_deg"], "rival row did not move"

	from wnn.control.controller_grid_search import ControllerGridSearch
	from wnn.control.evaluator import ControllerEvaluator
	gs = ControllerGridSearch(args, ec0, 31337002)
	g = gs._random_genome(gs._enumerate_points()[0])
	gs._evaluator._evaluate_core([g], write_back=True, return_stats=True)
	real = capture("score_controllers_cpu")
	try:
		wrows = []
		for e in (ec0, ec):
			ev = ControllerEvaluator(gs._probe_spec, num_eval_episodes=4, seed=99990101, episode_config=e,
			                         thresholds=gs.thresholds, max_train_workers=1, max_eval_workers_gpu=False)
			wrows.append(ev.score_genomes([g])[0])
	finally:
		A.score_controllers_cpu = real
	wk = calls["score_controllers_cpu"]
	for k in STRESS:
		assert tuple(wk[k]) == tuple(rk[k]) if isinstance(rk[k], tuple) else wk[k] == rk[k], f"{k}: WNN {wk[k]} vs rival {rk[k]}"
	assert wrows[0].mean_attitude_error_deg != wrows[1].mean_attitude_error_deg, "WNN report row did not move"
	print(f"✓ stress plant {STRESS}: the WNN report scorer and the mpcof rival receive the SAME kwargs; rival err "
	      f"{rows[0]['mean_attitude_error_deg']:.3f} -> {rows[1]['mean_attitude_error_deg']:.3f} deg, WNN err "
	      f"{wrows[0].mean_attitude_error_deg:.3f} -> {wrows[1].mean_attitude_error_deg:.3f} deg; trainer untouched")


def test_registry_provenance_wheel():
	from wnn.ram.experiments.params import KNOWN_PARAMS
	from wnn.control.flow_adapter import controller_spec_from_params
	from wnn.control.provenance import collect_provenance
	import wnn.control._accel as ra
	keys = ("controller_train_inertia_jitter", "controller_train_latency_range")
	assert set(keys) <= KNOWN_PARAMS
	for key in keys:
		try:
			controller_spec_from_params({key: 0.3})
		except ValueError as e:
			assert "not built" in str(e), e
		else:
			raise AssertionError(f"{key}: the flow adapter did not refuse it")
	assert "train_jitter=inertia:0.3 " in collect_provenance("x", train_jitter="inertia:0.3").line()
	assert "train_jitter" not in collect_provenance("x").line()
	# The plumbing landed in ABI 70; this tree's wheel must be the ABI its _accel expects.
	assert int(ra.require_accel().ABI_VERSION) == int(ra.EXPECTED_ABI) >= 70, (ra.EXPECTED_ABI,)
	base = dict(translation=True, af_mass=0.0393, af_k_thrust=0.2, motor_lag_s=0.2, dist_enabled=True)
	cfg = ra.RewardGatedConfigPacked(**base, train_inertia_jitter=0.3, train_latency_ms=(2.0, 12.0))
	assert round(cfg.train_inertia_jitter, 6) == 0.3 and tuple(cfg.train_latency_ms) == (2.0, 12.0)
	off = ra.RewardGatedConfigPacked()
	assert (off.train_inertia_jitter, off.train_latency_ms) == (0.0, None)
	for bad, why in (({**base, "dist_enabled": False}, "no weather"), ({**base, "translation": False}, "no translation")):
		try:
			ra.RewardGatedConfigPacked(**bad, train_inertia_jitter=0.3)
		except ValueError:
			pass
		else:
			raise AssertionError(why)
	print(f"✓ KNOWN_PARAMS + flow-adapter refusal, provenance train_jitter= (absent off), wheel ABI {ra.EXPECTED_ABI} "
	      "RewardGatedConfigPacked(train_inertia_jitter=, train_latency_ms=) accepted / refused")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--dump":
		import contextlib, io
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2])
		print(json.dumps(d))
		sys.exit(0)
	test_registry_provenance_wheel()
	off = test_flag_off_is_previous_build()
	test_zeros_spelled_out()
	test_spread_reaches_every_host(off)
	test_stress_plant_reaches_wnn_and_rival_identically()
	print("\nCTRL-89: cf21 flag-off pin vs the previous build + plant-jitter plumbing passed.")
