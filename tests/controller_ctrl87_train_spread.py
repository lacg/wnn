"""CTRL-87 (ABI 68): the TRAINING spread, Python side (prereg docs/ctrl87_nominal_mass_wide_spread_prereg.md,
Amendment 1).

  * CLI: --train-motor-asym-mag / --train-motor-lag-range default None (off); the resolved
    EpisodeConfig carries them; refusals (lag range without a lag / translation / weather,
    lo > hi, extended actuator, mag outside [0, 1));
  * TRAINING hosts: _train_asym_kwargs (the one source for DAgger, per-round eval, calibration
    sampler, MEMORY recorder) carries the training wear and the lag range — byte-identical off;
  * POOLS: with both flags ON, a REPORT evaluator (train_spread=False: TEST units, VAL
    stage-select, held-out rows) hands the scorer EXACTLY the kwargs of a flag-off run (preset
    ±0.20 wear draw, no lag range); a SEARCH evaluator (train_spread=True) hands it the ±0.30
    draw of the same stream and the lag range. phased_ga builds the three SEARCH evaluators
    (grid, arch stages, MEMORY) with train_spread=True and every report evaluator without;
  * provenance train_asym / train_lag (absent off), KNOWN_PARAMS, the flow adapter refuses;
  * the ABI-68 wheel accepts train_motor_lag_range on RewardGatedConfigPacked and the
    scorers, refuses a bad host, and the search scorer's per-episode lags cover the range.

Run: PYTHONPATH=/tmp/ctrl87_site:src python tests/controller_ctrl87_train_spread.py
"""

from __future__ import annotations

import inspect
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controller_ctrl86_flag_off as H  # noqa: E402  the flown l2b-family argv

SPREAD = ["--train-motor-asym-mag", "0.30", "--train-motor-lag-range", "0.15", "0.30"]


def _raises(fn, exc, text: str, what: str):
	try:
		fn()
	except exc as e:
		assert text in str(e), f"{what}: wrong refusal: {e}"
		return
	raise AssertionError(f"{what}: not refused")


def _args(extra: list):
	from wnn.control import phased_ga as pg
	argv = shlex.split(H.GT_ARGV) + extra
	args = pg.build_arg_parser().parse_args(argv)
	return args, pg.episode_config_from_args(args)


def test_cli_and_refusals():
	off_args, off = _args([])
	assert (off_args.train_motor_asym_mag, off_args.train_motor_lag_range) == (None, None)
	assert (off.train_motor_asym_mag, off.train_motor_lag_range) == (None, None) and not off.train_spread_on()
	assert off.search_disturbance() is off.disturbance
	_, on = _args(SPREAD)
	assert on.train_motor_asym_mag == 0.30 and on.train_motor_lag_range == (0.15, 0.30)
	assert on.train_asym_mag() == 0.30 and on.disturbance.motor_asym_mag == 0.2, "the preset must stay ±0.20"
	assert on.search_disturbance().motor_asym_mag == 0.30 and on.motor_lag_s == 0.20
	for extra, text in [(["--train-motor-lag-range", "0.3", "0.2"], "0 < lo <= hi"),
	                    (["--train-motor-lag-range", "0.15", "0.3", "--motor-lag-s", "0"], "motor_lag_s > 0"),
	                    (["--train-motor-lag-range", "0.15", "0.3", "--disturbance", "OFF"], "disturbance"),
	                    (["--train-motor-lag-range", "0.15", "0.3", "--motor-lag-down-s", "0.3"], "symmetric"),
	                    (["--train-motor-asym-mag", "1.0"], "[0, 1)"),
	                    (["--train-motor-asym-mag", "0.3", "--disturbance", "OFF"], "disturbance")]:
		_raises(lambda: _args(extra), SystemExit, text, " ".join(extra))
	print("✓ CLI: both flags default None; resolved onto EpisodeConfig (preset stays ±0.20, lag 0.20); refusals")


def test_training_hosts():
	from wnn.control.evaluator import _train_asym_kwargs, train_asym_token, train_lag_token
	_, off = _args([])
	_, on = _args(SPREAD)
	assert _train_asym_kwargs(off) == {"dist_motor_asym_mag": 0.2}, _train_asym_kwargs(off)
	assert _train_asym_kwargs(on) == {"dist_motor_asym_mag": 0.3, "train_motor_lag_range": (0.15, 0.3)}
	assert (train_asym_token(off), train_lag_token(off)) == ("per_episode:0.2", "")
	assert (train_asym_token(on), train_lag_token(on)) == ("per_episode:0.3", "uniform:0.15,0.3")
	# The three RewardGatedConfigPacked builders all take it from _train_asym_kwargs.
	import wnn.control.evaluator as E
	src = inspect.getsource(E)
	assert src.count("**_train_asym_kwargs(") == 3, src.count("**_train_asym_kwargs(")
	print("✓ training hosts: _train_asym_kwargs carries wear 0.3 + lag (0.15, 0.3) (off: {mag 0.2} only); 3 builders")


class _Capture:
	def __init__(self):
		self.calls = []

	def __call__(self, controllers, q0, omega0, n, steps, **kw):
		self.calls.append(dict(kw, _q0=list(q0), _n=n, _steps=steps))
		return [[0.0] * 20 for _ in controllers]


def _pool_kwargs(ec, train_spread: bool) -> dict:
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control import phased_ga as pg
	args, _ = _args([])
	pg._resolve_ctbr(args, ec)
	spec = pg.ControllerSpec(num_motors=4, levels_per_motor=16)
	ev = ControllerEvaluator(spec, num_eval_episodes=6, seed=99990101, episode_config=ec, num_eval_folds=5,
	                         train_spread=train_spread)
	ev._advance_fold()
	cap = _Capture()
	ev._score_population_rust([object()], cap)
	assert len(cap.calls) == 1, getattr(ev, "_rust_score_error", None)
	return cap.calls[0]


def test_report_pools_unchanged_search_pools_spread():
	_, off = _args([])
	_, on = _args(SPREAD)
	ref = _pool_kwargs(off, train_spread=False)
	assert "train_motor_lag_range" not in ref
	# REPORT (TEST units, VAL stage-select, held-out): EXACTLY the flag-off call.
	rep = _pool_kwargs(on, train_spread=False)
	assert repr(sorted(rep.items())) == repr(sorted(ref.items())), "a report pool moved with the spread on"
	# A SEARCH pool without flags is also the flag-off call (train_spread is inert off).
	assert repr(sorted(_pool_kwargs(off, train_spread=True).items())) == repr(sorted(ref.items()))
	# SEARCH with the spread: the same ICs / weather seed, ±0.30 wear of the same draw, the lag range.
	srch = _pool_kwargs(on, train_spread=True)
	assert srch["train_motor_lag_range"] == (0.15, 0.3)
	assert srch["dist_seed"] == ref["dist_seed"] and srch["_q0"] == ref["_q0"] and srch["ep_mass"] == ref["ep_mass"]
	r_dev = [a - 1.0 for a in ref["dist_motor_asym"]]
	s_dev = [a - 1.0 for a in srch["dist_motor_asym"]]
	assert all(abs(s - 1.5 * r) < 1e-12 for r, s in zip(r_dev, s_dev)), (r_dev, s_dev)
	rest = lambda d: {k: v for k, v in d.items() if k not in ("dist_motor_asym", "train_motor_lag_range")}
	assert repr(sorted(rest(srch).items())) == repr(sorted(rest(ref).items()))
	print(f"✓ pools: report kwargs == flag-off ({len(ref)} kwargs; wear {['%.4f' % a for a in ref['dist_motor_asym']]}); "
	      f"search = same stream at ±0.30 {['%.4f' % a for a in srch['dist_motor_asym']]} + lag range")


def test_phased_ga_wires_search_vs_report():
	from wnn.control import phased_ga as pg
	from wnn.control import controller_grid_search as cgs
	search = [pg._run_arch_phase, pg._run_memory_phase, cgs.ControllerGridSearch.__init__]
	report = [pg._holdout_report, pg._shell_holdout_compact, pg._train_arch_only]
	for f in search:
		assert "train_spread=True" in inspect.getsource(f), f"{f.__qualname__} is a SEARCH evaluator"
	for f in report:
		assert "train_spread" not in inspect.getsource(f), f"{f.__qualname__} is a REPORT evaluator"
	# Every ControllerEvaluator phased_ga / the grid build is one of the above.
	n = inspect.getsource(pg).count("ControllerEvaluator(") + inspect.getsource(cgs).count("ControllerEvaluator(")
	assert n == len(search) + len(report), n
	# VAL stage-select + TEST rows go through _holdout_report (the report evaluator).
	assert "_holdout_report(" in inspect.getsource(pg._maybe_holdout)
	print("✓ phased_ga: grid / arch stages / MEMORY = train_spread=True; held-out (TEST, VAL), shell, arch-only train = report")


def test_provenance_registry_adapter():
	from wnn.control.provenance import collect_provenance
	from wnn.ram.experiments.params import KNOWN_PARAMS
	from wnn.control.flow_adapter import controller_spec_from_params
	assert "train_lag=uniform:0.15,0.3 " in collect_provenance("x", train_lag="uniform:0.15,0.3").line()
	assert "train_lag" not in collect_provenance("x").line()
	keys = ("controller_train_motor_asym_mag", "controller_train_motor_lag_range")
	assert set(keys) <= KNOWN_PARAMS
	for key in keys:
		_raises(lambda: controller_spec_from_params({key: 0.3}), ValueError, "not built", key)
	print("✓ provenance train_lag= (absent off), KNOWN_PARAMS, flow adapter refuses the keys")


def test_wheel():
	import wnn.control._accel as ra
	# train_motor_lag_range landed in ABI 68; this tree's wheel must be the ABI its _accel expects.
	assert int(ra.require_accel().ABI_VERSION) == int(ra.EXPECTED_ABI) >= 68, (ra.EXPECTED_ABI,)
	base = dict(translation=True, af_mass=0.0393, af_k_thrust=0.2, motor_lag_s=0.2, dist_enabled=True)
	cfg = ra.RewardGatedConfigPacked(**base, train_motor_lag_range=(0.15, 0.3))
	assert tuple(round(x, 6) for x in cfg.train_motor_lag_range) == (0.15, 0.3)
	assert ra.RewardGatedConfigPacked().train_motor_lag_range is None
	_raises(lambda: ra.RewardGatedConfigPacked(**{**base, "dist_enabled": False}, train_motor_lag_range=(0.15, 0.3)),
	        ValueError, "weather", "no weather")
	_raises(lambda: ra.RewardGatedConfigPacked(**base, train_motor_lag_range=(0.3, 0.15)), ValueError, "lo <= hi", "lo > hi")
	print(f"✓ wheel ABI {ra.EXPECTED_ABI}: RewardGatedConfigPacked(train_motor_lag_range=) accepted / refused")


if __name__ == "__main__":
	test_cli_and_refusals()
	test_training_hosts()
	test_report_pools_unchanged_search_pools_spread()
	test_phased_ga_wires_search_vs_report()
	test_provenance_registry_adapter()
	test_wheel()
	print("\nCTRL-87 training spread (Python): all passed.")
