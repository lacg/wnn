"""Trainer-airframe parity (PAPER-CRITICAL gap, 19/09/2026).

The accelerated DAgger trainer (RewardGatedConfigPacked) used to receive ONLY
af_mass from the evaluator, so every --airframe run TRAINED on the synthetic
plant (k_thrust 2.4, inertia 2.3e-3, implied hover 0.200) and SCORED on the
airframe (cf21: k_thrust 0.2, inertia 3e-5, hover 0.694). This pins the fix in
BOTH directions, at the evaluator's OWN packing site:

  (a) cf21_brushless EpisodeConfig -> the trainer cfg carries the cf21 plant AND
      the firmware PID cascade the scorer's PID rival runs (af_pid_* identical to
      EpisodeConfig.airframe_kwargs), implied hover 0.694.
  (b) no-airframe EpisodeConfig     -> the trainer cfg is the Rust defaults
      (synthetic plant, no cascade), so the parity anchors stay bit-identical.
  (c) both evaluator ctor sites splat _plant_train_kwargs (source check — the
      batch site needs a materialized population to reach its ctor call).

Run: PYTHONPATH=src python tests/controller_trainer_airframe.py  (src, not src/wnn — see below)
"""

import math
import os
import re
import sys

# The parent of the `wnn` package, so THIS checkout shadows the venv's editable
# install (its .pth adds <live>/src; a src/wnn entry would not shadow wnn.control).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wnn.control._accel as ra  # noqa: E402
from wnn.control import evaluator as ev_mod  # noqa: E402
from wnn.control.airframe import Airframe  # noqa: E402
from wnn.control.evaluator import ControllerEvaluator, ControllerSpec  # noqa: E402
from wnn.control.reward_gated import RewardGatedConfig  # noqa: E402
from wnn.control.training import EpisodeConfig  # noqa: E402

FAILS = 0
SYNTHETIC = dict(af_arm_length=0.075, af_k_thrust=2.4, af_k_drag=0.05,
                 af_inertia=[0.0023, 0.0023, 0.0046], af_gravity=9.81)


def check(label: str, got, want, tol: float = 0.0) -> None:
	global FAILS
	if tol > 0.0:
		ok = abs(float(got) - float(want)) <= tol
	else:
		ok = got == want
	print(f"  {'ok  ' if ok else 'FAIL'} {label:<62} -> {got!r}" + ("" if ok else f" (expected {want!r})"))
	if not ok:
		FAILS += 1


class _Captured(Exception):
	pass


def capture_trainer_cfg(ec: EpisodeConfig):
	"""Drive the evaluator's real single-genome packing site (_train_genome_rust)
	and intercept the kwargs it hands RewardGatedConfigPacked, then build the
	REAL packed config from exactly those kwargs. The intercept raises so no
	training happens — the ctor call is the first thing the method does."""
	spec = ControllerSpec(levels_per_motor=8, bits_per_feature=4, input_window_k=2,
	                      state_neurons=4, state_bits_per_neuron=8, output_bits_per_neuron=8)
	ev = ControllerEvaluator(spec, episode_config=ec, max_eval_workers_gpu=False,
	                         rg_config=RewardGatedConfig(seed=0, episode_config=ec))
	box: dict = {}
	real = ra.RewardGatedConfigPacked

	class Capture:
		def __init__(self, **kw):
			box.update(kw)
			raise _Captured

	ra.RewardGatedConfigPacked = Capture
	try:
		ev._train_genome_rust(spec, [], [], None, None, 0)
	except _Captured:
		pass
	finally:
		ra.RewardGatedConfigPacked = real
	assert box, "the packing site never reached the ctor"
	return real(**box), box


def implied_hover(cfg, mass: float) -> float:
	"""hover pwm from thrust = k_thrust * pwm^2 per motor, 4 motors."""
	return math.sqrt(mass * cfg.af_gravity / (4.0 * cfg.af_k_thrust))


def test_cf21_trainer_matches_scorer() -> None:
	af = Airframe.preset("cf21_brushless")
	ec = EpisodeConfig(airframe=af, translation=True, target_altitude=1.0)
	cfg, kw = capture_trainer_cfg(ec)
	check("af_k_thrust is cf21's 0.2", cfg.af_k_thrust, 0.2, tol=1e-6)
	check("af_inertia[0] is cf21's 3.0e-5", cfg.af_inertia[0], 3.0e-5, tol=2e-7)
	check("af_arm_length is cf21's 0.0707", cfg.af_arm_length, 0.0707, tol=1e-4)
	check("af_mass is cf21's 0.0393", cfg.af_mass, 0.0393, tol=1e-7)
	check("af_dt is NOT a trainer kwarg (trainer takes dt)", "af_dt" in kw, False)
	check("dt is the episode dt", cfg.dt, ec.dt, tol=1e-12)
	scorer = ec.airframe_kwargs()
	for k in ("af_pid_att", "af_pid_rate"):
		check(f"{k} equals the scorer's cascade", list(cfg.__getattribute__(k)), list(scorer[k]))
	for k in ("af_pid_out_limit_n", "af_pid_hover_n", "af_pid_attitude_hz", "af_pid_lpf_hz"):
		check(f"{k} equals the scorer's", cfg.__getattribute__(k), scorer[k], tol=1e-12)
	check("af_pid_hover_n = m*g/4 (cascade hovers cf21)", cfg.af_pid_hover_n, af.mass * af.gravity / 4.0, tol=1e-9)
	check("implied trainer hover pwm is cf21's 0.694", implied_hover(cfg, af.mass), 0.694, tol=1e-3)
	plant = {k: (list(v) if isinstance(v, list) else v) for k, v in scorer.items() if k in SYNTHETIC}
	got = {k: (list(cfg.__getattribute__(k)) if k == "af_inertia" else cfg.__getattribute__(k)) for k in SYNTHETIC}
	same = all(abs(a - b) <= 1e-6 for k in SYNTHETIC
	           for a, b in zip(got[k] if k == "af_inertia" else [got[k]], plant[k] if k == "af_inertia" else [plant[k]]))
	check("all 5 plant fields equal the scorer's airframe_kwargs", same, True)


def test_cf21_attitude_only_still_flies_cf21() -> None:
	# Translation OFF: the stage-1 helper passes nothing; the airframe must still reach the trainer.
	af = Airframe.preset("cf21_brushless")
	cfg, kw = capture_trainer_cfg(EpisodeConfig(airframe=af))
	check("attitude-only: af_k_thrust still cf21's 0.2", cfg.af_k_thrust, 0.2, tol=1e-6)
	check("attitude-only: cascade supplied (attitude_hz 500)", cfg.af_pid_attitude_hz, 500.0, tol=1e-9)
	check("attitude-only: translation stays off", cfg.translation, False)
	check("attitude-only: af_mass stays the Rust default 0.0", cfg.af_mass, 0.0, tol=0.0)


def test_no_airframe_is_rust_defaults() -> None:
	cfg, kw = capture_trainer_cfg(EpisodeConfig())
	check("no airframe: no af_* kwarg passed at all", [k for k in kw if k.startswith("af_")], [])
	check("no airframe: af_k_thrust is the synthetic 2.4", cfg.af_k_thrust, SYNTHETIC["af_k_thrust"], tol=1e-6)
	check("no airframe: af_inertia is the synthetic 2.3e-3", cfg.af_inertia[0], 0.0023, tol=1e-6)
	check("no airframe: af_arm_length is the synthetic 0.075", cfg.af_arm_length, 0.075, tol=1e-6)
	check("no airframe: no cascade (attitude_hz 0 -> legacy PID teacher)", cfg.af_pid_attitude_hz, 0.0, tol=0.0)
	check("no airframe: implied hover at cf21 mass would be 0.200", implied_hover(cfg, 0.0393), 0.200, tol=1e-3)


def test_both_ctor_sites_use_the_plant_helper() -> None:
	src = open(ev_mod.__file__).read()
	sites = [m.start() for m in re.finditer(r"ra\.RewardGatedConfigPacked\(", src)]
	check("evaluator has exactly two trainer ctor sites", len(sites), 2)
	for i, pos in enumerate(sites):
		body = src[pos:src.index("\n\t\t)\n", pos)]
		check(f"site {i} splats _plant_train_kwargs", "**_plant_train_kwargs(rg.episode_config)" in body, True)
		check(f"site {i} does not splat _stage1_train_kwargs directly", "**_stage1_train_kwargs(" in body, False)


if __name__ == "__main__":
	for t in (test_cf21_trainer_matches_scorer, test_cf21_attitude_only_still_flies_cf21,
	          test_no_airframe_is_rust_defaults, test_both_ctor_sites_use_the_plant_helper):
		print(t.__name__)
		t()
	print(f"\n{'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}: controller_trainer_airframe")
	sys.exit(1 if FAILS else 0)
