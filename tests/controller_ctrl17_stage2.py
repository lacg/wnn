"""CTRL-17 stage-2 trainer fixes — the Python half (26/09/2026).

docs/ctrl17_stage2_trainer_audit.md. The Rust half (episode regime, eval, sampler,
recorder, byte-identity pins) is in `cargo test -p ram_controller` (ctrl17_*).
This pins what only Python can see:

  G1  fit_thresholds_from_pid_rollouts under translation + xy flies the Rust
      training-cascade sampler: the four xy ladders are NOT all-zero any more.
  G2  the translating recorder passes reference_cfg; the raw binding REFUSES
      stage-1 draws without it and a double-owned plant.
  G4  gate λ resolution (explicit / derived / off) and its plumbing into BOTH
      trainer ctor sites (_gate_lambda_kwargs), plus the unset flag default.
  G11 the --reward-lambda-pos guard names --reward-lambda-pos.

Run: PYTHONPATH=src python tests/controller_ctrl17_stage2.py
"""

import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wnn.control._accel as ra  # noqa: E402
from wnn.control import evaluator as ev_mod  # noqa: E402
from wnn.control.airframe import Airframe  # noqa: E402
from wnn.control.evaluator import ControllerSpec, fit_thresholds_from_pid_rollouts  # noqa: E402
from wnn.control.gate_lambdas import resolve_gate_lambdas  # noqa: E402
from wnn.control.reward_gated import RewardGatedConfig  # noqa: E402
from wnn.control.training import EpisodeConfig  # noqa: E402

FAILS = 0


def check(label: str, got, want, tol: float = 0.0) -> None:
	global FAILS
	ok = abs(float(got) - float(want)) <= tol if tol > 0.0 else got == want
	print(f"  {'ok  ' if ok else 'FAIL'} {label:<66} -> {got!r}" + ("" if ok else f" (expected {want!r})"))
	if not ok:
		FAILS += 1


def stage2_ec(steps: int = 1000) -> EpisodeConfig:
	return EpisodeConfig(airframe=Airframe.preset("cf21_brushless"), translation=True,
	                     steps_per_episode=steps, max_initial_tilt_rad=math.radians(5.0),
	                     max_initial_yaw_rad=math.radians(5.0), max_initial_body_rate=0.2,
	                     max_initial_yaw_rate=0.1, max_initial_alt_offset_m=0.3,
	                     max_initial_vz=0.2, collective_cmd_jitter=0.1, mass_jitter=0.15,
	                     max_initial_xy_offset_m=0.5)


def stage2_spec() -> ControllerSpec:
	return ControllerSpec(levels_per_motor=8, bits_per_feature=4, input_window_k=1,
	                      state_neurons=0, state_bits_per_neuron=0, output_bits_per_neuron=8,
	                      obs_collective_cmd=True, obs_alt_err=True, obs_vz=True,
	                      obs_pos_err_xy=True, obs_vel_xy=True)


class _Args:
	"""The phased_ga fields resolve_gate_lambdas reads (explicit, typed)."""
	def __init__(self, lam_alt, lam_pos, w_alt, w_pos, w_err=0.3125, w_stable=0.25, w_steady=0.4375):
		self.reward_lambda_alt, self.reward_lambda_pos = lam_alt, lam_pos
		self.fit_weight_alt, self.fit_weight_pos = w_alt, w_pos
		self.fit_weight_err_sq, self.fit_weight_stable, self.fit_weight_steady = w_err, w_stable, w_steady


def test_g4_resolution() -> None:
	ec, tilt = stage2_ec(), math.radians(30.0)
	a, p, _ = resolve_gate_lambdas(_Args(None, None, 0.0, 0.0), ec, tilt)
	check("unset λ + zero rank weights ⇒ attitude-only gate", (a, p), (0.0, 0.0))
	a, p, _ = resolve_gate_lambdas(_Args(0.0, 0.0, 0.1, 0.1), ec, tilt)
	check("explicit 0 wins over the rank weights", (a, p), (0.0, 0.0))
	a, p, _ = resolve_gate_lambdas(_Args(None, None, 0.1, 0.1), ec, tilt)
	check("derived λ_alt = (w/w_att)·2θ²/s²", a, 0.1 * 2.0 * tilt ** 2 / 0.3 ** 2, tol=1e-12)
	check("derived λ_pos = (w/w_att)·θ²/s²", p, 0.1 * tilt ** 2 / 0.5 ** 2, tol=1e-12)
	a, p, _ = resolve_gate_lambdas(_Args(16.0, None, 0.0, 0.0), EpisodeConfig(), tilt)
	check("translation off ⇒ 0 whatever the flags", (a, p), (0.0, 0.0))


class _Captured(Exception):
	pass


def capture_trainer_cfg(rg: RewardGatedConfig):
	"""Intercept the kwargs the single-genome site hands RewardGatedConfigPacked."""
	spec = ControllerSpec(levels_per_motor=8, bits_per_feature=4, input_window_k=2,
	                      state_neurons=4, state_bits_per_neuron=8, output_bits_per_neuron=8)
	ev = ev_mod.ControllerEvaluator(spec, episode_config=rg.episode_config,
	                                max_eval_workers_gpu=False, rg_config=rg)
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
	return real(**box)


def test_g4_plumbing() -> None:
	rg = RewardGatedConfig(seed=0, episode_config=stage2_ec())
	rg.gate_lambda_alt, rg.gate_lambda_pos = 0.25, 0.5
	cfg = capture_trainer_cfg(rg)
	check("packed lambda_alt = rg.gate_lambda_alt", cfg.lambda_alt, 0.25, tol=1e-7)
	check("packed lambda_pos = rg.gate_lambda_pos", cfg.lambda_pos, 0.5, tol=1e-7)
	rg2 = RewardGatedConfig(seed=0, episode_config=stage2_ec())
	cfg2 = capture_trainer_cfg(rg2)
	check("unresolved rg ⇒ the episode config's λ (0)", (cfg2.lambda_alt, cfg2.lambda_pos), (0.0, 0.0))
	src = open(ev_mod.__file__).read()
	n = src.count("**_gate_lambda_kwargs(rg),")
	check("both trainer ctor sites splat _gate_lambda_kwargs(rg)", n, 2)


def test_g4_g11_cli() -> None:
	from wnn.control import phased_ga as pg
	args = pg.build_arg_parser().parse_args([])
	check("--reward-lambda-alt defaults to UNSET", args.reward_lambda_alt, None)
	check("--reward-lambda-pos defaults to UNSET", args.reward_lambda_pos, None)
	check("unset λ ⇒ scorer's ec.lambda_alt 0.0", pg.episode_config_from_args(args).lambda_alt, 0.0)
	src = open(pg.__file__).read()
	check("G11: the λ_pos guard names --reward-lambda-pos",
	      bool(re.search(r'"--reward-lambda-pos requires --translation and --xy-offset', src)), True)


def span(xs) -> float:
	return max(xs) - min(xs) if xs else 0.0


def test_g1_fitter_flies_the_cascade() -> None:
	spec, ec = stage2_spec(), stage2_ec()
	th = fit_thresholds_from_pid_rollouts(spec, num_episodes=4, seed=3, episode_config=ec)
	bpf, nf = spec.bits_per_feature, spec.num_features()
	check("ladder length = nf·bpf", len(th), nf * bpf)
	for f, name in ((12, "e_x"), (13, "e_y"), (14, "v_x"), (15, "v_y"), (10, "alt_err"), (11, "vz")):
		check(f"G1: {name} ladder is not degenerate (span > 0)", span(th[f * bpf:(f + 1) * bpf]) > 1e-4, True)


def recorder_args(ec, n: int):
	from wnn.control.ga_memory import _recorder_plant_kwargs
	return _recorder_plant_kwargs(ec, n, 5)


def test_g2_recorder() -> None:
	from wnn.control.ga_memory import record_address_universe
	from wnn.control.evaluator import random_connectivity
	spec, ec = stage2_spec(), stage2_ec()
	th = fit_thresholds_from_pid_rollouts(spec, num_episodes=2, seed=3, episode_config=ec)
	sc, oc = random_connectivity(spec, seed=1)
	s_uni, o_uni = record_address_universe(spec, th, sc, oc, num_episodes=3, steps=600,
	                                       seed=5, episode_config=ec, teacher="mpcof")
	check("G2: the translating recorder records a non-empty universe", len(o_uni) > 0, True)
	plant = recorder_args(ec, 3)
	check("G2: reference_cfg rides with the draws", "reference_cfg" in plant, True)
	check("G2: no af_* alongside reference_cfg", [k for k in plant if k.startswith("af_")], [])
	c = ev_mod._feature_controller(spec)
	q, om = [[1.0, 0.0, 0.0, 0.0]] * 3, [[0.0, 0.0, 0.0]] * 3
	draws = {k: v for k, v in plant.items() if k != "reference_cfg"}
	for label, kw in (("stage-1 draws without reference_cfg", draws),
	                  ("reference_cfg + a second plant owner", {**plant, "af_k_thrust": 0.2})):
		try:
			ra.record_address_universe(c, q, om, [0.0, 0.0, 0.0], 50, **kw)
			check(f"G2: refuses {label}", "accepted", "ValueError")
		except ValueError:
			check(f"G2: refuses {label}", "ValueError", "ValueError")


if __name__ == "__main__":
	for t in (test_g4_resolution, test_g4_plumbing, test_g4_g11_cli,
	          test_g1_fitter_flies_the_cascade, test_g2_recorder):
		print(t.__name__)
		t()
	print(f"\n{'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}: controller_ctrl17_stage2")
	sys.exit(1 if FAILS else 0)
