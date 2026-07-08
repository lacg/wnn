"""GPU↔Python parity for the E5 RESIDUAL HYBRID (Phase 2).

The GPU rollout now composes the WNN output as a signed residual on an in-kernel
PID baseline (pid_step in controller_rollout.metal, mirror of AttitudePidRs::step_rs
= the Python AttitudePID). This asserts that scoring a hybrid via
score_controllers_metal(residual_enabled=True, pid_gains=...) matches the Python
make_residual_action_fn(make_pid_action_fn(AttitudePID), wnn) rolled out through
run_episode on the SAME ICs — i.e. the composition collapsed onto the Rust path.

f32 (GPU) vs f64 (Python) → assert on the AGGREGATE over episodes, same tolerance
family as the pure-WNN parity test.
"""
import math
import sys

import numpy as np

from wnn.control._accel import AttitudeSim, score_controllers_metal
from wnn.control.evaluator import (ControllerSpec, ControllerGenome, build_controller,
	fit_thresholds_from_pid_rollouts)
from wnn.control.genome import FiniteStateGenome
from wnn.control.training import (EpisodeConfig, run_episode, make_pid_action_fn,
	make_residual_action_fn, _sample_initial_state)
from wnn.control.dagger import _pd_config, _pid_plus_config
from wnn.control.pid import AttitudePID

SCALE, CLAMP = 1.0, 0.4


def gains_of(cfg):
	"""[kp_rp, ki_rp, kd_rp, iclamp_rp, kp_yaw, ki_yaw, kd_yaw, iclamp_yaw, hover, authority]."""
	return [cfg.roll.kp, cfg.roll.ki, cfg.roll.kd, cfg.roll.i_clamp,
	        cfg.yaw.kp, cfg.yaw.ki, cfg.yaw.kd, cfg.yaw.i_clamp,
	        cfg.hover_throttle, cfg.max_axis_authority]


def cpu_hybrid(controller, base_cfg, ep_seeds, ec):
	"""Python composed-hybrid aggregate over the SAME ICs the GPU uses."""
	sim = AttitudeSim()
	errs, stable = [], 0
	rises, sabs, srel, itaes = [], [], [], []
	base = AttitudePID(base_cfg)
	hy = make_residual_action_fn(make_pid_action_fn(base), controller, SCALE, CLAMP, 4)
	for s in ep_seeds:
		controller.reset(); base.reset()
		rng = np.random.default_rng(s)     # same seed → same ICs as the GPU q0/omega0
		res = run_episode(hy, sim, ec, rng=rng)
		errs.append(res.mean_attitude_error_rad)
		rises.append(res.rise_time_s); sabs.append(res.settle_time_abs2deg_s)
		srel.append(res.settle_time_rel5pct_s); itaes.append(res.itae)
		if (not res.diverged) and res.mean_attitude_error_rad <= math.radians(5.0):
			stable += 1
	n = len(ep_seeds)
	return dict(err=float(np.mean(errs)), stable=stable / n, rise=float(np.mean(rises)),
	            sabs=float(np.mean(sabs)), srel=float(np.mean(srel)), itae=float(np.mean(itaes)))


def main():
	spec = ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
		input_window_k=4, state_neurons=4, state_bits_per_neuron=24,
		output_bits_per_neuron=24, delta_control=False)
	E, STEPS, TILT = 12, 800, 15.0
	ec = EpisodeConfig(dt=0.001, steps_per_episode=STEPS,
		max_initial_tilt_rad=math.radians(TILT), max_initial_yaw_rad=math.radians(TILT),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)
	th = fit_thresholds_from_pid_rollouts(spec, num_episodes=8, seed=0)

	controllers = []
	for s in (1, 2):
		g = FiniteStateGenome.random(spec, seed=s)
		controllers.append(build_controller(ControllerGenome(spec=spec, thresholds=th,
			state_connections=g.state_connections, output_connections=g.output_connections)))

	rng = np.random.default_rng(123)
	ep_seeds = [int(rng.integers(0, 2**31)) for _ in range(E)]
	q0, omega0 = [], []
	for s in ep_seeds:
		r = np.random.default_rng(s)
		q, om = _sample_initial_state(r, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
			ec.max_initial_body_rate, ec.max_initial_yaw_rate)
		q0 += [float(x) for x in q]; omega0 += [float(x) for x in om]

	ok = True
	for bl_name, bl_cfg in (("pd", _pd_config()), ("pid+", _pid_plus_config())):
		pg = gains_of(bl_cfg)
		print(f"\n===== residual baseline = {bl_name}  gains={['%.3g'%x for x in pg]} =====")
		print(f"{'genome':<12}{'metric':<8}{'CPU':>12}{'GPU':>12}{'rel/abs diff':>14}")
		for ci, c in enumerate(controllers):
			cpu = cpu_hybrid(c, bl_cfg, ep_seeds, ec)
			gpu_rows = score_controllers_metal([c], q0, omega0, E, STEPS,
				residual_enabled=True, residual_scale=SCALE, residual_clamp=CLAMP, pid_gains=pg)
			r = gpu_rows[0]
			gpu = dict(err=r[1], stable=r[2], rise=r[6], sabs=r[7], srel=r[8], itae=r[9])
			# err in deg for readability; stable abs; transient integrals rel.
			de = abs(math.degrees(cpu['err']) - math.degrees(gpu['err']))
			ds = abs(cpu['stable'] - gpu['stable'])
			di = abs(cpu['itae'] - gpu['itae']) / max(abs(cpu['itae']), 1e-6)
			print(f"g{ci:<11}{'err°':<8}{math.degrees(cpu['err']):>12.3f}{math.degrees(gpu['err']):>12.3f}{de:>13.3f}°")
			print(f"{'':<12}{'stable':<8}{cpu['stable']*100:>11.0f}%{gpu['stable']*100:>11.0f}%{ds*100:>13.0f}%")
			print(f"{'':<12}{'itae':<8}{cpu['itae']:>12.4f}{gpu['itae']:>12.4f}{di*100:>13.2f}%")
			print(f"{'':<12}{'rise_s':<8}{cpu['rise']:>12.4f}{gpu['rise']:>12.4f}")
			print(f"{'':<12}{'settle2°':<8}{cpu['sabs']:>12.4f}{gpu['sabs']:>12.4f}")
			# Aggregate parity bounds (f32 feedback drift over 800 steps): err within
			# ~1°, stable within 1/12 episode, ITAE integral within a few %.
			if not (de < 1.0 and ds <= 1.0 / E + 1e-9 and di < 0.05):
				ok = False
	print(f"\nVERDICT: residual-hybrid GPU↔Python parity?  {'YES' if ok else 'NO'}")
	return 0 if ok else 1


if __name__ == "__main__":
	sys.exit(main())
