"""GPU↔CPU parity for the closed-loop controller eval (score_controllers_metal).

Feeds IDENTICAL per-episode initial conditions to both paths, then compares the
per-genome aggregates (mean_reward, mean_attitude_error_rad, stable_rate).

Parity target is STATISTICAL, not bit-exact: a 2000-step closed loop amplifies
tiny f32 op-order/transcendental differences, so stable (contracting) rollouts
track closely while tumbling (chaotic) ones may diverge. We assert the aggregate
metrics match within tolerance and report per-genome deltas. The untrained
controller (constant hover decode, no chaos) should match tightly — it isolates
the sim+decode port from the memory-lookup port.

Run:  python tests/test_controller_gpu_parity.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from ram_accelerator import AttitudeSim, score_controllers_metal
from wnn.control.evaluator import (
	ControllerSpec, ControllerGenome, build_controller,
	fit_thresholds_from_pid_rollouts,
)
from wnn.control.genome import FiniteStateGenome
from wnn.control.reward_gated import reward_gated_train, RewardGatedConfig
from wnn.control.training import (
	EpisodeConfig, run_episode, make_wnn_action_fn, _sample_initial_state,
)


def cpu_score(controller, ep_seeds, ec):
	"""Per-genome CPU aggregate over the SAME ICs the GPU will use."""
	sim = AttitudeSim()
	rewards, errs, stable = [], [], 0
	for s in ep_seeds:
		controller.reset()
		rng = np.random.default_rng(s)   # same seed → run_episode samples same ICs
		res = run_episode(make_wnn_action_fn(controller), sim, ec, rng=rng)
		rewards.append(res.cumulative_reward)
		errs.append(res.mean_attitude_error_rad)
		if (not res.diverged) and res.mean_attitude_error_rad <= math.radians(5.0):
			stable += 1
	n = len(ep_seeds)
	return float(np.mean(rewards)), float(np.mean(errs)), stable / n


def main():
	spec = ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
		input_window_k=4, state_neurons=4, state_bits_per_neuron=24,
		output_bits_per_neuron=24, delta_control=False)
	E, STEPS, TILT = 12, 800, 15.0
	ec = EpisodeConfig(dt=0.001, steps_per_episode=STEPS,
		max_initial_tilt_rad=math.radians(TILT), max_initial_yaw_rad=math.radians(TILT),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)
	th = fit_thresholds_from_pid_rollouts(spec, num_episodes=8, seed=0)

	# Build a mix: 1 untrained (isolates sim+decode) + 2 lightly-trained (exercises
	# the sparse memory binary-search path).
	controllers = []
	g0 = FiniteStateGenome.random(spec, seed=1)
	controllers.append(build_controller(ControllerGenome(spec=spec, thresholds=th,
		state_connections=g0.state_connections, output_connections=g0.output_connections)))
	rg = RewardGatedConfig(num_rounds=2, episodes_per_round=6, steps_per_episode=600,
		bptt_window=16, gate_mode="improvement", curriculum=False, full_tilt_deg=TILT,
		eval_episodes=4, episode_config=ec, progress=False)
	for s in (2, 3):
		g = FiniteStateGenome.random(spec, seed=s)
		c, _ = reward_gated_train(spec, th, g.state_connections, g.output_connections,
			RewardGatedConfig(**{**rg.__dict__, "seed": s}))
		controllers.append(c)

	# Shared per-episode ICs.
	rng = np.random.default_rng(123)
	ep_seeds = [int(rng.integers(0, 2**31)) for _ in range(E)]
	q0, omega0 = [], []
	for s in ep_seeds:
		r = np.random.default_rng(s)
		q, om = _sample_initial_state(r, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
			ec.max_initial_body_rate, ec.max_initial_yaw_rate)
		q0 += [float(x) for x in q]
		omega0 += [float(x) for x in om]

	# CPU aggregates.
	cpu = [cpu_score(c, ep_seeds, ec) for c in controllers]
	# GPU aggregates (same ICs, AttitudeSim defaults).
	gpu = score_controllers_metal(controllers, q0, omega0, E, STEPS)

	print(f"{'genome':<14}{'reward CPU':>12}{'reward GPU':>12}{'err° CPU':>10}{'err° GPU':>10}"
	      f"{'stbl CPU':>10}{'stbl GPU':>10}")
	max_rew_rel, max_err_abs, max_stbl_abs = 0.0, 0.0, 0.0
	labels = ["untrained", "trained s2", "trained s3"]
	for i, (lab, (cr, ce, cs), (gr, ge, gs)) in enumerate(zip(labels, cpu, gpu)):
		print(f"{lab:<14}{cr:>12.2f}{gr:>12.2f}{math.degrees(ce):>10.2f}{math.degrees(ge):>10.2f}"
		      f"{cs*100:>9.0f}%{gs*100:>9.0f}%")
		denom = max(abs(cr), 1.0)
		max_rew_rel = max(max_rew_rel, abs(cr - gr) / denom)
		max_err_abs = max(max_err_abs, abs(math.degrees(ce) - math.degrees(ge)))
		max_stbl_abs = max(max_stbl_abs, abs(cs - gs))

	print(f"\nmax reward rel-diff: {max_rew_rel*100:.2f}%   "
	      f"max mean-err abs-diff: {max_err_abs:.3f}°   max stable-rate diff: {max_stbl_abs*100:.0f}%")

	# Untrained is non-chaotic (constant hover) → should match tightly.
	un_rew_rel = abs(cpu[0][0] - gpu[0][0]) / max(abs(cpu[0][0]), 1.0)
	un_err_abs = abs(math.degrees(cpu[0][1]) - math.degrees(gpu[0][1]))
	print(f"\nUNTRAINED (sim+decode isolation): reward rel-diff {un_rew_rel*100:.3f}%, "
	      f"err abs-diff {un_err_abs:.4f}°")

	ok_untrained = un_rew_rel < 0.02 and un_err_abs < 0.5
	ok_aggregate = max_err_abs < 3.0 and max_stbl_abs <= 0.17  # ≤2/12 episodes flip
	print("\nVERDICT:")
	print(f"  untrained tight parity?  {'YES' if ok_untrained else 'NO'}")
	print(f"  aggregate statistical parity?  {'YES' if ok_aggregate else 'NO'}")
	return 0 if (ok_untrained and ok_aggregate) else 1


if __name__ == "__main__":
	sys.exit(main())
