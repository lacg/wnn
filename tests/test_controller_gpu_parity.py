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
from dataclasses import replace

import numpy as np

from wnn.control._accel import (
	AttitudeSim, disturbance_episode_seed, score_controllers_metal,
)
from wnn.control.evaluator import (
	ControllerSpec, ControllerGenome, build_controller,
	fit_thresholds_from_pid_rollouts,
)
from wnn.control.genome import FiniteStateGenome
from wnn.control.reward_gated import reward_gated_train, RewardGatedConfig
from wnn.control.training import (
	DisturbanceConfig, EpisodeConfig, run_episode, make_wnn_action_fn,
	_sample_initial_state,
)

# W2 disturbance case: FIXED motor_asym (no per-episode ±mag draw — the draw is
# numpy-rng based and the kernel can't mirror it) + a deterministic per-episode
# noise seed via disturbance_episode_seed — the exact channel-15 derivation the
# kernel applies, so CPU and GPU consume identical noise streams.
DIST_SEED = 777
DIST = DisturbanceConfig(
	tau_bias=(0.004, 0.0, 0.0), gust_sigma=0.02, gust_tau_c=0.1,
	motor_asym=(1.03, 0.97, 1.01, 0.99), motor_asym_mag=0.0,
	gyro_sigma=0.01, gyro_bias_walk=0.001, accel_sigma=0.1, seed=DIST_SEED,
)


def cpu_score(controller, ep_seeds, ec, dist=None):
	"""Per-genome CPU aggregate over the SAME ICs the GPU will use. When `dist`
	is set, episode ep_idx runs under DisturbanceConfig(episode_seed =
	disturbance_episode_seed(dist.seed, ep_idx)) — the GPU kernel's derivation."""
	from wnn.control._accel import monotonicity_violations
	sim = AttitudeSim()
	rewards, errs, jerks, monos, steadys, stable = [], [], [], [], [], 0
	rises, settleabs, settlerels, itaes, iaes, ises = [], [], [], [], [], []
	for ep_idx, s in enumerate(ep_seeds):
		controller.reset()
		rng = np.random.default_rng(s)   # same seed → run_episode samples same ICs
		ec_ep = ec
		if dist is not None:
			ep_dist = replace(dist, episode_seed=int(disturbance_episode_seed(dist.seed, ep_idx)))
			ec_ep = replace(ec, disturbance=ep_dist)
		res = run_episode(make_wnn_action_fn(controller), sim, ec_ep, rng=rng)
		rewards.append(res.cumulative_reward)
		errs.append(res.mean_attitude_error_rad)
		jerks.append(res.mean_pwm_jerk)
		steadys.append(res.mean_steady_error_rad)
		monos.append(float(monotonicity_violations(
			controller.get_last_output_cells(), controller.levels_per_motor, controller.num_motors)))
		# Transient-speed metrics — CPU oracle for the GPU rollout's trailing 6.
		rises.append(res.rise_time_s)
		settleabs.append(res.settle_time_abs2deg_s)
		settlerels.append(res.settle_time_rel5pct_s)
		itaes.append(res.itae); iaes.append(res.iae); ises.append(res.ise)
		if (not res.diverged) and res.mean_attitude_error_rad <= math.radians(5.0):
			stable += 1
	n = len(ep_seeds)
	return (float(np.mean(rewards)), float(np.mean(errs)), stable / n,
	        float(np.mean(jerks)), float(np.mean(monos)), float(np.mean(steadys)),
	        float(np.mean(rises)), float(np.mean(settleabs)), float(np.mean(settlerels)),
	        float(np.mean(itaes)), float(np.mean(iaes)), float(np.mean(ises)))


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

	# ---- clean case (pre-W2 behavior; dist args default to disabled) ----
	cpu = [cpu_score(c, ep_seeds, ec) for c in controllers]
	gpu = score_controllers_metal(controllers, q0, omega0, E, STEPS)
	ok_untrained, ok_aggregate = report_case("CLEAN", cpu, gpu, untrained_strict=True)

	# ---- W2 disturbance case: same ICs, same base seed on both paths ----
	cpu_d = [cpu_score(c, ep_seeds, ec, dist=DIST) for c in controllers]
	gpu_d = score_controllers_metal(
		controllers, q0, omega0, E, STEPS,
		dist_enabled=True,
		dist_tau_bias=[float(x) for x in DIST.tau_bias],
		dist_gust_sigma=float(DIST.gust_sigma),
		dist_gust_tau_c=float(DIST.gust_tau_c),
		dist_motor_asym=[float(x) for x in DIST.motor_asym],
		dist_gyro_sigma=float(DIST.gyro_sigma),
		dist_gyro_bias_walk=float(DIST.gyro_bias_walk),
		dist_accel_sigma=float(DIST.accel_sigma),
		dist_seed=DIST_SEED)
	# Untrained under weather is noise-driven (not contracting) → hold it to
	# the same STATISTICAL aggregate bounds as the trained genomes, not the
	# clean-case tight isolation check.
	_, ok_aggregate_d = report_case("DISTURBED (L~calibration probe)", cpu_d, gpu_d,
	                                untrained_strict=False)

	print("\nVERDICT:")
	print(f"  clean: untrained tight parity?  {'YES' if ok_untrained else 'NO'}")
	print(f"  clean: aggregate statistical parity?  {'YES' if ok_aggregate else 'NO'}")
	print(f"  disturbed: aggregate statistical parity?  {'YES' if ok_aggregate_d else 'NO'}")
	return 0 if (ok_untrained and ok_aggregate and ok_aggregate_d) else 1


def report_case(name, cpu, gpu, untrained_strict):
	"""Print the per-genome CPU/GPU table + aggregate deltas for one case.
	Returns (ok_untrained, ok_aggregate); ok_untrained is True when the strict
	check is skipped (untrained_strict=False)."""
	print(f"\n===== {name} =====")
	print(f"{'genome':<14}{'reward CPU':>12}{'reward GPU':>12}{'err° CPU':>10}{'err° GPU':>10}"
	      f"{'stbl CPU':>10}{'stbl GPU':>10}{'jerk CPU':>10}{'jerk GPU':>10}{'mono CPU':>10}{'mono GPU':>10}"
	      f"{'stdy° CPU':>10}{'stdy° GPU':>10}")
	max_rew_rel, max_err_abs, max_stbl_abs = 0.0, 0.0, 0.0
	max_jerk_abs, max_mono_abs, max_steady_abs = 0.0, 0.0, 0.0
	labels = ["untrained", "trained s2", "trained s3"]
	for i, (lab, cpu_row, gpu_row) in enumerate(zip(labels, cpu, gpu)):
		# Both rows are 12-metric now; the core parity assertions use the first 6.
		# (trailing 6 = rise/settle_abs/settle_rel/itae/iae/ise, transient-speed.)
		(cr, ce, cs, cj, cm, ct) = cpu_row[:6]
		(gr, ge, gs, gj, gm, gt) = gpu_row[:6]
		print(f"{lab:<14}{cr:>12.2f}{gr:>12.2f}{math.degrees(ce):>10.2f}{math.degrees(ge):>10.2f}"
		      f"{cs*100:>9.0f}%{gs*100:>9.0f}%{cj:>10.4f}{gj:>10.4f}{cm:>10.1f}{gm:>10.1f}"
		      f"{math.degrees(ct):>10.2f}{math.degrees(gt):>10.2f}")
		denom = max(abs(cr), 1.0)
		max_rew_rel = max(max_rew_rel, abs(cr - gr) / denom)
		max_err_abs = max(max_err_abs, abs(math.degrees(ce) - math.degrees(ge)))
		max_stbl_abs = max(max_stbl_abs, abs(cs - gs))
		max_jerk_abs = max(max_jerk_abs, abs(cj - gj))
		max_mono_abs = max(max_mono_abs, abs(cm - gm))
		max_steady_abs = max(max_steady_abs, abs(math.degrees(ct) - math.degrees(gt)))
	print(f"\nmax jerk abs-diff: {max_jerk_abs:.4f}   max mono abs-diff: {max_mono_abs:.1f}   "
	      f"max steady abs-diff: {max_steady_abs:.3f}°")
	print(f"max reward rel-diff: {max_rew_rel*100:.2f}%   "
	      f"max mean-err abs-diff: {max_err_abs:.3f}°   max stable-rate diff: {max_stbl_abs*100:.0f}%")

	ok_untrained = True
	if untrained_strict:
		# Untrained is non-chaotic (constant hover) → should match tightly.
		un_rew_rel = abs(cpu[0][0] - gpu[0][0]) / max(abs(cpu[0][0]), 1.0)
		un_err_abs = abs(math.degrees(cpu[0][1]) - math.degrees(gpu[0][1]))
		print(f"UNTRAINED (sim+decode isolation): reward rel-diff {un_rew_rel*100:.3f}%, "
		      f"err abs-diff {un_err_abs:.4f}°")
		ok_untrained = un_rew_rel < 0.02 and un_err_abs < 0.5
	ok_aggregate = max_err_abs < 3.0 and max_stbl_abs <= 0.17 and max_steady_abs < 3.0  # ≤2/12 episodes flip

	# --- Transient-speed metrics (indices 6..11) CPU↔GPU parity ---
	# ITAE/IAE/ISE are integrals → track tightly like reward. Rise/settle are
	# threshold-based → a near-band episode can flip under f32 rollout drift, so
	# assert on the AGGREGATE (mean over episodes), matching the steady tolerance.
	tnames = ["rise_s", "settle_abs_s", "settle_rel_s", "itae", "iae", "ise"]
	print(f"\n  transient-speed CPU↔GPU (aggregate):")
	max_t_rel = 0.0
	for j, tn in enumerate(tnames):
		k = 6 + j
		cvals = [row[k] for row in cpu]; gvals = [row[k] for row in gpu]
		cmean = sum(cvals) / len(cvals); gmean = sum(gvals) / len(gvals)
		rel = abs(cmean - gmean) / max(abs(cmean), 1e-6)
		max_t_rel = max(max_t_rel, rel)
		print(f"    {tn:<14} CPU {cmean:>10.5f}   GPU {gmean:>10.5f}   rel-diff {rel*100:>6.2f}%")
	# Integrals + aggregate settle/rise should agree within a few %% over 12 eps.
	ok_transient = max_t_rel < 0.05
	print(f"  max transient aggregate rel-diff: {max_t_rel*100:.2f}%  → {'OK' if ok_transient else 'FAIL'}")
	return ok_untrained, (ok_aggregate and ok_transient)


if __name__ == "__main__":
	sys.exit(main())
