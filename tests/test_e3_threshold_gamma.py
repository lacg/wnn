#!/usr/bin/env python3
"""E3 threshold-gamma warp — unit test.

Verifies (plan .claude/plans/controller_break_90_v2.md E3):
1. gamma=1.0 (default) → thresholds BIT-IDENTICAL to the pre-E3 fitter
   (parity anchor: the warp expression is skipped entirely).
2. gamma=2.0 → thresholds are DENSER near each feature's median: the middle
   half of the threshold ladder spans a smaller value range than at gamma=1.
3. spec_from_arch propagates threshold_gamma (stage chaining keeps the warp).

Uses the real fit_thresholds_from_pid_rollouts on a tiny s16 spec (2 episodes,
fast) — the PID rollout is deterministic given the seed.
"""

import sys

from wnn.control.evaluator import (
	ControllerSpec, fit_thresholds_from_pid_rollouts, spec_from_arch,
)


def spread_of_middle(ts: list[float]) -> float:
	"""Value span of the middle half of one feature's sorted threshold ladder."""
	s = sorted(ts)
	q = len(s) // 4
	return s[-q - 1] - s[q]


def main() -> int:
	failures = 0
	base = ControllerSpec(levels_per_motor=16, state_neurons=8,
	                      state_bits_per_neuron=16, output_bits_per_neuron=16)

	t_default = fit_thresholds_from_pid_rollouts(base, num_episodes=2, seed=7)
	t_gamma1 = fit_thresholds_from_pid_rollouts(
		ControllerSpec(levels_per_motor=16, state_neurons=8,
		               state_bits_per_neuron=16, output_bits_per_neuron=16,
		               threshold_gamma=1.0),
		num_episodes=2, seed=7)
	if t_default != t_gamma1:
		print("FAIL: gamma=1.0 is not bit-identical to the default fitter")
		failures += 1
	else:
		print(f"PASS: gamma=1.0 parity anchor ({len(t_default)} thresholds identical)")

	t_gamma2 = fit_thresholds_from_pid_rollouts(
		ControllerSpec(levels_per_motor=16, state_neurons=8,
		               state_bits_per_neuron=16, output_bits_per_neuron=16,
		               threshold_gamma=2.0),
		num_episodes=2, seed=7)
	bpf = base.bits_per_feature
	denser = 0
	for f in range(base.num_features()):
		lad1 = t_gamma1[f * bpf:(f + 1) * bpf]
		lad2 = t_gamma2[f * bpf:(f + 1) * bpf]
		s1, s2 = spread_of_middle(lad1), spread_of_middle(lad2)
		if s2 < s1 or (s1 == 0.0 and s2 == 0.0):  # constant features (target=0) tie at 0
			denser += 1
	nf = base.num_features()
	if denser < nf - 1:  # allow 1 pathological feature
		print(f"FAIL: gamma=2.0 middle-ladder denser on only {denser}/{nf} features")
		failures += 1
	else:
		print(f"PASS: gamma=2.0 densifies the middle ladder on {denser}/{nf} features")

	# 3. stage chaining propagates the warp
	class _G:  # minimal arch-genome stub for spec_from_arch
		output_neurons = 64
		state_neurons = 8
		state_bits_per_neuron = 16
		output_bits_per_neuron = 16
	child = spec_from_arch(_G(), ControllerSpec(threshold_gamma=1.7))
	if child.threshold_gamma != 1.7:
		print(f"FAIL: spec_from_arch dropped threshold_gamma (got {child.threshold_gamma})")
		failures += 1
	else:
		print("PASS: spec_from_arch propagates threshold_gamma=1.7")

	print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
	return 1 if failures else 0


if __name__ == "__main__":
	sys.exit(main())
