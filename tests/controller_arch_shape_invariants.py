#!/usr/bin/env python3
"""arch_shape_from_spec invariants — the two that cost box time on 04/08/2026.

Both bugs were of the SAME class: a fact about the architecture that the GA and the
experiment design each assumed, that nothing asserted. Neither was caught by the
CPU/GPU decode parity fixtures, because those test the forward path — these are
properties of the SEARCH SPACE, one layer up.

  1. OUTPUT QUANTUM vs DECODE. antagonist splits a motor's levels into E/I halves,
     so levels_per_motor MUST be even. The quantum that keeps neurogenesis stepping
     by whole even level counts was keyed on memory_mode == BINARY, which was an
     accurate proxy only while antagonist was welded to BINARY. --output-decode
     (ABI 21, 03/08) made it orthogonal; QUAD+antagonist then got q=4, the GA
     produced on=92 -> levels=23, and every affected genome fell back to the ~3x
     slower Python path mid-run.

  2. input_window_k DOES NOT REACH sn=0. The output layer samples its connections
     from `sensor_frame` (one frame) while only the state layer samples
     `sensor_window` (k frames). With no state layer, K is inert BY CONSTRUCTION.
     P3's pre-registered 2x2 had a (sn=0, K=8) corner that could not be a condition:
     it reproduced the K=4 arm bit-identically, held-out included.

Run: PYTHONPATH=src/wnn python3 tests/controller_arch_shape_invariants.py
"""
import sys

from wnn.control.evaluator import arch_shape_from_spec, random_connectivity
from wnn.control.phased_ga import _make_spec

# nf=15 pidmix — the feature set P1-P4 actually run.
PIDMIX = dict(obs_peraxis_p=True, obs_peraxis_i=True, obs_peraxis_yaw=False,
              obs_yaw_err=True, obs_yaw_err_i=True, bits_per_feature=8)
MODES = ["TERNARY", "QUAD_BINARY", "QUAD_WEIGHTED", "BINARY", "QSR", "PLN"]

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
	print(("  ok   " if cond else "  FAIL ") + msg)
	if not cond:
		_failures.append(msg)


def spec(mode="BINARY", decode=None, sn=0, k=4, levels=16, bits=30):
	return _make_spec(sn, levels, bits, True, 0.95, memory_mode=mode,
	                  output_decode=decode, input_window_k=k, **PIDMIX)


def test_antagonist_always_gets_the_even_quantum():
	"""ANY spec that resolves to antagonist needs quantum = 2*num_motors, whatever
	the memory mode. This is the assertion that would have caught the P4a fallback
	before it consumed box time."""
	print("\n[1] antagonist => quantum 2*num_motors, for EVERY memory mode")
	for mode in MODES:
		s = spec(mode=mode, decode="antagonist")
		q = arch_shape_from_spec(s).output_quantum
		check(q == s.num_motors * 2,
		      f"mode={mode:<14} decode=antagonist -> quantum {q} (want {s.num_motors * 2})")


def test_cumulative_gets_the_plain_quantum():
	print("\n[2] cumulative => quantum num_motors (BINARY is refused downstream, skip)")
	for mode in MODES:
		if mode == "BINARY":
			continue  # cumulative is refused for BINARY — its untrained bank floors
		s = spec(mode=mode, decode="cumulative")
		q = arch_shape_from_spec(s).output_quantum
		check(q == s.num_motors, f"mode={mode:<14} decode=cumulative -> quantum {q} (want {s.num_motors})")


def test_mode_defaults_reproduce_prior_cohorts():
	"""No --output-decode: BINARY must still resolve antagonist/8 and everything else
	cumulative/4, or every cohort measured before 03/08 stops reproducing."""
	print("\n[3] decode omitted => the mode's historical default (prior cohorts reproduce)")
	s = spec(mode="BINARY", decode=None)
	check(s.resolved_output_decode() == "antagonist", "BINARY default resolves antagonist")
	check(arch_shape_from_spec(s).output_quantum == 8, "BINARY default quantum 8")
	for mode in MODES:
		if mode == "BINARY":
			continue
		s = spec(mode=mode, decode=None)
		check(s.resolved_output_decode() == "cumulative", f"{mode:<14} default resolves cumulative")
		check(arch_shape_from_spec(s).output_quantum == 4, f"{mode:<14} default quantum 4")


def test_quantum_keeps_levels_even_under_antagonist():
	"""The POINT of the quantum: every output_neurons the GA can reach by stepping
	the quantum must decode to an even levels_per_motor. Walk the reachable ladder."""
	print("\n[4] every quantum step under antagonist yields EVEN levels_per_motor")
	s = spec(mode="QUAD_WEIGHTED", decode="antagonist")
	q = arch_shape_from_spec(s).output_quantum
	bad = [on for on in range(q, 256 + 1, q) if (on // s.num_motors) % 2 != 0]
	check(not bad, f"quantum={q}: no odd levels in on=[{q}..256] (offenders: {bad[:6]})")
	# And the failure the old quantum allowed, kept explicit so the regression is legible.
	check(92 % 4 == 0 and (92 // 4) % 2 != 0,
	      "on=92 (a multiple of the OLD q=4) gives levels=23 — odd, the shape that broke P4a")
	check(92 % q != 0, f"on=92 is NOT reachable under the fixed quantum {q}")


def test_input_window_k_is_inert_without_a_state_layer():
	"""P3's void corner. K widens ONLY the state layer's input pool; the output layer
	always samples one frame. At sn=0 there is no state layer, so K changes nothing —
	which is why P3_k8 reproduced A4 bit-identically, held-out included."""
	print("\n[5] input_window_k reaches the policy ONLY through a state layer")
	# NOTE the shape alone is the WRONG probe: state_input_space scales with k even at
	# sn=0, it is simply never consumed (no state neurons to address it). The property
	# that actually decides behaviour is the CONNECTIONS the search generates.
	c4, d4 = arch_shape_from_spec(spec(sn=8, k=4)), arch_shape_from_spec(spec(sn=8, k=8))
	check(d4.state_input_space == 2 * c4.state_input_space,
	      f"sn=8: doubling k doubles state_input_space ({c4.state_input_space} -> {d4.state_input_space})")
	check(c4.output_input_space == d4.output_input_space,
	      f"output_input_space is k-INVARIANT at {c4.output_input_space} (one frame, always)")

	s0_4 = random_connectivity(spec(sn=0, k=4), seed=7)
	s0_8 = random_connectivity(spec(sn=0, k=8), seed=7)
	check(s0_4 == s0_8,
	      "sn=0: k=4 and k=8 generate IDENTICAL connections — K is inert, which is why "
	      "P3_k8 reproduced A4 bit-identically (held-out included)")
	check(s0_4[0] == [], "sn=0: no state connections exist at all")
	s8_4 = random_connectivity(spec(sn=8, k=4), seed=7)
	s8_8 = random_connectivity(spec(sn=8, k=8), seed=7)
	check(s8_4 != s8_8, "sn=8: k=4 and k=8 generate DIFFERENT connections — K is a real axis")
	check(s8_4[1] != [] and s0_4[1] != [], "output connections exist in both cases")


def main() -> int:
	print("=" * 72)
	print("  arch_shape_from_spec invariants (quantum-vs-decode + K-vs-state-layer)")
	print("=" * 72)
	test_antagonist_always_gets_the_even_quantum()
	test_cumulative_gets_the_plain_quantum()
	test_mode_defaults_reproduce_prior_cohorts()
	test_quantum_keeps_levels_even_under_antagonist()
	test_input_window_k_is_inert_without_a_state_layer()
	print("\n" + "=" * 72)
	if _failures:
		print(f"  FAILED — {len(_failures)} assertion(s):")
		for f in _failures:
			print(f"    - {f}")
		return 1
	print("  ALL INVARIANTS HOLD")
	return 0


if __name__ == "__main__":
	sys.exit(main())
