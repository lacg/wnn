#!/usr/bin/env python3
"""--grid-output-neurons and --input-window-k: the two P2/P3 search axes.

WHY these two flags exist, and what makes them easy to get wrong:

--grid-output-neurons adds a THIRD Stage-0 grid axis. output_neurons is not a
free capacity dial: evaluator.spec_from_arch derives
    levels_per_motor = output_neurons // num_motors
so the axis sweeps PWM DECODE RESOLUTION. Two traps follow.
  1. A non-multiple of the output quantum floor-divides away silently — the user
     asks for 70 and searches 68. So non-multiples must RAISE.
  2. The quantum is num_motors, DOUBLED under BINARY: the antagonist E/I decode
     needs an even levels_per_motor for a symmetric split (odd L drifts neutral
     off 0.5), so arch_shape_from_spec sets output_quantum = 2*num_motors there.
     Validating against num_motors alone would admit an odd level count that
     neurogenesis can never hold.

--input-window-k un-hardcodes _make_spec's `input_window_k=4`. It grows the input
POOL linearly (k * num_features * bits_per_feature) but leaves the ADDRESS SPACE
2^(prefix+suffix) untouched, so the cost is sampling coverage, not memory. The
test pins that asymmetry: state_input_space scales with k, output_input_space
does not.

Both flags must be inert at their defaults — the dfa1l/l3dfeat cohorts already in
flight were measured without them, and a changed default would silently break
cross-cohort comparability. grid_search_parity.py covers the byte-identical grid;
this file pins the flag semantics.
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from wnn.control.phased_ga import (
	build_arg_parser, _make_spec, grid_output_neuron_axis, grid_point_count,
	apply_output_neuron_ceiling,
)
from wnn.control.evaluator import arch_shape_from_spec

FAILS = []


def check(label, got, expected):
	ok = got == expected
	print(f"  {'ok  ' if ok else 'FAIL'} {label:<58} -> {got} (expected {expected})")
	if not ok:
		FAILS.append(label)


def check_raises(label, fn):
	try:
		fn()
	except ValueError:
		print(f"  ok   {label:<58} -> ValueError")
		return
	print(f"  FAIL {label:<58} -> accepted (should have raised)")
	FAILS.append(label)


def _args(**kw):
	"""Parser defaults + overrides, so the test tracks the real CLI surface."""
	a = build_arg_parser().parse_args([])
	a._geometry_num_motors = 4
	for k, v in kw.items():
		setattr(a, k, v)
	return a


class _ArchCfg:
	def __init__(self, mx=256):
		self.max_output_neurons = mx
		self.min_output_neurons = 4


print("\n=== defaults are inert (in-flight cohorts stay comparable) ===")
d = _args()
check("--input-window-k default", d.input_window_k, 4)
check("--grid-output-neurons default", d.grid_output_neurons, None)
check("default axis = one point at num_motors*levels", grid_output_neuron_axis(d), [4 * d.levels])
check("default grid cardinality = len(sn)*len(bits)", grid_point_count(d),
      len(d.grid_state_neurons) * len(d.grid_bits))

print("\n=== --grid-output-neurons becomes a real third axis ===")
a = _args(grid_output_neurons=[64, 96, 128], grid_state_neurons=[8], grid_bits=[24, 30])
check("axis passes through", grid_output_neuron_axis(a), [64, 96, 128])
check("cardinality = 1*2*3", grid_point_count(a), 6)

print("\n=== output_neurons round-trips to levels_per_motor exactly ===")
for on, levels in ((64, 16), (96, 24), (128, 32)):
	s = _make_spec(8, on // 4, 24)
	check(f"on={on} -> levels_per_motor", s.levels_per_motor, levels)
	check(f"on={on} -> num_motors*levels", s.num_motors * s.levels_per_motor, on)

print("\n=== silent-truncation traps must RAISE, not floor-divide ===")
check_raises("70 is not a multiple of num_motors=4",
             lambda: grid_output_neuron_axis(_args(grid_output_neurons=[70])))
check_raises("0 is not positive",
             lambda: grid_output_neuron_axis(_args(grid_output_neurons=[0])))
check_raises("BINARY: 68 = 17 levels (odd) breaks the E/I split",
             lambda: grid_output_neuron_axis(_args(memory_mode="BINARY", grid_output_neurons=[68])))
check("BINARY: even level counts still accepted",
      grid_output_neuron_axis(_args(memory_mode="BINARY", grid_output_neurons=[64, 96])), [64, 96])

print("\n=== the ceiling refuses to silently clamp a grid axis ===")
c = _ArchCfg(256)
apply_output_neuron_ceiling(_args(grid_output_neurons=[128]), c)
check("grid max 128 under ceiling 256 -> untouched", c.max_output_neurons, 256)
c = _ArchCfg(256)
apply_output_neuron_ceiling(_args(max_output_neurons=128), c)
check("--max-output-neurons still lowers the ceiling", c.max_output_neurons, 128)
check_raises("grid max 512 above ceiling 256 -> refuse",
             lambda: apply_output_neuron_ceiling(_args(grid_output_neurons=[512]), _ArchCfg(256)))

print("\n=== --input-window-k grows the POOL, not the address space ===")
sh4 = arch_shape_from_spec(_make_spec(8, 16, 24, input_window_k=4))
sh8 = arch_shape_from_spec(_make_spec(8, 16, 24, input_window_k=8))
check("k reaches the spec", _make_spec(8, 16, 24, input_window_k=8).input_window_k, 8)
check("state_input_space is linear in k", sh8.state_input_space, 2 * sh4.state_input_space)
check("output_input_space is INDEPENDENT of k", sh8.output_input_space, sh4.output_input_space)
check("prefix_factor unchanged by k", sh8.prefix_factor, sh4.prefix_factor)

print("\n=== CLI parses both flags ===")
n = build_arg_parser().parse_args(
	["--input-window-k", "8", "--grid-output-neurons", "64", "96", "128"])
check("--input-window-k 8", n.input_window_k, 8)
check("--grid-output-neurons 64 96 128", n.grid_output_neurons, [64, 96, 128])

print()
if FAILS:
	print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
	sys.exit(1)
print("ALL PASS — defaults inert; both axes wired; truncation traps raise")
