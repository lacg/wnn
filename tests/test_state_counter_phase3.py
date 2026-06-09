"""Phase 3a — Counter substrate validation (the increment cascade counts).

Design doc: .claude/plans/controller_state_splitting_design.md (§10 Phase 3, §9).

Phase 3's Type-2 arm represents an INTEGRAL as a thermometer counter of state
neurons. Before wiring detection/install into the trainer, validate the novel
substrate behavior in isolation (Phase-1 style): hand-install a depth-L counter
and prove it COUNTS error-fires through the recurrence.

Counter (L level-neurons, 1-bit state each, sbpn=3):
  trigger = error feature (gyro[1] -> frame bit 1)
  level 0 :  on = self_0 OR error                      (always increment-enabled)
  level k :  on = self_k OR (error AND level_{k-1})     (gated by the level below)
The gate reads the PREVIOUS step's lower level (recurrence) -> exactly one level
advances per error-fire -> the thermometer encodes the COUNT (saturating at L).
No leak here (Phase 3c); once on, a level holds.
"""

from ram_accelerator import WnnController

NUM_FEATURES = 9
BPF = 1
K = 1
ERR_IDX = 1                       # gyro[1] -> frame bit 1 (the error trigger)
SENSOR_WINDOW = K * NUM_FEATURES * BPF   # = 9
L = 3                             # counter depth -> counts 0..3
SBPN = 3

TRUE, WEAK_FALSE = 3, 1


def level_self_idx(k):
	return SENSOR_WINDOW + k          # state-MSB feedback bit for level k


def build_counter_controller():
	"""L level-neurons wired as a gated thermometer counter on the error bit."""
	thresholds = [1e9] * (NUM_FEATURES * BPF)
	thresholds[ERR_IDX] = 0.5

	# Per-level connections (sbpn=3), MSB-first: [trigger, lower, self].
	# level 0 has no lower -> point "lower" at the trigger itself, so its
	# increment-enable (trigger AND lower) reduces to trigger.
	conns = []
	for k in range(L):
		lower = ERR_IDX if k == 0 else level_self_idx(k - 1)
		conns += [ERR_IDX, lower, level_self_idx(k)]

	num_motors, levels, obpn = 4, 2, 1
	output_connections = [0] * (num_motors * levels * obpn)

	c = WnnController(
		num_motors=num_motors, levels_per_motor=levels,
		bits_per_feature=BPF, input_window_k=K,
		state_neurons=L, state_bits_per_neuron=SBPN, output_bits_per_neuron=obpn,
		thresholds=thresholds,
		state_connections=conns,
		output_connections=output_connections,
	)

	# Install each level's increment+hold truth table over its 2^SBPN addresses.
	for k in range(L):
		base = k * SBPN
		kconns = conns[base:base + SBPN]
		for a in range(1 << SBPN):
			bits = [(a >> (SBPN - 1 - j)) & 1 for j in range(SBPN)]  # MSB-first
			trig = bits[0]
			lower = bits[1]
			selfb = bits[2]
			if k == 0:
				on = selfb or trig
			else:
				on = selfb or (trig and lower)
			c.write_state_cell(k, a, TRUE if on else WEAK_FALSE)
	return c


def thermometer(c):
	"""Read the emitted MSB of each level after the last step -> [b0,b1,...]."""
	addrs = dict(c.last_state_addresses())            # neuron -> addr
	cells = {(n, a): v for (n, a, v) in c.export_cells()[0]}
	return [(cells[(k, addrs[k])] >> 1) & 1 for k in range(L)]


def drive(c, error):
	gyro = [0.0, (1.0 if error else 0.0), 0.0]
	c.step(gyro, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
	therm = thermometer(c)
	return therm, sum(therm)


def test_3a_counter():
	print("=" * 70)
	print("  PHASE 3a — counter substrate validation (increment cascade counts)")
	print("=" * 70)
	c = build_counter_controller()
	c.reset()

	# Drive a sequence of error-fires and idles; the thermometer must track the
	# running count (saturating at L), and HOLD across idle steps.
	plan = [
		("err",  1), ("err",  2), ("err",  3), ("err",  3),   # count up, saturate at L=3
		("idle", 3), ("idle", 3),                              # hold across idle
		("err",  3),                                           # already saturated
	]
	print("\n  step | trig | thermometer | count | expect")
	fails = []
	for i, (kind, expect) in enumerate(plan):
		therm, count = drive(c, error=(kind == "err"))
		mark = "" if count == expect else "  <-- UNEXPECTED"
		print(f"   {i:3d} | {kind:4s} | {therm} |   {count}   |   {expect}{mark}")
		if count != expect:
			fails.append((i, kind, therm, count, expect))
		# thermometer must be monotone (unary): no gaps like [1,0,1]
		if therm != sorted(therm, reverse=True):
			fails.append((i, "non-unary", therm, count, expect))

	ok = not fails
	print("\n" + "-" * 70)
	if ok:
		print("  PHASE 3a PASS — the cascade counts error-fires through the recurrence,")
		print("  saturates at depth L, and holds across idle steps (unary thermometer).")
		print("  Proceed to Phase 3b (Type-2 detection + counter install + resolve).")
	else:
		print("  PHASE 3a FAIL")
		for f in fails:
			print("   -", f)
	return ok


if __name__ == "__main__":
	raise SystemExit(0 if test_3a_counter() else 1)
