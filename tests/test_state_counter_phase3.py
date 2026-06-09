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
MID = 0.5                          # neutral PWM at non-decision steps

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


# ===========================================================================
# Phase 3b — Type-2 detection + counter install + resolve (uses split_train)
# ===========================================================================
# A pure ACCUMULATOR conflict: at the decision frame (identical observation in
# every episode) the target PWM depends ONLY on how many times the error fired
# earlier — NOT on any single (bit, lag). The error-fire PATTERN varies across
# episodes with the same count, so no Type-1 separator exists; only the
# window-SUM of the error feature explains the PWM. The walk must fall through
# to Type-2 and install a counter.

DECISION_IDX = 6   # target[0] -> frame bit 6 (decision marker, same in all eps)


def build_counter_train_controller():
	"""state neurons 0..L-1 pre-wired as the counter chain on the error bit
	(what split_install_counter expects to find); output reads the top counter
	level + decision marker so it can map count -> PWM."""
	thresholds = [1e9] * (NUM_FEATURES * BPF)
	thresholds[ERR_IDX] = 0.5
	thresholds[DECISION_IDX] = 0.5

	conns = []
	for k in range(L):
		lower = ERR_IDX if k == 0 else level_self_idx(k - 1)
		conns += [ERR_IDX, lower, level_self_idx(k)]

	num_motors, levels, obpn = 4, 8, 4   # 8 levels so 4 distinct count-PWMs render
	# output-layer input = [frame(9) | state(L)]; state level k is at index 9+k.
	# every output neuron observes [level2, level1, level0, decision] — the whole
	# counter PLUS the decision marker (so it outputs the count-dependent PWM at
	# the decision frame and MID elsewhere, instead of conflating them).
	st = NUM_FEATURES * BPF
	output_connections = [st + 2, st + 1, st + 0, DECISION_IDX] * (num_motors * levels)

	return WnnController(
		num_motors=num_motors, levels_per_motor=levels,
		bits_per_feature=BPF, input_window_k=K,
		state_neurons=L, state_bits_per_neuron=SBPN, output_bits_per_neuron=obpn,
		thresholds=thresholds,
		state_connections=conns,
		output_connections=output_connections,
	)


def make_accumulator_episodes():
	"""One episode per error-fire PATTERN. Counts 0..3 each realized by SEVERAL
	distinct patterns so no single (bit,lag) predicts the count. Target PWM is
	monotone in the count. Decision marker at the last step (same in all eps)."""
	W = 4              # error window: steps 0..3; decision at step W
	length = W + 1
	# patterns grouped by count -> multiple lag-patterns per count. BALANCED to 3
	# episodes/count so each class gets equal QSR evidence in the output retrain
	# (an under-trained class lands at weak confidence, not its target).
	patterns = {
		0: [(), (), ()],
		1: [(0,), (1,), (3,)],
		2: [(0, 1), (1, 2), (0, 3)],
		3: [(0, 1, 2), (1, 2, 3), (0, 2, 3)],
	}
	pwm_for = {0: 0.05, 1: 0.35, 2: 0.65, 3: 0.95}

	gyros, accels, targets, pids = [], [], [], []
	for count, plist in patterns.items():
		for fires in plist:
			g = [[0.0, 0.0, 0.0] for _ in range(length)]
			a = [[0.0, 0.0, 0.0] for _ in range(length)]
			tg = [[0.0, 0.0, 0.0] for _ in range(length)]
			p = [[MID, MID, MID, MID] for _ in range(length)]
			for s in fires:
				g[s] = [0.0, 1.0, 0.0]          # error fire at step s
			tg[W] = [1.0, 0.0, 0.0]             # decision marker (identical everywhere)
			p[W] = [pwm_for[count]] * 4         # target depends only on COUNT
			gyros.append(g); accels.append(a); targets.append(tg); pids.append(p)
	return gyros, accels, targets, pids, W


def test_3b_type2_resolve():
	print("\n" + "=" * 70)
	print("  PHASE 3b — Type-2 detection + counter install + resolve")
	print("=" * 70)
	c = build_counter_train_controller()
	g, a, tg, p, W = make_accumulator_episodes()

	(before, after, mode, bit, levels_used, score, up, n_planted) = \
		c.split_train(g, a, tg, p, 0.1, 0.999, 0.9)
	print(f"\n  conflicts: before={before}  after={after}  mode={mode} (2=Type-2 counter)")
	print(f"  accumulator: bit={bit} (err={ERR_IDX})  levels={levels_used}  |corr|={score:.3f}  up={up}")
	print(f"  state neurons planted (counter levels): {n_planted}")

	# end-to-end: the controller's output is now MONOTONE in the error count.
	def out_for(fires):
		c.reset()
		length = W + 1
		last = None
		for t in range(length):
			gy = [0.0, 1.0, 0.0] if t in fires else [0.0, 0.0, 0.0]
			tgt = [1.0, 0.0, 0.0] if t == W else [0.0, 0.0, 0.0]
			last = c.step(gy, [0.0, 0.0, 0.0], tgt)
		return last[0]

	outs = [out_for(f) for f in [(), (1,), (1, 2), (1, 2, 3)]]
	print(f"\n  controller output @ decision by count: "
	      f"0->{outs[0]:.3f}  1->{outs[1]:.3f}  2->{outs[2]:.3f}  3->{outs[3]:.3f}")
	monotone = all(outs[i] < outs[i + 1] for i in range(3))

	ok = (
		before == 1 and after == 0 and mode == 2 and
		bit == ERR_IDX and score > 0.9 and up and
		n_planted == L and monotone
	)
	print("\n" + "-" * 70)
	if ok:
		print("  PHASE 3b PASS — no single bit separated the conflict; the walk detected")
		print("  the ACCUMULATED count, installed an integral (thermometer counter), and")
		print("  the controller output is now monotone in the error count.")
		print("  Proceed to Phase 3c (leaky decrement / anti-windup).")
	else:
		print("  PHASE 3b FAIL")
	return ok


# ===========================================================================
# Phase 3c — bidirectional counter (anti-windup / unwind on error reversal)
# ===========================================================================
# The increment-only counter (3a/3b) is a SATURATING integrator: it can't
# unwind. Real anti-windup needs DECREMENT on error reversal. We add a second
# trigger (err_dn) and a decrement rule: a level turns OFF when err_dn fires AND
# it is the TOP active level (self on, level above off). The recurrence reads
# the prior step's neighbor, so exactly one level moves per step in EITHER
# direction — the precise unwind, no decay clock, no top-search.
#
# level k observes [err_up, err_dn, lower, self, upper] (sbpn=5):
#   on = 0  if (err_dn AND self AND NOT upper)     # decrement: I'm the top -> unwind
#        1  elif self                               # hold
#        1  elif (err_up AND lower)                 # increment
#        0  else
#   lower = err_up (k=0, proxy) or level k-1 self ; upper = level k+1 self
#   (k=top) or a constant-0 bit (so NOT upper = 1 -> top always unwinds).

ERR_UP = 1     # gyro[1] -> increment trigger
ERR_DN = 2     # gyro[2] -> decrement trigger
CONST0 = 3     # accel[0], threshold 1e9 -> always 0 (the top level's "upper")
SBPN_BI = 5


def build_bidirectional_controller():
	thresholds = [1e9] * (NUM_FEATURES * BPF)
	thresholds[ERR_UP] = 0.5
	thresholds[ERR_DN] = 0.5
	# CONST0 stays at 1e9 -> its frame bit is always 0

	conns = []
	for k in range(L):
		lower = ERR_UP if k == 0 else level_self_idx(k - 1)
		upper = CONST0 if k == L - 1 else level_self_idx(k + 1)
		conns += [ERR_UP, ERR_DN, lower, level_self_idx(k), upper]

	num_motors, levels, obpn = 4, 2, 1
	output_connections = [0] * (num_motors * levels * obpn)

	c = WnnController(
		num_motors=num_motors, levels_per_motor=levels,
		bits_per_feature=BPF, input_window_k=K,
		state_neurons=L, state_bits_per_neuron=SBPN_BI, output_bits_per_neuron=obpn,
		thresholds=thresholds,
		state_connections=conns,
		output_connections=output_connections,
	)
	for k in range(L):
		for a in range(1 << SBPN_BI):
			b = [(a >> (SBPN_BI - 1 - j)) & 1 for j in range(SBPN_BI)]  # [up,dn,lower,self,upper]
			up, dn, lower, selfb, upper = b
			if dn and selfb and not upper:
				on = 0
			elif selfb:
				on = 1
			elif up and lower:
				on = 1
			else:
				on = 0
			c.write_state_cell(k, a, TRUE if on else WEAK_FALSE)
	return c


def test_3c_bidirectional():
	print("\n" + "=" * 70)
	print("  PHASE 3c — bidirectional counter (unwind on error reversal)")
	print("=" * 70)
	c = build_bidirectional_controller()
	c.reset()

	def drive_bi(direction):
		gyro = [0.0, 0.0, 0.0]
		if direction == "up":
			gyro[ERR_UP] = 1.0
		elif direction == "dn":
			gyro[ERR_DN] = 1.0
		c.step(gyro, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
		return thermometer(c)

	plan = [("up", 1), ("up", 2), ("up", 3), ("up", 3),     # ramp up, saturate
	        ("dn", 2), ("dn", 1), ("dn", 0), ("dn", 0)]      # unwind, floor at 0
	print("\n  step | trig | thermometer | count | expect")
	fails = []
	for i, (d, expect) in enumerate(plan):
		therm = drive_bi(d)
		count = sum(therm)
		mark = "" if count == expect else "  <-- UNEXPECTED"
		print(f"   {i:3d} | {d:4s} | {therm} |   {count}   |   {expect}{mark}")
		if count != expect or therm != sorted(therm, reverse=True):
			fails.append((i, d, therm, count, expect))

	ok = not fails
	print("\n" + "-" * 70)
	if ok:
		print("  PHASE 3c PASS — the counter ramps up AND unwinds one level/step on")
		print("  reversal (precise anti-windup), staying unary throughout. The integral")
		print("  is now bidirectional. Proceed to Phase 4 (consistency loop, k(e)).")
	else:
		print("  PHASE 3c FAIL")
		for f in fails:
			print("   -", f)
	return ok


if __name__ == "__main__":
	ok = test_3a_counter()
	ok = test_3b_type2_resolve() and ok
	ok = test_3c_bidirectional() and ok
	raise SystemExit(0 if ok else 1)
