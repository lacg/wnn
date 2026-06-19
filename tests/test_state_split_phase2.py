"""Phase 2 — Conflict scan + discriminative backward walk (Type-1 event arm).

Design doc: .claude/plans/controller_state_splitting_design.md (§10 Phase 2).

Synthetic delayed-response task (a clean Type-1 EVENT conflict):
  - A CUE appears at episode step 0 (gyro[0] high) in the "cue" episode only.
  - A DECISION frame appears at step 5 (target[0] high), IDENTICAL in both episodes.
  - The correct PWM at the decision frame depends on the cue seen 5 steps earlier:
      cue   -> HIGH pwm (0.9)
      no-cue-> LOW  pwm (0.1)
  A MEMORYLESS controller sees the same decision frame in both -> same output ->
  cannot satisfy both -> a conflict only STATE can resolve. The walk must find
  that gyro[0] at lag 5 is the discriminator, plant a latch, and resolve it.

2a (this file, first): SCAN — exactly one conflict at the decision frame.
2b: WALK + PLANT + output-retrain + re-scan — the conflict resolves.
"""

from wnn.control._accel import WnnController

NUM_FEATURES = 9
BPF = 1
K = 1
CUE_IDX = 0       # gyro[0] -> frame bit 0  (the delayed cue)
DECISION_IDX = 6  # target[0] -> frame bit 6 (the decision marker, same in both eps)
SELFLOOP_IDX = K * NUM_FEATURES * BPF + 0   # state-MSB feedback for neuron 0 = 9

L = 6             # episode length; decision at step 5
HIGH, LOW, MID = 0.9, 0.1, 0.5


def build_controller():
	"""1 state neuron pre-wired to observe {cue, self-loop} (so 2b can latch);
	output neurons observe {state, cue} (so they can use a planted distinction)."""
	thresholds = [1e9] * (NUM_FEATURES * BPF)
	thresholds[CUE_IDX] = 0.5
	thresholds[DECISION_IDX] = 0.5

	# state neuron 0 observes [cue(MSB), self_loop(LSB)]
	state_connections = [CUE_IDX, SELFLOOP_IDX]

	num_motors, levels, obpn = 4, 2, 2
	state_bit_in_out = NUM_FEATURES * BPF + 0   # state bit index in output-layer input = 9
	# every output neuron observes [state_bit, cue]
	output_connections = [state_bit_in_out, CUE_IDX] * (num_motors * levels)

	return WnnController(
		num_motors=num_motors, levels_per_motor=levels,
		bits_per_feature=BPF, input_window_k=K,
		state_neurons=1, state_bits_per_neuron=2, output_bits_per_neuron=obpn,
		thresholds=thresholds,
		state_connections=state_connections,
		output_connections=output_connections,
	)


def make_episodes():
	"""Two episodes: cue and no-cue. Returns (gyros, accels, targets, pid_pwms)
	as episode-major lists (the shape split_scan/ split_train expect)."""
	def blank3(): return [0.0, 0.0, 0.0]
	def blank4(): return [MID, MID, MID, MID]

	gyros, accels, targets, pids = [], [], [], []
	for cue in (True, False):
		g = [blank3() for _ in range(L)]
		a = [blank3() for _ in range(L)]
		tg = [blank3() for _ in range(L)]
		p = [blank4() for _ in range(L)]
		if cue:
			g[0] = [1.0, 0.0, 0.0]          # cue at step 0
		tg[L - 1] = [1.0, 0.0, 0.0]         # decision marker at step 5 (both eps)
		p[L - 1] = [HIGH] * 4 if cue else [LOW] * 4
		gyros.append(g); accels.append(a); targets.append(tg); pids.append(p)
	return gyros, accels, targets, pids


def test_2a_scan():
	print("=" * 70)
	print("  PHASE 2a — conflict scan (one event conflict at the decision frame)")
	print("=" * 70)
	c = build_controller()
	g, a, tg, p = make_episodes()
	total, conflicts = c.split_scan(g, a, tg, p, 0.1)

	print(f"\n  records: {total}  (expect {2 * L})")
	print(f"  conflicts: {len(conflicts)}  (expect 1)")
	for spread, coords in conflicts:
		print(f"    spread={spread:.3f}  instances={coords}")

	ok = (total == 2 * L) and (len(conflicts) == 1)
	if ok:
		spread, coords = conflicts[0]
		coordset = set(map(tuple, coords))
		ok = abs(spread - (HIGH - LOW)) < 1e-5 and coordset == {(0, L - 1), (1, L - 1)}
	print("\n" + "-" * 70)
	print("  PHASE 2a PASS" if ok else "  PHASE 2a FAIL")
	return ok


def decode_pwm_at_decision(c, gyros, accels, targets, ep):
	"""Drive one episode via step() on the trained controller and return motor-0
	PWM at the decision step (end-to-end output check)."""
	c.reset()
	last = None
	for t in range(L):
		last = c.step(gyros[ep][t], accels[ep][t], targets[ep][t])
	return last[0]


def test_2b_walk_plant_resolve():
	print("\n" + "=" * 70)
	print("  PHASE 2b — discriminative walk + latch plant + resolve")
	print("=" * 70)
	c = build_controller()
	g, a, tg, p = make_episodes()

	(before, after, mode, sbit, slag, sgain, shigh, n_planted) = c.split_train(g, a, tg, p, 0.1, 0.999, 0.9)
	print(f"\n  conflicts: before={before}  after={after}  mode={mode} (1=Type-1 latch)")
	print(f"  separator: bit={sbit} (cue={CUE_IDX})  lag={slag}  gain={sgain:.3f}  high_on={shigh}")
	print(f"  state neurons planted: {n_planted}")

	# end-to-end: the controller now outputs HIGH for the cue history, LOW otherwise
	pwm_cue = decode_pwm_at_decision(c, g, a, tg, ep=0)
	pwm_nocue = decode_pwm_at_decision(c, g, a, tg, ep=1)
	print(f"\n  controller output @ decision:  cue={pwm_cue:.3f}  no-cue={pwm_nocue:.3f}"
	      f"  (delta={pwm_cue - pwm_nocue:+.3f})")

	ok = (
		before == 1 and after == 0 and mode == 1 and
		sbit == CUE_IDX and slag == L - 1 and sgain > 0.999 and shigh and
		n_planted == 1 and
		pwm_cue > pwm_nocue + 0.1
	)
	print("\n" + "-" * 70)
	if ok:
		print("  PHASE 2b PASS — walk found the delayed cue, planted a latch,")
		print("  resolved the conflict; the controller's output now diverges on history.")
		print("  Proceed to Phase 3 (Type-2 integral counter, the steady-state deficit).")
	else:
		print("  PHASE 2b FAIL")
	return ok


if __name__ == "__main__":
	ok = test_2a_scan()
	ok = test_2b_walk_plant_resolve() and ok
	raise SystemExit(0 if ok else 1)
