"""Phase 4 — the multi-round consistency loop (k(e) greedy->batch).

Design doc: .claude/plans/controller_state_splitting_design.md (§7, §10 Phase 4).

Phases 2-3 each resolved ONE conflict per call. Phase 4 wraps that into a loop
that bootstraps from memoryless and keeps splitting until convergence, with a
k(round) schedule (greedy early -> batch late) and a `used`-neuron guard (the
collision rule: one distinction per neuron).

Test task — TWO independent delayed cues:
  cue A at step 0 (gyro[0]),  cue B at step 1 (gyro[1]),  decision at step 3.
  The decision-frame PWM depends on BOTH cues -> a 4-way conflict that a single
  latch cannot resolve. The loop must discover cue A (round 0), then cue B
  (round 1), planting a separate latch each, and converge with the controller
  output taking 4 distinct values.
"""

from wnn.control._accel import WnnController

NUM_FEATURES = 9
BPF = 1
K = 1
CUE_A = 0          # gyro[0] -> frame bit 0
CUE_B = 1          # gyro[1] -> frame bit 1
DECISION = 6       # target[0] -> frame bit 6
SENSOR = K * NUM_FEATURES * BPF          # 9
SELF0, SELF1 = SENSOR + 0, SENSOR + 1    # state-MSB feedback bits (9, 10)
L = 4              # episode length; decision at step 3
COPIES = 3         # episodes per (A,B) combo -> balanced QSR evidence for output

PWM = {(0, 0): 0.05, (0, 1): 0.35, (1, 0): 0.65, (1, 1): 0.95}


def build_controller():
	thresholds = [1e9] * (NUM_FEATURES * BPF)
	for b in (CUE_A, CUE_B, DECISION):
		thresholds[b] = 0.5
	# neuron 0 latches cue A {cueA, self0}; neuron 1 latches cue B {cueB, self1}
	state_connections = [CUE_A, SELF0, CUE_B, SELF1]
	num_motors, levels, obpn = 4, 8, 3
	# output observes [state0, state1, decision] -> count-combo -> PWM
	output_connections = [SENSOR + 0, SENSOR + 1, DECISION] * (num_motors * levels)
	return WnnController(
		num_motors=num_motors, levels_per_motor=levels,
		bits_per_feature=BPF, input_window_k=K,
		state_neurons=2, state_bits_per_neuron=2, output_bits_per_neuron=obpn,
		thresholds=thresholds,
		state_connections=state_connections,
		output_connections=output_connections,
	)


def make_episodes():
	gyros, accels, targets, pids = [], [], [], []
	for (ca, cb), p in PWM.items():
		for _ in range(COPIES):
			g = [[0.0, 0.0, 0.0] for _ in range(L)]
			a = [[0.0, 0.0, 0.0] for _ in range(L)]
			tg = [[0.0, 0.0, 0.0] for _ in range(L)]
			pw = [[0.5] * 4 for _ in range(L)]
			if ca:
				g[0] = [1.0, 0.0, 0.0]   # cue A at step 0
			if cb:
				g[1] = [0.0, 1.0, 0.0]   # cue B at step 1
			tg[L - 1] = [1.0, 0.0, 0.0]  # decision marker at step 3
			pw[L - 1] = [p] * 4
			gyros.append(g); accels.append(a); targets.append(tg); pids.append(pw)
	return gyros, accels, targets, pids


def out_for(c, ca, cb):
	c.reset()
	last = None
	for t in range(L):
		g = [0.0, 0.0, 0.0]
		if t == 0 and ca: g = [1.0, 0.0, 0.0]
		if t == 1 and cb: g = [0.0, 1.0, 0.0]
		tg = [1.0, 0.0, 0.0] if t == L - 1 else [0.0, 0.0, 0.0]
		last = c.step(g, [0.0, 0.0, 0.0], tg)
	return last[0]


def test_phase4_loop():
	print("=" * 70)
	print("  PHASE 4 — multi-round consistency loop (two cues -> two latches)")
	print("=" * 70)
	c = build_controller()
	g, a, tg, p = make_episodes()

	rounds, conflicts_final, planted, per_round, saturation, wishes = \
		c.split_train_loop(g, a, tg, p, 0.1, 0.999, 0.9, 8, 1)
	print(f"\n  rounds run        : {rounds}")
	print(f"  committed/round   : {per_round}")
	print(f"  distinctions planted: {planted}")
	print(f"  conflicts final   : {conflicts_final}")

	outs = {(ca, cb): out_for(c, ca, cb) for ca in (0, 1) for cb in (0, 1)}
	print("\n  controller output @ decision by (cueA,cueB):")
	for k in [(0, 0), (0, 1), (1, 0), (1, 1)]:
		print(f"    {k} -> {outs[k]:.3f}   (target {PWM[k]})")
	distinct = len(set(round(v, 3) for v in outs.values()))
	ordered = outs[(0, 0)] < outs[(0, 1)] < outs[(1, 0)] < outs[(1, 1)]

	ok = (
		conflicts_final == 0 and planted == 2 and rounds == 2 and
		distinct == 4 and ordered
	)
	print("\n" + "-" * 70)
	if ok:
		print("  PHASE 4 PASS — the loop bootstrapped from memoryless, discovered both")
		print("  cues over two rounds (one latch each, no neuron reuse), converged to")
		print("  zero conflicts; the controller output takes 4 distinct, ordered values.")
		print("  The state-splitting trainer is now a real multi-conflict trainer.")
	else:
		print("  PHASE 4 FAIL")
	assert ok, "test_state_loop_phase4 verdict was falsy"


if __name__ == "__main__":
	raise SystemExit(0 if test_phase4_loop() else 1)
