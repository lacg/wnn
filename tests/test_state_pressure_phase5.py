"""Phase 5a — the trainer emits GA-handshake pressure (design §8).

When split_train_loop cannot resolve a conflict, it asks the discriminative walk
over ALL frame bits what WOULD have separated it, and reports either:
  - CONNECTIVITY wish: a separating bit that no state neuron observes
    -> the GA should route a neuron to it.
  - SATURATION: a separating bit that IS observed but the trainer ran out of
    free/wired neurons -> the GA should grow state_neurons.

This is additive (touches nothing in the live GA loop). 5b wires the trainer
into the GA training path; 5c consumes these wishes in genome mutation.
"""

from ram_accelerator import WnnController

NF, BPF, K = 9, 1, 1
SENSOR = K * NF * BPF
DEC = 6


def _ctrl(state_neurons, sbpn, state_conns, thr_bits, num_motors=4, levels=2, obpn=2,
          out_conns=None):
	thresholds = [1e9] * (NF * BPF)
	for b in thr_bits:
		thresholds[b] = 0.5
	if out_conns is None:
		out_conns = [SENSOR + 0, DEC] * (num_motors * levels)
	return WnnController(
		num_motors=num_motors, levels_per_motor=levels,
		bits_per_feature=BPF, input_window_k=K,
		state_neurons=state_neurons, state_bits_per_neuron=sbpn, output_bits_per_neuron=obpn,
		thresholds=thresholds, state_connections=state_conns, output_connections=out_conns,
	)


def _delayed_cue_episodes(cue_bit, length=4):
	"""cue at step 0 (on the given frame feature), decision at the last step."""
	gyros, accels, targets, pids = [], [], [], []
	for cue in (True, False):
		g = [[0.0, 0.0, 0.0] for _ in range(length)]
		a = [[0.0, 0.0, 0.0] for _ in range(length)]
		tg = [[0.0, 0.0, 0.0] for _ in range(length)]
		p = [[0.5] * 4 for _ in range(length)]
		if cue:
			g[0] = [1.0, 0.0, 0.0]          # gyro[0] -> frame bit 0
		tg[length - 1] = [1.0, 0.0, 0.0]    # decision marker (frame bit 6)
		p[length - 1] = [0.9] * 4 if cue else [0.1] * 4
		gyros.append(g); accels.append(a); targets.append(tg); pids.append(p)
	return gyros, accels, targets, pids


def test_5a_connectivity_wish():
	print("=" * 70)
	print("  PHASE 5a (i) — connectivity wish (cue bit unobserved)")
	print("=" * 70)
	# the one state neuron observes bit 5 (an always-0 feature) + its self-loop,
	# NOT the cue (bit 0). So the trainer cannot resolve the delayed-cue conflict
	# and should WISH for bit 0.
	c = _ctrl(state_neurons=1, sbpn=2, state_conns=[5, SENSOR + 0], thr_bits=[0, DEC])
	g, a, tg, p = _delayed_cue_episodes(cue_bit=0)
	rounds, final, planted, per_round, saturation, wishes = \
		c.split_train_loop(g, a, tg, p, 0.1, 0.999, 0.9, 8, 1)
	print(f"\n  planted={planted}  conflicts_final={final}  saturation={saturation}  wishes={wishes}")
	ok = planted == 0 and final == 1 and saturation == 0 and wishes == [0]
	print("  -> wish for bit 0 (the cue)  " + ("OK" if ok else "FAIL"))
	return ok


def test_5a_saturation():
	print("\n" + "=" * 70)
	print("  PHASE 5a (ii) — saturation (separator observed, neurons exhausted)")
	print("=" * 70)
	# ONE state neuron observes BOTH cues (bit0, bit1) + self. A 4-way two-cue
	# task needs TWO distinctions but there is one neuron: the trainer plants cue A,
	# then cue B's separator is OBSERVED but no free neuron -> saturation.
	out = [SENSOR + 0, DEC] * (4 * 2)
	c = _ctrl(state_neurons=1, sbpn=3, state_conns=[0, 1, SENSOR + 0],
	          thr_bits=[0, 1, DEC], out_conns=out)
	# two-cue 4-way episodes (cue A @0, cue B @1, decision @3), 2 copies each
	gyros, accels, targets, pids = [], [], [], []
	PWM = {(0, 0): 0.05, (0, 1): 0.35, (1, 0): 0.65, (1, 1): 0.95}
	for (ca, cb), pv in PWM.items():
		for _ in range(2):
			gg = [[0.0, 0.0, 0.0] for _ in range(4)]
			aa = [[0.0, 0.0, 0.0] for _ in range(4)]
			tt = [[0.0, 0.0, 0.0] for _ in range(4)]
			pp = [[0.5] * 4 for _ in range(4)]
			if ca: gg[0] = [1.0, 0.0, 0.0]
			if cb: gg[1] = [0.0, 1.0, 0.0]
			tt[3] = [1.0, 0.0, 0.0]
			pp[3] = [pv] * 4
			gyros.append(gg); accels.append(aa); targets.append(tt); pids.append(pp)
	rounds, final, planted, per_round, saturation, wishes = \
		c.split_train_loop(gyros, accels, targets, pids, 0.1, 0.999, 0.9, 8, 1)
	print(f"\n  planted={planted}  conflicts_final={final}  saturation={saturation}  wishes={wishes}")
	ok = planted == 1 and final >= 1 and saturation >= 1 and wishes == []
	print("  -> planted one, remaining conflict is saturation (grow sn)  "
	      + ("OK" if ok else "FAIL"))
	return ok


if __name__ == "__main__":
	ok = test_5a_connectivity_wish()
	ok = test_5a_saturation() and ok
	print("\n" + "-" * 70)
	print("  PHASE 5a PASS — trainer emits connectivity + saturation pressure."
	      if ok else "  PHASE 5a FAIL")
	raise SystemExit(0 if ok else 1)
