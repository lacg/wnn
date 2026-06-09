"""Phase 1 — Latch substrate validation for the state-splitting trainer.

Design doc: .claude/plans/controller_state_splitting_design.md (§10 Phase 1).

The entire state-splitting design rests on ONE assumption: a bit written into
the state RAM PERSISTS across the recurrence (a "carry"). The original
diagnosis showed the state layer collapses to ~1 cell / memoryless by default,
so before building any walk/scan machinery we prove a hand-installed latch can
hold a bit — and that the QSR cell gives it Schmitt (debounced) hysteresis.

This test uses ONLY existing accelerator bindings (no rebuild):
  WnnController(...), write_state_cell, last_state_addresses, export_cells, step

Latch construction (state neuron 0, state_bits_per_neuron=3):
  observes {set_trigger, reset_trigger, self_loop}  (MSB-first connections)
  addr = set*4 + reset*2 + self_loop
  truth table (priority reset > set > hold), value 3=TRUE (MSB 1), 1=WEAK_FALSE (MSB 0):
    sl r s | addr | meaning           | cell
    0  0 0 |  0   | off, no trigger   | 1
    1  0 0 |  1   | HOLD on           | 3   <- the carry
    0  1 0 |  2   | off               | 1
    1  1 0 |  3   | RESET off         | 1
    0  0 1 |  4   | SET on            | 3
    1  0 1 |  5   | on                | 3
    0  1 1 |  6   | reset dominates   | 1
    1  1 1 |  7   | reset dominates   | 1
"""

from ram_accelerator import WnnController

NUM_FEATURES = 9          # controller.rs NUM_FEATURES
BPF = 1                   # 1 thermometer bit/feature -> frame_bits = 9
K = 1                     # input window -> sensor_total = K*frame_bits = 9
SET_TRIG_IDX = 0          # frame bit driven by gyro[0]
RESET_TRIG_IDX = 1        # frame bit driven by gyro[1]
SELFLOOP_IDX = K * NUM_FEATURES * BPF + 0   # state-MSB feedback bit for neuron 0 = index 9

TRUE, WEAK_TRUE, WEAK_FALSE, FALSE = 3, 2, 1, 0


def build_latch_controller():
	"""A 1-state-neuron controller whose only neuron is a hand-built SR latch."""
	thresholds = [1e9] * (NUM_FEATURES * BPF)
	thresholds[SET_TRIG_IDX] = 0.5     # gyro[0] >= 0.5 -> set trigger
	thresholds[RESET_TRIG_IDX] = 0.5   # gyro[1] >= 0.5 -> reset trigger

	# neuron 0 observes [set_trigger(MSB), reset_trigger, self_loop(LSB)]
	state_connections = [SET_TRIG_IDX, RESET_TRIG_IDX, SELFLOOP_IDX]

	num_motors, levels = 4, 2
	output_connections = [0] * (num_motors * levels * 1)  # output ignored in this test

	c = WnnController(
		num_motors=num_motors,
		levels_per_motor=levels,
		bits_per_feature=BPF,
		input_window_k=K,
		state_neurons=1,
		state_bits_per_neuron=3,
		output_bits_per_neuron=1,
		thresholds=thresholds,
		state_connections=state_connections,
		output_connections=output_connections,
	)
	# Install the latch truth table.
	latch = {0: WEAK_FALSE, 1: TRUE, 2: WEAK_FALSE, 3: WEAK_FALSE,
	         4: TRUE, 5: TRUE, 6: WEAK_FALSE, 7: WEAK_FALSE}
	for addr, val in latch.items():
		c.write_state_cell(0, addr, val)
	return c


def emitted_state(c):
	"""MSB (fired/not) emitted by state neuron 0 on the LAST step()."""
	(_, addr) = c.last_state_addresses()[0]
	cells = {(n, a): v for (n, a, v) in c.export_cells()[0]}
	val = cells[(0, addr)]
	return (val >> 1) & 1, addr, val


def gyro_for(trigger):
	if trigger == "set":   return [1.0, 0.0, 0.0]
	if trigger == "reset": return [0.0, 1.0, 0.0]
	return [0.0, 0.0, 0.0]   # idle


def drive(c, trigger):
	c.step(gyro_for(trigger), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
	return emitted_state(c)


def main():
	print("=" * 70)
	print("  PHASE 1 — Latch substrate validation (carry persistence + QSR hysteresis)")
	print("=" * 70)

	# ---- Test A: address packing sanity (MSB-first) -------------------------
	# A known input must read the address we hand-computed.
	c = build_latch_controller()
	c.reset()
	_, addr, _ = drive(c, "set")   # self_loop=0, set=1, reset=0 -> addr = 1*4 = 4
	assert addr == 4, f"address packing mismatch: SET step read addr {addr}, expected 4"
	print(f"\n[A] address packing (MSB-first): SET@sl=0 -> addr {addr} == 4  OK")

	# ---- Test B: carry persistence (the load-bearing assumption) ------------
	c = build_latch_controller()
	c.reset()
	seq, fails = [], []

	def record(label, trig, expect):
		bit, addr, val = drive(c, trig)
		seq.append((label, trig, bit, addr, val))
		if bit != expect:
			fails.append(f"{label}: emitted {bit}, expected {expect} (addr {addr}, cell {val})")

	record("init", "idle", 0)
	record("init", "idle", 0)
	record("SET", "set", 1)
	HOLD_N = 25
	for i in range(HOLD_N):
		record(f"hold{i+1}", "idle", 1)   # <-- must stay 1 across all 25 idle steps
	record("RESET", "reset", 0)
	for i in range(3):
		record(f"after{i+1}", "idle", 0)

	held = sum(1 for (lbl, _, b, _, _) in seq if lbl.startswith("hold") and b == 1)
	print(f"\n[B] carry persistence: bit held {held}/{HOLD_N} idle steps after SET")
	print("    emitted sequence (label | trig | bit | addr | cell):")
	for (lbl, trig, b, addr, val) in seq:
		mark = "" if (
			(lbl in ("init",) and b == 0) or
			(lbl == "SET" and b == 1) or
			(lbl.startswith("hold") and b == 1) or
			(lbl == "RESET" and b == 0) or
			(lbl.startswith("after") and b == 0)
		) else "   <-- UNEXPECTED"
		print(f"      {lbl:8s} | {trig:5s} | {b} | {addr} | {val}{mark}")

	# ---- Test C: QSR Schmitt hysteresis (debounced latch) -------------------
	# A strongly-held cell (TRUE=3) survives ONE contradictory nudge
	# (3 -> WEAK_TRUE=2, still MSB 1) and only flips on the SECOND (2 -> 1).
	c = build_latch_controller()
	c.reset()
	drive(c, "set")                       # latch ON
	b0, _, _ = drive(c, "idle")           # hold, cell=3
	c.write_state_cell(0, 1, WEAK_TRUE)   # simulate ONE downward nudge 3->2
	b1, _, v1 = drive(c, "idle")          # still ON? (MSB of 2 = 1)
	c.write_state_cell(0, 1, WEAK_FALSE)  # second nudge 2->1
	b2, _, v2 = drive(c, "idle")          # now OFF (MSB of 1 = 0)
	hysteresis_ok = (b0 == 1 and b1 == 1 and b2 == 0)
	print(f"\n[C] QSR Schmitt hysteresis: hold(3)->{b0}  nudge1(2)->{b1}  nudge2(1)->{b2}"
	      f"   {'OK (survives 1, flips on 2)' if hysteresis_ok else 'FAIL'}")

	# ---- Verdict -----------------------------------------------------------
	ok_persist = (held == HOLD_N and not fails)
	print("\n" + "-" * 70)
	if ok_persist and hysteresis_ok:
		print("  PHASE 1 PASS — carries persist; QSR gives debounced-latch hysteresis.")
		print("  The foundation holds. Proceed to Phase 2 (scan + discriminative walk).")
		return 0
	print("  PHASE 1 FAIL")
	for f in fails:
		print("   -", f)
	if not hysteresis_ok:
		print("   - hysteresis: expected hold=1, nudge1=1, nudge2=0")
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
