#!/usr/bin/env python3
"""--max-cells-strict: make the cell budget behave like a ceiling, not a threshold.

WHY: max_cells suppresses grows only once a genome is ALREADY at/over budget. A
genome just under it may still take a legal bits-grow, and that grow replicates
the grown layer x2^delta. With suffix_delta=2 that is x4, so a genome at ~145k
lands at ~580k against a 180k budget — measured 579,115 (3.22x) on the dfa1l
study cell 1layer_10feat_QUAD_s31337002. All 8 cells that overshot were QUAD and
no BINARY cell tripped 180k at all, so the granularity ablation was NOT
budget-matched: QUAD got up to 3x the memory and still lost every pair.

The gate is exercised through the real _grow_within_budget() on a real
RecurrentArchConfig, with a stub cells object supplying counts() — that is the
only method the projection reads, and stubbing it keeps the test free of a
controller wheel while still driving the production code path.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from wnn.control.recurrent_genome import RecurrentArchConfig, RecurrentArchGenome


class _StubCells:
	"""Minimal stand-in: the projection reads counts() and nothing else."""
	def __init__(self, state_cells, output_cells):
		self._s, self._o = state_cells, output_cells

	def counts(self):
		return (self._s, self._o)

	def cell_count(self):
		return self._s + self._o


def _genome(state_cells, output_cells):
	g = RecurrentArchGenome.__new__(RecurrentArchGenome)
	g.cells = _StubCells(state_cells, output_cells)
	return g


def _cfg(max_cells, strict):
	c = RecurrentArchConfig()
	c.max_cells = max_cells
	c.strict_cell_budget = strict
	return c


BUDGET = 180_000
FAILS = []


def check(label, got, expect):
	ok = got == expect
	print(f"  {'ok  ' if ok else 'FAIL'} {label:<58} -> {got} (expected {expect})")
	if not ok:
		FAILS.append(label)


def projected(state_cells, output_cells, delta, state_layer):
	return ((state_cells << delta) + output_cells if state_layer
	        else state_cells + (output_cells << delta))


print("=== DEFAULT (strict=False): historical behaviour, overshoot PRESERVED ===")
off = _cfg(BUDGET, False)
# The real incident shape: under budget, so the grow is legal and lands at 4x.
g = _genome(140_000, 5_000)
check("under budget, delta=+2 passes through unclamped", g._grow_within_budget(off, 2, True), 2)
check("  ...and that projects ABOVE budget (the bug)",
      projected(140_000, 5_000, 2, True) > BUDGET, True)
check("shrink untouched", g._grow_within_budget(off, -2, True), -2)
check("zero untouched", g._grow_within_budget(off, 0, True), 0)

print("\n=== STRICT (strict=True): grow clamped so POST-grow still fits ===")
on = _cfg(BUDGET, True)
g = _genome(140_000, 5_000)
# 140k<<1 = 280k (+5k) > 180k ; 140k<<0 = no grow -> 0
check("140k state, delta=+2 -> no grow fits", g._grow_within_budget(on, 2, True), 0)
check("  ...projection at the returned delta is within budget",
      projected(140_000, 5_000, 0, True) <= BUDGET, True)

g = _genome(80_000, 5_000)
# 80k<<1 = 160k +5k = 165k <= 180k ; 80k<<2 = 320k > 180k -> clamp 2 to 1
check("80k state, delta=+2 -> clamped to +1", g._grow_within_budget(on, 2, True), 1)
check("  ...projection within budget", projected(80_000, 5_000, 1, True) <= BUDGET, True)

g = _genome(20_000, 5_000)
# 20k<<2 = 80k +5k = 85k <= 180k -> full grow allowed
check("20k state, delta=+2 -> full grow allowed", g._grow_within_budget(on, 2, True), 2)

# Output layer uses the other side of the projection.
g = _genome(5_000, 80_000)
check("output layer, delta=+2 -> clamped to +1", g._grow_within_budget(on, 2, False), 1)
check("  ...projection within budget", projected(5_000, 80_000, 1, False) <= BUDGET, True)

g = _genome(5_000, 140_000)
check("output layer, 140k -> no grow fits", g._grow_within_budget(on, 2, False), 0)

print("\n=== INVARIANT: strict never returns a delta that overshoots ===")
bad = 0
for s in (1_000, 20_000, 80_000, 140_000, 179_000, 300_000):
	for o in (1_000, 20_000, 80_000, 140_000):
		for d in (1, 2, 3):
			for layer in (True, False):
				got = _genome(s, o)._grow_within_budget(on, d, layer)
				if got > 0 and projected(s, o, got, layer) > BUDGET:
					bad += 1
					print(f"    overshoot: s={s} o={o} d={d} layer={layer} -> {got}")
				if got > d:
					bad += 1
					print(f"    grew MORE than asked: s={s} o={o} d={d} -> {got}")
check("no (state,output,delta,layer) combo overshoots or over-grows", bad, 0)

print("\n=== shrinks and no-cells genomes are exempt in BOTH modes ===")
for mode, cfg in (("off", off), ("strict", on)):
	g = _genome(179_000, 179_000)
	check(f"[{mode}] shrink -1 passes through", g._grow_within_budget(cfg, -1, True), -1)
	gn = RecurrentArchGenome.__new__(RecurrentArchGenome)
	gn.cells = None
	check(f"[{mode}] no carried cells -> delta unchanged", gn._grow_within_budget(cfg, 3, True), 3)

print()
if FAILS:
	print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
	sys.exit(1)
print("ALL PASS — strict clamps the grow; default preserves the historical overshoot")
