"""R14a — regression tests for the `max_cells` per-genome cell budget.

Covers the gate that was documented in `RecurrentArchConfig` but DEAD until
75e50fbd, plus the R9 fix in 145b867b (per-suffix re-check on the mutated
intermediate).

What the budget IS: a clamp on STRUCTURAL grows only. `_mutate_neurons` /
`_mutate_bits` refuse to widen a genome that already carries `max_cells`
Lamarckian cells. It is NOT a cap on the cells a genome may end up holding —
training writes one cell per distinct visited input pattern and are not gated at
all, which is why the dfa1l campaign measured individual genomes at 211k against
a 180k budget (+17%). See memory `project_dfa1l_quad_oom_saga`.

Run:  /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python tests/test_max_cells_budget.py
"""

from __future__ import annotations

import numpy as np

from wnn.control.recurrent_genome import (
	RecurrentArchGenome, RecurrentArchShape, RecurrentArchConfig, MemoryPayload,
)
from wnn.ram.strategies.optimization_dimension import OptimizationDimension as PhaseType


SHAPE = RecurrentArchShape(prefix_factor=1, state_input_space=288,
                           output_input_space=72, output_quantum=4)

# Budget so large it can never bind — the "gate absent" control.
UNBOUNDED = 1_000_000_000


def _mk(rng, state_neurons=4, levels=2, ssuf=6, osuf=6) -> RecurrentArchGenome:
	return RecurrentArchGenome.random(
		SHAPE, state_neurons=state_neurons, output_neurons=levels * SHAPE.output_quantum,
		state_suffix=ssuf, output_suffix=osuf, rng=rng)


def _with_n_cells(g: RecurrentArchGenome, n_state: int, n_output: int,
                  ssuf: int, osuf: int) -> RecurrentArchGenome:
	"""Attach exactly n_state + n_output cells at addresses valid for the
	genome's suffix widths (address must be < 2^suffix on its layer)."""
	g = g.clone()
	su = [(i % g.state_neurons, i % (1 << ssuf)) for i in range(n_state)]
	ou = [(i % g.output_neurons, i % (1 << osuf)) for i in range(n_output)]
	# de-dup: (neuron, address) pairs must be unique or the payload collapses them
	su = sorted(set(su))[:n_state]
	ou = sorted(set(ou))[:n_output]
	g.cells = MemoryPayload(su, ou, [2] * len(su), [1] * len(ou))
	return g


def _bits_cfg(max_cells: int, suffix_delta: int = 2) -> RecurrentArchConfig:
	return RecurrentArchConfig(min_suffix=1, max_suffix=24, suffix_delta=suffix_delta,
	                           state_neuron_delta=0, output_block_delta=0,
	                           max_cells=max_cells)


def _neurons_cfg(max_cells: int) -> RecurrentArchConfig:
	return RecurrentArchConfig(min_state_neurons=2, max_state_neurons=12,
	                           min_output_neurons=4, max_output_neurons=64,
	                           state_neuron_delta=2, output_block_delta=2,
	                           suffix_delta=0, max_cells=max_cells)


# ===========================================================================
# The gate itself
# ===========================================================================

def test_gate_exempts_cellless_genome():
	"""Paradigm-A genomes (cells trained at eval, none carried) can't be
	measured, so they are exempt however small the budget."""
	rng = np.random.default_rng(0)
	g = _mk(rng)
	assert g.cells is None, "fresh random genome must carry no cells"
	assert not g._cells_at_budget(_bits_cfg(max_cells=1)), \
		"a cell-less genome must never be reported at budget"
	print("✓ gate_exempts_cellless_genome")


def test_gate_fires_at_and_above_budget():
	"""Boundary is >=, not >: a genome exactly at budget is at budget."""
	rng = np.random.default_rng(1)
	g = _with_n_cells(_mk(rng), n_state=20, n_output=10, ssuf=6, osuf=6)
	count = g.cells.cell_count()
	assert count == 30, f"expected 30 carried cells, got {count}"
	assert not g._cells_at_budget(_bits_cfg(max_cells=31)), "below budget must be free"
	assert g._cells_at_budget(_bits_cfg(max_cells=30)), "exactly at budget must fire (>=)"
	assert g._cells_at_budget(_bits_cfg(max_cells=29)), "above budget must fire"
	print("✓ gate_fires_at_and_above_budget")


# ===========================================================================
# Grows suppressed, shrinks preserved
# ===========================================================================

def test_bits_grow_suppressed_at_budget():
	"""At budget the STATE suffix may never widen — its gate reads the unmutated
	clone, so being at budget on entry is decisive. Shrinks must STILL happen, or
	selection loses its only way to slim an over-grown genome.

	The OUTPUT suffix is deliberately NOT asserted here: R9's re-check reads the
	mutated intermediate, so a state SHRINK that frees room legitimately re-opens
	the output grow. That behavior gets its own test below.
	"""
	rng = np.random.default_rng(2)
	g = _with_n_cells(_mk(rng, ssuf=8, osuf=8), n_state=20, n_output=10, ssuf=8, osuf=8)
	cfg = _bits_cfg(max_cells=g.cells.cell_count())   # exactly at budget
	saw_shrink = False
	cur = g
	for _ in range(200):
		at_budget = cur._cells_at_budget(cfg)
		m = cur._mutate_bits(rate=1.0, config=cfg, rng=rng)
		if at_budget:
			assert m.state_suffix_width <= cur.state_suffix_width, \
				f"state suffix grew at budget: {cur.state_suffix_width} -> {m.state_suffix_width}"
		if (m.state_suffix_width < cur.state_suffix_width
				or m.output_suffix_width < cur.output_suffix_width):
			saw_shrink = True
		m.assert_valid()
		cur = m
	assert saw_shrink, "budget must clamp grows only — shrinks must still occur"
	print("✓ bits_grow_suppressed_at_budget")


def test_state_shrink_reopens_output_grow():
	"""The other half of R9. Reading the MUTATED intermediate is bidirectional:
	a state-suffix shrink majority-collapses state cells, and if that drops the
	genome below budget the output grow is correctly re-enabled. A precomputed
	at_budget flag would wrongly keep it shut for the whole call.

	Asserted precisely: the output suffix widens ONLY when the count after the
	state step is under budget. The state step never touches output cells, so the
	intermediate count is (state cells after) + (output cells before).
	"""
	rng = np.random.default_rng(8)
	g = _with_n_cells(_mk(rng, ssuf=8, osuf=8), n_state=20, n_output=10, ssuf=8, osuf=8)
	cfg = _bits_cfg(max_cells=g.cells.cell_count())   # exactly at budget
	saw_reopened = 0
	cur = g
	for i in range(300):
		entered_at_budget = cur._cells_at_budget(cfg)
		s_before, o_before = cur.cells.counts()
		m = cur._mutate_bits(rate=1.0, config=cfg, rng=rng)
		s_after, _ = m.cells.counts()
		intermediate = s_after + o_before        # count the output gate actually saw
		if m.output_suffix_width > cur.output_suffix_width:
			assert intermediate < cfg.max_cells, (
				f"iter {i}: output suffix grew {cur.output_suffix_width} -> "
				f"{m.output_suffix_width} while the intermediate held {intermediate} "
				f">= budget {cfg.max_cells} (R9 regression: stale gate)")
			# Only when the call STARTED at budget does the grow prove a re-open;
			# the loop walks the genome, so later iterations may enter below it.
			if entered_at_budget:
				assert m.state_suffix_width < cur.state_suffix_width, (
					f"iter {i}: entered at budget and the output grew, but the state "
					f"suffix did not shrink to free the room — stale gate")
				saw_reopened += 1
		m.assert_valid()
		cur = m
	assert saw_reopened > 0, "fixture never exercised the shrink-reopens-grow path"
	print(f"✓ state_shrink_reopens_output_grow ({saw_reopened} cases)")


def test_neurons_grow_suppressed_at_budget():
	"""Same for neurogenesis: neither layer's neuron count may rise at budget."""
	rng = np.random.default_rng(3)
	g = _with_n_cells(_mk(rng, state_neurons=6, levels=4), n_state=20, n_output=10,
	                  ssuf=6, osuf=6)
	cfg = _neurons_cfg(max_cells=g.cells.cell_count())
	saw_shrink = False
	cur = g
	saw_at_budget = 0
	for _ in range(200):
		# _mutate_neurons keeps ONE shared gate read from the unmutated clone (R9
		# was scoped to _mutate_bits: appending EMPTY neuron blocks causes no
		# immediate cell jump, so a mid-call re-check would buy nothing). Both
		# layers therefore obey the ENTRY status — but only when we entered at
		# budget; the walked genome drifts below it after a shrink.
		if cur._cells_at_budget(cfg):
			saw_at_budget += 1
			m = cur._mutate_neurons(rate=1.0, config=cfg, rng=rng)
			assert m.state_neurons <= cur.state_neurons, \
				f"state neurons grew at budget: {cur.state_neurons} -> {m.state_neurons}"
			assert m.output_neurons <= cur.output_neurons, \
				f"output neurons grew at budget: {cur.output_neurons} -> {m.output_neurons}"
		else:
			m = cur._mutate_neurons(rate=1.0, config=cfg, rng=rng)
		if m.state_neurons < cur.state_neurons or m.output_neurons < cur.output_neurons:
			saw_shrink = True
		m.assert_valid()
		cur = m
	assert saw_at_budget > 0, "fixture never exercised an at-budget neurogenesis call"
	assert saw_shrink, "budget must clamp grows only — shrinks must still occur"
	print("✓ neurons_grow_suppressed_at_budget")


# ===========================================================================
# R9 — the per-suffix re-check
# ===========================================================================

def test_state_grow_reaching_budget_blocks_output_grow():
	"""R9. A single pre-computed at_budget let a state-suffix grow that REACHED
	the budget still permit the output-suffix grow in the SAME call — a one-step
	×2^suffix_delta overshoot on a second layer. The re-check reads the mutated
	intermediate, so the second grow sees the first grow's cells.

	Setup: start BELOW budget so the state gate opens, but close enough that the
	state grow's replicate-on-grow (×2^d) crosses it.
	"""
	rng = np.random.default_rng(4)
	base = _with_n_cells(_mk(rng, ssuf=6, osuf=6), n_state=20, n_output=10,
	                     ssuf=6, osuf=6)
	# 30 carried cells now; a state grow of d=1 replicates state cells -> 40+10=50.
	cfg = _bits_cfg(max_cells=35, suffix_delta=2)
	assert not base._cells_at_budget(cfg), "setup must start BELOW budget"

	checked = 0
	for seed in range(400):
		r = np.random.default_rng(1000 + seed)
		m = base._mutate_bits(rate=1.0, config=cfg, rng=r)
		grew_state = m.state_suffix_width > base.state_suffix_width
		if not grew_state:
			continue
		# cells after the state grow alone (output layer untouched by a state grow)
		s_after, o_after = m.cells.counts()
		if s_after + o_after < cfg.max_cells:
			continue                      # state grow did not reach budget; no claim
		checked += 1
		assert m.output_suffix_width <= base.output_suffix_width, (
			f"seed {seed}: state suffix grew to budget ({s_after + o_after} >= "
			f"{cfg.max_cells}) yet output suffix ALSO grew "
			f"{base.output_suffix_width} -> {m.output_suffix_width} (R9 regression)")
	assert checked > 0, "fixture never exercised the state-grow-reaches-budget path"
	print(f"✓ state_grow_reaching_budget_blocks_output_grow ({checked} cases)")


# ===========================================================================
# The gate must not perturb the deterministic RNG stream
# ===========================================================================

def test_budget_clamp_is_post_hoc_not_a_skipped_draw():
	"""`_cells_at_budget` reads cell_count only — no RNG draw — and the clamp is
	applied AFTER the delta is drawn. So a capped genome consumes exactly the
	draws an uncapped one does, and its width is the uncapped width clamped to
	"no grow". If the gate ever short-circuits the draw instead, the streams
	diverge and every downstream genome changes.

	Compare WIDTHS, not whole genomes: `set_*_suffix` draws extra bits from rng
	only when it actually grows, so the genomes legitimately differ.
	"""
	rng = np.random.default_rng(5)
	base = _with_n_cells(_mk(rng, ssuf=8, osuf=8), n_state=20, n_output=10,
	                     ssuf=8, osuf=8)
	free_cfg = _bits_cfg(max_cells=UNBOUNDED)
	cap_cfg = _bits_cfg(max_cells=base.cells.cell_count())   # at budget

	saw_clamped_grow = False
	for seed in range(300):
		free = base._mutate_bits(rate=1.0, config=free_cfg,
		                         rng=np.random.default_rng(2000 + seed))
		cap = base._mutate_bits(rate=1.0, config=cap_cfg,
		                        rng=np.random.default_rng(2000 + seed))
		# same seed -> same drawn deltas -> capped width is the free width with
		# grows clamped to the starting width.
		exp_s = min(free.state_suffix_width, base.state_suffix_width)
		exp_o = min(free.output_suffix_width, base.output_suffix_width)
		assert cap.state_suffix_width == exp_s, (
			f"seed {seed}: capped state width {cap.state_suffix_width} != clamped "
			f"free width {exp_s} — the gate perturbed the draw")
		assert cap.output_suffix_width == exp_o, (
			f"seed {seed}: capped output width {cap.output_suffix_width} != clamped "
			f"free width {exp_o} — the gate perturbed the draw")
		if free.state_suffix_width > base.state_suffix_width or \
				free.output_suffix_width > base.output_suffix_width:
			saw_clamped_grow = True
	assert saw_clamped_grow, "fixture never produced a grow for the clamp to bite on"
	print("✓ budget_clamp_is_post_hoc_not_a_skipped_draw")


def test_unbounded_budget_is_a_no_op():
	"""The default budget must leave behavior exactly as it was pre-75e50fbd."""
	rng = np.random.default_rng(6)
	base = _with_n_cells(_mk(rng, ssuf=8, osuf=8), n_state=20, n_output=10,
	                     ssuf=8, osuf=8)
	default_cfg = RecurrentArchConfig(min_suffix=1, max_suffix=24, suffix_delta=2,
	                                  state_neuron_delta=0, output_block_delta=0)
	explicit = _bits_cfg(max_cells=UNBOUNDED)
	assert default_cfg.max_cells == UNBOUNDED, \
		f"default max_cells must be effectively unbounded, got {default_cfg.max_cells}"
	for seed in range(100):
		a = base._mutate_bits(rate=1.0, config=default_cfg,
		                      rng=np.random.default_rng(3000 + seed))
		b = base._mutate_bits(rate=1.0, config=explicit,
		                      rng=np.random.default_rng(3000 + seed))
		assert a.state_suffix_width == b.state_suffix_width, f"seed {seed}: state width differs"
		assert a.output_suffix_width == b.output_suffix_width, f"seed {seed}: output width differs"
	print("✓ unbounded_budget_is_a_no_op")


# ===========================================================================
# TODO (yours) — the overshoot invariant
# ===========================================================================

def test_structural_overshoot_is_bounded():
	"""TODO — decide and assert what the budget actually PROMISES about cell
	count after a structural mutation.

	This is the judgment call the dfa1l campaign forced into the open, and it
	decides how we read every future cells[min-MAX] line:

	  (a) "cell_count never exceeds max_cells"        — FALSE as written; a grow
	      that starts below budget replicates ×2^delta and lands above it.
	  (b) "a genome AT budget never grows"            — what the code enforces
	      today; overshoot is bounded by one layer's ×2^suffix_delta.
	  (c) "overshoot <= max_cells * 2^suffix_delta"   — the numeric bound (b)
	      implies, assertable directly.

	`base` below starts just under budget with suffix_delta=2, so a state grow
	can multiply the state cells by up to 4. Write 5-10 lines that mutate it
	repeatedly and assert whichever bound we want to hold ourselves to.

	Why it matters: the campaign reported MAX-genome 211,168 against a 180,000
	budget. Under (b)/(c) that is expected and fine; under (a) it would read as
	a live bug and send someone hunting a leak that isn't there. Whichever we
	assert here becomes the documented contract.
	"""
	rng = np.random.default_rng(7)
	base = _with_n_cells(_mk(rng, ssuf=6, osuf=6), n_state=20, n_output=10,
	                     ssuf=6, osuf=6)
	cfg = _bits_cfg(max_cells=35, suffix_delta=2)
	assert not base._cells_at_budget(cfg), "setup must start BELOW budget"
	# YOUR ASSERTION HERE
	print("… structural_overshoot_is_bounded (not yet asserted)")


if __name__ == "__main__":
	test_gate_exempts_cellless_genome()
	test_gate_fires_at_and_above_budget()
	test_bits_grow_suppressed_at_budget()
	test_state_shrink_reopens_output_grow()
	test_neurons_grow_suppressed_at_budget()
	test_state_grow_reaching_budget_blocks_output_grow()
	test_budget_clamp_is_post_hoc_not_a_skipped_draw()
	test_unbounded_budget_is_a_no_op()
	test_structural_overshoot_is_bounded()
	print("\nAll max_cells budget-gate tests passed.")
