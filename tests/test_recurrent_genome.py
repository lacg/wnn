"""Structural-validity tests for the variable-shape RecurrentArchGenome.

These assert the FSM-coherence invariant holds BY CONSTRUCTION across every
operator (random / clone / mutate per phase / crossover) and that a materialized
genome round-trips into a real Rust WnnController and takes a forward step —
across a sweep of (state_neurons × levels × suffix-width) shapes.

Run:  PYTHONPATH=src/wnn python tests/test_recurrent_genome.py
"""

from __future__ import annotations

import numpy as np

from wnn.control.recurrent_genome import (
	RecurrentArchGenome, RecurrentArchShape, RecurrentArchConfig, MemoryPayload,
	_remap_grow, _remap_shrink, _remap_prefix_grow, _remap_prefix_shrink, _majority,
)
from wnn.ram.strategies.optimization_dimension import OptimizationDimension as PhaseType


# A drone-flavoured shape (num_motors=4 → output_quantum=4), but the genome
# itself stays domain-free. K=4, F=9, b=8 → state_input_space=288, frame=72.
SHAPE = RecurrentArchShape(prefix_factor=1, state_input_space=288,
                           output_input_space=72, output_quantum=4)
CONFIG = RecurrentArchConfig(min_state_neurons=2, max_state_neurons=12,
                             min_output_neurons=8, max_output_neurons=64,
                             min_suffix=1, max_suffix=24,
                             state_neuron_delta=2, output_block_delta=2, suffix_delta=3)


def _mk(rng, state_neurons=4, levels=4, ssuf=16, osuf=12) -> RecurrentArchGenome:
	return RecurrentArchGenome.random(
		SHAPE, state_neurons=state_neurons, output_neurons=levels * SHAPE.output_quantum,
		state_suffix=ssuf, output_suffix=osuf, rng=rng)


def _check_canonical_prefix(g: RecurrentArchGenome) -> None:
	"""Every neuron's materialized block must START with the canonical prefix."""
	sc, oc = g.to_connections()
	p = g.forced_prefix
	sbpn, obpn = g.state_bits_per_neuron, g.output_bits_per_neuron
	state_prefix = list(range(SHAPE.state_input_space, SHAPE.state_input_space + p))
	out_prefix = list(range(SHAPE.output_input_space, SHAPE.output_input_space + p))
	assert len(sc) == g.state_neurons * sbpn, "state_connections length mismatch"
	assert len(oc) == g.output_neurons * obpn, "output_connections length mismatch"
	for n in range(g.state_neurons):
		assert sc[n * sbpn:n * sbpn + p] == state_prefix, f"state neuron {n} prefix corrupted"
	for n in range(g.output_neurons):
		assert oc[n * obpn:n * obpn + p] == out_prefix, f"output neuron {n} prefix corrupted"


def test_random_valid():
	rng = np.random.default_rng(0)
	for sn in (2, 4, 7, 12):
		for lv in (2, 4, 8):
			for suf in (1, 8, 24):
				g = _mk(rng, state_neurons=sn, levels=lv, ssuf=suf, osuf=suf)
				g.assert_valid()
				_check_canonical_prefix(g)
				assert g.forced_prefix == sn
				assert g.output_neurons == lv * 4
	print("✓ random_valid")


def test_clone_is_deep_and_identical():
	rng = np.random.default_rng(1)
	g = _mk(rng)
	c = g.clone()
	assert c.fingerprint() == g.fingerprint()
	c.state_sampled[0][0] = -999  # mutate the copy
	assert g.state_sampled[0][0] != -999, "clone shares state list (not deep)"
	print("✓ clone_is_deep_and_identical")


def test_mutate_connections_preserves_shape():
	rng = np.random.default_rng(2)
	g = _mk(rng, state_neurons=5, levels=4, ssuf=16, osuf=12)
	m = g.mutate(PhaseType.CONNECTIONS, rate=1.0, config=CONFIG, rng=rng)
	m.assert_valid()
	_check_canonical_prefix(m)
	# CONNECTIONS must not touch counts or widths — only sampled values.
	assert (m.state_neurons, m.output_neurons) == (g.state_neurons, g.output_neurons)
	assert (m.state_suffix_width, m.output_suffix_width) == (g.state_suffix_width, g.output_suffix_width)
	assert m.fingerprint() != g.fingerprint(), "rate=1.0 should change some bits"
	print("✓ mutate_connections_preserves_shape")


def test_mutate_bits_uniform_and_floored():
	rng = np.random.default_rng(3)
	g = _mk(rng, state_neurons=4, levels=4, ssuf=8, osuf=8)
	for _ in range(50):
		g = g.mutate(PhaseType.BITS, rate=1.0, config=CONFIG, rng=rng)
		g.assert_valid()  # asserts uniform width + ≥1 floor + in-range
		_check_canonical_prefix(g)
		assert g.state_suffix_width <= min(CONFIG.max_suffix, SHAPE.state_input_space)
		assert g.output_suffix_width <= min(CONFIG.max_suffix, SHAPE.output_input_space)
	print("✓ mutate_bits_uniform_and_floored")


def test_mutate_neurons_global_reshape_and_survivor_preservation():
	rng = np.random.default_rng(4)
	g = _mk(rng, state_neurons=4, levels=4, ssuf=10, osuf=10)
	# Force a pure grow by clamping config to +delta only via repeated tries.
	grew = shrank = False
	for _ in range(200):
		m = g.mutate(PhaseType.NEURONS, rate=1.0, config=CONFIG, rng=rng)
		m.assert_valid()
		_check_canonical_prefix(m)
		assert m.forced_prefix == m.state_neurons  # prefix tracks neuron count
		assert m.output_neurons % SHAPE.output_quantum == 0
		# Survivors keep their suffix verbatim on growth (small-neighborhood rule).
		if m.state_neurons > g.state_neurons:
			grew = True
			keep = g.state_neurons
			assert [m.state_sampled[i] for i in range(keep)] == g.state_sampled, \
				"growth must preserve existing state neurons' suffixes"
		if m.state_neurons < g.state_neurons:
			shrank = True
		g = m
	assert grew and shrank, "sweep should exercise both grow and shrink"
	print("✓ mutate_neurons_global_reshape_and_survivor_preservation")


def test_crossover_different_shapes_valid():
	rng = np.random.default_rng(5)
	for _ in range(100):
		a = _mk(rng, state_neurons=int(rng.integers(2, 10)), levels=int(rng.integers(2, 8)),
		        ssuf=int(rng.integers(1, 20)), osuf=int(rng.integers(1, 20)))
		b = _mk(rng, state_neurons=int(rng.integers(2, 10)), levels=int(rng.integers(2, 8)),
		        ssuf=int(rng.integers(1, 20)), osuf=int(rng.integers(1, 20)))
		child = RecurrentArchGenome.crossover(a, b, rng)
		child.assert_valid()
		_check_canonical_prefix(child)
		# Child must adopt exactly one parent's shape.
		assert (child.state_neurons, child.output_neurons,
		        child.state_suffix_width, child.output_suffix_width) in {
			(a.state_neurons, a.output_neurons, a.state_suffix_width, a.output_suffix_width),
			(b.state_neurons, b.output_neurons, b.state_suffix_width, b.output_suffix_width),
		}
	print("✓ crossover_different_shapes_valid")


def test_wnn_controller_roundtrip():
	"""Materialized genome must build a real Rust WnnController and take a step,
	across a neuron × level × bit sweep. This is the ultimate validity proof:
	the Rust ctor rejects any connectivity whose length ≠ neurons × bits."""
	from wnn.control._accel import WnnController  # noqa: import here so the file imports without the accel
	from wnn.control.evaluator import (
		ControllerSpec, controller_genome_from_arch, build_controller, NUM_FEATURES,
	)
	rng = np.random.default_rng(6)
	base = ControllerSpec(num_motors=4, levels_per_motor=4, bits_per_feature=8, input_window_k=4)
	thresholds = list(np.linspace(-1.0, 1.0, NUM_FEATURES * base.bits_per_feature))
	for sn in (2, 4, 6):
		for lv in (2, 4):
			# suffix kept comfortably above the floor for both layers
			g = _mk(rng, state_neurons=sn, levels=lv, ssuf=12, osuf=10)
			cg = controller_genome_from_arch(g, base, thresholds)
			c = build_controller(cg)  # Rust ctor validates connectivity lengths
			assert isinstance(c, WnnController)
			c.reset()
			pwm = c.step([0.01, -0.02, 0.0], [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
			assert len(pwm) == base.num_motors
			assert all(0.0 <= float(p) <= 1.0 for p in pwm), "PWM out of [0,1]"
	print("✓ wnn_controller_roundtrip")


# ===========================================================================
# Step 4a — unified genome cells payload + best-effort remap
# ===========================================================================

def _with_cells(g, su, ou, sv, ov):
	g = g.clone()
	g.cells = MemoryPayload(list(su), list(ou), list(sv), list(ov))
	return g


# MemoryPayload stores universes as (N,2) uint64 arrays and values as a uint8
# array (since 8c87e273, 156 -> 17 B/cell). These adapt them back to the plain
# Python shapes the assertions below are written against. Comparing the arrays
# directly does NOT work and does not always fail loudly:
#   arr == [(0,1)]      -> elementwise array -> "truth value is ambiguous"
#   arr == []           -> "operands could not be broadcast (0,2) vs (0,)"
#   dict(zip(arr, ...)) -> "unhashable type: numpy.ndarray" (rows are arrays)
#   sv + ov             -> elementwise ADDITION, not list concatenation -- this
#                          one silently computes a wrong assertion rather than
#                          raising, which is why it must go through _vals().
def _uni(u):
	"""Universe as a list of (neuron, address) int tuples."""
	return [(int(n), int(a)) for n, a in u]


def _vals(*arrays):
	"""One flat list of int values, concatenated across the given arrays."""
	return [int(x) for arr in arrays for x in arr]


# ---- pure remap-math tests (the risky part) -------------------------------

def test_remap_grow_replicates():
	# d=2: each cell → 4 children A·4+child, same value.
	u, v = [(0, 5), (1, 0)], [3, 1]
	nu, nv = _remap_grow(u, v, 2)
	assert nu == [(0, 20), (0, 21), (0, 22), (0, 23), (1, 0), (1, 1), (1, 2), (1, 3)]
	assert nv == [3, 3, 3, 3, 1, 1, 1, 1]
	print("✓ remap_grow_replicates")


def test_remap_shrink_majority():
	# d=1: addresses 4,5 → 2 ; values [0,0,3] collide at key (0,2)→majority 0.
	u = [(0, 4), (0, 5), (0, 6)]
	v = [0, 0, 3]
	nu, nv = _remap_shrink(u, v, 1)
	d = dict(zip(nu, nv))
	assert d[(0, 2)] == 0, "4,5 → 2 majority of [0,0] = 0"   # 4>>1=2, 5>>1=2
	assert d[(0, 3)] == 3, "6 → 3"                            # 6>>1=3
	print("✓ remap_shrink_majority")


def test_majority_tiebreak_lower():
	assert _majority([3, 1]) == 1, "tie → lower value"
	assert _majority([2, 2, 0]) == 2, "clear majority"
	print("✓ majority_tiebreak_lower")


def test_remap_prefix_grow_formula():
	# w=3, +1 state neuron: new_addr = (P<<5) | (2<<3) | S, P=A>>3, S=A&7.
	A = (0b10 << 3) | 0b110  # P=2, S=6  → A=22
	nu, nv = _remap_prefix_grow([(0, A)], [3], k=1, w=3)
	assert nu == [(0, (2 << 5) | (2 << 3) | 6)]   # 64 | 16 | 6 = 86
	assert nv == [3]
	print("✓ remap_prefix_grow_formula")


def test_remap_prefix_shrink_inverts_grow():
	# Grow then shrink at neutral branch returns the original key (round-trip).
	u, v = [(0, 22), (1, 13)], [3, 2]
	gu, gv = _remap_prefix_grow(u, v, k=1, w=3)
	su, sv = _remap_prefix_shrink(gu, gv, k=1, w=3)
	assert dict(zip(su, sv)) == {(0, 22): 3, (1, 13): 2}, "prefix grow→shrink round-trips"
	print("✓ remap_prefix_shrink_inverts_grow")


# ---- mutation integration with cells --------------------------------------

def test_mutate_bits_remaps_cells():
	rng = np.random.default_rng(10)
	g = _mk(rng, state_neurons=3, levels=2, ssuf=6, osuf=6)
	# tiny universe at known addresses
	g = _with_cells(g, su=[(0, 1), (1, 7)], ou=[(0, 3)], sv=[2, 3], ov=[1])
	cfg = RecurrentArchConfig(min_suffix=1, max_suffix=24, suffix_delta=2,
	                          state_neuron_delta=0, output_block_delta=0)
	for _ in range(60):
		m = g.mutate(PhaseType.BITS, rate=1.0, config=cfg, rng=rng)
		m.assert_valid()           # validates cell addresses < 2^bits, alignment
		_check_canonical_prefix(m)
		g = m
	print("✓ mutate_bits_remaps_cells")


def test_mutate_neurons_state_remaps_both_layers():
	rng = np.random.default_rng(11)
	g = _mk(rng, state_neurons=3, levels=2, ssuf=5, osuf=5)
	g = _with_cells(g, su=[(0, 1), (1, 2)], ou=[(0, 4), (3, 1)], sv=[3, 1], ov=[2, 0])
	cfg = RecurrentArchConfig(min_state_neurons=2, max_state_neurons=8,
	                          state_neuron_delta=1, output_block_delta=0, suffix_delta=0)
	saw_grow = saw_shrink = False
	for _ in range(200):
		m = g.mutate(PhaseType.NEURONS, rate=1.0, config=cfg, rng=rng)
		m.assert_valid()
		_check_canonical_prefix(m)
		if m.state_neurons > g.state_neurons:
			saw_grow = True
		if m.state_neurons < g.state_neurons:
			saw_shrink = True
			# removed state neurons keep no cells
			assert all(n < m.state_neurons for (n, _a) in m.cells.state_universe)
		g = m
	assert saw_grow and saw_shrink
	print("✓ mutate_neurons_state_remaps_both_layers")


def test_mutate_neurons_output_preserves_state_cells():
	rng = np.random.default_rng(12)
	g = _mk(rng, state_neurons=3, levels=4, ssuf=5, osuf=5)   # 16 output neurons
	state_cells = [(0, 1), (2, 3)]
	g = _with_cells(g, su=state_cells, ou=[(0, 2), (15, 1)], sv=[3, 2], ov=[1, 0])
	cfg = RecurrentArchConfig(min_output_neurons=4, max_output_neurons=64,
	                          state_neuron_delta=0, output_block_delta=2, suffix_delta=0)
	for _ in range(80):
		m = g.mutate(PhaseType.NEURONS, rate=1.0, config=cfg, rng=rng)
		m.assert_valid()
		# state cells are NEVER touched by output neurogenesis
		assert _uni(m.cells.state_universe) == state_cells
		assert all(n < m.output_neurons for (n, _a) in _uni(m.cells.output_universe))
		g = m
	print("✓ mutate_neurons_output_preserves_state_cells")


def test_mutate_connections_drops_changed_neurons():
	rng = np.random.default_rng(13)
	g = _mk(rng, state_neurons=4, levels=2, ssuf=8, osuf=6)
	g = _with_cells(g, su=[(0, 1), (1, 2), (2, 3), (3, 0)], ou=[(0, 1)],
	                sv=[1, 2, 3, 0], ov=[2])
	m = g.mutate(PhaseType.CONNECTIONS, rate=1.0, config=RecurrentArchConfig(), rng=rng)
	m.assert_valid()
	# rate=1.0 scrambles every neuron's suffix → all cells dropped
	assert _uni(m.cells.state_universe) == [] and _uni(m.cells.output_universe) == []
	# rate=0.0 changes nothing → all cells survive
	m0 = g.mutate(PhaseType.CONNECTIONS, rate=0.0, config=RecurrentArchConfig(), rng=rng)
	assert _uni(m0.cells.state_universe) == _uni(g.cells.state_universe)
	print("✓ mutate_connections_drops_changed_neurons")


def test_mutate_memory_nudges_values_only():
	rng = np.random.default_rng(14)
	g = _mk(rng, state_neurons=3, levels=2, ssuf=6, osuf=6)
	g = _with_cells(g, su=[(0, 1), (1, 2), (2, 3)], ou=[(0, 1), (1, 2)],
	                sv=[0, 0, 0], ov=[3, 3])
	m = g.mutate(PhaseType.MEMORY, rate=1.0, config=RecurrentArchConfig(), rng=rng)
	m.assert_valid()
	# architecture + universe unchanged; values moved by ±1, clamped 0..3
	assert _uni(m.cells.state_universe) == _uni(g.cells.state_universe)
	assert (m.state_neurons, m.output_neurons) == (g.state_neurons, g.output_neurons)
	assert all(0 <= v <= 3 for v in _vals(m.cells.state_values, m.cells.output_values))
	assert all(abs(a - b) <= 1 for a, b in zip(_vals(m.cells.state_values), _vals(g.cells.state_values)))
	# MEMORY on a genome without cells must error clearly
	try:
		_mk(rng).mutate(PhaseType.MEMORY, 1.0, RecurrentArchConfig(), rng)
		raise SystemExit("MEMORY without cells should raise")
	except ValueError:
		pass
	print("✓ mutate_memory_nudges_values_only")


def test_crossover_memory_per_cell():
	rng = np.random.default_rng(15)
	g = _mk(rng, state_neurons=3, levels=2, ssuf=6, osuf=6)
	a = _with_cells(g, su=[(0, 1), (1, 2)], ou=[(0, 1)], sv=[0, 0], ov=[0])
	b = _with_cells(g, su=[(0, 1), (1, 2)], ou=[(0, 1)], sv=[3, 3], ov=[3])
	child = RecurrentArchGenome.crossover_memory(a, b, rng)
	child.assert_valid()
	assert all(v in (0, 3) for v in _vals(child.cells.state_values, child.cells.output_values))
	print("✓ crossover_memory_per_cell")


def test_clone_fingerprint_with_cells():
	rng = np.random.default_rng(16)
	g = _with_cells(_mk(rng), su=[(0, 1)], ou=[(0, 1)], sv=[2], ov=[1])
	c = g.clone()
	assert c.fingerprint() == g.fingerprint()
	# Stage B: cells live in a Rust handle and the numpy fields are on-demand
	# COPIES — in-place array writes are not a mutation route any more. Mutate
	# the clone through a handle op instead and verify the original is untouched.
	c.cells.drop_changed_state([0])
	assert int(g.cells.state_values[0]) == 2, "clone shares cells (not deep)"
	assert len(c.cells.state_values) == 0 and len(g.cells.state_values) == 1
	assert c.fingerprint() != g.fingerprint(), "cells must be in fingerprint"
	print("✓ clone_fingerprint_with_cells")


# ---- the strong one: replicate-on-grow preserves behavior EXACTLY ----------

def _record_universe(g, base, thresholds, steps=60, seed=0):
	"""Run a short rollout and capture every (neuron,address) the layers read;
	assign random QSR values → a universe the controller actually exercises."""
	from wnn.control.evaluator import controller_genome_from_arch, build_controller
	c = build_controller(controller_genome_from_arch(g, base, thresholds))
	c.reset()
	su, ou = set(), set()
	for _ in range(steps):
		c.step([0.05, -0.03, 0.02], [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
		su.update(tuple(x) for x in c.last_state_addresses())
		ou.update(tuple(x) for x in c.last_output_addresses())
	su, ou = sorted(su), sorted(ou)
	rng = np.random.default_rng(seed)
	return MemoryPayload(su, ou,
	                     [int(v) for v in rng.integers(0, 4, len(su))],
	                     [int(v) for v in rng.integers(0, 4, len(ou))])


def _trajectory(g, base, thresholds, steps=60):
	from wnn.control.evaluator import controller_genome_from_arch, build_controller
	c = build_controller(controller_genome_from_arch(g, base, thresholds))
	c.reset()
	traj = []
	for _ in range(steps):
		traj.append(tuple(float(p) for p in c.step([0.05, -0.03, 0.02], [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])))
	return traj


def test_remove_state_neuron_surgical():
	"""Excise a MID-array state neuron: its 1-bit prefix bit is deleted from every
	address, survivors reindex. Worked example (n=3, w=2 suffix, prefix_factor=1 →
	sbpn=5; remove k=1): address layout [n0 n1 n2 s1 s0], remove neuron 1 deletes
	bit 3. p_lsb = (1·3+2) − 1 − 1·1 = 3, nbits=1. Neuron0 stays index 0, neuron2
	→ index 1; delete bit 3: a → ((a>>4)<<3) | (a&7)."""
	sh = RecurrentArchShape(prefix_factor=1, state_input_space=64, output_input_space=64, output_quantum=4)
	g = RecurrentArchGenome(sh, state_neurons=3, output_neurons=4,
	                        state_sampled=[[0, 1], [2, 3], [4, 5]],
	                        output_sampled=[[0, 1], [2, 3], [4, 5], [6, 7]])
	# state cells: neuron0@22 (0b10110), neuron1@5 (dropped), neuron2@27 (0b11011)
	g.cells = MemoryPayload([(0, 22), (1, 5), (2, 27)], [(0, 27)], [1, 2, 3], [2])
	g.assert_valid()
	g.remove_state_neuron(1, np.random.default_rng(0))
	g.assert_valid()
	assert g.state_neurons == 2
	# 22 → ((22>>4)<<3)|(22&7) = (1<<3)|6 = 14 ; 27 → (1<<3)|3 = 11
	assert dict(zip(_uni(g.cells.state_universe), _vals(g.cells.state_values))) == {(0, 14): 1, (1, 11): 3}, \
		f"got {dict(zip(_uni(g.cells.state_universe), _vals(g.cells.state_values)))}"
	assert dict(zip(_uni(g.cells.output_universe), _vals(g.cells.output_values))) == {(0, 11): 2}
	print("✓ remove_state_neuron_surgical")


def test_behavior_preserved_under_bits_grow():
	"""A BITS-grow replicates each cell across the 2^d new low-bit children, so a
	controller built from the remapped cells must produce an IDENTICAL trajectory
	(the appended bits never change the read value). The deepest correctness proof."""
	from wnn.control.evaluator import ControllerSpec, NUM_FEATURES
	rng = np.random.default_rng(17)
	base = ControllerSpec(num_motors=4, levels_per_motor=4, bits_per_feature=8, input_window_k=4)
	thresholds = list(np.linspace(-1.0, 1.0, NUM_FEATURES * base.bits_per_feature))
	g = _mk(rng, state_neurons=3, levels=4, ssuf=10, osuf=8)
	g.cells = _record_universe(g, base, thresholds)
	before = _trajectory(g, base, thresholds)
	# Force a pure +d grow on BOTH layers (no shrink, no other dims).
	cfg = RecurrentArchConfig(min_suffix=g.state_suffix_width + 2, max_suffix=40,
	                          suffix_delta=2, state_neuron_delta=0, output_block_delta=0)
	grown = g.mutate(PhaseType.BITS, rate=1.0, config=cfg, rng=rng)
	grown.assert_valid()
	assert grown.state_suffix_width > g.state_suffix_width, "must have grown"
	after = _trajectory(grown, base, thresholds)
	assert after == before, "replicate-on-grow must preserve the trajectory exactly"
	print("✓ behavior_preserved_under_bits_grow")


if __name__ == "__main__":
	test_random_valid()
	test_clone_is_deep_and_identical()
	test_mutate_connections_preserves_shape()
	test_mutate_bits_uniform_and_floored()
	test_mutate_neurons_global_reshape_and_survivor_preservation()
	test_crossover_different_shapes_valid()
	test_wnn_controller_roundtrip()
	# step 4a — cells + remap
	test_remap_grow_replicates()
	test_remap_shrink_majority()
	test_majority_tiebreak_lower()
	test_remap_prefix_grow_formula()
	test_remap_prefix_shrink_inverts_grow()
	test_mutate_bits_remaps_cells()
	test_mutate_neurons_state_remaps_both_layers()
	test_mutate_neurons_output_preserves_state_cells()
	test_mutate_connections_drops_changed_neurons()
	test_mutate_memory_nudges_values_only()
	test_crossover_memory_per_cell()
	test_clone_fingerprint_with_cells()
	test_remove_state_neuron_surgical()
	test_behavior_preserved_under_bits_grow()
	print("\nAll RecurrentArchGenome structural-validity tests passed.")
