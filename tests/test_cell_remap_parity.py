"""Bit-exact parity: Rust cell_remap_* vs the Python _remap_* originals.

These remaps run inside the GA, so ANY divergence changes genome lineage and
silently invalidates comparisons against every prior controller result. Two
properties are load-bearing and easy to break in a port:

  * output ORDER on the collapse paths is first-encounter (Python dict insertion
    order), NOT sorted -- values are positional and fingerprint() hashes the
    buffers, so reordering changes genome identity;
  * majority ties resolve to the LOWER value.

Randomised over shapes chosen to force collisions (small address spaces, repeated
neurons), because the collapse paths are exactly where an order or tie-break bug
hides.

Run:  PYTHONPATH=src/wnn python -m pytest tests/test_cell_remap_parity.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from wnn.control.recurrent_genome import (
	_remap_grow, _remap_shrink, _remap_prefix_grow, _remap_prefix_shrink,
	_remap_delete_bit_window, _drop_neurons_ge, _drop_changed_neurons, _majority,
)

ra = pytest.importorskip("ram_controller")


def _split(universe, values):
	"""Python (list-of-(n,a), list-of-v) -> Rust column form."""
	ns = [int(n) for n, _a in universe]
	addrs = [int(a) for _n, a in universe]
	return ns, addrs, [int(v) for v in values]


def _join(ns, addrs, vals):
	"""Rust column form -> the Python shape, for direct ==."""
	return [(int(n), int(a)) for n, a in zip(ns, addrs)], [int(v) for v in vals]


def _rand_cells(rng, count, n_neurons, addr_bits):
	"""Deliberately small address space so collapses actually collide."""
	universe, seen = [], set()
	for _ in range(count):
		n = int(rng.integers(0, n_neurons))
		a = int(rng.integers(0, 1 << addr_bits))
		if (n, a) in seen:          # universes are unique (n, a) pairs
			continue
		seen.add((n, a))
		universe.append((n, a))
	values = [int(rng.integers(0, 4)) for _ in universe]   # QSR 0..3
	return universe, values


CASES = [(seed, cnt, nn, bits) for seed in range(12)
         for cnt, nn, bits in ((40, 3, 6), (200, 5, 8), (500, 8, 10))]


@pytest.mark.parametrize("seed,cnt,nn,bits", CASES)
def test_grow_shrink_parity(seed, cnt, nn, bits):
	rng = np.random.default_rng(seed)
	u, v = _rand_cells(rng, cnt, nn, bits)
	ns, addrs, vals = _split(u, v)
	for d in (0, 1, 2, 3):
		assert _join(*ra.cell_remap_grow(ns, addrs, vals, d)) == _remap_grow(u, v, d)
		assert _join(*ra.cell_remap_shrink(ns, addrs, vals, d)) == _remap_shrink(u, v, d)


@pytest.mark.parametrize("seed,cnt,nn,bits", CASES)
def test_prefix_parity(seed, cnt, nn, bits):
	rng = np.random.default_rng(seed)
	u, v = _rand_cells(rng, cnt, nn, bits)
	ns, addrs, vals = _split(u, v)
	for pf in (1, 2):
		for k in (0, 1, 2):
			for w in (2, 4):
				assert _join(*ra.cell_remap_prefix_grow(ns, addrs, vals, k, w, pf)) \
				       == _remap_prefix_grow(u, v, k, w, pf)
				assert _join(*ra.cell_remap_prefix_shrink(ns, addrs, vals, k, w, pf)) \
				       == _remap_prefix_shrink(u, v, k, w, pf)


@pytest.mark.parametrize("seed,cnt,nn,bits", CASES)
def test_delete_window_and_drops_parity(seed, cnt, nn, bits):
	rng = np.random.default_rng(seed)
	u, v = _rand_cells(rng, cnt, nn, bits)
	ns, addrs, vals = _split(u, v)
	for p_lsb in (0, 1, 3, 5):
		for nbits in (1, 2):
			assert _join(*ra.cell_remap_delete_bit_window(ns, addrs, vals, p_lsb, nbits)) \
			       == _remap_delete_bit_window(u, v, p_lsb, nbits)
	for limit in (0, 1, nn // 2, nn):
		assert _join(*ra.cell_drop_neurons_ge(ns, addrs, vals, limit)) \
		       == _drop_neurons_ge(u, v, limit)
	for changed in ([], [0], list(range(0, nn, 2)), list(range(nn))):
		assert _join(*ra.cell_drop_changed_neurons(ns, addrs, vals, changed)) \
		       == _drop_changed_neurons(u, v, set(changed))


def test_majority_parity_exhaustive():
	"""Every multiset over QSR 0..3 up to size 5 -- the tie-break must match."""
	from itertools import combinations_with_replacement
	for size in range(1, 6):
		for combo in combinations_with_replacement(range(4), size):
			assert ra.cell_majority(list(combo)) == _majority(list(combo)), combo


def test_collapse_order_is_first_encounter_not_sorted():
	"""Guards the property a HashMap port would silently break.

	The input is ordered so first-encounter and sorted order DISAGREE: the high
	bucket (0,3) is seen before (0,2). An earlier version of this test used an
	input whose first-encounter order happened to already be sorted, so it would
	have passed against a sorting implementation -- i.e. it guarded nothing.
	"""
	u = [(0, 12), (0, 8), (0, 9), (1, 4)]           # >>2 -> 3, 2, 2, 1
	v = [2, 1, 3, 0]
	ns, addrs, vals = _split(u, v)
	got_u, got_v = _join(*ra.cell_remap_shrink(ns, addrs, vals, 2))
	assert (got_u, got_v) == _remap_shrink(u, v, 2)
	assert got_u == [(0, 3), (0, 2), (1, 1)]        # first-encounter
	assert got_u != sorted(got_u), "input must distinguish the two orderings"
	assert got_v == [2, 1, 0]                       # bucket {1,3} ties -> lower = 1


def test_grow_overflow_raises_instead_of_wrapping():
	"""Python raises OverflowError when MemoryPayload stores the bigint; Rust must
	not silently wrap the address into a different cell."""
	with pytest.raises(OverflowError):
		ra.cell_remap_grow([0], [1 << 63], [3], 2)
