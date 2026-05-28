"""Tests for RecurrentArchGenome.crossover_average — the user-spec'd average-shape
crossover added 2026-05-28.

The "average" semantics differ from the existing one-parent-shape crossover:
- target shape = element-wise average of parent shapes
- per-neuron suffixes sampled from parent positions, resized to target width
- cells dropped (None) — address universe doesn't align across shapes
"""

from __future__ import annotations

import numpy as np
import pytest

from wnn.control.recurrent_genome import (
	RecurrentArchGenome,
	RecurrentArchShape,
)


@pytest.fixture
def shape() -> RecurrentArchShape:
	return RecurrentArchShape(
		prefix_factor=2, state_input_space=64, output_input_space=64, output_quantum=4
	)


def _mk(shape, sn, on, ssuf, osuf, seed):
	rng = np.random.default_rng(seed)
	return RecurrentArchGenome.random(shape, state_neurons=sn, output_neurons=on,
	                                  state_suffix=ssuf, output_suffix=osuf, rng=rng)


def test_average_shape_picks_average_state_neurons(shape):
	"""4 + 8 state neurons → child has 6 (the average)."""
	a = _mk(shape, sn=4, on=16, ssuf=8, osuf=8, seed=1)
	b = _mk(shape, sn=8, on=32, ssuf=10, osuf=12, seed=2)
	rng = np.random.default_rng(42)
	c = RecurrentArchGenome.crossover_average(a, b, rng)
	c.assert_valid()
	assert c.state_neurons == 6, f"expected avg state_neurons=6, got {c.state_neurons}"


def test_average_shape_rounds_output_to_quantum(shape):
	"""output_neurons must be a multiple of output_quantum (4); 16+32=48, avg=24, which IS divisible by 4."""
	a = _mk(shape, sn=4, on=16, ssuf=8, osuf=8, seed=1)
	b = _mk(shape, sn=8, on=32, ssuf=10, osuf=12, seed=2)
	rng = np.random.default_rng(42)
	c = RecurrentArchGenome.crossover_average(a, b, rng)
	c.assert_valid()
	assert c.output_neurons == 24, f"expected avg output=24, got {c.output_neurons}"
	assert c.output_neurons % shape.output_quantum == 0


def test_average_shape_rounds_off_quantum(shape):
	"""output_neurons avg = 22, rounds to nearest multiple of 4 → 24 or 20.
	Defensively: must be a valid multiple of quantum."""
	a = _mk(shape, sn=4, on=16, ssuf=8, osuf=8, seed=1)
	b = _mk(shape, sn=4, on=28, ssuf=8, osuf=8, seed=2)
	rng = np.random.default_rng(42)
	c = RecurrentArchGenome.crossover_average(a, b, rng)
	c.assert_valid()
	assert c.output_neurons in (20, 24), f"expected 20 or 24, got {c.output_neurons}"
	assert c.output_neurons % shape.output_quantum == 0


def test_average_suffix_widths(shape):
	"""suffix widths averaged: (8 + 12) / 2 = 10."""
	a = _mk(shape, sn=4, on=16, ssuf=8, osuf=8, seed=1)
	b = _mk(shape, sn=8, on=32, ssuf=12, osuf=12, seed=2)
	rng = np.random.default_rng(42)
	c = RecurrentArchGenome.crossover_average(a, b, rng)
	c.assert_valid()
	assert c.state_suffix_width == 10, f"expected state_suffix=10, got {c.state_suffix_width}"
	assert c.output_suffix_width == 10, f"expected output_suffix=10, got {c.output_suffix_width}"


def test_same_shape_parents_produce_same_shape_child(shape):
	"""Same-shape special case: child has parent shape, per-neuron suffix uniform-picked."""
	a = _mk(shape, sn=4, on=16, ssuf=8, osuf=8, seed=1)
	b = _mk(shape, sn=4, on=16, ssuf=8, osuf=8, seed=2)
	rng = np.random.default_rng(42)
	c = RecurrentArchGenome.crossover_average(a, b, rng)
	c.assert_valid()
	assert c.state_neurons == 4
	assert c.output_neurons == 16
	assert c.state_suffix_width == 8
	assert c.output_suffix_width == 8


def test_cells_are_dropped(shape):
	"""crossover_average drops cells — addresses don't align with the new shape."""
	from wnn.control.recurrent_genome import MemoryPayload
	a = _mk(shape, sn=4, on=16, ssuf=8, osuf=8, seed=1)
	a.cells = MemoryPayload([(0, 5)], [(0, 7)], [3], [2])  # some toy cells
	b = _mk(shape, sn=8, on=32, ssuf=12, osuf=12, seed=2)
	b.cells = MemoryPayload([(0, 11)], [(0, 13)], [1], [0])  # different cells
	rng = np.random.default_rng(42)
	c = RecurrentArchGenome.crossover_average(a, b, rng)
	c.assert_valid()
	assert c.cells is None, "crossover_average must drop cells for the variable-shape case"


def test_per_neuron_suffix_widths_are_uniform(shape):
	"""All state suffixes have the same width; same for output (genome invariant)."""
	a = _mk(shape, sn=4, on=16, ssuf=6, osuf=8, seed=1)
	b = _mk(shape, sn=10, on=24, ssuf=14, osuf=16, seed=2)
	rng = np.random.default_rng(42)
	c = RecurrentArchGenome.crossover_average(a, b, rng)
	c.assert_valid()
	state_widths = {len(s) for s in c.state_sampled}
	output_widths = {len(s) for s in c.output_sampled}
	assert len(state_widths) == 1, f"non-uniform state widths: {state_widths}"
	assert len(output_widths) == 1, f"non-uniform output widths: {output_widths}"


def test_suffix_bits_are_in_range(shape):
	"""All sampled bits stay within [0, input_space) — never wander out of bounds."""
	a = _mk(shape, sn=4, on=16, ssuf=8, osuf=8, seed=1)
	b = _mk(shape, sn=8, on=32, ssuf=12, osuf=12, seed=2)
	rng = np.random.default_rng(42)
	for trial in range(20):
		c = RecurrentArchGenome.crossover_average(a, b, rng)
		c.assert_valid()
		for s in c.state_sampled:
			assert all(0 <= x < shape.state_input_space for x in s)
		for o in c.output_sampled:
			assert all(0 <= x < shape.output_input_space for x in o)


def test_no_duplicate_bits_within_suffix(shape):
	"""A neuron's suffix has DISTINCT bit indices (no duplicate sampled bits)."""
	a = _mk(shape, sn=4, on=16, ssuf=8, osuf=8, seed=1)
	b = _mk(shape, sn=8, on=32, ssuf=12, osuf=12, seed=2)
	rng = np.random.default_rng(42)
	for trial in range(20):
		c = RecurrentArchGenome.crossover_average(a, b, rng)
		c.assert_valid()
		for s in c.state_sampled:
			assert len(set(s)) == len(s), f"duplicate state bits: {s}"
		for o in c.output_sampled:
			assert len(set(o)) == len(o), f"duplicate output bits: {o}"


def test_stress_many_random_pairs(shape):
	"""Robustness: 100 random parent pairs must all produce valid children."""
	rng = np.random.default_rng(123)
	for _ in range(100):
		sn_a = int(rng.integers(2, 16))
		sn_b = int(rng.integers(2, 16))
		on_a = int(rng.integers(1, 16)) * shape.output_quantum
		on_b = int(rng.integers(1, 16)) * shape.output_quantum
		ssuf_a = int(rng.integers(2, 20))
		ssuf_b = int(rng.integers(2, 20))
		osuf_a = int(rng.integers(2, 20))
		osuf_b = int(rng.integers(2, 20))
		a = _mk(shape, sn=sn_a, on=on_a, ssuf=ssuf_a, osuf=osuf_a, seed=int(rng.integers(0, 10000)))
		b = _mk(shape, sn=sn_b, on=on_b, ssuf=ssuf_b, osuf=osuf_b, seed=int(rng.integers(0, 10000)))
		c = RecurrentArchGenome.crossover_average(a, b, rng)
		c.assert_valid()
		assert c.state_neurons == max(1, (sn_a + sn_b) // 2)


if __name__ == "__main__":
	import sys
	pytest.main([__file__, "-v"] + sys.argv[1:])
