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
	RecurrentArchGenome, RecurrentArchShape, RecurrentArchConfig,
)
from wnn.ram.strategies.connectivity.adaptive_cluster import PhaseType


# A drone-flavoured shape (num_motors=4 → output_quantum=4), but the genome
# itself stays domain-free. K=4, F=9, b=8 → state_input_space=288, frame=72.
SHAPE = RecurrentArchShape(prefix_factor=2, state_input_space=288,
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
				assert g.forced_prefix == 2 * sn
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
		assert m.forced_prefix == 2 * m.state_neurons  # prefix tracks neuron count
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
	from ram_accelerator import WnnController  # noqa: import here so the file imports without the accel
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


if __name__ == "__main__":
	test_random_valid()
	test_clone_is_deep_and_identical()
	test_mutate_connections_preserves_shape()
	test_mutate_bits_uniform_and_floored()
	test_mutate_neurons_global_reshape_and_survivor_preservation()
	test_crossover_different_shapes_valid()
	test_wnn_controller_roundtrip()
	print("\nAll RecurrentArchGenome structural-validity tests passed.")
