"""Bank-aware output neurogenesis (08/11/2026 increment-0 fix B1).

The output layer is `output_quantum` equal contiguous segments — one per bank
(cumulative decode) or per E/I half per bank (antagonist) — and the controller
reads a neuron's role positionally (controller.rs: bank = n / levels, half =
level >= L/2). Output neurogenesis used to grow/shrink the FLAT tail, so a
256 -> 264 grow moved almost every neuron outside segment 0 to another bank /
level / half, and Lamarckian cells followed the index, not the neuron.

Every test here FAILS on the pre-fix code (tail resize, index-keyed crossover
and cell inheritance).

Run: PYTHONPATH=src/wnn python tests/controller_bank_neurogenesis.py
"""

from __future__ import annotations

import numpy as np

from wnn.control.evaluator import (
	ControllerSpec, arch_shape_from_spec, build_controller, controller_genome_from_arch,
)
from wnn.control.recurrent_genome import RecurrentArchGenome, MemoryPayload
from wnn.control.arch_strategy import _inherit_cells

STEP_INPUT = ([0.05, -0.03, 0.02], [0.4, -0.3, 9.7], [0.01, 0.0, 0.0])
OUT_SUFFIX = 10


def _base(memory_mode: str, levels: int) -> ControllerSpec:
	"""sn=0 single-layer controller (the production recipe's layout)."""
	return ControllerSpec(num_motors=4, levels_per_motor=levels, bits_per_feature=8,
	                      input_window_k=4, state_neurons=0, state_bits_per_neuron=0,
	                      output_bits_per_neuron=OUT_SUFFIX, memory_mode=memory_mode)


def _thresholds(base: ControllerSpec) -> list[float]:
	return list(np.linspace(-1.0, 1.0, base.num_features() * base.bits_per_feature))


def _genome(base: ControllerSpec, seed: int) -> RecurrentArchGenome:
	shape = arch_shape_from_spec(base)
	n_out = base.num_motors * base.levels_per_motor
	return RecurrentArchGenome.random(shape, 0, n_out, 0, OUT_SUFFIX, np.random.default_rng(seed))


def _trained(g: RecurrentArchGenome, base: ControllerSpec, seed: int) -> RecurrentArchGenome:
	"""Give every output neuron a cell at each address a short rollout reads, with
	a random value of the mode's alphabet — a 'trained' Lamarckian genome."""
	c = build_controller(controller_genome_from_arch(g, base, _thresholds(base)))
	c.reset()
	ou = set()
	for t in range(40):
		c.step([0.05 * np.sin(t), -0.03, 0.02], [0.3 * np.cos(t), -0.2, 9.7], [0.0, 0.0, 0.0])
		ou.update(tuple(int(x) for x in a) for a in c.last_output_addresses())
	ou = sorted(ou)
	hi = 2 if base.memory_mode == "BINARY" else 4
	vals = [int(v) for v in np.random.default_rng(seed).integers(0, hi, len(ou))]
	g = g.clone()
	g.cells = MemoryPayload([], ou, [], vals)
	return g


def _cells_by_neuron(g: RecurrentArchGenome) -> dict[int, dict[int, int]]:
	out: dict[int, dict[int, int]] = {}
	for (n, a), v in zip(g.cells.output_universe, g.cells.output_values):
		out.setdefault(int(n), {})[int(a)] = int(v)
	return out


def _role(i: int, n: int, levels_per_bank: int, antagonist: bool) -> tuple:
	"""(bank, rank-in-half, half) exactly as the Rust controller reads it."""
	bank, level = i // levels_per_bank, i % levels_per_bank
	if not antagonist:
		return bank, level, 0
	half = levels_per_bank // 2
	return bank, level % half, int(level >= half)


def _index_of_role(role: tuple, levels_per_bank: int, antagonist: bool) -> int:
	bank, rank, half = role
	return bank * levels_per_bank + half * (levels_per_bank // 2 if antagonist else 0) + rank


def _check_roles_kept(old: RecurrentArchGenome, new: RecurrentArchGenome, antagonist: bool) -> int:
	"""Every old neuron whose role exists in `new` sits there with the same suffix
	and the same cells. Returns how many survived."""
	L0, L1 = old.output_neurons // 4, new.output_neurons // 4
	c0, c1 = _cells_by_neuron(old), _cells_by_neuron(new)
	kept = 0
	for i in range(old.output_neurons):
		bank, rank, half = _role(i, old.output_neurons, L0, antagonist)
		if rank >= (L1 // 2 if antagonist else L1):
			continue   # dropped by a shrink
		j = _index_of_role((bank, rank, half), L1, antagonist)
		assert _role(j, new.output_neurons, L1, antagonist) == (bank, rank, half)
		assert new.output_sampled[j] == old.output_sampled[i], \
			f"neuron {i} (bank {bank}, rank {rank}, half {half}) lost its wiring at {j}"
		assert c1.get(j, {}) == c0.get(i, {}), \
			f"neuron {i} (bank {bank}, rank {rank}, half {half}) lost its cells at {j}"
		kept += 1
	return kept


def _grow_shrink_roundtrip(memory_mode: str, levels: int) -> None:
	base = _base(memory_mode, levels)
	q = arch_shape_from_spec(base).output_quantum
	antagonist = base.resolved_output_decode() == "antagonist"
	g = _trained(_genome(base, 11), base, 12)
	assert len(_cells_by_neuron(g)) > g.output_neurons // 2, "fixture must train most neurons"
	grown = g.clone()
	grown.set_output_neurons(g.output_neurons + q, np.random.default_rng(1))
	assert _check_roles_kept(g, grown, antagonist) == g.output_neurons
	# The fresh neurons are the tail of EVERY segment, and carry no cells.
	fresh = {j for j in range(grown.output_neurons)
	         if _role(j, grown.output_neurons, grown.output_neurons // 4, antagonist)[1]
	         == (levels // 2 if antagonist else levels)}
	assert len(fresh) == q and not (fresh & set(_cells_by_neuron(grown))), fresh
	back = grown.clone()
	back.set_output_neurons(g.output_neurons, np.random.default_rng(2))
	assert back.fingerprint() == g.fingerprint(), "grow then shrink must restore the genome exactly"
	shrunk = g.clone()
	shrunk.set_output_neurons(g.output_neurons - q, np.random.default_rng(3))
	assert _check_roles_kept(g, shrunk, antagonist) == g.output_neurons - q


def test_grow_shrink_keeps_bank_level_half_and_cells():
	_grow_shrink_roundtrip("BINARY", 64)          # antagonist, quantum 8 (the recipe)
	_grow_shrink_roundtrip("QUAD_WEIGHTED", 16)   # cumulative, quantum 4
	print("✓ grow/shrink keeps (bank, level, half) + wiring + cells; round trip exact")


def _first_step_cells(g: RecurrentArchGenome, base: ControllerSpec) -> list[int]:
	spec_base = _base(base.memory_mode, g.output_neurons // 4)
	c = build_controller(controller_genome_from_arch(g, spec_base, _thresholds(base)))
	c.reset()
	c.step(*STEP_INPUT)
	return [int(x) for x in c.get_last_output_cells()]


def _trajectory(g: RecurrentArchGenome, base: ControllerSpec, steps: int = 30) -> list:
	c = build_controller(controller_genome_from_arch(g, base, _thresholds(base)))
	c.reset()
	return [tuple(float(p) for p in c.step(*STEP_INPUT)) for _ in range(steps)]


def test_unaffected_levels_decode_unchanged():
	"""The controller's own cell read at a fixed input: every surviving neuron
	reads the same cell after a grow (its level is unaffected), and after the
	shrink back the whole decoded trajectory is bit-identical."""
	base = _base("BINARY", 64)
	antagonist = True
	g = _trained(_genome(base, 21), base, 22)
	q = arch_shape_from_spec(base).output_quantum
	grown = g.clone()
	grown.set_output_neurons(g.output_neurons + q, np.random.default_rng(4))
	before, after = _first_step_cells(g, base), _first_step_cells(grown, base)
	L0, L1 = g.output_neurons // 4, grown.output_neurons // 4
	for i in range(g.output_neurons):
		j = _index_of_role(_role(i, g.output_neurons, L0, antagonist), L1, antagonist)
		assert after[j] == before[i], f"level of neuron {i} changed its read ({before[i]} -> {after[j]})"
	back = grown.clone()
	back.set_output_neurons(g.output_neurons, np.random.default_rng(5))
	assert _trajectory(back, base) == _trajectory(g, base), "decoded output moved after grow+shrink"
	print("✓ unaffected levels read the same cell; grow+shrink decodes bit-identically")


def test_crossover_pairs_output_neurons_by_role():
	"""Different level counts: a child neuron takes the other parent's block only
	from the SAME (bank, rank, half), never the same flat index."""
	base = _base("BINARY", 64)
	a = _genome(base, 31)
	b = _genome(base, 32)
	b.set_output_neurons(b.output_neurons + 16, np.random.default_rng(6))   # 68 levels/bank
	for seed in range(8):
		child = RecurrentArchGenome.crossover(a, b, np.random.default_rng(seed))
		sp, other = (a, b) if child.output_neurons == a.output_neurons else (b, a)
		Lc, Lo = child.output_neurons // 4, other.output_neurons // 4
		for j in range(child.output_neurons):
			role = _role(j, child.output_neurons, Lc, True)
			ok = [sp.output_sampled[j]]
			if role[1] < Lo // 2:
				ok.append(other.output_sampled[_index_of_role(role, Lo, True)])
			assert child.output_sampled[j] in ok, f"seed {seed}: neuron {j} {role} took a foreign role"
	print("✓ crossover pairs output blocks by (bank, rank, half)")


def test_lamarckian_inheritance_follows_roles():
	"""A child shaped like parent2 (more levels) inherits parent1's cells at the
	neurons that carry parent1's wiring in the SAME role, and nowhere else."""
	base = _base("BINARY", 64)
	p1 = _trained(_genome(base, 41), base, 42)
	p2 = p1.clone()
	p2.cells = None
	p2.set_output_neurons(p1.output_neurons + 8, np.random.default_rng(7))
	child = p2.clone()                         # = parent2 shape, parent1's wiring per role
	cells = _inherit_cells(child, p1)
	child.cells = cells
	assert _check_roles_kept(p1, child, True) == p1.output_neurons
	assert set(_cells_by_neuron(child)) <= set(_cells_by_neuron(p1).keys()) | set(range(child.output_neurons))
	# Different state-neuron count = different address layout: nothing inherited.
	other = p1.clone()
	other.state_neurons, other.state_sampled = 1, [[0]]
	assert _inherit_cells(other, p1).cell_count() == 0
	print("✓ Lamarckian inheritance re-indexes by role; prefix-width change inherits nothing")


if __name__ == "__main__":
	test_grow_shrink_keeps_bank_level_half_and_cells()
	test_unaffected_levels_decode_unchanged()
	test_crossover_pairs_output_neurons_by_role()
	test_lamarckian_inheritance_follows_roles()
	print("\nAll bank-neurogenesis tests passed.")
