"""CTRL-45 axis receptive fields: the invariant holds after EVERY operator that can
create or move an output connection, off-mode draws nothing, cells of re-wired
neurons are dropped, and the v1-incompatible combinations are refused.

Invariant: output neuron n of bank b = n // (N / 4) holds at most
floor(rf_cross x width) connections whose feature family (feature_layout.FAMILY)
shares no axis with BANK_FAMILIES[b].

Run: PYTHONPATH=src python tests/controller_receptive_fields.py
"""

from __future__ import annotations

import numpy as np

from wnn.control.arch_strategy import (
	ControllerArchGAStrategy, default_controller_arch_config, rf_summary,
)
from wnn.control.evaluator import ControllerSpec, arch_shape_from_spec, spec_from_arch
from wnn.control.recurrent_genome import RecurrentArchGenome, MemoryPayload, _rf_max_cross
from wnn.ram.strategies.optimization_dimension import OptimizationDimension as Dim

WIDTH = 24
RECIPE_OBS = dict(obs_peraxis_p=True, obs_peraxis_i=True, obs_peraxis_yaw=False, obs_yaw_err=True,
                  obs_yaw_err_i=True, obs_collective_cmd=True, obs_alt_err=True, obs_vz=True,
                  obs_pos_err_xy=True, obs_vel_xy=True)


def _spec(rf: str = "axis", cross: float = 0.2, levels: int = 64) -> ControllerSpec:
	"""The CTRL-45 recipe layout (22 features, BINARY antagonist, sn=0). decouple
	stands in for ctbr: the same 4 axis banks, no airframe needed."""
	return ControllerSpec(num_motors=4, levels_per_motor=levels, bits_per_feature=8,
	                      input_window_k=4, state_neurons=0, state_bits_per_neuron=0,
	                      output_bits_per_neuron=WIDTH, memory_mode="BINARY",
	                      decouple_outputs=True, receptive_fields=rf, rf_cross=cross, **RECIPE_OBS)


def _random(spec: ControllerSpec, seed: int, cfg=None) -> RecurrentArchGenome:
	cfg = cfg or default_controller_arch_config(spec)
	n = spec.num_motors * spec.levels_per_motor
	return RecurrentArchGenome.random(arch_shape_from_spec(spec), 0, n, 0, WIDTH,
	                                  np.random.default_rng(seed), config=cfg)


def _cross_counts(g: RecurrentArchGenome, cfg) -> list[int]:
	bpf, banks = cfg.bits_per_feature, cfg.rf_bank_masks
	per_bank = g.output_neurons // len(banks)
	return [sum(1 for b in suf if not (cfg.rf_feature_masks[b // bpf] & banks[n // per_bank]))
	        for n, suf in enumerate(g.output_sampled)]


def _allowed(cfg, width: int, bank_mask: int) -> int:
	"""The budget, or more when the family owns fewer bits than width - budget (the
	documented deterministic fallback after a large BITS grow)."""
	pool = sum(cfg.bits_per_feature for m in cfg.rf_feature_masks if m & bank_mask)
	return max(_rf_max_cross(cfg, width), width - pool)


def _assert_invariant(g: RecurrentArchGenome, cfg, what: str) -> None:
	per_bank = g.output_neurons // len(cfg.rf_bank_masks)
	for n, c in enumerate(_cross_counts(g, cfg)):
		allowed = _allowed(cfg, g.output_suffix_width, cfg.rf_bank_masks[n // per_bank])
		assert c <= allowed, f"{what}: neuron {n} holds {c} cross-family connections > {allowed}"
	for suf in g.output_sampled:
		assert len(set(suf)) == len(suf), f"{what}: duplicate connection"


def test_invariant_after_every_operator():
	spec = _spec()
	cfg = default_controller_arch_config(spec)
	cfg.max_output_neurons, cfg.output_block_delta, cfg.suffix_delta = 512, 2, 4
	a, b = _random(spec, 1), _random(spec, 2)
	_assert_invariant(a, cfg, "random")
	assert max(_cross_counts(a, cfg)) == _rf_max_cross(cfg, WIDTH), "init = exactly the budget"
	rng = np.random.default_rng(3)
	widths = set()
	for step in range(40):
		for dim in (Dim.NEURONS, Dim.BITS, Dim.CONNECTIONS, Dim.CLUSTER):
			a = a.mutate(dim, 1.0, cfg, rng)
			_assert_invariant(a, cfg, f"mutate {dim.name} #{step}")
		b = b.mutate(Dim.NEURONS, 1.0, cfg, rng)
		for op in ("crossover", "crossover_average"):
			child = getattr(RecurrentArchGenome, op)(a, b, rng, cfg)
			_assert_invariant(child, cfg, f"{op} #{step} ({a.output_neurons} x {b.output_neurons})")
		widths.add(a.output_suffix_width)
	assert len(widths) > 3, f"the sweep must move the width (saw {sorted(widths)})"
	print(f"✓ invariant holds after random / mutate (all dims) / crossover / crossover_average "
	      f"(widths {min(widths)}..{max(widths)})")


def test_off_mode_draws_nothing():
	spec = _spec(rf="off")
	cfg = default_controller_arch_config(spec)
	assert cfg.receptive_fields == "off" and cfg.rf_bank_masks == ()
	g = _random(spec, 4)
	rng = np.random.default_rng(5)
	before = rng.bit_generator.state
	g2 = g.clone().apply_receptive_fields(cfg, rng)
	assert rng.bit_generator.state == before, "off mode consumed an RNG draw"
	assert g2.fingerprint() == g.fingerprint()
	print("✓ off: no RNG draw, genome untouched")


def test_rewired_neurons_lose_their_cells():
	spec = _spec()
	cfg = default_controller_arch_config(spec)
	free = _random(spec, 6, cfg=default_controller_arch_config(_spec(rf="off")))
	free.cells = MemoryPayload([], [(n, 1) for n in range(free.output_neurons)], [],
	                           [1] * free.output_neurons)
	before = [list(s) for s in free.output_sampled]
	free.apply_receptive_fields(cfg, np.random.default_rng(7))
	changed = {n for n in range(free.output_neurons) if free.output_sampled[n] != before[n]}
	kept = {int(n) for n, _a in free.cells.output_universe}
	assert changed and not (changed & kept), "a re-wired neuron kept its cells"
	assert kept == set(range(free.output_neurons)) - changed
	_assert_invariant(free, cfg, "projection of a spread genome")
	print(f"✓ projection re-wired {len(changed)} spread neurons and dropped exactly their cells")


def test_strategy_crossover_and_spec_chain_keep_rf():
	"""The GA strategy hands its config to crossover, and spec_from_arch (the stage
	chain) keeps the RF + GA connectivity fields (it used to drop four of them)."""
	spec = _spec()
	strat = ControllerArchGAStrategy(spec, Dim.NEURONS, seed=1, lamarckian=True)
	strat._on_reseed(0)
	a, b = _random(spec, 8), _random(spec, 9)
	b.set_output_neurons(b.output_neurons + 8, np.random.default_rng(10), strat._arch_config)
	child = strat.crossover_genomes(a, b)
	_assert_invariant(child, strat._arch_config, "strategy crossover")
	full = ControllerSpec(**{**spec.__dict__, "conn_policy": "min_per_cluster", "conn_policy_min": 2,
	                         "conn_mutation_scope": "feature", "feature_balance_ratio": 1.5})
	chained = spec_from_arch(child, full)
	for f in ("receptive_fields", "rf_cross", "conn_policy", "conn_policy_min",
	          "conn_mutation_scope", "feature_balance_ratio"):
		assert getattr(chained, f) == getattr(full, f), f"spec_from_arch dropped {f}"
	assert chained.levels_per_motor == child.output_neurons // 4
	print("✓ strategy crossover projects; spec_from_arch carries rf + conn fields across stages")


def test_genesis_projects():
	"""Stats-guided genesis (Lamarckian flow phases) grows the suffix with FREE bits;
	its exit re-projects. Also exercises the sn=0 assert_valid fix."""
	from types import SimpleNamespace
	from wnn.control.arch_adaptation import ControllerAdaptationStrategy, AdaptationConfig
	spec = _spec()
	st = ControllerAdaptationStrategy.__new__(ControllerAdaptationStrategy)
	st._cfg = AdaptationConfig(genesis_mode="synaptogenesis", bits_margin=40, bit_step=8)
	st._arch_config = default_controller_arch_config(spec)
	st._np_rng = np.random.default_rng(11)
	g = _random(spec, 12)
	stats = SimpleNamespace(state_cell_counts=[], output_cell_counts=[64] * g.output_neurons)
	out = st._genesis(g, stats)
	assert out.output_suffix_width > g.output_suffix_width, "synaptogenesis must have grown"
	_assert_invariant(out, st._arch_config, "genesis")
	print(f"✓ genesis exit projects (suffix {g.output_suffix_width} -> {out.output_suffix_width})")


def test_refusals_and_startup_line():
	from wnn.control.phased_ga import build_arg_parser, _refuse_rf_combos, rf_token
	ok = ["--receptive-fields", "axis", "--output-mode", "ctbr"]
	_refuse_rf_combos(build_arg_parser().parse_args(ok))
	for extra in (["--conn-policy", "min2"], ["--conn-policy", "min1"], ["--conn-policy", "framed1"],
	              ["--feature-balance-ratio", "2"],
	              ["--rf-cross", "1.5"]):
		try:
			_refuse_rf_combos(build_arg_parser().parse_args(ok + extra))
		except SystemExit:
			continue
		raise AssertionError(f"{extra} must be refused with --receptive-fields axis")
	try:
		_refuse_rf_combos(build_arg_parser().parse_args(["--receptive-fields", "axis"]))
		raise AssertionError("PWM mode without --decouple-outputs must be refused")
	except SystemExit:
		pass
	_refuse_rf_combos(build_arg_parser().parse_args(["--receptive-fields", "axis", "--decouple-outputs"]))
	assert rf_token(build_arg_parser().parse_args(ok)) == "axis:cross=0.2"
	assert rf_token(build_arg_parser().parse_args([])) == "off"
	line = rf_summary(_spec(), [WIDTH])
	assert "T/R/P/Y = 4/7/7/4" in line and "4/24" in line, line
	print(f"✓ refusals (min1/min2/framed1/balance/pwm/rf-cross range) + provenance rf= + startup: {line}")


if __name__ == "__main__":
	test_invariant_after_every_operator()
	test_off_mode_draws_nothing()
	test_rewired_neurons_lose_their_cells()
	test_strategy_crossover_and_spec_chain_keep_rf()
	test_genesis_projects()
	test_refusals_and_startup_line()
	print("\nAll receptive-field tests passed.")
