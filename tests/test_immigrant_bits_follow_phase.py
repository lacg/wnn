"""IDS-17 (16/09/2026): GA immigrants must follow the phase, not default_bits.

A neurons phase (optimize_bits=False) at min_bits=34 used to breed immigrants
with the hard-coded ArchitectureConfig.default_bits (=8) — 34/8 hybrids that the
tracker then reported as "27" (rounded mean). The fix routes every random-genome
builder AND ClusterGenome._mutate_neurons through ONE rule (mode_or_midpoint:
the reference population's per-cluster mode, else the band midpoint, clamped —
the Rust neurons-operator convention), validates the defaults instead of
clamping them, and tracks heterogeneous clusters exactly. No GPU, no training.
"""

import json
import random

import pytest

from wnn.ram.genome import ClusterGenome, AdaptiveClusterConfig, PhaseType, mode_or_midpoint
from wnn.ram.strategies.connectivity.framework import GAConfig
from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig
from wnn.ram.strategies.connectivity.architecture_ga import ArchitectureGAStrategy
from wnn.ram.experiments.experiment import _phase_defaults


def _genome(bits: list[int], neurons: list[int] | None = None) -> ClusterGenome:
	"""Single-cluster genome (ids_single_cluster) unless `neurons` says otherwise."""
	return ClusterGenome(bits_per_neuron=bits, neurons_per_cluster=neurons or [len(bits)], connections=None)


def _strategy(
	min_bits: int, max_bits: int, optimize_bits: bool, bits_grid: list[int] | None = None,
	num_clusters: int = 1, immigrant_fraction: float = 0.0,
) -> ArchitectureGAStrategy:
	arch = ArchitectureConfig(
		num_clusters=num_clusters, min_bits=min_bits, max_bits=max_bits,
		min_neurons=4, max_neurons=40, default_bits=(min_bits + max_bits) // 2, default_neurons=10,
		optimize_bits=optimize_bits, optimize_neurons=not optimize_bits,
		bits_grid=bits_grid, total_input_bits=128,
	)
	cfg = GAConfig(population_size=4, generations=2, immigrant_fraction=immigrant_fraction)
	return ArchitectureGAStrategy(arch_config=arch, ga_config=cfg, seed=7)


# ---- (a) neurons phase: bits follow the population -------------------------

def test_neurons_phase_immigrant_takes_population_bits():
	s = _strategy(min_bits=34, max_bits=64, optimize_bits=False)
	population = [_genome([34] * 30) for _ in range(6)]
	for _ in range(8):
		g = s.create_random_genome(reference=population)
		assert set(g.bits_per_neuron) == {34}, g.bits_per_neuron
		assert 4 <= g.neurons_per_cluster[0] <= 40


def test_neurons_phase_no_population_uses_band_midpoint():
	s = _strategy(min_bits=34, max_bits=34, optimize_bits=False)
	g = s.create_random_genome(reference=None)
	assert set(g.bits_per_neuron) == {34}
	s2 = _strategy(min_bits=10, max_bits=20, optimize_bits=False)
	assert set(s2.create_random_genome().bits_per_neuron) == {15}


def test_neurons_phase_mixed_population_takes_mode():
	s = _strategy(min_bits=4, max_bits=64, optimize_bits=False)
	population = [_genome([34] * 27 + [8] * 7), _genome([34] * 20 + [8] * 3), _genome([12] * 10)]
	g = s.create_random_genome(reference=population)
	assert set(g.bits_per_neuron) == {34}


def test_neurons_phase_mode_is_per_cluster():
	s = _strategy(min_bits=4, max_bits=64, optimize_bits=False, num_clusters=2)
	population = [_genome([34] * 5 + [12] * 5, neurons=[5, 5]) for _ in range(3)]
	g = s.create_random_genome(reference=population)
	assert set(g.bits_for_cluster(0)) == {34} and set(g.bits_for_cluster(1)) == {12}


def test_population_bits_outside_band_are_clamped():
	s = _strategy(min_bits=34, max_bits=40, optimize_bits=False)
	assert set(s.create_random_genome(reference=[_genome([8] * 10)]).bits_per_neuron) == {34}
	assert set(s.create_random_genome(reference=[_genome([96] * 10)]).bits_per_neuron) == {40}


def test_generic_ga_immigrant_path_passes_live_population():
	"""generic_ga's immigrant slot must hand the population to the builder."""
	s = _strategy(min_bits=34, max_bits=64, optimize_bits=False, immigrant_fraction=1.0)
	s._rng = random.Random(3)

	class M:
		ce = 1.0

	population = [(_genome([34] * 30), M()) for _ in range(4)]
	s._build_viable_population = (lambda target_size, generator_fn, batch_fn=None, single_fn=None,
	                              min_accuracy=None, generation=0, total_generations=0:
	                              [(generator_fn(), None) for _ in range(target_size)])
	s._batch_evaluate_fn = None
	s._evaluate_fn = None
	offspring = s._generate_offspring(population=population, n_needed=5, threshold=0.0, generation=1)
	assert len(offspring) == 5
	assert all(set(g.bits_per_neuron) == {34} for g, _ in offspring)


# ---- (b) bits phase: sample from bits_grid ∩ band -------------------------

def test_bits_phase_samples_from_grid_within_band():
	s = _strategy(min_bits=10, max_bits=20, optimize_bits=True, bits_grid=[4, 8, 12, 16, 20, 24])
	seen = set()
	for _ in range(20):
		seen.update(s.create_random_genome().bits_per_neuron)
	assert seen <= {12, 16, 20} and len(seen) > 1


def test_bits_phase_empty_grid_falls_back_to_uniform_band():
	s = _strategy(min_bits=10, max_bits=20, optimize_bits=True, bits_grid=[])
	seen = set()
	for _ in range(20):
		seen.update(s.create_random_genome().bits_per_neuron)
	assert seen <= set(range(10, 21)) and len(seen) > 1


# ---- (c) validate, do not clamp --------------------------------------------

def test_post_init_rejects_default_bits_outside_band():
	with pytest.raises(ValueError, match="default_bits=8.*min_bits=34"):
		ArchitectureConfig(num_clusters=1, min_bits=34, max_bits=64, default_bits=8)
	with pytest.raises(ValueError, match="default_neurons=5.*min_neurons=10"):
		ArchitectureConfig(num_clusters=1, min_neurons=10, max_neurons=40, default_neurons=5)


def test_post_init_passes_on_derived_defaults():
	seeds = [_genome([34] * 30), _genome([34] * 28 + [8] * 2)]
	default_bits, default_neurons = _phase_defaults(seeds, 34, 64, 4, 40)
	assert (default_bits, default_neurons) == (34, 30)
	cfg = ArchitectureConfig(num_clusters=1, min_bits=34, max_bits=64, min_neurons=4, max_neurons=40,
	                         default_bits=default_bits, default_neurons=default_neurons)
	assert cfg.default_bits == 34
	# No seeds at all → band midpoints, still inside the band.
	assert _phase_defaults([], 34, 64, 4, 40) == (49, 22)


# ---- shared rule ------------------------------------------------------------

def test_mode_or_midpoint_rule():
	assert mode_or_midpoint([], 34, 64) == 49
	assert mode_or_midpoint([34, 34, 8], 4, 64) == 34
	assert mode_or_midpoint([8, 34], 4, 64) == 8      # tie → smallest, deterministic
	assert mode_or_midpoint([8, 8, 8], 34, 64) == 34  # clamped up
	with pytest.raises(ValueError):
		mode_or_midpoint([1], 10, 5)


def test_mutate_neurons_grows_with_cluster_mode_bits():
	g = _genome([34] * 27 + [8] * 7)
	cfg = AdaptiveClusterConfig(min_bits=34, max_bits=64, min_neurons=4, max_neurons=200)
	grown = None
	for seed in range(50):
		cand = g.mutate(PhaseType.NEURONS, 1.0, cfg, 128, random.Random(seed))
		if cand.total_neurons > g.total_neurons:
			grown = cand
			break
	assert grown is not None
	assert grown.bits_per_neuron[:34] == g.bits_per_neuron
	assert set(grown.bits_per_neuron[34:]) == {34}


# ---- (d) tracked genomes serialize exactly ---------------------------------

def test_tracked_genome_serialization_is_exact_for_hybrid():
	s = _strategy(min_bits=4, max_bits=64, optimize_bits=False)
	tiers = json.loads(s.genome_to_config(_genome([34] * 27 + [8] * 7)).to_json())
	assert [(t["neurons"], t["bits"], t["clusters"], t["start_cluster"], t["end_cluster"]) for t in tiers] \
		== [(27, 34, 1, 0, 1), (7, 8, 1, 0, 1)]
	assert 27 not in {t["bits"] for t in tiers}  # the rounded mean is gone


def test_tracked_genome_homogeneous_shape_unchanged():
	s = _strategy(min_bits=4, max_bits=64, optimize_bits=False, num_clusters=3)
	gc = s.genome_to_config(_genome([34] * 30, neurons=[10, 10, 10]))
	assert json.loads(gc.to_json()) == [
		{"tier": 0, "clusters": 3, "neurons": 10, "bits": 34, "start_cluster": 0, "end_cluster": 3}]
	assert gc.total_neurons == 30 and gc.total_clusters == 3


def test_db_to_ga_checkpoint_reads_both_shapes():
	import importlib.util
	from pathlib import Path
	spec = importlib.util.spec_from_file_location(
		"db_to_ga_checkpoint", Path(__file__).resolve().parent.parent / "scripts" / "db_to_ga_checkpoint.py")
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	exact = json.dumps([
		{"tier": 0, "clusters": 1, "neurons": 27, "bits": 34, "start_cluster": 0, "end_cluster": 1},
		{"tier": 0, "clusters": 1, "neurons": 7, "bits": 8, "start_cluster": 0, "end_cluster": 1}])
	assert mod.tiers_to_arch(exact) == ([34] * 27 + [8] * 7, [34])
	legacy = json.dumps([{"tier": 0, "clusters": 2, "neurons": 3, "bits": 16}])
	assert mod.tiers_to_arch(legacy) == ([16] * 6, [3, 3])


if __name__ == "__main__":
	pytest.main([__file__, "-v"])
