"""Unit tests for the random-search baseline (RAID'26 Review C).

GAConfig(random_search=True) must turn the GA loop into best-of-N random
sampling with the SAME evaluation protocol: offspring slots are fresh
create_random_genome() calls, the Rust breeding path (search_offspring) is
bypassed, and everything else (batch evaluator, viability filter, μ+λ pool)
is untouched. No GPU/training — generator wiring is checked directly.
"""

import pytest

from wnn.ram.strategies.connectivity.framework import GAConfig
from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig
from wnn.ram.strategies.connectivity.architecture_ga import ArchitectureGAStrategy


def _strategy(random_search: bool) -> ArchitectureGAStrategy:
	arch = ArchitectureConfig(
		num_clusters=2, min_bits=4, max_bits=8, min_neurons=2, max_neurons=4,
		default_bits=4, default_neurons=2, total_input_bits=32,
		optimize_bits=True, optimize_neurons=True,
	)
	cfg = GAConfig(population_size=4, generations=3, random_search=random_search)
	return ArchitectureGAStrategy(arch_config=arch, ga_config=cfg, seed=42)


def _capture_build(strategy):
	"""Monkeypatch _build_viable_population to capture the generator_fn used."""
	captured = {}

	def fake_build(target_size, generator_fn, batch_fn=None, single_fn=None,
	               min_accuracy=None, generation=0, total_generations=0):
		captured["generator_fn"] = generator_fn
		return [(generator_fn(), None) for _ in range(target_size)]

	strategy._build_viable_population = fake_build
	strategy._batch_evaluate_fn = None
	strategy._evaluate_fn = None
	return captured


def test_random_search_uses_create_random_genome():
	s = _strategy(random_search=True)
	captured = _capture_build(s)
	offspring = s._generate_offspring(population=[], n_needed=3, threshold=0.0, generation=1)
	# Bound method comparison: __func__ identity is the strongest available check.
	assert captured["generator_fn"].__func__ is type(s).create_random_genome
	assert len(offspring) == 3


def test_random_search_bypasses_rust_breeding():
	"""With a cached_evaluator present, random_search must NOT call search_offspring."""
	class ExplodingEvaluator:
		total_input_bits = 32
		def search_offspring(self, **kwargs):
			raise AssertionError("Rust breeding path must be bypassed under random_search")

	s = _strategy(random_search=True)
	s._cached_evaluator = ExplodingEvaluator()
	_capture_build(s)
	offspring = s._generate_offspring(population=[], n_needed=2, threshold=0.0, generation=1)
	assert len(offspring) == 2


def test_ga_default_keeps_breeding_generator():
	s = _strategy(random_search=False)
	captured = _capture_build(s)
	# Seed a tiny population so tournament selection has parents. Metrics-like
	# objects only need .ce for the base path's population conversion.
	class M:  # noqa: D401 — minimal stand-in
		ce = 1.0
	genomes = [s.create_random_genome() for _ in range(3)]
	s._current_fitness_scores = [0.1, 0.2, 0.3]
	population = [(g, M()) for g in genomes]
	s._generate_offspring(population=population, n_needed=2, threshold=0.0, generation=1)
	# Default path uses the local breeding closure, not the bound method.
	assert getattr(captured["generator_fn"], "__func__", None) is not type(s).create_random_genome


def test_random_genomes_are_diverse():
	"""Fresh random genomes must not be clones of one another (sanity)."""
	s = _strategy(random_search=True)
	fps = {tuple(g.bits_per_neuron) + tuple(g.neurons_per_cluster)
	       for g in (s.create_random_genome() for _ in range(16))}
	assert len(fps) > 1


if __name__ == "__main__":
	pytest.main([__file__, "-v"])
