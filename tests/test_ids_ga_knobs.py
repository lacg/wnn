"""IDS-26 GA knobs (11/10/2026): ga_dedupe_neurons, immigrant_fraction, ga_joint_mutation.

All default OFF. Unit tests check the Python wiring with a recording fake
evaluator (no wheel needed): defaults reach Rust exactly as before (no new
RNG draws, no immigrants, same phase), and each knob reaches the right place.
The operators themselves (and the bit-identity proof) are Rust tests in
ram_core core/neighbor_search_golden.rs + neighbor_search_knob_tests.rs.
"""

import itertools

import pytest

from wnn.ram.experiments.params import KNOWN_PARAMS
from wnn.ram.experiments.experiment import ExperimentConfig, ExperimentType
from wnn.ram.strategies.connectivity.framework import GAConfig
from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig
from wnn.ram.strategies.connectivity.architecture_ga import ArchitectureGAStrategy
from wnn.ram.strategies.optimization_dimension import PhaseType
from wnn.ram.metrics import IDSMetrics


class RecordingEvaluator:
	"""Stands in for IDSEvaluator: records calls, returns fresh distinct genomes."""
	total_input_bits = 32

	def __init__(self, strategy):
		self._strategy = strategy
		self.search_calls = []
		self.immigrant_calls = []

	def search_offspring(self, population, target_count, max_attempts, accuracy_threshold, min_bits, max_bits,
	                     min_neurons, max_neurons, bits_mutation_rate, neurons_mutation_rate, crossover_rate,
	                     tournament_size, train_subset_idx, eval_subset_idx, seed, logger, generation,
	                     total_generations, return_best_n, mutable_clusters, phase_type, fitness_scores,
	                     cluster_crossover_ratio, pool_shuffle_ratio, assortative_mating_ratio, dedupe_neurons,
	                     dedupe_fill):
		self.search_calls.append(dict(target_count=target_count, phase_type=phase_type,
		                              bits_rate=bits_mutation_rate, neurons_rate=neurons_mutation_rate,
		                              dedupe=dedupe_neurons, fill=dedupe_fill, train_idx=train_subset_idx))
		genomes = [_with_metrics(self._strategy.create_random_genome()) for _ in range(target_count)]
		return type("R", (), {"genomes": genomes})()

	def evaluate_offspring_batch(self, genomes, train_subset_idx):
		self.immigrant_calls.append((list(genomes), train_subset_idx))
		return [IDSMetrics(ce=0.5, acc=0.9, f1=0.8, fpr=0.1) for _ in genomes]


def _with_metrics(genome):
	genome.metrics = IDSMetrics(ce=0.4, acc=0.9, f1=0.8, fpr=0.1)
	return genome


def _strategy(immigrant_fraction=0.0, dedupe=False, joint=False, opt=(True, True, False), fill=True):
	arch = ArchitectureConfig(
		num_clusters=1, min_bits=4, max_bits=34, min_neurons=5, max_neurons=500,
		default_bits=10, default_neurons=20, total_input_bits=32,
		optimize_neurons=opt[0], optimize_bits=opt[1], optimize_connections=opt[2],
		dedupe_neurons=dedupe, dedupe_fill=fill, joint_mutation=joint,
	)
	cfg = GAConfig(population_size=10, generations=3, immigrant_fraction=immigrant_fraction, mutation_rate=0.1)
	s = ArchitectureGAStrategy(arch_config=arch, ga_config=cfg, seed=42)
	s._ensure_rng()
	s._phase_train_idx = 3
	s._offspring_seed = 1234
	s._cached_evaluator = RecordingEvaluator(s)
	return s


def _population(s, n=6):
	return [(_with_metrics(s.create_random_genome()), IDSMetrics(ce=0.4 + i / 100, acc=0.9, f1=0.8, fpr=0.1))
	        for i in range(n)]


# ---- registry / config defaults ----------------------------------------------

def test_params_registered():
	assert {"ga_dedupe_neurons", "ga_dedupe_fill", "immigrant_fraction", "ga_joint_mutation"} <= KNOWN_PARAMS


def test_defaults_are_off():
	e = ExperimentConfig(name="t", experiment_type=ExperimentType.GA)
	assert (e.ga_dedupe_neurons, e.ga_dedupe_fill, e.immigrant_fraction, e.ga_joint_mutation) == (False, True, 0.0, False)
	a = ArchitectureConfig(num_clusters=1)
	assert (a.dedupe_neurons, a.dedupe_fill, a.joint_mutation) == (False, True, False)
	assert GAConfig().immigrant_fraction == 0.0


# ---- phase selection ---------------------------------------------------------

@pytest.mark.parametrize("opt", list(itertools.product([False, True], repeat=3)))
def test_joint_off_keeps_derived_phase_for_every_flag_combo(opt):
	s = _strategy(opt=opt)
	assert s._phase_type == s._derive_phase_type()
	assert s._phase_type != PhaseType.JOINT


def test_joint_flag_selects_joint_phase_and_all_rates():
	s = _strategy(joint=True)
	assert s._phase_type == PhaseType.JOINT == 5
	s._generate_offspring(_population(s), n_needed=8, threshold=0.0, generation=0)
	call = s._cached_evaluator.search_calls[0]
	assert call["phase_type"] == 5
	assert (call["bits_rate"], call["neurons_rate"]) == (0.1, 0.1)


def test_neurons_phase_rates_unchanged():
	s = _strategy(opt=(True, False, False))
	s._generate_offspring(_population(s), n_needed=8, threshold=0.0, generation=0)
	call = s._cached_evaluator.search_calls[0]
	assert (call["phase_type"], call["bits_rate"], call["neurons_rate"]) == (0, 0.0, 0.1)


# ---- dedupe ------------------------------------------------------------------

@pytest.mark.parametrize("flag", [False, True])
def test_dedupe_flag_reaches_rust(flag):
	s = _strategy(dedupe=flag)
	s._generate_offspring(_population(s), n_needed=8, threshold=0.0, generation=0)
	assert s._cached_evaluator.search_calls[0]["dedupe"] is flag


# ---- immigrants --------------------------------------------------------------

def test_immigrants_off_is_untouched_path():
	s = _strategy(immigrant_fraction=0.0)
	pop = _population(s)
	state_before = s._rng.getstate()
	offspring = s._generate_offspring(pop, n_needed=8, threshold=0.0, generation=0)
	ev = s._cached_evaluator
	assert ev.search_calls[0]["target_count"] == 8
	assert ev.immigrant_calls == []
	assert len(offspring) == 8
	# The fake evaluator itself draws from s._rng to build its genomes; replay
	# those draws and the state must match — i.e. the strategy drew nothing else.
	s._rng.setstate(state_before)
	for _ in range(8):
		s.create_random_genome()
	assert s._rng.getstate() == _rng_after(s, pop)


def _rng_after(s, pop):
	"""RNG state after an identical f=0 call on an identically-seeded strategy."""
	t = _strategy(immigrant_fraction=0.0)
	_population(t)  # same draws as building `pop`
	t._generate_offspring(pop, n_needed=8, threshold=0.0, generation=0)
	return t._rng.getstate()


def test_immigrants_take_their_slots_and_use_the_offspring_fold():
	s = _strategy(immigrant_fraction=0.25)
	offspring = s._generate_offspring(_population(s), n_needed=8, threshold=0.0, generation=0)
	ev = s._cached_evaluator
	assert ev.search_calls[0]["target_count"] == 6  # 8 - round(0.25 * 8)
	assert len(ev.immigrant_calls) == 1
	immigrants, fold = ev.immigrant_calls[0]
	assert fold == s._phase_train_idx == ev.search_calls[0]["train_idx"]
	assert len(immigrants) == 2
	assert len(offspring) == 8
	assert all(any(g is o for o, _ in offspring) for g in immigrants)


def test_immigrants_follow_the_phase_bands():
	s = _strategy(immigrant_fraction=0.5, opt=(True, False, False))
	s._generate_offspring(_population(s), n_needed=8, threshold=0.0, generation=0)
	for g in s._cached_evaluator.immigrant_calls[0][0]:
		assert 5 <= g.neurons_per_cluster[0] <= 500
		assert all(4 <= b <= 34 for b in g.bits_per_neuron)


def test_immigrant_fraction_validated():
	with pytest.raises(ValueError):
		_strategy(immigrant_fraction=1.5)


@pytest.mark.parametrize("fill", [True, False])
def test_dedupe_fill_reaches_rust(fill):
	s = _strategy(dedupe=True, fill=fill)
	s._generate_offspring(_population(s), n_needed=8, threshold=0.0, generation=0)
	assert s._cached_evaluator.search_calls[0]["fill"] is fill


def test_dedupe_refused_on_python_fallback():
	"""No cached evaluator → Python breeding, which has no dedupe operator."""
	s = _strategy(dedupe=True)
	s._cached_evaluator = None
	with pytest.raises(NotImplementedError):
		s._generate_offspring(_population(s), n_needed=4, threshold=0.0, generation=0)


def test_joint_immigrants_sample_bits_in_band():
	"""Under JOINT the phase evolves bits too, so immigrants sample bits in the
	band rather than inheriting the population mode (GA-Neurons flags)."""
	s = _strategy(immigrant_fraction=1.0, joint=True, opt=(True, False, False))
	pop = _population(s)
	s._generate_offspring(pop, n_needed=8, threshold=0.0, generation=0)
	immigrant_bits = {b for g in s._cached_evaluator.immigrant_calls[0][0] for b in g.bits_per_neuron}
	assert len(immigrant_bits) > 1 and all(4 <= b <= 34 for b in immigrant_bits)
