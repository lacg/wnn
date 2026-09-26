"""Shared GA template: stop-and-resume must CONTINUE the run, not re-seed it (WNN-1).

Both substrates (IDS ArchitectureGAStrategy, controller ControllerArchGAStrategy)
checkpoint and resume through GenericGAStrategy / OptimizationTemplate. This pins
the contract on a toy GA driven through the REAL optimize() loop:

  * a run stopped at generation k and resumed from the checkpoint the shared path
    stamped is identical, generation for generation, to the uninterrupted run —
    same best-fitness history, same final population, same early-stopper and
    adaptive-scaler state. That needs (a) the generation counter, (b) the full
    tracker + scaler snapshots and (c) per-generation counter-RNG seeds, i.e. no
    RNG state in the checkpoint at all;
  * the same seed twice gives the same run (IDS offspring used a wall-clock seed);
  * a different seed gives a different run (the seed actually reaches the draws);
  * the snapshots round-trip on their own, including a FRACTIONAL patience counter
    that the old restore() truncated to int.

Needs a worker wheel exporting counter_rng_draw_u64 (ABI >= 14).
"""

import copy

import pytest

from wnn.ram.metrics import IDSMetrics as Metrics
from wnn.ram.strategies.connectivity.framework.configs import GAConfig
from wnn.ram.strategies.connectivity.framework import AdaptiveScaler
from wnn.ram.strategies.connectivity.framework.adaptive_scaling import AdaptiveLevel
from wnn.ram.strategies.connectivity.generic_ga import GenericGAStrategy
from wnn.ram.strategies.phased import PhaseCheckpoint

GENES, TARGET = 12, 7


def _needs_counter_rng():
	import ram_accelerator
	if not hasattr(ram_accelerator, "counter_rng_draw_u64"):
		pytest.skip("installed worker wheel predates counter_rng exports (ABI < 14)")


class ToyGenome:
	def __init__(self, vals):
		self.vals = list(vals)

	def key(self):
		return tuple(self.vals)


def _metrics(g: ToyGenome) -> Metrics:
	err = sum((v - TARGET) ** 2 for v in g.vals) / GENES
	acc = sum(v == TARGET for v in g.vals) / GENES
	return Metrics(ce=float(err), acc=float(acc))


class MemoryCheckpoints:
	"""Stands in for PhasedCheckpointManager: keeps every stamped checkpoint."""

	def __init__(self):
		self.by_gen = {}

	def maybe_save(self, generation, ckpt):
		self.by_gen[generation] = copy.deepcopy(ckpt)

	def save(self, ckpt):
		self.by_gen[ckpt.iterations_run] = copy.deepcopy(ckpt)


class ToyGA(GenericGAStrategy):
	"""Every random choice goes through self._rng — the template's generator."""

	@property
	def name(self) -> str:
		return "ToyGA-Resume"

	def clone_genome(self, g):
		return ToyGenome(g.vals)

	def mutate_genome(self, g, rate):
		return ToyGenome([self._rng.randint(0, 9) if self._rng.random() < rate else v for v in g.vals])

	def crossover_genomes(self, p1, p2):
		return ToyGenome([a if self._rng.random() < 0.5 else b for a, b in zip(p1.vals, p2.vals)])

	def create_random_genome(self):
		return ToyGenome([self._rng.randint(0, 9) for _ in range(GENES)])

	def _build_checkpoint(self, generation, genomes, ctx, complete):
		return PhaseCheckpoint(phase_key="toy", phase_name=self.name, strategy_type="GA",
		                       best_genome=ctx.get("best_genome"),
		                       final_population=[ToyGenome(g.vals) for g in genomes])


def _config():
	# patience/check_interval small so the early-stopper and scaler actually move
	return GAConfig(population_size=16, generations=10, patience=3, check_interval=2,
	                mutation_rate=0.2, elitism_pct=0.2)


def _batch_eval(genomes, min_accuracy=0.0, generation=None, total_generations=None):
	return [_metrics(g) for g in genomes]


def _initial_population(seed: int):
	import random
	r = random.Random(seed)
	return [ToyGenome([r.randint(0, 9) for _ in range(GENES)]) for _ in range(16)]


def _run(seed, initial_population, resume_ckpt=None):
	ga = ToyGA(_config(), seed=seed)
	ga._checkpoint_mgr = MemoryCheckpoints()
	if resume_ckpt is not None:
		ga.resume_state_from_checkpoint(resume_ckpt)
	res = ga.optimize(initial_population=initial_population, batch_evaluate_fn=_batch_eval)
	return ga, res


def _population_keys(res):
	return sorted(g.key() for g in (res.final_population or []))


def _history_from(res, start_gen):
	return [(g, round(f, 12)) for g, f in res.history if g >= start_gen]


@pytest.mark.parametrize("run_seed,pop_seed", [(1234, 99), (77, 5), (31337002, 42)])
def test_resume_at_every_generation_is_bit_identical_to_straight_through(run_seed, pop_seed):
	"""Stop at EVERY generation the straight run checkpointed and resume from it."""
	_needs_counter_rng()
	init = _initial_population(pop_seed)
	straight_ga, straight = _run(run_seed, init)
	stamped = sorted(straight_ga._checkpoint_mgr.by_gen)
	assert stamped, "the shared path stamped no checkpoint"
	for resume_at in stamped:
		ckpt = straight_ga._checkpoint_mgr.by_gen[resume_at]
		assert ckpt.iterations_run == resume_at
		assert set(ckpt.extra["ga_state"]) == {"stopper", "scaler", "incumbent"}

		_, resumed = _run(run_seed, [ToyGenome(g.vals) for g in ckpt.final_population], resume_ckpt=ckpt)

		where = f"seed {run_seed}, resumed at gen {resume_at}"
		assert _population_keys(resumed) == _population_keys(straight), where
		assert _history_from(resumed, resume_at + 1) == _history_from(straight, resume_at + 1), where
		assert resumed.iterations_run == straight.iterations_run, where
		assert resumed.best_genome.key() == straight.best_genome.key(), where


def test_same_seed_same_run_different_seed_different_run():
	_needs_counter_rng()
	init = _initial_population(7)
	_, a = _run(555, init)
	_, b = _run(555, init)
	_, c = _run(556, init)
	assert _population_keys(a) == _population_keys(b)
	assert a.history == b.history
	assert _population_keys(a) != _population_keys(c)


def test_stages_sharing_a_seed_do_not_share_draws():
	_needs_counter_rng()
	ga = ToyGA(_config(), seed=1)
	other = ToyGA(_config(), seed=1)
	other.__class__ = type("ToyGAOther", (ToyGA,), {"name": property(lambda self: "ToyGA-Other")})
	assert ga._derive_seed(3, ga.RNG_STREAM_PYTHON) != other._derive_seed(3, ga.RNG_STREAM_PYTHON)
	assert ga._derive_seed(3, ga.RNG_STREAM_PYTHON) != ga._derive_seed(3, ga.RNG_STREAM_NUMPY)
	assert ga._derive_seed(3, ga.RNG_STREAM_PYTHON) != ga._derive_seed(4, ga.RNG_STREAM_PYTHON)


def test_early_stopper_snapshot_keeps_fractional_patience_and_watermarks():
	ga = ToyGA(_config(), seed=1)
	stopper = ga._setup_early_stopping(1.0)
	stopper._patience_counter = 2.5
	stopper._mag_watermarks = {"err": 1.42, "stable": 0.988}
	stopper._prev_best = 0.75
	snap = stopper.state()

	fresh = ga._setup_early_stopping(9.0)
	fresh.restore_state(snap)
	assert fresh._patience_counter == 2.5
	assert fresh._mag_watermarks == {"err": 1.42, "stable": 0.988}
	assert fresh._prev_best == 0.75
	assert fresh._initial_fitness == 1.0
	assert fresh.state() == snap


def test_pre_wnn1_checkpoint_restores_counter_only():
	ga = ToyGA(_config(), seed=1)
	legacy = PhaseCheckpoint(phase_key="toy", phase_name="x", strategy_type="GA",
	                         iterations_run=5, extra={"patience_counter": 2})
	ga.resume_state_from_checkpoint(legacy)
	assert ga._resume_start_gen == 5
	assert ga._resume_ga_state == {"stopper": {"patience_counter": 2}}
	stopper = ga._setup_early_stopping(1.0)
	stopper.restore_state(ga._resume_ga_state["stopper"])
	assert stopper._patience_counter == 2.0


def test_adaptive_scaler_snapshot_round_trip():
	s = AdaptiveScaler(base_population=50, base_mutation=0.1, name="t")
	s.update(AdaptiveLevel.WARNING)
	snap = s.state()
	fresh = AdaptiveScaler(base_population=50, base_mutation=0.1, name="t")
	fresh.restore_state(snap)
	assert (fresh.level, fresh.population, fresh.mutation_rate) == (s.level, s.population, s.mutation_rate)
	assert fresh.population == 57 and abs(fresh.mutation_rate - 0.15) < 1e-12
