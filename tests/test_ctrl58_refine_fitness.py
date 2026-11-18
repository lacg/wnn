"""CTRL-58 (Luiz 18/11/2026): the staged REFINEMENT fitness — default off, bit-identical.

Pinned:
  * OFF: a controller GA with the refine mixin but no latch runs EXACTLY the base loop
    (history, final population, every patience check), and check_magnitude without
    steady is the legacy err+stable check;
  * the trigger fires exactly when >= frac of the TRAIN pool meets all four bounds for K
    consecutive generations (or the trigger stage starts), and the latch never reverts;
  * the tightened four-way gate ranks every infeasible genome behind every feasible one,
    infeasible ordered by the summed normalised violation;
  * the weights are applied as specified (steady .50 / reward .25 / tail .15 / pos .10;
    stable in the gate only);
  * at the switch the frozen-best mark and the patience counter reset, and magnitude
    patience watches steady from then on;
  * --refine-select, the launch refusals, provenance and the params registry.
Needs the installed ram_controller (fitness_combine); no GPU, no sim.
"""
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wnn.control.refine_config import RefineConfig, refine_token, validate_refine_args
from wnn.control.refine_fitness import FitnessCalculatorControllerRefine
from wnn.control.refine_latch import RefineLatch
from wnn.control.refine_mixin import RefineFitnessMixin
from wnn.ram.fitness import FitnessCalculatorControllerHarmonic as FC
from wnn.ram.metrics import ControllerMetrics
from wnn.ram.strategies.connectivity.framework.early_stopping import (
	EarlyStoppingConfig, EarlyStoppingTracker)
from wnn.ram.strategies.connectivity.generic_ga import GenericGAStrategy

CFG = RefineConfig(w_tail=0.0)          # the flyable default (no train-pool tail row yet)
_REAL_CHECK = EarlyStoppingTracker.check_magnitude   # captured once: _run re-patches per call


def _m(stable=1.0, err=1.0, steady=0.5, pos=0.3, reward=-1.0, tail=None):
	return ControllerMetrics(reward=reward, stable_rate=stable, mean_attitude_error_deg=err,
	                         mean_steady_error_deg=steady, mean_position_error_m=pos,
	                         episode_err_p99_deg=tail)


def _refine(cfg=CFG, with_tail=False):
	return FitnessCalculatorControllerRefine(cfg, zrank_clamp=3.0, zrank_mad_floor=False,
	                                         zrank_pos_floor=0.0, pos_metric="pos", with_tail=with_tail)


# ---- a tiny deterministic GA over 2-vectors (pure Python; ram_controller ranks) --------

def _metrics_of(g) -> ControllerMetrics:
	a, b = g
	err = 0.5 + 3.0 * a
	return _m(stable=1.0 if err < 2.0 else 0.9, err=err, steady=0.3 + 2.0 * b,
	          pos=0.30, reward=-(err ** 2) / 10.0)


class _ToyGA(GenericGAStrategy):
	def __init__(self, config, seed=7):
		super().__init__(config=config, seed=seed, logger=lambda _m: None)
		self._stage_name = "neurons"

	@property
	def name(self) -> str:
		return "ToyGA"

	def _counter_rng(self):
		from wnn.control import _accel
		return _accel.require_accel()

	def clone_genome(self, g):
		return tuple(g)

	def mutate_genome(self, g, rate):
		return tuple(min(1.0, max(0.0, x + self._rng.gauss(0.0, 0.1))) for x in g)

	def crossover_genomes(self, p1, p2):
		return tuple((x + y) / 2.0 for x, y in zip(p1, p2))

	def create_random_genome(self):
		return (self._rng.random(), self._rng.random())


class _ToyRefineGA(RefineFitnessMixin, _ToyGA):
	pass


def _ga_config(gens=8):
	from wnn.control.ga_strategy import default_controller_ga_config
	c = default_controller_ga_config(
		population_size=10, generations=gens, weight_err_sq=0.2557, weight_stable=0.2045,
		weight_steady=0.358, weight_pos=0.1818, aggregation="zscore",
		gate_stable_min=0.70, gate_err_max=8.0)
	c.patience, c.check_interval, c.magnitude_aware_patience = 50, 1, True
	return c


def _run(strategy, monkeypatch):
	"""optimize() on a fixed seed population; returns (result, check_magnitude calls)."""
	calls = []
	real = _REAL_CHECK

	def spy(self, iteration, err, stable, steady_deg=None, steady_floor=None):
		calls.append((iteration, err, stable, steady_deg, steady_floor, self._patience_counter,
		              None if self._mag_watermarks is None else dict(self._mag_watermarks)))
		return real(self, iteration, err, stable, steady_deg=steady_deg, steady_floor=steady_floor)
	monkeypatch.setattr(EarlyStoppingTracker, "check_magnitude", spy)
	rng = random.Random(3)
	seed = [(rng.random(), rng.random()) for _ in range(10)]
	batch = lambda gs, min_accuracy=0.0, generation=None, total_generations=None: [_metrics_of(g) for g in gs]
	res = strategy.optimize(evaluate_fn=lambda g: -_metrics_of(g).reward,
	                        batch_evaluate_fn=batch, initial_population=seed)
	return res, calls


# ---- 1. OFF ⇒ identical ---------------------------------------------------------------

def test_off_is_the_base_loop(monkeypatch):
	base, base_calls = _run(_ToyGA(_ga_config()), monkeypatch)
	mixed, mixed_calls = _run(_ToyRefineGA(_ga_config()), monkeypatch)
	assert mixed.history == base.history
	assert mixed.final_population == base.final_population
	assert [m.reward for m in mixed.population_metrics] == [m.reward for m in base.population_metrics]
	assert mixed_calls == base_calls
	assert all(c[3] is None and c[4] is None for c in mixed_calls), "steady never watched when off"


def test_check_magnitude_without_steady_is_legacy():
	def tracker():
		t = EarlyStoppingTracker(EarlyStoppingConfig(patience=3, check_interval=1, magnitude_aware=True),
		                         lambda _m: None, "T")
		t.reset(1.0)
		return t
	a, b = tracker(), tracker()
	for i, (e, s) in enumerate([(5.0, 0.5), (4.0, 0.6), (4.0, 0.6), (3.9, 0.6)]):
		assert a.check_magnitude(i, e, s) == b.check_magnitude(i, e, s, steady_deg=None, steady_floor=0.15)
		assert a.state() == b.state()


def test_flags_default_off_and_token_empty():
	from wnn.control.phased_ga import build_arg_parser, build_refine_latch
	args = build_arg_parser().parse_args([])
	assert not args.refine_fitness and not args.refine_select
	assert build_refine_latch(args) is None
	assert refine_token(args) == ""
	validate_refine_args(args)    # off: never refuses


# ---- 2. trigger + latch --------------------------------------------------------------

def _pool(n_ok, n_bad):
	return [_m(1.0, 1.0, 0.5, 0.3)] * n_ok + [_m(1.0, 2.5, 0.5, 0.3)] * n_bad


def _latch(cfg=CFG, log=None):
	lines = [] if log is None else log
	return RefineLatch(cfg, lambda: _refine(cfg), pos_metric="pos", log=lines.append), lines


def test_trigger_fires_after_k_consecutive_gens_only():
	latch, lines = _latch(replace(CFG, trigger_gens=3, trigger_stage="none"))
	latch.stage_start("neurons")
	assert not latch.observe("neurons", 0, _pool(8, 2))      # streak 1 (80% meets 80%)
	assert not latch.observe("neurons", 1, _pool(8, 2))      # streak 2
	assert not latch.observe("neurons", 2, _pool(7, 3))      # 70% < 80%: streak broken
	assert not latch.observe("neurons", 3, _pool(9, 1))      # 1
	assert not latch.observe("neurons", 4, _pool(10, 0))     # 2
	assert latch.observe("neurons", 5, _pool(8, 2))          # 3 → fires on THIS gen
	assert latch.fired and latch.fired_gen == 6 and latch.fired_stage == "neurons"
	assert lines == [l for l in lines if l.startswith("[refine] trigger fired gen 6 stage NEURONS reason ")]
	assert len(lines) == 1


@pytest.mark.parametrize("bad", [dict(stable=0.995), dict(err=1.6), dict(steady=1.05), dict(pos=0.41)])
def test_each_bound_must_hold(bad):
	latch, _ = _latch(replace(CFG, trigger_gens=1, trigger_frac=1.0, trigger_stage="none"))
	latch.stage_start("bits")
	m = dict(stable=1.0, err=1.0, steady=0.5, pos=0.3)
	assert not latch.observe("bits", 0, [_m(**{**m, **bad})] * 4)
	assert latch.observe("bits", 1, [_m(**m)] * 4)


def test_latch_never_reverts_and_carries_into_later_stages():
	latch, lines = _latch(replace(CFG, trigger_gens=1, trigger_stage="none"))
	latch.stage_start("neurons")
	assert latch.observe("neurons", 0, _pool(10, 0))
	for g in range(1, 5):
		assert not latch.observe("neurons", g, _pool(0, 10)), "no second fire, no revert"
	assert latch.fired
	assert latch.stage_start("bits") and latch.stage_start("memory")
	assert len(lines) == 1


def test_stage_start_trigger_and_streak_restart():
	latch, lines = _latch(replace(CFG, trigger_gens=2))
	assert not latch.stage_start("neurons")
	assert not latch.observe("neurons", 0, _pool(10, 0))     # streak 1 ...
	assert not latch.stage_start("connections")              # ... restarts at a new stage
	assert not latch.observe("connections", 0, _pool(10, 0))
	assert latch.stage_start("memory")                       # default trigger stage
	assert lines == ["[refine] trigger fired gen 0 stage MEMORY reason stage start (MEMORY)"]


# ---- 3. tightened gate + 4. weights ---------------------------------------------------

def _violation(m, c=CFG):
	v = lambda x, b: max(0.0, x - b) / b
	return (max(0.0, c.gate_stable - m.stable_rate) / c.gate_stable + v(m.mean_attitude_error_deg, c.gate_err)
	        + v(m.mean_steady_error_deg, c.gate_steady) + v(m.mean_position_error_m, c.gate_pos))


def test_infeasible_rank_behind_every_feasible_ordered_by_violation():
	feas = [_m(1.0, 1.0, 0.5, 0.30, -1.0), _m(0.99, 1.4, 0.9, 0.38, -1.3),
	        _m(1.0, 0.8, 0.7, 0.25, -0.8), _m(1.0, 1.2, 0.4, 0.35, -1.1)]
	infeas = [_m(0.98, 1.0, 0.5, 0.30), _m(1.0, 1.6, 0.5, 0.30), _m(1.0, 1.0, 1.3, 0.30),
	          _m(1.0, 1.0, 0.5, 0.60), _m(0.90, 3.0, 2.0, 0.90)]
	pool = [infeas[0], feas[0], infeas[1], feas[1], infeas[2], feas[2], infeas[3], feas[3], infeas[4]]
	f = _refine().fitness(pool)
	fs = [f[i] for i, m in enumerate(pool) if m in feas]
	inf = [(f[i], _violation(m)) for i, m in enumerate(pool) if m in infeas]
	assert max(fs) < min(s for s, _ in inf)
	assert [v for _, v in sorted(inf)] == sorted(v for _, v in inf), "Deb's rule 3: less violation first"
	# Feasible members ranked over the FEASIBLE SUBSET only (the wheel's normalisation).
	assert fs == _refine().fitness([m for m in pool if m in feas])


def test_weights_are_the_specified_columns():
	from wnn.control._accel import fitness_combine
	cfg = RefineConfig()                                    # spec weights incl. tail .15
	pool = [_m(1.0, 1.0, 0.50, 0.30, -1.0, 2.0), _m(1.0, 1.2, 0.45, 0.32, -1.2, 2.6),
	        _m(0.99, 0.9, 0.70, 0.28, -0.9, 1.9), _m(1.0, 1.4, 0.60, 0.36, -1.4, 3.1),
	        _m(1.0, 1.1, 0.55, 0.31, -1.1, 2.2)]
	cols = ([m.reward for m in pool] + [m.mean_steady_error_deg for m in pool]
	        + [m.mean_position_error_m for m in pool] + [m.episode_err_p99_deg for m in pool])
	want = list(fitness_combine(cols, len(pool), [0.25, 0.50, 0.10, 0.15],
	                            [True, False, False, False], "zscore", 3.0))
	assert _refine(cfg, with_tail=True).fitness(pool) == want
	# Without the tail it is exactly the base ZRank on reward/steady/pos, stable unweighted.
	no_tail = FC(weight_err_sq=0.25, weight_steady=0.50, weight_pos=0.10, aggregation="zscore")
	assert _refine(cfg, with_tail=False).fitness(pool) == no_tail.fitness(pool)


def test_stable_lives_in_the_gate_only():
	a = [_m(1.0, 1.0, 0.5, 0.3, -1.0), _m(1.0, 1.2, 0.6, 0.31, -1.2), _m(1.0, 0.9, 0.7, 0.29, -0.9)]
	b = [replace(a[0], stable_rate=0.99)] + a[1:]
	assert _refine().fitness(a) == _refine().fitness(b)


def test_missing_tail_is_refused_not_dropped():
	with pytest.raises(ValueError, match="episode_err_p99_deg"):
		_refine(RefineConfig(), with_tail=True).fitness([_m(), _m(err=1.2)])


def test_name_carries_the_gate_and_weights():
	n = _refine(RefineConfig(), with_tail=True).name
	assert n.startswith("Refine[gate=(st>=0.99,err<=1.5°,steady<=1°,pos<=0.4m)]ZRank(")
	assert "steady=0.5" in n and "pos=0.1" in n and "tail_p99=0.15" in n and "stable" not in n


# ---- 5. the switch inside the GA loop ---------------------------------------------------

def _wired(strategy, cfg, lines):
	strategy._refine_latch = RefineLatch(cfg, lambda: _refine(cfg), pos_metric="pos", log=lines.append)
	strategy._refine_active = False
	return strategy


def test_switch_resets_patience_and_watches_steady(monkeypatch):
	loose = replace(CFG, trigger_frac=0.5, trigger_stable=0.5, trigger_err=10.0, trigger_steady=10.0,
	                trigger_pos=1.0, trigger_gens=2, trigger_stage="none")
	lines, resets = [], []
	real_reset = EarlyStoppingTracker.reset
	monkeypatch.setattr(EarlyStoppingTracker, "reset",
	                    lambda self, f: (resets.append(f), real_reset(self, f))[1])
	ga = _wired(_ToyRefineGA(_ga_config()), loose, lines)
	res, calls = _run(ga, monkeypatch)
	assert lines and lines[0].startswith("[refine] trigger fired gen 2 stage NEURONS")
	assert len(resets) == 2, "setup + exactly one reset at the switch"
	before = [c for c in calls if c[0] < 1]
	after = [c for c in calls if c[0] >= 1]
	assert all(c[3] is None for c in before), "steady unwatched before the switch"
	assert all(c[3] is not None and c[4] == loose.patience_steady_floor for c in after)
	assert after[0][5] == 0 and after[0][6] is None, "counter and watermarks restart at the switch"
	assert "steady" in after[-1][6]
	assert res.history[2][1] == resets[1], "the frozen-best mark restarts in the new frame"
	assert ga._fitness_calculator.name.startswith("Refine[")


def test_trigger_stage_ranks_refined_from_gen0(monkeypatch):
	lines = []
	ga = _wired(_ToyRefineGA(_ga_config(gens=3)), replace(CFG, trigger_stage="neurons"), lines)
	_, calls = _run(ga, monkeypatch)
	assert lines == ["[refine] trigger fired gen 0 stage NEURONS reason stage start (NEURONS)"]
	assert all(c[3] is not None for c in calls)
	assert ga._fitness_calculator.name.startswith("Refine[")


# ---- stage-select, refusals, provenance, registry ------------------------------------

def _args(*argv):
	from wnn.control.phased_ga import build_arg_parser
	return build_arg_parser().parse_args(list(argv))


def test_refine_select_uses_the_refine_calculator_without_tail():
	from wnn.control.phased_ga import _stage_select_calculator
	pool = [_m(1.0, 1.0, 0.5, 0.3, -1.0), _m(1.0, 1.2, 0.6, 0.31, -1.2), _m(0.9, 0.9, 0.7, 0.29, -0.9)]
	calc = _stage_select_calculator(_args("--refine-select", "--select-tail", "p99"))
	assert isinstance(calc, FitnessCalculatorControllerRefine) and calc.weight_tail == 0.0
	default = _stage_select_calculator(_args("--fit-weight-steady", "0.358", "--fit-aggregation", "zscore"))
	assert not isinstance(default, FitnessCalculatorControllerRefine)
	assert default.fitness(pool) == FC(weight_err_sq=1.0, weight_steady=0.358, aggregation="zscore").fitness(pool)


@pytest.mark.parametrize("argv, why", [
	(["--refine-fitness"], "refine-w-tail"),
	(["--refine-fitness", "--refine-w-tail", "0", "--strategy", "ts"], "ts"),
	(["--refine-fitness", "--refine-w-tail", "0", "--resume-from-emergency", "x"], "resume"),
	(["--refine-select"], "p99"),
	(["--refine-fitness", "--refine-w-tail", "0", "--refine-trigger-frac", "0"], "frac"),
	(["--refine-fitness", "--refine-w-tail", "0", "--refine-gate-stable", "99"], "FRACTION"),
])
def test_launch_refusals(argv, why):
	with pytest.raises(SystemExit, match=why):
		validate_refine_args(_args(*argv))


def test_provenance_and_params_registry():
	from wnn.control.provenance import collect_provenance
	from wnn.control.flow_adapter import _refuse_unbuilt_keys
	from wnn.ram.experiments.params import KNOWN_PARAMS
	assert "refine=" not in collect_provenance("x").line()
	tok = refine_token(_args("--refine-fitness", "--refine-w-tail", "0"))
	assert f"refine={tok} " in collect_provenance("x", refine=tok).line()
	for k in ("controller_refine_fitness", "controller_refine_select", "controller_refine_w_tail",
	          "controller_refine_trigger_stage", "controller_refine_patience_steady_floor"):
		assert k in KNOWN_PARAMS
	with pytest.raises(ValueError):
		_refuse_unbuilt_keys({"controller_refine_fitness": True})
