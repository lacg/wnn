"""Stage selection: the union ranking must see top-K of EVERY stage, and pop[0]
must always compete.

Regression cover for the 09/08/2026 defects (see _select_headline_stage and
ControllerOrchestrator._release_prior_populations):

  A  only one candidate per stage reached the ranking, compressing every metric
     onto 3 rank slots, so a .20-weight term could outvote a .40-weight one;
  C  that one candidate was `best_genome`, while the published triple and the
     exported committee member are `final_population[0]` — measured different in
     3/3 stages of CMT_mpc_cf21_brushless_L4C_s31337004.

These tests monkeypatch the val scorer, so they run without a simulator and
without taking the box's one controller slot.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

from wnn.control import phased_ga as P
from wnn.control.controller_orchestrator import ControllerOrchestrator


def _metric(err, steady, stable=1.0, jerk=1.0, mono=0.0):
	return SimpleNamespace(reward=-err, acc=stable, stable_rate=stable,
	                       mean_attitude_error_deg=err, mean_steady_error_deg=steady,
	                       motor_jerk_mean=jerk, mono_violations_total=mono,
	                       mean_effort=0.5)


def _genome(name):
	return SimpleNamespace(name=name, cells=None)


def _entries(pops):
	"""[(label, spec, res)] with res.final_population = pops[label]."""
	return [(label, SimpleNamespace(tag=label),
	         SimpleNamespace(best_genome=_genome(f"{label}-BEST"), final_population=pop))
	        for label, pop in pops.items()]


def _args(**kw):
	base = dict(fit_weight_err_sq=0.4, fit_weight_stable=0.3, fit_weight_jerk=0.2,
	            fit_weight_mono=0.1, fit_weight_steady=0.0, fit_weight_effort=0.0,
	            stage_select_top_k=3)
	base.update(kw)
	return SimpleNamespace(**base)


@pytest.fixture
def seen(monkeypatch):
	"""Capture every genome the selector scores; return per-genome metrics by name."""
	scored = []
	table = {}

	def fake_holdout(args, ec, spec, genome, final_population, vs, train, stage_label=""):
		scored.append(genome.name)
		return table.get(genome.name, _metric(2.0, 2.0))

	monkeypatch.setattr(P, "_holdout_report", fake_holdout)
	monkeypatch.setattr(P, "_maybe_holdout", lambda *a, **k: _metric(1.0, 1.0))
	return SimpleNamespace(scored=scored, table=table)


def _run(entries, args, holdouts=None):
	seeds = SimpleNamespace(val=1000, train=7)
	return P._select_headline_stage(args, None, seeds, entries, holdouts or {})


def test_top_k_of_every_stage_enters_the_ranking(seen):
	"""9 candidates from 3 stages x top-3 — not 3."""
	pops = {lbl: [_genome(f"{lbl}#{i}") for i in range(5)]
	        for lbl in ("GRID", "NEURONS", "MEMORY")}
	_run(_entries(pops), _args())
	unique = set(seen.scored)
	assert len(unique) == 9, f"expected top-3 x 3 stages, scored {sorted(unique)}"
	for lbl in ("GRID", "NEURONS", "MEMORY"):
		assert {f"{lbl}#{i}" for i in range(3)} <= unique
		assert f"{lbl}#3" not in unique, "K must bound the candidates"


def test_pop0_always_competes_not_best_genome(seen):
	"""Defect C: the ranked genome must be population[0], never the divergent
	best_genome, because pop[0] is what gets published and exported."""
	pops = {lbl: [_genome(f"{lbl}#{i}") for i in range(3)]
	        for lbl in ("NEURONS", "MEMORY")}
	_run(_entries(pops), _args())
	assert "NEURONS#0" in seen.scored and "MEMORY#0" in seen.scored
	assert not any(n.endswith("-BEST") for n in seen.scored), \
		"best_genome must not be scored while a population is present"


def test_falls_back_to_best_genome_without_a_population(seen):
	"""Older checkpoints and resumed runs carry no population — still scoreable."""
	entries = [("MEMORY", SimpleNamespace(tag="MEMORY"),
	            SimpleNamespace(best_genome=_genome("MEMORY-BEST"), final_population=None))]
	assert _run(entries, _args()) == "MEMORY"
	assert seen.scored == ["MEMORY-BEST"] * 5   # 5 val seeds


def test_winner_maps_back_to_its_stage(seen):
	"""A runner-up winning must headline its own STAGE, not a '#2' label."""
	pops = {"NEURONS": [_genome(f"NEURONS#{i}") for i in range(3)],
	        "MEMORY": [_genome(f"MEMORY#{i}") for i in range(3)]}
	# Make MEMORY#2 the clear best on every ranked component.
	seen.table["MEMORY#2"] = _metric(0.5, 0.3, stable=1.0, jerk=0.1, mono=0)
	assert _run(_entries(pops), _args()) == "MEMORY"


def test_steady_weight_can_flip_the_choice(seen):
	"""B: with steady weighted, a steady-dominant candidate wins; with steady at
	0 (the C10 default) it does not. This is the axis an S16-style sweep tests."""
	pops = {"NEURONS": [_genome("NEURONS#0")], "MEMORY": [_genome("MEMORY#0")]}
	# NEURONS: better jerk/mono. MEMORY: much better steady, slightly worse jerk.
	seen.table["NEURONS#0"] = _metric(2.0, 2.25, stable=1.0, jerk=0.1, mono=0)
	seen.table["MEMORY#0"] = _metric(2.0, 1.53, stable=1.0, jerk=0.9, mono=1)
	assert _run(_entries(pops), _args(fit_weight_steady=0.0)) == "NEURONS"
	assert _run(_entries(pops), _args(fit_weight_steady=0.35)) == "MEMORY"


def test_release_keeps_top_k_not_everything_and_not_nothing():
	"""The retention that makes the above possible: K survive, the rest are freed."""
	orch = ControllerOrchestrator.__new__(ControllerOrchestrator)
	orch._args = _args()
	res = SimpleNamespace(final_population=[_genome(f"g{i}") for i in range(50)],
	                      population_metrics=[{}] * 50)
	orch._res_by_stage = {1: res}
	orch._release_prior_populations(current_stage=4)
	assert len(res.final_population) == 3, "top-K must survive for stage selection"
	assert [g.name for g in res.final_population] == ["g0", "g1", "g2"]
	assert res.population_metrics is None, "the metrics list is still released"


def test_release_leaves_the_current_stage_untouched():
	orch = ControllerOrchestrator.__new__(ControllerOrchestrator)
	orch._args = _args()
	res = SimpleNamespace(final_population=[_genome(f"g{i}") for i in range(50)],
	                      population_metrics=None)
	orch._res_by_stage = {4: res}
	orch._release_prior_populations(current_stage=4)
	assert len(res.final_population) == 50, "the running stage still needs its pool"
