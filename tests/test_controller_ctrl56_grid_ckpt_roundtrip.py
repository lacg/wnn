"""CTRL-56 regression (pytest, self-contained): a stage-0 GRID checkpoint must carry
trained cells for EVERY saved genome, and a reloaded genome must score exactly like
the in-memory one.

The bug (17/11/2026): the GRID search never writes cells back; the held-out trained
only pop[0] + its --holdout-pop-sample draw, so GRID#2 (and 41 others of 50) were
saved with cells=None and reloaded as EMPTY controllers in score-only tools.

Unlike tests/controller_ctrl56_ckpt_cells.py (a script over a REAL pre-fix run
checkpoint, skipped when the fixture is gone), this builds a tiny 4-genome grid
population from scratch, so it always runs:
  * the arch-only entries are materialized on COPIES (live genomes untouched);
  * save -> STRICT reload: every genome has cells, fingerprints equal the saved ones;
  * every reloaded genome scores identically to its in-memory twin;
  * an arch-only entry saved anyway is refused by the strict loader, naming it;
  * a resume strips exactly the materialized entries back to arch-only.

Run: PYTHONPATH=src pytest -q tests/test_controller_ctrl56_grid_ckpt_roundtrip.py
"""
import copy
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("WNN_CONTROLLER_GPU_EVAL", "0")

pytest.importorskip("ram_controller")

from wnn.control import phased_ga as P   # noqa: E402
from wnn.control.arch_strategy import controller_ga_neurons   # noqa: E402
from wnn.control.ckpt_cells import ArchOnlyEntryError, strip_materialized_cells   # noqa: E402
from wnn.control.evaluator import ControllerEvaluator, ControllerSpec   # noqa: E402

TRAIN_SEED = 31337001
TEST_SEED = 99990101
N_GENOMES = 4
TRAINED_IN_RUN = (0, 1)   # what the held-out trained in a real run (pop[0] + the draw)


def _args(td: str):
	a = P.build_arg_parser().parse_args([
		"--steps", "60", "--eval-episodes", "2", "--test-episodes", "2",
		"--calib-episodes", "2", "--holdout-pop-sample", "2", "--num-eval-folds", "5",
		"--train-workers", "1", "--levels", "4"])
	a.save_stage_checkpoints = td
	a._dhat_b = None
	a._conn_policy, a._conn_policy_min = P.conn_policy_from_name(getattr(a, "conn_policy", "spread"))
	return a


def _spec(a) -> ControllerSpec:
	return ControllerSpec(num_motors=4, levels_per_motor=a.levels, bits_per_feature=8,
	                      input_window_k=4, state_neurons=0,
	                      state_bits_per_neuron=12, output_bits_per_neuron=12)


def _score(a, ec, spec, genome):
	thr = P._test_thresholds(a, ec, spec, TEST_SEED, TRAIN_SEED, use_score=True)
	ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=TEST_SEED,
	                         episode_config=ec, thresholds=thr, rg_config=P._rg_config(a, ec, TEST_SEED),
	                         max_train_workers=1, num_eval_folds=a.num_eval_folds)
	m = ev.score_genomes([genome])[0]
	return (m.stable_rate, m.mean_attitude_error_deg, m.mean_steady_error_deg, m.reward)


@pytest.fixture(scope="module")
def grid_run(tmp_path_factory):
	"""A tiny 'grid population as the run left it': entries 0-1 trained, 2-3 arch-only."""
	a = _args(str(tmp_path_factory.mktemp("ctrl56")))
	ec = P.episode_config_from_args(a)
	P._resolve_ctbr(a, ec)
	spec = _spec(a)
	strat = controller_ga_neurons(spec, seed=0)
	live = [strat.create_random_genome() for _ in range(N_GENOMES)]
	thr = P._test_thresholds(a, ec, spec, TRAIN_SEED, TRAIN_SEED, use_score=True)
	P._train_arch_only(a, ec, spec, [live[i] for i in TRAINED_IN_RUN], TRAIN_SEED, thr)
	return SimpleNamespace(a=a, ec=ec, spec=spec, live=live)


def _materialize_and_save(r):
	before = [g.fingerprint() for g in r.live]
	mat, idx = P._materialize_arch_only_cells(r.a, r.ec, r.spec, r.live, TRAIN_SEED)
	P._save_stage_checkpoint(r.a, 0, "grid", r.spec,
	                         SimpleNamespace(best_genome=mat[0], final_population=mat), None,
	                         cells_materialized_idx=idx)
	return mat, idx, before


def test_grid_ckpt_every_genome_has_cells_and_scores_identically(grid_run):
	r = grid_run
	assert r.live[2].cells is None, "fixture: GRID#2 must start arch-only (the CTRL-56 shape)"
	mat, idx, before = _materialize_and_save(r)
	assert idx == [2, 3]
	assert [g.fingerprint() for g in r.live] == before and r.live[2].cells is None, \
		"live population must stay as flown (NEURONS seeds from it)"
	path = os.path.join(r.a.save_stage_checkpoints, "stage0_grid.yaml.gz")
	_l, spec2, res2, _it = P._stage_entry_from_checkpoint(path)   # STRICT: must not raise
	back = res2.final_population
	assert len(back) == N_GENOMES
	assert all(g.cells is not None for g in back), "every saved GRID genome must carry cells"
	assert [g.cells.fingerprint() for g in back] == [g.cells.fingerprint() for g in mat]
	assert [g.fingerprint() for g in back] == [g.fingerprint() for g in mat]
	for i in range(N_GENOMES):
		assert _score(r.a, r.ec, spec2, back[i]) == _score(r.a, r.ec, r.spec, mat[i]), f"GRID#{i}"


def test_materialized_cells_equal_a_solo_retrain(grid_run):
	"""The checkpoint holds what stage-select / --recalc-headline compute for a lone candidate."""
	r = grid_run
	mat, idx = P._materialize_arch_only_cells(r.a, r.ec, r.spec, r.live, TRAIN_SEED)
	thr = P._test_thresholds(r.a, r.ec, r.spec, TRAIN_SEED, TRAIN_SEED, use_score=True)
	for i in idx:
		solo = copy.copy(r.live[i])
		P._train_arch_only(r.a, r.ec, r.spec, [solo], TRAIN_SEED, thr)
		assert solo.cells.fingerprint() == mat[i].cells.fingerprint(), f"GRID#{i}"


def test_resume_strips_materialized_entries(grid_run):
	r = grid_run
	_mat, idx, before = _materialize_and_save(r)
	pay = P._ctl_load(os.path.join(r.a.save_stage_checkpoints, "stage0_grid.yaml.gz"))
	assert strip_materialized_cells(pay) == idx
	assert [g.cells is None for g in pay["population"]] == [False, False, True, True]
	assert [g.fingerprint() for g in pay["population"]] == before


def test_arch_only_entry_is_refused_by_strict_loader(grid_run, tmp_path, capsys):
	"""If an arch-only entry IS saved, the writer warns and the strict loader refuses it."""
	r = grid_run
	a = copy.copy(r.a)
	a.save_stage_checkpoints = str(tmp_path)
	P._save_stage_checkpoint(a, 0, "grid", r.spec,
	                         SimpleNamespace(best_genome=r.live[0], final_population=r.live), None)
	assert "[ckpt-cells] WARNING" in capsys.readouterr().out
	with pytest.raises(ArchOnlyEntryError, match="GRID#2"):
		P._stage_entry_from_checkpoint(str(tmp_path / "stage0_grid.yaml.gz"))
	_l, _s, res, _it = P._stage_entry_from_checkpoint(str(tmp_path / "stage0_grid.yaml.gz"),
	                                                  allow_arch_only=True)
	assert [g.cells is None for g in res.final_population] == [False, False, True, True]
