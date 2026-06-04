"""Integration test for the curriculum proper-cancel → abort → resume flow.

Stubs _run_one_stage so no real GA/training runs (milliseconds, GPU-free).
Verifies:
  * a PROPER cancel mid-stage aborts the run, writes a resume checkpoint that
    re-runs the INCOMPLETE stage (next_index points at it, not past it), and
    records only the COMPLETED stages — never a fake degenerate RESULT;
  * relaunching with that checkpoint continues to completion (DONE).
"""

import importlib.util
import pickle
import types
from pathlib import Path

import pytest

from wnn.control import cancel_state
from wnn.ram.metrics import Metrics

_SPEC = importlib.util.spec_from_file_location(
	"run_curriculum_ga", str(Path(__file__).parent / "run_curriculum_ga.py"))
cur = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cur)


class _FakeRes:
	def __init__(self):
		self.best_genome = object()
		self.final_population = [object(), object()]
		self.final_fitness = -2.0     # ce = -reward → reward = +2.0
		self.final_accuracy = 0.8
		self.iterations_run = 30


class _FakeEv:
	num_eval_episodes = 100
	def evaluate_batch(self, genomes, **_kw):
		return [Metrics(ce=-2.0, acc=0.8, fitness=2.0, mean_attitude_error_deg=3.0)
		        for _ in genomes]


def _args(tmp, resume=None):
	return types.SimpleNamespace(
		pop=4, elitism=0.2, crossover_rate=0.5, train_workers=1, num_eval_folds=1,
		save_dir=str(tmp), resume=resume, base_seed=42, seed=42,
	)


@pytest.fixture(autouse=True)
def _clean():
	cancel_state.reset_sigterm()
	yield
	cancel_state.reset_sigterm()


def test_proper_cancel_aborts_and_writes_resume(tmp_path, monkeypatch):
	# Stage B fires a proper cancel mid-stage; A completes normally.
	def fake_stage(args, stage, weights, spec, seed, initial_population=None, stage_label="", resume_start_gen=0):
		if stage.name == "B":
			cancel_state.mark_sigterm(15)
		return _FakeRes(), _FakeEv(), 1.0, None
	monkeypatch.setattr(cur, "_run_one_stage", fake_stage)
	monkeypatch.setattr(cur, "_print_curriculum_report", lambda *a, **k: None)

	weights = {"err": 0.5, "stable": 0.4, "jerk": 0.05, "mono": 0.05}
	outcome = cur.run_full_curriculum(_args(tmp_path), weights, seed=42)

	assert outcome == "ABORTED"
	rp = cur._resume_path(tmp_path)
	assert rp.exists()
	rs = pickle.loads(rp.read_bytes())
	# Stage A (index 0) completed; B (index 1) was incomplete → re-run at 1.
	assert rs["next_index"] == 1
	assert len(rs["stage_records"]) == 1
	assert rs["stage_records"][0]["name"] == "A"


def test_resume_runs_to_completion(tmp_path, monkeypatch):
	# First: produce a resume checkpoint (cancel at B).
	def fake_stage_cancel(args, stage, weights, spec, seed, initial_population=None, stage_label="", resume_start_gen=0):
		if stage.name == "B":
			cancel_state.mark_sigterm(15)
		return _FakeRes(), _FakeEv(), 1.0, None
	monkeypatch.setattr(cur, "_run_one_stage", fake_stage_cancel)
	monkeypatch.setattr(cur, "_print_curriculum_report", lambda *a, **k: None)
	weights = {"err": 0.5, "stable": 0.4, "jerk": 0.05, "mono": 0.05}
	cur.run_full_curriculum(_args(tmp_path), weights, seed=42)
	rp = cur._resume_path(tmp_path)
	assert rp.exists()

	# Now resume with NO cancel → should finish all remaining stages (B..E).
	cancel_state.reset_sigterm()
	def fake_stage_clean(args, stage, weights, spec, seed, initial_population=None, stage_label="", resume_start_gen=0):
		return _FakeRes(), _FakeEv(), 1.0, None
	monkeypatch.setattr(cur, "_run_one_stage", fake_stage_clean)

	recorded = {}
	def capture_report(stage_records, cumulative_wall, seed):
		recorded["n"] = len(stage_records)
		recorded["names"] = [r["name"] for r in stage_records]
	monkeypatch.setattr(cur, "_print_curriculum_report", capture_report)

	outcome = cur.run_full_curriculum(_args(tmp_path, resume=str(rp)), weights={}, seed=42)
	assert outcome == "DONE"
	# A (from before) + B,C,D,E (this run) = all 5 stages.
	assert recorded["n"] == 5
	assert recorded["names"] == ["A", "B", "C", "D", "E"]
	# Success tidies the resume file away.
	assert not rp.exists()
