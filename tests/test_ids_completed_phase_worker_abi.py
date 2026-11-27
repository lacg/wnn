"""IDS-81 step 2: a COMPLETED phase result (exp_NN/*.json.gz, e.g. the grid
search) records the worker ABI, and a resume never reuses one written under
another ABI (or none): Flow re-runs that phase and every later one (a later
GA phase still resumes its own same-ABI in-progress checkpoint).

Run: PYTHONPATH=src/wnn python -m pytest tests/test_ids_completed_phase_worker_abi.py -q
"""
import gzip
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import wnn.ram.experiments.flow as flow_mod
import wnn.ram.strategies.phased.worker_abi_guard as guard
from wnn.ram.experiments.experiment import Experiment, ExperimentResult
from wnn.ram.experiments.flow import Flow
from wnn.ram.strategies.phased.worker_abi_guard import (
	WORKER_ABI_KEY, completed_phase_refusal, stamp_metadata,
)

ABI_N = 30
NAMES = ["Grid Search (neurons x bits)", "GA Neurons", "GA Bits"]


def _write_result(exp_dir: Path, abi=None, name="grid.json.gz"):
	exp_dir.mkdir(parents=True, exist_ok=True)
	meta = {"elapsed_seconds": 1.0}
	if abi is not None:
		meta[WORKER_ABI_KEY] = abi
	with gzip.open(exp_dir / name, "wt", encoding="utf-8") as f:
		json.dump({"phase_result": {"phase_name": "x"}, "_metadata": meta}, f)


def _flow(ckpt_dir: Path, lines: list) -> Flow:
	"""A Flow with only what _determine_start_idx touches (no tracker → the
	auto-resume half returns 0; explicit resume_from drives the index)."""
	f = Flow.__new__(Flow)
	f.checkpoint_dir = ckpt_dir
	f.config = SimpleNamespace(experiments=[SimpleNamespace(name=n) for n in NAMES])
	f.tracker, f._flow_id = None, None
	f.log = lines.append
	return f


def _start(ckpt_dir, resume_from, installed, monkeypatch, lines=None):
	monkeypatch.setattr(flow_mod, "installed_worker_abi", lambda: installed)
	return Flow._determine_start_idx(_flow(ckpt_dir, lines if lines is not None else []), resume_from)


def test_stamp_metadata_copies_and_records():
	meta = {"elapsed_seconds": 2.0}
	out = stamp_metadata(meta, ABI_N)
	assert out[WORKER_ABI_KEY] == ABI_N and WORKER_ABI_KEY not in meta
	assert WORKER_ABI_KEY not in stamp_metadata(meta, None)


def test_refusal_on_missing_untagged_and_other_abi():
	d = Path(tempfile.mkdtemp())
	assert "no completed-phase result" in completed_phase_refusal(d / "exp_00", ABI_N)
	_write_result(d / "exp_00")
	assert "no worker ABI" in completed_phase_refusal(d / "exp_00", ABI_N)
	_write_result(d / "exp_01", ABI_N - 1)
	assert f"worker ABI {ABI_N - 1}" in completed_phase_refusal(d / "exp_01", ABI_N)
	_write_result(d / "exp_02", ABI_N)
	assert completed_phase_refusal(d / "exp_02", ABI_N) is None


def test_non_result_gz_files_are_ignored():
	d = Path(tempfile.mkdtemp()) / "exp_00"
	_write_result(d, ABI_N)
	with gzip.open(d / "other.json.gz", "wt", encoding="utf-8") as f:
		json.dump({"population": []}, f)
	assert completed_phase_refusal(d, ABI_N) is None


def test_untagged_grid_is_rerun_on_resume(monkeypatch):
	"""The pre-IDS-81 case: a completed grid from an older (untagged) worker."""
	d, lines = Path(tempfile.mkdtemp()), []
	_write_result(d / "exp_00")
	assert _start(d, 1, ABI_N, monkeypatch, lines) == 0
	assert lines and lines[0].startswith("[CKPT] ⚠️ experiment 0 (Grid Search")
	# The message names the re-run range and the one thing NOT redone: a later
	# GA phase still resumes an in-progress checkpoint tagged with this ABI.
	assert f"re-running experiments 0..{len(NAMES) - 1}" in lines[0]
	assert f"in-progress checkpoint tagged ABI {ABI_N}" in lines[0]
	assert "from scratch" not in lines[0]


def test_other_abi_phase_and_everything_after_rerun(monkeypatch):
	d = Path(tempfile.mkdtemp())
	_write_result(d / "exp_00", ABI_N)
	_write_result(d / "exp_01", ABI_N - 1)
	assert _start(d, 2, ABI_N, monkeypatch) == 1
	assert _start(d, 2, ABI_N - 1, monkeypatch) == 0


def test_same_abi_is_reused(monkeypatch):
	d = Path(tempfile.mkdtemp())
	_write_result(d / "exp_00", ABI_N)
	_write_result(d / "exp_01", ABI_N)
	assert _start(d, 2, ABI_N, monkeypatch) == 2


def test_no_accelerator_means_no_check(monkeypatch):
	d = Path(tempfile.mkdtemp())
	_write_result(d / "exp_00")
	assert _start(d, 1, None, monkeypatch) == 1


def test_experiment_save_stamps_the_installed_abi(monkeypatch):
	from wnn.ram.genome import ClusterGenome
	monkeypatch.setattr(guard, "installed_worker_abi", lambda: ABI_N)
	g = ClusterGenome(bits_per_neuron=[8] * 4, neurons_per_cluster=[4], connections=list(range(32)))
	result = ExperimentResult(
		experiment_name="Grid Search", strategy_type="grid", initial_fitness=None, final_fitness=0.5,
		final_accuracy=0.9, improvement_percent=0.0, iterations_run=1, best_genome=g,
		final_population=[g], final_threshold=0.5, elapsed_seconds=1.0)
	d = Path(tempfile.mkdtemp()) / "exp_00"
	exp = SimpleNamespace(checkpoint_dir=d, log=lambda m: None, dashboard_client=None, experiment_id=None)
	Experiment._save_checkpoint(exp, result)
	assert completed_phase_refusal(d, ABI_N) is None
	assert "worker ABI 30" in completed_phase_refusal(d, ABI_N + 1)
