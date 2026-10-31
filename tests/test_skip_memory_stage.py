"""--skip-stages memory SKIPS the MEMORY stage (31/10/2026).

Before this, the orchestrator only honoured skips for arch/neurons stages: a
`memory` entry was dropped silently and the stage ran anyway (CTRL-8 smoke,
30/10). Luiz: "if you pass it, it should skip". With MEMORY skipped, the run's
final result (best genome + population, --save-winner) is the last stage that
ran. Unknown stage names are refused instead of being ignored.

Run: PYTHONPATH=src python -m pytest tests/test_skip_memory_stage.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("ram_controller")

from wnn.control import phased_ga as P

TINY = ["--grid-state-neurons", "0", "--max-state-neurons", "0",
        "--grid-bits", "8", "--grid-output-neurons", "8", "--max-output-neurons", "8",
        "--neurons-gens", "1", "--neurons-patience", "1",
        "--memory-gens", "1", "--memory-patience", "1", "--pop", "4", "--grid-top-k", "4",
        "--eval-episodes", "2", "--memory-eval-episodes", "2", "--report-episodes", "2",
        "--steps", "50", "--universe-episodes", "2", "--report-seeds", "11",
        "--base-seed", "5", "--runs", "1", "--check-interval", "1",
        "--train-workers", "2", "--grid-min-suffix", "1"]


def _main(monkeypatch, tmp_path, extra):
	monkeypatch.setenv("WNN_DB_PATH", str(tmp_path / "seeds.db"))
	monkeypatch.setattr(sys, "argv", ["phased_ga", *TINY, *extra])
	P.main()


def test_memory_skip_runs_no_memory_stage_and_saves_neurons_winner(monkeypatch, tmp_path):
	# No capsys: the optimizer's logger binds the stream it first sees, and a
	# captured stream closed after this test breaks the next test's logging.
	def _no_memory(*a, **k):
		raise AssertionError("MEMORY ran although --skip-stages memory was passed")
	monkeypatch.setattr(P, "_run_memory_phase", _no_memory)
	winner = tmp_path / "w.yaml.gz"
	_main(monkeypatch, tmp_path, ["--skip-stages", "bits,connections,memory",
	                              "--save-winner", str(winner)])
	assert winner.exists(), "--save-winner must publish the last stage that ran"


def test_unknown_skip_stage_is_refused(monkeypatch, tmp_path):
	with pytest.raises(SystemExit, match="unknown stage"):
		_main(monkeypatch, tmp_path, ["--skip-stages", "bits,memroy"])
