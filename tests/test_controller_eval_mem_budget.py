"""Memory-aware eval budget (01/11/2026): WNN_CTRL_EVAL_BUDGET = unset (legacy 6 GB) |
"<GB>" | "auto" (a fraction of the memory available now, clamped), re-read at every packing.

Run: PYTHONPATH=src python -m pytest tests/test_controller_eval_mem_budget.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.control.evaluator import ControllerEvaluator as CE

GB = 1024 ** 3


def _budget(monkeypatch, value):
	if value is None:
		monkeypatch.delenv("WNN_CTRL_EVAL_BUDGET", raising=False)
	else:
		monkeypatch.setenv("WNN_CTRL_EVAL_BUDGET", value)
	return CE._eval_budget_bytes(CE.__new__(CE))


def test_unset_is_the_legacy_fixed_budget(monkeypatch):
	assert _budget(monkeypatch, None) == CE.EVAL_BUDGET_BYTES


def test_fixed_gb(monkeypatch):
	assert _budget(monkeypatch, "12") == 12 * GB


def test_invalid_is_refused(monkeypatch):
	with pytest.raises(ValueError):
		_budget(monkeypatch, "-1")


@pytest.mark.parametrize("avail,expect", [(10 * GB, 5 * GB), (1 * GB, CE.AUTO_BUDGET_MIN_BYTES),
                                          (200 * GB, CE.AUTO_BUDGET_MAX_BYTES)])
def test_auto_tracks_available_memory_and_clamps(monkeypatch, avail, expect):
	import psutil
	monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=avail))
	assert _budget(monkeypatch, "auto") == expect
