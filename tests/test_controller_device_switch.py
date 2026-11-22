"""CTRL-54 follow-up (Luiz 22/11/2026): a controller run never switches scoring device mid-run.

Metal failing must raise (not silently re-score on the Rust CPU scorer or the Python loop)
unless WNN_CONTROLLER_ALLOW_DEVICE_SWITCH=1.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "wnn"))
sys.path.insert(0, str(ROOT / "src"))

from wnn.control import _accel  # noqa: E402
from wnn.control.evaluator import ControllerEvaluator  # noqa: E402


def _fake(metal_result, cpu_result, retried: list):
	"""Minimal stand-in carrying exactly what score_population reads on the Rust path."""
	fake = SimpleNamespace(max_eval_workers_gpu=True, eval_mode="gpu", _rust_score_error=RuntimeError("cmd buffer"))
	fake._score_population_rust = lambda ctrls, scorer: metal_result if scorer is _accel.score_controllers_metal else cpu_result
	fake._retry_on_rust_cpu = lambda ctrls, scorer: retried.append(scorer) or cpu_result
	fake._refuse_device_switch = lambda target: ControllerEvaluator._refuse_device_switch(fake, target)
	fake._refuse_python_fallback = lambda: (_ for _ in ()).throw(RuntimeError("python loop refused"))
	return fake


def test_metal_failure_raises_by_default(monkeypatch):
	monkeypatch.delenv("WNN_CONTROLLER_ALLOW_DEVICE_SWITCH", raising=False)
	retried = []
	with pytest.raises(RuntimeError, match="refusing to re-score on the Rust CPU scorer"):
		ControllerEvaluator.score_population(_fake(None, ["cpu"], retried), ["c"])
	assert retried == []


def test_metal_success_is_untouched(monkeypatch):
	monkeypatch.delenv("WNN_CONTROLLER_ALLOW_DEVICE_SWITCH", raising=False)
	assert ControllerEvaluator.score_population(_fake(["gpu"], ["cpu"], []), ["c"]) == ["gpu"]


def test_opt_in_keeps_the_old_cpu_retry(monkeypatch):
	monkeypatch.setenv("WNN_CONTROLLER_ALLOW_DEVICE_SWITCH", "1")
	retried = []
	assert ControllerEvaluator.score_population(_fake(None, ["cpu"], retried), ["c"]) == ["cpu"]
	assert retried == [_accel.score_controllers_cpu]
