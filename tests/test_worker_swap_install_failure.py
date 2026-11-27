"""IDS-81 fix 10: scripts/worker_swap.py on a FAILED wheel install must revert
the staged patch, still requeue + relaunch on the old wheel, and write the
marker with install_failed. The pip step is mocked; git runs for real on a
throwaway repo. No worker is touched (pid lookup / relaunch / requeue mocked).

Run: python -m pytest tests/test_worker_swap_install_failure.py -q
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "worker_swap.py"
ORIGINAL = "x = 1\n"


def _load_swap():
	spec = importlib.util.spec_from_file_location("worker_swap_under_test", SCRIPT)
	mod = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = mod  # @dataclass resolves annotations via sys.modules
	spec.loader.exec_module(mod)
	return mod


def _git(repo: Path, *args: str) -> str:
	return subprocess.run(["git", "-C", str(repo), *args], check=True,
	                      capture_output=True, text=True).stdout


def _repo_with_patch(tmp: Path) -> tuple[Path, Path]:
	repo = tmp / "repo"
	repo.mkdir()
	_git(repo, "init", "-q")
	(repo / "mod.py").write_text(ORIGINAL)
	_git(repo, "add", "mod.py")
	_git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
	(repo / "mod.py").write_text("x = 2\n")
	patch = tmp / "staged.patch"
	patch.write_text(_git(repo, "diff"))
	_git(repo, "checkout", "--", "mod.py")
	return repo, patch


def _mock_side_effects(swap, calls: dict, install_ok: bool) -> None:
	swap.running_flow_ids = lambda db: calls.setdefault("running", [])
	swap.find_worker_pid = lambda: None
	swap.install_wheel = lambda venv, wheel: install_ok
	swap.requeue_interrupted = lambda args, interrupted: calls.setdefault("requeue", list(interrupted))
	swap.relaunch_worker = lambda *a: calls.setdefault("relaunch", 4242)


def _run(tmp: Path, monkeypatch, install_ok: bool):
	swap = _load_swap()
	repo, patch = _repo_with_patch(tmp)
	wheel = tmp / "ram_accelerator-0-cp312-none-any.whl"
	wheel.write_bytes(b"")
	marker = tmp / "marker.json"
	calls: dict = {}
	_mock_side_effects(swap, calls, install_ok)
	monkeypatch.setattr(sys, "argv", [
		"worker_swap.py", "--stop-now", "--install-wheel", str(wheel),
		"--apply-patch", str(patch), "--project", str(repo), "--db", str(tmp / "x.db"),
		"--venv", str(tmp / "venv"), "--marker", str(marker)])
	rc = swap.main()
	return rc, repo, json.loads(marker.read_text()), calls


def test_install_failure_reverts_patch_and_still_relaunches(tmp_path, monkeypatch):
	rc, repo, m, calls = _run(tmp_path, monkeypatch, install_ok=False)
	assert (repo / "mod.py").read_text() == ORIGINAL, "patch must be reverted"
	assert _git(repo, "status", "--porcelain") == ""
	assert "requeue" in calls and calls["relaunch"] == 4242
	assert m["install_failed"] is True and m["patch_reverted"] is True
	assert m["installed_wheel"] is None and m["applied_patch"] is None
	assert m["new_worker_pid"] == 4242 and m["restarted"] is True
	assert rc == 3


def test_install_success_keeps_patch(tmp_path, monkeypatch):
	rc, repo, m, calls = _run(tmp_path, monkeypatch, install_ok=True)
	assert (repo / "mod.py").read_text() == "x = 2\n"
	assert m["install_failed"] is False and m["patch_reverted"] is None
	assert m["installed_wheel"] and m["applied_patch"] and m["patch_failed"] is False
	assert calls["relaunch"] == 4242 and rc == 0
