"""R9 provenance: the line phased_ga prints must be greppable, single-token per
field, fail-safe, and round-trip through the ladder's sed captures.

Run: PYTHONPATH=src/wnn python tests/controller_provenance.py
"""

import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "wnn"))

from wnn.control.provenance import (  # noqa: E402
	PROVENANCE_PREFIX, UNKNOWN, ControllerProvenance, collect_provenance, _git_head, _module_sha256,
)

FAILS = 0


def check(label: str, got, want) -> None:
	global FAILS
	ok = got == want
	print(f"  {'ok  ' if ok else 'FAIL'} {label:<60} -> {got!r}" + ("" if ok else f" (expected {want!r})"))
	if not ok:
		FAILS += 1


def test_line_is_single_token_per_field() -> None:
	p = ControllerProvenance("ram_controller-2026.212.37", 27, "08cdd0462eac0d15", "3b5fc37d+dirty", "CRN(all 5 pools/gen)")
	line = p.line()
	check("line starts with the prefix", line.startswith(PROVENANCE_PREFIX + " "), True)
	fields = dict(re.findall(r" (\w+)=(\S+)", line.split(" fitness_pools=")[0]))
	check("wheel/abi/sha/git are space-free tokens", fields,
	      {"wheel": "ram_controller-2026.212.37", "abi": "27", "wheel_sha256": "08cdd0462eac0d15", "git": "3b5fc37d+dirty"})
	check("fitness_pools is the LAST field (may contain spaces)", line.endswith(" fitness_pools=CRN(all 5 pools/gen)"), True)
	check("no double quotes anywhere (the marker is printf'd JSON)", '"' in line, False)


def test_collect_never_raises_and_reports_live_wheel() -> None:
	p = collect_provenance("rotation(1 pool/gen)")
	check("fitness_pools passed through", p.fitness_pools, "rotation(1 pool/gen)")
	check("abi is an int", isinstance(p.abi, int), True)
	check("wheel names the dist", p.wheel.startswith("ram_controller-") or p.wheel.startswith("ram_accelerator-") or p.wheel == UNKNOWN, True)
	check("sha is 16 hex or unknown", bool(re.fullmatch(r"[0-9a-f]{16}", p.wheel_sha256)) or p.wheel_sha256 == UNKNOWN, True)
	check("git is a short sha (+dirty) or unknown", bool(re.fullmatch(r"[0-9a-f]{7,12}(\+dirty)?", p.git)) or p.git == UNKNOWN, True)


def test_git_head_outside_a_repo_is_unknown() -> None:
	with tempfile.TemporaryDirectory() as td:
		check("no repo -> unknown, not an exception", _git_head(td), UNKNOWN)


def test_git_head_dirty_suffix() -> None:
	with tempfile.TemporaryDirectory() as td:
		env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
		subprocess.run(["git", "init", "-q", td], check=True)
		open(os.path.join(td, "a"), "w").write("1\n")
		subprocess.run(["git", "-C", td, "add", "a"], check=True)
		subprocess.run(["git", "-C", td, "commit", "-q", "-m", "x"], check=True, env=env)
		clean = _git_head(td)
		check("clean tree -> bare short sha", bool(re.fullmatch(r"[0-9a-f]{7,12}", clean)), True)
		open(os.path.join(td, "a"), "w").write("2\n")
		check("edited tracked file -> +dirty", _git_head(td), clean + "+dirty")
		open(os.path.join(td, "untracked"), "w").write("u\n")
		os.remove(os.path.join(td, "a")); open(os.path.join(td, "a"), "w").write("1\n")
		check("untracked file alone does NOT dirty (logs/markers live in-tree)", _git_head(td), clean)


def test_module_sha_handles_package_init() -> None:
	with tempfile.TemporaryDirectory() as td:
		so = os.path.join(td, "x.cpython-313-darwin.so")
		open(so, "wb").write(b"abc")

		class Mod:
			__file__ = os.path.join(td, "__init__.py")
		check("hashes the .so beside __init__.py", _module_sha256(Mod), "ba7816bf8f01cfea")

		class NoFile:
			pass
		check("module without __file__ -> unknown", _module_sha256(NoFile), UNKNOWN)


if __name__ == "__main__":
	for name, fn in list(globals().items()):
		if name.startswith("test_") and callable(fn):
			print(f"=== {name}")
			fn()
	print()
	if FAILS:
		print(f"FAILED ({FAILS})"); sys.exit(1)
	print("ALL PASS — provenance is greppable, single-token, and fail-safe")
