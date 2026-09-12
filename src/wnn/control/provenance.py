"""Run provenance for controller markers (multi-axis programme spec, rule R9).

Every banked marker has to say WHICH CODE produced it: the controller wheel
(name-version), its ABI, a hash of the compiled extension actually loaded, the
git HEAD of the Python tree, and the fitness-pool scheme the search ran under.
Without these, "the four anchor markers predate the 08/09 label-fix wheel" is a
sentence in a memory note instead of a field a script can check.

The chain of custody is .out -> grep -> marker JSON (scripts/controller_arm_lib.sh),
so this module prints ONE greppable line, `[provenance] key=value ...`, next to the
existing `fitness_pools=` header. Every accessor is fail-safe: a run must never
die at startup because git is absent or the wheel lacks a metadata record — a
missing value prints as `unknown`, which is itself a finding, not a crash.
"""

import hashlib
import importlib.metadata
import os
import subprocess
from dataclasses import dataclass


PROVENANCE_PREFIX = "[provenance]"
UNKNOWN = "unknown"
_SHA_PREFIX_HEX = 16


@dataclass(frozen=True)
class ControllerProvenance:
	"""What produced a run. Frozen so a marker cannot be re-attributed after the fact."""

	wheel: str          # "ram_controller-2026.212.37" (dist name-version) or "unknown"
	abi: int            # ABI_VERSION of the loaded extension; 0 when unavailable
	wheel_sha256: str   # first 16 hex of sha256 over the compiled .so actually imported
	git: str            # short HEAD of the Python tree, "+dirty" when the tree has edits
	fitness_pools: str  # the same label the "Pop=..." header prints

	def line(self) -> str:
		"""The single greppable line the ladder copies into the marker."""
		return (f"{PROVENANCE_PREFIX} wheel={self.wheel} abi={self.abi} "
		        f"wheel_sha256={self.wheel_sha256} git={self.git} "
		        f"fitness_pools={self.fitness_pools}")


def collect_provenance(fitness_pools: str) -> ControllerProvenance:
	"""Gather every field, never raising: each accessor degrades to UNKNOWN / 0."""
	wheel, abi, sha = _wheel_identity()
	return ControllerProvenance(
		wheel=wheel,
		abi=abi,
		wheel_sha256=sha,
		git=_git_head(os.path.dirname(os.path.abspath(__file__))),
		fitness_pools=fitness_pools,
	)


def _wheel_identity() -> tuple[str, int, str]:
	"""(dist name-version, ABI, .so hash) of the controller extension in use.

	Goes through the facade so the transition fallback (combined ram_accelerator
	wheel) is reported as what it is, not as a ram_controller build."""
	try:
		from wnn.control import _accel
		mod = _accel.require_accel()
	except Exception:
		return UNKNOWN, 0, UNKNOWN
	name = getattr(mod, "__name__", UNKNOWN).split(".")[0]
	abi = int(getattr(mod, "ABI_VERSION", 0) or 0)
	return _dist_label(name), abi, _module_sha256(mod)


def _dist_label(dist_name: str) -> str:
	try:
		return f"{dist_name}-{importlib.metadata.version(dist_name)}"
	except Exception:
		return f"{dist_name}-{UNKNOWN}"


def _module_sha256(mod) -> str:
	"""Hash of the compiled extension file. `mod.__file__` may be the package
	__init__.py (maturin layout) — hash the .so beside it in that case."""
	try:
		path = getattr(mod, "__file__", None)
		if path is None:
			return UNKNOWN
		if path.endswith("__init__.py"):
			path = _first_extension_in(os.path.dirname(path))
		if path is None:
			return UNKNOWN
		with open(path, "rb") as fh:
			return hashlib.sha256(fh.read()).hexdigest()[:_SHA_PREFIX_HEX]
	except Exception:
		return UNKNOWN


def _first_extension_in(directory: str) -> str | None:
	for entry in sorted(os.listdir(directory)):
		if entry.endswith(".so") or entry.endswith(".pyd"):
			return os.path.join(directory, entry)
	return None


def _git_head(source_dir: str) -> str:
	"""Short HEAD of the tree the Python was imported from, '+dirty' if edited.
	A 2 s timeout so a hung git (network FS, lock) cannot stall a run."""
	try:
		head = _git(source_dir, "rev-parse", "--short", "HEAD")
		if not head:
			return UNKNOWN
		dirty = _git(source_dir, "status", "--porcelain", "--untracked-files=no")
		return head + ("+dirty" if dirty else "")
	except Exception:
		return UNKNOWN


def _git(cwd: str, *args: str) -> str:
	result = subprocess.run(
		["git", *args], cwd=cwd, capture_output=True, text=True, timeout=2.0, check=False,
	)
	return result.stdout.strip() if result.returncode == 0 else ""
