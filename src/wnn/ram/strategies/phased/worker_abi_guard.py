"""Worker-ABI lineage guard for IDS GA checkpoints (IDS-81 fix 1, 27/11/2026).

A per-generation GA checkpoint used to carry no record of the worker wheel
(``ram_accelerator`` ABI) that produced it. A flow paused under ABI N and
resumed after a worker swap to ABI N+1 then continued the SAME GA population
on a different trainer — one result, two trainer lineages, nothing in the
checkpoint or the DB to tell.

The guard is two pure functions plus one lookup:
  * ``stamp_worker_abi`` — copy of a checkpoint with ``extra["worker_abi"]``;
  * ``worker_abi_refusal`` — why a loaded checkpoint must NOT be resumed under
    the installed ABI (mismatch, or no ABI at all = legacy), else None;
  * ``installed_worker_abi`` — the installed wheel's ABI via ``wnn.accel``.

``PhasedCheckpointManager(worker_abi=...)`` wires them into save/load. Only
the IDS strand passes ``worker_abi``; the controller strand (``ram_controller``,
its own ABI) leaves it None and is unaffected.

The same lineage rule covers a COMPLETED phase (IDS-81 step 2): a finished
experiment's result file (``exp_NN/*.json.gz``, written by
``Experiment._save_checkpoint``) records the ABI in ``_metadata`` via
``stamp_metadata``; on resume ``completed_phase_refusal`` rejects one from
another ABI (or none) and ``Flow`` re-runs that phase and everything after
it instead of reusing any completed result. A later GA phase still resumes its
own in-progress checkpoint when that checkpoint passes ``worker_abi_refusal``
(tagged with the installed ABI); otherwise it starts fresh.
"""

import gzip
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

from wnn.ram.strategies.phased.checkpoint import PhaseCheckpoint

WORKER_ABI_KEY = "worker_abi"


def installed_worker_abi() -> Optional[int]:
	"""ABI_VERSION of the installed worker wheel, or None when the accelerator
	is unavailable (WNN_ALLOW_PY_FALLBACK runs — never reported anyway)."""
	from wnn.accel import accel_or_none
	try:
		ra = accel_or_none()
	except ImportError:
		return None
	return None if ra is None else int(getattr(ra, "ABI_VERSION", 0))


def stamp_worker_abi(ckpt: PhaseCheckpoint, abi: int) -> PhaseCheckpoint:
	"""A copy of ``ckpt`` whose extra records the worker ABI that wrote it
	(copy, not mutation: async saves snapshot the caller's object)."""
	return replace(ckpt, extra={**(ckpt.extra or {}), WORKER_ABI_KEY: int(abi)})


def worker_abi_refusal(ckpt: PhaseCheckpoint, installed: int) -> Optional[str]:
	"""Reason this checkpoint must not be resumed under ``installed``, or None."""
	return _abi_refusal((ckpt.extra or {}).get(WORKER_ABI_KEY), installed)


def stamp_metadata(metadata: dict, abi: Optional[int]) -> dict:
	"""A completed-phase ``_metadata`` dict recording the worker ABI (a copy;
	unchanged when the ABI is unknown — such a file is refused on resume)."""
	return dict(metadata) if abi is None else {**metadata, WORKER_ABI_KEY: int(abi)}


def completed_phase_refusal(exp_dir: Path, installed: int) -> Optional[str]:
	"""Reason the completed phase result in ``exp_dir`` must NOT be reused
	under ``installed`` (none / unreadable, no ABI, other ABI), or None. Every
	phase-result file there must pass (the loader takes the first it globs);
	other ``*.json.gz`` files (no ``phase_result`` key) are not results."""
	files = sorted(Path(exp_dir).glob("*.json.gz")) if Path(exp_dir).is_dir() else []
	verdicts = [v for v in (_result_file_refusal(p, installed) for p in files) if v is not None]
	refused = [reason for _, reason in verdicts if reason is not None]
	if refused:
		return refused[0]
	return None if verdicts else f"no completed-phase result file to verify, installed ABI {installed}"


def _result_file_refusal(path: Path, installed: int) -> Optional[tuple]:
	"""None when ``path`` is not a phase-result file, else (path, reason-or-None)."""
	try:
		with gzip.open(path, "rt", encoding="utf-8") as f:
			data = json.load(f)
	except (OSError, ValueError) as e:
		return path, f"{path.name}: completed-phase result unreadable ({e}), installed ABI {installed}"
	if "phase_result" not in data:
		return None
	reason = _abi_refusal((data.get("_metadata") or {}).get(WORKER_ABI_KEY), installed)
	return path, None if reason is None else f"{path.name}: {reason}"


def _abi_refusal(saved: Optional[int], installed: int) -> Optional[str]:
	if saved is None:
		return f"checkpoint records no worker ABI (legacy), installed ABI {installed}"
	if int(saved) != int(installed):
		return f"checkpoint saved under worker ABI {saved}, installed ABI {installed}"
	return None
