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
"""

from dataclasses import replace
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
	saved = (ckpt.extra or {}).get(WORKER_ABI_KEY)
	if saved is None:
		return f"checkpoint records no worker ABI (legacy), installed ABI {installed}"
	if int(saved) != int(installed):
		return f"checkpoint saved under worker ABI {saved}, installed ABI {installed}"
	return None
