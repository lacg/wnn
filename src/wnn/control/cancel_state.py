"""Process-level "proper cancel" signal for controller / curriculum runs.

Background (01/06/2026)
-----------------------
The Rust ``ram_accelerator`` cancel flag (``is_cancelled`` / ``set_cancel_flag``
/ ``reset_cancel_flag``) is a process-wide ``AtomicBool``. Controller training
(``reward_gated`` / ``bptt``) and scoring poll it per step, so when it is set
the remaining work becomes a no-op and controllers come back **untrained** —
which the closed-loop scorer then reports as a degenerate genome (reward≈0 /
err≈0 over a too-short window, or the explicit 180°/-inf sentinel). This is the
controller analog of the IDS "F1=0.49" untrained-scoring bug.

The trouble is the Rust flag alone cannot tell us *why* it was set:

* A **proper cancel** — a real ``SIGTERM`` / ``SIGINT`` delivered to *this*
  process — means "stop, dump state, exit so we can resume."
* A **spurious cancel** — the flag observed set without any signal to this
  process (a stale flag, an internal error path, a cross-talk bug) — means
  "the results are untrained garbage; reset and retry, don't treat it as a
  shutdown."

IDS distinguishes these because a genuine shutdown also sets the worker's
Python ``_stop_current_flow`` (which unwinds the GA loop), so resetting the
Rust flag to retrain never defeats a real stop. The controller curriculum had
no such second signal — this module is it.

The curriculum's signal handler calls :func:`mark_sigterm`; the evaluator's
cancel-guard treats ``is_cancelled() and not sigterm_received()`` as spurious
(reset + retry) and ``is_cancelled() and sigterm_received()`` as proper (return
sentinels so the GA stops, then dump + resume).
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_sigterm_received: bool = False
_signum: int | None = None


def mark_sigterm(signum: int | None = None) -> None:
	"""Record that a real OS termination signal hit this process.

	Called from the SIGTERM/SIGINT handler. Idempotent."""
	global _sigterm_received, _signum
	with _lock:
		_sigterm_received = True
		if signum is not None:
			_signum = signum


def sigterm_received() -> bool:
	"""True iff a real SIGTERM/SIGINT was delivered to this process.

	This is the "proper cancel" predicate — distinct from the Rust
	``is_cancelled()`` flag, which can also be set spuriously."""
	with _lock:
		return _sigterm_received


def last_signum() -> int | None:
	"""The signal number of the most recent proper cancel (or None)."""
	with _lock:
		return _signum


def reset_sigterm() -> None:
	"""Clear the proper-cancel flag (call once at startup, or on resume)."""
	global _sigterm_received, _signum
	with _lock:
		_sigterm_received = False
		_signum = None
