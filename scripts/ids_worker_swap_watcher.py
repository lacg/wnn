"""IDS worker resource-swap watcher (31/05/2026 ad-hoc).

Polls the dashboard DB for flow 4042 (XDS-unsw-random-64b-Wc) completion.
When that flow finishes:
  1. SIGTERM the running worker (PID auto-detected).
  2. Wait for the worker to exit cleanly.
  3. Relaunch the worker with RAYON_NUM_THREADS=10 (was unbound; the
     co-resident curriculum-GA now caps itself at RAYON=1 × 3 Python
     workers = 3 cores, leaving 13 for the IDS worker — plenty).
  4. Flip flow 4041 (next-pickup, parked at status='pending' before this
     watcher started) back to 'queued' so the new worker grabs it.

Idempotent on restart: re-running while flow 4042 is still in flight
just re-enters the poll loop; safe to background.
"""
from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

DB = Path("/Users/lacg/wnn/db/wnn.db")
TARGET_FLOW = 4042   # flow that must finish before swap
NEXT_FLOW   = 4041   # parked at 'pending' pre-watcher; flipped back to 'queued' after restart
RAYON_THREADS_NEW = "10"

POLL_SECS  = 30
GRACE_SECS = 20  # how long to wait for SIGTERM'd worker to exit before SIGKILL


def get_flow_status(flow_id: int) -> str | None:
	con = sqlite3.connect(str(DB))
	row = con.execute("SELECT status FROM flows WHERE id=?", (flow_id,)).fetchone()
	con.close()
	return row[0] if row else None


def find_worker_pid() -> int | None:
	"""Find the python -m wnn.ram.experiments.worker process. Returns the
	Python interpreter PID (not the zsh launcher), which is what we SIGTERM."""
	out = subprocess.run(
		["pgrep", "-af", "wnn.ram.experiments.worker"],
		capture_output=True, text=True
	).stdout
	for line in out.splitlines():
		parts = line.split(None, 1)
		if not parts:
			continue
		pid = int(parts[0])
		# Filter to the Python interpreter (the zsh launcher's cmdline also
		# contains the worker module name).
		cmdline = parts[1] if len(parts) > 1 else ""
		if "Python.framework" in cmdline or cmdline.lstrip().startswith("/opt"):
			return pid
	# Fallback: return the first match.
	for line in out.splitlines():
		parts = line.split(None, 1)
		if parts:
			return int(parts[0])
	return None


def stop_worker(pid: int) -> None:
	"""SIGTERM the worker, escalate to SIGKILL after GRACE_SECS if still alive."""
	print(f"[swap-watcher] SIGTERM worker PID={pid}", flush=True)
	try:
		os.kill(pid, signal.SIGTERM)
	except ProcessLookupError:
		return
	for _ in range(GRACE_SECS):
		time.sleep(1)
		try:
			os.kill(pid, 0)
		except ProcessLookupError:
			print(f"[swap-watcher] worker {pid} exited", flush=True)
			return
	print(f"[swap-watcher] worker still alive after {GRACE_SECS}s — SIGKILL", flush=True)
	try:
		os.kill(pid, signal.SIGKILL)
	except ProcessLookupError:
		pass


def relaunch_worker_with_rayon(threads: str) -> None:
	"""Launch a fresh worker process with RAYON_NUM_THREADS set. Uses the
	same activation pattern as the original launch (per CLAUDE.md venv
	setup)."""
	print(f"[swap-watcher] relaunching worker with RAYON_NUM_THREADS={threads}", flush=True)
	cmd = (
		"cd /Users/lacg/wnn && "
		"unset CONDA_PREFIX && "
		"source wnn/bin/activate && "
		"export PYTHONPATH=\"$(pwd)/src/wnn:$PYTHONPATH\" && "
		f"export RAYON_NUM_THREADS={threads} && "
		"nohup python -u -B -m wnn.ram.experiments.worker --no-ssl-verify "
		"</dev/null >>/tmp/wnn_worker.log 2>&1 &"
	)
	subprocess.Popen(["zsh", "-c", cmd], start_new_session=True)
	time.sleep(3)
	new_pid = find_worker_pid()
	print(f"[swap-watcher] new worker PID={new_pid}", flush=True)


def requeue_flow(flow_id: int) -> None:
	"""Flip the parked flow back to 'queued' so the new worker picks it up.
	id-DESC pickup means this is the next thing the new worker will grab."""
	con = sqlite3.connect(str(DB))
	con.execute("UPDATE flows SET status='queued' WHERE id=? AND status='pending'", (flow_id,))
	con.commit()
	row = con.execute("SELECT id, status FROM flows WHERE id=?", (flow_id,)).fetchone()
	con.close()
	print(f"[swap-watcher] flow {flow_id} now status={row[1]!r}", flush=True)


def main() -> int:
	print(f"[swap-watcher] watching flow {TARGET_FLOW}; will swap worker to "
	      f"RAYON_NUM_THREADS={RAYON_THREADS_NEW} on completion", flush=True)
	while True:
		status = get_flow_status(TARGET_FLOW)
		print(f"[swap-watcher] flow {TARGET_FLOW} status={status!r}", flush=True)
		if status == "completed":
			break
		if status in ("failed", "cancelled"):
			print(f"[swap-watcher] flow {TARGET_FLOW} ended as {status!r} — still proceeding with swap.",
			      flush=True)
			break
		if status is None:
			print(f"[swap-watcher] flow {TARGET_FLOW} no longer exists! Aborting.", flush=True)
			return 2
		time.sleep(POLL_SECS)

	pid = find_worker_pid()
	if pid is None:
		print(f"[swap-watcher] no worker process found; just flipping {NEXT_FLOW} to queued",
		      flush=True)
	else:
		stop_worker(pid)

	relaunch_worker_with_rayon(RAYON_THREADS_NEW)
	requeue_flow(NEXT_FLOW)
	print(f"[swap-watcher] swap complete.", flush=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
