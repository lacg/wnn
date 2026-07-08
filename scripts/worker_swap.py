"""Reusable worker hot-swap watcher.

Waits for a flow to finish, stops the IDS/XDS worker cleanly, optionally installs
a freshly-built accelerator wheel, relaunches the worker (detached, PPID=1), and
re-queues any flow the stop interrupted — via the dashboard API, NOT raw SQL.

Replaces the old hardcoded ids_worker_swap_watcher.py (flows 4042/4041 baked in,
direct-SQL requeue). Everything is now a CLI arg; nothing is hardcoded.

Examples
--------
# Watch the currently-running flow; when it ends, swap in a new wheel + restart:
  python scripts/worker_swap.py --auto-detect-running \
      --install-wheel /path/to/ram_accelerator-*.whl --rayon-threads 10

# Watch a specific flow, stop only (no relaunch — manual control afterwards):
  python scripts/worker_swap.py --watch-flow 4321 --no-restart

Detach it (survives the launching session):
  python scripts/detach_launch.py -- python scripts/worker_swap.py ...
  (or: setsid nohup python scripts/worker_swap.py ... &)

Notes
-----
* Status detection reads the SQLite DB (read-only) — robust if the dashboard
  restarts. Re-queue uses POST /api/flows/:id/restart (keeps the checkpoint), the
  canonical path that applies the lifecycle logic (CLAUDE.md Rule 2: never flip
  flow rows by hand).
* Writes a completion marker (--marker) so a monitor/cron can detect the swap and
  trigger follow-up work (this script can't send chat notifications itself).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TERMINAL = {"completed", "failed", "cancelled"}


def get_flow_status(db: Path, flow_id: int) -> str | None:
	con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
	try:
		row = con.execute("SELECT status FROM flows WHERE id=?", (flow_id,)).fetchone()
	finally:
		con.close()
	return row[0] if row else None


def running_flow_ids(db: Path) -> list[int]:
	con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
	try:
		rows = con.execute("SELECT id FROM flows WHERE status='running'").fetchall()
	finally:
		con.close()
	return [int(r[0]) for r in rows]


def find_worker_pid() -> int | None:
	"""PID of the Python worker interpreter (not the zsh launcher)."""
	out = subprocess.run(["pgrep", "-af", "wnn.ram.experiments.worker"],
	                     capture_output=True, text=True).stdout
	rows = [ln.split(None, 1) for ln in out.splitlines() if ln.strip()]
	for pid_s, cmd in [(r[0], r[1] if len(r) > 1 else "") for r in rows]:
		if "Python.framework" in cmd or cmd.lstrip().startswith("/opt") or cmd.lstrip().startswith("/Volumes"):
			return int(pid_s)
	return int(rows[0][0]) if rows else None


def flow_runner_pids() -> list[int]:
	"""PIDs of any flow_runner subprocesses (the worker's per-flow children)."""
	out = subprocess.run(["pgrep", "-f", "wnn.ram.experiments.flow_runner"],
	                     capture_output=True, text=True).stdout
	return [int(x) for x in out.split()]


def _kill_pid(pid: int, grace_secs: int, label: str) -> None:
	try:
		os.kill(pid, signal.SIGTERM)
	except ProcessLookupError:
		return
	for _ in range(grace_secs):
		time.sleep(1)
		try:
			os.kill(pid, 0)
		except ProcessLookupError:
			print(f"[swap] {label} {pid} exited cleanly", flush=True)
			return
	print(f"[swap] {label} {pid} still alive after {grace_secs}s — SIGKILL", flush=True)
	try:
		os.kill(pid, signal.SIGKILL)
	except ProcessLookupError:
		pass


def stop_worker(pid: int, grace_secs: int) -> None:
	# Snapshot the worker's flow_runner children BEFORE killing it: SIGKILL does
	# NOT propagate to children, so a worker stuck mid-gen (past its grace) would
	# orphan its flow_runner (PPID=1), leaving it running the OLD wheel alongside
	# the relaunched worker — double memory + a stale 'running' DB row. We reap
	# them explicitly. Safe because stop_worker runs BEFORE relaunch, so every
	# flow_runner here belongs to the worker being replaced.
	children = flow_runner_pids()
	print(f"[swap] SIGTERM worker PID={pid} (flow_runner children: {children or 'none'})", flush=True)
	_kill_pid(pid, grace_secs, "worker")
	for cpid in flow_runner_pids():
		if cpid in children:
			print(f"[swap] reaping orphaned flow_runner {cpid}", flush=True)
			_kill_pid(cpid, grace_secs, "flow_runner")


def install_wheel(venv: Path, wheel: Path) -> None:
	print(f"[swap] pip install --force-reinstall {wheel.name}", flush=True)
	pip = venv / "bin" / "pip"
	r = subprocess.run([str(pip), "install", "--force-reinstall", "--no-deps", str(wheel)],
	                   capture_output=True, text=True)
	print((r.stdout or "")[-400:], flush=True)
	if r.returncode != 0:
		print(f"[swap] ⚠️ pip install FAILED: {r.stderr[-600:]}", flush=True)
		raise SystemExit(3)


def relaunch_worker(project: Path, venv: Path, rayon: str | None, worker_args: str,
                    log: Path) -> int | None:
	env_lines = [
		f"cd {project}",
		"unset CONDA_PREFIX",
		f"source {venv}/bin/activate",
		'export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"',
	]
	if rayon:
		env_lines.append(f"export RAYON_NUM_THREADS={rayon}")
	env_lines.append(
		f"nohup python -u -B -m wnn.ram.experiments.worker {worker_args} "
		f"</dev/null >>{log} 2>&1 &")
	cmd = " && ".join(env_lines)
	print(f"[swap] relaunching worker (rayon={rayon or 'unset'}) → {log}", flush=True)
	subprocess.Popen(["zsh", "-c", cmd], start_new_session=True)
	time.sleep(3)
	pid = find_worker_pid()
	print(f"[swap] new worker PID={pid}", flush=True)
	return pid


def api_restart_flow(url: str, flow_id: int, verify: bool) -> bool:
	"""POST /api/flows/:id/restart {from_beginning:false} — requeue from checkpoint."""
	body = json.dumps({"from_beginning": False}).encode()
	req = urllib.request.Request(f"{url}/api/flows/{flow_id}/restart", data=body,
	                             headers={"Content-Type": "application/json"}, method="POST")
	ctx = None
	if url.startswith("https") and not verify:
		ctx = ssl.create_default_context()
		ctx.check_hostname = False
		ctx.verify_mode = ssl.CERT_NONE
	try:
		with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
			ok = 200 <= resp.status < 300
			print(f"[swap] requeue flow {flow_id}: HTTP {resp.status}", flush=True)
			return ok
	except Exception as e:  # noqa: BLE001 — report + continue; caller logs the miss
		print(f"[swap] ⚠️ requeue flow {flow_id} failed: {e}", flush=True)
		return False


def write_marker(marker: Path | None, payload: dict) -> None:
	if marker is None:
		return
	marker.write_text(json.dumps(payload, indent=2))
	print(f"[swap] wrote marker {marker}", flush=True)


def resolve_watch_flow(args, db: Path) -> int:
	if getattr(args, "stop_now", False):
		running = running_flow_ids(db)
		print(f"[swap] --stop-now: stopping the worker immediately (running={running} "
		      f"will be re-queued from checkpoint).", flush=True)
		return -1
	if args.watch_flow is not None:
		return args.watch_flow
	running = running_flow_ids(db)
	if not running:
		print("[swap] --auto-detect-running: no flow is currently 'running'. "
		      "Nothing to wait for — proceeding to swap immediately.", flush=True)
		return -1
	# Watch the lowest-id running flow (FIFO head; the others requeue after stop).
	wf = min(running)
	print(f"[swap] auto-detected running flow {wf} (running={running})", flush=True)
	return wf


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	g = ap.add_mutually_exclusive_group(required=True)
	g.add_argument("--watch-flow", type=int, help="Flow ID to wait for completion.")
	g.add_argument("--auto-detect-running", action="store_true",
	               help="Watch the currently-running flow (lowest id).")
	g.add_argument("--stop-now", action="store_true",
	               help="Stop the worker IMMEDIATELY (skip waiting for the running flow); "
	                    "interrupted flows are re-queued from checkpoint.")
	ap.add_argument("--install-wheel", type=Path, default=None,
	                help="Wheel to pip-install --force-reinstall after stopping (optional).")
	ap.add_argument("--rayon-threads", type=str, default=None,
	                help="RAYON_NUM_THREADS for the relaunched worker (optional).")
	ap.add_argument("--no-restart", action="store_true",
	                help="Stop the worker but do NOT relaunch (manual control).")
	ap.add_argument("--dashboard-url", default="https://localhost:3000")
	ap.add_argument("--ssl-verify", action="store_true",
	                help="Verify TLS (default: off, matching --no-ssl-verify localhost).")
	ap.add_argument("--worker-args", default="--url https://localhost:3000 --no-ssl-verify",
	                help="Args passed to the relaunched worker module.")
	ap.add_argument("--db", type=Path, default=Path("/Users/lacg/wnn/db/wnn.db"))
	ap.add_argument("--project", type=Path, default=Path("/Users/lacg/wnn"))
	ap.add_argument("--venv", type=Path, default=Path("/Users/lacg/wnn/wnn"))
	ap.add_argument("--worker-log", type=Path, default=Path("/tmp/wnn_worker.log"))
	ap.add_argument("--marker", type=Path, default=None,
	                help="Write a JSON completion marker here (for a monitor/cron).")
	ap.add_argument("--poll-secs", type=int, default=30)
	ap.add_argument("--grace-secs", type=int, default=20)
	args = ap.parse_args()

	if args.install_wheel is not None and not args.install_wheel.exists():
		print(f"[swap] ERROR: --install-wheel not found: {args.install_wheel}", file=sys.stderr)
		return 2

	watch = resolve_watch_flow(args, args.db)

	# 1) Wait for the watched flow to reach a terminal state.
	if watch >= 0:
		print(f"[swap] watching flow {watch} (poll {args.poll_secs}s) …", flush=True)
		while True:
			st = get_flow_status(args.db, watch)
			print(f"[swap] flow {watch} status={st!r}", flush=True)
			if st is None:
				print(f"[swap] flow {watch} vanished — proceeding with swap.", flush=True)
				break
			if st in TERMINAL:
				break
			time.sleep(args.poll_secs)

	# 2) Snapshot flows the worker still owns (will be interrupted by the stop).
	interrupted = [f for f in running_flow_ids(args.db) if f != watch]

	# 3) Stop the worker.
	pid = find_worker_pid()
	if pid is None:
		print("[swap] no worker process found (already down).", flush=True)
	else:
		stop_worker(pid, args.grace_secs)

	# 4) Optional wheel install (worker is down → safe to replace the shared .so).
	if args.install_wheel is not None:
		install_wheel(args.venv, args.install_wheel)

	# 5) Relaunch the worker on the (possibly new) wheel.
	new_pid = None
	if args.no_restart:
		print("[swap] --no-restart: leaving the worker DOWN.", flush=True)
	else:
		new_pid = relaunch_worker(args.project, args.venv, args.rayon_threads,
		                          args.worker_args, args.worker_log)

	# 6) Re-queue interrupted flows via the API (keep checkpoints).
	requeued = []
	if not args.no_restart:
		for fid in interrupted:
			if api_restart_flow(args.dashboard_url, fid, args.ssl_verify):
				requeued.append(fid)

	write_marker(args.marker, {
		"watched_flow": watch,
		"interrupted": interrupted,
		"requeued": requeued,
		"installed_wheel": str(args.install_wheel) if args.install_wheel else None,
		"new_worker_pid": new_pid,
		"restarted": not args.no_restart,
	})
	print(f"[swap] DONE — watched={watch} interrupted={interrupted} requeued={requeued} "
	      f"wheel={'yes' if args.install_wheel else 'no'} new_pid={new_pid}", flush=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
