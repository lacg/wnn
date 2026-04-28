"""Watcher: full autonomous post-r124 pipeline.

Polls the DB every 5 minutes. When flow 1686 (r124) reaches status='completed':
  1. Stop worker
  2. Run scripts/apply_per_class_integration.py:
     - Edits ids_evaluator.py + experiment.py (per-class integration)
     - Edits adaptive_cluster.py (genome-storage JSON fix)
     - Edits dashboard types.ts + experiment Svelte (per-class display)
     - ast.parse + smoke test (UNSW small load)
     - git commit + push
     - On failure: reverts ALL files via git checkout, continues with old code
  3. Restart worker (loads new Python on success, old on revert)
  4. Restart dashboard backend (picks up nothing in Rust; vite hot-reloads
     Svelte/types.ts independently)
  5. Wait for dashboard to be up (poll until ready)
  6. Run scripts/queue_all_post_r124_flows.py — race-safe order:
     a. Create 2 Phase D as PENDING (lower IDs)
     b. Create 112 PUB50 as PENDING (higher IDs)
     c. Queue PUB50 first  ← worker can now pick; highest IDs run first
     d. Queue Phase D last ← still has lower IDs; runs after all PUB50
     Avoids the race where Phase D is picked before PUB50 exists.
  7. Exit — worker runs all 114 queued flows sequentially.
     Worker picks highest ID first (ORDER BY id DESC) → PUB50 runs before Phase D.

DOES NOT run per-class on r125/r124 (bencorn dataset superseded).
DOES integrate per-class into worker validation for ALL future flows.

Safe to run while r124 is still going — it just polls. Will sit idle waiting.

Usage:
    cd /Users/lacg/wnn
    source wnn-venv/bin/activate    # actually /Users/lacg/wnn-venv
    nohup python scripts/auto_per_class_when_r124_done.py > /tmp/auto_per_class.log 2>&1 &
"""

import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import signal

DB_PATH = Path("/Users/lacg/wnn/db/wnn.db")
ANALYSIS_DIR = Path("/Users/lacg/wnn/analysis")
SCRIPT = Path("/Users/lacg/wnn/scripts/per_class_analysis.py")
POLL_SECS = 300  # 5 min
TARGET_FLOW = 1686       # r124 — the one we wait for
ALSO_ANALYZE = [1687, 1686]  # r125 first (older), then r124


def log(msg):
	stamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
	print(f"[{stamp}] {msg}", flush=True)


def get_flow_status(flow_id: int) -> str:
	con = sqlite3.connect(str(DB_PATH))
	row = con.execute("SELECT status FROM flows WHERE id = ?", (flow_id,)).fetchone()
	con.close()
	return row[0] if row else "missing"


def any_flow_running() -> bool:
	con = sqlite3.connect(str(DB_PATH))
	row = con.execute("SELECT count(*) FROM flows WHERE status = 'running'").fetchone()
	con.close()
	return row[0] > 0


def run_analysis(flow_id: int) -> int:
	"""Run per_class_analysis.py for one flow. Returns exit code."""
	ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
	stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
	out_file = ANALYSIS_DIR / f"per_class_flow{flow_id}_{stamp}.md"
	log(f"Running per_class_analysis for flow {flow_id} → {out_file}")
	cmd = [
		sys.executable, str(SCRIPT),
		"--flow-id", str(flow_id),
		"--metrics", "f1_macro", "fpr", "accuracy",
		"--out", str(out_file),
	]
	env = os.environ.copy()
	env["PYTHONPATH"] = "/Users/lacg/wnn/src/wnn:" + env.get("PYTHONPATH", "")
	p = subprocess.run(cmd, env=env, capture_output=True, text=True)
	if p.returncode != 0:
		log(f"  Flow {flow_id} analysis FAILED (exit={p.returncode})")
		log(f"    stderr tail:\n{p.stderr[-2000:]}")
	else:
		log(f"  Flow {flow_id} analysis OK ({out_file.stat().st_size if out_file.exists() else '?'} bytes)")
	return p.returncode


def find_worker_pid() -> int | None:
	"""Find the FlowWorker process PID."""
	try:
		out = subprocess.check_output(["pgrep", "-f", "wnn.ram.experiments.worker"], text=True)
		pids = [int(p) for p in out.strip().split("\n") if p.strip()]
		return pids[0] if pids else None
	except subprocess.CalledProcessError:
		return None


def find_dashboard_pid() -> int | None:
	"""Find the wnn-dashboard backend (Rust binary) PID."""
	try:
		out = subprocess.check_output(["pgrep", "-f", "wnn-dashboard"], text=True)
		pids = [int(p) for p in out.strip().split("\n") if p.strip()]
		return pids[0] if pids else None
	except subprocess.CalledProcessError:
		return None


def stop_dashboard():
	pid = find_dashboard_pid()
	if not pid:
		log("  No dashboard process found.")
		return
	log(f"  Stopping dashboard (PID {pid}) with SIGTERM...")
	try:
		os.kill(pid, signal.SIGTERM)
	except ProcessLookupError:
		return
	for _ in range(20):
		try:
			os.kill(pid, 0)
		except ProcessLookupError:
			log("  Dashboard exited cleanly.")
			return
		time.sleep(1)
	log("  Dashboard didn't exit in 20s — sending SIGKILL.")
	try: os.kill(pid, signal.SIGKILL)
	except ProcessLookupError: pass


def restart_dashboard():
	"""Launch wnn-dashboard from /Users/lacg/wnn/dashboard/ (so DATABASE_URL ../db/wnn.db resolves)."""
	log("  Launching dashboard backend...")
	logfile = Path("/tmp/dashboard_post_r124.out")
	proc = subprocess.Popen(
		["./target/release/wnn-dashboard"],
		cwd="/Users/lacg/wnn/dashboard",
		stdout=open(logfile, "w"),
		stderr=subprocess.STDOUT,
		start_new_session=True,
	)
	log(f"  Dashboard started PID {proc.pid}, log: {logfile}")
	# Poll for HTTPS readiness up to 30s
	import urllib.request
	import ssl
	ctx = ssl.create_default_context()
	ctx.check_hostname = False
	ctx.verify_mode = ssl.CERT_NONE
	for _ in range(30):
		try:
			urllib.request.urlopen("https://localhost:3000/api/flows?limit=1", context=ctx, timeout=2)
			log("  ✓ Dashboard responding.")
			return
		except Exception:
			time.sleep(1)
	log("  ⚠ Dashboard did not respond within 30s — continuing anyway.")


def stop_worker():
	pid = find_worker_pid()
	if not pid:
		log("  No worker process found — already stopped.")
		return
	log(f"  Stopping worker (PID {pid}) with SIGTERM...")
	try:
		os.kill(pid, signal.SIGTERM)
	except ProcessLookupError:
		return
	# Wait up to 5 min for graceful exit
	for _ in range(60):
		try:
			os.kill(pid, 0)
		except ProcessLookupError:
			log("  Worker exited cleanly.")
			return
		time.sleep(5)
	log("  Worker didn't exit after 5min — sending SIGKILL.")
	try: os.kill(pid, signal.SIGKILL)
	except ProcessLookupError: pass


def restart_worker():
	"""Launch a fresh worker (picks up latest code, including neto_full/subsample dataset names)."""
	log("  Launching new worker...")
	env = os.environ.copy()
	env["PYTHONPATH"] = "/Users/lacg/wnn/src/wnn:" + env.get("PYTHONPATH", "")
	# Need venv-activated python — use absolute path
	python = "/Users/lacg/wnn-venv/bin/python"
	logfile = Path("/tmp/worker_post_r124.out")
	# Use Popen with start_new_session for proper detachment
	proc = subprocess.Popen(
		[python, "-u", "-m", "wnn.ram.experiments.worker"],
		env=env,
		cwd="/Users/lacg/wnn",
		stdout=open(logfile, "w"),
		stderr=subprocess.STDOUT,
		start_new_session=True,
	)
	log(f"  Worker started PID {proc.pid}, log: {logfile}")
	# Sanity check after 5s
	time.sleep(5)
	if proc.poll() is None:
		log("  ✓ Worker alive after 5s.")
	else:
		log(f"  ✗ Worker died with exit code {proc.returncode}. Log tail:")
		log(open(logfile).read()[-1500:])


def run_subprocess(name: str, script_path: str, env_extra: dict | None = None, capture: bool = True) -> int:
	env = os.environ.copy()
	env["PYTHONPATH"] = "/Users/lacg/wnn/src/wnn:" + env.get("PYTHONPATH", "")
	if env_extra: env.update(env_extra)
	log(f"  Running {name}: {script_path}")
	if capture:
		p = subprocess.run([sys.executable, script_path], env=env, capture_output=True, text=True)
		if p.returncode == 0:
			log(f"    ✓ {name} OK. stdout tail:\n{p.stdout[-800:]}")
		else:
			log(f"    ✗ {name} FAILED (exit={p.returncode}). stderr tail:\n{p.stderr[-1500:]}")
		return p.returncode
	else:
		p = subprocess.run([sys.executable, script_path], env=env)
		return p.returncode


def main():
	log(f"Watcher started. Polling every {POLL_SECS}s for flow {TARGET_FLOW} completion.")
	while True:
		status = get_flow_status(TARGET_FLOW)
		if status == "completed":
			log(f"Flow {TARGET_FLOW} is COMPLETED.")
			# Wait for worker to be fully idle (no other running flows)
			while any_flow_running():
				log("  Other flow(s) still running — waiting another poll cycle.")
				time.sleep(POLL_SECS)

			# ── Step 1: Stop worker ───────────────────────────────────────
			log("Step 1/6: Stopping worker.")
			stop_worker()

			# ── Step 2: Apply per-class + genome-storage + dashboard integration ──
			# This script edits 5 files (ids_evaluator.py, experiment.py,
			# adaptive_cluster.py, types.ts, +page.svelte), runs a smoke test,
			# and commits + pushes. On failure, all files are reverted via
			# `git checkout HEAD --` and the script exits non-zero. The
			# watcher continues either way (Step 3+) — restart with old code on revert.
			log("Step 2/6: Applying per-class + genome-storage + dashboard integration.")
			apply_rc = run_subprocess(
				"apply_integration",
				str(Path(__file__).resolve().parent / "apply_per_class_integration.py"),
			)
			if apply_rc == 0:
				log("  ✓ Apply succeeded.")
			else:
				log(f"  ⚠ Apply FAILED (exit={apply_rc}). Continuing with OLD code.")

			# ── Step 3: Restart worker (loads new Python on success, old on revert) ──
			log("Step 3/6: Restarting worker with latest committed code.")
			restart_worker()

			# ── Step 4: Restart dashboard backend (vite hot-reloads frontend on its own) ──
			log("Step 4/6: Restarting dashboard backend.")
			stop_dashboard()
			restart_dashboard()

			# ── Step 5: Create + queue ALL post-r124 flows in race-safe order ──
			# Single combined script:
			#   1. Create Phase D as PENDING (worker can't pick pending)
			#   2. Create 112 PUB50 as PENDING (higher IDs since created later)
			#   3. Queue PUB50 first → worker picks PUB50 highest-ID first
			#   4. Queue Phase D last → still has lower IDs, runs after all PUB50
			# Avoids the race where Phase D would be picked before PUB50 exists.
			log("Step 5/5: Creating + queueing all flows (race-safe order).")
			run_subprocess("queue_all_post_r124", str(Path(__file__).resolve().parent / "queue_all_post_r124_flows.py"))

			log("All steps complete. Worker is running queued flows. Watcher exiting.")
			log("Order: PUB50 1.43M flows run FIRST (highest IDs), then 2 Phase D 46M flows.")
			return
		elif status in ("failed", "cancelled"):
			log(f"Flow {TARGET_FLOW} ended in {status} state — exiting without further action.")
			return
		else:
			log(f"  flow {TARGET_FLOW} status={status} — sleeping {POLL_SECS}s")
			time.sleep(POLL_SECS)


if __name__ == "__main__":
	main()
