"""Watcher: auto-chain post-r124 work when flow 1686 finishes.

Polls the DB every 5 minutes. When flow 1686 (r124) reaches status='completed':
  1. Stop the worker
  2. Restart the worker (picks up new ciciot2023_neto_full + neto_subsample
     dataset names AND any pending Python code changes for built-in per-class)
  3. Queue 2 new Phase D flows on neto-full (~7-14 days each)
  4. Queue 112 PUB50 flows on neto-subsample (~6-7 days)

Then exits — the worker runs all queued flows sequentially.

Per-class on r125+r124 is INTENTIONALLY SKIPPED — those ran on the superseded
bencorn-MERGED dataset (45M / 39 features) which we've replaced with neto-full
(46.7M / 46 features). Per-class will be computed by the worker's own validation
phase on every new flow (assuming worker code has been updated for it). Otherwise
per_class_analysis.py remains available for ad-hoc use on any flow.

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

			# Step 1: Stop worker
			log("Step 1/3: Stopping worker.")
			stop_worker()

			# Step 2: Restart worker — picks up latest code (new dataset names + any
			# per-class integration committed to main since the running worker started)
			log("Step 2/3: Restarting worker with latest committed code.")
			restart_worker()

			# Step 3: Queue all the new flows
			log("Step 3/3: Queueing new flows (2 Phase D on neto-full + 112 PUB50 on neto-subsample).")
			run_subprocess("queue_phase_d_neto_full", str(Path(__file__).resolve().parent / "queue_phase_d_neto_full_flows.py"))
			run_subprocess("queue_pub50_neto_subsample", str(Path(__file__).resolve().parent / "queue_pub50_neto_subsample_flows.py"))

			log("All steps complete. Worker is running queued flows. Watcher exiting.")
			log("Per-class for r125+r124 SKIPPED (bencorn dataset superseded). Per-class on")
			log("new flows will come from the worker's own validation phase if integrated.")
			return
		elif status in ("failed", "cancelled"):
			log(f"Flow {TARGET_FLOW} ended in {status} state — exiting without further action.")
			return
		else:
			log(f"  flow {TARGET_FLOW} status={status} — sleeping {POLL_SECS}s")
			time.sleep(POLL_SECS)


if __name__ == "__main__":
	main()
