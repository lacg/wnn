"""Watcher: auto-run per_class_analysis.py for r125 + r124 when r124 finishes.

Polls the DB every 5 minutes. When flow 1686 (r124) reaches status='completed',
verifies the worker is idle (no other running flows), then sequentially:
  1. Run per_class_analysis.py --flow-id 1687 (r125)
  2. Run per_class_analysis.py --flow-id 1686 (r124)
Writes results to /Users/lacg/wnn/analysis/per_class_r{125,124}_<timestamp>.md.

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
			# Worker idle — safe to use GPU
			log("Worker idle. Starting per-class analyses.")
			for fid in ALSO_ANALYZE:
				if get_flow_status(fid) != "completed":
					log(f"  Skipping flow {fid} (status={get_flow_status(fid)})")
					continue
				run_analysis(fid)

			# Chain into camera-ready draft generation
			log("Per-class analyses done. Generating camera-ready draft...")
			env = os.environ.copy()
			env["PYTHONPATH"] = "/Users/lacg/wnn/src/wnn:" + env.get("PYTHONPATH", "")
			p = subprocess.run(
				[sys.executable, str(Path(__file__).resolve().parent / "draft_camera_ready_update.py")],
				env=env, capture_output=True, text=True,
			)
			if p.returncode == 0:
				log("  ✓ Camera-ready draft generated.")
				log(f"    stdout tail:\n{p.stdout[-1500:]}")
			else:
				log(f"  ✗ Draft generation FAILED (exit={p.returncode}). stderr:\n{p.stderr[-1500:]}")
			log("Watcher complete. Exiting.")
			return
		elif status in ("failed", "cancelled"):
			log(f"Flow {TARGET_FLOW} ended in {status} state — exiting without analysis.")
			return
		else:
			log(f"  flow {TARGET_FLOW} status={status} — sleeping {POLL_SECS}s")
			time.sleep(POLL_SECS)


if __name__ == "__main__":
	main()
