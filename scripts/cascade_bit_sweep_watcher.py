"""Autonomous cascade watcher for the bit-width sweep.

Watches the queue for the next bit-width's two seeds to complete, pulls
their best val_cal F1 (across all 5 best-genome types), and decides
whether to queue the next bit-width:

  Decision rule:
    if max val_cal F1 (over both seeds × 5 genomes) >= 90.5%  → queue next
    else                                                       → stop

Cascade progression: 92 → 96 → 104 → 112  (then stop unconditionally).

The threshold (90.5%) is set just below 88b r202's typical (~91.5%) to
allow for seed variance — if either seed hits 90.5+, that's a "matches
or exceeds 88b" signal worth extending on.

Designed to be nohup'd so it survives Claude session exit:
    nohup python3 scripts/cascade_bit_sweep_watcher.py \\
        > /tmp/cascade_watcher.log 2>&1 &

Polls every 5 min. Sends a brief one-liner to stdout on every state change
(queue, finish, decision) — tail the log to follow progress.
"""

import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB = "/Users/lacg/wnn/db/wnn.db"
DASHBOARD = "https://localhost:3000"
THRESHOLD = 90.5  # min val_cal F1% to extend cascade
POLL_SEC = 300    # 5 min between polls

# Each entry: (bit-width, list of pre-existing flow_ids — empty means we'll queue)
CASCADE_ORDER = [
	(92, [1834, 1835]),  # already queued
	(96, []),
	(104, []),
	(112, []),
]


def log(msg: str):
	ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print(f"[{ts}] {msg}", flush=True)


def db_query(sql, params=()):
	con = sqlite3.connect(DB)
	con.row_factory = sqlite3.Row
	cur = con.cursor()
	cur.execute(sql, params)
	rows = cur.fetchall()
	con.close()
	return rows


def flow_status(fid: int) -> str:
	rows = db_query("SELECT status FROM flows WHERE id = ?", (fid,))
	return rows[0]["status"] if rows else "unknown"


def flow_max_val_cal_f1(fid: int) -> float:
	"""Return max val_cal F1 across all 5 best-genome types (GA Neurons phase)."""
	rows = db_query(
		"""SELECT vs.threshold_metadata
		   FROM validation_summaries vs
		   JOIN experiments e ON e.id = vs.experiment_id
		   WHERE vs.flow_id = ?
		     AND e.phase_type = 'ga_neurons'
		     AND vs.validation_point = 'final'""",
		(fid,),
	)
	best = 0.0
	for r in rows:
		try:
			tm = json.loads(r["threshold_metadata"])
			vc = tm.get("val_cal", {})
			f1 = vc.get("f1")
			if f1 is not None and f1 * 100 > best:
				best = f1 * 100
		except Exception:
			pass
	return best


def queue_bits(n_bits: int) -> list[int]:
	"""POST 2 flows for the given bit-width via dashboard API. Returns flow IDs."""
	con = sqlite3.connect(DB)
	row = con.execute("SELECT config_json FROM flows WHERE id = 1812").fetchone()
	con.close()
	if not row:
		raise RuntimeError("r1812 not found — can't clone params")
	base = json.loads(row[0])["params"]

	experiments = [
		{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
		{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
	]

	ids = []
	for seed in [201, 202]:
		params = dict(base)
		params["ids_n_bits"] = n_bits
		params["seed"] = seed
		body = {
			"name": f"BITSWEEP-neto-sub-{n_bits:02d}b-r{seed}",
			"description": (
				f"Autonomous cascade extension: ids_n_bits={n_bits}, seed={seed}. "
				f"Triggered after the prior step hit val_cal F1 >= {THRESHOLD}%."
			),
			"config": {"template": "ids-binary-2-phase", "params": params},
			"experiments": experiments,
		}
		resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
		if resp.status_code not in (200, 201):
			raise RuntimeError(f"flow create failed: {resp.status_code} {resp.text[:200]}")
		fid = resp.json()["id"]
		# Flip to queued
		resp2 = requests.post(
			f"{DASHBOARD}/api/flows/{fid}/restart",
			json={}, verify=False, timeout=15,
		)
		if resp2.status_code not in (200, 201):
			raise RuntimeError(f"flow {fid} /restart failed: {resp2.status_code}")
		ids.append(fid)
	return ids


def wait_completed(fid1: int, fid2: int) -> bool:
	"""Poll until both flows leave running. Returns True if both completed cleanly."""
	while True:
		s1, s2 = flow_status(fid1), flow_status(fid2)
		if s1 in ("completed", "failed") and s2 in ("completed", "failed"):
			ok = s1 == "completed" and s2 == "completed"
			log(f"  flow {fid1}={s1}, flow {fid2}={s2}  {'✓' if ok else '✗'}")
			return ok
		time.sleep(POLL_SEC)


def main():
	log(f"Cascade watcher starting. Threshold={THRESHOLD}%. Cascade={[c[0] for c in CASCADE_ORDER]}b")
	for bits, prior_ids in CASCADE_ORDER:
		# Queue if not already
		if prior_ids:
			ids = prior_ids
			log(f"{bits}b already queued: {ids}")
		else:
			log(f"queueing {bits}b...")
			try:
				ids = queue_bits(bits)
				log(f"  queued {bits}b: {ids}")
			except Exception as e:
				log(f"  FAILED to queue {bits}b: {e}")
				return 1

		# Wait for both to complete
		log(f"waiting for {bits}b r{ids[0]}, r{ids[1]} to complete (poll every {POLL_SEC}s)...")
		if not wait_completed(ids[0], ids[1]):
			log(f"{bits}b had a failed flow — stopping cascade.")
			return 1

		# Get max val_cal F1 across both flows
		f1s = [flow_max_val_cal_f1(fid) for fid in ids]
		max_f1 = max(f1s)
		log(f"{bits}b val_cal F1 (best per flow): r{ids[0]}={f1s[0]:.2f}%, r{ids[1]}={f1s[1]:.2f}% → max={max_f1:.2f}%")

		# Decision
		if max_f1 >= THRESHOLD:
			log(f"  ≥ {THRESHOLD}% — extending cascade.")
		else:
			log(f"  < {THRESHOLD}% — STOPPING cascade. Last bit-width with strong result: {bits}b.")
			return 0

	log("Cascade reached upper bound (112b). Stopping unconditionally — needs human review for further extension.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
