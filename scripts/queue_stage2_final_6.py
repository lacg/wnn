"""Queue stage-2 final 6 flows after 100b verdict — Case A locked.

100b mean F1 = 88.68±0.44, below 64b's 89.54. Top-7 unchanged:
  88b, 104b, 80b, 60b, 96b, 48b, 64b

The first 5 (88/104/80/60/96b) have stage-2 already running (queued by
queue_stage2_locked_5.py + flipped via stage2_flip_watcher.sh).

This script adds the remaining 2 candidates × 3 fresh seeds = 6 flows:
  48b ×3 (203/204/205)  — cheapest deployable, completes top-6 by mean F1
  64b ×3 (203/204/205)  — only sub-5% FPR specialist (FPR uniqueness)

Per CLAUDE.md Rule 2: dashboard POST /api/flows.
"""

import json
import sqlite3
import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"

FINAL_BITS = [48, 64]
SEEDS = [203, 204, 205]

con = sqlite3.connect(str(DB_PATH))
row = con.execute("SELECT config_json FROM flows WHERE id = 1812").fetchone()
con.close()
if not row:
	print("ERROR: r1812 not found.")
	sys.exit(1)
base_params = dict(json.loads(row[0])["params"])

experiments = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]

created = []
for n_bits in FINAL_BITS:
	for seed in SEEDS:
		params = dict(base_params)
		params["ids_n_bits"] = n_bits
		params["seed"] = seed
		name = f"BITSWEEP-neto-sub-{n_bits:02d}b-r{seed}"
		body = {
			"name": name,
			"description": (
				f"Stage 2 final 6 — Case A (100b below 64b, top-7 unchanged). "
				f"ids_n_bits={n_bits}, seed={seed}."
			),
			"config": {"template": "ids-binary-2-phase", "params": params},
			"experiments": experiments,
		}
		print(f"Creating: {name}...")
		resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
		if resp.status_code in (200, 201):
			fid = resp.json()["id"]
			created.append((fid, n_bits, seed, name))
			print(f"  ✓ id={fid}")
		else:
			print(f"  ✗ Failed ({resp.status_code}): {resp.text[:200]}")
			sys.exit(2)

# Flip pending → queued (gentle pacing — avoid the burst-POST race we hit before)
import time
print()
print("Flipping pending → queued...")
for fid, n_bits, seed, name in created:
	resp = requests.post(
		f"{DASHBOARD}/api/flows/{fid}/restart",
		json={}, verify=False, timeout=15,
	)
	time.sleep(1)
	# Verify
	r = requests.get(f"{DASHBOARD}/api/flows/{fid}", verify=False, timeout=15)
	status = r.json().get("status", "unknown") if r.ok else "(unknown)"
	print(f"  {fid:>5}  ({n_bits:>2}b r{seed}) → {status}")

print()
print("Final state:")
con = sqlite3.connect(str(DB_PATH))
for fid, *_ in sorted(created, key=lambda x: -x[0]):
	r = con.execute("SELECT id, name, status FROM flows WHERE id = ?", (fid,)).fetchone()
	exps = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,)).fetchone()[0]
	flag = "✓" if exps == 2 else "✗"
	print(f"  {flag} id={r[0]}  status={r[2]:<10}  experiments={exps}  {r[1]}")
con.close()
