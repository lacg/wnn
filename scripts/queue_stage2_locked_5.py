"""Queue stage-2 locked candidates (5 bit-widths × 3 fresh seeds = 15 flows).

These are the candidates that stay in top-7 regardless of where 100b lands:
  88b, 104b, 80b, 60b, 96b — locked positions 1-5 by mean train_cal F1.

Seeds 203/204/205 are fresh (not yet used). Combined with existing
seeds 201/202 these give n=5 per bit-width for stage 2 statistics.

Status='pending' so the worker doesn't pick them up until we explicitly
flip them. Flip happens when r1839 (100b r202) transitions queued→running
— that's when the watcher fires.

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

LOCKED_BITS = [88, 104, 80, 60, 96]   # ranked 1-5 by mean F1, locked regardless of 100b
SEEDS = [203, 204, 205]                # fresh seeds (201/202 already done in stage 1)

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
for n_bits in LOCKED_BITS:
	for seed in SEEDS:
		params = dict(base_params)
		params["ids_n_bits"] = n_bits
		params["seed"] = seed
		name = f"BITSWEEP-neto-sub-{n_bits:02d}b-r{seed}"
		body = {
			"name": name,
			"description": (
				f"Stage 2 (n=5 expansion). ids_n_bits={n_bits}, seed={seed}. "
				f"Locked top-5 candidate; will run after watcher flips pending→queued "
				f"once r1839 (100b r202) starts."
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

# Verify all are pending (NOT calling /restart — leave them pending).
print()
print("Final state — all should be 'pending' (worker won't pick them up):")
con = sqlite3.connect(str(DB_PATH))
for fid, *_ in sorted(created, key=lambda x: -x[0]):
	r = con.execute("SELECT id, name, status FROM flows WHERE id = ?", (fid,)).fetchone()
	exps = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,)).fetchone()[0]
	flag = "✓" if exps == 2 and r[2] == "pending" else "✗"
	print(f"  {flag} id={r[0]}  status={r[2]:<10}  experiments={exps}  {r[1]}")
con.close()

# Summary for the watcher script
ids = sorted([fid for fid, *_ in created], reverse=True)
print()
print(f"Pending flow IDs ({len(ids)}): {ids}")
print(f"Watcher trigger: when r1839 (100b r202) status changes from 'queued' to 'running'.")
