"""Stage 3: ±2b around 96b → 94b and 98b × seeds 201-207 = 14 flows.

96b's deployable champion (90.31/4.40/94.79) is the rare/hard achievement.
Stage 3 tests whether ±2b can sharpen the F1 ceiling or deployable point.
8-bit periodicity predicts 94b dips (between 88b peak and 96b peak) and
98b sits in the 96b plateau — interesting either way.

Per CLAUDE.md Rule 2: dashboard POST /api/flows.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"

PM2_BITS = [94, 98]
SEEDS = list(range(201, 209))  # n=8 to match top-4

con = sqlite3.connect(str(DB_PATH))
row = con.execute("SELECT config_json FROM flows WHERE id = 1812").fetchone()
con.close()
if not row:
	print("ERROR: r1812 base config not found.")
	sys.exit(1)
base_params = dict(json.loads(row[0])["params"])

experiments = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]

# Alternate creation order so id-DESC produces alternation:
# Worker pulls highest id first, so create lowest-priority first.
# We want highest id = 94b r208 (runs first), then 98b r208, then 94b r207, ...
# So create order is reversed: lowest seed first, alternating 98b before 94b per seed.
create_order = []
for seed in SEEDS:  # 201, 202, ..., 208
	create_order.append((98, seed))  # created first → lower id → runs later
	create_order.append((94, seed))  # created second → higher id → runs first
# Reverse pair order so 94b runs first per seed pair (i.e., 94b r208 has highest id)
created = []
for n_bits, seed in create_order:
		params = dict(base_params)
		params["ids_n_bits"] = n_bits
		params["seed"] = seed
		name = f"BITSWEEP-neto-sub-{n_bits:02d}b-r{seed}"
		body = {
			"name": name,
			"description": (
				f"Stage 3 ±2b around 96b winner. ids_n_bits={n_bits}, seed={seed}."
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
			time.sleep(1)
		else:
			print(f"  ✗ Failed ({resp.status_code}): {resp.text[:200]}")
			sys.exit(2)

print()
print("Flipping pending → queued...")
for fid, n_bits, seed, name in created:
	requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
	time.sleep(1)

print()
print("Final state:")
con = sqlite3.connect(str(DB_PATH))
for fid, n_bits, seed, name in sorted(created, key=lambda x: -x[0]):
	r = con.execute("SELECT id, name, status FROM flows WHERE id = ?", (fid,)).fetchone()
	exps = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,)).fetchone()[0]
	flag = "✓" if exps == 2 and r[2] in ("queued", "running") else "✗"
	print(f"  {flag} id={r[0]}  status={r[2]:<10}  experiments={exps}  {r[1]}")
con.close()
