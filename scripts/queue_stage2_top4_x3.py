"""Queue stage-2 top-4 expansion: 104b, 96b, 88b, 80b × seeds 206/207/208.

User decision after morning synthesis: top-4 candidates (88b mean leader, 104b
ceiling, 96b speed/FPR, 80b balanced) deserve 3 more fresh seeds each before
locking the stage-3 ±2b call.

Per CLAUDE.md Rule 2: dashboard POST /api/flows with experiments inline.
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

TOP4_BITS = [104, 96, 88, 80]
SEEDS = [206, 207, 208]

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

created = []
for n_bits in TOP4_BITS:
	for seed in SEEDS:
		params = dict(base_params)
		params["ids_n_bits"] = n_bits
		params["seed"] = seed
		name = f"BITSWEEP-neto-sub-{n_bits:02d}b-r{seed}"
		body = {
			"name": name,
			"description": (
				f"Stage 2 top-4 expansion (n=8 target). "
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
			time.sleep(1)
		else:
			print(f"  ✗ Failed ({resp.status_code}): {resp.text[:200]}")
			sys.exit(2)

print()
print("Verifying experiments attached + status:")
con = sqlite3.connect(str(DB_PATH))
for fid, n_bits, seed, name in sorted(created, key=lambda x: -x[0]):
	r = con.execute("SELECT id, name, status FROM flows WHERE id = ?", (fid,)).fetchone()
	exps = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,)).fetchone()[0]
	flag = "✓" if exps == 2 else "✗"
	print(f"  {flag} id={r[0]}  status={r[2]:<10}  experiments={exps}  {r[1]}")
con.close()
