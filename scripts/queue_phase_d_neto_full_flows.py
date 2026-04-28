"""Queue 2 new Phase D WNN flows on the canonical 46.7M neto-full dataset.

Same r98-style config (cloned from flow 1156) but pointed at
ciciot2023_neto_full instead of ciciot2023_full. Two seeds for variance.

Names: PUB-neto-full-46M-8b-r{144,145} (past r124/r125 + buffer)

Per CLAUDE.md Rule 2: ALWAYS create flows via dashboard POST /api/flows.
"""

import requests, urllib3, json, sys, sqlite3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"

# Clone r98's params, override what changes for neto-full
con = sqlite3.connect(str(DB_PATH))
row = con.execute("SELECT config_json FROM flows WHERE id = 1156").fetchone()
con.close()
if not row:
	print("ERROR: r98 (id=1156) not found"); sys.exit(1)
base_params = dict(json.loads(row[0])["params"])

base_params["ids_dataset"] = "ciciot2023_neto_full"
base_params["ids_split"] = "random_3way"
base_params["ids_invalid_encoding"] = "single_bit"  # explicit (auto-defaults same)

SEEDS = [144, 145]

experiments = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]

created_ids = []
for seed in SEEDS:
	params = dict(base_params)
	params["seed"] = seed
	name = f"PUB-neto-full-46M-8b-r{seed}"
	body = {
		"name": name,
		"description": f"Phase D / neto-full (canonical Neto 46.7M, 46 features) / single_bit / seed={seed}. "
					   "Compare against r125+r124 (canonical-neto bencorn 45M) and r98 baseline.",
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": experiments,
	}
	print(f"Creating flow: {name}...")
	resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
	if resp.status_code in (200, 201):
		fid = resp.json().get("id")
		print(f"  ✓ Created flow id={fid}")
		# Set status to queued so worker picks it up
		r2 = requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=10)
		if r2.status_code == 200:
			print(f"  ✓ Queued flow {fid}")
			created_ids.append(fid)
		else:
			print(f"  ⚠ Restart-to-queue failed: {r2.status_code} {r2.text[:200]}")
	else:
		print(f"  ✗ Failed ({resp.status_code}): {resp.text[:300]}")
		sys.exit(2)

print(f"\nCreated + queued {len(created_ids)} flows: {created_ids}")
