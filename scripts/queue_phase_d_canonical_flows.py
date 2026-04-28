"""Queue Phase D WNN flows on CIC-IoT-2023-canonical-neto via dashboard API.

Creates 2 flows (2 different seeds) of the headline experiment:
  ids_dataset = "ciciot2023_canonical"
  ids_invalid_encoding = "single_bit"   (auto-defaulted, made explicit here)
  ids_split = "random_3way"             (canonical-neto's preferred split)
  All other params = identical to r98 (flow id 1156)

Flow names: PUB-canonical-46M-8b-r{124,125}

Per CLAUDE.md Rule 2: ALWAYS create flows via dashboard POST /api/flows
(never direct SQL inserts) so defaults + experiment plumbing get applied.
"""

import requests, urllib3, json, sys, sqlite3, os
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"

# Pull r98's params verbatim, override only what changes for canonical-neto
con = sqlite3.connect(str(DB_PATH))
row = con.execute("SELECT config_json FROM flows WHERE id = 1156").fetchone()
con.close()
if not row:
	print("ERROR: r98 (id=1156) not found in DB — cannot clone params.")
	sys.exit(1)
r98_config = json.loads(row[0])
base_params = dict(r98_config["params"])

# Overrides for canonical-neto:
base_params["ids_dataset"] = "ciciot2023_canonical"
base_params["ids_split"] = "random_3way"   # canonical-neto's preferred (3-way available)
base_params["ids_invalid_encoding"] = "single_bit"  # explicit (worker would auto-default this anyway)

# Two seeds — pick fresh ones (r98 used 98; we use 124, 125 — past the r113-r123 UNSW range)
SEEDS = [124, 125]

experiments = [
	{
		"name": "Grid Search (neurons x bits)",
		"experiment_type": "grid_search",
		"phase_type": "grid_search",
	},
	{
		"name": "GA Neurons",
		"experiment_type": "ga",
		"phase_type": "ga_neurons",
	},
]

created_ids = []
for seed in SEEDS:
	params = dict(base_params)
	params["seed"] = seed
	name = f"PUB-canonical-46M-8b-r{seed}"
	body = {
		"name": name,
		"description": f"Phase D / canonical-neto / single_bit / seed={seed}. "
					   "Compare against r98 (38.5M, NaN-dropped) for the +6.5M-row + flag-bit ablation.",
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": experiments,
	}
	print(f"Creating flow: {name}...")
	resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
	if resp.status_code in (200, 201):
		flow = resp.json()
		fid = flow.get("id") if isinstance(flow, dict) else None
		print(f"  ✓ Created flow id={fid}")
		created_ids.append(fid)
	else:
		print(f"  ✗ Failed ({resp.status_code}): {resp.text[:300]}")
		sys.exit(2)

print(f"\nCreated {len(created_ids)} flows: {created_ids}")
print("Verify in DB:")
con = sqlite3.connect(str(DB_PATH))
for fid in created_ids:
	row = con.execute("SELECT id, name, status FROM flows WHERE id = ?", (fid,)).fetchone()
	exps = con.execute("SELECT id, name, phase_type FROM experiments WHERE flow_id = ? ORDER BY sequence_order", (fid,)).fetchall()
	print(f"  Flow {row[0]}: {row[1]} ({row[2]}) — {len(exps)} experiments: {[(e[0], e[2]) for e in exps]}")
con.close()
