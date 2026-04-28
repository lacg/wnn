"""Queue 112 PUB50 WNN flows on neto-subsample (1.43M, 46 features).

Replicates the original PUB50-ciciot-random-r{001..112} batch but pointed at
ciciot2023_neto_subsample (canonical Neto 1.43M, 46 features) instead of
the bencorn-derived ciciot2023 (1.3M, 39 features).

Names: PUB50-neto-sub-ciciot-random-r{001..112}

Per CLAUDE.md Rule 2: ALWAYS create flows via dashboard POST /api/flows.
Each flow is queued (restart→queued) so the worker picks them up sequentially.
"""

import requests, urllib3, json, sys, sqlite3, time
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"

# Clone the original PUB50-ciciot-random-r001 config
con = sqlite3.connect(str(DB_PATH))
row = con.execute(
	"SELECT config_json FROM flows WHERE name = 'PUB50-ciciot-random-r001'"
).fetchone()
con.close()
if not row:
	print("ERROR: PUB50-ciciot-random-r001 not found"); sys.exit(1)
base_params = dict(json.loads(row[0])["params"])

# Override for neto-subsample
base_params["ids_dataset"] = "ciciot2023_neto_subsample"
base_params["ids_split"] = "random"
base_params["ids_invalid_encoding"] = "single_bit"  # auto-default but explicit for clarity

# 112 flows, seeds 1..112
N_FLOWS = 112

experiments = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]

created_ids = []
for seed in range(1, N_FLOWS + 1):
	params = dict(base_params)
	params["seed"] = seed
	name = f"PUB50-neto-sub-ciciot-random-r{seed:03d}"
	body = {
		"name": name,
		"description": f"PUB50-style 112-flow batch on neto-subsample (1.43M, 46 features) / "
					   f"single_bit / seed={seed}. Drop-in replacement for PUB50-ciciot-random.",
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": experiments,
	}
	resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
	if resp.status_code in (200, 201):
		fid = resp.json().get("id")
		# Queue immediately
		r2 = requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=10)
		if r2.status_code == 200:
			created_ids.append(fid)
			if seed % 10 == 0 or seed == 1 or seed == N_FLOWS:
				print(f"  Queued [{seed:3d}/{N_FLOWS}]: flow id={fid} ({name})", flush=True)
		else:
			print(f"  ⚠ Restart failed for {fid}: {r2.status_code}")
	else:
		print(f"  ✗ Create failed for seed {seed}: {resp.status_code} {resp.text[:200]}")
	time.sleep(0.05)  # gentle on the dashboard

print(f"\nCreated + queued {len(created_ids)}/{N_FLOWS} flows. ID range: {min(created_ids)}-{max(created_ids)}")
