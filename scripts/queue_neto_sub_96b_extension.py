"""Add 96b × 2 seeds — confirm/refute the 92b inflection.

92b dropped to 89.30±0.39 from 88b's 91.49±0.22. Both 92b seeds agreed
(89.22 / 89.61), so it's not seed roulette. But 76b also dipped to 89.30
(same exact mean!) — could be a 'bumpy curve' pattern rather than clean
saturation past 88b.

96b is the disambiguator:
  if 96b ≥ 90.5%  →  curve is bumpy; bit-width matters more than 'larger
                     is better'; 76b and 92b were unlucky local minima.
  if 96b < 90.5%  →  smooth saturation past 88b; 92b's drop was the real
                     plateau end.

Cost: 2 flows × ~92m ≈ 3h.

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

SEEDS = [201, 202]
N_BITS = 96

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
for seed in SEEDS:
	params = dict(base_params)
	params["ids_n_bits"] = N_BITS
	params["seed"] = seed
	name = f"BITSWEEP-neto-sub-{N_BITS:02d}b-r{seed}"
	body = {
		"name": name,
		"description": (
			f"Disambiguator for 92b inflection. ids_n_bits={N_BITS}, seed={seed}. "
			f"If 96b ≥ 90.5% the curve is bumpy (76b and 92b were unlucky local "
			f"minima); if 96b < 90.5% smooth saturation past 88b."
		),
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": experiments,
	}
	print(f"Creating: {name}...")
	resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
	if resp.status_code in (200, 201):
		fid = resp.json()["id"]
		created.append((fid, seed, name))
		print(f"  ✓ id={fid}")
	else:
		print(f"  ✗ Failed ({resp.status_code}): {resp.text[:300]}")
		sys.exit(2)

print()
print("Flipping to status='queued'...")
for fid, seed, name in created:
	resp = requests.post(
		f"{DASHBOARD}/api/flows/{fid}/restart",
		json={}, verify=False, timeout=15,
	)
	status_str = "queued" if resp.status_code in (200, 201) else f"failed ({resp.status_code})"
	print(f"  {fid:>5}  ({N_BITS}b r{seed}) -> {status_str}")

print()
print("Final state:")
con = sqlite3.connect(str(DB_PATH))
for fid, *_ in sorted(created, key=lambda x: -x[0]):
	r = con.execute(
		"SELECT id, name, status FROM flows WHERE id = ?", (fid,),
	).fetchone()
	exps = con.execute(
		"SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,),
	).fetchone()[0]
	flag = "✓" if exps == 2 else "✗"
	print(f"  {flag} id={r[0]}  status={r[2]:<10}  experiments={exps}  {r[1]}")
con.close()
