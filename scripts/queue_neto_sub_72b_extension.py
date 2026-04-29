"""Add 72b × 2 flows to the in-progress neto-subsample bit sweep.

The 64b results showed real promise (r1812 hit 89.09% F1 at 5.07% FPR
deployable on best_fitness fixed_05) so we want to see if 72b extends the
trend or shows diminishing returns.

Cost: ~1.7h per flow vs 64b's ~1.5h (12.5% more input bits → 12.5% more
training time, roughly). 2 seeds = ~3.4h total — runs after the currently
in-flight 32b r201 and ahead of 16b/24b/48b/4b in the existing queue
(higher id → runs first under ORDER BY id DESC).

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

SEEDS = [201, 202]  # same seed pattern as the existing bit-sweep flows
N_BITS = 72

# Clone r1812 (id 1812 — 64b r201, the best-performing 64b flow we have so far).
con = sqlite3.connect(str(DB_PATH))
row = con.execute("SELECT config_json FROM flows WHERE id = 1812").fetchone()
con.close()
if not row:
	print("ERROR: r1812 (64b r201) not found.")
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
	# Keep min_bits/max_bits at 4/34 — same as the rest of the bit sweep,
	# isolates the encoding effect from the GA bound effect.
	name = f"BITSWEEP-neto-sub-{N_BITS:02d}b-r{seed}"
	body = {
		"name": name,
		"description": (
			f"UV/IR extension of bit-width sanity sweep on neto-subsample (1.43M, 46f). "
			f"ids_n_bits={N_BITS}, seed={seed}. Test whether the 64b gains "
			f"(r1812: 89.09% F1 / 5.07% FPR deployable) extend at 72b or hit "
			f"diminishing returns."
		),
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": experiments,
	}
	print(f"Creating: {name}...")
	resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
	if resp.status_code in (200, 201):
		fid = resp.json().get("id")
		created.append((fid, seed, name))
		print(f"  ✓ id={fid}")
	else:
		print(f"  ✗ Failed ({resp.status_code}): {resp.text[:300]}")
		sys.exit(2)

# Flip to queued via /restart (per CLAUDE.md Rule 2 — never SQL).
print()
print("Flipping to status='queued'...")
for fid, seed, name in created:
	resp = requests.post(
		f"{DASHBOARD}/api/flows/{fid}/restart",
		json={}, verify=False, timeout=15,
	)
	if resp.status_code in (200, 201):
		print(f"  ✓ id={fid} ({N_BITS}b r{seed}) → queued")
	else:
		print(f"  ✗ id={fid}: {resp.status_code} {resp.text[:200]}")

# Verify experiments + run order
print()
print("Final state (highest id runs first under ORDER BY id DESC):")
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

print()
print(f"Done. {len(created)} 72b flows queued.")
print("Run order after current 32b r201 finishes: 72b r202 → 72b r201 → 16b r202 → ...")
