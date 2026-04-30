"""Add 76b and 80b × 2 seeds to confirm the upper boundary of the plateau.

Current 2-seed coverage shows 48-72b in a tight ~90% F1 plateau. The
question 76b/80b answers: does the plateau extend further, or does the
encoder finally over-resolve at 76+? Either answer is a clean paper
result.

After these complete, the next stage is: pick top-3 bit-widths by mean
train_cal F1 and run 5 more seeds (seeds 203-207) at each, for tight
variance estimates on the final paper-headline candidates.

Cost: 4 flows × ~90min ≈ 6h.

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
BIT_WIDTHS = [76, 80]

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
for n_bits in BIT_WIDTHS:
	for seed in SEEDS:
		params = dict(base_params)
		params["ids_n_bits"] = n_bits
		params["seed"] = seed
		name = f"BITSWEEP-neto-sub-{n_bits:02d}b-r{seed}"
		body = {
			"name": name,
			"description": (
				f"Upper-bound check on the bit-width plateau at neto-subsample "
				f"(1.43M, 46f). ids_n_bits={n_bits}, seed={seed}. "
				f"Confirms whether 48-72b plateau extends to 76-80b or the "
				f"encoder finally over-resolves at 76+."
			),
			"config": {"template": "ids-binary-2-phase", "params": params},
			"experiments": experiments,
		}
		print(f"Creating: {name}...")
		resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
		if resp.status_code in (200, 201):
			fid = resp.json().get("id")
			created.append((fid, n_bits, seed, name))
			print(f"  ✓ id={fid}")
		else:
			print(f"  ✗ Failed ({resp.status_code}): {resp.text[:300]}")
			sys.exit(2)

print()
print("Flipping to status='queued'...")
for fid, n_bits, seed, name in created:
	resp = requests.post(
		f"{DASHBOARD}/api/flows/{fid}/restart",
		json={}, verify=False, timeout=15,
	)
	status_str = "queued" if resp.status_code in (200, 201) else f"failed ({resp.status_code})"
	print(f"  {fid:>5}  ({n_bits:>2}b r{seed}) -> {status_str}")

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
print(f"Done. {len(created)} bit-sweep flows queued.")
