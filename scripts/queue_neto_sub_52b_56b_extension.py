"""Add 52b and 56b × 2 seeds each to the in-progress neto-subsample bit sweep.

Combined with the 60b/68b queued earlier, gives us the full 32→72 curve at
8-bit resolution. Specifically tests whether the 64-72b near-tie at
~91% F1 is part of a plateau that extends down to ~52-56b, or whether
the curve climbs steeply between 32b and 60b.

Cost: 4 flows × ~90min ≈ 6h. Queued (status=queued) — under ORDER BY id
DESC, these run BEFORE the still-pending 60b/68b flows since they get
the higher ids. Per the overnight context this is fine — overall queue
runs sequentially regardless of order; user can flip priorities on
wake-up if needed.

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
BIT_WIDTHS = [52, 56]  # interpolate between 32b/48b/60b

# Clone r1812 (best-performing 64b reference) — same dataset, K-fold,
# weights, hyperparameters; only ids_n_bits + seed change.
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
		# Keep min_bits/max_bits at PUB50 defaults (4, 34) — same as the rest
		# of the bit sweep, isolates the encoding effect from the GA bound effect.
		name = f"BITSWEEP-neto-sub-{n_bits:02d}b-r{seed}"
		body = {
			"name": name,
			"description": (
				f"Bit-width interpolation between 48b/60b on neto-subsample "
				f"(1.43M, 46f). ids_n_bits={n_bits}, seed={seed}. Together "
				f"with 60b/68b runs, traces the full 32→72b curve at 8-bit "
				f"resolution to see whether the 64-72b plateau extends downward."
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

# Flip pending → queued
print()
print("Flipping to status='queued'...")
for fid, n_bits, seed, name in created:
	resp = requests.post(
		f"{DASHBOARD}/api/flows/{fid}/restart",
		json={}, verify=False, timeout=15,
	)
	status_str = "queued" if resp.status_code in (200, 201) else f"failed ({resp.status_code})"
	print(f"  {fid:>5}  ({n_bits:>2}b r{seed}) -> {status_str}")

# Verify experiments + final state.
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
