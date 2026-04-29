"""Queue 2 flows of per-feature adaptive thermometer (auto:128b cap) on
neto-subsample.

Set ids_n_bits="auto" + ids_auto_max_bits=128 — the encoder allocates
per-feature widths capped at 128 (highly-discriminative features can use
up to 128b, low-cardinality features like flag-bits stay at 4-8b). Total
input bits expected to be smaller than uniform 64b while extracting more
signal.

IMPORTANT: queued as STATUS='pending' (not 'queued'). The currently
running worker has the OLD imports of wnn.ids.ciciot2023 cached, which
predates the auto_max_bits plumbing change in this commit. These flows
must NOT be picked up until the worker is restarted with the new code.

Workflow (manual, after current bit-sweep completes):
  1. Wait for the bit-sweep flows to finish (~12h)
  2. Kill + restart the worker (so new ciciot2023 / worker.py imports load)
  3. Flip these 2 flows from pending -> queued (POST /api/flows/<id>/restart)
  4. They run with auto_max_bits=128 properly plumbed through

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

SEEDS = [301, 302]  # distinct from bit-sweep's 201/202

# Clone r1812 (the best-performing 64b flow) — same dataset, K-fold, weights.
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
	params["ids_n_bits"] = "auto"          # trigger per-feature adaptive width
	params["ids_auto_max_bits"] = 128      # cap (rationale in saved memory)
	params["seed"] = seed
	# Keep min_bits/max_bits at 4/34 (PUB50 defaults) — isolate encoding effect.
	name = f"BITSWEEP-neto-sub-auto128-r{seed}"
	body = {
		"name": name,
		"description": (
			f"Per-feature adaptive thermometer with 128b cap on neto-subsample (1.43M, 46f). "
			f"ids_n_bits=auto, ids_auto_max_bits=128, seed={seed}. Test whether "
			f"the auto encoder beats uniform 64b/72b by allocating bits where they "
			f"matter (discriminative features) and saving them where they don't "
			f"(low-cardinality flag bits). Saved memory note "
			f"project_post_submission_thermometer.md has the rationale."
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

# Verify experiments + final state.  Note: NOT calling /restart, so they stay pending.
print()
print("Final state (status=pending — worker will NOT pick these up):")
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
print(f"Done. {len(created)} auto:128b flows created in PENDING state.")
print()
print("To activate them later (after bit-sweep + worker restart):")
print("  for fid in", [c[0] for c in created], ":")
print("      curl -sk -X POST https://localhost:3000/api/flows/$fid/restart -H 'Content-Type: application/json' -d '{}'")
