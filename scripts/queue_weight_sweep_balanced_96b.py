"""Balanced-low-F1=FPR sweep on 96b — 6 variants × 8 seeds = 48 flows.

Tests hypothesis: low-but-balanced F1=FPR (0.05 / 0.10 / 0.15) with
CE+Acc carrying the bulk of the fitness pressure. Two ratios per level:
  CE-dominant (3:1 CE:Acc) and Acc-dominant (1:3 CE:Acc).

CE-dominant tests "calibration drives sub-5%" cleanly.
Acc-dominant tests whether Acc-as-F1-proxy can also surface sub-5%.

All 6 variants × 8 seeds = 48 flows ≈ 73h. Plan: prune as we go from
n=2-3 to drop variants that aren't producing sub-5% signal.

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

# (label, ce, acc, f1, fpr) — declared order is run-order priority within each seed.
# Worker pulls id DESC, and we create in REVERSED order, so first listed = runs first.
# (B05-CE has the most extreme test of the calibration hypothesis → runs first)
VARIANTS = [
	("B05-CE", 0.675, 0.225, 0.05, 0.05),
	("B05-AC", 0.225, 0.675, 0.05, 0.05),
	("B10-CE", 0.60,  0.20,  0.10, 0.10),
	("B10-AC", 0.20,  0.60,  0.10, 0.10),
	("B15-CE", 0.525, 0.175, 0.15, 0.15),
	("B15-AC", 0.175, 0.525, 0.15, 0.15),
]
SEEDS = list(range(201, 209))  # 8 seeds

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

# Create order: lowest id first per seed → highest id (B05-CE) runs first.
# Iterate seeds ascending, variants in REVERSE declared order within each seed.
create_order = []
for seed in SEEDS:
	for variant in reversed(VARIANTS):
		create_order.append((variant, seed))

created = []
for (label, ce, acc, f1, fpr), seed in create_order:
	if abs((ce + acc + f1 + fpr) - 1.0) > 1e-9:
		print(f"ERROR: weights for {label} sum to {ce+acc+f1+fpr}, not 1.0")
		sys.exit(3)
	params = dict(base_params)
	params["ids_n_bits"] = 96
	params["seed"] = seed
	params["fitness_weight_ce"] = ce
	params["fitness_weight_acc"] = acc
	params["fitness_weight_f1"] = f1
	params["fitness_weight_fpr"] = fpr
	name = f"WSWEEP-96b-{label}-r{seed}"
	body = {
		"name": name,
		"description": (
			f"Balanced low-F1=FPR sweep. variant={label} "
			f"(ce={ce}, acc={acc}, f1={f1}, fpr={fpr}), seed={seed}."
		),
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": experiments,
	}
	print(f"Creating: {name}...")
	resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
	if resp.status_code in (200, 201):
		fid = resp.json()["id"]
		created.append((fid, label, seed, name))
		print(f"  ✓ id={fid}")
		time.sleep(1)
	else:
		print(f"  ✗ Failed ({resp.status_code}): {resp.text[:200]}")
		sys.exit(2)

print()
print("Flipping pending → queued...")
for fid, label, seed, name in created:
	requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
	time.sleep(0.8)

print()
print("First 12 flows in id-DESC = run order:")
con = sqlite3.connect(str(DB_PATH))
for fid, label, seed, name in sorted(created, key=lambda x: -x[0])[:12]:
	r = con.execute("SELECT id, name, status FROM flows WHERE id = ?", (fid,)).fetchone()
	exps = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,)).fetchone()[0]
	flag = "✓" if exps == 2 and r[2] in ("queued", "running") else "✗"
	print(f"  {flag} id={r[0]}  status={r[2]:<10}  experiments={exps}  {r[1]}")
con.close()
