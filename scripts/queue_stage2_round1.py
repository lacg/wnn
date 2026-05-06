"""Stage 2 round 1: F2R + F35 + CE20-revised + CE20-balanced × 1 seed each = 8 flows.

Single-seed initial sweep to identify which configs are worth extending.
Run order priority (id-DESC): most-likely-to-succeed first.

Variants (all sum to 1.00):
  CE20rev      0.20/0.20/0.40/0.20  - hybrid (f1=2*fpr=2*ce, balanced CE/Acc)
  F35-CE30     0.30/0.20/0.35/0.15  - CE10's F1 + bumped CE + halved FPR
  F2R-high-eq  0.275/0.275/0.30/0.15 - f1=2*fpr, near-CE10 territory
  F35-CE35     0.35/0.20/0.35/0.10  - More CE, lower FPR
  CE20bal      0.20/0.15/0.325/0.325 - CE20 with F1=FPR balance
  F2R-mid-eq   0.35/0.35/0.20/0.10  - f1=2*fpr, mid level
  F35-CE25-A25 0.25/0.25/0.35/0.15  - Balanced CE+Acc, F1=0.35
  F2R-low-eq   0.425/0.425/0.10/0.05 - f1=2*fpr, weak F1 (likely fail)

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

# Declared in REVERSE run-order: lowest id created first → runs last.
# Worker pulls id-DESC, so the LAST item in this list will run FIRST.
VARIANTS_REVERSED = [
	("F2R-low-eq",   0.425, 0.425, 0.10,  0.05),   # runs last (likely fail)
	("F35-CE25-A25", 0.25,  0.25,  0.35,  0.15),
	("F2R-mid-eq",   0.35,  0.35,  0.20,  0.10),
	("CE20bal",      0.20,  0.15,  0.325, 0.325),
	("F35-CE35",     0.35,  0.20,  0.35,  0.10),
	("F2R-high-eq",  0.275, 0.275, 0.30,  0.15),
	("F35-CE30",     0.30,  0.20,  0.35,  0.15),
	("CE20rev",      0.20,  0.20,  0.40,  0.20),   # runs first (most promising)
]
SEED = 208  # single seed for round 1

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
for label, ce, acc, f1, fpr in VARIANTS_REVERSED:
	if abs((ce + acc + f1 + fpr) - 1.0) > 1e-9:
		print(f"ERROR: weights for {label} sum to {ce+acc+f1+fpr}, not 1.0")
		sys.exit(3)
	params = dict(base_params)
	params["ids_n_bits"] = 96
	params["seed"] = SEED
	params["fitness_weight_ce"] = ce
	params["fitness_weight_acc"] = acc
	params["fitness_weight_f1"] = f1
	params["fitness_weight_fpr"] = fpr
	name = f"WSWEEP-96b-{label}-r{SEED}"
	body = {
		"name": name,
		"description": (
			f"Stage 2 round 1. variant={label} "
			f"(ce={ce}, acc={acc}, f1={f1}, fpr={fpr}), seed={SEED}."
		),
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": experiments,
	}
	print(f"Creating: {name}...")
	resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
	if resp.status_code in (200, 201):
		fid = resp.json()["id"]
		created.append((fid, label, name))
		print(f"  ✓ id={fid}")
		time.sleep(1)
	else:
		print(f"  ✗ Failed ({resp.status_code}): {resp.text[:200]}")
		sys.exit(2)

print()
print("Flipping pending → queued...")
for fid, label, name in created:
	requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
	time.sleep(0.8)

print()
print("Final state (id-DESC = run order):")
con = sqlite3.connect(str(DB_PATH))
for fid, label, name in sorted(created, key=lambda x: -x[0]):
	r = con.execute("SELECT id, name, status FROM flows WHERE id = ?", (fid,)).fetchone()
	exps = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,)).fetchone()[0]
	flag = "✓" if exps == 2 and r[2] in ("queued", "running") else "✗"
	print(f"  {flag} id={r[0]}  status={r[2]:<10}  experiments={exps}  {r[1]}")
con.close()
