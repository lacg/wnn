"""Weight sweep on 96b: 4 variants × 8 seeds = 32 flows, alternating order.

Baseline (CE10: 0.1/0.2/0.35/0.35) already has n=8 from stage-2 — anchor
for comparison, no new flows.

Variants tested:
  CE20  (0.20/0.10/0.30/0.40)  - historical PUB-baseline / cicids set
  F1H   (0.10/0.10/0.50/0.30)  - F1-heavy: chase ceiling above 91.85
  FPRH  (0.10/0.10/0.30/0.50)  - FPR-heavy: chase deployable below 4.40%
  CE40  (0.40/0.00/0.30/0.30)  - CE-dominant, zero-acc (PUB-top20-CE4F3R3 set)

Alternating order: by 4 flows in, n=1 of each; by 8 flows, n=2; etc.
User can PAUSE a variant (set status back to pending) if it flatlines.

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

# (label, ce, acc, f1, fpr) — ordered for alternation
VARIANTS = [
	("CE20",  0.20, 0.10, 0.30, 0.40),
	("F1H",   0.10, 0.10, 0.50, 0.30),
	("FPRH",  0.10, 0.10, 0.30, 0.50),
	("CE40",  0.40, 0.00, 0.30, 0.30),
]
SEEDS = list(range(201, 209))  # 8 seeds matching top-4 stage-2

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

# Create order: lowest id first, so id-DESC produces:
#   v[0] r208 (runs first), v[1] r208, v[2] r208, v[3] r208, v[0] r207, ...
# So creation iterates: seed ascending, variant DESCending within seed.
create_order = []
for seed in SEEDS:                         # 201..208
	for variant in reversed(VARIANTS):     # CE40, FPRH, F1H, CE20 — so CE20 ends up highest id
		create_order.append((variant, seed))

created = []
for (label, ce, acc, f1, fpr), seed in create_order:
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
			f"Weight sweep on 96b. variant={label} "
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
print("Final state (id-DESC = run order):")
con = sqlite3.connect(str(DB_PATH))
for fid, label, seed, name in sorted(created, key=lambda x: -x[0]):
	r = con.execute("SELECT id, name, status FROM flows WHERE id = ?", (fid,)).fetchone()
	exps = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,)).fetchone()[0]
	flag = "✓" if exps == 2 and r[2] in ("queued", "running") else "✗"
	print(f"  {flag} id={r[0]}  status={r[2]:<10}  experiments={exps}  {r[1]}")
con.close()
