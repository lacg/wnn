"""Queue the neto-subsample bit-width sanity sweep.

12 flows total (2× each of [4, 16, 24, 32, 48, 64] thermometer bit-widths)
to test whether `ids_n_bits=8` (the current PUB50 default) is leaving F1
on the table OR if the train_cal/val_cal divergence we see at 8b is a
score-resolution artifact that goes away with more bits.

Creation order is **lowest bits → highest bits**, because the dashboard
worker's poll uses `ORDER BY id DESC`. With 4b having the lowest id and
64b the highest, the worker dequeues 64b first → 32b → 16b → 24b → 48b
→ 4b — exactly the "highest bits first" run order we want.

Status = `queued` from the start, so these jump in front of the remaining
~99 PUB50 flows (lower ids 1690-1794). The currently-running PUB50 flow
finishes naturally; right after, the worker pulls 64b r1 (highest id).

`min_bits=4, max_bits=34` are kept at PUB50 defaults — isolating the
*encoding* effect from the *GA-search-bound* effect for this probe.

Per CLAUDE.md Rule 2: dashboard POST /api/flows (never direct SQL).
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

# Creation order: 4 → 48 → 24 → 16 → 32 → 64 (lowest id first; 64 ends up highest)
BIT_SEQUENCE = [4, 48, 24, 16, 32, 64]
SEEDS_PER_BITS = [201, 202]  # 2 flows per bit setting

# Clone r112 (1801) — most recent completed neto-subsample flow with all params we want.
con = sqlite3.connect(str(DB_PATH))
row = con.execute("SELECT config_json FROM flows WHERE id = 1801").fetchone()
con.close()
if not row:
	print("ERROR: r112 (id=1801) not found in DB — cannot clone params.")
	sys.exit(1)

base_params = dict(json.loads(row[0])["params"])

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

created = []
for n_bits in BIT_SEQUENCE:
	for seed in SEEDS_PER_BITS:
		params = dict(base_params)
		params["ids_n_bits"] = n_bits
		params["seed"] = seed
		# Keep min_bits/max_bits at PUB50 defaults (4, 34) so we isolate the
		# encoding effect, not the GA bound effect. If 64b looks promising,
		# rerun with widened bounds in a follow-up batch.
		name = f"BITSWEEP-neto-sub-{n_bits:02d}b-r{seed}"
		body = {
			"name": name,
			"description": (
				f"Bit-width sanity sweep on neto-subsample (1.43M, 46f). "
				f"ids_n_bits={n_bits}, seed={seed}. Test whether the "
				f"train_cal/val_cal FPR divergence at 8b is a "
				f"score-resolution artifact."
			),
			"config": {"template": "ids-binary-2-phase", "params": params},
			"experiments": experiments,
		}
		print(f"Creating: {name}...")
		resp = requests.post(
			f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30,
		)
		if resp.status_code in (200, 201):
			flow = resp.json()
			fid = flow.get("id") if isinstance(flow, dict) else None
			created.append((fid, n_bits, seed, name))
			print(f"  ✓ id={fid}")
		else:
			print(f"  ✗ Failed ({resp.status_code}): {resp.text[:300]}")
			sys.exit(2)

# All flows land as "pending" by default. Flip to "queued" via /restart so the
# worker actually picks them up. We use /restart (not direct SQL) to follow
# CLAUDE.md Rule 2.
print()
print("Flipping to status='queued' so the worker picks them up...")
for fid, n_bits, seed, name in created:
	resp = requests.post(
		f"{DASHBOARD}/api/flows/{fid}/restart",
		json={}, verify=False, timeout=15,
	)
	if resp.status_code in (200, 201):
		print(f"  ✓ id={fid} ({n_bits}b r{seed}) → queued")
	else:
		print(f"  ✗ id={fid}: {resp.status_code} {resp.text[:200]}")

# Verify ordering
print()
print("Final state (highest id runs first via ORDER BY id DESC):")
con = sqlite3.connect(str(DB_PATH))
for fid, *_ in sorted(created, key=lambda x: -x[0]):
	r = con.execute(
		"SELECT id, name, status FROM flows WHERE id = ?", (fid,),
	).fetchone()
	if r:
		print(f"  id={r[0]}  status={r[2]:<10}  {r[1]}")
con.close()

print(f"\nDone. {len(created)} bit-sweep flows queued.")
print("Run order will be: 64b → 32b → 16b → 24b → 48b → 4b (highest bits first).")
