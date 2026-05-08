"""Commit n=112 fill on C35 (the camera-ready statistical baseline).

Plan:
  1. Pause CE10 r209/r210/r211 (queued — 3 flows). Frees worker to focus on C35.
  2. Resume C35 r213/r214 (currently paused — 2 flows).
  3. Queue 97 new C35 flows with RANDOM seeds (deterministic via master_seed=20260508
     so the seed set is reproducible).

Final tally after all run:
  9 done + 1 running + 3 queued (C35 r209-r211) + 2 resumed + 97 new = 112

Naming: WSWEEP-96b-C35-rNNNNN where N is a 5-digit random seed.

Seeds drawn from [1000, 99999], non-overlapping with existing 201-214.
Master RNG state pinned to master_seed=20260508 (today's date YYYYMMDD).

Per CLAUDE.md Rule 2: dashboard POST /api/flows.
"""

import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"

# C35 weights
WEIGHTS = {
	"fitness_weight_ce": 0.35,
	"fitness_weight_acc": 0.30,
	"fitness_weight_f1": 0.30,
	"fitness_weight_fpr": 0.05,
}

# Master seed for reproducibility — anyone can re-derive the exact seed set
MASTER_SEED = 20260508
N_NEW_FLOWS = 97
SEED_RANGE = (1000, 99999)
EXCLUDED_SEEDS = set(range(201, 215))  # avoid overlap with existing C35 flows

experiments = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def main():
	con = sqlite3.connect(str(DB_PATH))
	row = con.execute("SELECT config_json FROM flows WHERE id = 1812").fetchone()
	if not row:
		print("ERROR: r1812 base config not found.")
		sys.exit(1)
	base_params = dict(json.loads(row[0])["params"])

	# Step 1: pause CE10 r209/r210/r211
	print("=" * 60)
	print("STEP 1: Pause CE10 r209/r210/r211 (free worker for C35)")
	print("=" * 60)
	for fid in [2033, 2035, 2037]:
		r = requests.patch(f"{DASHBOARD}/api/flows/{fid}", json={"status": "pending"}, verify=False, timeout=15)
		row = con.execute("SELECT name FROM flows WHERE id = ?", (fid,)).fetchone()
		print(f"  PATCH {fid} ({row[0] if row else '?'}) → {r.status_code}")
		time.sleep(0.4)

	# Step 2: resume C35 r213/r214
	print()
	print("=" * 60)
	print("STEP 2: Resume C35 r213/r214 (paused → queued)")
	print("=" * 60)
	for fid in [2040, 2041]:
		r = requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		print(f"  POST {fid}/restart → {r.status_code}")
		time.sleep(0.4)

	# Step 3: generate random seeds
	rng = random.Random(MASTER_SEED)
	# Sample without replacement from valid range (excluding already-used)
	candidates = [s for s in range(SEED_RANGE[0], SEED_RANGE[1] + 1) if s not in EXCLUDED_SEEDS]
	new_seeds = rng.sample(candidates, N_NEW_FLOWS)
	print()
	print("=" * 60)
	print(f"STEP 3: Queue {N_NEW_FLOWS} new C35 flows with random seeds")
	print(f"        Master seed: {MASTER_SEED}  Range: {SEED_RANGE}")
	print(f"        First 10 seeds: {new_seeds[:10]}")
	print("=" * 60)

	created = []
	for i, seed in enumerate(new_seeds):
		params = dict(base_params)
		params["ids_n_bits"] = 96
		params["seed"] = seed
		params.update(WEIGHTS)
		name = f"WSWEEP-96b-C35-r{seed:05d}"
		body = {
			"name": name,
			"description": f"C35 PUB112 fill ({i+1}/{N_NEW_FLOWS}). seed={seed}.",
			"config": {"template": "ids-binary-2-phase", "params": params},
			"experiments": experiments,
		}
		try:
			resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
			if resp.status_code in (200, 201):
				fid = resp.json()["id"]
				created.append((fid, seed, name))
				if (i + 1) % 10 == 0:
					print(f"  ...{i+1}/{N_NEW_FLOWS} created (latest id={fid}, seed={seed})")
				time.sleep(0.3)
			else:
				print(f"  ✗ Failed for seed={seed}: {resp.status_code}")
				sys.exit(2)
		except Exception as e:
			print(f"  ✗ Exception for seed={seed}: {e}")
			sys.exit(3)

	# Flip pending → queued
	print()
	print(f"Flipping {len(created)} pending → queued...")
	for j, (fid, seed, name) in enumerate(created):
		requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		time.sleep(0.2)
		if (j + 1) % 20 == 0:
			print(f"  ...{j+1}/{len(created)} queued")

	# Verify
	queued_count = con.execute(
		"SELECT COUNT(*) FROM flows WHERE name LIKE 'WSWEEP-96b-C35-r%' AND status='queued'"
	).fetchone()[0]
	pending_count = con.execute(
		"SELECT COUNT(*) FROM flows WHERE name LIKE 'WSWEEP-96b-C35-r%' AND status='pending'"
	).fetchone()[0]
	completed_count = con.execute(
		"SELECT COUNT(*) FROM flows WHERE name LIKE 'WSWEEP-96b-C35-r%' AND status='completed'"
	).fetchone()[0]
	running_count = con.execute(
		"SELECT COUNT(*) FROM flows WHERE name LIKE 'WSWEEP-96b-C35-r%' AND status='running'"
	).fetchone()[0]

	print()
	print("=" * 60)
	print("FINAL C35 STATE")
	print("=" * 60)
	print(f"  Completed: {completed_count}")
	print(f"  Running:   {running_count}")
	print(f"  Queued:    {queued_count}")
	print(f"  Pending:   {pending_count}")
	print(f"  TOTAL:     {completed_count + running_count + queued_count + pending_count}")
	print(f"  → Target n=112 reached: {completed_count + running_count + queued_count + pending_count >= 112}")
	con.close()


if __name__ == "__main__":
	main()
