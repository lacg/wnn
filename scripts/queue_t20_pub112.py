"""Queue 112 fresh 250n×100b flows on neto-subsample with canonical TOP20.

Replaces the TOP19-tagged cohort (43 done + 69 paused). New cohort uses the
canonical TOP20 list re-derived from neto-full on 13/05/2026
(scripts/derive_top20_neto_full.py).

Naming: WSWEEP-T20-96b-C35-250n100b-r{SEED}
- T20 prefix differentiates from the obsolete WSWEEP-96b-C35-250n100b-r* cohort
- Same C35 fitness weights, same 250n×100b architecture ceiling, same seed
  methodology (random seeds from master_seed=20260513).

Per CLAUDE.md Rule 2: dashboard POST /api/flows (with experiments array).
"""

import argparse
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
SOURCE_FLOW = "WSWEEP-96b-C35-250n100b-r94805"  # template (any TOP19 flow's config works as a base)

# C35 weights (CE-leading)
WEIGHTS = {
	"fitness_weight_ce": 0.35,
	"fitness_weight_acc": 0.30,
	"fitness_weight_f1": 0.30,
	"fitness_weight_fpr": 0.05,
}

MASTER_SEED = 20260513  # today's date YYYYMMDD; reproducibility anchor
DEFAULT_N_FLOWS = 112
SEED_RANGE = (1000, 99999)
# Exclude seeds already used in the TOP19 cohort to avoid name collisions
EXCLUDED_SEEDS_SQL = "SELECT name FROM flows WHERE name LIKE 'WSWEEP-96b-C35-250n100b-r%'"

experiments = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--features", default="top20",
	                    choices=["top20", "top20_mi8b", "top20_mi96b"],
	                    help="Feature selection (default top20 RF; mi96b for ablation)")
	parser.add_argument("--n-flows", type=int, default=DEFAULT_N_FLOWS,
	                    help=f"Number of flows to queue (default {DEFAULT_N_FLOWS})")
	parser.add_argument("--name-suffix", default="",
	                    help="Append suffix to flow name (e.g., '-MI96b' to mark ablation)")
	args = parser.parse_args()

	con = sqlite3.connect(str(DB_PATH))
	row = con.execute("SELECT config_json FROM flows WHERE name = ?", (SOURCE_FLOW,)).fetchone()
	if not row:
		print(f"ERROR: source flow {SOURCE_FLOW} not found.")
		sys.exit(1)
	base_params = dict(json.loads(row[0])["params"])
	# Pin to canonical settings; feature_selection overridable via CLI
	base_params["ids_dataset"] = "ciciot2023_neto_subsample"
	base_params["ids_feature_selection"] = args.features
	base_params["ids_n_bits"] = 96
	base_params.update(WEIGHTS)

	# Excluded seeds = any seed already used in a 250n×100b flow (TOP19 cohort)
	excluded = set()
	for (name,) in con.execute(EXCLUDED_SEEDS_SQL).fetchall():
		# name format: WSWEEP-96b-C35-250n100b-r{SEED}
		try:
			seed = int(name.rsplit("-r", 1)[1])
			excluded.add(seed)
		except (ValueError, IndexError):
			pass
	print(f"Excluded {len(excluded)} seeds from prior TOP19 cohort.")

	rng = random.Random(MASTER_SEED + (hash(args.features) & 0xffff))  # different seeds per feature selection
	candidates = [s for s in range(SEED_RANGE[0], SEED_RANGE[1] + 1) if s not in excluded]
	new_seeds = rng.sample(candidates, args.n_flows)
	print(f"Drew {len(new_seeds)} new seeds (master_seed={MASTER_SEED}, features={args.features}).")
	print(f"  First 10: {new_seeds[:10]}")
	print()

	created = []
	for i, seed in enumerate(new_seeds):
		params = dict(base_params)
		params["seed"] = seed
		name = f"WSWEEP-T20-96b-C35-250n100b{args.name_suffix}-r{seed}"
		body = {
			"name": name,
			"description": f"C35 250n×100b PUB{args.n_flows} with features={args.features}. seed={seed}. master_seed={MASTER_SEED}. ({i+1}/{args.n_flows}).",
			"config": {"template": "ids-binary-2-phase", "params": params},
			"experiments": experiments,
		}
		try:
			resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
			if resp.status_code in (200, 201):
				fid = resp.json()["id"]
				created.append((fid, seed, name))
				if (i + 1) % 10 == 0:
					print(f"  ...{i+1}/{args.n_flows} created (latest id={fid}, seed={seed})")
				time.sleep(0.3)
			else:
				print(f"  ✗ Failed for seed={seed}: {resp.status_code}  body={resp.text[:200]}")
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
	print(f"  ...{len(created)}/{len(created)} queued")

	# Verify
	print()
	print("=" * 60)
	print("T20 COHORT STATE")
	print("=" * 60)
	c = con.execute("SELECT status, COUNT(*) FROM flows WHERE name LIKE 'WSWEEP-T20-96b-C35-250n100b-r%' GROUP BY status").fetchall()
	for status, count in c:
		print(f"  {status}: {count}")


if __name__ == "__main__":
	main()
