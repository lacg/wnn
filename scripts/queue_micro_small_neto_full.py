"""Queue Micro + Small rerun flows on canonical neto-full or neto-subsample.

Usage:
    # Default: 46M neto-full, therm=8, name prefix EVAL46MNETO
    python queue_micro_small_neto_full.py
    # 1.43M neto-subsample
    python queue_micro_small_neto_full.py --dataset neto_subsample
    # therm=96 ablation
    python queue_micro_small_neto_full.py --therm 96


Background: Table 5's Micro/Small/Peak/Search WNN rows were originally run on
`ciciot2023_full` (bencorn's 38.5M lossy reorg). The baselines in the same
table were run on `ciciot2023_neto_full` (canonical 46.7M Neto). This script
re-runs just the Micro + Small architectures on neto-full to fix the
apples-to-oranges comparison. Peak + Search will be replaced by 250n×100b
architecture results separately.

Architectures (same as the paper, 8-bit thermometer for all):
  Micro:
    - 5n × 4b   : seeds 1..10 (paper reports mean±std over 10 seeds)
    - 100n × 4b : seed 1
    - 200n × 4b : seed 1  ← "800-byte detector" headline
    - 300n × 4b : seed 1
    - 500n × 4b : seed 1
  Small:
    - 5n × 12b  : seed 1
    - 100n × 8b : seed 1
    - 300n × 8b : seed 1
    - 400n × 8b : seed 1  ← "25 KB / 83.13% F1" headline

Total: 14 single-genome flows (10 + 4 + 4). They go to the BACK of the worker
queue and will run after the current 250n×100b backlog finishes.

Naming: EVAL46MNETO-<arch>-s<seed>
Per CLAUDE.md Rule 2: dashboard POST /api/flows.
"""

import argparse
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
SOURCE_FLOW = "EVAL46M-46M-5n4b-s1-SWEEP46M-MICRO"  # template to copy params from


# (n, b, [seeds], tag) — tag is for the flow name
ARCHS = [
	# Micro (5n×4b is the only multi-seed; the rest are single seed)
	(5, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Micro"),
	(100, 4, [1], "Micro"),
	(200, 4, [1], "Micro"),
	(300, 4, [1], "Micro"),
	(500, 4, [1], "Micro"),
	# Small
	(5, 12, [1], "Small"),
	(100, 8, [1], "Small"),
	(300, 8, [1], "Small"),
	(400, 8, [1], "Small"),
]

# Single eval flow → single experiment (just grid_search, no GA refinement needed
# since these are eval-only with locked architecture)
experiments = [
	{"name": "Grid Search (locked arch)", "experiment_type": "grid_search", "phase_type": "grid_search"},
]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--therm", type=int, default=8,
	                    help="Thermometer bit-width per feature (default 8; 96 for ablation)")
	parser.add_argument("--dataset", default="neto_full", choices=["neto_full", "neto_subsample"],
	                    help="Dataset variant: neto_full (46M) or neto_subsample (1.43M)")
	parser.add_argument("--features", default="top20",
	                    choices=["top20", "top20_mi8b", "top20_mi96b"],
	                    help="Feature selection: top20 (RF importance) or quantization-aware MI variants")
	args = parser.parse_args()
	therm = args.therm
	ds_key = "ciciot2023_" + args.dataset
	name_prefix = "EVAL46MNETO" if args.dataset == "neto_full" else "EVAL13MNETO"
	if args.features != "top20":
		name_prefix += "-" + args.features.replace("top20_", "").upper()

	con = sqlite3.connect(str(DB_PATH))
	row = con.execute("SELECT config_json FROM flows WHERE name = ?", (SOURCE_FLOW,)).fetchone()
	if not row:
		print(f"ERROR: source flow {SOURCE_FLOW} not found.")
		sys.exit(1)
	base_cfg = json.loads(row[0])
	base_params = dict(base_cfg["params"])

	# Patch base_params for chosen dataset + feature selection + thermometer width
	base_params["ids_dataset"] = ds_key
	base_params["ids_feature_selection"] = args.features
	base_params["ids_n_bits"] = therm

	# Name suffix differentiates therm=8 (default, no suffix) vs other widths (-t{therm}b)
	therm_suffix = "" if therm == 8 else f"-t{therm}b"

	print("=" * 70)
	print(f"QUEUE: Micro + Small rerun on ciciot2023_neto_full (canonical 46.7M)")
	print(f"Thermometer width: {therm} bits per feature  |  Template: {SOURCE_FLOW}")
	print(f"Total flows to queue: {sum(len(s) for _, _, s, _ in ARCHS)}")
	print("=" * 70)
	print()

	# Generate (architecture, seed) pairs in order
	pairs = []
	for n, b, seeds, tag in ARCHS:
		for s in seeds:
			pairs.append((n, b, s, tag))

	created = []
	for i, (n, b, seed, tag) in enumerate(pairs):
		params = dict(base_params)
		params["neurons_grid"] = [n]
		params["bits_grid"] = [b]
		params["min_neurons"] = n
		params["max_neurons"] = n
		params["min_bits"] = b
		params["max_bits"] = b
		params["seed"] = seed
		params["population_size"] = 1
		params["ga_generations"] = 1

		name = f"{name_prefix}-{n}n{b}b-s{seed}-{tag.upper()}{therm_suffix}"
		body = {
			"name": name,
			"description": f"Table 5 {tag} on canonical {ds_key} (TOP20 post-13/05/2026). arch={n}n×{b}b, seed={seed}, therm={therm}b.",
			"config": {"template": "ids-binary-2-phase", "params": params},
			"experiments": experiments,
		}
		try:
			resp = requests.post(
				f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30,
			)
			if resp.status_code in (200, 201):
				fid = resp.json()["id"]
				created.append((fid, name))
				print(f"  [{i+1:2d}/{len(pairs)}] created id={fid:<5} {name}")
				time.sleep(0.3)
			else:
				print(f"  ✗ Failed for {name}: {resp.status_code}  body={resp.text[:200]}")
				sys.exit(2)
		except Exception as e:
			print(f"  ✗ Exception for {name}: {e}")
			sys.exit(3)

	# Flip pending → queued
	print()
	print(f"Flipping {len(created)} pending → queued...")
	for j, (fid, name) in enumerate(created):
		requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		time.sleep(0.2)
		if (j + 1) % 5 == 0:
			print(f"  ...{j+1}/{len(created)} queued")
	print(f"  ...{len(created)}/{len(created)} queued")

	# Verify
	print()
	print("=" * 70)
	print("STATE AFTER QUEUING")
	print("=" * 70)
	q = con.execute(
		"SELECT COUNT(*) FROM flows WHERE name LIKE 'EVAL46MNETO-%' AND status='queued'"
	).fetchone()[0]
	p = con.execute(
		"SELECT COUNT(*) FROM flows WHERE name LIKE 'EVAL46MNETO-%' AND status='pending'"
	).fetchone()[0]
	print(f"  queued : {q}")
	print(f"  pending: {p}")
	print()
	print("Worker will pick these up after the current 250n×100b backlog finishes.")
	con.close()


if __name__ == "__main__":
	main()
