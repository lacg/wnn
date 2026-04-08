"""
Create a batch of PUB50-style ciciot flows with 2-bit thermometer encoding.

Mirrors the exact config of the existing 8-bit PUB50 ciciot batch (grid
search + GA neurons, 2-phase template, balanced fitness weights, single
cluster, K-fold 5, 250 GA generations, patience 5) with the ONLY change
being ids_n_bits=2 instead of 8.

Default: creates 10 flows as a statistical validation test. If the 10-run
batch confirms the 2-bit lift observed in the 46M sweep, re-run with
--count 112 --start-seed 11 to complete a full PUB50-matching batch
(112 total flows, 10+102, preserving the test seeds).

Usage:
    python scripts/create_pub50_2b_batch.py                   # 10 flows, seeds 1-10
    python scripts/create_pub50_2b_batch.py --count 10        # same, explicit
    python scripts/create_pub50_2b_batch.py --count 102 --start-seed 11  # fill to 112
    python scripts/create_pub50_2b_batch.py --dry-run         # print, don't POST
"""

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request


def build_flow_request(seed: int, thermometer_bits: int = 2, name_prefix: str = "PUB50-2b-ciciot-random") -> dict:
	"""Build the POST body for a PUB50-style flow with configurable thermometer width.

	Matches the exact parameters of the existing 8-bit PUB50 ciciot batch
	(see scripts/watch_fitness_sweep.sh and flow ID 711+ in the database).
	Only ids_n_bits differs from the 8-bit baseline.
	"""
	return {
		"name": f"{name_prefix}-r{seed:03d}",
		"description": (
			f"PUB50-style statistical batch with {thermometer_bits}-bit thermometer encoding "
			f"(vs 8-bit default). Same GA + grid search pipeline as the existing PUB50 ciciot "
			f"batch; seed {seed}. Created to statistically validate the 2-bit thermometer "
			f"finding observed in the 46M THERMO46MLOW sweep on 2026-04-08."
		),
		"config": {
			"template": "ids-binary-2-phase",
			"params": {
				# Dataset
				"ids_dataset": "ciciot2023",
				"ids_split": "random",
				"ids_n_bits": thermometer_bits,     # THE CHANGE: 8 → 2
				"ids_feature_selection": "top20",
				"ids_classification": "binary",

				# Architecture search range (same as PUB50)
				"min_neurons": 5,
				"max_neurons": 500,
				"min_bits": 4,
				"max_bits": 34,

				# Architecture
				"architecture_type": "ids",
				"ids_single_cluster": True,

				# Training
				"balance_classes": True,
				"neuron_sample_rate": 0.25,
				"context_size": 4,

				# K-fold + statistical protocol
				"ids_k_folds": 5,
				"ids_kfold_per_gen": 5,
				"ids_num_parts": 5,
				"ids_val_fraction": 0.25,

				# Balanced fitness weights (same as PUB50 ciciot batch)
				"fitness_calculator": "harmonic_rank",
				"fitness_weight_ce": 0.1,
				"fitness_weight_f1": 0.35,
				"fitness_weight_fpr": 0.35,
				"fitness_weight_acc": 0.2,

				# Threshold sweep settings
				"threshold_start": 0,
				"threshold_step": 1,

				# GA settings (same as PUB50)
				"ga_generations": 250,
				"population_size": 50,
				"neighbors_per_iter": 50,
				"patience": 5,
				"fitness_percentile": 0.75,
				"cluster_crossover_ratio": 0.5,
				"assortative_mating_ratio": 0.85,
				"pool_shuffle_ratio": 0.8,
				"phase_order": "neurons_first",
				"adaptation_iterations": 50,
				"min_accuracy_floor": 0,

				# Reproducibility
				"seed": seed,
			},
		},
		# Two-phase: grid search over (neurons, bits) then GA neurons refinement
		"experiments": [
			{
				"name": "Grid Search (neurons x bits)",
				"phase_type": "grid_search",
				"experiment_type": "grid_search",
			},
			{
				"name": "GA Neurons",
				"phase_type": "ga_neurons",
				"experiment_type": "ga",
			},
		],
	}


def post_flow(url: str, body: dict) -> dict:
	data = json.dumps(body).encode("utf-8")
	req = urllib.request.Request(
		url,
		data=data,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	ctx = ssl.create_default_context()
	ctx.check_hostname = False
	ctx.verify_mode = ssl.CERT_NONE
	try:
		with urllib.request.urlopen(req, context=ctx) as resp:
			return json.loads(resp.read().decode("utf-8"))
	except urllib.error.HTTPError as e:
		print(f"ERROR: HTTP {e.code} {e.reason}", file=sys.stderr)
		print(e.read().decode("utf-8"), file=sys.stderr)
		sys.exit(1)


def main():
	parser = argparse.ArgumentParser(description="Create a PUB50-style ciciot batch with configurable thermometer")
	parser.add_argument("--count", type=int, default=10,
		help="Number of flows to create (default: 10)")
	parser.add_argument("--start-seed", type=int, default=1,
		help="First seed value (default: 1). Use 11 to add to an existing 10-run batch.")
	parser.add_argument("--thermometer-bits", type=int, default=2,
		help="Thermometer encoding width (default: 2)")
	parser.add_argument("--name-prefix", default=None,
		help="Flow name prefix (default: PUB50-{thermo}b-ciciot-random)")
	parser.add_argument("--api-url", default="https://localhost:3000/api/flows",
		help="Dashboard API URL")
	parser.add_argument("--queue", action="store_true",
		help="After creation, set flows to status='queued' via the DB directly")
	parser.add_argument("--dry-run", action="store_true",
		help="Print the request bodies without POSTing")
	args = parser.parse_args()

	name_prefix = args.name_prefix or f"PUB50-{args.thermometer_bits}b-ciciot-random"

	print(f"Creating {args.count} flows: seeds {args.start_seed}..{args.start_seed + args.count - 1}")
	print(f"  Name prefix: {name_prefix}")
	print(f"  Thermometer: {args.thermometer_bits}-bit")
	print(f"  Dataset:     ciciot2023 (1.3M subsample, random 80/20)")
	print(f"  Template:    ids-binary-2-phase")
	print()

	created_ids = []
	for seed in range(args.start_seed, args.start_seed + args.count):
		body = build_flow_request(seed=seed, thermometer_bits=args.thermometer_bits, name_prefix=name_prefix)
		if args.dry_run:
			print(f"[dry-run] would create: {body['name']}")
			continue
		resp = post_flow(args.api_url, body)
		flow_id = resp.get("id", "?")
		created_ids.append(flow_id)
		print(f"  ✓ Created flow {flow_id}: {body['name']}")

	if args.dry_run:
		print("\n(dry run — no flows actually created)")
		return

	print(f"\nCreated {len(created_ids)} flows (IDs: {min(created_ids)}..{max(created_ids)})")

	if args.queue:
		import sqlite3
		from pathlib import Path
		db_path = Path(__file__).resolve().parent.parent / "db" / "wnn.db"
		conn = sqlite3.connect(str(db_path))
		cur = conn.cursor()
		placeholders = ",".join("?" for _ in created_ids)
		cur.execute(
			f"UPDATE flows SET status='queued' WHERE id IN ({placeholders}) AND status='pending'",
			created_ids,
		)
		changes = cur.rowcount
		conn.commit()
		conn.close()
		print(f"Queued {changes} flows (worker will pick them up in ID-descending order).")
	else:
		print(f"\nFlows are 'pending'. To queue them:")
		print(f"  sqlite3 db/wnn.db \"UPDATE flows SET status='queued' WHERE name LIKE '{name_prefix}%' AND status='pending';\"")


if __name__ == "__main__":
	main()
