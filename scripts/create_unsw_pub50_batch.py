"""
Create the UNSW-NB15 PUB50-style 112-run statistical batch.

Uses the same balanced-fitness + fitness-aligned threshold protocol as
the CIC-IoT PUB50 batch, with the thermometer width chosen from the
UNSW mini-sweep results (scripts/create_unsw_thermo_minisweep.py).

Default: 8-bit thermometer (override with --thermometer-bits after
mini-sweep identifies the best width).

Each flow is a 2-phase run (grid search + GA neurons) on the UNSW-NB15
random 80/20 split with top-20 features and the corrected feature names
(all 20/20 matching the HuggingFace dataset).

UNSW is smaller than CIC-IoT (~1.27M train vs 1.07M), so per-flow time
should be ~15-25 min. 112 flows → ~28-47 hours total.

Usage:
    python scripts/create_unsw_pub50_batch.py                     # 112 flows, 8b, pending
    python scripts/create_unsw_pub50_batch.py --thermometer-bits 2 # override width
    python scripts/create_unsw_pub50_batch.py --count 10           # test batch
    python scripts/create_unsw_pub50_batch.py --queue              # also queue
    python scripts/create_unsw_pub50_batch.py --dry-run
"""

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request


def build_flow_request(seed: int, thermometer_bits: int = 8,
                       name_prefix: str = None, split: str = "random") -> dict:
	"""Build the POST body for a UNSW PUB50-style flow."""
	if name_prefix is None:
		name_prefix = f"PUB50-{thermometer_bits}b-unsw-{split}"

	return {
		"name": f"{name_prefix}-r{seed:03d}",
		"description": (
			f"UNSW-NB15 {split} with balanced fitness weights "
			f"(0.1/0.35/0.35/0.2), fitness-aligned threshold, and "
			f"{thermometer_bits}-bit thermometer encoding. Part of the "
			f"112-run PUB50-style statistical batch; seed {seed}."
		),
		"config": {
			"template": "ids-binary-2-phase",
			"params": {
				# Dataset
				"ids_dataset": "unsw-nb15",
				"ids_split": split,
				"ids_n_bits": thermometer_bits,
				"ids_feature_selection": "top20",
				"ids_classification": "binary",

				# Architecture search range (same as PUB50 ciciot)
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

				# Balanced fitness weights (the fitness fix)
				"fitness_calculator": "harmonic_rank",
				"fitness_weight_ce": 0.1,
				"fitness_weight_f1": 0.35,
				"fitness_weight_fpr": 0.35,
				"fitness_weight_acc": 0.2,

				# Threshold sweep
				"threshold_start": 0,
				"threshold_step": 1,

				# GA settings (same as PUB50 ciciot)
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
		url, data=data,
		headers={"Content-Type": "application/json"}, method="POST",
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
	parser = argparse.ArgumentParser(
		description="Create UNSW-NB15 PUB50-style 112-run statistical batch")
	parser.add_argument("--count", type=int, default=112,
		help="Number of flows to create (default: 112)")
	parser.add_argument("--start-seed", type=int, default=1,
		help="First seed value (default: 1)")
	parser.add_argument("--thermometer-bits", type=int, default=8,
		help="Thermometer encoding width (default: 8, override after mini-sweep)")
	parser.add_argument("--split", default="random",
		help="Dataset split: 'random' or 'temporal' (default: random)")
	parser.add_argument("--name-prefix", default=None,
		help="Flow name prefix (default: PUB50-{thermo}b-unsw-{split})")
	parser.add_argument("--api-url", default="https://localhost:3000/api/flows",
		help="Dashboard API URL")
	parser.add_argument("--queue", action="store_true",
		help="Set flows to status='queued' after creation")
	parser.add_argument("--dry-run", action="store_true",
		help="Print flow names without creating")
	args = parser.parse_args()

	name_prefix = args.name_prefix or f"PUB50-{args.thermometer_bits}b-unsw-{args.split}"

	print(f"Creating {args.count} UNSW-NB15 PUB50 flows:")
	print(f"  Seeds:       {args.start_seed}..{args.start_seed + args.count - 1}")
	print(f"  Name prefix: {name_prefix}")
	print(f"  Split:       {args.split}")
	print(f"  Thermometer: {args.thermometer_bits}-bit")
	print(f"  Dataset:     UNSW-NB15 ({args.split}, top-20 features)")
	print(f"  Fitness:     balanced (0.1/0.35/0.35/0.2), fitness-aligned threshold")
	print(f"  Template:    ids-binary-2-phase (grid search + GA neurons)")
	print()

	created_ids = []
	for seed in range(args.start_seed, args.start_seed + args.count):
		body = build_flow_request(
			seed=seed,
			thermometer_bits=args.thermometer_bits,
			name_prefix=name_prefix,
			split=args.split,
		)
		if args.dry_run:
			print(f"  [dry-run] would create: {body['name']}")
			continue
		resp = post_flow(args.api_url, body)
		flow_id = resp.get("id", "?")
		created_ids.append(flow_id)
		print(f"  Created flow {flow_id}: {body['name']}")

	if args.dry_run:
		print(f"\n(dry run — {args.count} flows would be created)")
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
