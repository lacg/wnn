"""
Create a UNSW-NB15 random 80/20 thermometer mini-sweep.

Purpose: before committing to a full 112-run UNSW random PUB50 batch
with a chosen thermometer width, run a quick 6-width × 2-seed pre-test
to identify which thermometer width is best on UNSW (may not be the
same as ciciot, which favoured 2-bit).

Produces 12 flows total:
    6 widths × 2 seeds = 12 flows
    widths: 2, 4, 8, 16, 32, 64 bits per feature

Each flow is a FULL PUB50-style 2-phase run (grid search + GA neurons),
matching the methodology of the existing PUB50 ciciot batch:
    - balanced fitness weights (0.1/0.35/0.35/0.2)
    - fitness-aligned threshold
    - single cluster
    - k_folds=5, kfold_per_gen=5
    - 250 GA generations, patience=5
    - min/max neurons 5/500, min/max bits 4/34

UNSW is smaller than ciciot (~258K rows vs 1.3M), so per-flow time
should be ~15-25 min. 12 flows → ~3-4 hours total.

Usage:
    python scripts/create_unsw_thermo_minisweep.py             # 12 flows, pending
    python scripts/create_unsw_thermo_minisweep.py --queue     # also queue
    python scripts/create_unsw_thermo_minisweep.py --dry-run
"""

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request


def build_flow_request(seed: int, thermometer_bits: int) -> dict:
	"""Build the POST body for a UNSW random 80/20 PUB50-style flow."""
	return {
		"name": f"UNSW-fitfix-t{thermometer_bits}b-random-r{seed:03d}",
		"description": (
			f"UNSW-NB15 random 80/20 with balanced fitness weights "
			f"(0.1/0.35/0.35/0.2), fitness-aligned threshold, and "
			f"{thermometer_bits}-bit thermometer encoding. Part of the "
			f"UNSW thermometer mini-sweep to identify the best width "
			f"before committing to a full 112-run batch. Same GA + grid "
			f"search pipeline as PUB50 ciciot; seed {seed}."
		),
		"config": {
			"template": "ids-binary-2-phase",
			"params": {
				# Dataset
				"ids_dataset": "unsw-nb15",
				"ids_split": "random",
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
		url, data=data, headers={"Content-Type": "application/json"}, method="POST",
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
	parser = argparse.ArgumentParser(description="Create UNSW thermometer mini-sweep")
	parser.add_argument("--seeds", type=int, default=2, help="Seeds per thermometer width (default: 2)")
	parser.add_argument("--widths", type=str, default="2,4,8,16,32,64",
		help="Comma-separated thermometer widths (default: 2,4,8,16,32,64)")
	parser.add_argument("--api-url", default="https://localhost:3000/api/flows")
	parser.add_argument("--queue", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	widths = [int(w) for w in args.widths.split(",")]
	print(f"Creating UNSW thermometer mini-sweep:")
	print(f"  Widths:  {widths}")
	print(f"  Seeds:   {args.seeds} per width")
	print(f"  Total:   {len(widths) * args.seeds} flows")
	print(f"  Dataset: UNSW-NB15 random 80/20")
	print(f"  Fitness: balanced (0.1/0.35/0.35/0.2), fitness-aligned threshold")
	print()

	created_ids = []
	for width in widths:
		for seed in range(1, args.seeds + 1):
			body = build_flow_request(seed=seed, thermometer_bits=width)
			if args.dry_run:
				print(f"  [dry-run] would create: {body['name']}")
				continue
			resp = post_flow(args.api_url, body)
			flow_id = resp.get("id", "?")
			created_ids.append(flow_id)
			print(f"  ✓ Created flow {flow_id}: {body['name']}")

	if args.dry_run:
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
		print(f"  sqlite3 db/wnn.db \"UPDATE flows SET status='queued' WHERE name LIKE 'UNSW-fitfix-%' AND status='pending';\"")


if __name__ == "__main__":
	main()
