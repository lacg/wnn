"""
Create a single-genome evaluation flow for CIC-IoT-2023 (full 46M dataset).

POSTs to the dashboard API to create a flow with one Grid Search experiment
containing exactly one (neurons, bits) point and no GA refinement. This is
the workhorse for the 46M Pareto sweep — call it once per (n, b, seed) combo.

Usage:
    python scripts/create_46M_eval_flow.py --neurons 200 --bits 4 --seed 42
    python scripts/create_46M_eval_flow.py --neurons 5 --bits 4 --seed 1 --name CV1

Optional --dataset flag lets you target the 1.3M subsample instead for
quick smoke tests:
    python scripts/create_46M_eval_flow.py --neurons 5 --bits 4 --seed 1 --dataset ciciot2023
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def build_flow_request(name: str, description: str, neurons: int, bits: int,
                       seed: int, dataset: str) -> dict:
	"""Build the POST body for a single-genome 46M evaluation flow.

	Mirrors the PUB50 ciciot flow config (balanced fitness weights,
	fitness-aligned threshold, 5x K-fold) but constrains the grid to a
	single (neurons, bits) point and skips GA refinement.
	"""
	return {
		"name": name,
		"description": description,
		"config": {
			"template": "ids-binary-2-phase",
			"params": {
				# Dataset
				"ids_dataset": dataset,
				"ids_split": "random",
				"ids_n_bits": 8,
				"ids_feature_selection": "top20",
				"ids_classification": "binary",

				# Single-point grid (not a real grid)
				"min_neurons": neurons,
				"max_neurons": neurons,
				"min_bits": bits,
				"max_bits": bits,

				# Architecture
				"architecture_type": "ids",
				"ids_single_cluster": True,

				# Training
				"balance_classes": True,
				"neuron_sample_rate": 0.25,
				"context_size": 4,

				# K-fold within 80% train, 20% held-out for final report
				"ids_k_folds": 5,
				"ids_kfold_per_gen": 5,
				"ids_num_parts": 5,
				"ids_val_fraction": 0.25,

				# Balanced fitness weights (same as PUB50)
				"fitness_calculator": "harmonic_rank",
				"fitness_weight_ce": 0.1,
				"fitness_weight_f1": 0.35,
				"fitness_weight_fpr": 0.35,
				"fitness_weight_acc": 0.2,

				# Threshold sweep settings
				"threshold_start": 0,
				"threshold_step": 1,

				# GA settings (only used by Grid Search for population init)
				"ga_generations": 1,  # minimal — Grid Search uses pop_size combinations
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
		# Only Grid Search — no GA Neurons phase. The single-point grid acts as a
		# direct train/eval of the requested (neurons, bits) configuration.
		"experiments": [
			{
				"name": "Grid Search (1 point)",
				"phase_type": "grid_search",
				"experiment_type": "grid_search",
			},
		],
	}


def post_flow(url: str, body: dict) -> dict:
	"""POST the flow to the dashboard API and return the response."""
	data = json.dumps(body).encode("utf-8")
	req = urllib.request.Request(
		url,
		data=data,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	# Dashboard uses a self-signed cert; disable verification for localhost.
	import ssl
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
	parser = argparse.ArgumentParser(description="Create a single-genome 46M eval flow")
	parser.add_argument("--neurons", type=int, required=True, help="Number of neurons")
	parser.add_argument("--bits", type=int, required=True, help="Bits per neuron address")
	parser.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility)")
	parser.add_argument("--dataset", default="ciciot2023_full",
		choices=["ciciot2023", "ciciot2023_full"],
		help="ciciot2023 = 1.3M subsample, ciciot2023_full = 46M (default)")
	parser.add_argument("--name", default=None,
		help="Custom flow name suffix (default: auto-generated)")
	parser.add_argument("--api-url", default="https://localhost:3000/api/flows",
		help="Dashboard API URL (default: https://localhost:3000/api/flows)")
	args = parser.parse_args()

	# Auto-generate name if not provided
	dataset_label = "46M" if args.dataset == "ciciot2023_full" else "1p3M"
	suffix = f"-{args.name}" if args.name else ""
	name = f"EVAL46M-{dataset_label}-{args.neurons}n{args.bits}b-s{args.seed}{suffix}"

	description = (
		f"Single-genome evaluation: {args.neurons}n × {args.bits}b on "
		f"{args.dataset} (seed={args.seed}). Grid Search with 1 point, no GA. "
		f"Memory footprint: {args.neurons * (1 << args.bits) * 2 // 8} bytes."
	)

	body = build_flow_request(
		name=name,
		description=description,
		neurons=args.neurons,
		bits=args.bits,
		seed=args.seed,
		dataset=args.dataset,
	)

	print(f"Creating flow: {name}")
	print(f"  Memory: {args.neurons * (1 << args.bits) * 2 // 8} bytes")
	print(f"  Dataset: {args.dataset}")
	print(f"  POST {args.api_url}")
	resp = post_flow(args.api_url, body)
	flow_id = resp.get("id", "?")
	print(f"\n✓ Created flow {flow_id}: {name}")
	print(f"  View at: https://localhost:3000/flows/{flow_id}")


if __name__ == "__main__":
	main()
