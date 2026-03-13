#!/usr/bin/env python3
"""
Backfill best_genomes leaderboard from existing validation_summaries.

Reads validation_summaries with validation_point='final', finds the
corresponding checkpoint files, extracts genome data, and submits
to the best_genomes leaderboard via the dashboard API.

Idempotent — UNIQUE constraint prevents duplicates on re-run.

Usage:
	python scripts/backfill_best_genomes.py [--dashboard-url URL] [--no-ssl-verify]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path


def main():
	parser = argparse.ArgumentParser(description="Backfill best genomes from existing experiments")
	parser.add_argument("--dashboard-url", default="https://localhost:3000")
	parser.add_argument("--no-ssl-verify", action="store_true")
	parser.add_argument("--dry-run", action="store_true", help="Print what would be submitted")
	args = parser.parse_args()

	from wnn.ram.experiments.dashboard_client import DashboardClient, DashboardClientConfig

	client_config = DashboardClientConfig(
		base_url=args.dashboard_url,
		verify_ssl=not args.no_ssl_verify,
	)
	client = DashboardClient(client_config)

	# Verify connectivity
	if not client.ping():
		print(f"Error: Cannot reach dashboard at {args.dashboard_url}")
		sys.exit(1)

	print("Fetching experiments and validation summaries...")

	# Get all experiments
	experiments = client._request("GET", "/api/experiments") or []
	print(f"  Found {len(experiments)} experiments")

	# Get all checkpoints
	checkpoints = client._request("GET", "/api/checkpoints") or []
	print(f"  Found {len(checkpoints)} checkpoints")

	# Build checkpoint lookup: experiment_id -> list of checkpoints
	ckpt_by_exp = {}
	for ckpt in checkpoints:
		exp_id = ckpt.get("experiment_id")
		if exp_id not in ckpt_by_exp:
			ckpt_by_exp[exp_id] = []
		ckpt_by_exp[exp_id].append(ckpt)

	# Process each experiment's validation summaries
	submitted = 0
	skipped = 0
	failed = 0
	categories_seen = set()

	for exp in experiments:
		exp_id = exp["id"]
		exp_name = exp.get("name", "")
		flow_id = exp.get("flow_id")
		arch_type = exp.get("architecture_type", "tiered")

		# Get validation summaries for this experiment
		summaries = client._request("GET", f"/api/experiments/{exp_id}/summaries") or []
		final_summaries = [s for s in summaries if s.get("validation_point") == "final"]

		if not final_summaries:
			continue

		# Determine task_type and stage
		task_type = "ids" if arch_type == "ids" else "lm"
		if exp_name.startswith("S1:") or exp_name.startswith("S1 "):
			stage = "stage_1"
		elif exp_name.startswith("S2:") or exp_name.startswith("S2 "):
			stage = "stage_2"
		else:
			stage = "stage_0"

		for summary in final_summaries:
			genome_type = summary.get("genome_type", "best_ce")
			genome_hash = summary.get("genome_hash", "")
			ce = summary.get("ce", 0)
			accuracy = summary.get("accuracy", 0)
			f1_macro = summary.get("f1_macro")
			fpr = summary.get("fpr")

			# Determine metric from genome_type
			metric = genome_type.replace("best_", "")
			if metric == "acc":
				metric = "accuracy"

			# Try to find checkpoint with genome data
			genome_data = None
			exp_checkpoints = ckpt_by_exp.get(exp_id, [])
			for ckpt in exp_checkpoints:
				if ckpt.get("checkpoint_type") == "experiment_end":
					file_path = ckpt.get("file_path", "")
					genome_data = _load_genome_from_checkpoint(file_path, genome_type)
					if genome_data:
						break

			payload = {
				"task_type": task_type,
				"stage": stage,
				"metric": metric,
				"genome_hash": genome_hash,
				"ce": ce,
				"accuracy": accuracy,
				"f1_macro": f1_macro,
				"fpr": fpr,
				"flow_id": flow_id,
				"experiment_id": exp_id,
			}
			if genome_data:
				payload["genome_data"] = genome_data

			categories_seen.add((task_type, stage, metric))

			if args.dry_run:
				print(f"  Would submit: {task_type}/{stage}/{metric} hash={genome_hash[:8]}... CE={ce:.4f}")
				submitted += 1
				continue

			try:
				result = client.submit_best_genomes([payload])
				accepted = result.get("accepted", [])
				if accepted:
					submitted += 1
				else:
					skipped += 1
			except Exception as e:
				print(f"  Error submitting {genome_hash[:8]}...: {e}")
				failed += 1

	print(f"\nBackfill complete:")
	print(f"  Submitted: {submitted}")
	print(f"  Skipped:   {skipped}")
	print(f"  Failed:    {failed}")

	# Recalculate rankings for all affected categories
	if not args.dry_run and categories_seen:
		print(f"\nRecalculating rankings for {len(categories_seen)} categories...")
		for task_type, stage, metric in sorted(categories_seen):
			try:
				result = client.recalculate_best_genomes(task_type, stage, metric)
				ranked = result.get("ranked", 0)
				print(f"  {task_type}/{stage}/{metric}: {ranked} ranked")
			except Exception as e:
				print(f"  Error recalculating {task_type}/{stage}/{metric}: {e}")


def _load_genome_from_checkpoint(file_path: str, genome_type: str) -> dict | None:
	"""Try to load genome data from a checkpoint file."""
	try:
		path = Path(file_path)
		if not path.exists():
			return None

		if path.suffix == '.gz':
			with gzip.open(path, 'rt', encoding='utf-8') as f:
				data = json.load(f)
		else:
			with open(path, 'r') as f:
				data = json.load(f)

		# Look for genome in phase_result or final format
		genome_dict = None
		if 'phase_result' in data:
			pr = data['phase_result']
			if genome_type == 'best_ce' and 'best_genome' in pr:
				genome_dict = pr['best_genome']
		elif 'final' in data:
			final = data['final']
			if 'genome' in final:
				genome_dict = final['genome']

		if not genome_dict:
			return None

		# Extract relevant fields
		result = {}
		if 'connections' in genome_dict:
			result['connections_json'] = ','.join(str(c) for c in genome_dict['connections'])
		if 'bits_per_neuron' in genome_dict:
			result['total_neurons'] = sum(genome_dict.get('neurons_per_cluster', []))
			result['total_clusters'] = len(genome_dict.get('neurons_per_cluster', []))

		# Build tiers_json from bits_per_neuron + neurons_per_cluster
		bpn = genome_dict.get('bits_per_neuron', [])
		npc = genome_dict.get('neurons_per_cluster', [])
		if npc:
			tiers = []
			neuron_offset = 0
			for i, n in enumerate(npc):
				b = bpn[neuron_offset] if neuron_offset < len(bpn) else 16
				tiers.append({"tier": i, "clusters": 1, "neurons": n, "bits": b})
				neuron_offset += n
			result['tiers_json'] = json.dumps(tiers)

		result['config_hash'] = genome_dict.get('config_hash', '')
		result['architecture_type'] = 'bitwise'

		return result if result.get('connections_json') else None

	except Exception:
		return None


if __name__ == "__main__":
	main()
