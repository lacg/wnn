#!/usr/bin/env python3
"""
Retroactive computation of combined (end-to-end) metrics for multi-stage flows.

Loads checkpoint files for the last experiment of each stage, identifies
best_ce / best_acc / best_fitness genomes from each stage's population,
then calls compute_combined_metrics() for each genome type pair.

Results are stored in the dashboard via POST /api/flows/{id}/combined-validations.

Usage:
    cd wnn/
    source wnn/bin/activate
    export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"
    python scripts/compute_combined_validations.py --flow-id 49
"""

import argparse
import gzip
import json
import sys
from pathlib import Path


def load_checkpoint(path: Path) -> dict:
	"""Load a checkpoint file (supports .json.gz)."""
	if path.suffix == '.gz':
		with gzip.open(path, 'rt', encoding='utf-8') as f:
			return json.load(f)
	else:
		with open(path, 'r') as f:
			return json.load(f)


def find_best_genomes(checkpoint_data: dict):
	"""Extract best_ce, best_acc, best_fitness genomes from checkpoint.

	Returns dict mapping genome_type -> ClusterGenome
	"""
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	from wnn.ram.experiments.phased_search import PhaseResult

	phase_result = PhaseResult.deserialize(checkpoint_data['phase_result'])
	best_fitness_genome = phase_result.best_genome

	results = {"best_fitness": best_fitness_genome}

	# Try to use pre-saved best_ce/best_acc genomes first
	if 'best_ce_genome' in checkpoint_data:
		results["best_ce"] = ClusterGenome.deserialize(checkpoint_data['best_ce_genome'])
	if 'best_acc_genome' in checkpoint_data:
		results["best_acc"] = ClusterGenome.deserialize(checkpoint_data['best_acc_genome'])

	# Fall back to population_metrics if pre-saved genomes not available
	if "best_ce" not in results or "best_acc" not in results:
		pop = phase_result.final_population
		metrics = checkpoint_data.get('population_metrics')
		if pop and metrics and len(pop) == len(metrics):
			if "best_ce" not in results:
				best_ce_idx = min(range(len(metrics)), key=lambda i: metrics[i][0])
				results["best_ce"] = pop[best_ce_idx]
			if "best_acc" not in results:
				best_acc_idx = max(range(len(metrics)), key=lambda i: metrics[i][1])
				results["best_acc"] = pop[best_acc_idx]
		else:
			# Last resort: use best_fitness for all types
			if "best_ce" not in results:
				results["best_ce"] = best_fitness_genome
			if "best_acc" not in results:
				results["best_acc"] = best_fitness_genome

	return results


def main():
	parser = argparse.ArgumentParser(description="Compute combined validations for multi-stage flows")
	parser.add_argument("--flow-id", type=int, required=True, help="Flow ID to compute for")
	parser.add_argument("--dashboard-url", default="https://localhost:3000", help="Dashboard API URL")
	parser.add_argument("--dry-run", action="store_true", help="Print results without storing")
	parser.add_argument("--checkpoint-dir", type=str, help="Override checkpoint directory path")
	args = parser.parse_args()

	# Get flow info from dashboard
	from wnn.ram.experiments.dashboard_client import DashboardClient, DashboardClientConfig

	client = DashboardClient(DashboardClientConfig(
		base_url=args.dashboard_url,
		verify_ssl=False,
	), logger=print)

	flow = client.get_flow(args.flow_id)
	if not flow:
		print(f"Error: Flow {args.flow_id} not found")
		sys.exit(1)

	params = flow.get('config', {}).get('params', {})
	architecture_type = params.get('architecture_type', 'tiered')
	if architecture_type != 'multi_stage':
		print(f"Error: Flow {args.flow_id} is not multi-stage (type={architecture_type})")
		sys.exit(1)

	num_stages = params.get('num_stages', 2)
	context_size = params.get('context_size', 4)
	stage_k = params.get('stage_k')
	stage_cluster_type = params.get('stage_cluster_type')
	memory_mode_str = params.get('memory_mode', 'QUAD_WEIGHTED')
	neuron_sample_rate = params.get('neuron_sample_rate', 0.25)

	print(f"Flow {args.flow_id}: {flow.get('name', '?')}")
	print(f"  Architecture: multi_stage, stages={num_stages}, context={context_size}")
	print(f"  Stage K: {stage_k}")
	print(f"  Memory mode: {memory_mode_str}, sample_rate={neuron_sample_rate}")
	print()

	# Find checkpoint directory
	if args.checkpoint_dir:
		ckpt_base = Path(args.checkpoint_dir)
	else:
		# List experiments to find the checkpoint dir
		experiments = client.list_flow_experiments(args.flow_id)
		if not experiments:
			print("Error: No experiments found for this flow")
			sys.exit(1)

		# Try to find checkpoint from the experiments' checkpoints
		checkpoints = client.list_checkpoints(limit=200)
		flow_exp_ids = {e['id'] for e in experiments}
		flow_ckpts = [c for c in checkpoints if c.get('experiment_id') in flow_exp_ids]

		if flow_ckpts:
			# Derive base dir from first checkpoint path
			first_path = Path(flow_ckpts[0]['file_path'])
			# Path is like: .../checkpoints/dirname/exp_XX/file.json.gz
			ckpt_base = first_path.parent.parent
			print(f"  Checkpoint dir (auto-detected): {ckpt_base}")
		else:
			print("Error: Could not determine checkpoint directory. Use --checkpoint-dir")
			sys.exit(1)

	# Load last checkpoint of each stage
	# Convention: experiments are numbered exp_00 through exp_N
	# Stage boundary: S0 ends at exp_09 (or similar), S1 starts at exp_10
	# We need the LAST experiment of each stage
	stage_checkpoints: dict[int, dict] = {}

	# Get experiments sorted by sequence order
	experiments = client.list_flow_experiments(args.flow_id)
	if not experiments:
		print("Error: No experiments found")
		sys.exit(1)

	# Group experiments by target_stage (from name pattern like "S0: ..." or "S1: ...")
	exp_by_stage: dict[int, list] = {}
	for exp in experiments:
		name = exp.get('name', '')
		for s in range(num_stages):
			if f"S{s}" in name:
				exp_by_stage.setdefault(s, []).append(exp)
				break

	for stage in range(num_stages):
		stage_exps = exp_by_stage.get(stage, [])
		if not stage_exps:
			print(f"Error: No experiments found for stage {stage}")
			sys.exit(1)

		# Find the last experiment of this stage (by sequence_order)
		last_exp = max(stage_exps, key=lambda e: e.get('sequence_order', 0))
		last_exp_idx = last_exp.get('sequence_order', 0)
		exp_dir = ckpt_base / f"exp_{last_exp_idx:02d}"

		print(f"  Stage {stage}: last experiment = exp_{last_exp_idx:02d} ({last_exp.get('name', '?')})")

		# Find checkpoint file in that directory
		ckpt_files = list(exp_dir.glob("*.json.gz"))
		if not ckpt_files:
			print(f"Error: No checkpoint files in {exp_dir}")
			sys.exit(1)

		ckpt_file = ckpt_files[0]  # Should be exactly one
		print(f"    Loading: {ckpt_file.name}")
		stage_checkpoints[stage] = load_checkpoint(ckpt_file)

	# Extract genomes per type per stage
	print()
	stage_genomes: dict[int, dict[str, object]] = {}
	for stage in range(num_stages):
		stage_genomes[stage] = find_best_genomes(stage_checkpoints[stage])
		genomes = stage_genomes[stage]
		print(f"  Stage {stage} genomes: {', '.join(genomes.keys())}")

	# Create evaluator for combined metrics
	print()
	print("Creating MultiStageEvaluator...")
	from wnn.ram.architecture.multistage_evaluator import MultiStageEvaluator
	from wnn.ram.experiments.experiment import StageMode

	# Load tokenizer + data
	from wnn.ram.experiments.worker import FlowWorker
	worker = FlowWorker.__new__(FlowWorker)
	worker._log = print

	# Load data the same way as the worker
	print("Loading GPT-2 tokenizer + WikiText-2...")
	import tiktoken
	tokenizer = tiktoken.get_encoding("gpt2")
	from datasets import load_dataset
	dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
	train_text = "\n\n".join([t for t in dataset["train"]["text"] if t.strip()])
	test_text = "\n\n".join([t for t in dataset["test"]["text"] if t.strip()])
	train_tokens = tokenizer.encode(train_text)
	test_tokens = tokenizer.encode(test_text)
	vocab_size = tokenizer.n_vocab

	memory_mode_map = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2}
	memory_mode = memory_mode_map.get(memory_mode_str, 2)

	# Parse stage_mode
	raw_mode = params.get("stage_mode")
	if isinstance(raw_mode, str):
		mode_map = {"input_concat": StageMode.INPUT_CONCAT, "selector": StageMode.SELECTOR}
		stage_mode = [mode_map.get(raw_mode, StageMode.INPUT_CONCAT)]
	elif isinstance(raw_mode, list):
		mode_map = {"input_concat": StageMode.INPUT_CONCAT, "selector": StageMode.SELECTOR}
		stage_mode = [mode_map.get(m, StageMode.INPUT_CONCAT) if isinstance(m, str) else m for m in raw_mode]
	else:
		stage_mode = None

	evaluator = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=test_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		num_stages=num_stages,
		stage_k=stage_k,
		stage_cluster_type=stage_cluster_type,
		stage_mode=stage_mode,
		target_stage=0,
		num_parts=1,
		num_eval_parts=1,
		memory_mode=memory_mode,
		neuron_sample_rate=neuron_sample_rate,
	)
	print(f"  MultiStageEvaluator ready: stages={num_stages}")

	# Compute combined metrics for each genome type
	print()
	print("=" * 70)
	print("  Computing Combined Metrics")
	print("=" * 70)

	for genome_type in ("best_ce", "best_acc", "best_fitness"):
		# Build genome pair: one from each stage
		genome_pair = []
		for stage in range(num_stages):
			g = stage_genomes[stage].get(genome_type)
			if g is None:
				print(f"  {genome_type}: Missing genome for stage {stage}, skipping")
				break
			genome_pair.append(g)
		else:
			# All stages have genomes
			try:
				result = evaluator.compute_combined_metrics(genome_pair)
				ce = result.ce
				acc = result.accuracy
				per_stage_ce = [result.cluster_ce, result.within_ce]

				print(f"  {genome_type}: CE={ce:.4f}, ACC={acc:.2%}, S0={per_stage_ce[0]:.4f}, S1={per_stage_ce[1]:.4f}")

				if not args.dry_run:
					client.create_combined_validation(
						flow_id=args.flow_id,
						genome_type=genome_type,
						combined_ce=ce,
						combined_accuracy=acc,
						per_stage_ce=per_stage_ce,
					)
					print(f"    -> Stored in dashboard")

			except Exception as e:
				print(f"  {genome_type}: Failed - {e}")

	print()
	print("Done!")


if __name__ == "__main__":
	main()
