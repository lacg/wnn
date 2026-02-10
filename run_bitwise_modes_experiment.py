#!/usr/bin/env python3
"""
Bitwise Memory Modes Experiment — Compare TERNARY, QUAD_BINARY, QUAD_WEIGHTED
with varying neuron_sample_rate and bits_per_neuron on WikiText-2.

Phase 1: Mode × sample_rate comparison (9 configs)
Phase 2: bits_per_neuron sweep for top-3 from Phase 1
Phase 3: neuron count sweep for top-3 from Phases 1-2

Usage:
	python run_bitwise_modes_experiment.py [--phase 1|2|3|all] [--output results.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# WikiText-2 data loading (reuse existing infrastructure)
def load_wikitext2_tokens(tokenizer_name="gpt2"):
	"""Load WikiText-2 train/test tokens using GPT-2 tokenizer."""
	try:
		from datasets import load_dataset
		from transformers import AutoTokenizer
	except ImportError:
		print("ERROR: datasets and transformers required. Install with:")
		print("  pip install datasets transformers")
		sys.exit(1)

	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

	train_text = "\n".join(dataset["train"]["text"])
	test_text = "\n".join(dataset["test"]["text"])

	train_tokens = tokenizer.encode(train_text)
	test_tokens = tokenizer.encode(test_text)

	vocab_size = tokenizer.vocab_size
	print(f"Loaded WikiText-2: {len(train_tokens):,} train, {len(test_tokens):,} test tokens, vocab={vocab_size}")

	return train_tokens, test_tokens, vocab_size


def run_single_config(
	train_tokens, test_tokens, vocab_size,
	neurons_per_cluster, bits_per_neuron, context_size,
	memory_mode, neuron_sample_rate,
	label,
):
	"""Run a single config and return results dict."""
	from wnn.ram.core.models import BitwiseRAMLM
	from wnn.ram.core import MemoryMode

	mode_names = {0: "TERNARY", 1: "QUAD_BINARY", 2: "QUAD_WEIGHTED"}

	print(f"\n{'='*70}")
	print(f"Config: {label}")
	print(f"  mode={mode_names[memory_mode]}, rate={neuron_sample_rate}, "
		  f"neurons={neurons_per_cluster}, bits={bits_per_neuron}, ctx={context_size}")
	print(f"{'='*70}")

	model = BitwiseRAMLM(
		vocab_size=vocab_size,
		context_size=context_size,
		neurons_per_cluster=neurons_per_cluster,
		bits_per_neuron=bits_per_neuron,
		memory_mode=memory_mode,
		neuron_sample_rate=neuron_sample_rate,
	)
	print(f"Model: {model}")

	# Train
	t0 = time.time()
	model.reset_memory()
	train_stats = model.train_epoch_fast(
		token_ids=train_tokens,
		batch_size=2000,
		verbose=True,
	)
	train_time = time.time() - t0
	print(f"Training time: {train_time:.1f}s")

	# Evaluate
	t0 = time.time()
	eval_stats = model.evaluate_fast(
		token_ids=test_tokens,
		batch_size=5000,
		verbose=True,
		per_bit=True,
	)
	eval_time = time.time() - t0
	print(f"Eval time: {eval_time:.1f}s")

	result = {
		"label": label,
		"mode": mode_names[memory_mode],
		"memory_mode": memory_mode,
		"neuron_sample_rate": neuron_sample_rate,
		"neurons_per_cluster": neurons_per_cluster,
		"bits_per_neuron": bits_per_neuron,
		"context_size": context_size,
		"cross_entropy": eval_stats["cross_entropy"],
		"perplexity": eval_stats["perplexity"],
		"accuracy": eval_stats["accuracy"],
		"mean_bit_accuracy": eval_stats.get("mean_bit_accuracy", None),
		"per_bit_accuracy": eval_stats.get("per_bit_accuracy", None),
		"train_modified": train_stats["modified"],
		"train_examples": train_stats["examples"],
		"train_time": train_time,
		"eval_time": eval_time,
	}

	return result


def print_summary_table(results, title="Results"):
	"""Print a formatted summary table."""
	print(f"\n{'='*90}")
	print(f"  {title}")
	print(f"{'='*90}")
	print(f"{'Label':<35} {'Mode':<15} {'Rate':>5} {'CE':>8} {'PPL':>10} {'Acc':>7} {'BitAcc':>7}")
	print(f"{'-'*35} {'-'*15} {'-'*5} {'-'*8} {'-'*10} {'-'*7} {'-'*7}")

	# Sort by CE (lower is better)
	for r in sorted(results, key=lambda x: x["cross_entropy"]):
		bit_acc = f"{r['mean_bit_accuracy']:.2%}" if r.get("mean_bit_accuracy") else "N/A"
		print(f"{r['label']:<35} {r['mode']:<15} {r['neuron_sample_rate']:>5.1f} "
			  f"{r['cross_entropy']:>8.4f} {r['perplexity']:>10.2f} "
			  f"{r['accuracy']:>7.2%} {bit_acc:>7}")

	print(f"{'='*90}")


def phase1_mode_comparison(train_tokens, test_tokens, vocab_size, context_size=4):
	"""Phase 1: Compare all 9 mode×rate configs."""
	print("\n" + "#" * 70)
	print("# PHASE 1: Mode × Sample Rate Comparison")
	print("#" * 70)

	neurons = 1000
	bits = 10

	configs = [
		(0, 1.0, "TERNARY_rate1.0"),
		(0, 0.5, "TERNARY_rate0.5"),
		(0, 0.3, "TERNARY_rate0.3"),
		(1, 1.0, "QUAD_BINARY_rate1.0"),
		(1, 0.5, "QUAD_BINARY_rate0.5"),
		(1, 0.3, "QUAD_BINARY_rate0.3"),
		(2, 1.0, "QUAD_WEIGHTED_rate1.0"),
		(2, 0.5, "QUAD_WEIGHTED_rate0.5"),
		(2, 0.3, "QUAD_WEIGHTED_rate0.3"),
	]

	results = []
	for mode, rate, label in configs:
		r = run_single_config(
			train_tokens, test_tokens, vocab_size,
			neurons, bits, context_size,
			mode, rate, label,
		)
		results.append(r)

	print_summary_table(results, "Phase 1: Mode × Sample Rate")
	return results


def phase2_bits_sweep(train_tokens, test_tokens, vocab_size, top3_configs, context_size=4):
	"""Phase 2: bits_per_neuron sweep for top-3 from Phase 1."""
	print("\n" + "#" * 70)
	print("# PHASE 2: bits_per_neuron Sweep (top-3 from Phase 1)")
	print("#" * 70)

	neurons = 1000
	bits_values = [6, 8, 10, 12, 14, 16]

	results = []
	for cfg in top3_configs:
		mode = cfg["memory_mode"]
		rate = cfg["neuron_sample_rate"]
		mode_name = cfg["mode"]

		for bits in bits_values:
			label = f"{mode_name}_rate{rate}_bits{bits}"
			r = run_single_config(
				train_tokens, test_tokens, vocab_size,
				neurons, bits, context_size,
				mode, rate, label,
			)
			results.append(r)

	print_summary_table(results, "Phase 2: bits_per_neuron Sweep")
	return results


def phase3_neuron_sweep(train_tokens, test_tokens, vocab_size, top3_configs, context_size=4):
	"""Phase 3: neuron count sweep for top-3 from Phases 1-2."""
	print("\n" + "#" * 70)
	print("# PHASE 3: Neuron Count Sweep (top-3 from Phases 1-2)")
	print("#" * 70)

	neuron_values = [200, 500, 1000]

	results = []
	for cfg in top3_configs:
		mode = cfg["memory_mode"]
		rate = cfg["neuron_sample_rate"]
		bits = cfg["bits_per_neuron"]
		mode_name = cfg["mode"]

		for neurons in neuron_values:
			label = f"{mode_name}_rate{rate}_bits{bits}_n{neurons}"
			r = run_single_config(
				train_tokens, test_tokens, vocab_size,
				neurons, bits, context_size,
				mode, rate, label,
			)
			results.append(r)

	print_summary_table(results, "Phase 3: Neuron Count Sweep")
	return results


def main():
	parser = argparse.ArgumentParser(description="Bitwise Memory Modes Experiment")
	parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "all"],
						help="Which phase(s) to run")
	parser.add_argument("--output", type=str, default="experiments/bitwise_modes_results.json",
						help="Output JSON file")
	parser.add_argument("--context", type=int, default=4, help="Context size")
	args = parser.parse_args()

	# Load data
	train_tokens, test_tokens, vocab_size = load_wikitext2_tokens()

	all_results = {}

	# Phase 1
	if args.phase in ("1", "all"):
		p1 = phase1_mode_comparison(train_tokens, test_tokens, vocab_size, args.context)
		all_results["phase1"] = p1

		# Select top-3 by CE
		top3 = sorted(p1, key=lambda x: x["cross_entropy"])[:3]
		print("\nTop-3 from Phase 1:")
		for i, t in enumerate(top3):
			print(f"  #{i+1}: {t['label']} — CE={t['cross_entropy']:.4f}")
		all_results["phase1_top3"] = [t["label"] for t in top3]

	# Phase 2
	if args.phase in ("2", "all"):
		if "phase1" not in all_results:
			# Load Phase 1 results
			if os.path.exists(args.output):
				with open(args.output) as f:
					all_results = json.load(f)
			else:
				print("ERROR: Phase 1 results not found. Run phase 1 first.")
				sys.exit(1)

		p1 = all_results["phase1"]
		top3 = sorted(p1, key=lambda x: x["cross_entropy"])[:3]

		p2 = phase2_bits_sweep(train_tokens, test_tokens, vocab_size, top3, args.context)
		all_results["phase2"] = p2

		# Select top-3 by CE across Phase 2
		top3_p2 = sorted(p2, key=lambda x: x["cross_entropy"])[:3]
		print("\nTop-3 from Phase 2:")
		for i, t in enumerate(top3_p2):
			print(f"  #{i+1}: {t['label']} — CE={t['cross_entropy']:.4f}")
		all_results["phase2_top3"] = [t["label"] for t in top3_p2]

	# Phase 3
	if args.phase in ("3", "all"):
		if "phase2" not in all_results:
			if os.path.exists(args.output):
				with open(args.output) as f:
					all_results = json.load(f)
			else:
				print("ERROR: Phase 2 results not found. Run phase 2 first.")
				sys.exit(1)

		p2 = all_results["phase2"]
		top3 = sorted(p2, key=lambda x: x["cross_entropy"])[:3]

		p3 = phase3_neuron_sweep(train_tokens, test_tokens, vocab_size, top3, args.context)
		all_results["phase3"] = p3

	# Save results
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w") as f:
		json.dump(all_results, f, indent=2)
	print(f"\nResults saved to {output_path}")

	# Final summary
	if "phase1" in all_results:
		print_summary_table(all_results["phase1"], "FINAL — Phase 1")
	if "phase2" in all_results:
		print_summary_table(all_results["phase2"], "FINAL — Phase 2")
	if "phase3" in all_results:
		print_summary_table(all_results["phase3"], "FINAL — Phase 3")


if __name__ == "__main__":
	main()
