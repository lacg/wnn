#!/usr/bin/env python3
"""
Focused Masked vs Unmasked Evaluation.

Compares three evaluation modes:
1. Softmax + EMPTY=0.0 (what grid sweep uses — untrained clusters get exp(0)=1)
2. Sum-norm + EMPTY=0.0 (TRUE masking — untrained clusters get 0, excluded)
3. Softmax + EMPTY=0.5 (no masking — untrained clusters get exp(0.5)=1.65)

Key question: Does masking change which bit configuration is best?
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "wnn"))

import torch
from torch.nn.functional import softmax

import ram_accelerator
from wnn.ram.core.models import RAMLM


def load_wikitext2_tokens(split: str = "validation") -> list[int]:
	from datasets import load_dataset
	ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
	text = "\n\n".join(ds["text"])
	from transformers import GPT2TokenizerFast
	tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
	tokens = tokenizer.encode(text)
	print(f"Loaded {split}: {len(tokens):,} tokens")
	return tokens


def build_cluster_order(train_tokens: list[int], vocab_size: int = 50257) -> list[int]:
	counts = Counter(train_tokens)
	seen = set(t for t, _ in counts.most_common())
	unseen = [t for t in range(vocab_size) if t not in seen]
	return [t for t, _ in counts.most_common()] + unseen


def force_sparse_mode(model):
	layer = model.layer
	if layer._use_sparse and layer._sparse_memory is not None:
		return
	rust_tier_configs = [
		(tc.cluster_end, tc.neurons_per_cluster, tc.bits_per_neuron)
		for tc in layer.tier_configs
	]
	layer._sparse_memory = ram_accelerator.TieredSparseMemory(
		rust_tier_configs, layer.num_clusters
	)
	layer._use_sparse = True


def evaluate_all_modes(
	model, val_tokens: list[int], batch_size: int = 5000,
) -> dict:
	"""Evaluate model with all three normalization modes in a single pass."""
	total_examples = len(val_tokens) - model.context_size
	all_bits = model.encode_sequence(val_tokens)
	targets = torch.tensor(val_tokens[model.context_size:], dtype=torch.long)

	# Accumulators for each mode
	softmax_e0_ce = 0.0  # softmax + EMPTY=0.0 (grid sweep method)
	sumnorm_e0_ce = 0.0  # sum-norm + EMPTY=0.0 (true masking)
	softmax_e5_ce = 0.0  # softmax + EMPTY=0.5 (unmasked)

	softmax_e0_correct = 0
	sumnorm_e0_correct = 0
	softmax_e5_correct = 0

	total_nonzero = 0

	num_batches = (total_examples + batch_size - 1) // batch_size

	for batch_idx in range(num_batches):
		start = batch_idx * batch_size
		end = min(start + batch_size, total_examples)
		batch_bits = all_bits[start:end]
		batch_targets = targets[start:end]
		bi = torch.arange(len(batch_targets))

		# --- Mode 1: Softmax + EMPTY=0.0 (grid sweep method) ---
		ram_accelerator.set_empty_value(0.0)
		scores_e0 = model.forward(batch_bits)  # [batch, vocab_size]

		# Count non-zero clusters (with EMPTY=0.0)
		total_nonzero += (scores_e0 > 1e-9).sum(dim=1).float().sum().item()

		probs_sm0 = softmax(scores_e0, dim=-1)
		tp = probs_sm0[bi, batch_targets].clamp(min=1e-10)
		softmax_e0_ce += -torch.log(tp).sum().item()
		softmax_e0_correct += (probs_sm0.argmax(dim=1) == batch_targets).sum().item()

		# --- Mode 2: Sum-norm + EMPTY=0.0 (true masking) ---
		sums = scores_e0.sum(dim=1, keepdim=True).clamp(min=1e-10)
		probs_sn0 = scores_e0 / sums
		tp = probs_sn0[bi, batch_targets].clamp(min=1e-10)
		sumnorm_e0_ce += -torch.log(tp).sum().item()
		sumnorm_e0_correct += (scores_e0.argmax(dim=1) == batch_targets).sum().item()

		# --- Mode 3: Softmax + EMPTY=0.5 (unmasked) ---
		ram_accelerator.set_empty_value(0.5)
		scores_e5 = model.forward(batch_bits)

		probs_sm5 = softmax(scores_e5, dim=-1)
		tp = probs_sm5[bi, batch_targets].clamp(min=1e-10)
		softmax_e5_ce += -torch.log(tp).sum().item()
		softmax_e5_correct += (probs_sm5.argmax(dim=1) == batch_targets).sum().item()

	ram_accelerator.set_empty_value(0.0)  # restore default

	avg_nonzero = total_nonzero / total_examples

	return {
		"softmax_e0_ce": round(softmax_e0_ce / total_examples, 4),
		"softmax_e0_acc": round(softmax_e0_correct / total_examples, 4),
		"sumnorm_e0_ce": round(sumnorm_e0_ce / total_examples, 4),
		"sumnorm_e0_acc": round(sumnorm_e0_correct / total_examples, 4),
		"softmax_e5_ce": round(softmax_e5_ce / total_examples, 4),
		"softmax_e5_acc": round(softmax_e5_correct / total_examples, 4),
		"avg_nonzero_clusters": round(avg_nonzero, 1),
		"effective_vocab_pct": round(avg_nonzero / model.vocab_size * 100, 2),
	}


def main():
	parser = argparse.ArgumentParser(description="Masked vs Unmasked evaluation")
	parser.add_argument("--input", type=str, default="experiments/tier_bits_grid_fast.json")
	parser.add_argument("--output", type=str, default="experiments/masked_eval_results.json")
	parser.add_argument("--top-n", type=int, default=15)
	parser.add_argument("--train-fraction", type=float, default=0.25)
	parser.add_argument("--top-k", type=int, default=200)
	parser.add_argument("--val-tokens", type=int, default=50000)
	args = parser.parse_args()

	with open(args.input) as f:
		sweep_data = json.load(f)

	results = sweep_data["results"]
	by_ce = sorted(results, key=lambda x: x["avg_ce"])
	by_acc = sorted(results, key=lambda x: -x["avg_accuracy"])

	selected = {}
	for r in by_ce[:args.top_n]:
		selected[r["name"]] = (r["t0_bits"], r["t1_bits"], r["t2_bits"])
	for r in by_acc[:args.top_n]:
		selected[r["name"]] = (r["t0_bits"], r["t1_bits"], r["t2_bits"])
	mid = len(results) // 2
	for r in by_ce[mid - 3 : mid + 3]:
		selected[r["name"]] = (r["t0_bits"], r["t1_bits"], r["t2_bits"])

	configs = list(selected.items())
	configs.sort(key=lambda c: sum(c[1]))

	print("=" * 70)
	print(f"Masked vs Unmasked Evaluation — {len(configs)} configs")
	print("Three modes:")
	print("  1. Softmax + EMPTY=0.0  (grid sweep method)")
	print("  2. Sum-norm + EMPTY=0.0 (TRUE masking)")
	print("  3. Softmax + EMPTY=0.5  (no masking)")
	print("=" * 70)

	train_tokens = load_wikitext2_tokens("train")
	val_tokens = load_wikitext2_tokens("validation")
	cluster_order = build_cluster_order(train_tokens)

	n = int(len(train_tokens) * args.train_fraction)
	train_tokens = train_tokens[:n]
	val_tokens = val_tokens[:args.val_tokens]
	print(f"Using {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")

	all_results = []

	for i, (name, (t0, t1, t2)) in enumerate(configs):
		total = t0 + t1 + t2
		print(f"\n[{i+1}/{len(configs)}] Config {name} (total={total} bits)")

		tiers = [(100, 15, t0), (400, 10, t1), (None, 5, t2)]

		t_start = time.time()
		model = RAMLM(
			vocab_size=50257, context_size=4, tiers=tiers,
			cluster_order=cluster_order, rng=0,
		)
		force_sparse_mode(model)
		model.train_epoch_fast_auto(
			train_tokens, global_top_k=args.top_k, batch_size=5000, verbose=False,
		)
		train_time = time.time() - t_start

		evals = evaluate_all_modes(model, val_tokens)

		result = {
			"name": name,
			"t0_bits": t0, "t1_bits": t1, "t2_bits": t2,
			"total_bits": total,
			"train_time": round(train_time, 1),
			**evals,
		}
		all_results.append(result)

		print(f"  Softmax+E0.0: CE={evals['softmax_e0_ce']:.4f}  Acc={evals['softmax_e0_acc']:.2%}  (grid sweep)")
		print(f"  SumNorm+E0.0: CE={evals['sumnorm_e0_ce']:.4f}  Acc={evals['sumnorm_e0_acc']:.2%}  (true mask)")
		print(f"  Softmax+E0.5: CE={evals['softmax_e5_ce']:.4f}  Acc={evals['softmax_e5_acc']:.2%}  (unmasked)")
		print(f"  EffVocab: {evals['avg_nonzero_clusters']:.0f} tokens ({evals['effective_vocab_pct']:.1f}%)")

		output_path = Path(args.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with open(output_path, "w") as f:
			json.dump({
				"experiment": "masked_vs_unmasked_eval",
				"configs_completed": len(all_results),
				"configs_total": len(configs),
				"results": all_results,
			}, f, indent=2)

	# Final summaries
	print(f"\n{'='*80}")
	print("RANKING COMPARISON (sorted by each metric)")
	print(f"{'='*80}")

	for mode_name, ce_key, acc_key in [
		("Softmax + EMPTY=0.0 (grid sweep)", "softmax_e0_ce", "softmax_e0_acc"),
		("Sum-norm + EMPTY=0.0 (TRUE masking)", "sumnorm_e0_ce", "sumnorm_e0_acc"),
		("Softmax + EMPTY=0.5 (unmasked)", "softmax_e5_ce", "softmax_e5_acc"),
	]:
		print(f"\n--- {mode_name} ---")
		print(f"{'Rank':>4} {'Config':<12} {'Total':>5} {'CE':>10} {'Acc':>8} {'EffVocab':>9}")
		print("-" * 55)
		for rank, r in enumerate(sorted(all_results, key=lambda x: x[ce_key]), 1):
			print(f"{rank:>4} {r['name']:<12} {r['total_bits']:>5} {r[ce_key]:>10.4f} "
				  f"{r[acc_key]:>8.2%} {r['avg_nonzero_clusters']:>7.0f}")

	# Rank correlation
	print(f"\n{'='*80}")
	print("RANK CORRELATION")
	print(f"{'='*80}")
	names_by_sm0 = [r["name"] for r in sorted(all_results, key=lambda x: x["softmax_e0_ce"])]
	names_by_sn0 = [r["name"] for r in sorted(all_results, key=lambda x: x["sumnorm_e0_ce"])]
	names_by_sm5 = [r["name"] for r in sorted(all_results, key=lambda x: x["softmax_e5_ce"])]

	def rank_corr(a, b):
		"""Spearman rank correlation."""
		rank_a = {name: i for i, name in enumerate(a)}
		rank_b = {name: i for i, name in enumerate(b)}
		n = len(a)
		d_sq = sum((rank_a[name] - rank_b[name]) ** 2 for name in a)
		return 1 - 6 * d_sq / (n * (n**2 - 1))

	print(f"  Softmax+E0.0 vs SumNorm+E0.0: rho = {rank_corr(names_by_sm0, names_by_sn0):.4f}")
	print(f"  Softmax+E0.0 vs Softmax+E0.5: rho = {rank_corr(names_by_sm0, names_by_sm5):.4f}")
	print(f"  SumNorm+E0.0 vs Softmax+E0.5: rho = {rank_corr(names_by_sn0, names_by_sm5):.4f}")

	print(f"\nFull results: {args.output}")


if __name__ == "__main__":
	main()
