#!/usr/bin/env python3
"""
WS3: RAM as Embedding Layer Experiment.

Compares:
(a) RAMTransformerLM with RAM features (frozen RAM + trainable projection + transformer)
(b) TinyTransformerLM with standard embeddings (same transformer architecture)

Measures:
- Convergence speed (loss curve over epochs)
- Final CE/PPL/accuracy
- Ablation: RAM projection vs random projection

Usage:
	python tests/test_ram_transformer.py
	python tests/test_ram_transformer.py --epochs 5
	python tests/test_ram_transformer.py --output experiments/ram_transformer_results.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "wnn"))

from wnn.ram.core.models import RAMLM


def load_wikitext2_tokens(split: str = "validation") -> list[int]:
	"""Load WikiText-2 tokens using GPT-2 tokenizer."""
	from datasets import load_dataset
	ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
	text = "\n\n".join(ds["text"])
	from transformers import GPT2TokenizerFast
	tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
	tokens = tokenizer.encode(text)
	print(f"Loaded {split}: {len(tokens):,} tokens")
	return tokens


def create_ram_model(train_tokens: list[int]) -> RAMLM:
	"""Create and train default RAMLM."""
	counts = Counter(train_tokens)
	vocab_size = 50257
	seen_tokens = set(t for t, _ in counts.most_common())
	unseen_tokens = [t for t in range(vocab_size) if t not in seen_tokens]
	cluster_order = [t for t, _ in counts.most_common()] + unseen_tokens

	model = RAMLM(
		vocab_size=vocab_size,
		context_size=4,
		tiers=[(100, 15, 20), (400, 10, 12), (None, 5, 8)],
		cluster_order=cluster_order,
	)
	print(f"\n{model}\n")
	print("Training RAM model...")
	t0 = time.time()
	model.train_epoch_fast_auto(train_tokens, global_top_k=1000, batch_size=5000)
	print(f"RAM training: {time.time() - t0:.1f}s\n")
	return model


def main():
	parser = argparse.ArgumentParser(description="WS3: RAM embedding experiment")
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--output", type=str, default="experiments/ram_transformer_results.json")
	parser.add_argument("--feature-dim", type=int, default=256)
	parser.add_argument("--seq-len", type=int, default=32)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--model-path", type=str, help="Path to saved RAMLM")
	args = parser.parse_args()

	print("=" * 60)
	print("WS3: RAM as Embedding Layer Experiment")
	print("=" * 60)

	# Load data
	train_tokens = load_wikitext2_tokens("train")
	val_tokens = load_wikitext2_tokens("validation")

	# Create RAM model
	if args.model_path:
		print(f"Loading RAM from {args.model_path}...")
		ram = RAMLM.load(args.model_path)
	else:
		ram = create_ram_model(train_tokens)

	# --- Model A: RAMTransformerLM (RAM features) ---
	print("\n--- Model A: RAMTransformerLM (RAM features) ---")
	from wnn.ram.core.models.ram_transformer_lm import RAMTransformerLM

	model_a = RAMTransformerLM(
		ram,
		feature_dim=args.feature_dim,
		n_heads=4,
		n_layers=4,
		d_ff=512,
		max_seq_len=128,
	)
	print(f"  Trainable parameters: {model_a.num_parameters():,}")

	a_curves = []
	for epoch in range(args.epochs):
		print(f"\n  Epoch {epoch + 1}/{args.epochs}:")
		t0 = time.time()
		train_stats = model_a.train_epoch(
			train_tokens,
			batch_size=args.batch_size,
			seq_len=args.seq_len,
			lr=1e-4,
		)
		train_time = time.time() - t0

		eval_stats = model_a.evaluate(val_tokens, batch_size=args.batch_size, seq_len=args.seq_len)
		a_curves.append({
			"epoch": epoch + 1,
			"train_loss": train_stats["avg_loss"],
			"val_ce": eval_stats["cross_entropy"],
			"val_ppl": eval_stats["perplexity"],
			"val_acc": eval_stats["accuracy"],
			"train_time_s": round(train_time, 1),
		})

	# --- Model B: TinyTransformerLM (standard embeddings) ---
	print("\n\n--- Model B: TinyTransformerLM (standard embeddings) ---")
	from wnn.ram.core.models.tiny_transformer import TinyTransformerLM

	model_b = TinyTransformerLM(
		vocab_size=50257,
		d_model=args.feature_dim,
		n_heads=4,
		n_layers=4,
		d_ff=512,
		max_context=128,
	)
	print(f"  Trainable parameters: {model_b.num_parameters():,}")

	b_curves = []
	for epoch in range(args.epochs):
		print(f"\n  Epoch {epoch + 1}/{args.epochs}:")
		t0 = time.time()
		train_stats = model_b.train_epoch(
			train_tokens,
			batch_size=args.batch_size,
			seq_len=args.seq_len,
			lr=1e-4,
		)
		train_time = time.time() - t0

		eval_stats = model_b.evaluate(val_tokens, batch_size=args.batch_size, seq_len=args.seq_len)
		b_curves.append({
			"epoch": epoch + 1,
			"train_loss": train_stats["avg_loss"],
			"val_ce": eval_stats["cross_entropy"],
			"val_ppl": eval_stats["perplexity"],
			"val_acc": eval_stats["accuracy"],
			"train_time_s": round(train_time, 1),
		})

	# --- Comparison ---
	print("\n" + "=" * 80)
	print("COMPARISON: RAM Features vs Standard Embeddings")
	print("=" * 80)
	print(f"\n{'Epoch':<8} {'RAM CE':>10} {'Std CE':>10} {'RAM PPL':>12} {'Std PPL':>12} {'RAM Acc':>10} {'Std Acc':>10}")
	print("-" * 72)

	for a, b in zip(a_curves, b_curves):
		print(f"{a['epoch']:<8} {a['val_ce']:>10.4f} {b['val_ce']:>10.4f} "
			  f"{a['val_ppl']:>12.2f} {b['val_ppl']:>12.2f} "
			  f"{a['val_acc']:>10.2%} {b['val_acc']:>10.2%}")

	# Final comparison
	final_a = a_curves[-1]
	final_b = b_curves[-1]
	ce_diff = final_a['val_ce'] - final_b['val_ce']
	print(f"\nFinal CE difference (A - B): {ce_diff:+.4f}")
	if ce_diff < 0:
		print("  -> RAM features converge to BETTER CE")
	elif ce_diff > 0:
		print("  -> Standard embeddings achieve BETTER CE")
	else:
		print("  -> Both are equivalent")

	# Check convergence speed
	if len(a_curves) > 1:
		a_improvement = a_curves[0]['val_ce'] - a_curves[1]['val_ce']
		b_improvement = b_curves[0]['val_ce'] - b_curves[1]['val_ce']
		print(f"\nEpoch 1->2 CE improvement:")
		print(f"  RAM features: {a_improvement:.4f}")
		print(f"  Standard:     {b_improvement:.4f}")

	# Save results
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	output = {
		"experiment": "WS3_ram_as_embedding",
		"config": {
			"feature_dim": args.feature_dim,
			"n_heads": 4,
			"n_layers": 4,
			"epochs": args.epochs,
			"seq_len": args.seq_len,
			"batch_size": args.batch_size,
		},
		"model_a_ram_features": {
			"trainable_params": model_a.num_parameters(),
			"convergence_curve": a_curves,
			"final": a_curves[-1],
		},
		"model_b_standard_embeddings": {
			"trainable_params": model_b.num_parameters(),
			"convergence_curve": b_curves,
			"final": b_curves[-1],
		},
		"comparison": {
			"final_ce_diff": round(ce_diff, 4),
			"ram_features_better": ce_diff < 0,
		},
	}

	with open(output_path, "w") as f:
		json.dump(output, f, indent=2)
	print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
	main()
