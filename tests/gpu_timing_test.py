#!/usr/bin/env python3
"""
GPU Timing Test - Measures where time is spent during hybrid evaluation.

Run with: WNN_GROUP_TIMING=1 python tests/gpu_timing_test.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'wnn'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genomes', type=int, default=5)
    parser.add_argument('--train-tokens', type=int, default=500000)
    parser.add_argument('--eval-tokens', type=int, default=50000)
    args = parser.parse_args()

    # Enable detailed timing
    os.environ['WNN_GROUP_TIMING'] = '1'
    os.environ['WNN_EVAL_TIMING'] = '1'

    print("=" * 70)
    print("GPU Timing Investigation")
    print("=" * 70)

    # Load data
    print("Loading data...")
    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(dataset["train"]["text"])
    eval_text = "\n".join(dataset["validation"]["text"])

    train_tokens = enc.encode(train_text)[:args.train_tokens]
    eval_tokens = enc.encode(eval_text)[:args.eval_tokens]
    vocab_size = enc.n_vocab

    print(f"Train: {len(train_tokens):,} tokens, Eval: {len(eval_tokens):,} tokens")

    # Create cache
    import ram_accelerator
    cluster_order = list(range(vocab_size))
    cache = ram_accelerator.TokenCacheWrapper(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        test_tokens=eval_tokens,
        vocab_size=vocab_size,
        context_size=4,
        cluster_order=cluster_order,
        num_parts=4,
        num_negatives=5,
        seed=42,
    )
    total_input_bits = cache.total_input_bits()
    print(f"Total input bits: {total_input_bits}")

    # Tier config - all sparse (bits > 12) to test batched sparse
    tier_config = [(100, 15, 20), (400, 10, 15), (vocab_size - 500, 5, 13)]

    # Generate genomes
    import random
    rng = random.Random(42)

    genomes_bits = []
    genomes_neurons = []
    genomes_connections = []

    for _ in range(args.genomes):
        bits = []
        neurons = []
        for clusters, n, b in tier_config:
            bits.extend([b] * clusters)
            neurons.extend([n] * clusters)
        total_conns = sum(b * n for b, n in zip(bits, neurons))
        conns = [rng.randint(0, total_input_bits - 1) for _ in range(total_conns)]
        genomes_bits.extend(bits)
        genomes_neurons.extend(neurons)
        genomes_connections.extend(conns)

    print(f"\nGenomes: {args.genomes}")
    print(f"Connections per genome: {len(genomes_connections) // args.genomes:,}")
    print()

    # Run evaluation with detailed timing
    print("=" * 70)
    print("Running evaluation with timing enabled...")
    print("(Look for [EVAL_WORKER] and [EVAL_HYBRID] messages)")
    print("=" * 70)
    print()

    start = time.perf_counter()
    results = cache.evaluate_genomes_hybrid(
        genomes_bits_flat=genomes_bits,
        genomes_neurons_flat=genomes_neurons,
        genomes_connections_flat=genomes_connections,
        num_genomes=args.genomes,
        train_subset_idx=0,
        eval_subset_idx=0,
        empty_value=0.5,
    )
    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print(f"Total time: {elapsed:.2f}s ({elapsed/args.genomes:.2f}s per genome)")
    print("=" * 70)
    print()

    print("Results (first 3 genomes):")
    for i, (ce, acc) in enumerate(results[:3]):
        print(f"  Genome {i}: CE={ce:.4f}, Acc={acc*100:.2f}%")

    print()
    print("Analysis points:")
    print("- GPU time: Metal kernel execution (sparse or dense)")
    print("- CPU time: Fallback when GPU unavailable")
    print("- Scatter time: Copying results to all_scores array")
    print()
    print("If GPU time >> scatter time, batching multiple genomes")
    print("in one kernel could help by amortizing kernel launch overhead.")
    print()
    print("If scatter time is significant, we need a different approach")
    print("(e.g., keeping results in GPU memory between groups).")


if __name__ == '__main__':
    main()
