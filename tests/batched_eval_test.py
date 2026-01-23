#!/usr/bin/env python3
"""
Batched Evaluation Test - Compares sequential vs batched GPU evaluation.

This test uses UNIFORM config (same bits/neurons for all clusters) to enable
true batching where multiple genomes are evaluated in a single GPU dispatch.

Run with: python tests/batched_eval_test.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'wnn'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genomes', type=int, default=10)
    parser.add_argument('--train-tokens', type=int, default=500000)
    parser.add_argument('--eval-tokens', type=int, default=50000)
    parser.add_argument('--bits', type=int, default=15)
    parser.add_argument('--neurons', type=int, default=10)
    args = parser.parse_args()

    # Enable timing for debugging
    os.environ['WNN_GROUP_TIMING'] = '1'
    os.environ['WNN_EVAL_TIMING'] = '1'

    print("=" * 70)
    print("Batched vs Sequential Evaluation Test")
    print("=" * 70)
    print(f"Config: uniform {args.bits} bits, {args.neurons} neurons per cluster")
    print()

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

    # UNIFORM config - all clusters have same (neurons, bits)
    # This enables true batching!
    tier_config = [(vocab_size, args.neurons, args.bits)]

    # Generate genomes
    import random
    rng = random.Random(42)

    genomes_bits = []
    genomes_neurons = []
    genomes_connections = []

    for _ in range(args.genomes):
        bits = [args.bits] * vocab_size
        neurons = [args.neurons] * vocab_size
        total_conns = args.bits * args.neurons * vocab_size
        conns = [rng.randint(0, total_input_bits - 1) for _ in range(total_conns)]
        genomes_bits.extend(bits)
        genomes_neurons.extend(neurons)
        genomes_connections.extend(conns)

    total_neurons = args.neurons * vocab_size
    total_conns_per_genome = args.bits * total_neurons
    print(f"\nGenomes: {args.genomes}")
    print(f"Total neurons per genome: {total_neurons:,}")
    print(f"Connections per genome: {total_conns_per_genome:,}")
    print()

    # Run evaluation with sequential processing (batch_size=1)
    print("=" * 70)
    print("Sequential evaluation (batch_size=1)...")
    print("=" * 70)
    os.environ['WNN_BATCH_SIZE'] = '1'

    start = time.perf_counter()
    results_seq = cache.evaluate_genomes_hybrid(
        genomes_bits_flat=genomes_bits,
        genomes_neurons_flat=genomes_neurons,
        genomes_connections_flat=genomes_connections,
        num_genomes=args.genomes,
        train_subset_idx=0,
        eval_subset_idx=0,
        empty_value=0.5,
    )
    elapsed_seq = time.perf_counter() - start
    print(f"\nSequential total: {elapsed_seq:.2f}s ({elapsed_seq/args.genomes:.2f}s per genome)")
    print()

    # Run with larger batch sizes to see if parallel training helps
    for batch_size in [2, 5, 10]:
        if batch_size > args.genomes:
            continue
        print(f"Batch size {batch_size}...")
        os.environ['WNN_BATCH_SIZE'] = str(batch_size)

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
        speedup = elapsed_seq / elapsed if elapsed > 0 else 0
        print(f"Batch {batch_size}: {elapsed:.2f}s ({elapsed/args.genomes:.2f}s per genome) - {speedup:.2f}x vs sequential")
        print()

    print("=" * 70)
    print("Results (first 3 genomes):")
    for i, (ce, acc) in enumerate(results_seq[:3]):
        print(f"  Genome {i}: CE={ce:.4f}, Acc={acc*100:.2f}%")

    print()
    print("Analysis:")
    print("- With UNIFORM config, all clusters use same (neurons, bits)")
    print("- This means a single GPU kernel call per genome (vs 3 for tiered)")
    print("- Batching at training level helps with parallel training")
    print("- True GPU batching (multiple genomes in one dispatch) could help more")


if __name__ == '__main__':
    main()
