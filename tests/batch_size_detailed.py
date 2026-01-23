#!/usr/bin/env python3
"""
Detailed batch size investigation - measures training vs evaluation time separately.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'wnn'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genomes', type=int, default=10)
    parser.add_argument('--batch-sizes', type=str, default='1,2,5,10')
    parser.add_argument('--train-tokens', type=int, default=500000)
    parser.add_argument('--eval-tokens', type=int, default=50000)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    print("=" * 70)
    print("Detailed Batch Size Investigation")
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

    # Tier config
    tier_config = [(100, 15, 20), (400, 10, 12), (vocab_size - 500, 5, 8)]

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

    # Test each batch size with detailed timing
    print("=" * 70)
    print(f"{'Batch':<8} {'Total':>10} {'Per-Gen':>10} {'CPU%':>8} {'Notes'}")
    print("=" * 70)

    for batch_size in batch_sizes:
        os.environ['WNN_BATCH_SIZE'] = str(batch_size)

        # Force fresh start
        import gc
        gc.collect()

        # Time the full evaluation
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

        per_genome = elapsed / args.genomes

        # Calculate effective parallelism
        # With batch_size=N, we expect N genomes processed "together"
        # If per-genome time increases with batch_size, there's contention
        num_batches = (args.genomes + batch_size - 1) // batch_size

        notes = ""
        if batch_size == 1:
            baseline_per_genome = per_genome
        else:
            slowdown = per_genome / baseline_per_genome
            if slowdown > 1.1:
                notes = f"({slowdown:.1f}x slower than batch=1)"

        print(f"{batch_size:<8} {elapsed:>10.2f}s {per_genome:>10.2f}s {'-':>8} {notes}")

    print("=" * 70)
    print()
    print("Analysis:")
    print("- If larger batches are slower, there's likely memory/cache contention")
    print("- The parallel training (rayon) may be competing for memory bandwidth")
    print("- DashMap (sparse memory) may have lock contention with many writers")
    print()

    # Now let's test with RUST_LOG to see internal timing
    print("To investigate further, we could:")
    print("1. Add timing instrumentation to Rust code (train vs eval)")
    print("2. Monitor memory bandwidth during execution")
    print("3. Profile with perf/instruments")

if __name__ == '__main__':
    main()
