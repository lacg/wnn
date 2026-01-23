#!/usr/bin/env python3
"""
Quick batch size performance test for the persistent eval worker.

Tests different batch sizes and measures:
- Time per genome
- Memory usage
- Throughput

Usage:
    python tests/batch_size_test.py [--genomes N] [--batch-sizes 10,25,50]
"""

import argparse
import os
import sys
import time
import psutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'wnn'))

def get_memory_gb():
    """Get current process memory in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def run_batch_test(num_genomes: int, batch_size: int, cache, tier_config: list, total_input_bits: int):
    """Run evaluation with specific batch size and return timing."""
    import random
    import ram_accelerator

    # Set batch size via env var
    os.environ['WNN_BATCH_SIZE'] = str(batch_size)

    # Generate random genomes with tiered config
    vocab_size = sum(t[0] for t in tier_config)
    genomes_bits = []
    genomes_neurons = []
    genomes_connections = []

    rng = random.Random(42)
    for _ in range(num_genomes):
        bits = []
        neurons = []
        for clusters, default_neurons, default_bits in tier_config:
            bits.extend([default_bits] * clusters)
            neurons.extend([default_neurons] * clusters)

        # Random connections
        total_conns = sum(b * n for b, n in zip(bits, neurons))
        conns = [rng.randint(0, total_input_bits - 1) for _ in range(total_conns)]

        genomes_bits.extend(bits)
        genomes_neurons.extend(neurons)
        genomes_connections.extend(conns)

    # Measure evaluation time
    mem_before = get_memory_gb()
    start = time.perf_counter()

    results = cache.evaluate_genomes_hybrid(
        genomes_bits_flat=genomes_bits,
        genomes_neurons_flat=genomes_neurons,
        genomes_connections_flat=genomes_connections,
        num_genomes=num_genomes,
        train_subset_idx=0,
        eval_subset_idx=0,
        empty_value=0.5,
    )

    elapsed = time.perf_counter() - start
    mem_after = get_memory_gb()

    return {
        'batch_size': batch_size,
        'num_genomes': num_genomes,
        'total_time': elapsed,
        'time_per_genome': elapsed / num_genomes,
        'mem_before_gb': mem_before,
        'mem_after_gb': mem_after,
        'mem_delta_gb': mem_after - mem_before,
    }

def main():
    parser = argparse.ArgumentParser(description='Batch size performance test')
    parser.add_argument('--genomes', type=int, default=10, help='Number of genomes to evaluate')
    parser.add_argument('--batch-sizes', type=str, default='10,25,50', help='Comma-separated batch sizes to test')
    parser.add_argument('--context', type=int, default=4, help='Context size')
    parser.add_argument('--token-parts', type=int, default=4, help='Token parts for encoding')
    parser.add_argument('--tier-config', type=str, default='100,15,20;400,10,12;rest,5,8',
                        help='Tier config: clusters,neurons,bits;...')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    print("=" * 60)
    print("Batch Size Performance Test")
    print("=" * 60)
    print(f"Genomes to evaluate: {args.genomes}")
    print(f"Batch sizes to test: {batch_sizes}")
    print(f"Tier config: {args.tier_config}")
    print()

    # Load data and create cache
    print("Loading data...")
    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

    train_text = "\n".join(dataset["train"]["text"])
    eval_text = "\n".join(dataset["validation"]["text"])

    # Use configurable data size
    max_train = int(os.environ.get('TEST_TRAIN_TOKENS', '50000'))
    max_eval = int(os.environ.get('TEST_EVAL_TOKENS', '10000'))
    train_tokens = enc.encode(train_text)[:max_train]
    eval_tokens = enc.encode(eval_text)[:max_eval]
    vocab_size = enc.n_vocab

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Eval tokens: {len(eval_tokens):,}")
    print(f"Vocab size: {vocab_size:,}")
    print()

    # Parse tier config
    tier_config = []
    remaining_clusters = vocab_size
    for part in args.tier_config.split(';'):
        clusters_str, neurons, bits = part.split(',')
        if clusters_str == 'rest':
            clusters = remaining_clusters
        else:
            clusters = int(clusters_str)
        tier_config.append((clusters, int(neurons), int(bits)))
        remaining_clusters -= clusters

    total_clusters = sum(t[0] for t in tier_config)
    print(f"Tier config parsed: {tier_config}")
    print(f"Total clusters: {total_clusters}")
    print()

    # Create token cache
    print("Creating token cache...")
    import ram_accelerator

    # Create cluster order (identity mapping)
    cluster_order = list(range(vocab_size))

    cache = ram_accelerator.TokenCacheWrapper(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        test_tokens=eval_tokens,  # Use eval as test for quick test
        vocab_size=vocab_size,
        context_size=args.context,
        cluster_order=cluster_order,
        num_parts=args.token_parts,
        num_negatives=5,
        seed=42,
    )

    # Get actual total_input_bits from cache (computed as context_size * ceil(log2(vocab_size)))
    total_input_bits = cache.total_input_bits()
    print(f"Total input bits: {total_input_bits}")
    print()

    # Run tests
    print("=" * 60)
    print("Running batch size tests...")
    print("=" * 60)

    results = []
    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size}...")

        # Clear any cached state
        import gc
        gc.collect()

        result = run_batch_test(
            num_genomes=args.genomes,
            batch_size=batch_size,
            cache=cache,
            tier_config=tier_config,
            total_input_bits=total_input_bits,
        )
        results.append(result)

        print(f"  Time: {result['total_time']:.2f}s total, {result['time_per_genome']:.2f}s/genome")
        print(f"  Memory: {result['mem_before_gb']:.1f}GB -> {result['mem_after_gb']:.1f}GB (delta: {result['mem_delta_gb']:+.1f}GB)")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Batch Size':>12} | {'Total Time':>12} | {'Per Genome':>12} | {'Mem Delta':>12}")
    print("-" * 60)

    best = min(results, key=lambda r: r['time_per_genome'])
    for r in results:
        marker = " <-- BEST" if r == best else ""
        print(f"{r['batch_size']:>12} | {r['total_time']:>10.2f}s | {r['time_per_genome']:>10.2f}s | {r['mem_delta_gb']:>+10.1f}GB{marker}")

    print()
    print(f"Recommended batch size: {best['batch_size']} ({best['time_per_genome']:.2f}s/genome)")

    return 0

if __name__ == '__main__':
    sys.exit(main())
