"""
End-to-end smoke for the WNN_GPU_BATCHED_TRAIN=1 wired path.

Calls ram_accelerator.evaluate_genomes_parallel_hybrid with synthetic
single-cluster (IDS-style) inputs at varying scales to confirm:
1. Option B path produces the same fitness as the existing CPU+GPU path.
2. Wall time is no longer pathologically slow (was 62s for e=50K before fix).

Each scenario runs twice: once with WNN_GPU_BATCHED_TRAIN unset (baseline) and once
with WNN_GPU_BATCHED_TRAIN=1 (Option B). Reports timing + fitness for both.
"""
import os
import sys
import time
import numpy as np

# Make ram_accelerator importable
import ram_accelerator as ra


def make_scenario(num_genomes, num_train, num_eval, num_neurons, bits_per_neuron,
                  total_input_bits=768, seed=12345):
    """Build inputs for evaluate_genomes_parallel_hybrid.

    Single-cluster, uniform per-neuron bits, random connections, random data.
    Returns the kwargs dict ready to splat.
    """
    rng = np.random.default_rng(seed)

    # Per-genome flat layouts
    # genomes_bits_flat: total_neurons × num_genomes entries
    total_neurons = num_neurons * num_genomes
    genomes_bits_flat = np.full(total_neurons, bits_per_neuron, dtype=np.uint64).tolist()
    # genomes_neurons_flat: num_genomes entries (single cluster)
    genomes_neurons_flat = np.full(num_genomes, num_neurons, dtype=np.uint64).tolist()
    # genomes_connections_flat: num_genomes × num_neurons × bits_per_neuron i64
    conns_per_genome = num_neurons * bits_per_neuron
    total_conns = num_genomes * conns_per_genome
    genomes_connections_flat = rng.integers(0, total_input_bits, size=total_conns, dtype=np.int64).tolist()

    # Data: packed_bits-like uint8 (one byte per bit; total_input_bits per example)
    # The Rust side uses PackedBits::from_bool_bytes — each byte is a bit (1 or 0).
    train_bits_per_example = total_input_bits
    train_input_bits = (rng.integers(0, 2, size=num_train * train_bits_per_example, dtype=np.uint8))
    train_targets = (rng.integers(0, 5, size=num_train, dtype=np.int64) == 0).astype(np.int64)  # ~20% positive class
    train_negatives = np.zeros(1, dtype=np.int64)

    eval_input_bits = (rng.integers(0, 2, size=num_eval * train_bits_per_example, dtype=np.uint8))
    eval_targets = (rng.integers(0, 5, size=num_eval, dtype=np.int64) == 0).astype(np.int64)

    return {
        "genomes_bits_flat": genomes_bits_flat,
        "genomes_neurons_flat": genomes_neurons_flat,
        "genomes_connections_flat": genomes_connections_flat,
        "num_genomes": num_genomes,
        "num_clusters": 1,
        "train_input_bits": train_input_bits,
        "train_targets": train_targets,
        "train_negatives": train_negatives,
        "num_train": num_train,
        "num_negatives": 0,
        "eval_input_bits": eval_input_bits,
        "eval_targets": eval_targets,
        "num_eval": num_eval,
        "total_input_bits": total_input_bits,
        "empty_value": 0.5,
        "neuron_sample_rate": 1.0,
        "rng_seed": 42,
    }


def run_once(kwargs, option_b: bool, label: str):
    if option_b:
        os.environ["WNN_GPU_BATCHED_TRAIN"] = "1"
        os.environ["WNN_GPU_BATCHED_TRAIN_TRACE"] = "1"
    else:
        os.environ.pop("WNN_GPU_BATCHED_TRAIN", None)
        os.environ.pop("WNN_GPU_BATCHED_TRAIN_TRACE", None)
    t0 = time.time()
    try:
        result = ra.evaluate_genomes_parallel_hybrid(**kwargs)
        elapsed = time.time() - t0
        print(f"  [{label}] wall={elapsed:.2f}s; fitness={result}")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{label}] EXCEPTION after {elapsed:.2f}s: {e}")
        return None, elapsed


def main():
    scenarios = [
        # (num_genomes, num_train, num_eval, label)
        (1,  5_000,  1_000,  "ng=1 e=5K"),
        (1,  50_000, 10_000, "ng=1 e=50K"),
        (1,  200_000, 40_000, "ng=1 e=200K"),
        # Skipping 500K and 1.5M for the initial smoke — those would take a
        # while and the pattern at 50K + 200K is enough to confirm the fix.
        (4,  50_000, 10_000, "ng=4 e=50K"),
        (16, 50_000, 10_000, "ng=16 e=50K"),
    ]
    n_per_g = 100
    b = 48

    for (ng, ne_train, ne_eval, label) in scenarios:
        print(f"\n=== Scenario: {label} (n_per_g={n_per_g}, b={b}) ===")
        kwargs = make_scenario(ng, ne_train, ne_eval, n_per_g, b)
        # Run baseline first
        baseline, t_base = run_once(kwargs, option_b=False, label="Baseline")
        # Then Option B
        optb, t_optb = run_once(kwargs, option_b=True, label="OptionB ")
        # Compare
        if baseline is not None and optb is not None:
            speedup = t_base / max(t_optb, 1e-3)
            # Fitness is list of tuples (ce, acc, f1, fpr, ...); compare first two
            ce_diff = abs(baseline[0][0] - optb[0][0]) if baseline and optb else float("nan")
            acc_diff = abs(baseline[0][1] - optb[0][1]) if baseline and optb else float("nan")
            print(f"  Speedup: {speedup:.2f}x | ΔCE={ce_diff:.4f} | ΔAcc={acc_diff:.4f}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
