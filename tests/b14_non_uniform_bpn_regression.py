"""
B14 regression test: non-uniform per-neuron bits within same (n, max_bits) shape.

Bug history: B14 (15/05/2026) used shape key (neurons_per_cluster, max_bits)
to group heterogeneous batches. But two genomes with same n and same max_b
can have different individual bpn arrays. batched_train_offspring builds
connections at sum(bpn) length, but ConfigGroup.conn_size() returns
total_neurons × max_bits — a mismatch panics in evaluate_group_sparse_gpu
with "range end index N out of range for slice of length M".

Fix: shape key now includes the full bpn vector, so genomes with
divergent bpn signatures land in separate groups.

This test reproduces the buggy regime and verifies the fix.
"""
import os
import numpy as np
import ram_accelerator as ra


def make_mixed_bpn_batch(seed=42):
    """3 genomes, all n=50 + max_b=48, but with different bpn signatures.
    Same shape key under buggy code; different signatures under fixed code."""
    rng = np.random.default_rng(seed)
    ib = 1920
    bits_flat: list = []
    bits_flat.extend([32] + [48] * 49)        # genome A
    bits_flat.extend([48] * 50)               # genome B
    bits_flat.extend([32, 32] + [48] * 48)    # genome C
    return {
        "genomes_bits_flat": bits_flat,
        "genomes_neurons_flat": [50, 50, 50],
        "genomes_connections_flat": rng.integers(0, ib, size=sum(bits_flat), dtype=np.int64).tolist(),
        "num_genomes": 3,
        "num_clusters": 1,
        "train_input_bits": rng.integers(0, 2, size=50_000 * ib, dtype=np.uint8),
        "train_targets": (rng.random(50_000) < 0.15).astype(np.int64),
        "train_negatives": np.zeros(1, dtype=np.int64),
        "num_train": 50_000,
        "num_negatives": 0,
        "eval_input_bits": rng.integers(0, 2, size=10_000 * ib, dtype=np.uint8),
        "eval_targets": (rng.random(10_000) < 0.15).astype(np.int64),
        "num_eval": 10_000,
        "total_input_bits": ib,
        "empty_value": 0.5,
        "neuron_sample_rate": 0.25,
        "rng_seed": 42,
    }


def main() -> None:
    kw = make_mixed_bpn_batch()
    os.environ["WNN_GPU_BATCHED_TRAIN"] = "off"
    cpu = ra.evaluate_genomes_parallel_hybrid(**kw)
    os.environ.pop("WNN_GPU_BATCHED_TRAIN", None)
    default = ra.evaluate_genomes_parallel_hybrid(**kw)
    max_dce = max(abs(c[0] - d[0]) for c, d in zip(cpu, default))
    print(f"max|dCE| (CPU vs B14) = {max_dce:.4f}")
    assert max_dce < 1e-3, f"PARITY FAIL: dCE={max_dce}"
    print("PASS")


if __name__ == "__main__":
    main()
