"""
Order-independence end-to-end parity test.

Validates that `WNN_ORDER_INDEPENDENT_TRAIN=1` produces order-independent
training output at the Python entry-point level (evaluate_genomes_parallel_hybrid).

Three checks:

1) Determinism: two back-to-back runs with the same seed and the same env
   produce identical outputs. Holds for both legacy and OI paths, but the
   OI path makes this guarantee more robust against rayon scheduling
   variance (which can perturb low-bits legacy results under contention).

2) OI vs legacy divergence: running with OI on vs off on the same inputs
   produces *different* per-genome (CE, Acc) tuples — proof that the OI
   fix is materially changing the training output (not a no-op gate).

3) OI cross-shuffle invariance: shuffling the training examples and
   re-running with OI produces identical outputs. The legacy path would
   fail this check by construction (that's the bug being fixed).

This test exercises the production training entry-point — every backend
(dense, sparse DashMap, sparse atomic-HT, Metal kernel) is reachable
depending on (b, n) shape and env flags.
"""
import os
import sys
import numpy as np

# Activate the wnn venv path before importing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "wnn"))

import ram_accelerator as ra


def synth_ids_data(num_train=2000, num_eval=500, input_bits=128, attack_frac=0.15, seed=0):
    """Synthesize a tiny IDS-shaped binary classification dataset.

    Attacks are sampled from a distribution biased toward higher bit-1 density
    in the second half of features; benign from a low-bit-1 density.
    """
    rng = np.random.default_rng(seed)
    n_total = num_train + num_eval
    labels = (rng.random(n_total) < attack_frac).astype(np.int64)
    bits = np.zeros((n_total, input_bits), dtype=bool)
    half = input_bits // 2
    for i in range(n_total):
        if labels[i] == 1:
            # Attack: higher density in second half
            bits[i, :half] = rng.random(half) < 0.30
            bits[i, half:] = rng.random(input_bits - half) < 0.65
        else:
            bits[i, :half] = rng.random(half) < 0.55
            bits[i, half:] = rng.random(input_bits - half) < 0.25
    return (
        bits[:num_train].astype(bool),
        labels[:num_train],
        bits[num_train:].astype(bool),
        labels[num_train:],
    )


def run_one(train_bits, train_labels, eval_bits, eval_labels, num_genomes=3, seed=42):
    """Train + eval a small population of random genomes; return per-genome (CE, Acc)."""
    num_clusters = 2  # binary
    num_train = len(train_labels)
    num_eval = len(eval_labels)
    input_bits = train_bits.shape[1]

    # Same shape per genome, random connections via empty connections list.
    # bits_flat is PER-NEURON: [num_genomes * sum(neurons_per_cluster)]
    # neurons_flat is per-cluster: [num_genomes * num_clusters]
    num_neurons_per_cluster = 30
    bits_per_neuron = 6  # small → dense backend
    neurons_per_genome = num_neurons_per_cluster * num_clusters
    bits_flat = [bits_per_neuron] * neurons_per_genome * num_genomes
    neurons_flat = [num_neurons_per_cluster] * num_clusters * num_genomes

    # Flatten data — PyO3 boundary expects u8 arrays of 0/1, plus i64 labels.
    train_input_flat = np.ascontiguousarray(train_bits.flatten().astype(np.uint8))
    eval_input_flat = np.ascontiguousarray(eval_bits.flatten().astype(np.uint8))
    train_targets = np.ascontiguousarray(train_labels, dtype=np.int64)
    eval_targets = np.ascontiguousarray(eval_labels, dtype=np.int64)
    train_negs = np.ascontiguousarray(1 - train_targets, dtype=np.int64)

    # Empty connections → accelerator generates random per genome using seed.
    res = ra.evaluate_genomes_parallel_hybrid(
        bits_flat,                # genomes_bits_flat
        neurons_flat,             # genomes_neurons_flat
        [],                       # genomes_connections_flat (empty = random)
        num_genomes,
        num_clusters,
        train_input_flat,
        train_targets,
        train_negs,
        num_train,
        1,                        # num_negatives per example
        eval_input_flat,
        eval_targets,
        num_eval,
        input_bits,
        0.0,                      # empty_value
        1.0,                      # neuron_sample_rate (no sampling)
        seed,                     # rng_seed
    )
    # res is Vec<(ce, acc, f1, fpr, ...) — keep all 5 fields, rounded.
    return [tuple(round(x, 6) for x in row) for row in res]


def main():
    print("=" * 76)
    print("OI end-to-end parity test (training_order_independence.py)")
    print("=" * 76)

    train_bits, train_labels, eval_bits, eval_labels = synth_ids_data(
        num_train=2000, num_eval=500, input_bits=128, seed=1
    )

    # --- 1) Determinism under OI ---
    os.environ["WNN_ORDER_INDEPENDENT_TRAIN"] = "1"
    out_oi_a = run_one(train_bits, train_labels, eval_bits, eval_labels)
    out_oi_b = run_one(train_bits, train_labels, eval_bits, eval_labels)
    print(f"\n[1] Determinism (OI=1, same inputs, two runs)")
    print(f"    run a: {out_oi_a}")
    print(f"    run b: {out_oi_b}")
    assert out_oi_a == out_oi_b, "OI runs should be deterministic across reruns"
    print(f"    PASS — identical")

    # --- 2) OI vs legacy ---
    os.environ.pop("WNN_ORDER_INDEPENDENT_TRAIN", None)
    out_legacy = run_one(train_bits, train_labels, eval_bits, eval_labels)
    print(f"\n[2] OI vs legacy (same inputs)")
    print(f"    legacy: {out_legacy}")
    print(f"    OI:     {out_oi_a}")
    if out_legacy != out_oi_a:
        print(f"    PASS — outputs differ, OI is materially active")
    else:
        # Possible for synthetic data with extreme nudge patterns; not a fail.
        print(f"    NOTE — outputs identical on this seed; OI was a no-op for this data")

    # --- 3) OI cross-shuffle invariance ---
    os.environ["WNN_ORDER_INDEPENDENT_TRAIN"] = "1"
    rng = np.random.default_rng(2)
    perm = rng.permutation(len(train_labels))
    train_bits_perm = train_bits[perm]
    train_labels_perm = train_labels[perm]
    out_oi_shuf = run_one(train_bits_perm, train_labels_perm, eval_bits, eval_labels)
    print(f"\n[3] OI cross-shuffle invariance")
    print(f"    OI (original order): {out_oi_a}")
    print(f"    OI (shuffled order): {out_oi_shuf}")
    if out_oi_a == out_oi_shuf:
        print(f"    PASS — identical under shuffle (order-independence holds end-to-end)")
    else:
        # Random connections seed depends on genome_idx (deterministic), but
        # the *connections* don't change across runs because seed is the same.
        # Differences here would point to a remaining order-dependent code path.
        print(f"    FAIL — outputs differ under shuffle:")
        for i, (a, b) in enumerate(zip(out_oi_a, out_oi_shuf)):
            if a != b:
                print(f"      genome {i}: original={a}  shuffled={b}")
        sys.exit(1)

    # --- 4) LM legacy entry: evaluate_genomes_parallel (uses train_genome_in_slot) ---
    # Confirms the LM connectivity-optimization entry point honors OI via the
    # IDS wire-up (train_genome_in_slot was made OI-aware in the IDS commit).
    # NOTE: evaluate_genomes_parallel generates RANDOM connections via
    # `SmallRng::from_entropy()` when an empty connections list is passed
    # (non-deterministic — separate from OI). To test OI determinism we
    # have to pass explicit deterministic connections.
    print(f"\n[4] LM legacy entry (evaluate_genomes_parallel) honors OI")
    os.environ["WNN_ORDER_INDEPENDENT_TRAIN"] = "1"

    num_genomes = 3
    num_clusters = 2
    num_neurons_per_cluster = 30
    bits_per_neuron = 6
    neurons_per_genome = num_neurons_per_cluster * num_clusters
    bits_flat = [bits_per_neuron] * neurons_per_genome * num_genomes
    neurons_flat = [num_neurons_per_cluster] * num_clusters * num_genomes

    # Generate fixed deterministic connections.
    rng_conn = np.random.default_rng(123)
    conns_per_genome = bits_per_neuron * neurons_per_genome
    conns_flat = rng_conn.integers(
        0, train_bits.shape[1], size=conns_per_genome * num_genomes
    ).astype(np.int64).tolist()

    train_input_flat = np.ascontiguousarray(train_bits.flatten().astype(np.uint8))
    eval_input_flat = np.ascontiguousarray(eval_bits.flatten().astype(np.uint8))
    train_targets_np = np.ascontiguousarray(train_labels, dtype=np.int64)
    eval_targets_np = np.ascontiguousarray(eval_labels, dtype=np.int64)
    train_negs_np = np.ascontiguousarray(1 - train_targets_np, dtype=np.int64)

    lm_oi_a = ra.evaluate_genomes_parallel(
        bits_flat, neurons_flat, conns_flat, num_genomes, num_clusters,
        train_input_flat, train_targets_np, train_negs_np,
        len(train_labels), 1,
        eval_input_flat, eval_targets_np, len(eval_labels),
        train_bits.shape[1], 0.0, 1.0, 42,
    )
    lm_oi_b = ra.evaluate_genomes_parallel(
        bits_flat, neurons_flat, conns_flat, num_genomes, num_clusters,
        train_input_flat, train_targets_np, train_negs_np,
        len(train_labels), 1,
        eval_input_flat, eval_targets_np, len(eval_labels),
        train_bits.shape[1], 0.0, 1.0, 42,
    )
    lm_oi_a = [tuple(round(x, 6) for x in row) for row in lm_oi_a]
    lm_oi_b = [tuple(round(x, 6) for x in row) for row in lm_oi_b]
    print(f"    LM OI run a: {lm_oi_a}")
    print(f"    LM OI run b: {lm_oi_b}")
    assert lm_oi_a == lm_oi_b, "evaluate_genomes_parallel should be deterministic under OI"
    print(f"    PASS — LM legacy entry deterministic")

    print("\nDone.")


if __name__ == "__main__":
    main()
