"""
B5f — multi-class end-to-end parity through evaluate_genomes_parallel_hybrid.

Mirrors what a real CIC-IoT-2023 8-class flow would do, but driven from
Python without queueing through the dashboard. Validates that:
1. Multi-cluster Option B path produces identical CE/Acc per genome to baseline
2. Imbalanced class distributions work (mimics CIC-IoT's ~86% benign)
3. Multi-cluster path handles K∈{4, 8} at sample_rate=0.25 (production default)

If this passes, multi-cluster Option B is wired-path-correct. A real flow
would only add: HF dataset loading, threshold calibration, per-class F1.
"""
import os
import time
import numpy as np
import ram_accelerator as ra


def make_multiclass(num_genomes, num_train, num_eval, num_clusters,
                    neurons_per_cluster, b, num_negatives,
                    sr=0.25, class_skew=None, total_input_bits=1920, seed=42):
    """Synthetic multi-class data with optional class imbalance.

    class_skew: None for uniform, or a probability vector (len=num_clusters).
                CIC-IoT-2023-like would be ~[0.86, 0.02, 0.05, 0.01, ...].
    """
    rng = np.random.default_rng(seed)
    total_neurons_per_genome = num_clusters * neurons_per_cluster
    total = num_genomes * total_neurons_per_genome

    # Class distribution
    if class_skew is None:
        train_targets = rng.integers(0, num_clusters, size=num_train, dtype=np.int64)
        eval_targets = rng.integers(0, num_clusters, size=num_eval, dtype=np.int64)
    else:
        skew = np.asarray(class_skew) / np.sum(class_skew)
        train_targets = rng.choice(num_clusters, size=num_train, p=skew).astype(np.int64)
        eval_targets = rng.choice(num_clusters, size=num_eval, p=skew).astype(np.int64)

    # Generate train_negatives: for each example, pick `num_negatives` distinct
    # non-target clusters
    train_negatives = np.zeros(num_train * num_negatives, dtype=np.int64)
    for ex in range(num_train):
        target = int(train_targets[ex])
        pool = [c for c in range(num_clusters) if c != target]
        rng.shuffle(pool)
        for k in range(num_negatives):
            train_negatives[ex * num_negatives + k] = pool[k % len(pool)]

    return {
        'genomes_bits_flat': np.full(total, b, dtype=np.uint64).tolist(),
        'genomes_neurons_flat': (np.full(num_genomes * num_clusters, neurons_per_cluster, dtype=np.uint64).tolist()),
        'genomes_connections_flat': rng.integers(0, total_input_bits, size=num_genomes*total_neurons_per_genome*b, dtype=np.int64).tolist(),
        'num_genomes': num_genomes,
        'num_clusters': num_clusters,
        'train_input_bits': rng.integers(0, 2, size=num_train*total_input_bits, dtype=np.uint8),
        'train_targets': train_targets,
        'train_negatives': train_negatives,
        'num_train': num_train,
        'num_negatives': num_negatives,
        'eval_input_bits': rng.integers(0, 2, size=num_eval*total_input_bits, dtype=np.uint8),
        'eval_targets': eval_targets,
        'num_eval': num_eval,
        'total_input_bits': total_input_bits,
        'empty_value': 0.5,
        'neuron_sample_rate': sr,
        'rng_seed': 42,
    }


CIC_IOT_DIST = [0.86, 0.04, 0.03, 0.03, 0.02, 0.01, 0.005, 0.005]  # 8 classes


print("=" * 80)
print("B5f — multi-class end-to-end parity (production-like)")
print("=" * 80)

configs = [
    # (ng, K, n/c, e, b, negs, sr, skew, label)
    (4,  4, 25, 5000,  24, 3, 0.25, None,        "ng=4  K=4 n/c=25 e=5K   negs=3 sr=0.25 uniform"),
    (4,  8, 12, 5000,  32, 7, 0.25, None,        "ng=4  K=8 n/c=12 e=5K   negs=7 sr=0.25 uniform (full)"),
    (4,  8, 12, 5000,  32, 2, 0.25, None,        "ng=4  K=8 n/c=12 e=5K   negs=2 sr=0.25 uniform (sparse)"),
    (16, 8, 12, 10000, 32, 2, 0.25, CIC_IOT_DIST, "ng=16 K=8 n/c=12 e=10K  negs=2 sr=0.25 CIC-IoT-skew"),
    (16, 8, 12, 20000, 48, 7, 0.25, CIC_IOT_DIST, "ng=16 K=8 n/c=12 e=20K  negs=7 sr=0.25 CIC-IoT-skew"),
]

for (ng, k, npc, e, b, negs, sr, skew, label) in configs:
    kw = make_multiclass(ng, e, e // 5, k, npc, b, negs, sr, skew)
    os.environ.pop('WNN_OPTION_B', None)
    t0 = time.time()
    base = ra.evaluate_genomes_parallel_hybrid(**kw)
    t_base = time.time() - t0
    os.environ['WNN_OPTION_B'] = '1'
    t0 = time.time()
    optb = ra.evaluate_genomes_parallel_hybrid(**kw)
    t_optb = time.time() - t0
    max_dce = max(abs(b_[0] - o[0]) for b_, o in zip(base, optb))
    max_dacc = max(abs(b_[1] - o[1]) for b_, o in zip(base, optb))
    status = "OK" if max_dce < 1e-4 and max_dacc < 1e-4 else "DIVERGED"
    speedup = t_base / max(t_optb, 0.001)
    print(f"  [{status}] {label}")
    print(f"    base={t_base:.2f}s optB={t_optb:.2f}s ({speedup:.2f}x) | max|ΔCE|={max_dce:.4f} max|ΔAcc|={max_dacc:.4f}")
    print(f"    base[0]={base[0][:2]}  optB[0]={optb[0][:2]}")

print("\nDone.")
