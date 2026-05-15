"""
B6 + B9 validation after MSB-first fix.

Confirms:
1. Parity (exact match GPU↔CPU after MSB-fix) still holds with B6's parallel export
2. B9 enables ng up to 50 batched dispatches with linear scaling
3. Production-scale shapes work end-to-end via evaluate_genomes_parallel_hybrid

B6: export_per_neuron runs in parallel across neurons (rayon).
B9: pool_size cap raised from cpu_cores=16 to 50 when WNN_GPU_BATCHED_TRAIN=1.
"""
import os
import time
import numpy as np
import ram_accelerator as ra


def make(num_genomes, num_train, num_eval, num_neurons, b, sr=0.25, seed=12345, ib=1920):
    rng = np.random.default_rng(seed)
    total = num_genomes * num_neurons
    return {
        'genomes_bits_flat': np.full(total, b, dtype=np.uint64).tolist(),
        'genomes_neurons_flat': np.full(num_genomes, num_neurons, dtype=np.uint64).tolist(),
        'genomes_connections_flat': rng.integers(0, ib, size=num_genomes*num_neurons*b, dtype=np.int64).tolist(),
        'num_genomes': num_genomes, 'num_clusters': 1,
        'train_input_bits': rng.integers(0, 2, size=num_train*ib, dtype=np.uint8),
        'train_targets': (rng.random(num_train) < 0.15).astype(np.int64),
        'train_negatives': np.zeros(1, dtype=np.int64),
        'num_train': num_train, 'num_negatives': 0,
        'eval_input_bits': rng.integers(0, 2, size=num_eval*ib, dtype=np.uint8),
        'eval_targets': (rng.random(num_eval) < 0.15).astype(np.int64),
        'num_eval': num_eval, 'total_input_bits': ib,
        'empty_value': 0.5, 'neuron_sample_rate': sr, 'rng_seed': 42,
    }


print("=" * 80)
print("B6 + B9 validation (after MSB-first fix)")
print("=" * 80)

# --- Section 1: Parity tests at production scale verify B6 doesn't break correctness ---
print("\n--- 1. Parity at production scale (sr=0.25, post-MSB-fix) ---")
parity_configs = [
    (16, 100, 50_000, 48, 1920, "ng=16 n=100 e=50K b=48"),
    (16, 60, 100_000, 48, 1920, "ng=16 n=60 e=100K b=48 (cohort)"),
    (16, 60, 228_000, 48, 1920, "ng=16 n=60 e=228K b=48 (TOP20 per-fold)"),
]
for (ng, n, e, b, ib, label) in parity_configs:
    r = ra.run_marker_train_batched_parity_test(ng, n, e, b, ib, 12345, 0.25, 42)
    for name, ok, detail, gpu_ms, cpu_ms in r:
        status = "PASS" if ok else "FAIL"
        speedup = cpu_ms / max(gpu_ms, 0.001)
        print(f"  [{status}] {label}: gpu={gpu_ms:.0f}ms cpu={cpu_ms:.0f}ms (parity speedup vs serial CPU: {speedup:.1f}x)")

# --- Section 2: B9 effect — ng=50 batched dispatch ---
print("\n--- 2. B9: ng=50 batched dispatch works (parity at scale) ---")
b9_configs = [
    (50, 30, 50_000, 48, 1920, "ng=50 n=30  e=50K  b=48 (B9 maximum batch)"),
    (50, 60, 30_000, 48, 1920, "ng=50 n=60  e=30K  b=48"),
    (32, 60, 100_000, 48, 1920, "ng=32 n=60  e=100K b=48"),
]
for (ng, n, e, b, ib, label) in b9_configs:
    r = ra.run_marker_train_batched_parity_test(ng, n, e, b, ib, 12345, 0.25, 42)
    for name, ok, detail, gpu_ms, cpu_ms in r:
        status = "PASS" if ok else "FAIL"
        speedup = cpu_ms / max(gpu_ms, 0.001)
        print(f"  [{status}] {label}: gpu={gpu_ms:.0f}ms cpu={cpu_ms:.0f}ms ({speedup:.1f}x)")

# --- Section 3: End-to-end via evaluate_genomes_parallel_hybrid, Option B vs Baseline ---
print("\n--- 3. End-to-end at high b (deterministic regime): Option B vs Baseline ---")
e2e_configs = [
    # (ng, n, e, b, label)
    (1,  60, 100_000, 48, "ng=1  n=60 e=100K b=48"),
    (16, 60, 100_000, 48, "ng=16 n=60 e=100K b=48 (typical batched generation)"),
    (50, 60, 50_000, 48, "ng=50 n=60 e=50K  b=48 (B9 batched generation)"),
]
for (ng, n, e, b, label) in e2e_configs:
    kw = make(ng, e, e//5, n, b, sr=0.25)
    # baseline
    os.environ.pop('WNN_GPU_BATCHED_TRAIN', None)
    t0 = time.time()
    base = ra.evaluate_genomes_parallel_hybrid(**kw)
    t_base = time.time() - t0
    # option B
    os.environ['WNN_GPU_BATCHED_TRAIN'] = '1'
    t0 = time.time()
    optb = ra.evaluate_genomes_parallel_hybrid(**kw)
    t_optb = time.time() - t0
    # compare
    max_dce = max(abs(b_[0] - o[0]) for b_, o in zip(base, optb))
    max_dacc = max(abs(b_[1] - o[1]) for b_, o in zip(base, optb))
    speedup = t_base / max(t_optb, 0.001)
    status = "OK" if max_dce < 1e-4 and max_dacc < 1e-4 else "DIVERGED"
    print(f"  [{status}] {label}: base={t_base:.1f}s optB={t_optb:.1f}s ({speedup:.2f}x) | max|ΔCE|={max_dce:.4f} max|ΔAcc|={max_dacc:.4f}")

print("\nDone.")
