"""
Option A — production-scale Option B parity with sample_rate=0.25.

Validates exact GPU↔CPU match on shapes that mirror the TOP20 cohort and
multi-class flows under the production neuron_sample_rate=0.25 default.

Compares against the actually-running cohort shape (exp 8654 flow 2667):
  - 250n × 100b max architecture, GA picks ~50-60n × ~46-48b
  - Per-fold train examples ~228K (T20 1.43M × 0.8 / 5 folds)
  - sample_rate=0.25 (production default — discovered today's session)
  - ng up to 16 (cpu_cores cap)
"""
import time
import ram_accelerator as ra

print("=" * 78)
print("Option A — production-scale Option B parity (sample_rate=0.25)")
print("=" * 78)

print("\n--- Single-cluster (binary IDS), production shape ---")
# Production: TOP20 cohort, T20 features = 20 thermometers × 8 bits = 160 input bits.
# Actual T20 with 96b thermometers: 96 × 20 = 1920 bits. Conservative test: 768.
configs = [
    # (ng, n, e, b, total_input_bits, sample_rate, rng_seed, label)
    (1,  100, 50_000,  48, 768, 0.25, 42, "ng=1  e=50K  b=48 sr=0.25"),
    (4,  100, 50_000,  48, 768, 0.25, 42, "ng=4  e=50K  b=48 sr=0.25"),
    (16, 100, 50_000,  48, 768, 0.25, 42, "ng=16 e=50K  b=48 sr=0.25"),
    (16, 60,  100_000, 48, 1920, 0.25, 42, "ng=16 e=100K b=48 sr=0.25 (cohort-shape)"),
    (16, 60,  228_000, 48, 1920, 0.25, 42, "ng=16 e=228K b=48 sr=0.25 (PRODUCTION per-fold)"),
]
for (ng, n, e, b, ib, sr, seed, label) in configs:
    t0 = time.time()
    try:
        r = ra.run_marker_train_batched_parity_test(ng, n, e, b, ib, 12345, sr, seed)
        dt = time.time() - t0
        for name, ok, detail, gpu_ms, cpu_ms in r:
            status = "PASS" if ok else "FAIL"
            speedup = cpu_ms / max(gpu_ms, 0.001)
            print(f"  [{status}] {label}: gpu={gpu_ms:.0f}ms cpu={cpu_ms:.0f}ms speedup={speedup:.1f}x py_wall={dt:.1f}s")
            if not ok:
                print(f"    detail: {detail[:300]}")
    except Exception as ex:
        print(f"  [EXC] {label}: {ex}")

print("\n--- Multi-cluster (K=8, e.g., CIC-IoT-2023), production shape ---")
configs_mc = [
    # (ng, K, n/c, e, b, ib, sr, rng_seed, label)
    (4,  8, 12, 50_000, 48, 768, 0.25, 42, "ng=4  K=8 n/c=12 e=50K"),
    (16, 8, 12, 50_000, 48, 768, 0.25, 42, "ng=16 K=8 n/c=12 e=50K"),
    (16, 8, 12, 100_000, 48, 1920, 0.25, 42, "ng=16 K=8 n/c=12 e=100K (CIC-IoT-shape)"),
]
for (ng, k, npc, e, b, ib, sr, seed, label) in configs_mc:
    t0 = time.time()
    try:
        r = ra.run_marker_train_multicluster_parity_test(ng, k, npc, e, b, ib, 12345, sr, seed)
        dt = time.time() - t0
        for name, ok, detail, gpu_ms, cpu_ms in r:
            status = "PASS" if ok else "FAIL"
            speedup = cpu_ms / max(gpu_ms, 0.001)
            print(f"  [{status}] {label}: gpu={gpu_ms:.0f}ms cpu={cpu_ms:.0f}ms speedup={speedup:.1f}x py_wall={dt:.1f}s")
            if not ok:
                print(f"    detail: {detail[:300]}")
    except Exception as ex:
        print(f"  [EXC] {label}: {ex}")

print("\nDone.")
