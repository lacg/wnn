"""
Option B timing sweep — runs the existing run_marker_train_batched_parity_test
with controlled (num_genomes, num_examples) combinations. The parity test
calls MarkerTrainer::train directly without the adaptive.rs / wiring glue,
so it isolates whether the bug is in the kernel/dispatch or in the surrounding
batched_train_offspring path.

Pattern from memory note (with surrounding glue, single-genome):
    e=5k:  80ms        (fast)
    e=50k: 62000ms     (~62 sec slow)
    e=200k: 2500ms     (fast)
    e=500k: 68000ms    (~68 sec slow)
    e=1.5M: HUNG

Goal: see if the parity test reproduces this pattern. If YES → kernel bug;
if NO → glue bug in batched_train_offspring or its caller.
"""
import os
import sys
import time

os.environ.setdefault("WNN_OPTION_B_TRACE", "1")

# Make ram_accelerator importable
import ram_accelerator as ra

NUM_NEURONS = 100
BITS_PER_NEURON = 48
TOTAL_INPUT_BITS = 96 * 8  # 96b thermometers × 8 features = 768 bits
SEED = 12345

# Sweep configurations
CONFIGS = [
    (1, 5_000),
    (1, 50_000),
    (1, 200_000),
    (1, 500_000),
    (1, 1_500_000),
    (4, 50_000),
    (16, 50_000),
    (50, 50_000),
    (50, 200_000),
]

def main():
    print(f"# Option B kernel timing sweep")
    print(f"# n={NUM_NEURONS}, b={BITS_PER_NEURON}, input_bits={TOTAL_INPUT_BITS}")
    print(f"# format: (num_genomes, num_examples) -> gpu_ms, cpu_ms, speedup, parity_ok")
    print()

    for (ng, ne) in CONFIGS:
        sys.stdout.write(f"=== num_genomes={ng}, num_examples={ne} ===\n")
        sys.stdout.flush()
        t0 = time.time()
        try:
            results = ra.run_marker_train_batched_parity_test(
                ng, NUM_NEURONS, ne, BITS_PER_NEURON, TOTAL_INPUT_BITS, SEED
            )
            t_total = time.time() - t0
            for (name, ok, detail, gpu_ms, cpu_ms) in results:
                speedup = cpu_ms / max(gpu_ms, 0.001)
                status = "OK" if ok else "FAIL"
                print(f"  [{status}] gpu={gpu_ms:.2f}ms cpu={cpu_ms:.2f}ms speedup={speedup:.2f}x py_wall={t_total*1000:.0f}ms")
                print(f"  detail: {detail}")
        except Exception as e:
            t_total = time.time() - t0
            print(f"  [EXCEPTION after {t_total:.1f}s] {e}")
        sys.stdout.flush()
        print()

if __name__ == "__main__":
    main()
