"""Phase 0 parity harness — Path 2 (dense → Option B / MarkerHashTable).

Compares the per-genome (ce, acc, f1, fpr, fitness) outputs of:

  - **CPU dense baseline** (`WNN_GPU_BATCHED_TRAIN=off`):
    the existing reference path, end-to-end on CPU.

  - **Option B path** (`WNN_GPU_BATCHED_TRAIN=force`):
    forces the MarkerHashTable / batched_train_offspring GPU path.

If both produce the same metrics within tolerance for a workload, then
Option B is a faithful replacement for the dense path at that shape —
exactly what Path 2 migration needs.

Run:
    python tests/path2_parity.py

Tolerances:
  - CE absolute Δ ≤ 5e-3 (rayon order-dependence at low b allows small drift)
  - Acc/F1/FPR absolute Δ ≤ 5e-3 (~0.5 pp)
  - Larger tol at b ≤ 12 due to known QUAD non-determinism (see memory).
"""

import os
import sys
from collections import namedtuple

import numpy as np

# Ensure Rust accelerator is importable
import ram_accelerator as ra


GenomeResult = namedtuple("GenomeResult", "ce acc f1 fpr fitness")


def _make_workload(num_genomes: int, num_neurons: int, num_train: int,
                   num_eval: int, bits_per_neuron: int, total_input_bits: int,
                   seed: int) -> dict:
    """Synthetic binary IDS workload, single cluster."""
    rng = np.random.default_rng(seed)
    train_bits = rng.integers(0, 2, size=num_train * total_input_bits, dtype=np.uint8)
    eval_bits = rng.integers(0, 2, size=num_eval * total_input_bits, dtype=np.uint8)
    # 15% attack ratio (similar to neto-sub binary class distribution)
    train_targets = (rng.random(num_train) < 0.15).astype(np.int64)
    eval_targets = (rng.random(num_eval) < 0.15).astype(np.int64)
    # Connections: random selection of total_input_bits per neuron
    bpn = [bits_per_neuron] * (num_genomes * num_neurons)
    neurons = [num_neurons] * num_genomes  # 1 cluster, num_neurons per genome
    conns = rng.integers(0, total_input_bits, size=num_genomes * num_neurons * bits_per_neuron, dtype=np.int64)
    return {
        "genomes_bits_flat": bpn,
        "genomes_neurons_flat": neurons,
        "genomes_connections_flat": conns.tolist(),
        "num_genomes": num_genomes,
        "num_clusters": 1,
        "train_input_bits": train_bits,
        "train_targets": train_targets,
        "train_negatives": np.zeros(1, dtype=np.int64),
        "num_train": num_train,
        "num_negatives": 0,
        "eval_input_bits": eval_bits,
        "eval_targets": eval_targets,
        "num_eval": num_eval,
        "total_input_bits": total_input_bits,
        "empty_value": 0.5,
        "neuron_sample_rate": 0.25,
        "rng_seed": 42,
    }


def _run(workload: dict, mode: str) -> list[GenomeResult]:
    """Run evaluate_genomes_parallel_hybrid in the given mode.

    mode ∈ {"off", "force"}:
      off   → WNN_GPU_BATCHED_TRAIN=off  (CPU dense baseline)
      force → WNN_GPU_BATCHED_TRAIN=force (Option B / GPU batched-train)
    """
    saved = os.environ.get("WNN_GPU_BATCHED_TRAIN")
    try:
        os.environ["WNN_GPU_BATCHED_TRAIN"] = mode
        # Also disable B12 hybrid to keep paths clean
        os.environ["WNN_HYBRID"] = "0"
        results = ra.evaluate_genomes_parallel_hybrid(**workload)
        return [GenomeResult(*r) for r in results]
    finally:
        if saved is None:
            os.environ.pop("WNN_GPU_BATCHED_TRAIN", None)
        else:
            os.environ["WNN_GPU_BATCHED_TRAIN"] = saved
        os.environ.pop("WNN_HYBRID", None)


def _diff(label: str, dense: list[GenomeResult], optb: list[GenomeResult],
          tol_ce: float = 5e-3, tol_metric: float = 5e-3) -> bool:
    """Assert parity, return True if pass."""
    if len(dense) != len(optb):
        print(f"  [FAIL] {label}: length mismatch ({len(dense)} vs {len(optb)})")
        return False

    ok = True
    max_dce = max_dacc = max_df1 = max_dfpr = 0.0
    for i, (d, o) in enumerate(zip(dense, optb)):
        dce = abs(d.ce - o.ce)
        dacc = abs(d.acc - o.acc)
        df1 = abs(d.f1 - o.f1) if (d.f1 is not None and o.f1 is not None) else 0.0
        dfpr = abs(d.fpr - o.fpr) if (d.fpr is not None and o.fpr is not None) else 0.0
        max_dce = max(max_dce, dce)
        max_dacc = max(max_dacc, dacc)
        max_df1 = max(max_df1, df1)
        max_dfpr = max(max_dfpr, dfpr)
        if dce > tol_ce or dacc > tol_metric or df1 > tol_metric or dfpr > tol_metric:
            print(f"    [g{i}] dCE={dce:.5f} dAcc={dacc:.5f} dF1={df1:.5f} dFPR={dfpr:.5f}"
                  f"  dense=({d.ce:.4f},{d.acc:.4f},{d.f1:.4f},{d.fpr:.4f})"
                  f"  optb=({o.ce:.4f},{o.acc:.4f},{o.f1:.4f},{o.fpr:.4f})")
            ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: max(|dCE|={max_dce:.5f} |dAcc|={max_dacc:.5f}"
          f" |dF1|={max_df1:.5f} |dFPR|={max_dfpr:.5f})")
    return ok


def main() -> int:
    print("=" * 78)
    print("Phase 0 — Path 2 parity harness (dense CPU baseline vs Option B GPU)")
    print("=" * 78)

    # Shape grid: cover the cohort's typical architectures + b∈{8,16,32,48,64,96}
    # b ≤ 12 has known non-determinism (memory: baseline-nondeterministic-low-b)
    # — use looser tolerance.
    configs = [
        # (label, num_genomes, n, b, num_train, num_eval, total_input_bits)
        ("ng=4  n=50  b=16",  4, 50,  16, 10_000, 2_000, 320),
        ("ng=4  n=50  b=32",  4, 50,  32, 10_000, 2_000, 768),
        ("ng=4  n=50  b=48",  4, 50,  48, 10_000, 2_000, 1920),
        ("ng=4  n=100 b=48",  4, 100, 48, 10_000, 2_000, 1920),
        ("ng=8  n=100 b=48",  8, 100, 48, 20_000, 4_000, 1920),
        ("ng=4  n=200 b=48",  4, 200, 48, 20_000, 4_000, 1920),
        ("ng=4  n=100 b=64",  4, 100, 64, 10_000, 2_000, 1920),
        ("ng=4  n=100 b=96",  4, 100, 96, 10_000, 2_000, 1920),
        ("ng=4  n=250 b=48",  4, 250, 48, 20_000, 4_000, 1920),
    ]

    pass_count = 0
    fail_count = 0
    for label, ng, n, b, ntr, nev, ib in configs:
        print(f"\n--- {label} ---")
        try:
            wl = _make_workload(ng, n, ntr, nev, b, ib, seed=12345)
            dense = _run(wl, "off")
            optb = _run(wl, "force")
            # Looser tolerance at b ≤ 12 (QUAD non-determinism per memory)
            tol_ce = 5e-2 if b <= 12 else 5e-3
            ok = _diff(label, dense, optb, tol_ce=tol_ce)
            pass_count += int(ok)
            fail_count += int(not ok)
        except Exception as e:
            print(f"  [EXC] {label}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print()
    print("=" * 78)
    print(f"Path 2 parity: {pass_count}/{pass_count + fail_count} configs passed")
    print("=" * 78)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
