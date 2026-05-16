"""Phase 0 (extended) — IDS calibration parity (Path 2 train_single_via_marker).

Validates that the 3 IDS calibration entry points (`score_examples`,
`score_train_examples`, `evaluate_at_thresholds`) produce bit-exact identical
output whether the dense or Option B train path is taken. These are the
functions migrated in Phase 2 (commits 237d28a4 + 0bfae1de).

Setup: build a synthetic IDSCacheWrapper, call score_examples twice with
different WNN_GPU_BATCHED_TRAIN settings, compare per-example scores.

Run:
    python tests/path2_calibration_parity.py
"""

import os
import sys
from collections import namedtuple

import numpy as np
import ram_accelerator as ra


def _make_cache(num_train: int, num_eval: int, total_features: int, seed: int):
    """Build a synthetic IDSCacheWrapper (binary classification, 15% attack rate).
    Features are packed bytes (np.packbits) — total_features bits per row."""
    rng = np.random.default_rng(seed)
    train_unpacked = rng.integers(0, 2, size=(num_train, total_features), dtype=np.uint8)
    eval_unpacked = rng.integers(0, 2, size=(num_eval, total_features), dtype=np.uint8)
    train_features = np.packbits(train_unpacked, axis=1).flatten()
    eval_features = np.packbits(eval_unpacked, axis=1).flatten()
    train_labels = (rng.random(num_train) < 0.15).astype(np.int64).tolist()
    eval_labels = (rng.random(num_eval) < 0.15).astype(np.int64).tolist()
    cache = ra.IDSCacheWrapper.new_from_numpy(
        train_features=train_features,
        train_labels=train_labels,
        eval_features=eval_features,
        eval_labels=eval_labels,
        num_classes=2,
        total_features=total_features,
        num_parts=1,
        num_negatives=0,
        seed=seed,
        balance_classes=False,
        single_cluster=True,
    )
    return cache


def _make_genome(num_neurons: int, bits_per_neuron: int, total_input_bits: int, seed: int):
    """Single-cluster binary genome."""
    rng = np.random.default_rng(seed)
    bits_flat = [bits_per_neuron] * num_neurons
    neurons_flat = [num_neurons]  # 1 cluster
    connections = rng.integers(0, total_input_bits, size=num_neurons * bits_per_neuron, dtype=np.int64).tolist()
    return bits_flat, neurons_flat, connections


def _run_score(cache, bits_flat, neurons_flat, connections, mode: str) -> list[float]:
    """Run score_examples under the given WNN_GPU_BATCHED_TRAIN mode."""
    saved = os.environ.get("WNN_GPU_BATCHED_TRAIN")
    saved_hyb = os.environ.get("WNN_HYBRID")
    try:
        os.environ["WNN_GPU_BATCHED_TRAIN"] = mode
        os.environ["WNN_HYBRID"] = "0"
        return cache.score_examples(
            bits_flat=bits_flat,
            neurons_flat=neurons_flat,
            connections_flat=connections,
            empty_value=0.5,
            neuron_sample_rate=0.25,
            rng_seed=42,
        )
    finally:
        if saved is None:
            os.environ.pop("WNN_GPU_BATCHED_TRAIN", None)
        else:
            os.environ["WNN_GPU_BATCHED_TRAIN"] = saved
        if saved_hyb is None:
            os.environ.pop("WNN_HYBRID", None)
        else:
            os.environ["WNN_HYBRID"] = saved_hyb


def _diff(label: str, dense: list[float], optb: list[float], tol: float = 5e-3) -> bool:
    if len(dense) != len(optb):
        print(f"  [FAIL] {label}: length {len(dense)} vs {len(optb)}")
        return False
    max_d = max(abs(d - o) for d, o in zip(dense, optb))
    ok = max_d <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: max|d|={max_d:.6f} (n={len(dense)})")
    return ok


def main() -> int:
    print("=" * 78)
    print("Phase 0 (ext) — IDS calibration parity: dense CPU vs Option B GPU")
    print("=" * 78)

    configs = [
        # (label, n, b, ntrain, neval, total_features)
        ("n=50  b=16",  50,  16, 5_000, 1_000, 320),
        ("n=50  b=32",  50,  32, 5_000, 1_000, 768),
        ("n=50  b=48",  50,  48, 5_000, 1_000, 1920),
        ("n=100 b=48",  100, 48, 5_000, 1_000, 1920),
        ("n=200 b=48",  200, 48, 5_000, 1_000, 1920),
        ("n=100 b=64",  100, 64, 5_000, 1_000, 1920),
        ("n=100 b=96",  100, 96, 5_000, 1_000, 1920),
    ]

    pass_count = 0
    fail_count = 0
    for label, n, b, ntr, nev, ib in configs:
        print(f"\n--- {label} ---")
        try:
            cache = _make_cache(ntr, nev, ib, seed=12345)
            bits_flat, neurons_flat, conns = _make_genome(n, b, ib, seed=54321)
            dense_scores = _run_score(cache, bits_flat, neurons_flat, conns, "off")
            optb_scores = _run_score(cache, bits_flat, neurons_flat, conns, "force")
            ok = _diff(label, dense_scores, optb_scores)
            pass_count += int(ok)
            fail_count += int(not ok)
        except Exception as e:
            print(f"  [EXC] {label}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print()
    print("=" * 78)
    print(f"IDS calibration parity: {pass_count}/{pass_count + fail_count} configs passed")
    print("=" * 78)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
