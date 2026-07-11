"""Numerical equivalence: OLD multi-call path vs NEW evaluate_at_thresholds.

For a small UNSW genome, compare metrics from:
  OLD: 3 separate evaluate_batch_full calls (train_cal / fixed=0.5 / oracle=-1.0)
       + score_examples + score_train_examples
  NEW: 1 evaluate_at_thresholds call returning (eval_scores, train_scores, metrics)

Tolerance: 1e-10 (same training seed → same memory → identical scores → identical
metrics). Any wider divergence indicates the new path drifts from the old path
and needs investigation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "wnn"))

import numpy as np
import ram_accelerator
from wnn.ram.architecture.ids_evaluator import IDSEvaluator
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


TOL = 1e-10


def make_synthetic_dataset(n_train=2000, n_eval=500, n_features=64, seed=42):
    """Build a tiny synthetic IDS dataset directly compatible with IDSEvaluator."""
    rng = np.random.default_rng(seed)
    # Feature pattern: class 0 mostly zeros with some noise; class 1 mostly ones with noise
    y_train = rng.integers(0, 2, n_train).astype(np.int64)
    y_eval = rng.integers(0, 2, n_eval).astype(np.int64)
    X_train = rng.random((n_train, n_features)) > 0.5
    X_eval = rng.random((n_eval, n_features)) > 0.5
    # Bias features so class 1 has higher density on first 16 bits
    for i in range(n_train):
        if y_train[i] == 1:
            X_train[i, :16] = rng.random(16) > 0.3
        else:
            X_train[i, :16] = rng.random(16) > 0.7
    for i in range(n_eval):
        if y_eval[i] == 1:
            X_eval[i, :16] = rng.random(16) > 0.3
        else:
            X_eval[i, :16] = rng.random(16) > 0.7

    class _Dummy:
        pass
    d = _Dummy()
    d.X_train = X_train.astype(np.uint8)
    d.X_test = X_eval.astype(np.uint8)
    d.y_train_binary = y_train
    d.y_test_binary = y_eval
    d.y_train_multi = y_train
    d.y_test_multi = y_eval
    d.category_names = ["normal", "attack"]
    return d


def make_small_genome(n_features=64, n_neurons=5, n_bits=4, seed=42):
    """Build a tiny single-cluster genome."""
    rng = np.random.default_rng(seed)
    bits_per_neuron = [n_bits] * n_neurons
    neurons_per_cluster = [n_neurons]  # single cluster
    connections = []
    for _ in range(n_neurons):
        connections.extend(rng.choice(n_features, n_bits, replace=False).tolist())
    return ClusterGenome(
        bits_per_neuron=bits_per_neuron,
        neurons_per_cluster=neurons_per_cluster,
        connections=connections,
    )


def main():
    print("Building synthetic dataset...")
    ds = make_synthetic_dataset()
    print(f"  train: {ds.X_train.shape}, eval: {ds.X_test.shape}")

    print("\nCreating IDSEvaluator (single-cluster, binary)...")
    evaluator = IDSEvaluator(
        dataset=ds,
        classification="binary",
        single_cluster=True,
        num_parts=1,
        seed=42,
        neuron_sample_rate=1.0,  # full-rate so seeds give deterministic memory
    )

    genome = make_small_genome()

    # ── OLD path ───────────────────────────────────────────────────────────
    print("\n[OLD path] 3 separate evaluate_batch_full calls + score_examples + score_train_examples")
    old_train_cal = evaluator.evaluate_batch_full([genome])[0]
    old_train_t = genome.threshold  # set by evaluate_batch_full when override is None
    old_fixed = evaluator.evaluate_batch_full([genome], override_threshold=0.5)[0]
    old_oracle = evaluator.evaluate_batch_full([genome], override_threshold=-1.0)[0]
    old_oracle_t = genome.threshold
    old_eval_scores = evaluator.score_examples(genome)
    old_train_scores = evaluator.score_train_examples(genome)

    print(f"  train_cal: ce={old_train_cal.ce:.10f} acc={old_train_cal.acc:.10f} "
          f"f1={old_train_cal.f1:.10f} fpr={old_train_cal.fpr:.10f} t={old_train_t:.10f}")
    print(f"  fixed_05:  ce={old_fixed.ce:.10f} acc={old_fixed.acc:.10f} "
          f"f1={old_fixed.f1:.10f} fpr={old_fixed.fpr:.10f}")
    print(f"  val_cal:   ce={old_oracle.ce:.10f} acc={old_oracle.acc:.10f} "
          f"f1={old_oracle.f1:.10f} fpr={old_oracle.fpr:.10f} t={old_oracle_t:.10f}")
    print(f"  eval_scores: len={len(old_eval_scores)} first5={old_eval_scores[:5]}")
    print(f"  train_scores: len={len(old_train_scores)} first5={old_train_scores[:5]}")

    # ── NEW path ───────────────────────────────────────────────────────────
    print("\n[NEW path] single evaluate_at_thresholds returning eval+train scores + metrics")
    # We call with [-1.0, 0.5] for oracle + fixed to compare those metrics.
    # train_cal needs train_scores → derive in Python first.
    eval_scores, train_scores, _val_scores, metrics = evaluator.evaluate_at_thresholds(
        genome, [-1.0, 0.5],
    )
    new_oracle, new_fixed = metrics

    # Derive train_cal threshold + metrics in Python via the helpers
    train_t_new, _, _ = ram_accelerator.find_optimal_threshold_f1_py(
        train_scores, evaluator._y_train,
    )
    new_tc_ce, new_tc_acc, new_tc_f1, new_tc_fpr = ram_accelerator.compute_binary_metrics_at_threshold_py(
        eval_scores, evaluator._y_test, train_t_new, 0,
    )

    print(f"  train_cal: ce={new_tc_ce:.10f} acc={new_tc_acc:.10f} "
          f"f1={new_tc_f1:.10f} fpr={new_tc_fpr:.10f} t={train_t_new:.10f}")
    print(f"  fixed_05:  ce={new_fixed.ce:.10f} acc={new_fixed.acc:.10f} "
          f"f1={new_fixed.f1:.10f} fpr={new_fixed.fpr:.10f}")
    print(f"  val_cal:   ce={new_oracle.ce:.10f} acc={new_oracle.acc:.10f} "
          f"f1={new_oracle.f1:.10f} fpr={new_oracle.fpr:.10f} t={new_oracle.threshold:.10f}")
    print(f"  eval_scores: len={len(eval_scores)} first5={eval_scores[:5]}")
    print(f"  train_scores: len={len(train_scores)} first5={train_scores[:5]}")

    # ── Compare ────────────────────────────────────────────────────────────
    def cmp(name, old, new, tol=TOL):
        diff = abs(float(old) - float(new))
        status = "OK" if diff < tol else "DIVERGENT"
        print(f"  {name}: |{old:.12f} - {new:.12f}| = {diff:.2e}  [{status}]")
        return diff < tol

    print("\n[COMPARE] (tolerance=1e-10)")
    all_ok = True

    # Eval scores arrays
    old_eval = np.asarray(old_eval_scores)
    new_eval = np.asarray(eval_scores)
    max_eval_diff = float(np.max(np.abs(old_eval - new_eval)))
    print(f"  eval_scores max abs diff: {max_eval_diff:.2e}")
    all_ok &= max_eval_diff < TOL

    old_tr = np.asarray(old_train_scores)
    new_tr = np.asarray(train_scores)
    max_tr_diff = float(np.max(np.abs(old_tr - new_tr)))
    print(f"  train_scores max abs diff: {max_tr_diff:.2e}")
    all_ok &= max_tr_diff < TOL

    print("\n  --- train_cal metrics ---")
    all_ok &= cmp("ce ", old_train_cal.ce, new_tc_ce)
    all_ok &= cmp("acc", old_train_cal.acc, new_tc_acc)
    all_ok &= cmp("f1 ", old_train_cal.f1, new_tc_f1)
    all_ok &= cmp("fpr", old_train_cal.fpr, new_tc_fpr)
    all_ok &= cmp("t  ", old_train_t, train_t_new)

    print("\n  --- fixed_05 metrics ---")
    all_ok &= cmp("ce ", old_fixed.ce, new_fixed.ce)
    all_ok &= cmp("acc", old_fixed.acc, new_fixed.acc)
    all_ok &= cmp("f1 ", old_fixed.f1, new_fixed.f1)
    all_ok &= cmp("fpr", old_fixed.fpr, new_fixed.fpr)

    print("\n  --- val_cal (oracle) metrics ---")
    all_ok &= cmp("ce ", old_oracle.ce, new_oracle.ce)
    all_ok &= cmp("acc", old_oracle.acc, new_oracle.acc)
    all_ok &= cmp("f1 ", old_oracle.f1, new_oracle.f1)
    all_ok &= cmp("fpr", old_oracle.fpr, new_oracle.fpr)
    all_ok &= cmp("t  ", old_oracle_t, new_oracle.threshold)

    print("\n" + ("=" * 60))
    print("ALL EQUIVALENT" if all_ok else "DIVERGENCE DETECTED — investigate")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
