"""
Reproduce the production bug: Option B produces degraded F1/FPR vs baseline
at sample_rate=0.25, high-b configs.

Calls evaluate_genomes_parallel_hybrid in both modes (Option B ON vs OFF)
with identical synthetic data + sampling. Compares headline CE/Acc per
genome. If they diverge, we've reproduced.
"""
import os
import sys
import time
import numpy as np
import ram_accelerator as ra


def make_scenario(num_genomes, num_train, num_eval, num_neurons, bits_per_neuron,
                  total_input_bits=1920, sample_rate=0.25, seed=12345):
    rng = np.random.default_rng(seed)
    total_neurons = num_neurons * num_genomes
    genomes_bits_flat = np.full(total_neurons, bits_per_neuron, dtype=np.uint64).tolist()
    genomes_neurons_flat = np.full(num_genomes, num_neurons, dtype=np.uint64).tolist()
    conns_per_genome = num_neurons * bits_per_neuron
    total_conns = num_genomes * conns_per_genome
    genomes_connections_flat = rng.integers(0, total_input_bits, size=total_conns, dtype=np.int64).tolist()

    # Thermometer-like data: each example value v in [0,1], thermometer = v*8 ones followed by zeros.
    # Mimics real T20 data correlation pattern better than uniform random bits.
    feats = 20  # 20 features
    bits_per_feat = total_input_bits // feats  # 96 each
    train_input_bits = np.zeros(num_train * total_input_bits, dtype=np.uint8)
    for ex in range(num_train):
        for f in range(feats):
            v = rng.random()
            n_ones = int(v * bits_per_feat)
            base = ex * total_input_bits + f * bits_per_feat
            train_input_bits[base:base + n_ones] = 1
    train_targets = (rng.random(num_train) < 0.15).astype(np.int64)  # 15% attack
    train_negatives = np.zeros(1, dtype=np.int64)

    eval_input_bits = np.zeros(num_eval * total_input_bits, dtype=np.uint8)
    for ex in range(num_eval):
        for f in range(feats):
            v = rng.random()
            n_ones = int(v * bits_per_feat)
            base = ex * total_input_bits + f * bits_per_feat
            eval_input_bits[base:base + n_ones] = 1
    eval_targets = (rng.random(num_eval) < 0.15).astype(np.int64)

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
        "neuron_sample_rate": sample_rate,
        "rng_seed": 42,
    }


def run(kwargs, option_b: bool, label: str):
    if option_b:
        os.environ["WNN_OPTION_B"] = "1"
    else:
        os.environ.pop("WNN_OPTION_B", None)
    t0 = time.time()
    try:
        result = ra.evaluate_genomes_parallel_hybrid(**kwargs)
        dt = time.time() - t0
        # result is list of (ce, acc, ...) per genome
        ce = result[0][0]
        acc = result[0][1]
        print(f"  [{label}] wall={dt:.1f}s CE={ce:.4f} Acc={acc:.4f}")
        return ce, acc
    except Exception as e:
        dt = time.time() - t0
        print(f"  [{label}] EXC after {dt:.1f}s: {e}")
        return None, None


configs = [
    # (n, e, b, label)
    (50, 50_000, 4,  "n=50 b=4   e=50K  (should match)"),
    (50, 50_000, 16, "n=50 b=16  e=50K  (should match)"),
    (50, 50_000, 32, "n=50 b=32  e=50K  (suspected divergence)"),
    (50, 50_000, 48, "n=50 b=48  e=50K  (suspected divergence)"),
    (50, 100_000, 48, "n=50 b=48  e=100K (suspected divergence)"),
]

print("=" * 80)
print("Option B vs baseline at sample_rate=0.25 with thermometer-like data")
print("=" * 80)
for (n, e, b, label) in configs:
    print(f"\n=== {label} ===")
    kw = make_scenario(num_genomes=1, num_train=e, num_eval=int(e*0.2), num_neurons=n, bits_per_neuron=b)
    ce_base, acc_base = run(kw, option_b=False, label="Baseline")
    ce_optb, acc_optb = run(kw, option_b=True,  label="Option B")
    if ce_base is not None and ce_optb is not None:
        d_ce = abs(ce_base - ce_optb)
        d_acc = abs(acc_base - acc_optb)
        status = "OK" if d_ce < 1e-4 and d_acc < 1e-4 else "DIVERGED"
        print(f"  [{status}] |ΔCE|={d_ce:.4f} |ΔAcc|={d_acc:.4f}")
    sys.stdout.flush()

print("\nDone.")
