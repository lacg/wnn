# Adaptation Mechanisms: Dynamic Architecture-Aware Synaptogenesis & Neurogenesis

## Overview

The WNN adaptation system implements **Lamarckian/Baldwin evolution** — architectures adapt during evaluation, and the adapted fitness feeds back into GA/TS selection. This creates evolutionary pressure for **adaptable** architectures, not just performant ones.

Two mechanisms operate at different granularities:

| Mechanism | Granularity | What it does |
|-----------|-------------|--------------|
| **Synaptogenesis** | Per-neuron connections | Prune low-information connections, grow connections where underfitting |
| **Neurogenesis** | Per-cluster neurons | Add neurons to underfitting clusters, remove redundant neurons |

## Why Dynamic Thresholds

### The Problem: Absolute Thresholds Fail at Scale

The original implementation used hardcoded absolute thresholds tuned for small architectures (5-30 neurons, 8-16 bits). With 200 neurons × 20 bits:

| Threshold | Hardcoded Value | Actual Metric | Triggers? |
|-----------|----------------|---------------|-----------|
| `prune_entropy_threshold = 0.05` | — | ~1.0 | Never |
| `grow_fill_threshold = 0.8` | — | max 14.3%* | Never |
| `neuron_uniqueness_threshold = 0.05` | — | ~0.50 | Never |
| `max_neurons = 30` | — | 200 current | Blocks growth |

*20-bit address space = 2^20 = 1,048,576 cells. With 150K examples, maximum possible fill = 150K/1M = 14.3%.

**Result**: ~60 seconds per generation wasted computing stats that produce zero adaptation.

### The Fix: Architecture-Aware Relative Thresholds

All thresholds now scale with architecture parameters:

| Parameter | Formula | Example (20-bit, 150K examples) |
|-----------|---------|--------------------------------|
| Prune threshold | `median_entropy × 0.3` | ~0.3 (reachable) |
| Grow fill threshold | `expected_fill × 0.5` | 14.3% × 0.5 = 7.1% (reachable) |
| Growth cap | `initial_neurons × 1.5` | 200 × 1.5 = 300 |
| Prune candidates | Bottom 10% by score | Always ~10% eligible |

## Design Principles (Literature-Informed)

| Principle | Source | Application |
|-----------|--------|-------------|
| **Percentile-based pruning** | SET (Mocanu 2018), RigL (Evci 2020) | Prune bottom X% by metric, not absolute threshold |
| **Relative to expected capacity** | Address space math | Fill thresholds scale with `min(1, examples/2^bits)` |
| **Competition-gated removal** | Biology: JAK2/STAT1 pathway | Only remove if neuron score < cluster_mean × factor |
| **Cosine annealing** | RigL: `rate(t) = r₀/2·(1+cos(πt/T))` | Aggressive early, frozen last 25% of post-warmup |
| **Contribution = uniqueness × accuracy** | Fisher Information pruning | Combined metric avoids removing diverse OR accurate neurons |

## Cosine Schedule (Warmup-Aware)

The warmup period does NOT count against the active schedule:

```
total=250, warmup=10, stabilize_fraction=0.25

post_warmup = 250 - 10 = 240
active_window = 240 × 0.75 = 180 gens
active_start = gen 10 (first post-warmup)
active_end = 10 + 180 = gen 190

Gen 10 (first active): progress=0.0, rate=1.0
Gen 100:               progress=0.5, rate=0.5
Gen 190:               progress=1.0, rate=0.0
Gen 191-250:           frozen (rate=0.0, skip stats entirely)
```

**Key optimization**: When `rate = 0.0` (warmup + stabilization = ~35% of gens), all stats computation is skipped entirely, saving ~60s/gen.

## Configuration Parameters

### Default Values and Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `prune_entropy_ratio = 0.3` | SET: 30% replacement. Prunes connections with entropy < 30% of neuron's median |
| `grow_fill_utilization = 0.5` | 50% of expected fill. For 20-bit: 0.5 × 14.3% = 7.1% (reachable) |
| `grow_error_baseline = 0.35` | Below random (0.5) but above well-trained (~0.2). NEAT-inspired |
| `neuron_prune_percentile = 0.1` | Bottom 10% candidates. Lottery Ticket: 80-96% prunable, we're conservative |
| `neuron_removal_factor = 0.5` | Competition gate (JAK2/STAT1). Must be < 50% of cluster mean |
| `max_growth_ratio = 1.5` | Overproduction 1.5-3x (Cowan 1984). Conservative 1.5x |
| `stabilize_fraction = 0.25` | RigL: stops at 75%. Biology: adult brain stabilizes architecture |

### Expected Fill Rate

The theoretical fill for a neuron with `b` bits and `N` training examples:

```
expected_fill = min(1.0, N / 2^b)
```

| Bits | Address Space | Fill @ 150K examples |
|------|--------------|---------------------|
| 8 | 256 | 100% (saturated) |
| 12 | 4,096 | 100% (saturated) |
| 16 | 65,536 | 100% (saturated) |
| 20 | 1,048,576 | 14.3% |
| 24 | 16,777,216 | 0.9% |

## GPU+CPU Hybrid Stats

### Architecture

```
Current:  per-genome sequential GPU dispatch (50 genomes × 2 passes = 100 GPU calls)
New:      batched single dispatch (1 Pass 1 + 1 Pass 2 = 2 GPU calls)
```

**Shared data** (same for all genomes): `packed_input`, `target_bits`, `sample_indices` — ~60-80% of GPU data.

**Per-genome data** (concatenated): neuron memory, connections — only differs between genomes.

The Metal shader processes each neuron independently via `NeuronMeta` (containing all per-neuron offsets), so batching multiple genomes is transparent — just concatenate all neurons.

## References

- Mocanu, D.C. et al. (2018). "Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science." *Nature Communications*.
- Evci, U. et al. (2020). "Rigging the Lottery: Making All Tickets Winners." *ICML*.
- Frankle, J. & Carlin, M. (2019). "The Lottery Ticket Hypothesis." *ICLR*.
- Stanley, K.O. & Miikkulainen, R. (2002). "Evolving Neural Networks through Augmenting Topologies." *Evolutionary Computation* (NEAT).
- Cowan, W.M. et al. (1984). "Regressive events in neurogenesis." *Science*.
- Baldwin, J.M. (1896). "A New Factor in Evolution." *American Naturalist*.

For weekly research updates and deeper analysis: [llm-optimizer](https://lacg.github.io/llm-optimizer)
