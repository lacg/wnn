# Phase Design Review: What Each Phase Does & Should Do

## Purpose

This document maps out exactly what each optimization phase does today, so we can
discuss whether the separation of concerns is correct and identify improvements.

---

## Phase Order (per stage)

```
Grid Search -> GA Neurons -> Neurogenesis -> TS Neurons
           -> GA Bits -> Synaptogenesis -> TS Bits
           -> GA Connections -> Axonogenesis -> TS Connections
```

---

## 1. Grid Search

**What it does:** Evaluates every (neurons, bits) combination from a grid, ranks by
fitness, builds a population of 50 genomes with exponential representation (best
config gets ~35 copies, 2nd gets ~8, etc.).

**What it mutates:** Nothing - pure evaluation.

**Connections:** Each genome gets fresh RANDOM connections. The original (1 per config)
and all copies have independent random connections.

**Output:** 50 genomes with metrics, passed to GA Neurons.

**Known issue:** Copies have random connections that score ~5.38, while lucky originals
score ~5.24. This dilutes the GA's starting population.

---

## 2. GA Phases (ga_neurons, ga_bits, ga_connections)

### Core Algorithm

Population-based evolutionary optimization with:
- **Tournament selection** (size 3): pick 3 random, return best
- **Crossover** (70% rate): single-point at CLUSTER boundary
- **Mutation**: per-cluster with `mutation_rate` probability
- **Elitism**: top 10 (20%) preserved each generation

### What Each GA Phase Mutates

| Phase | bits_mutation_rate | neurons_mutation_rate | connections | Delta |
|-------|-------------------|----------------------|-------------|-------|
| ga_neurons | 0.0 | mutation_rate | only via neuron add/remove | neurons +/-3 |
| ga_bits | mutation_rate | 0.0 | only via bit count change | bits +/-3 |
| ga_connections | 0.0 | 0.0 | direct connection mutation | +/-2 positions |

### Crossover Details (Rust, `neighbor_search.rs:676`)

Cluster-level single-point crossover:
- Pick random crossover_point in [1, num_clusters)
- Child gets parent1's clusters [0..point) and parent2's clusters [point..end)
- This copies: neurons_per_cluster, bits_per_neuron, AND connections

**Key:** Crossover preserves connections from both parents. No new random connections
are created during crossover.

### Mutation Details

**When neurons change** (`adjust_connections_per_neuron`, Rust line 301):
- Existing neurons: connections PRESERVED exactly (no drift, since bits don't change)
- New neurons (added): connections CLONED from random template neuron in same cluster,
  with mandatory +/-1 or +/-2 drift on EVERY connection
- Removed neurons: connections deleted

**When bits change** (`adjust_connections_per_neuron`, Rust line 348):
- Existing connections: 10% chance of +/-2 drift per connection
- New connections (for added bits): fully random

**When only connections change** (`_mutate_connections_only`, Python line 811):
- Each connection has `mutation_rate` chance of +/-2 perturbation

### The ga_neurons Problem

In ga_neurons mode:
- `bits_mutation_rate = 0.0` -> bits never change -> existing connections preserved exactly
- `neurons_mutation_rate > 0` -> neuron counts change per cluster
- When neuron count increases: NEW neuron gets cloned connections with mandatory drift
- When neuron count decreases: neuron removed entirely

The population from grid search has ~10 genomes with good connections (CE 5.24) and
~40 with random connections (CE 5.38). Tournament selection picks mostly mediocre
parents. Offspring from mediocre parents are mediocre. The 10 elites persist unchanged
but can't spread their good connections.

**Question for discussion:** Should ga_neurons exist as a separate phase, or should
neuron count optimization be handled by neurogenesis only?

---

## 3. TS Phases (ts_neurons, ts_bits, ts_connections)

### Core Algorithm

Population-based Tabu Search with (mu + lambda) replacement:
- Start from single best genome + seeded population from previous GA phase
- Each iteration: generate `neighbors_per_iter` (50) neighbors via mutation
- Select elite sources (top 20% of population) as mutation bases
- Evaluate all neighbors, rank by fitness
- Merge neighbors + population, keep top N
- Add best neighbor's move to tabu list

### What Each TS Phase Mutates

Same as GA but NO crossover - mutation only:

| Phase | Mutates | Delta |
|-------|---------|-------|
| ts_neurons | neurons_per_cluster | +/-3 per cluster |
| ts_bits | bits_per_neuron | +/-3 per neuron |
| ts_connections | individual connections | +/-2 per connection |

### Tabu Mechanism

- **Move info:** tuple of cluster indices that were mutated
- **Tabu check:** move is tabu if >50% of its cluster indices overlap with any recent move
- **Tabu list size:** 10 (FIFO, oldest dropped)

### Key Difference from GA

GA uses crossover to COMBINE good solutions. TS uses single-point mutation from best
solutions. TS is better for fine-tuning; GA is better for exploration.

**Question for discussion:** Does TS suffer the same problem as GA when seeded with
a population of mixed-quality connections?

---

## 4. Neurogenesis

### Core Algorithm

Statistics-guided neuron addition and removal. Uses Rust in-place adaptation during
training -- NOT random mutation like GA/TS.

### Neuron ADDITION (adaptation.rs:959)

For each cluster, add a neuron if ALL conditions met:
1. Cluster NOT in cooldown (5 iterations after last change)
2. `error_rate > 0.5 * cluster_error_factor` (default 0.7 -> error > 35%)
3. `mean_fill_rate > expected_fill * cluster_fill_utilization` (default 0.5)
4. `current_neurons < initial_neurons * max_growth_ratio` (default 1.5x)
5. `random() < adaptation_rate` (cosine schedule: aggressive early, fades out)

**How new neurons are created:**
- Find BEST-PERFORMING neuron in cluster (highest accuracy)
- CLONE it (same bit count)
- Perturb each connection: delta in {-2, -1, 0, 0, 0, 1, 2} (biased toward no change)
- Up to `max_neurons_per_pass` (3) neurons added per cluster per pass

### Neuron REMOVAL / Apoptosis (adaptation.rs:985)

For each cluster, remove a neuron if:
1. Cluster NOT in cooldown
2. `current_neurons > min_neurons`
3. Neuron's SCORE = uniqueness * accuracy
4. Score is in bottom `neuron_prune_percentile` (10%) of cluster
5. Score < `cluster_mean * neuron_removal_factor` (default 0.5)

**Removal criteria:** Targets neurons that are both REDUNDANT (low uniqueness) and
INACCURATE (low accuracy). A neuron that's redundant but accurate is kept. A neuron
that's inaccurate but unique is kept.

### Adaptation Rate Schedule (Cosine Annealing)

```
Gen 0-9 (warmup):     rate = 0.0 (no changes)
Gen 10-37 (active):   rate = cosine(0->pi) = 1.0->0.0
Gen 38-49 (stabilize): rate = 0.0 (frozen)
```

### Key Difference from GA

| Aspect | GA Neurons | Neurogenesis |
|--------|-----------|--------------|
| Decision basis | Random mutation | Statistics (error, fill, uniqueness) |
| New neuron connections | Clone + mandatory drift | Clone + biased-toward-zero drift |
| Removal criteria | Random (via mutation) | Worst-scoring (redundant + inaccurate) |
| Scope | Whole population each gen | In-place during training |
| Rate control | Fixed mutation_rate | Cosine annealing schedule |

**Question for discussion:** Neurogenesis is strictly more intelligent than ga_neurons
for neuron count optimization. Should ga_neurons be replaced by neurogenesis entirely?

---

## 5. Synaptogenesis

### What it Does

Modifies the NUMBER of connections per neuron (bits_per_neuron). This is the same
dimension as ga_bits but uses statistics instead of random mutation.

### Connection PRUNING (adaptation.rs:532)

For each neuron with bits > min_bits:
1. Compute median entropy of its connections
2. Find lowest-entropy connection
3. If entropy < median * prune_entropy_ratio (0.3):
   - REMOVE that connection (bits_per_neuron -= 1)

**Rationale:** Low-entropy connections carry little information. The input bit is
either always 0 or always 1 for the training data, so the neuron's address doesn't
meaningfully depend on it.

### Connection GROWING (adaptation.rs:550)

For each neuron with bits < max_bits:
1. If fill_rate > expected * grow_fill_utilization (0.5)
   AND error_rate > grow_error_baseline (0.35):
   - Find highest-entropy UNCONNECTED input bit
   - ADD that connection (bits_per_neuron += 1)

**Rationale:** High fill + high error = neuron has enough training data but still
makes mistakes. Adding a high-entropy (informative) connection gives it more
discriminating power.

### Key Difference from GA Bits

| Aspect | GA Bits | Synaptogenesis |
|--------|---------|---------------|
| Decision basis | Random mutation | Entropy analysis |
| Which connections change | Random neurons | Worst-performing neurons |
| New connections | Random bits | Highest-entropy unused bits |
| Removed connections | Random bits | Lowest-entropy bits |

---

## 6. Axonogenesis

### What it Does

REWIRES individual connections (replaces which input bit a connection points to)
without changing the number of connections. Same dimension as ga_connections but
statistics-guided.

### Three-Stage Algorithm (adaptation.rs:654)

**Stage 1 - Find weak connections:**
- Compute median entropy per neuron
- Find connections with entropy < median * 0.3
- Keep top 2 weakest

**Stage 2 - Test replacements:**
- For each weak connection x each candidate replacement:
  - Compute accuracy delta across sampled training examples
  - Skip if delta <= 0

**Stage 3 - Redundancy penalty:**
- Compute Jaccard similarity between candidate and existing connections
- Adjusted score = delta * (1.0 - 0.5 * max_jaccard)
- Penalizes redundant connections

**Stage 4 - Rewire:**
- Replace weak connection with best candidate

### Key Difference from GA Connections

| Aspect | GA Connections | Axonogenesis |
|--------|---------------|-------------|
| Decision basis | Random +/-2 perturbation | MI-guided replacement |
| Scope | Any connection | Only weak connections |
| Replacement | Adjacent bit (+/-2) | Any unconnected high-entropy bit |
| Redundancy check | None | Jaccard penalty |

---

## Summary: Separation of Concerns

| Dimension | GA Phase | *genesis Phase | TS Phase |
|-----------|----------|---------------|----------|
| **Neuron count** | ga_neurons (random) | neurogenesis (stats) | ts_neurons (mutation) |
| **Bit count** | ga_bits (random) | synaptogenesis (entropy) | ts_bits (mutation) |
| **Connections** | ga_connections (random) | axonogenesis (MI) | ts_connections (mutation) |

### Discussion Points for Tomorrow

1. **GA vs *genesis overlap:** Each GA phase and its corresponding *genesis phase
   optimize the same dimension. GA uses random search, *genesis uses statistics.
   Should we keep both, or replace GA with *genesis?

2. **ga_neurons specifically:** Changing neuron counts necessarily creates new neurons
   with random-ish connections. This degrades CE, making the phase ineffective when
   connections dominate performance. Neurogenesis handles this better because it
   clones the BEST neuron and uses smarter perturbation.

3. **Crossover in GA:** Currently cluster-level. Would neuron-level crossover
   (mixing individual neurons between parents) be more useful?

4. **Grid search expansion:** Currently creates copies with fully random connections.
   Should copies clone+perturb the original's connections, or should fewer copies
   be created?

5. **Phase order:** Currently neurons_first: GA_neurons -> neurogenesis -> TS_neurons
   -> GA_bits -> synaptogenesis -> TS_bits -> GA_connections -> axonogenesis ->
   TS_connections. Is this the right order?

6. **Train subset consistency:** Fixed within each phase (our recent fix). Each phase
   gets its own fixed train subset via `next_train_idx()`.
