# Fitness Calculators

## The Problem

During GA/TS optimization, genomes must be **ranked** to decide which survive (elitism), which become parents (selection), and which are "best" (result). But we track two objectives:

- **CE (Cross-Entropy)**: How well the model predicts the next token (lower = better)
- **Accuracy**: Fraction of predictions where the top-1 token is correct (higher = better)

These don't always agree. A genome might have the lowest CE but mediocre accuracy, or vice versa. The fitness calculator defines **how to combine them into a single ranking**.

## Available Calculators

### CE (`FitnessCalculatorType.CE`)

The simplest approach: rank purely by cross-entropy, ignore accuracy entirely.

```
fitness = CE    (lower = better)
```

**When to use:** When you only care about the probability distribution quality, not top-1 predictions. This was the original default.

**Trade-off:** Can produce genomes with excellent CE but poor accuracy — the model becomes good at spreading probability mass but bad at concentrating it on the right answer.

---

### Harmonic Rank (`FitnessCalculatorType.HARMONIC_RANK`)

The **default** calculator. Ranks genomes by **harmonic mean of their CE rank and accuracy rank** within the population. This is a rank-based multi-objective approach.

**Step 1 — Rank each metric independently:**

| Genome | CE    | Acc   | CE Rank | Acc Rank |
|--------|-------|-------|---------|----------|
| A      | 10.32 | 0.01% | 1       | 5        |
| B      | 10.35 | 0.04% | 3       | 1        |
| C      | 10.34 | 0.03% | 2       | 2        |
| D      | 10.36 | 0.02% | 4       | 3        |
| E      | 10.38 | 0.01% | 5       | 4        |

CE rank 1 = lowest CE (best). Accuracy rank 1 = highest accuracy (best).

**Step 2 — Compute harmonic mean of ranks:**

```
HM(rank_ce, rank_acc) = 2 / (1/rank_ce + 1/rank_acc)
                       = 2 × rank_ce × rank_acc / (rank_ce + rank_acc)
```

| Genome | CE Rank | Acc Rank | HM   |
|--------|---------|----------|------|
| A      | 1       | 5        | 1.67 |
| B      | 3       | 1        | 1.50 |
| C      | 2       | 2        | 2.00 |
| D      | 4       | 3        | 3.43 |
| E      | 5       | 4        | 4.44 |

**Result:** B wins (HM=1.50) — not the best CE, not the best accuracy, but the best *balance*.

**Why harmonic mean?** The harmonic mean penalizes imbalance more strongly than the arithmetic mean. Being rank 1 in CE but rank 50 in accuracy gives HM ≈ 1.96 (bad), whereas arithmetic mean = 25.5 would look "okay". The harmonic mean says: you must be good at *both*.

#### Weighted Harmonic Rank

When weights differ from the default (1.0, 1.0), the formula becomes:

```
WHM = (w_ce + w_acc) / (w_ce/rank_ce + w_acc/rank_acc)
```

With `w_ce=1.2, w_acc=1.0` (CE is 20% more important):

| Genome | CE Rank | Acc Rank | WHM (1.2, 1.0) | WHM (1.0, 1.0) |
|--------|---------|----------|-----------------|-----------------|
| A      | 1       | 5        | **1.57**        | 1.67            |
| B      | 3       | 1        | 1.69            | **1.50**        |

With equal weights, B wins. With CE weighted 20% higher, A wins — its rank-1 CE matters more.

**Configuration:**
```python
GAConfig(
    fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
    fitness_weight_ce=1.0,   # Higher = CE matters more
    fitness_weight_acc=1.0,  # Higher = accuracy matters more
)
```

---

### Normalized (`FitnessCalculatorType.NORMALIZED`)

Normalizes CE and accuracy to [0, 1] scale (relative to current population), then combines with weighted arithmetic mean.

```
norm_ce  = (ce - min_ce) / (max_ce - min_ce)     → 0 = best CE, 1 = worst
norm_acc = (max_acc - acc) / (max_acc - min_acc)  → 0 = best acc, 1 = worst

fitness = (w_ce × norm_ce + w_acc × norm_acc) / (w_ce + w_acc)
```

**Difference from Harmonic Rank:** Uses actual values (normalized) rather than rank positions. A genome that's 10x better in CE than the 2nd-place will get a proportionally better score, not just rank 1. Harmonic Rank is ordinal (only relative position matters); Normalized is cardinal (magnitude matters).

**Trade-off:** Sensitive to outliers. One genome with extremely bad CE can compress the normalization range, making all other differences look tiny.

---

### Normalized Harmonic (`FitnessCalculatorType.NORMALIZED_HARMONIC`)

Same normalization as `NORMALIZED`, but combines using harmonic mean instead of arithmetic:

```
fitness = (w_ce + w_acc) / (w_ce/norm_ce + w_acc/norm_acc)
```

This gives the strongest penalty for imbalance: a genome must have good normalized scores in *both* CE and accuracy.

**Trade-off:** Can be unstable when a genome is near-perfect in one metric (norm → 0), as the harmonic mean approaches 0 regardless of the other metric. An epsilon (1e-6) prevents division by zero but the behavior near zero can still dominate.

---

### Accuracy Floor (wrapper)

Not a standalone calculator — wraps any of the above and enforces a minimum accuracy threshold. Genomes below the floor get `fitness = ∞` (effectively eliminated).

```python
# Genomes with accuracy < 0.3% are eliminated regardless of CE
GAConfig(
    fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
    min_accuracy_floor=0.003,
)
```

**Why?** Prevents a pathological optimization direction where reducing neuron activity produces more uniform predictions (lower CE) but near-random accuracy. The floor ensures the GA can't "cheat" by predicting uniform distributions.

## Comparison

| Calculator | Basis | Combines via | Outlier-robust | Penalizes imbalance |
|------------|-------|-------------|----------------|---------------------|
| CE | Values | N/A (single metric) | Yes | N/A |
| Harmonic Rank | Ranks | Harmonic mean | Yes | Strong |
| Normalized | Values | Arithmetic mean | No | Weak |
| Normalized Harmonic | Values | Harmonic mean | No | Very strong |

**Recommendation:** `HARMONIC_RANK` (the default) is the safest choice — it's robust to outliers and naturally balances both objectives. Use `CE` only when accuracy genuinely doesn't matter. Use `NORMALIZED` variants when the magnitude of differences (not just ordering) should influence selection.

## How Fitness Interacts with GA and TS

The fitness calculator is used in several places:

- **GA elitism**: Top 20% of the population by fitness score are preserved (see [GA docs](ga.md#elitism))
- **GA selection**: Tournament selection compares fitness scores
- **TS source selection**: When using [cooperative multi-start](ts.md#cooperative-multi-start-variant), the reference set is the top N% of the neighbor cache by fitness
- **TS aspiration**: A tabu move is allowed if it produces fitness better than the current global best
- **Both**: Best genome tracking (overall result) uses fitness ranking
