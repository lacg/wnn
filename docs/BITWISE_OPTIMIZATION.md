# Bitwise Architecture Optimization — Research Progress

**Started:** 2026-02-10
**Status:** Active (v3 running)

---

## What Is BitwiseRAMLM?

A 16-cluster language model where each cluster has independently configurable:
- **Neurons per cluster** (10–300)
- **Bits per neuron** (10–24)
- **Connection patterns** (which input bits each neuron observes)

Unlike the tiered RAMLM (which groups tokens by frequency into 2-5 tiers with uniform configs per tier), BitwiseRAMLM treats each of the 16 output-bit clusters independently. The 16 clusters correspond to `bits_needed(vocab_size)` — each cluster learns one bit of the output token encoding.

**Evaluation backend:** Rust + Metal GPU via `BitwiseEvaluator`. Training uses DashMap (lock-free), evaluation exports to sorted arrays for GPU binary search.

---

## 7-Phase Optimization Pipeline

| Phase | What's Optimized | Method | Seeds From |
|-------|-----------------|--------|------------|
| 1 | Grid search (neurons × bits) | Exhaustive 4×4 grid | — |
| 2 | Neurons per cluster | GA (250 gens, pop=50) | Phase 1 top-3 |
| 3 | Neurons refinement | TS (250 iters, 30 neighbors) | Phase 2 best |
| 4 | Bits per cluster | GA | Phase 3 best |
| 5 | Bits refinement | TS | Phase 4 best |
| 6 | Connections | GA | Phase 5 best |
| 7 | Connections refinement | TS | Phase 6 best |

Each GA/TS phase uses:
- **HARMONIC_RANK** fitness: weighted harmonic mean of CE rank + Acc rank
- **Progressive threshold**: 3.00% → 3.25% accuracy over 250 gens (0.001%/gen)
- **Patience**: 5 check intervals before early stop
- **Data rotation**: 36 train subsets + 6 eval subsets (prevents memorization)
- **Full validation**: After each phase, 1-3 genomes evaluated on held-out validation set

---

## Experiment Results

### v1: First run (2026-02-10)
- **Config**: 50 gens, pop=50, patience=2
- **Result**: CE=9.2650, PPL=10,561, Acc=5.20%
- **Grid search best**: n=200, b=20, CE=9.1461

### v2: Overnight run (2026-02-11 02:47)
- **Config**: 50 gens, pop=50, patience=2
- **Grid search**: Same rankings — n=200/b=20 wins
- **Phase 2 GA Neurons**: CE 9.153 → 9.145 (+0.09%)
- **Key issue**: Threshold started at 0% (should be 3%), ran too fast at 0.01%/gen

### v3: Current run (2026-02-11, in progress)
- **Config**: 250 gens/iters, pop=50, patience=5
- **Fixes applied**:
  - Threshold starts at 3%, rate 0.001%/gen (3.00% → 3.25% over 250 gens)
  - Standard `PhaseComparisonTable` for summary (not ad-hoc)
  - Per-batch logging enabled (shows `[Gen XX/250] 50 genomes in 42s`)
  - JSON serialization fixed (`PhaseMetrics` → `asdict()`)
- **Phase 1 Grid**: Best n=200, b=20, CE=9.1425
- **Phase 2 GA Neurons**: Running, Gen ~5/250 so far

---

## Key Findings

### 1. Grid Search Rankings Are Stable

Across all three runs, the Phase 1 grid produces the same top configs:

| Rank | Config | CE | Acc |
|------|--------|-----|------|
| 1 | n=200, b=20 | 9.142 | 6.42% |
| 2 | n=150, b=20 | 9.151 | 6.32% |
| 3 | n=200, b=18 | 9.153 | 6.55% |

**Takeaway**: More neurons + more bits = better, up to the point where address space becomes sparse. The 200/20 config has `2^20 = 1M` addresses per neuron with `200 × 16 = 3,200` total neurons.

### 2. BitwiseEvaluator Uses Python Fallback (Not Rust Offspring)

Unlike the n-gram `CachedEvaluator` which has `search_offspring()`/`search_neighbors()` for Rust-accelerated offspring generation, `BitwiseEvaluator` does NOT implement these methods. This means:

- All GA offspring generation falls through to the **Python path** in `GenericGAStrategy._generate_offspring()`
- This uses `_build_viable_population()` → `batch_fn()` → `BitwiseEvaluator.evaluate_batch()`
- The Rust+Metal acceleration is used for **evaluation** (train + score), but **genome generation** (crossover, mutation, tournament selection) is in Python

**Performance implication**: ~42s per generation for 50 genomes. Not as fast as Rust offspring search would be, but acceptable.

### 3. Progressive Threshold Matters

| Setting | Start | End (250 gens) | Effect |
|---------|-------|----------------|--------|
| v2 (broken) | 0.00% | 2.50% | Too permissive early, too aggressive late |
| v3 attempt 1 | 3.00% | 5.50% | 0.01%/gen too aggressive for TS |
| **v3 final** | **3.00%** | **3.25%** | **0.001%/gen — gentle pressure** |

The threshold filters out genomes with accuracy below the threshold. Too aggressive = kills diversity early. Too gentle = no selection pressure. The 0.001%/gen rate means by the time TS starts (phases 3/5/7), the threshold is still very manageable (~3.1-3.2%).

### 4. HARMONIC_RANK Balances CE and Accuracy

Previous runs using pure CE fitness collapsed accuracy to 0%. HARMONIC_RANK prevents this by requiring genomes to be good at BOTH metrics:

```
WHM = (w_ce + w_acc) / (w_ce/rank_ce + w_acc/rank_acc)
```

With equal weights, a genome that's rank 1 in CE but rank 50 in accuracy (WHM=1.96) loses to one that's rank 3 in both (WHM=3.0 → actually 3.0 > 1.96, so rank 1 CE still wins... but the gap is narrowed significantly compared to pure CE).

### 5. Standard Reporting Classes Prevent Drift

Using `PhaseComparisonTable` from `reporting.py` ensures every phase shows three rows:
- **Best fitness** (by harmonic rank)
- **Best CE** (lowest cross-entropy genome)
- **Best Acc** (highest accuracy genome)

Ad-hoc code tended to only show best CE, hiding accuracy degradation.

---

## Architecture Decisions

### Why 16 Clusters?

`bits_needed(vocab_size)` for GPT-2 (50,257 tokens) = 16 bits = 16 clusters. Each cluster predicts one bit of the 16-bit output encoding. This is simpler than tiered architectures (which group by frequency) but gives each output bit its own neuron/bit budget.

### Why QUAD_WEIGHTED Memory Mode?

Mode 2 (QUAD_WEIGHTED) uses 4 states per memory cell: FALSE(0), WEAK_FALSE(1), WEAK_TRUE(2), TRUE(3). This provides softer boundaries than binary (TRUE/FALSE) without the information loss of ternary (TRUE/FALSE/EMPTY).

### Why 36 Train Subsets + 6 Eval Subsets?

Data rotation prevents the optimizer from overfitting to a single train/eval split. Each generation sees a different ~67K-token slice of the 2.4M training tokens, scored on a different ~47K-token slice of the 285K test tokens.

---

## Next Steps (After v3 Completes)

### Context Sweep (`run_bitwise_context_sweep.py`)

Script is written and committed. After v3 finishes:

1. Read top-3 configs from v3 JSON (`phase1_grid.results`)
2. For each config, run full 7-phase pipeline at context sizes [2, 3, 5, 6, 7, 8, 16]
3. Skip context=4 (pull from v3 as baseline)
4. Produce comparison table: final CE/Acc for each context size

**Hypothesis**: Shorter context (2-3) may work better due to higher training density per address. This aligns with DeepSeek Engram findings (2-3 n-grams optimal).

### Gating Integration

After context sweep, integrate RAM-based gating:
- Gate neurons decide per-cluster whether to trust the prediction
- Allows selective abstention on low-confidence clusters
- Already implemented in `src/wnn/ram/core/gating.py`

### DeepSeek Engram Comparison

Benchmark our gated RAM WNN against DeepSeek's Engram on WikiText-2. Target: match or beat their 5-10% relative improvement.

---

## Code Changes Log

| Date | Commit | Change |
|------|--------|--------|
| 2026-02-10 | — | Initial `run_bitwise_optimization.py` |
| 2026-02-11 | `877d00b` | Fix threshold: start at 3% instead of 0% |
| 2026-02-11 | `847a9e2` | Reduce threshold rate: 0.01% → 0.001%/gen |
| 2026-02-11 | `52ea51c` | Add context sweep script |
| 2026-02-11 | `eca7114` | PhaseComparisonTable, per-batch logging, JSON fix |
