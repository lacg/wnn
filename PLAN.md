# Analysis: Why Accuracy Isn't Improving

## Summary of Findings

**ROOT CAUSE IDENTIFIED**: The Rust `search_offspring` function is mutating ALL 50,257 clusters instead of just the tier0 (first 100) clusters.

---

## 1. Is Only Tier 0 Being Affected?

**NO - BUG FOUND!**

The log confirms tier0-only mode is configured:
```
DEBUG: optimize_tier0_only=True, tier_config=[(100, 15, 20), (400, 10, 12), (None, 5, 8)]
Tier0-only mode: mutating first 100 clusters
```

However, the **Rust `search_offspring` function ignores this setting**. In `lib.rs:3163-3164`:
```rust
let num_clusters = if !population.is_empty() {
    population[0].0.len()  // Returns 50,257, not 100!
```

Then in `neighbor_search.rs:198`:
```rust
for i in 0..config.num_clusters {  // Iterates over ALL 50,257 clusters!
    if rng.gen::<f64>() < config.neurons_mutation_rate {
        // Mutates neurons for ALL clusters, not just tier0
```

**Impact**: With 10% mutation rate:
- Expected per offspring: 10 tier0 clusters mutated
- Actual per offspring: ~5,026 clusters mutated (all tiers!)

---

## 2. Are Bits/Neurons Changing Drastically?

**BITS**: Fixed correctly (bits_mutation_rate=0.0)

**NEURONS**: Yes, but across ALL tiers (bug):
- Tier0: 15 neurons → mutated within [5-15] ✓
- Tier1: 10 neurons → **being mutated even though it shouldn't be!** ✗
- Tier2: 5 neurons → **being mutated even though it shouldn't be!** ✗

The delta range is ±2 (10% of 5+15=20), which is reasonable for tier0 but disruptive for tier1/2.

---

## 3. Elite Selection in First 3 Generations

**ELITE SELECTION IS CORRECT** (using NormalizedHarmonic with ce=0, acc=1):

| Gen | Elite #1 Acc | Elite #2 Acc | Elite #3 Acc |
|-----|--------------|--------------|--------------|
| 1   | 0.8341%      | 0.7421%      | 0.7421%      |
| 2   | 0.8341%      | 0.7421%      | 0.7421%      |
| 3   | 0.8341%      | 0.7421%      | 0.7421%      |
| 7   | **0.8361%**  | 0.8341%      | 0.7421%      |
| 23  | **0.9201%**  | 0.8361%      | 0.8341%      |

The elites ARE being selected by accuracy (since ce=0, acc=1). Small improvements do occur (0.8341% → 0.9201% by Gen 23), but very slowly.

---

## 4. Why Is Accuracy Struggling Even with ce=0, acc=1?

**Because most offspring have TERRIBLE accuracy due to the tier1/tier2 mutations**:

| Gen | Best Offspring Acc | Worst Offspring Acc | Elite Acc |
|-----|-------------------|---------------------|-----------|
| 1   | 0.61%             | 0.21%               | 0.83%     |
| 2   | 0.58%             | 0.08%               | 0.83%     |
| 3   | 0.35%             | 0.03%               | 0.83%     |
| 4   | 0.60%             | **0.006%**          | 0.83%     |

Offspring accuracy is **consistently worse** than the elites because mutations to tier1/tier2 are destroying accuracy while tier0 changes might be beneficial.

---

## 5. Are Neurons Actually Changing?

**YES, but in the WRONG tiers!**

For each offspring, ~10% of ALL clusters get neuron mutations:
- ~10 tier0 clusters (correct)
- ~40 tier1 clusters (WRONG - should be fixed)
- ~5,000 tier2 clusters (WRONG - should be fixed)

---

## 6. How Are Grouping Strategies Working?

The coalescing groups neurons by buckets (1-5→5, 6-10→10, etc.), but this is orthogonal to the mutation bug. The grouping works correctly for evaluation - the problem is that mutations are creating too many different configurations.

---

## Recommended Fix

Add `mutable_clusters` parameter to Rust `search_offspring`:

### In `lib.rs`:
```rust
// Add to signature:
mutable_clusters: Option<usize>,  // None = all clusters

// In GAConfig creation:
let effective_mutable = mutable_clusters.unwrap_or(num_clusters);
let ga_config = neighbor_search::GAConfig {
    num_clusters,
    mutable_clusters: effective_mutable,  // Add this field
    ...
};
```

### In `neighbor_search.rs`:
```rust
pub struct GAConfig {
    pub num_clusters: usize,
    pub mutable_clusters: usize,  // Add this
    ...
}

// In mutate_genome:
for i in 0..config.mutable_clusters {  // Was: config.num_clusters
    // Only mutate first N clusters (tier0)
```

### In Python `cached_evaluator.py`:
```python
def search_offspring(
    ...
    mutable_clusters: Optional[int] = None,  # Add this
```

---

## Files to Modify

1. `src/wnn/ram/strategies/accelerator/lib.rs` - Add parameter, pass to GAConfig
2. `src/wnn/ram/strategies/accelerator/neighbor_search.rs` - Add field, use in mutation loop
3. `src/wnn/ram/architecture/cached_evaluator.py` - Add parameter, pass to Rust
4. `src/wnn/ram/strategies/connectivity/architecture_strategies.py` - Pass mutable_clusters to search_offspring

---

## Verification

After fix, offspring accuracy should be comparable to elite accuracy (same tier1/tier2, varied tier0 only).
