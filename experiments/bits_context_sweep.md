# RAMLM Bits & Context Sweep Experiments

## Purpose
Test how bits_per_neuron and context_size affect model performance.

## Baseline Configuration
- Mode: fast (50k train, 10k test/val)
- Architecture: Tiered (100×11, 400×7, rest×5)
- Strategy: GA,TS
- GA: 5 pop × 5 gens
- TS: 10 neighbors × 3 iters

---

## Results Summary

### 5 Neurons/Cluster (rest tier)

| Mode | Bits | Ctx | Tiers | Total Neurons | Time | Val PPL | Val Acc | Test PPL | Test Acc | Notes |
|------|------|-----|-------|---------------|------|---------|---------|----------|----------|-------|
| fast | 8    | 4   | 100×11, 400×7, rest×5 | 252,685 | ~15m | 40,942  | 6.72%   | 40,506   | 6.28%    | Baseline |
| fast | 10   | 4   | 100×11, 400×7, rest×5 | 252,685 | ~30m | 43,204  | 8.62%   | 42,961   | 8.52%    | More bits = worse PPL, better acc |
| fast | 12   | 4   | 100×11, 400×7, rest×5 | 252,685 | ~60m | 45,516  | 9.84%   | 45,434   | 9.24%    | Trend continues |
| fast | 8    | 5   | 100×11, 400×7, rest×5 | 252,685 | ~30m | 41,071  | 6.21%   | 40,611   | 5.95%    | +1 context, no improvement |
| fast | 8    | 6   | 100×11, 400×7, rest×5 | 252,685 | ~30m | 41,074  | 6.75%   | 40,574   | 6.18%    | +2 context, no improvement |

### 3 Neurons/Cluster (rest tier only)

| Mode | Bits | Ctx | Tiers | Total Neurons | Time | Val PPL | Val Acc | Test PPL | Test Acc | Notes |
|------|------|-----|-------|---------------|------|---------|---------|----------|----------|-------|
| fast | 8    | 4   | 100×11, 400×7, rest×3 | 153,171 | ~5m  | 40,906  | 6.12%   | 40,539   | 5.71%    | PPL same, acc slightly worse |
| fast | 10   | 4   | 100×11, 400×7, rest×3 | 153,171 | ~10m | 43,237  | 8.34%   | 43,024   | 7.89%    | PPL same, acc slightly worse |
| fast | 12   | 4   | 100×11, 400×7, rest×3 | 153,171 | ~24m | 45,606  | 8.44%   | 45,409   | 8.60%    | PPL same, acc slightly worse |

### 5n vs 3n Comparison (rest tier only)

| Bits | 5n PPL | 5n Acc | 3n PPL | 3n Acc | PPL Δ | Acc Δ | Neurons Δ |
|------|--------|--------|--------|--------|-------|-------|-----------|
| 8    | 40,506 | 6.28%  | 40,539 | 5.71%  | +0.1% | -9.1% | -39% |
| 10   | 42,961 | 8.52%  | 43,024 | 7.89%  | +0.1% | -7.4% | -39% |
| 12   | 45,434 | 9.24%  | 45,409 | 8.60%  | -0.1% | -6.9% | -39% |

**Conclusion:** 3 neurons/cluster in rest tier gives nearly identical PPL with ~7-9% worse accuracy, but uses **39% fewer neurons** and runs **3-4x faster**.

### 3 Neurons/Cluster (tier 1 + rest)

| Mode | Bits | Ctx | Tiers | Total Neurons | Time | Val PPL | Val Acc | Test PPL | Test Acc | Notes |
|------|------|-----|-------|---------------|------|---------|---------|----------|----------|-------|
| fast | 8    | 4   | 100×11, 400×3, rest×3 | 151,571 | ~5m  | 40,981  | 5.39%   | 40,523   | 4.94%    | PPL same, acc worse |
| fast | 10   | 4   | 100×11, 400×3, rest×3 | 151,571 | ~10m | 43,242  | 7.80%   | 43,024   | 7.50%    | PPL same, acc worse |
| fast | 12   | 4   | 100×11, 400×3, rest×3 | 151,571 | ~24m | 45,370  | 8.32%   | 45,244   | 8.04%    | PPL same, acc worse |

### 3 Neurons/Cluster (all tiers)

| Mode | Bits | Ctx | Tiers | Total Neurons | Time | Val PPL | Val Acc | Test PPL | Test Acc | Notes |
|------|------|-----|-------|---------------|------|---------|---------|----------|----------|-------|
| fast | 8    | 4   | 100×3, 400×3, rest×3 | 150,771 | ~5m  | 40,650  | 2.74%   | 40,194   | 1.95%    | Best PPL! Worst acc |
| fast | 10   | 4   | 100×3, 400×3, rest×3 | 150,771 | ~10m | 42,783  | 4.89%   | 42,527   | 4.68%    | Good PPL, poor acc |
| fast | 12   | 4   | 100×3, 400×3, rest×3 | 150,771 | ~24m | 45,094  | 5.36%   | 44,807   | 4.75%    | Good PPL, poor acc |

### 1 Neuron/Cluster (all tiers)

| Mode | Bits | Ctx | Tiers | Total Neurons | Time | Val PPL | Val Acc | Test PPL | Test Acc | Notes |
|------|------|-----|-------|---------------|------|---------|---------|----------|----------|-------|
| fast | 8    | 4   | 100×1, 400×1, rest×1 | 50,257  | ~1m  | 39,888  | 0.14%   | 39,538   | 0.15%    | BEST PPL! ~random acc |
| fast | 10   | 4   | 100×1, 400×1, rest×1 | 50,257  | ~2m  | 41,721  | 0.53%   | 41,346   | 0.57%    | No voting = no acc |
| fast | 12   | 4   | 100×1, 400×1, rest×1 | 50,257  | ~8m  | 43,443  | 1.09%   | 43,114   | 1.20%    | No voting = no acc |

### Full Summary: Neurons vs PPL vs Accuracy

| Config | 8-bit PPL | 8-bit Acc | 10-bit PPL | 10-bit Acc | 12-bit PPL | 12-bit Acc |
|--------|-----------|-----------|------------|------------|------------|------------|
| 5n baseline | 40,506 | 6.28% | 42,961 | 8.52% | 45,434 | 9.24% |
| 3n (rest) | 40,539 | 5.71% | 43,024 | 7.89% | 45,409 | 8.60% |
| 3n (tier1+rest) | 40,523 | 4.94% | 43,024 | 7.50% | 45,244 | 8.04% |
| 3n (all) | 40,194 | 1.95% | 42,527 | 4.68% | 44,807 | 4.75% |
| 1n (all) | 39,538 | 0.15% | 41,346 | 0.57% | 43,114 | 1.20% |

**Key Insights:**
1. **PPL improves with fewer neurons** - 1n gives best PPL (39,538 vs 40,506 baseline)
2. **Accuracy requires voting** - 1n has ~random accuracy (0.15%), 3n is poor (1.95%)
3. **Tier 0 neurons matter most for accuracy** - going 11n→3n in tier 0 destroys accuracy
4. **Trade-off is severe** - 2.4% PPL improvement costs 97% accuracy with 1n

---

## Mechanism Analysis: Why PPL and Accuracy Trade Off

### Why Fewer Neurons → Better PPL

**The Voting/Hedging Mechanism:**

When multiple neurons per cluster have different connectivity maps, they see different "views" of the input and often disagree on predictions:

```
Example: Token cluster with 5 neurons
  Neuron 1 (sees bits 2,5,11): predicts "the" with 100% confidence
  Neuron 2 (sees bits 3,8,14): predicts "a" with 100% confidence
  Neuron 3 (sees bits 1,6,12): predicts "the" with 100% confidence
  Neuron 4 (sees bits 4,9,15): predicts "an" with 100% confidence
  Neuron 5 (sees bits 2,7,13): predicts "the" with 100% confidence

  Aggregated output: "the" 60%, "a" 20%, "an" 20%
```

This **hedging** behavior hurts PPL:
- Cross-entropy = -log(P(correct))
- If correct answer is "the": -log(0.60) = 0.51 nats
- With 1 neuron (100% confident): -log(1.0) = 0.0 nats

**With 1 neuron per cluster:**
- Prediction is always 100% or 0% (binary RAM output)
- No probability dilution from disagreeing neurons
- When correct → perfect cross-entropy (0 nats)
- Result: Lower PPL (39,538 vs 40,506 baseline)

**But accuracy collapses:**
- Accuracy only cares about argmax (is top prediction correct?)
- With 5 neurons voting, the "ensemble" can correct individual neuron errors
- With 1 neuron, a single wrong RAM lookup = wrong answer
- Result: 0.15% accuracy (essentially random guessing)

### Why More Bits → Worse PPL but Better Accuracy

**The Collision/Isolation Mechanism:**

Bits per neuron determines the address space size:
- 8 bits → 256 addresses
- 10 bits → 1,024 addresses
- 12 bits → 4,096 addresses

**Fewer bits = More collisions = Implicit smoothing:**

```
8-bit neuron: Many different contexts hash to same address
  Context "the quick brown" → address 42
  Context "a fast brown"    → address 42  (collision!)

  RAM[42] trained on both → predicts average of their targets
  Result: Smoother, less confident predictions
```

**More bits = Fewer collisions = Spikier predictions:**

```
12-bit neuron: Fewer collisions, more isolated predictions
  Context "the quick brown" → address 2847
  Context "a fast brown"    → address 1392  (no collision)

  Each address trained on specific context → confident predictions
  Result: Spikier, more confident predictions
```

**The trade-off in action:**

| Scenario | 8-bit (smooth) | 12-bit (spiky) |
|----------|----------------|----------------|
| Correct prediction | P=0.6, CE=0.51 | P=0.9, CE=0.11 |
| Wrong prediction | P=0.3 on wrong, CE=1.2 | P=0.9 on wrong, CE=2.3 |

- **Accuracy favors spiky**: Higher confidence on correct → more wins in argmax
- **PPL penalizes confident wrong**: A 90% confident wrong answer is worse than 30%

**Why collisions help PPL:**
- Collisions force probability mass to spread across likely tokens
- This "hedges" against confident wrong answers
- Even when wrong, the model isn't catastrophically confident
- Net effect: Lower cross-entropy on average

**Why isolation helps accuracy:**
- Spikier predictions win more argmax battles
- Even if PPL is higher due to occasional confident mistakes
- Net effect: Higher accuracy

### Summary: The Fundamental Trade-off

| Metric | Optimized by | Mechanism |
|--------|--------------|-----------|
| **PPL** | Fewer neurons + fewer bits | Avoid hedging + leverage collision smoothing |
| **Accuracy** | More neurons + more bits | Ensemble voting + isolated confident predictions |

**Best configurations by goal:**

| Goal | Config | PPL | Accuracy |
|------|--------|-----|----------|
| Best PPL | 1n-all, 8-bit | 39,538 | 0.15% |
| Best Accuracy | 5n-baseline, 12-bit | 45,434 | 9.24% |
| Balanced | 5n-baseline, 8-bit | 40,506 | 6.28% |

**The insight for architecture design:**
- More neurons = better accuracy (voting ensemble)
- More bits = better accuracy but worse PPL (isolation)
- Tier 0 (top 100 tokens) needs more neurons for accuracy (49% of data)
- Rare tokens (tier 2) can use fewer neurons with minimal PPL impact

---

## Key Observations

### Bits Comparison (complete)
- **8 bits**: PPL 40,506, Acc 6.28%
- **10 bits**: PPL 42,961, Acc 8.52%
- **12 bits**: PPL 45,434, Acc 9.24%

**Trend confirmed:** More bits = worse PPL but better accuracy!
- PPL increases ~12% per +2 bits (40K → 43K → 45K)
- Accuracy increases ~47% per +2 bits (6.28% → 8.52% → 9.24%)
- This suggests higher bits = more precise predictions but fewer address collisions = less probability spreading

---

## Detailed Results

### Test 1: 8 bits × 4 context (baseline)
- Status: ✅ Complete
- Config: `--tiered "100,11,8;400,7,8;rest,5,8" --context 4`
- Results:
  - Val PPL: 40,942 | Val Acc: 6.72%
  - Test PPL: 40,506 | Test Acc: 6.28%

### Test 2: 10 bits × 4 context
- Status: ✅ Complete
- Config: `--tiered "100,11,10;400,7,10;rest,5,10" --context 4`
- Results:
  - Val PPL: 43,204 | Val Acc: 8.62%
  - Test PPL: 42,961 | Test Acc: 8.52%

### Test 3: 12 bits × 4 context
- Status: ✅ Complete
- Config: `--tiered "100,11,12;400,7,12;rest,5,12" --context 4`
- Results:
  - Val PPL: 45,516 | Val Acc: 9.84%
  - Test PPL: 45,434 | Test Acc: 9.24%
- Notes: 14x slower than 8-bit (135s vs 10s training) due to 16x larger address space

### Test 4: 8 bits × 5 context
- Status: ✅ Complete
- Config: `--tiered "100,11,8;400,7,8;rest,5,8" --context 5`
- Results:
  - Val PPL: 41,071 | Val Acc: 6.21%
  - Test PPL: 40,611 | Test Acc: 5.95%
- Notes: More context doesn't help

### Test 5: 8 bits × 6 context
- Status: ✅ Complete
- Config: `--tiered "100,11,8;400,7,8;rest,5,8" --context 6`
- Results:
  - Val PPL: 41,074 | Val Acc: 6.75%
  - Test PPL: 40,574 | Test Acc: 6.18%
- Notes: More context doesn't help

---

## Roadmap: Next Experiments

### 1. Per-Tier PPL/Accuracy Metrics
**Goal:** Understand which tier contributes most to errors

**Implementation:**
- Modify evaluation to track per-tier statistics
- Separate predictions by token frequency tier
- Metrics: PPL, accuracy, confidence distribution per tier

**Hypothesis:**
- Tier 0 (top 100, 49% of data) dominates overall metrics
- Tier 2 (rare tokens) likely has worst accuracy but minimal PPL impact
- This will inform where to allocate neurons/bits

### 2. Memory Hashing for 20-60 Bits
**Goal:** Scale beyond 12 bits without exponential memory growth

**Current limitation:**
- 12 bits = 4,096 addresses per neuron = 4KB RAM per neuron
- 20 bits = 1M addresses = 1MB per neuron (50GB for 50K vocab!)
- 60 bits = infeasible direct storage

**Proposed solutions:**

| Approach | Description | Trade-off |
|----------|-------------|-----------|
| **LSH buckets** | Hash 60-bit address to 12-bit bucket | Controlled collisions |
| **Bloom filters** | Probabilistic membership | False positives, no values |
| **Sparse storage** | Only store non-empty addresses | Memory proportional to data |
| **Hierarchical** | Tree of smaller RAMs | Slower lookup, more flexible |

**Experiment:** Compare 12-bit direct vs 20-bit with LSH hashing

### 3. Asymmetric Tier Configurations
**Goal:** Optimize neuron allocation based on tier importance

**Experiments to run:**

| Config | Rationale |
|--------|-----------|
| 11n tier0, 1n rest | Maximize tier 0 accuracy (49% of data) |
| 11n tier0, 3n tier1, 1n rest | Gradual reduction |
| 5n tier0, 5n tier1, 1n rest | Flat top tiers, minimal rest |

**Hypothesis:** Tier 0 neurons matter most; can reduce tier 2 to 1n with minimal accuracy loss

### 4. Hybrid Bit Configurations
**Goal:** Different bits per tier to balance PPL/accuracy

**Experiments to run:**

| Config | Rationale |
|--------|-----------|
| 12-bit tier0, 8-bit rest | High accuracy for common tokens, smooth rare |
| 8-bit tier0, 12-bit rest | Opposite: smooth common, precise rare |
| 10-bit all tiers | Uniform middle ground |

**Hypothesis:** 12-bit tier0 + 8-bit rest may give best of both worlds

### 5. Optimal Tier 0 Neuron Count
**Goal:** Find diminishing returns point for tier 0 neurons

**Experiments to run:**
- 100×3, 100×5, 100×7, 100×9, 100×11, 100×15, 100×20 (tier 0 only)
- Keep tier 1 and rest at 1n to isolate tier 0 effect

**Expected outcome:** Accuracy improvement curve that flattens around some neuron count

---

## Future Directions

### Architecture Improvements
- [ ] Learned connectivity via backprop (vs random + optimization)
- [ ] Attention over context tokens (vs fixed concatenation)
- [ ] Hierarchical token clustering (vs frequency-based tiers)

### Training Improvements
- [ ] Curriculum learning: easy contexts first
- [ ] Hard example mining: focus on high-loss contexts
- [ ] Ensemble of models with different random seeds

### Evaluation Improvements
- [ ] Per-token-frequency accuracy curves
- [ ] Confidence calibration analysis
- [ ] Error analysis: which contexts fail?
