# Overnight Sweep Experiments

High-capacity experiments using the SPARSE memory backend (Rust FxHashMap) to push beyond the 10-bit dense limit.

## Memory Backend Thresholds

| Backend | Bits | Memory | Use Case |
|---------|------|--------|----------|
| DENSE | 1-10 | O(2^bits × neurons) | Small address spaces |
| SPARSE | 11-30 | O(written_cells) | Large address spaces |
| LSH | 31-60 | O(buckets) | Extreme configurations |

## Experiment Sets

### Quick Set (~4-6 hours)
4 experiments for weeknight runs.

| Name | Tier Config | Context | Rationale |
|------|-------------|---------|-----------|
| `tier0_16bit` | 100,15,16; 400,10,10; rest,5,8 | 4 | 16-bit SPARSE tier0 = 65K addresses per neuron |
| `balanced_14bit` | 100,12,14; 400,8,12; rest,5,10 | 4 | All tiers SPARSE with gradient 14→12→10 |
| `neurons_20_tier0` | 100,20,12; 400,12,10; rest,7,8 | 4 | High neuron count for better voting |
| `context_6_sparse` | 100,12,14; 400,8,10; rest,5,8 | 6 | Larger context window |

### Standard Set (~8-10 hours)
Quick + 2 additional experiments for full overnight.

| Name | Tier Config | Context | Rationale |
|------|-------------|---------|-----------|
| `tier0_18bit` | 100,15,18; 400,10,12; rest,5,8 | 4 | 18-bit = 262K addresses per neuron |
| `neurons_25_gradient` | 100,25,14; 400,15,10; rest,7,8 | 4 | Maximum neurons with SPARSE |

### Extended Set (~16-20 hours)
Standard + 4 additional experiments for weekend runs.

| Name | Tier Config | Context | Rationale |
|------|-------------|---------|-----------|
| `tier0_20bit` | 100,15,20; 400,10,12; rest,5,8 | 4 | 20-bit = 1M addresses per neuron |
| `all_sparse_16bit` | 100,15,16; 400,12,16; rest,8,16 | 4 | Uniform 16-bit across all tiers |
| `context_8_high_cap` | 100,15,12; 400,10,10; rest,7,8 | 8 | Maximum context window |
| `extreme_tier0` | 100,30,16; 400,12,12; rest,5,8 | 5 | 30 neurons, 16 bits - push limits |

## Usage

```bash
# Quick overnight (4 experiments, ~4-6 hours)
python tests/run_overnight_sweep.py --full-data --set quick

# Standard overnight (6 experiments, ~8-10 hours)
python tests/run_overnight_sweep.py --full-data --set standard

# Extended/weekend (10 experiments, ~16-20 hours)
python tests/run_overnight_sweep.py --full-data --set extended

# Single experiment test (fast mode)
python tests/run_overnight_sweep.py --experiments tier0_16bit

# Multiple specific experiments
python tests/run_overnight_sweep.py --full-data --experiments tier0_16bit,tier0_18bit
```

## Output

Results are saved to `experiments/overnight_sweep_results.json` with intermediate saves after each experiment.

## Baseline Results (Per-Tier Sweep)

Results from 15 experiments using DENSE backend (fast mode):

| Experiment | PPL | Accuracy | T0 PPL | T0 Acc | Notes |
|------------|-----|----------|--------|--------|-------|
| neurons_3_uniform | **40,515** | 2.70% | 34,072 | 5.7% | Best PPL |
| neurons_5_uniform | 40,564 | 4.99% | 34,063 | 10.5% | |
| neurons_7_uniform | 40,614 | 5.47% | 34,239 | 11.5% | |
| **baseline_8bit** | 40,651 | 6.93% | 34,286 | 14.6% | Reference |
| neurons_11_uniform | 40,654 | 6.59% | 34,288 | 13.9% | |
| context_5 | 40,752 | 6.09% | 34,355 | 12.8% | |
| context_6 | 40,835 | 6.29% | 34,387 | 13.2% | |
| bits_10_uniform | 43,294 | 7.87% | 38,041 | 16.5% | |
| bits_12_uniform | 45,753 | **9.37%** | 42,127 | **19.4%** | Best Acc |

### Key Findings

1. **Neurons vs PPL**: Fewer neurons (3-5) slightly improve PPL but significantly hurt accuracy
2. **Bits vs Accuracy**: More bits (10-12) improve accuracy by 35-50% but hurt PPL by 7-12%
3. **Context**: 5-6 tokens shows minimal improvement over 4 tokens
4. **Trade-off**: Clear PPL↔Accuracy trade-off with current DENSE backend

### Implications for Overnight Experiments

The SPARSE backend experiments (16-20 bits) may show different patterns because:
- Higher bits = more address space = less collision = potentially better PPL
- SPARSE only stores written cells, avoiding the "EMPTY cell" penalty
- Combined with more neurons, may break the PPL↔Accuracy trade-off

## Hypotheses

1. **More bits = better discrimination**: 16-20 bits should significantly outperform 8-bit for tier0 (frequent tokens) because each neuron can distinguish more patterns.

2. **Diminishing returns on neurons**: Going from 11→20 neurons should help, but 20→30 may show diminishing returns.

3. **Context trade-off**: Larger context provides more information but expands address space. Context 6-8 may help if bits are high enough.

4. **Tier0 focus**: Since tier0 handles 47% of data (frequent tokens), investing capacity there should yield the best PPL improvement.

---

## Full Data Results (January 2026)

### Complete Sweep Results

Full WikiText-2 dataset (2.4M train, 251K val, 288K test tokens) with GA+TS optimization.
Quick answer on your question: GPT-2 small (124M params) scores ~29.4 PPL / ~3.38 CE on WikiText-2 test. The larger variants: medium (355M) ~22.8 PPL, large (774M) ~19.9 PPL,  
  XL (1.5B) ~18.3 PPL.

#### Overall Rankings (by Test PPL)

| Rank | Experiment | Config | Ctx | Test PPL | Test Acc | Notes |
|------|------------|--------|-----|----------|----------|-------|
| 🥇 **1** | **tier0_20bit** | `100,15,20;400,10,12;rest,5,8` | 4 | **36,853** | **5.28%** | ⭐ Best overall |
| 🥈 2 | asymmetric_extreme_t0 | `100,25,24;400,8,10;rest,5,8` | 8 | 39,503 | 1.35% | 25n×24b tier0 |
| 🥉 3 | context_8_high_cap | `100,15,12;400,10,10;rest,7,8` | 8 | 39,508 | 1.26% | Higher context |
| 4 | asymmetric_expanded_t0 | `200,25,20;300,8,10;rest,4,8` | 8 | 40,047 | 1.26% | 200 tokens tier0 |
| 5 | extreme_tier0 | `100,30,16;400,12,12;rest,5,8` | 5 | 40,934 | 0.13% | 30 neurons |
| 6 | five_tier_gradient | `50,37,22;50,31,20;400,11,12;20000,7,10;rest,3,11` | 8 | 41,001 | 0.83% | 5-tier |
| 7 | two_tier_simple | `500,15,16;rest,4,6` | 8 | 41,883 | 0.03% | 2-tier only |
| 8 | neurons_20_tier0 | `100,20,12;400,12,10;rest,7,8` | 4 | 46,358 | 0.02% | |
| 9 | context_6_sparse | `100,12,14;400,8,10;rest,5,8` | 6 | 46,584 | 0.02% | |
| 10 | neurons_25_gradient | `100,25,14;400,15,10;rest,7,8` | 4 | 46,677 | 0.02% | |
| 11 | tier0_16bit | `100,15,16;400,10,10;rest,5,8` | 4 | 46,787 | 0.01% | |
| 12 | all_sparse_16bit | `100,15,16;400,12,16;rest,8,16` | 4 | 47,060 | 2.70% | Uniform 16b |
| 13 | tier0_18bit | `100,15,18;400,10,12;rest,5,8` | 4 | 47,075 | 0.02% | |
| 14 | balanced_14bit | `100,12,14;400,8,12;rest,5,10` | 4 | 47,181 | 0.13% | |
| 15 | uniform_20bit_ctx8 | `100,15,20;400,10,20;rest,5,20` | 8 | 49,675 | 1.81% | ❌ Uniform worse |

#### Per-Tier Breakdown (Top 3 Experiments)

**🥇 tier0_20bit** (Best PPL: 36,853)
| Tier | Clusters | Neurons | Bits | Data % | PPL | Accuracy |
|------|----------|---------|------|--------|-----|----------|
| 0 | 100 | 15 | 20 | 46.5% | 34,067 | **11.23%** |
| 1 | 400 | 10 | 12 | 13.0% | 35,415 | 0.24% |
| 2 | 49,757 | 5 | 8 | 40.4% | 40,862 | 0.05% |

**🥈 context_8_high_cap** (PPL: 39,508)
| Tier | Clusters | Neurons | Bits | Data % | PPL | Accuracy |
|------|----------|---------|------|--------|-----|----------|
| 0 | 100 | 15 | 12 | 46.5% | 36,991 | 2.71% |
| 1 | 400 | 10 | 10 | 13.0% | 37,076 | 0.01% |
| 2 | 49,757 | 7 | 8 | 40.4% | 43,498 | 0.00% |

**❌ uniform_20bit_ctx8** (PPL: 49,675 - Failure Case)
| Tier | Clusters | Neurons | Bits | Data % | PPL | Accuracy |
|------|----------|---------|------|--------|-----|----------|
| 0 | 100 | 15 | 20 | 46.5% | 49,232 | 3.84% |
| 1 | 400 | 10 | 20 | 13.0% | 49,736 | 0.18% |
| 2 | 49,757 | 5 | 20 | 40.4% | 50,171 | 0.01% |

### Key Insights

#### ⭐ Asymmetric Architecture Wins

The **tier0_20bit** experiment demonstrates that asymmetric bit allocation dramatically outperforms uniform configurations:

| Config | Tier 0 | Tier 1 | Tier 2 | Test PPL |
|--------|--------|--------|--------|----------|
| **Asymmetric (winner)** | 20 bits | 12 bits | 8 bits | **36,853** |
| Uniform 20-bit | 20 bits | 20 bits | 20 bits | 49,675 |

**Why asymmetric works:**
- **Tier 0** (100 frequent tokens, 46% of data): Each token seen ~11K times → can fill large address spaces → benefits from 20 bits
- **Tier 2** (50K rare tokens, 40% of data): Each token seen ~20 times → can't fill even 8-bit address spaces → more bits = more empty cells = worse predictions

#### ❌ Uniform High-Bit Configurations Fail

The uniform_20bit experiment showed **worse results** than tier0_20bit:
- PPL increased 35% (36,853 → 49,675)
- Tier 0 accuracy dropped 66% (11.23% → 3.84%)

**Root cause**: Rare tokens in tier-2 have too few training examples to fill the 2^20 = 1M address space. Most cells remain empty, leading to near-random predictions.

#### ❌ Pushing Beyond Sweet Spot (Priority 4 Results)

The asymmetric experiments (completed 2026-01-11) tested pushing tier0 capacity further:

| Experiment | Change from Champion | Result | Finding |
|------------|---------------------|--------|---------|
| asymmetric_extreme_t0 | 15n→25n, 20b→24b | PPL +7.2% | Too much capacity |
| asymmetric_expanded_t0 | 100→200 tokens | PPL +8.7% | Spreads data too thin |
| two_tier_simple | 2 tiers only | PPL +13.6% | Loses tier granularity |

**Key insight**: The champion's configuration (15 neurons, 20 bits for top 100 tokens) is near-optimal. Going beyond shows diminishing returns - more capacity without proportionally more data leads to underfitting.

#### Next Experiment (Priority 5)

Fine-grained 5-tier architecture to test more granular frequency-based allocation:

| Name | Config | Ctx | Rationale |
|------|--------|-----|-----------|
| five_tier_gradient | `50,37,22;50,31,20;400,11,12;20000,7,10;rest,3,11` | 8 | 5-tier with neuron gradient |

Tier breakdown:
- Tier 0: 50 most frequent (37n×22b) - maximum capacity
- Tier 1: next 50 (31n×20b) - high capacity
- Tier 2: next 400 (11n×12b) - medium
- Tier 3: next 20K (7n×10b) - low
- Tier 4: rest ~30K (3n×11b) - minimal

### Hypotheses Status (Updated)

| Hypothesis | Status | Finding |
|------------|--------|---------|
| More bits = better | ⚠️ **Depends** | More bits help **only for frequent tokens** with enough training data |
| Uniform bits | ❌ **Rejected** | Asymmetric allocation dramatically outperforms uniform |
| Tier0 focus | ✅ **Confirmed** | Investing capacity in tier0 yields best returns |
| Context 8 | ⚠️ **Mixed** | Helps PPL but needs right bit configuration |
| More neurons/bits always better | ❌ **Rejected** | 15n×20b optimal; 25n×24b performs worse |
| Expanding tier0 tokens | ❌ **Rejected** | 100 tokens optimal; 200-500 spreads data too thin |

### Usage

```bash
# Quick sweep (4 experiments, ~4-6 hours)
python tests/ramlm_full_benchmark.py --sweep --set quick

# Extended sweep (10 experiments, ~16-20 hours)
python tests/ramlm_full_benchmark.py --sweep --set extended

# Run 5-tier experiment
python tests/ramlm_full_benchmark.py --sweep --experiments five_tier_gradient \
    --ga-gens 1000 --ts-iters 1000 --patience 5
```

---

## Bitwise Context Sweep (2026-02-11)

### Background

Systematic evaluation of context sizes 2-16 using the bitwise RAMLM architecture with full WikiText-2 data (2.4M train tokens). Each context size was tested with a 16-config grid search (neurons: 50, 100, 150, 200 x bits: 14, 16, 18, 20) followed by full 7-phase optimization (GA+TS for neurons, bits, and connections).

### Results Summary

| Context | Best CE | Best PPL | Best Acc | Winner Config | Total Input Bits |
|---------|---------|----------|----------|---------------|-----------------|
| **2** | **9.0844** | **8,816** | **10.04%** | n=200, b=18 | 32 |
| **3** | **9.0862** | **8,833** | **7.84%** | n=200, b=20 | 48 |
| 4 | 9.1430 | 9,349 | 6.46% | n=200, b=20 | 64 |
| 5 | 9.2042 | 9,938 | 5.08% | n=200, b=20 | 80 |
| 6 | 9.2477 | 10,381 | 4.05% | n=200, b=20 | 96 |
| 7 | 9.2741 | 10,659 | 3.21% | n=200, b=20 | 112 |
| 8 | 9.2926 | 10,857 | 2.64% | n=200, b=20 | 128 |
| 16 | 9.3685 | 11,713 | 0.34% | n=200, b=20 | 256 |

### Winners

| Metric | Context | Value | Config |
|--------|---------|-------|--------|
| Best CE/PPL | 2 | CE=9.0844, PPL=8,816 | n=200, b=18 |
| Best Accuracy | 2 | 10.04% | n=200, b=18 |
| Runner-up CE | 3 | CE=9.0862, PPL=8,833 | n=200, b=20 |
| Runner-up Acc | 3 | 7.84% | n=200, b=20 |

**Context=2 wins on both CE and accuracy.** Context=3 is a close second on CE (only +0.002 nats) but loses ~22% accuracy.

### Context Degradation Analysis

| Context | PPL vs ctx=2 | Acc vs ctx=2 | CE Penalty |
|---------|-------------|-------------|------------|
| 2 | baseline | baseline | — |
| 3 | +0.2% | -21.9% | +0.002 |
| 4 | +6.0% | -35.7% | +0.059 |
| 5 | +12.7% | -49.4% | +0.120 |
| 6 | +17.7% | -59.7% | +0.163 |
| 7 | +20.9% | -68.0% | +0.190 |
| 8 | +23.2% | -73.7% | +0.208 |
| 16 | +32.9% | -96.6% | +0.284 |

### Key Insights

1. **Context=2 is the sweet spot**: Both best CE and best accuracy. The 32-bit address space (2 tokens x 16 clusters) is small enough to be well-covered by 2.4M training examples.

2. **Monotonic degradation**: Every additional context token makes things worse — both CE and accuracy degrade consistently. There are no "jumps" where more context suddenly helps.

3. **Address space explosion**: Context=2 has 2^32 possible addresses. Context=8 has 2^128. With only 2.4M training examples, larger context = sparser memory = more EMPTY cells = worse predictions.

4. **Accuracy collapses faster than CE**: Going from ctx=2 to ctx=8 loses 74% of accuracy but only 23% of PPL. This is because accuracy requires the correct token to be the argmax winner — harder with sparser predictions.

5. **All configs converge on n=200, b=20**: Except ctx=2 which favors b=18. More neurons and bits help because they compensate for context-induced sparsity.

6. **Contrast with transformers**: GPT-2 small uses context=1024 and achieves PPL ~29.4. RAM WNNs cannot use selective attention — all context bits contribute to the address, making longer context a liability rather than an asset.

### Comparison with Tiered Architecture

These results use uniform (non-tiered) bitwise RAMLM. The best tiered result (EMPTY=0.0, 5-tier, context=4) achieved PPL=26,986 — significantly better than the uniform ctx=2 result (PPL=8,816... wait, that's actually better!).

**Important note**: The bitwise RAMLM PPL numbers are NOT directly comparable to the tiered RAMLM numbers because:
- Bitwise RAMLM uses per-bit output encoding (16 binary predictions per token)
- Tiered RAMLM uses per-cluster output (50K-way softmax)
- CE is computed differently (binary CE vs categorical CE)

The bitwise CE ~9.08 corresponds to the per-bit cross-entropy across 16 output bits, not the categorical CE across 50K tokens.

---

## Per-Cluster Optimization Experiments (2026-01-12)

### Context

Per-cluster (PC) optimization uses GA+TS to optimize neuron connectivity for each output cluster independently. This is the correct granularity for language modeling where each cluster predicts a specific token.

**Key question:** Can per-cluster connectivity optimization improve PPL (not just accuracy)?

### Fitness Function Comparison

Tested different fitness functions with uniform architecture (3 neurons × 8 bits, 50K clusters):

| Fitness Function | Val PPL Δ | Val Acc Δ | Notes |
|------------------|-----------|-----------|-------|
| No optimization (baseline) | 0% | 2.18-2.83% | Initial random connectivity |
| **SIMPLE** | +7.88% ❌ | -43% ❌ | `Σvote_correct - Σvote_wrong` |
| **SIMPLE + Groups(50)** | +9.33% ❌ | -64% ❌ | Joint optimization of 50 clusters |
| **ACCURACY** | +3.25% ❌ | +67% ✓ | Count top-1 wins |
| **Global CE** | +1.38% ❌ | +32.5% ✓ | Softmax over all 50K clusters |

### Key Findings

#### ❌ Per-Cluster Optimization Degrades PPL

All fitness functions tested resulted in **worse PPL** after optimization:

1. **SIMPLE fitness** (`+vote when correct, -vote when wrong`):
   - Optimizes individual cluster discrimination
   - Caused PPL to increase by 7.88%
   - Reason: Doesn't account for false positives from other clusters

2. **ACCURACY fitness** (count how often cluster wins top-1):
   - Best accuracy improvement (+67%)
   - Still degraded PPL (+3.25%)
   - Reason: Optimizes for rank position, not probability calibration

3. **Global CE fitness** (true softmax over all 50K clusters):
   - Most theoretically correct (matches eval metric)
   - Still degraded PPL (+1.38%)
   - Only 5% of clusters improved (2613/50257)
   - Reason: Per-cluster optimization can't coordinate between clusters

#### Why Per-Cluster Fails

The fundamental issue is **coordination**. When optimizing cluster A:
- If A's connectivity changes to increase its votes
- Other clusters' relative votes change too
- But those other clusters were optimized with different connectivity
- After retraining, the global distribution shifts unpredictably

**Analogy**: Like tuning each instrument in an orchestra separately - they might each sound "better" alone but worse together.

#### ✓ Accuracy vs PPL Trade-off

Interestingly, accuracy improved substantially even as PPL degraded:
- ACCURACY fitness: +67% accuracy, +3.25% PPL
- Global CE: +32.5% accuracy, +1.38% PPL

This suggests:
- Connectivity optimization CAN find better discriminative features
- But the probability calibration (softmax distribution) gets worse
- The model becomes "more confident but less calibrated"

### Architecture Comparison

| Architecture | Global Baseline? | Fitness Mode | Results |
|--------------|------------------|--------------|---------|
| Uniform (--mode fast) | ✓ (fixed 2026-01-12) | Global CE | PPL +1.38% |
| Tiered (sweep experiments) | ✓ (always had) | Per-tier CE | See sweep results above |

**Note:** The full-data sweep experiments (tier0_20bit, etc.) used tiered architecture with GA+TS optimization. Those experiments showed different dynamics because:
1. Different tiers have different cluster frequencies
2. Optimization happens per-tier, not per-cluster
3. Frequent tokens (tier 0) benefit more from optimization

### Implications

1. **Per-cluster connectivity optimization alone is insufficient** for language modeling
2. **Need coordinated optimization** across clusters (e.g., joint loss, alternating optimization)
3. **Tiered architecture with asymmetric capacity** remains the best approach (see tier0_20bit results)
4. **Future direction**: Consider whole-model optimization or learned connectivity initialization

### Commands Used

```bash
# ACCURACY fitness
python tests/ramlm_full_benchmark.py --mode fast --optimize --strategy PC --fitness ACCURACY

# Global CE fitness
python tests/ramlm_full_benchmark.py --mode fast --optimize --strategy PC --fitness CE

# SIMPLE fitness
python tests/ramlm_full_benchmark.py --mode fast --optimize --strategy PC --fitness SIMPLE
```

---

## 5-Tier Simpler Configuration (2026-01-12)

### Motivation

The previous 5-tier experiment (`five_tier_gradient`) used very high neuron counts (37n, 31n) that may have been overkill. Testing a simpler gradient that better matches data density.

### Configuration

**five_tier_simple**: `50,15,20;50,13,18;400,9,10;20000,7,9;rest,5,8`

| Tier | Clusters | Neurons | Bits | Data % | Rationale |
|------|----------|---------|------|--------|-----------|
| 0 | 50 | 15 | 20 | 42.1% | Same as champion tier0_20bit |
| 1 | 50 | 13 | 18 | 4.6% | Slight step down |
| 2 | 400 | 9 | 10 | 13.0% | Medium capacity |
| 3 | 20,000 | 7 | 9 | 36.8% | Low capacity |
| 4 | 29,757 | 5 | 8 | 3.4% | Minimal (rest) |

### Results

#### Before Optimization (Promising!)

| Metric | Value | vs tier0_20bit |
|--------|-------|----------------|
| **Val PPL** | 36,643 | **-0.6%** ✓ |
| **Val Acc** | 4.86% | -8% |
| **Test PPL** | (not measured) | - |

**Per-Tier Initial Results:**
| Tier | PPL | Accuracy |
|------|-----|----------|
| 0 | 32,974 | 11.34% |
| 1 | 34,527 | 0.72% |
| 2 | 34,903 | 0.27% |
| 3 | 41,228 | 0.04% |
| 4 | 49,235 | 0.00% |

The initial PPL (36,643) was **better than the champion** tier0_20bit (36,853) before any optimization!

#### After GA+TS Optimization (Window-based)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Val PPL | 36,643 | 39,353 | **+7.40%** ❌ |
| Val Acc | 4.86% | 2.68% | **-45%** ❌ |
| Test PPL | - | 39,459 | - |
| Test Acc | - | 2.72% | - |

**GA+TS hurt BOTH metrics!**

### The Optimization-Retraining Mismatch Problem

**Why did optimization make things worse?**

| Step | Data Size | Modified Cells |
|------|-----------|----------------|
| Initial training | 2.4M examples | 25,399,131 |
| Optimization | 10K examples | - |
| Retrain after opt | 2.4M examples | 18,223,400 ❌ |

The "optimized" connectivity caused **28% fewer memory cells** to be written during retraining - meaning more address collisions and lost information.

**Root cause**: Connectivity optimized on a 10K-token window doesn't transfer to the full 2.4M-token dataset. The optimizer finds patterns that work locally but cause more collisions at scale.

### Overnight Full-Data Optimization Run (In Progress)

Testing the hypothesis that optimizing on **full data** might actually help:

```bash
python tests/ramlm_full_benchmark.py \
  --full-data \
  --tiered "50,15,20;50,13,18;400,9,10;20000,7,9;rest,5,8" \
  --optimize --strategy GA,TS \
  --opt-train-windows 12000 \   # ~2.4M tokens (full train)
  --opt-eval-windows 700 \      # ~140K tokens (half of test)
  --ga-gens 1000 --ts-iters 1000 --patience 5 \
  --per-tier
```

**Key differences from standard run:**
- Train windows: 12,000 (2.4M tokens) instead of 50 (10K tokens)
- Eval windows: 700 (140K tokens) instead of 20 (4K tokens)
- 1000 generations/iterations with patience 5

**Started:** 2026-01-12 21:22
**Expected duration:** ~12-24 hours

### Key Insight

**The 5-tier simpler config achieved the best pre-optimization PPL (36,643)**, but optimization on small windows hurt it. If full-data optimization works, this architecture could beat the current champion.

### Implications

1. **Architecture design matters more than connectivity optimization** (at least with small-window optimization)
2. **Initial random connectivity may be surprisingly good** when trained on full data
3. **Small-window optimization doesn't transfer** to full dataset
4. **Full-data optimization is the real test** - running overnight

---

## EMPTY Value Experiments (2026-01-13)

### Background

The EMPTY value determines what probability an unwritten memory cell contributes during softmax prediction. Previous experiments used EMPTY=0.5 (neutral), but this inflates the denominator for rare tokens with many empty cells.

### The Finding

| Configuration | Test PPL | Test Acc | Change |
|---------------|----------|----------|--------|
| EMPTY=0.5 (no opt) | 36,646 | 5.30% | Baseline |
| EMPTY=0.5 (full opt) | 39,487 | 3.97% | +7.8% worse |
| **EMPTY=0.0 (no opt)** | **26,986** | **4.86%** | **-26.4% better** ⭐ |
| EMPTY=0.0 (quick opt) | 33,443 | 0.48% | +24% worse than EMPTY=0.0 no opt |

### 🏆 New Champion: EMPTY=0.0 Without Optimization

**EMPTY=0.0 achieves PPL 26,986 — a 26% improvement over EMPTY=0.5!**

This is now the best result achieved with the 5-tier architecture:
```
Configuration: 50,15,20;50,13,18;400,9,10;20000,7,9;rest,5,8
Context: 4
EMPTY value: 0.0
Optimization: NONE
```

### Per-Tier Results (EMPTY=0.0, No Optimization)

| Tier | Clusters | Neurons | Bits | Data % | PPL | Accuracy |
|------|----------|---------|------|--------|-----|----------|
| 0 | 50 | 15 | 20 | 42.0% | 21,697 | **11.40%** |
| 1 | 50 | 13 | 18 | 4.6% | 23,181 | 0.59% |
| 2 | 400 | 9 | 10 | 13.0% | 24,737 | 0.19% |
| 3 | 20,000 | 7 | 9 | 36.9% | 34,353 | 0.05% |
| 4 | 29,757 | 5 | 8 | 3.6% | 48,275 | 0.13% |
| **TOTAL** | 50,257 | | | 100.0% | **26,986** | **4.86%** |

### Why EMPTY=0.0 Works

When a memory cell is EMPTY (never written), it means:
- The network has no training evidence for that address
- With EMPTY=0.5: contributes 0.5 to numerator, inflating softmax denominator
- With EMPTY=0.0: contributes 0.0, effectively "abstaining" from the vote

For rare tokens (tiers 3-4) with many empty cells:
- EMPTY=0.5: Many neurons vote 0.5, diluting the meaningful votes
- EMPTY=0.0: Only trained cells vote, improving signal-to-noise ratio

### Why Optimization Still Hurts

Even with EMPTY=0.0, optimization degraded PPL from 26,986 → 33,443 (+24%):

| Factor | Impact |
|--------|--------|
| **Data mismatch** | Optimizer sees 50 generations × limited windows |
| **Overfitting** | Connectivity optimized for specific patterns |
| **Cell reduction** | Modified 18M cells vs 25M with random connectivity |

### Key Conclusions

1. **EMPTY=0.0 is strictly better** than EMPTY=0.5 for this architecture
2. **No optimization > optimization** - random connectivity + full training wins
3. **Architecture + EMPTY tuning > connectivity optimization**

### Updated Overall Rankings

| Rank | Experiment | Config | Test PPL | Notes |
|------|------------|--------|----------|-------|
| 🥇 **1** | **five_tier_EMPTY0** | `50,15,20;...;rest,5,8` | **26,986** | ⭐ New champion! |
| 🥈 2 | tier0_20bit | `100,15,20;400,10,12;rest,5,8` | 36,853 | Previous best |
| 🥉 3 | five_tier_simple | `50,15,20;...;rest,5,8` | 36,643 | EMPTY=0.5 |
| 4 | asymmetric_extreme_t0 | `100,25,24;...` | 39,503 | |
| 5 | context_8_high_cap | `100,15,12;...` | 39,508 | |

### Commands Used

```bash
# EMPTY=0.0 without optimization (NEW CHAMPION)
python tests/ramlm_full_benchmark.py \
  --full-data \
  --tiered "50,15,20;50,13,18;400,9,10;20000,7,9;rest,5,8" \
  --empty 0.0 \
  --per-tier

# EMPTY=0.0 with optimization (degraded results)
python tests/ramlm_full_benchmark.py \
  --full-data \
  --tiered "50,15,20;50,13,18;400,9,10;20000,7,9;rest,5,8" \
  --empty 0.0 \
  --optimize --strategy GA,TS \
  --ga-gens 50 --ts-iters 50 --patience 1 \
  --per-tier
```

---

## Failed Experiment: Per-Cluster Architecture Search (2026-01-14)

### Background

Attempted to optimize (bits, neurons) per individual token cluster (50K clusters) using Tabu Search on AdaptiveClusteredRAM architecture.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| Runtime | 3.5+ hours | Killed at iteration 11/100 |
| CPU Time | 3,391 minutes | ~56 hours CPU time |
| Best CE | 10.5461 | Stuck since iteration 4 |
| Test Accuracy | ~0.04% | Essentially random guessing |
| Improvement | 0% | Flat fitness landscape |

### Why It Failed

1. **Search space too large**: 50K clusters × 2 parameters (bits, neurons) = 100K dimensional space
2. **Flat fitness landscape**: CE ~10.5 ≈ log(50257) means barely better than random
3. **No gradient signal**: 0.04% accuracy = ~20 correct predictions out of 50K, statistical noise
4. **Time per evaluation**: ~17s per genome × 20 neighbors × 100 iterations = would need ~9 more hours

### Key Insight

Per-token granularity is too fine when the model can't predict anything. The tiered approach works because:
- Groups tokens by frequency (coarser granularity = stronger signal)
- Tier 0 (100 tokens) has 11% accuracy → real optimization signal
- Per-cluster has 0.04% accuracy → no signal to optimize

### Conclusion

**Don't optimize at per-token level.** Use tiered approach with 3-5 tiers, not 50K clusters.

---

## Architecture Verification (2026-01-14)

### Confirmed: All Current Benchmarks Use Real RAM WNN

| Script | Architecture | Status |
|--------|--------------|--------|
| `ramlm_benchmark.py` | RAMLM → RAMClusterLayer → Memory | ✅ Real RAM |
| `ramlm_full_benchmark.py` | RAMLM → TieredRAMClusterLayer → Memory | ✅ Real RAM |
| `run_adaptive_search.py` | AdaptiveRAMLMWrapper → AdaptiveClusteredRAM → Memory | ✅ Real RAM |

### Deleted: 17 Counter-Based Test Scripts

The following scripts used `defaultdict(Counter)` (n-gram counting, NOT real RAM WNN):

```
tests/ram_lm_v2.py           tests/language_model.py
tests/ram_lm_combined.py     tests/language_model_word.py
tests/ram_lm_deep_analysis.py tests/babi_tasks.py
tests/ram_lm_final.py        tests/code_completion.py
tests/ram_lm_hybrid_connectivity.py tests/listops_benchmark.py
tests/ram_lm_improved.py     tests/real_world_benchmark.py
tests/ram_lm_optimized.py    tests/scaling_benchmark.py
tests/ram_lm_proper.py       tests/scan_benchmark.py
                             tests/theorem_proving.py
```

### Real RAM WNN vs Counter-Based

| Aspect | Counter-based (deleted) | Real RAM WNN (kept) |
|--------|------------------------|---------------------|
| Storage | `dict[addr → Counter]` | `Memory` with ternary cells |
| Cell states | Counts (0, 1, 2...) | TRUE / FALSE / EMPTY |
| Output | Probability from counts | Boolean or "don't know" |
| Training | Count increment | EDRA backpropagation |

---

## 🔴 Critical Bug: Random Connection Regeneration (2026-01-15)

### Discovery

While running adaptive architecture search (GA→TS pipeline), Tabu Search showed **zero improvement** after 1200+ genome evaluations despite Metal acceleration achieving ~5s per genome.

### Investigation

Observed symptoms:
- GA improved from 10.96 → 10.54 CE (initial generations)
- TS showed 0% improvement after 40+ iterations
- Neighbors evaluated had similar CE to current best (within noise)
- No clear optimization signal despite evaluating many neighbors

### Root Cause

In `src/wnn/ram/strategies/accelerator/adaptive.rs:673`:

```rust
// THE BUG - new random connections for EVERY genome evaluation!
let mut rng = rand::rngs::SmallRng::from_entropy();
```

**The Problem:**
When TS mutates a genome (changing one cluster's bits by +1), the "neighbor" gets:
1. Architecture: `[8, 8, 8, ..., 9, ..., 8]` (one cluster changed from 8→9 bits)
2. Connections: **Completely random!** (regenerated from entropy)

The fitness of a genome depends on BOTH architecture AND connectivity. With random connections, what TS thinks are "neighbors" are actually completely different models. There is no gradient to follow.

**Analogy:** Imagine optimizing a neural network's architecture while randomly reinitializing all weights each evaluation. The loss would be pure noise.

### Why GA Still Worked (Partially)

GA showed initial improvement because:
1. Population diversity explores architecture space broadly
2. Selection pressure still favors lower CE even with noise
3. But improvement stalls once easy wins are found

TS failed because it relies on local search - neighbors must be actually similar.

### Solution: Connection-Preserving Genomes

Connections must be part of genome state:

```python
@dataclass
class ClusterGenome:
    bits_per_cluster: List[int]
    neurons_per_cluster: List[int]
    connections: List[List[int]]  # NEW: inherited and mutated
```

Operations:
1. **Initialize:** Create random connections once when genome is born
2. **Crossover:** Inherit connections from parents (with crossover point)
3. **Mutate:** Small perturbations (+/-1, +/-2) to connection indices
4. **Evaluate:** Use genome's stored connections, not random ones

### Proposed Phased Approach

Instead of optimizing everything at once:

| Phase | What's Optimized | What's Fixed | Rationale |
|-------|------------------|--------------|-----------|
| 1a | Neurons per cluster | Bits=8, random conn | Coarse architecture |
| 1b | Bits per cluster | Neurons from 1a, random conn | Fine architecture |
| 2 | Connections | Architecture from 1b | Connectivity patterns |

Each phase inherits the best solution from the previous phase.

### Performance Improvements Made

While investigating, several optimizations were implemented:

| Improvement | Before | After | Speedup |
|-------------|--------|-------|---------|
| Metal GPU for dense groups | 15s/genome | 5s/genome | ~3x |
| Sequential genome eval | OOM | Stable | ∞ |
| Duplicate logging removed | 2x logs | 1x logs | cleaner |
| Early stop indicators | None | 🟢🟡🔴 | visible |

### Next Steps

1. Implement `connections` field in `ClusterGenome`
2. Modify `adaptive.rs` to accept connections as input
3. Implement connection mutation with small deltas
4. Test phased optimization approach

### Commands Used

```bash
# Ran overnight - observed no TS improvement
python tests/run_adaptive_search.py

# Monitoring
tail -f logs/2026/01/15/adaptive_search_*.log
```

---

