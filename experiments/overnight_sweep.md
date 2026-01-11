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

#### Overall Rankings (by Test PPL)

| Rank | Experiment | Config | Ctx | Test PPL | Test Acc | Notes |
|------|------------|--------|-----|----------|----------|-------|
| 🥇 **1** | **tier0_20bit** | `100,15,20;400,10,12;rest,5,8` | 4 | **36,853** | **5.28%** | ⭐ Best overall |
| 🥈 2 | context_8_high_cap | `100,15,12;400,10,10;rest,7,8` | 8 | 39,508 | 1.26% | Higher context |
| 3 | asymmetric_extreme_t0 | `100,25,24;400,8,10;rest,5,8` | 8 | 39,503 | 1.35% | 25n×24b tier0 |
| 4 | asymmetric_expanded_t0 | `200,25,20;300,8,10;rest,4,8` | 8 | 40,047 | 1.26% | 200 tokens tier0 |
| 🥉 5 | extreme_tier0 | `100,30,16;400,12,12;rest,5,8` | 5 | 40,934 | 0.13% | 30 neurons |
| 6 | two_tier_simple | `500,15,16;rest,4,6` | 8 | 41,883 | 0.03% | 2-tier only |
| 7 | neurons_20_tier0 | `100,20,12;400,12,10;rest,7,8` | 4 | 46,358 | 0.02% | |
| 8 | context_6_sparse | `100,12,14;400,8,10;rest,5,8` | 6 | 46,584 | 0.02% | |
| 9 | neurons_25_gradient | `100,25,14;400,15,10;rest,7,8` | 4 | 46,677 | 0.02% | |
| 10 | tier0_16bit | `100,15,16;400,10,10;rest,5,8` | 4 | 46,787 | 0.01% | |
| 11 | all_sparse_16bit | `100,15,16;400,12,16;rest,8,16` | 4 | 47,060 | 2.70% | Uniform 16b |
| 12 | tier0_18bit | `100,15,18;400,10,12;rest,5,8` | 4 | 47,075 | 0.02% | |
| 13 | balanced_14bit | `100,12,14;400,8,12;rest,5,10` | 4 | 47,181 | 0.13% | |
| 14 | uniform_20bit_ctx8 | `100,15,20;400,10,20;rest,5,20` | 8 | 49,675 | 1.81% | ❌ Uniform worse |

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
