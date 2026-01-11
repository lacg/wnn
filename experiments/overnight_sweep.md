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

### Standard Sweep + Hybrid + Extended Results

Full WikiText-2 dataset (2.4M train, 251K val, 288K test tokens) with GA+TS optimization.

#### Overall Rankings (by Validation PPL)

| Rank | Experiment | Config | Ctx | Val PPL | Test PPL | Accuracy | Improv |
|------|------------|--------|-----|---------|----------|----------|--------|
| 🥇 **1** | **hybrid** | `100,20,12;400,8,10;rest,5,8` | 4 | **46,346** | **46,267** | **0.04%** | 0.63% |
| 🥈 2 | neurons_20_tier0 | `100,20,12;400,12,10;rest,7,8` | 4 | 46,405 | 46,358 | 0.02% | 0.63% |
| 🥉 3 | context_6_sparse | `100,12,14;400,8,10;rest,5,8` | 6 | 46,619 | 46,584 | 0.02% | 0.36% |
| 4 | neurons_25_gradient | `100,25,14;400,15,10;rest,7,8` | 4 | 46,725 | 46,677 | 0.02% | 0.55% |
| 5 | tier0_16bit (ext) | `100,15,16;400,10,10;rest,5,8` | 4 | 46,804 | 46,757 | 0.01% | 0.48% |
| 6 | tier0_16bit (std) | `100,15,16;400,10,10;rest,5,8` | 4 | 46,826 | 46,787 | 0.01% | 0.46% |
| 7 | tier0_18bit | `100,15,18;400,10,12;rest,5,8` | 4 | 47,117 | 47,075 | 0.02% | 0.41% |
| 8 | balanced_14bit (std) | `100,12,14;400,8,12;rest,5,10` | 4 | 47,248 | 47,181 | 0.13% | 0.94% |
| 9 | balanced_14bit (ext) | `100,12,14;400,8,12;rest,5,10` | 4 | 47,256 | 47,195 | 0.07% | 0.87% |

#### Best Per-Tier Results

| Tier | Best PPL | From Experiment | Config |
|------|----------|-----------------|--------|
| **Tier 0** | 48,977 | hybrid | 100 × 20n × 12b |
| **Tier 1** | 44,622 | context_6_sparse | 400 × 8n × 10b |
| **Tier 2** | 43,829 | hybrid | rest × 5n × 8b |

#### Hybrid Experiment Details

The hybrid config combines best-performing settings from each tier:
- **Tier 0**: 20 neurons × 12 bits (from neurons_20_tier0)
- **Tier 1**: 8 neurons × 10 bits (from context_6_sparse)
- **Tier 2**: 5 neurons × 8 bits (from context_6_sparse)

**Results:**
| Metric | Initial | Final | Change |
|--------|---------|-------|--------|
| Val PPL | 46,642 | 46,346 | -0.63% |
| Val Acc | 0.00% | 0.04% | +708% |
| Test PPL | - | 46,267 | - |
| Test Acc | - | 0.04% | - |

**Per-Tier Test Breakdown:**
| Tier | Clusters | Neurons | Bits | Data % | PPL | Accuracy |
|------|----------|---------|------|--------|-----|----------|
| 0 | 100 | 20 | 12 | 46.5% | 48,977 | 0.07% |
| 1 | 400 | 8 | 10 | 13.0% | 44,665 | 0.01% |
| 2 | 49,757 | 5 | 8 | 40.4% | 43,829 | 0.02% |

### Key Findings (Updated)

1. **20 neurons optimal for Tier 0**: Beats 12, 15, and even 25 neurons
2. **8 neurons optimal for Tier 1**: Outperforms 10, 12, 15 neurons
3. **Lower bits (8-12) beat higher bits (14-18)**: Contrary to initial hypothesis
4. **Context 6 helps marginally**: But neuron tuning has bigger impact
5. **Hybrid approach works**: Combining best per-tier configs yields best overall result

### Hypotheses Status

| Hypothesis | Status | Finding |
|------------|--------|---------|
| More bits = better discrimination | ❌ **Rejected** | Lower bits (8-12) performed better |
| Diminishing returns on neurons | ✅ **Confirmed** | 20n optimal, 25n worse |
| Context trade-off | ⚠️ **Partial** | ctx=6 helped slightly, more testing needed |
| Tier0 focus | ✅ **Confirmed** | Tier 0 config most impactful on overall PPL |
