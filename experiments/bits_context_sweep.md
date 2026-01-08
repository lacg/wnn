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

### 3 Neurons/Cluster (tier 1 + rest) - pending

| Mode | Bits | Ctx | Tiers | Total Neurons | Time | Val PPL | Val Acc | Test PPL | Test Acc | Notes |
|------|------|-----|-------|---------------|------|---------|---------|----------|----------|-------|
| fast | 8    | 4   | 100×11, 400×3, rest×3 | - | - | -  | -   | -   | -    | Pending |
| fast | 10   | 4   | 100×11, 400×3, rest×3 | - | - | -  | -   | -   | -    | Pending |
| fast | 12   | 4   | 100×11, 400×3, rest×3 | - | - | -  | -   | -   | -    | Pending |

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
