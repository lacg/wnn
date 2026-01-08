# RAMLM Bits & Context Sweep Experiments

## Purpose
Test how bits_per_neuron and context_size affect model performance.

## Baseline Configuration
- Mode: fast (50k train, 10k test/val)
- Architecture: Tiered (100×11, 400×7, rest×5)
- Strategy: GA,TS
- GA: 5 pop × 5 gens
- TS: 5 neighbors × 3 iters

---

## Results Summary

| Bits | Context | Total Neurons | Val PPL | Val Acc | Test PPL | Test Acc | Notes |
|------|---------|---------------|---------|---------|----------|----------|-------|
| 8    | 4       | 252,685       | -       | -       | -        | -        | Baseline |
| 10   | 4       | 252,685       | -       | -       | -        | -        | +2 bits |
| 12   | 4       | 252,685       | -       | -       | -        | -        | +4 bits |
| 8    | 5       | 252,685       | -       | -       | -        | -        | +1 context |
| 8    | 6       | 252,685       | -       | -       | -        | -        | +2 context |

---

## Detailed Results

### Test 1: 8 bits × 4 context (baseline)
- Status: Pending
- Config: `--tiered "100,11,8;400,7,8;rest,5,8" --context 4`

### Test 2: 10 bits × 4 context
- Status: Pending
- Config: `--tiered "100,11,10;400,7,10;rest,5,10" --context 4`

### Test 3: 12 bits × 4 context
- Status: Pending
- Config: `--tiered "100,11,12;400,7,12;rest,5,12" --context 4`

### Test 4: 8 bits × 5 context
- Status: Pending
- Config: `--tiered "100,11,8;400,7,8;rest,5,8" --context 5`

### Test 5: 8 bits × 6 context
- Status: Pending
- Config: `--tiered "100,11,8;400,7,8;rest,5,8" --context 6`
