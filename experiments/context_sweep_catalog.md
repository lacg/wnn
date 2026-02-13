# Bitwise N-gram Context Sweep Catalog

## Overview

Systematic sweep of context sizes 2-16 for the bitwise RAMLM architecture.
All experiments use QUAD_WEIGHTED memory, HARMONIC_RANK fitness, vocab=50257, clusters=16.

## Phase 1 Grid Search Results (No GA/TS Optimization)

### Best CE by Context Size

| Context | Input Bits | Best CE | Neurons | Bits | Acc at Best CE | Data Split |
|---------|-----------|---------|---------|------|---------------|------------|
| **2** | 32 | **9.0844** | 200 | 18 | 10.04% | tp=36, ep=6 |
| **3** | 48 | **9.0862** | 200 | 20 | 7.84% | tp=36, ep=6 |
| 4 | 64 | 9.1430 | 200 | 20 | 6.46% | tp=1, ep=1 (full) |
| 5 | 80 | 9.2042 | 200 | 20 | 5.08% | tp=36, ep=6 |
| 6 | 96 | 9.2477 | 200 | 20 | 4.46% | tp=36, ep=6 |
| 7 | 112 | 9.2741 | 200 | 20 | 3.73% | tp=36, ep=6 |
| 8 | 128 | 9.2926 | 200 | 20 | 3.01% | tp=36, ep=6 |
| 16 | 256 | 9.3685 | 200 | 14 | 0.87% | tp=36, ep=6 |

### Best Accuracy by Context Size

| Context | Best Acc | Neurons | Bits | CE at Best Acc |
|---------|---------|---------|------|---------------|
| **2** | **10.40%** | 200 | 20 | 9.1394 |
| **3** | 7.96% | 200 | 18 | 9.0898 |
| 4 | 6.69% | 200 | 18 | 9.1529 |
| 5 | 5.39% | 200 | 18 | 9.2204 |
| 6 | 4.46% | 200 | 20 | 9.2477 |
| 7 | 3.73% | 200 | 20 | 9.2741 |
| 8 | 3.01% | 200 | 20 | 9.2926 |
| 16 | 1.82% | 50 | 20 | 9.3319 |

### Key Observations

1. **Context 2 and 3 are nearly tied on CE** (9.084 vs 9.086), but ctx2 has much better accuracy (10.4% vs 8.0%)
2. **Performance degrades monotonically** with more context tokens beyond 2-3
3. **Sweet spot is 18-20 bits** for best CE; 20 bits for best accuracy at most context sizes
4. **200 neurons consistently best** across all context sizes
5. **Context 16 shifts to fewer bits** (14) because the address space is already enormous (2^14 * 256 input bits)

### Data Split Caveat

- Context 2,3,5,6,7,8,16 used **sampled data**: `train_parts=36, eval_parts=6` (~67k train, ~48k eval per subset)
- Context 4 used **full data**: `train_parts=1, eval_parts=1` (~2.4M train, ~286k eval)
- Full data ctx4 with GA optimization reached CE=9.126, beating sampled ctx2/3 grid search (9.084)
- **Fair comparison requires same data split** - ctx2/3 with full data would likely be better than 9.084

## Experiment Files

| Context | File | Status |
|---------|------|--------|
| 2 | `bitwise_ngram_ctx2.json` | Phase 1 only |
| 3 | `bitwise_ngram_ctx3.json` | Phase 1 only |
| 4 | `bitwise_fulldata_v4.json` | Phase 1 (full data) |
| 4 | `bitwise_fulldata_v5.json` | Phases 1-7 in progress (PID 76075) |
| 5 | `bitwise_ngram_ctx5.json` | Phase 1 only |
| 6 | `bitwise_ngram_ctx6.json` | Phase 1 only |
| 7 | `bitwise_ngram_ctx7.json` | Phase 1 only |
| 8 | `bitwise_ngram_ctx8.json` | Phase 1 only |
| 16 | `bitwise_ngram_ctx16.json` | Phase 1 only |

## Sampling Rate Comparison (ctx=4, 200n, 20b, full data)

From session 2026-02-12:

| Rate | CE | Acc | Time |
|------|-----|------|------|
| 1.00 | 9.233 | 7.41% | 152s |
| 0.25 | 9.127 | 7.18% | 133s |
| 0.10 | 9.129 | 6.74% | 102s |
| 0.01 | 9.391 | 3.91% | 64s |

Rate 0.25 is the sweet spot: best CE with 1.15x speedup vs rate=1.0.

## High Neuron Count Test (ctx=2, rate=0.1, full data)

| Neurons | Bits | CE | Acc | Time |
|---------|------|-----|------|------|
| 10,000 | 4 | 9.632 | 0.87% | 481s |
| 10,000 | 8 | 9.401 | 6.34% | 692s |

4-bit address space (16 addresses) is too small even with 10k neurons.
8-bit (256 addresses) performs reasonably but worse than 200n/18b (CE 9.084).

## Next Steps

- [ ] Run ctx=2 with full data (train_parts=1) to get fair comparison with ctx=4
- [ ] Run ctx=2 with train_parts=4,eval_parts=1 as efficient alternative
- [ ] Compare train_parts={1,4,10,20} on same config to find data efficiency sweet spot
- [ ] Run full GA/TS optimization on best context size with optimal data split
