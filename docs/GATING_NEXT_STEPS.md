# Gating Implementation - Next Steps

## Current Status (2026-01-30)

✅ **Completed:**
- RAM-based gating implementation (Python + Rust + Metal GPU)
- Hybrid CPU+GPU acceleration with auto mode selection
- Gating integration into phased search pipeline
- Dashboard UI for enabling gating in flows
- Parser fix for Live UI accuracy display
- Trend chart fix (cumulative min CE / max Acc)

---

## Immediate Next Steps

### 1. Run a Gated Test (Validate Integration)

**Purpose:** Verify gating works end-to-end before longer experiments.

**Quick test configuration:**
- GA Generations: 100
- TS Iterations: 100
- Patience: 5
- Tier Config: `100,15,20;400,10,12;rest,5,8`
- ✅ Enable Gating Layer
- Neurons per Gate: 8
- Bits per Neuron: 12
- Threshold: 0.5

**Expected behavior:**
1. 6 main phases run (neurons → bits → connections optimization)
2. Gating Training Phase runs after phase 6 completes
3. Logs show: `[Gate Stats] Active: X% | Tier0: X% | Tier1: X% | Tier2: X%`
4. Final results include gating stats

**How to run:**
- Dashboard: http://localhost:3000/flows/new (with gating enabled)
- CLI: `python run_coarse_fine_search.py --enable-gating --ga-gens 100 --ts-iters 100 --patience 5`

**Duration:** ~30-60 minutes

---

## After Gated Test Validates

### 2. Ablation Benchmark

Compare gated vs ungated performance on same configuration:

| Run | Config | Purpose |
|-----|--------|---------|
| A | Ungated baseline | Current best (already have results) |
| B | Gated (8 neurons, 12 bits) | Test gating improvement |
| C | Gated + context_size=3 | Shorter n-gram (Engram uses 2-3) |
| D | Gated + context_size=2 | Even shorter n-gram |

**Metrics to compare:**
- Cross-Entropy (CE) - lower is better
- Accuracy (%) - higher is better
- Per-tier accuracy breakdown
- Gate activation rates

### 3. N-gram Size Experiments

Engram paper found 2-3 n-grams optimal. Current default is 4.

| N-gram | Address Bits | Training Density | Notes |
|--------|--------------|------------------|-------|
| 2 | 32 bits | Very high | More collisions = sharing, Engram uses this |
| 3 | 48 bits | High | Engram also uses this |
| 4 | 64 bits | Medium | Current default |
| 5 | 80 bits | Low | May be too sparse |

**Why shorter n-grams may work better:**
- More examples per address = better training coverage
- Collisions help generalization (similar contexts share memory)
- Gating can filter bad matches (collisions less problematic)

---

## Medium-term Improvements

### 4. Semantic Bit Representations

Use learned encodings so similar tokens have similar bit patterns:
- `ram_learned` encoding from representations module
- `mutual_info` encoding based on co-occurrence
- Enables better generalization through address space sharing

### 5. Per-tier Gating

Different gating strategies per tier:
- Tier 0 (frequent): Fine-grained gating, more neurons
- Tier 2 (rare): Coarse gating, fewer neurons
- Matches data density to gating capacity

### 6. Soft Gating

Continuous gates [0,1] instead of binary:
- Smoother optimization signal
- May help during training
- Can threshold to binary at inference

---

## Long-term: DeepSeek Engram Comparison

### Setup Engram Baseline

```bash
# Clone Engram repo
git clone https://github.com/deepseek-ai/Engram
cd Engram

# Run their benchmark on same WikiText-2 data
python eval_engram.py --dataset wikitext2 --model_size small
```

### Comparison Methodology

1. Run RAM WNN ungated → get CE₁, Acc₁
2. Run RAM WNN gated → get CE₂, Acc₂
3. Run DeepSeek Engram → get CE₃, Acc₃
4. Compute improvement: `(CE₁ - CE₂) / CE₁` vs `(baseline - Engram) / baseline`

**Target:** Match or beat Engram's relative improvement (5-10% from paper).

---

## Paper Documentation

When results are ready, document in `docs/paper/`:
- Architecture diagram
- Ablation results table
- Per-tier accuracy breakdown
- Gate activation heatmaps
- Comparison with Engram

See plan file for full paper outline: `.claude/plans/cosmic-nibbling-leaf.md`
