# Research Insights: RAM WNNs for Language Modeling

This document captures key research findings from the RAM WNN language modeling experiments.

## Table of Contents

1. [Fundamental Limitation: Why RAM WNNs Cannot Match Transformers](#fundamental-limitation)
2. [Asymmetric Tiered Architecture](#asymmetric-tiered-architecture)
3. [Context Length Analysis](#context-length-analysis)
4. [Hybrid Architecture Vision](#hybrid-architecture-vision)

---

## Fundamental Limitation: Why RAM WNNs Cannot Match Transformers for LM {#fundamental-limitation}

**Date:** 2026-01-21
**Status:** Confirmed finding

### The Performance Gap

| Metric | RAM WNN (Best) | GPT-2 Small | Gap |
|--------|---------------|-------------|-----|
| WikiText-2 PPL | ~36,853 | ~29 | **1,270x worse** |
| Context window | 4 tokens | 1,024 tokens | 256x less |
| Architecture | Lookup tables | Matrix multiplication | Different paradigm |

**Conclusion: Pure RAM WNNs cannot achieve transformer-level language modeling performance.**

### The Fundamental Barriers

These are **mathematical limitations**, not engineering problems to solve:

#### 1. Address Space Explosion (Insurmountable)

```
Context × Bits/Token = Address Bits → Address Space

4 tokens × 16 bits = 64 bits  → 2^64 addresses   (barely fillable with all human text)
8 tokens × 16 bits = 128 bits → 2^128 addresses  (more than atoms in universe)
```

Language modeling requires 100+ tokens of context. This is mathematically impossible with address-based lookup.

#### 2. No Selective Attention (Architectural)

| Transformer | RAM WNN |
|-------------|---------|
| Computes relevance per token | ALL bits contribute to address |
| Ignores irrelevant context | Cannot ignore anything |
| Dynamic focus per query | Fixed addressing |

Transformers can attend to "the **dog**" 10 tokens back while ignoring noise. RAM WNNs must incorporate every bit into the address—irrelevant context pollutes the lookup.

#### 3. Why State Layers Don't Solve This

**Initial hope:** Add a recurrent state layer to "compress" longer context into a fixed-size state, enabling n-gram lookups with encoded history.

**Why it fails:**

1. **State must encode 1024 tokens of context** → requires massive state space
2. **Learning the encoding** still requires seeing the patterns → same data sparsity problem
3. **Sequential processing**: 1024 tokens ÷ 4-gram = 256 sequential lookups per prediction
4. **Performance**: 256 sequential lookups ≠ "just a lookup" anymore

| Approach | Operations per Token | Parallelizable? |
|----------|---------------------|-----------------|
| Transformer | 1 matrix multiply | Yes (batch) |
| RAM + State | ~256 sequential lookups | No (sequential) |

The "instant lookup" advantage disappears when you need hundreds of them sequentially.

### Research Value

This exploration was not wasted. We learned:

1. **Why the limitation exists** - Mathematical (address space), not engineering
2. **Connectivity = static attention** - Theoretical insight connecting paradigms
3. **Tiered architecture value** - Asymmetric allocation based on data density
4. **Optimization techniques** - GA/TS for discrete architecture search (reusable)
5. **Hybrid potential** - Clear vision for where RAM WNNs fit in larger systems

---

## Asymmetric Tiered Architecture {#asymmetric-tiered-architecture}

**Date:** 2026-01-11
**Status:** Validated

### The Finding

| Configuration | Tier 0 | Tier 1 | Tier 2 | Test PPL |
|---------------|--------|--------|--------|----------|
| **Asymmetric (best)** | 20 bits | 12 bits | 8 bits | **36,853** |
| Uniform 20-bit | 20 bits | 20 bits | 20 bits | 49,675 |

The asymmetric config achieves **35% better PPL** than uniform.

### Why This Works

The key is **training data density per address space**:

| Tier | Tokens | Data % | Examples/Token | Can Fill |
|------|--------|--------|----------------|----------|
| Tier 0 | 100 frequent | 46% | ~11,000 | 2^20 addresses ✓ |
| Tier 1 | 400 medium | 13% | ~800 | 2^12 addresses ✓ |
| Tier 2 | 50K rare | 40% | ~20 | 2^8 addresses ✗ |

### Design Principle

**Match address space size to training data density:**
- High-frequency tiers → more bits (can utilize the capacity)
- Low-frequency tiers → fewer bits (can't fill large spaces anyway)

---

## Context Length Analysis {#context-length-analysis}

**Date:** 2026-01-21
**Status:** Confirmed

### Why Transformers Scale with Context

Transformers use **selective attention**:
1. Compute relevance scores between all token pairs
2. Attend strongly to relevant tokens, weakly to irrelevant ones
3. Dynamically focus on different parts of context for different queries

Longer context = more opportunities to find relevant information, without penalty for irrelevant tokens.

### Why RAM WNNs Don't Scale with Context

RAM WNNs use **address-based lookup**:
1. Concatenate ALL context bits into an address
2. Look up that exact address in memory
3. Cannot ignore any bits - all contribute to the address

Longer context = exponentially larger address space = more EMPTY cells = worse predictions.

### Experimental Evidence

From overnight sweeps:
```
context=4:  Best PPL 36,853 ✓
context=8:  Higher PPL (worse)
context=16: Even higher PPL (even worse)
```

### Connectivity as Static Attention

The GA/TS connectivity optimization is a form of **static attention**:
- Each neuron's connectivity defines which bits it "attends to"
- Optimization finds the most informative bit subsets
- Unlike transformers, this is fixed per neuron (not dynamic per input)

---

## Hybrid Architecture Vision {#hybrid-architecture-vision}

**Date:** 2026-01-21
**Status:** Proposed future direction

### What RAM WNNs ARE Good For

| Strength | Why | Example Use |
|----------|-----|-------------|
| **Binary/discrete patterns** | Native representation | Hash tables, pattern matching |
| **Fixed short context** | No address explosion | Classification, lookup |
| **Interpretability** | Explicit memory cells | Debugging, verification |
| **Fast pattern retrieval** | O(1) lookup | Caching frequent patterns |
| **No floating point** | Bit operations only | Edge devices, FPGAs |

### The Hybrid Architecture

**Key insight:** Use RAM WNNs for what they're good at, transformers for the rest.

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Tokens                                               │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────────┐                                   │
│   │  RAM Pattern Cache  │ ◄── Fast lookup for frequent      │
│   │  (short n-grams)    │     patterns (cache hit = done)   │
│   └──────────┬──────────┘                                   │
│              │ cache miss                                    │
│              ▼                                               │
│   ┌─────────────────────┐                                   │
│   │    Transformer      │ ◄── Long-range dependencies,      │
│   │  (attention layers) │     complex reasoning             │
│   └──────────┬──────────┘                                   │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────┐                                   │
│   │  RAM Output Layer   │ ◄── Fast final classification,   │
│   │  (frequent tokens)  │     interpretable decisions       │
│   └──────────┬──────────┘                                   │
│              │                                               │
│              ▼                                               │
│        Output Token                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Potential Benefits

1. **Cache frequent patterns**: Most language is repetitive; RAM caches common n-grams
2. **Fast path for easy cases**: "the" → next word often predictable from 2-3 grams
3. **Slow path for hard cases**: Transformer handles long-range, complex dependencies
4. **Interpretable caching**: Can inspect what patterns are cached and why

### Future Research Directions

| Direction | Feasibility | Value |
|-----------|-------------|-------|
| Pure RAM LM | ❌ Not viable | Proven impossible |
| RAM as cache layer | ✅ Promising | Speed up frequent patterns |
| RAM for classification | ✅ Good fit | Short context, interpretable |
| RAM for tokenization | ✅ Natural fit | Pattern matching |
| RAM for edge/embedded | ✅ Strong | No FP, interpretable |
| Hybrid transformer+RAM | 🔬 Research | Best of both worlds |

---

---

## Bitwise Architecture Optimization {#bitwise-optimization}

**Date:** 2026-02-10 — ongoing
**Status:** Active experiments

See **[docs/BITWISE_OPTIMIZATION.md](BITWISE_OPTIMIZATION.md)** for full details.

### Key Findings

1. **Grid search rankings are stable**: n=200/b=20 consistently wins across runs (CE~9.14)
2. **Progressive threshold is critical**: Must start at 3% with gentle ramp (0.001%/gen)
3. **HARMONIC_RANK prevents accuracy collapse**: Previous CE-only fitness destroyed accuracy
4. **BitwiseRAMLM approach**: 16 independent clusters, each with configurable neurons/bits/connections
5. **Context sweep planned**: Test [2,3,4,5,6,7,8,16] n-grams with best architecture

### Connection to Hybrid Vision

The bitwise optimization work feeds directly into the **hybrid architecture vision**:
- Optimized RAM WNN serves as the fast pattern cache layer
- Context sweep determines optimal n-gram size for the cache
- Gating (Engram-inspired) enables selective use of RAM predictions
- Hard cases fall through to transformer backbone

---

## Changelog

- **2026-02-11**: Added bitwise optimization section, linked to detailed doc
- **2026-01-21**: Added fundamental limitation analysis, hybrid architecture vision
- **2026-01-11**: Added asymmetric tiered architecture finding
