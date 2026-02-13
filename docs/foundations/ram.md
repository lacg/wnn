# RAM Neurons

## What is a RAM Neuron?

A **RAM (Random Access Memory) neuron** is a computational unit that uses a lookup table instead of weighted connections. Given a set of binary inputs, the neuron computes an address, looks up that address in its memory, and returns the stored value.

```
Input bits:  [1, 0, 1]
                │
                ▼
         Address = 101₂ = 5
                │
                ▼
    ┌───────────────────┐
    │  RAM Memory (2³)  │
    │  ───────────────  │
    │  Addr 0: FALSE    │
    │  Addr 1: TRUE     │
    │  Addr 2: FALSE    │
    │  Addr 3: TRUE     │
    │  Addr 4: FALSE    │
    │  Addr 5: TRUE  ◄──┤── Output: TRUE
    │  Addr 6: EMPTY    │
    │  Addr 7: FALSE    │
    └───────────────────┘
```

A neuron with **n** input connections has **2^n** memory addresses. Each address stores a value learned during training.

## How It Differs from Weighted Neurons

| Aspect | Weighted Neuron | RAM Neuron |
|--------|----------------|------------|
| **Computation** | Weighted sum + activation | Address lookup |
| **Parameters** | Continuous weights | Binary/ternary memory cells |
| **Learning** | Gradient descent | Memory writes + connectivity optimization |
| **Inputs** | Real-valued | Binary |
| **Output** | Real-valued | Binary or probabilistic (via neuron ensemble) |
| **Inference speed** | O(n) multiply-accumulate | O(1) memory lookup |

## Partial Connectivity: The Generalization Mechanism

The key insight of RAM-based neural networks is that **partial connectivity enables generalization**.

### Fully Connected = Memorization

If a neuron observes ALL input bits, each unique input maps to a unique address. The neuron memorizes each training example exactly but cannot generalize to unseen inputs.

### Partially Connected = Generalization

If a neuron observes only a SUBSET of input bits (e.g., 3 out of 48), then many different inputs that share the same values at those 3 positions will map to the **same address** and trigger the **same response**.

```
Example: Neuron observes bits [2, 5, 11] out of 48 total

Input A: ...1..0.....1...  → Address = 101₂ = 5 → TRUE
Input B: ...1..0.....1...  → Address = 101₂ = 5 → TRUE  (same!)
         ↑     ↑         ↑
        bit2  bit5     bit11

Despite being completely different inputs, A and B trigger
the same response because they match at positions [2, 5, 11].
The neuron has learned a FEATURE — the pattern at those positions.
```

This is the fundamental generalization mechanism. The neuron learns a **feature** (the pattern at its connected positions), and any input exhibiting that feature triggers the same response — even inputs never seen during training.

### Which Bits to Connect?

The choice of which input bits each neuron observes is critical. Random connectivity gives baseline performance, but **optimized connectivity** dramatically improves it. This is where metaheuristic optimization (GA, TS, SA) comes in — see [Connectivity Optimization](optimization.md).

## Memory States

In this project, RAM neurons use multi-state memory cells:

### Ternary Mode (3 states)
| Value | Meaning | Output |
|-------|---------|--------|
| FALSE (0) | Trained negative | 0.0 |
| TRUE (1) | Trained positive | 1.0 |
| EMPTY (2) | Untrained | 0.5 (configurable) |

### Quad Mode (4 states)
| Value | Meaning | Weight |
|-------|---------|--------|
| STRONG_FALSE (0) | Confident negative | 0.0 |
| WEAK_FALSE (1) | Tentative negative | 0.25 |
| WEAK_TRUE (2) | Tentative positive | 0.75 |
| STRONG_TRUE (3) | Confident positive | 1.0 |

Quad mode enables **incremental learning** via nudging: each training example nudges the cell one step toward the target direction, rather than overwriting it. This provides more nuanced confidence estimation and better generalization.

## Output Clustering

A single neuron outputs a binary value. To produce probabilistic outputs (e.g., for language modeling), we use **output clustering**: multiple neurons per output class, with their responses averaged.

```
Output bit 5 prediction:
  Neuron 0: TRUE  (1.0)
  Neuron 1: FALSE (0.0)
  Neuron 2: TRUE  (1.0)
  Neuron 3: TRUE  (1.0)
  ─────────────────────
  Average:  0.75  → P(bit 5 = 1) = 75%
```

More neurons per cluster = smoother probability estimates, but also more memory and computation.

## In This Project

The project applies RAM neurons to **language modeling**:

- **Input**: Binary-encoded context tokens (e.g., 4 tokens x 16 bits = 64 input bits)
- **Output**: 16 output clusters (one per bit of the predicted token), each with multiple neurons
- **Training**: Memory writes from token sequences
- **Connectivity**: Optimized via GA/TS/SA to find the most informative bit subsets

## References

- Aleksander, I., & Morton, H. (1990). *An Introduction to Neural Computing.* Chapman & Hall.
- Aleksander, I., & Stonham, T. J. (1979). "Guide to pattern recognition using random-access memories." *Computers and Digital Techniques*, 2(1).
- Ludermir, T. B., Carvalho, A., Braga, A. P., & Souto, M. (1999). "Weightless neural models: a review of current and past works." *Neural Computing Surveys*, 2. UC Berkeley ICSI.
