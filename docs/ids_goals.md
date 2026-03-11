# IDS Goals — UNSW-NB15

## Standard Split (175K train / 82K test — what we use)

This is the harder, more realistic split with temporal separation between train/test.

### Literature Baselines

| Model                  | Accuracy | F1-Macro | FPR    | Source                        |
|------------------------|----------|----------|--------|-------------------------------|
| Random Forest          | 87.2%    | ~87%     | ~12%   | Our reproduction (Week 18)    |
| XGBoost                | 87.3%    | ~87%     | ~12%   | Our reproduction (Week 18)    |
| SVM                    | ~86%     | ~85%     | ~15%   | Kasongo & Sun 2020            |
| DNN (sequential)       | ~88%     | ~87%     | ~10%   | Zoghi & Serpen 2024           |
| BiLSTM (deep learning) | ~89%     | ~89%     | ~8%    | Abdalgawad et al. 2024        |

### Our WNN Targets (Binary)

| Metric                    | Target | Stretch Goal | Basis                                    |
|---------------------------|--------|--------------|------------------------------------------|
| **Accuracy**              | ≥ 88%  | ≥ 90%        | Beat RF (87.2%) and XGBoost (87.3%)      |
| **F1-Macro**              | ≥ 87%  | ≥ 89%        | Match or beat classical ML (~87%)        |
| **FPR**                   | ≤ 12%  | ≤ 8%         | Match classical ML; stretch = BiLSTM     |

Matching RF/XGBoost (~87% F1, ~12% FPR) validates the architecture.
**Beating them is publishable** — no prior WNN work reports results on the standard split.

## Random Split (90/10 deduplicated — FWIW comparison)

Much easier — inflated numbers due to data leakage between train/test:

| Model          | Accuracy | F1     | FPR  |
|----------------|----------|--------|------|
| RF / XGBoost   | 98-99%   | 97-99% | <2%  |
| Deep Learning  | 98-99%   | 98-99% | <1%  |
| FWIW WNN       | 98.5%    | —      | —    |

Target: ≥ 98.5% (match FWIW).

## Research FPR vs Production FPR

**Our FPR targets above (8–12%) are research metrics on a balanced test set.**

Production deployment is a fundamentally different regime:

- In production, 95–99% of traffic is normal
- A 10% FPR on 1M flows/day = **~100,000 false alerts/day**
- Industry guidance: <1% FPR is "good", <0.1% is "excellent"
- **Alert fatigue** is the #1 operational killer of IDS tools — teams turn off noisy detectors

This is why production IDS uses **cascading architectures**:

```
Tier 1: WNN on FPGA (5ns, inspects EVERY packet)
  ↓ flags ~5% suspicious
Tier 2: Deep model on CPU/GPU (re-examines flagged subset only)
  ↓ confirms ~0.1% as true alerts
SOC analyst
```

The WNN's value is not standalone FPR but **throughput × recall at the first tier** —
inspecting every packet at line rate rather than sampling 1 in 100.

Production FPR targets (for the cascading system):

| Tier          | FPR Target | Role                                  |
|---------------|------------|---------------------------------------|
| Tier 1 (WNN)  | ≤ 5%       | High recall, flags suspicious traffic |
| Tier 2 (Deep) | ≤ 0.1%     | High precision, confirms real threats |
| Combined      | ≤ 0.1%     | What the SOC analyst actually sees    |

---

## Experiment Types

We have **4 IDS experiment types**, progressing from simple to complex:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Type 1: Flat Binary (Tiered)         ← CURRENT, running today    │
│  Type 2: Flat Binary (Bitwise)        ← CURRENT, running today    │
│  Type 3: Hierarchical S0→S1 (Tiered)  ← NEW                       │
│  Type 4: Hierarchical S0→S1 (Bitwise) ← NEW                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Type 1 & 2: Flat Binary (current)

Single-stage, 2 classes (Normal vs Attack). One genome, one optimization pipeline.
- **Tiered**: clusters grouped by frequency tiers, shared bits/neurons per tier
- **Bitwise**: per-cluster bits/neurons, fully heterogeneous

Metrics: F1-Macro (2-class), FPR, Accuracy.

### Type 3 & 4: Hierarchical S0 → S1 (new)

Two independently-optimized stages, each with its own genome:

```
                    ┌──────────────────────────────────┐
                    │  S0: Binary (Normal vs Attack)   │
All 329-bit flows → │  Own genome (tiered or bitwise)  │
                    │  Own 7-phase optimization         │
                    └──────────┬───────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │ Normal              │ Attack
                    │ (done)              ↓
                    │              ┌─────────────────────────────┐
                    │              │  S1: 9 Attack Types          │
                    │              │  Own genome (tiered or bitwise│)
                    │              │  Own 7-phase optimization     │
                    │              └─────────────────────────────┘
                    │                     │
                    ↓                     ↓
               "Normal"          "Exploits" / "Generic" / "Worms" / ...
```

### Why hierarchical beats flat 10-class

1. **Class imbalance**: Flat 10-class is dominated by Normal (56K) and Generic (17K).
   Rare attacks (Worms=44, Shellcode=378) get ignored.
2. **FPR is controlled by S0 alone**: S1 never sees normal flows, so it cannot
   create false positives. The most critical metric depends on one focused stage.
3. **Balanced training data per stage**: S0 sees Normal vs Attack (~56K vs 119K).
   S1 sees only attacks — still imbalanced but without Normal drowning everything.
4. **Independent optimization**: Each stage's genome is optimized via its own
   GA/TS pipeline with stage-specific fitness.

---

## Design Decisions

### D1: S0 Operating Point — Configurable, Not Hardcoded

Rather than choosing between balanced F1 and recall-biased, **make it a config parameter
and let experiments decide**:

```
s0_fitness_mode: "balanced" | "recall_biased"
```

- **balanced**: Standard `ids_security` fitness (F1 × (1-FPR)²)
- **recall_biased**: Penalizes missed attacks more heavily than false positives.
  e.g. `recall_weight × Recall + (1 - recall_weight) × (1 - FPR)` with `recall_weight=0.7`

Run both modes and compare:
- Balanced S0 → lower FPR, but some attacks slip past S1
- Recall-biased S0 → higher FPR, but S1 sees (almost) all attacks

The combined 10-class F1 tells us which approach wins overall.

### D2: S1 Training Data — Configurable, Test Both

Two modes for what S1 trains on:

```
s1_training_data: "all_attacks" | "s0_filtered"
```

- **all_attacks**: S1 trains on ALL attack examples from the training set,
  regardless of what S0 would classify them as. Simpler, more training data,
  S1 learns from examples S0 might miss.

- **s0_filtered**: S1 trains ONLY on examples that S0 classified as attacks.
  More realistic (matches inference-time data distribution), but S1 never
  learns from attacks S0 misses. Also tests: what about Normal flows that
  S0 wrongly sends to S1? S1 must handle those gracefully.

Run both and compare. Hypothesis: `all_attacks` wins for S1 F1 (more data),
but `s0_filtered` may produce better *combined* metrics since S1 is calibrated
to S0's actual output distribution.

### D3: Dashboard Tracking — Separate Experiments Per Stage

Each stage runs as a **separate experiment** within a flow:

```
Flow: "IDS Hierarchical Tiered"
├── Experiment 1: S0 Binary (GA Neurons → GA Bits → TS → Connections)
│   ├── Iteration curves: F1-Macro, FPR, Recall (2-class)
│   ├── Seeded Population with per-genome F1/FPR
│   └── Validation: init + final summaries
├── Experiment 2: S1 Attack Types (GA Neurons → GA Bits → TS → Connections)
│   ├── Iteration curves: F1-Macro (9-class), per-class F1
│   ├── Seeded Population with per-genome F1/FPR
│   └── Validation: init + final summaries
└── Combined Validation: overall F1-Macro, FPR, Accuracy (10-class)
```

Full observability per stage — each gets its own iteration charts, seeded
population table, and validation progression in the dashboard.

---

## Per-Stage Metrics & Targets

### S0 (Binary: Normal vs Attack)

| Metric     | Target | Stretch | Notes                                          |
|------------|--------|---------|------------------------------------------------|
| F1-Macro   | ≥ 87%  | ≥ 89%   | Same as binary targets above                   |
| FPR        | ≤ 12%  | ≤ 8%    | % Normal misclassified as Attack               |
| Recall     | ≥ 95%  | ≥ 98%   | Must be high — missed attacks never reach S1   |

### S1 (Multi-class: 9 Attack Types)

| Metric              | Target | Stretch | Notes                                  |
|---------------------|--------|---------|----------------------------------------|
| F1-Macro (9 class)  | ≥ 80%  | ≥ 85%   | Hard due to Worms (44 examples)        |
| Per-class F1 (rare) | ≥ 50%  | ≥ 70%   | Worms, Shellcode, Backdoors            |
| Per-class F1 (common)| ≥ 85% | ≥ 90%   | Generic, Exploits, Fuzzers             |

S1 only trains/evaluates on attack flows. Cost-sensitive fitness can boost rare classes.

### Combined (Overall 10-class)

| Metric          | Target | Stretch | How computed                            |
|-----------------|--------|---------|-----------------------------------------|
| Overall F1-Macro| ≥ 85%  | ≥ 88%   | F1 across all 10 classes                |
| Overall FPR     | ≤ 12%  | ≤ 8%    | = S0 FPR (S1 doesn't affect normals)   |
| Overall Accuracy| ≥ 88%  | ≥ 90%   | Correct class for all flows             |

---

## UNSW-NB15 Attack Class Distribution (Training Set)

| Class          | Count  | % of Attacks | S1 Strategy                    |
|----------------|--------|--------------|--------------------------------|
| Generic        | 18,871 | 15.8%        | Standard neurons/bits          |
| Exploits       | 11,132 | 9.3%         | Standard neurons/bits          |
| Fuzzers        | 6,062  | 5.1%         | Standard neurons/bits          |
| DoS            | 4,089  | 3.4%         | Moderate neurons, fewer bits   |
| Reconnaissance | 3,496  | 2.9%         | Moderate neurons, fewer bits   |
| Analysis       | 677    | 0.6%         | Few neurons, few bits          |
| Backdoors      | 583    | 0.5%         | Few neurons, few bits          |
| Shellcode      | 378    | 0.3%         | Few neurons, few bits          |
| Worms          | 44     | 0.04%        | Minimal (6 bits, 2-3 neurons)  |

For S1, asymmetric initialization per class based on data density is critical.
Worms with 44 examples cannot fill a 2^12 address space — use 6-bit neurons.

---

## Experiment Matrix

Full set of experiments to run:

| # | Type | Architecture | S0 Mode | S1 Data | Key Question |
|---|------|-------------|---------|---------|--------------|
| 1 | Flat binary | Tiered | — | — | Baseline: can WNN beat RF? |
| 2 | Flat binary | Bitwise | — | — | Does per-cluster heterogeneity help? |
| 3 | Hierarchical | Tiered | Balanced | All attacks | Multi-class with standard fitness |
| 4 | Hierarchical | Tiered | Recall-biased | All attacks | Does biasing S0 help combined F1? |
| 5 | Hierarchical | Tiered | Balanced | S0-filtered | Does realistic S1 training help? |
| 6 | Hierarchical | Tiered | Recall-biased | S0-filtered | Best of both biases? |
| 7 | Hierarchical | Bitwise | Balanced | All attacks | Bitwise version of #3 |
| 8 | Hierarchical | Bitwise | Recall-biased | All attacks | Bitwise version of #4 |

Priority order: 1 → 2 → 3 → 7 → 4 → 8 → 5 → 6

## Sources

- [Zoghi & Serpen 2024 — Building an IDS on UNSW-NB15](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.8242)
- [Abdalgawad et al. 2024 — Enhanced IDS Performance](https://www.mdpi.com/1999-4893/17/2/64)
- [Kasongo & Sun 2020 — Feature Selection on UNSW-NB15](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00379-6)
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [Check Point — FPR in Cybersecurity](https://www.checkpoint.com/cyber-hub/cyber-security/what-is-a-false-positive-rate-in-cybersecurity/)
- [Fidelis Security — Reducing IDS False Positives](https://fidelissecurity.com/cybersecurity-101/network-security/reducing-false-positives-in-intrusion-detection-systems/)
