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

### Our WNN Targets

| Metric                    | Target | Stretch Goal | Basis                                    |
|---------------------------|--------|--------------|------------------------------------------|
| **Accuracy**              | ≥ 88%  | ≥ 90%        | Beat RF (87.2%) and XGBoost (87.3%)      |
| **F1-Macro**              | ≥ 87%  | ≥ 89%        | Match or beat classical ML (~87%)        |
| **FPR**                   | ≤ 12%  | ≤ 8%         | Match classical ML; stretch = BiLSTM     |
| Multi-class (10 classes)  | ≥ 85%  | ≥ 88% (F1)   | Novel benchmark (no WNN prior art)       |

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

## Multi-Class Architecture: Hierarchical S0 → S1

For multi-class IDS (10 classes), we use the existing multi-stage evaluator
with two independently-optimized stages:

```
                    ┌─────────────────────────────┐
                    │  S0: Binary (Normal vs Attack) │
All 329-bit flows → │  Tiered or Bitwise genome     │
                    │  Optimized for recall + low FPR│
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │ Normal              │ Attack
                    │ (done)              ↓
                    │              ┌──────────────────────────┐
                    │              │  S1: 9 Attack Types       │
                    │              │  Tiered or Bitwise genome │
                    │              │  Optimized for per-class F1│
                    │              └──────────────────────────┘
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

### Per-Stage Metrics

**S0 (Binary: Normal vs Attack)**

| Metric     | Target | Stretch | Notes                                          |
|------------|--------|---------|------------------------------------------------|
| F1-Macro   | ≥ 87%  | ≥ 89%   | Same as binary targets above                   |
| FPR        | ≤ 12%  | ≤ 8%    | % Normal misclassified as Attack               |
| Recall     | ≥ 95%  | ≥ 98%   | Must be high — missed attacks never reach S1   |

S0 should be biased toward **high recall** (catch all attacks), tolerating slightly
higher FPR. Better to send a normal flow to S1 than to miss an attack entirely.

**S1 (Multi-class: 9 Attack Types)**

| Metric              | Target | Stretch | Notes                                  |
|---------------------|--------|---------|----------------------------------------|
| F1-Macro (9 class)  | ≥ 80%  | ≥ 85%   | Hard due to Worms (44 examples)        |
| Per-class F1 (rare) | ≥ 50%  | ≥ 70%   | Worms, Shellcode, Backdoors            |
| Per-class F1 (common)| ≥ 85% | ≥ 90%   | Generic, Exploits, Fuzzers             |

S1 only trains/evaluates on attack flows. Cost-sensitive fitness can boost rare classes.

**Combined (Overall 10-class)**

| Metric          | Target | Stretch | How computed                            |
|-----------------|--------|---------|-----------------------------------------|
| Overall F1-Macro| ≥ 85%  | ≥ 88%   | F1 across all 10 classes                |
| Overall FPR     | ≤ 12%  | ≤ 8%    | = S0 FPR (S1 doesn't affect normals)   |
| Overall Accuracy| ≥ 88%  | ≥ 90%   | Correct class for all flows             |

### Dashboard Tracking

Each stage runs as a **separate experiment** within a flow:

```
Flow: "IDS Multi-Class 10-class"
├── Experiment 1: S0 Binary (GA Neurons → GA Bits → TS → Connections)
│   └── Metrics: F1-Macro, FPR, Recall (2-class)
├── Experiment 2: S1 Attack Types (GA Neurons → GA Bits → TS → Connections)
│   └── Metrics: F1-Macro, per-class F1 (9-class)
└── Combined Validation: overall F1-Macro, FPR, Accuracy (10-class)
```

This gives each stage its own iteration curves, seeded population, and validation
summaries in the dashboard — full observability per stage.

### UNSW-NB15 Attack Class Distribution (Training Set)

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

## Sources

- [Zoghi & Serpen 2024 — Building an IDS on UNSW-NB15](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.8242)
- [Abdalgawad et al. 2024 — Enhanced IDS Performance](https://www.mdpi.com/1999-4893/17/2/64)
- [Kasongo & Sun 2020 — Feature Selection on UNSW-NB15](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00379-6)
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [Check Point — FPR in Cybersecurity](https://www.checkpoint.com/cyber-hub/cyber-security/what-is-a-false-positive-rate-in-cybersecurity/)
- [Fidelis Security — Reducing IDS False Positives](https://fidelissecurity.com/cybersecurity-101/network-security/reducing-false-positives-in-intrusion-detection-systems/)
