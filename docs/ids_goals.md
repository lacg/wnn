# IDS Performance Goals

Last updated: 2026-05-31 (UNSW-NB15 temporal baselines re-measured, 30-seed cohort)

## Literature Performance & Our Goals

### CICIDS2017

| Method | Split | Acc | F1 | FPR | Model Size |
|---|---|---:|---:|---:|---|
| XGBoost | random | 99.95% | ~99% | <1% | ~10MB |
| Random Forest | random | 99.90% | 97.4% | ~1% | ~50MB |
| CNN-BiLSTM | random | 98.7% | ~98% | 1.4% | ~5MB |
| LSTM | random | 99.0% | ~98% | ~2% | ~5MB |
| RF/XGB | **temporal** | **~85-90%*** | **~85-88%*** | **~10-15%*** | ~50MB |

*Temporal estimates — almost no papers use day-based splits. This is our opportunity.

**Important**: Most CICIDS2017 papers use random splits AND have [known labeling errors](https://hal.science/hal-03775466v1/document). The temporal split is largely unexplored.

| Our Goals — CICIDS2017 | Target | Stretch |
|---|---:|---:|
| F1 (temporal, oracle) | 85% | 90% |
| F1 (random) | 95% | 98% |
| FPR (temporal, oracle) | <15% | <10% |
| Model size | <1KB | <500B |

### CIC-IoT-2023

**Important**: Dataset is ~97-98% attack traffic. Accuracy >98% is trivially achievable by always predicting "attack". F1 is the meaningful metric. Almost no papers report FPR. No papers use temporal splits. Data leakage is a documented concern (dedicated study, ResearchGate 2024).

#### Original Paper (Neto et al. 2023, random 80/20, StandardScaler)

| Method | Acc | F1 | FPR | Source |
|---|---:|---:|---:|---|
| Random Forest | 99.68% | 96.53% | — | Neto et al. 2023 (original) |
| AdaBoost | 99.59% | 95.63% | — | Neto et al. 2023 (original) |
| DNN | 99.44% | 94.03% | — | Neto et al. 2023 (original) |
| Logistic Regression | 98.90% | 87.63% | — | Neto et al. 2023 (original) |
| Perceptron | 98.18% | 81.05% | — | Neto et al. 2023 (original) |

#### Follow-up Papers (all random splits)

| Method | Acc | F1 | FPR | Source |
|---|---:|---:|---:|---|
| DL (auto features) | 99.71% | 98.47% | — | Ferreira et al. 2023 |
| CNN-LSTM | 98.42% | 98.57% | 9.17% | Gueriani et al. 2024 |
| DCNN | 99.50% | 89.72%* | — | Bayaraa et al. 2024 |
| XGBoost+RF ensemble | 99.87% | 99.85%** | — | Anis 2024 (thesis) |
| CNN | 99.40% | — | — | Ayo et al. 2024 |

*attack-class F1 only (benign-class F1 = 99.75%)
**likely inflated by data leakage (near-duplicate flows in random split)

#### Our Results (pending)

| Method | Acc | F1 | FPR | Notes |
|---|---:|---:|---:|---|
| Our WNN (8-bit) | — | — | — | 112 flows queued, starts after CICIDS |

#### Our Goals

| Our Goals — CIC-IoT-2023 | Target | Stretch |
|---|---:|---:|
| F1 (random) | 93% | 96% |
| FPR (random) | <5% | <2% |

### UNSW-NB15 (achieved 2026-05-31, n=30 seed cohort)

| Our Results — UNSW-NB15 | Server (n=30) | Edge / FPGA (5n pick) | FPR-extreme (16n pick) |
|---|---:|---:|---:|
| F1 (temporal, val_cal best_ce) | **88.87% ±0.23%** | **87.88%** | 87.70% |
| FPR (temporal, val_cal best_ce) | **8.77% ±1.14%** | **6.94%** | **4.63%** |
| Max F1 single seed | 89.34% (flow 2994) | — | — |
| Model size | ~1.3 MB | **~2 KB** | ~5 KB |
| F1 (random, 1 flow) | 93.86% | — | — |

**F1 ceiling note**: No XDS-unsw-temporal config crossed 90% F1 across all widths and weight schemes evaluated (16b-Wb peak 89.34, 8b-Wc peak 89.50, 96b-Wc peak 89.69). RF and XGBoost on the same split plateau at 84-86% F1. The temporal split is genuinely hard; the differentiator is FPR and model size, not F1.

---

## UNSW-NB15 Detailed Baselines (Standard Temporal Split)

Updated 2026-05-31: RF and XGBoost rows now reflect direct measurement on the temporal
split via `scripts/verify_unsw_temporal_baselines.py` (top-20 features, 16-bit thermometer,
matched preprocessing to our WNN). Prior "~87% / ~12%" entries were estimates and
substantially underestimated FPR.

| Model | Accuracy | F1-Macro | FPR | Model Size | Source |
|---|---|---|---|---|---|
| Random Forest (measured) | 86.39% | **86.05%** | **25.41%** | 138 MB | This work, 100 trees, max_depth=None |
| XGBoost (measured) | 85.34% | **84.89%** | **28.57%** | 0.27 MB | This work, 100 trees, max_depth=6, lr=0.1 |
| SVM | ~86% | ~85% | ~15% | — | Kasongo & Sun 2020 |
| DNN (sequential) | ~88% | ~87% | ~10% | ~2 MB | Zoghi & Serpen 2024 |
| BiLSTM (deep learning) | ~89% | ~89% | ~8% | ~5 MB | Abdalgawad et al. 2024 |
| **WNN server (n=30 seeds)** | — | **88.87±0.23%** | **8.77±1.14%** | **~1.3 MB** | This work, 16b-Wb 500n34b OI |
| **WNN edge / FPGA (5n)** | — | **87.88%** | **6.94%** | **~2 KB** | This work, flow 2994 r56926 |

## Research FPR vs Production FPR

**Our FPR targets above (8-12%) are research metrics on a balanced test set.**

Production deployment is a fundamentally different regime:

- In production, 95-99% of traffic is normal
- A 10% FPR on 1M flows/day = ~100,000 false alerts/day
- Industry guidance: <1% FPR is "good", <0.1% is "excellent"
- **Alert fatigue** is the #1 operational killer of IDS tools

This is why production IDS uses **cascading architectures**:

```
Tier 1: WNN on FPGA (5ns, inspects EVERY packet)
  -> flags ~5% suspicious
Tier 2: Deep model on CPU/GPU (re-examines flagged subset only)
  -> confirms ~0.1% as true alerts
SOC analyst
```

## Sources

- [Zoghi & Serpen 2024](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.8242)
- [Abdalgawad et al. 2024](https://www.mdpi.com/1999-4893/17/2/64)
- [Kasongo & Sun 2020](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00379-6)
- [CICIDS2017 Labeling Errors](https://hal.science/hal-03775466v1/document)
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [Neto et al. 2023 — CICIoT2023 original paper](https://www.mdpi.com/1424-8220/23/13/5941)
- [Ferreira et al. 2023 — Auto vs hand-crafted features](https://arxiv.org/html/2312.00034v1)
- [Gueriani et al. 2024 — CNN-LSTM IDS](https://arxiv.org/abs/2405.18624)
- [Bayaraa et al. 2024 — DCNN binary+multiclass](https://link.springer.com/article/10.1186/s13635-024-00184-1)
- [Ayo et al. 2024 — DNN/CNN/LSTM comparison](https://www.mdpi.com/2227-9709/11/2/32)
