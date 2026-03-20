# IDS Performance Goals

Last updated: 2026-03-20

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

| Method | Split | Acc | F1 | FPR | Model Size |
|---|---|---:|---:|---:|---|
| XGBoost + RF | random | 99%+ | 99%+ | <1% | ~10MB |
| CatBoost | random | ~96% | ~96% | ~4% | ~10MB |
| ANN | random | ~95% | ~95% | ~5% | ~1MB |

| Our Goals — CIC-IoT-2023 | Target | Stretch |
|---|---:|---:|
| F1 (random) | 93% | 96% |
| FPR (random) | <5% | <2% |
| Model size | <1KB | <500B |

### UNSW-NB15 (existing, for reference)

| Our Goals — UNSW-NB15 | Current (34 flows) | Target (100 flows) | Stretch |
|---|---:|---:|---:|
| F1 (temporal, oracle) | 87.10% ±2.55% | 88% | 90% |
| F1 (random) | 93.86% (1 flow) | 95% | 98% |
| FPR (temporal, oracle) | 10.89% ±4.08% | <10% | <7% |

---

## UNSW-NB15 Detailed Baselines (Standard Temporal Split)

| Model | Accuracy | F1-Macro | FPR | Source |
|---|---|---|---|---|
| Random Forest | 87.2% | ~87% | ~12% | Our reproduction |
| XGBoost | 87.3% | ~87% | ~12% | Our reproduction |
| SVM | ~86% | ~85% | ~15% | Kasongo & Sun 2020 |
| DNN (sequential) | ~88% | ~87% | ~10% | Zoghi & Serpen 2024 |
| BiLSTM (deep learning) | ~89% | ~89% | ~8% | Abdalgawad et al. 2024 |

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
