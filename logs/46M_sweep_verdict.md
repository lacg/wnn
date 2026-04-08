# 46M Pareto Sweep Verdict

**Sweep tag:** SWEEP46M
**Flows:** 22 total

## Flow status

| Status | Count |
|---|---:|
| completed | 22 |

## Reference: headline cross-validation (flow 1166)

The 200n × 4b architecture on 46M (flow 1166, single-genome CV) gave:
- F1 = 82.20%, FPR = 6.06%, Acc = 97.22% (train_cal threshold)

The sweep below should show how this baseline scales across memory tiers.

## Results by tier (train_cal threshold — deployable)

### MICRO

| Config | Mem | Seeds | F1 | FPR | Acc |
|---|---|---:|---:|---:|---:|
| 5n × 4b | 20B | 10 | 66.66±6.36% | 8.84±12.30% | 90.68±4.49% |
| 100n × 4b | 400B | 1 | 80.32% | 2.85% | 96.66% |
| 200n × 4b | 800B | 1 | 79.37% | 2.38% | 96.38% |
| 300n × 4b | 1.2KB | 1 | 79.73% | 2.90% | 96.50% |
| 500n × 4b | 2.0KB | 1 | 79.60% | 3.58% | 96.49% |

### SMALL

| Config | Mem | Seeds | F1 | FPR | Acc |
|---|---|---:|---:|---:|---:|
| 5n × 12b | 5.0KB | 1 | 80.32% | 6.02% | 96.76% |
| 100n × 8b | 6.2KB | 1 | 79.89% | 1.68% | 96.51% |
| 300n × 8b | 18.8KB | 1 | 81.58% | 2.11% | 96.96% |
| 400n × 8b | 25.0KB | 1 | 83.13% | 2.06% | 97.33% |

### PEAK

| Config | Mem | Seeds | F1 | FPR | Acc |
|---|---|---:|---:|---:|---:|
| 96n × 32b | 96.0GB | 1 | 84.36% | 1.61% | 97.59% |
| 198n × 32b | 198.0GB | 1 | 84.68% | 2.00% | 97.67% |
| 245n × 32b | 245.0GB | 1 | 84.79% | 1.59% | 97.68% |
| 500n × 34b | 2000.0GB | 1 | 84.73% | 1.91% | 97.68% |

## PEAK tier: three operating modes (train_cal / fixed_05 / val_cal)

| Config | Mode | F1 | FPR | Acc |
|---|---|---:|---:|---:|
| 96n × 32b | train_cal | 84.36% | 1.61% | 97.59% |
| 96n × 32b | fixed_05 | 86.34% | 8.66% | 98.14% |
| 96n × 32b | val_cal | 84.54% | 1.87% | 97.64% |
| | | | | |
| 198n × 32b | train_cal | 84.68% | 2.00% | 97.67% |
| 198n × 32b | fixed_05 | 86.11% | 6.11% | 98.04% |
| 198n × 32b | val_cal | 84.64% | 1.51% | 97.65% |
| | | | | |
| 245n × 32b | train_cal | 84.79% | 1.59% | 97.68% |
| 245n × 32b | fixed_05 | 86.30% | 11.00% | 98.17% |
| 245n × 32b | val_cal | 84.20% | 1.36% | 97.55% |
| | | | | |
| 500n × 34b | train_cal | 84.73% | 1.91% | 97.68% |
| 500n × 34b | fixed_05 | 86.64% | 9.83% | 98.21% |
| 500n × 34b | val_cal | 84.48% | 1.69% | 97.62% |
| | | | | |

## 34b saturation probe verdict

- **Best 32b peak (train_cal):** 245n × 32b — F1 84.79%, FPR 1.59%, Acc 97.68%
- **Best 34b peak (train_cal):** 500n × 34b — F1 84.73%, FPR 1.91%, Acc 97.68%
- **Delta (34 - 32):** -0.06pp F1

**🤝 TIE:** 34b and 32b are within noise. Extra bits don't hurt, but don't help either. Thermometer encoding caps effective discrimination.

## Literature comparison (Neto et al. 2023, same dataset, random 80/20)

| Method | F1 | Acc | Notes |
|---|---:|---:|---|
| Perceptron | 81.05% | 98.18% | ~10 KB |
| Logistic Reg. | 87.63% | 98.90% | ~1 KB |
| DNN | 94.03% | 99.44% | ~5 MB |
| AdaBoost | 95.63% | 99.59% | ~10 MB |
| Random Forest | 96.53% | 99.68% | ~50 MB |

**Our best 32b peak:** 96n × 32b @ empirical_cumulative
  F1 = 86.96%, FPR = 14.38%, Acc = 98.35%

