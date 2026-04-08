# 46M Cross-Validation Verdict

**Flow:** 1165 -- EVAL46M-46M-200n4b-s42-CV-pipeline
**Architecture:** 200 neurons x 4 bits (800 bytes)
**Dataset:** CIC-IoT-2023 full 46M (random 80/20)

## Reference headline (from project_200n4b_46M_result.md, 2026-04-05)

| Metric | Headline |
|---|---|
| F1 | 82.33% |
| FPR | 6.67% |
| Acc | 97.27% |

## Cross-validation results (best_* genomes)

| Genome | F1 | FPR | Acc | delta F1 vs headline |
|---|---:|---:|---:|---:|
| best_acc | 82.10% | 2.67% | 97.10% | -0.23pp |
| best_ce | 78.73% | 1.70% | 96.18% | -3.60pp |
| best_f1 | 82.10% | 2.67% | 97.10% | -0.23pp |
| best_fitness | 82.10% | 2.67% | 97.10% | -0.23pp |
| best_fpr | 81.67% | 3.01% | 97.01% | -0.66pp |

## Threshold mode breakdown (best_fitness genome)

| Threshold | F1 | FPR | Acc |
|---|---:|---:|---:|
| train_cal | 82.10% | 2.67% | 97.10% |
| fixed_05 | 53.03% | 95.91% | 97.13% |
| platt | 49.28% | 100.00% | 97.15% |
| beta | 76.73% | 49.52% | 97.61% |
| empirical | 49.28% | 100.00% | 97.15% |
| empirical_cumulative | 81.72% | 26.60% | 97.71% |
| val_cal | 80.46% | 0.85% | 96.63% |

## Verdict

**MATCH**

best_fitness F1 (82.10%) is within 1pp of the headline (82.33%). Pipeline cross-validates successfully.
