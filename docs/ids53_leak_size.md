# IDS-53 leak size — RF/XGB, legacy vs train_only top-20 (08/11/2026)

**Answer: the feature-selection leak did NOT materially inflate results.**
- Max inflation is ≤0.08 pp F1 / ≤0.07 pp FPR, and only on cicids 8-bit thermometers.
- cicids raw and both ciciot encodings score equal or BETTER with train_only.
- The no-leak control (UNSW-random, list change only) moves RF by +0.32 pp, so the gaps are list composition, not leak.

The legacy baselines are essentially unbiased. Cite the train_only rows going forward, since that is what the new WNN cells use.

## Method

- Code path: scripts/verify_ids_baselines.py, wrapped by scripts/ids53_leak_baselines.py.
- RF: 100 trees. XGB: 100 trees, depth 6, lr 0.1.
- val_cal threshold chosen on VAL, scored on TEST.
- Seeds 42–46; split fixed (HF random_3way), so the SDs reflect model seeds only.
- All 8 legacy seed-42 rows reproduce the banked 28/10 table exactly.
- 46 min wall time, nice 15, 3 threads.

## cicids (TEST, val_cal, %)

| enc | model | list | F1 | FPR |
|---|---|---|---|---|
| raw | RF | legacy | 99.85 | 0.09 |
| raw | RF | train_only | 99.86 | 0.08 |
| raw | XGB | legacy | 99.71 | 0.09 |
| raw | XGB | train_only | 99.73 | 0.08 |
| 8b | RF | legacy | 99.52 | 0.21 |
| 8b | RF | train_only | 99.44 | 0.26 |
| 8b | XGB | legacy | 99.44 | 0.23 |
| 8b | XGB | train_only | 99.36 | 0.30 |

## ciciot (TEST, val_cal, %)

| enc | model | list | F1 | FPR |
|---|---|---|---|---|
| raw | RF | legacy | 95.56 | 6.47 |
| raw | RF | train_only | 95.78 | 5.82 |
| raw | XGB | legacy | 94.55 | 7.90 |
| raw | XGB | train_only | 94.70 | 7.89 |
| 8b | RF | legacy | 88.69 | 15.53 |
| 8b | RF | train_only | 89.28 | 14.92 |
| 8b | XGB | legacy | 88.85 | 15.10 |
| 8b | XGB | train_only | 89.25 | 15.60 |

## UNSW-random control (no leak)

| model | legacy | train_only | ΔF1 |
|---|---|---|---|
| RF | 95.84 / 0.32 | 96.16 / 0.28 | +0.32 |
| XGB | 95.60 / 0.43 | 95.65 / 0.38 | +0.05 |

## Overlap

| dataset | test rows in the ranking partition | natural train→test duplicates |
|---|---|---|
| cicids | 76.3% | 13.4% |
| ciciot | 83.8% | 4.0% |

## Caveats

- Not measured on the WNN itself; the design doc's estimate (a) or an optional IDS53 legacy cell would cover that.
- Significance is not tested: XGB has SD 0 and only the model seed varies.
- Per-seed rows are in docs/ids53_leak/.
