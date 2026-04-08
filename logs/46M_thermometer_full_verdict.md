# 46M Full Thermometer Encoding Sweep Verdict

Combined 6-point thermometer-width curve (2/4/8/16/32/64-bit) on the
full CIC-IoT-2023 46M dataset, across two architectures.

**Data sources:**
- 8-bit baselines from the Pareto sweep (flows 1184, 1185, 1188)
- 16/32/64-bit from THERMO46M (flows 1189-1200)
- 2/4-bit from THERMO46MLOW (flows 1201-1208)

## 96n × 32b

| Thermometer | Seeds | F1 | FPR | Acc |
|---|---:|---:|---:|---:|
| 2-bit | 2 | 85.63±0.06% | 1.13±0.08% | 97.85±0.01% |
| 4-bit | 2 | 84.24±0.28% | 1.33±0.33% | 97.56±0.05% |
| 8-bit | 1 | 84.36% | 1.61% | 97.59% |
| 16-bit | 2 | 84.13±0.08% | 1.17±0.22% | 97.53±0.02% |
| 32-bit | 2 | 84.53±0.25% | 1.50±0.27% | 97.63±0.06% |
| 64-bit | 2 | 84.59±0.13% | 1.69±0.17% | 97.64±0.03% |

F1 range across widths: **1.62pp**

## 400n × 8b

| Thermometer | Seeds | F1 | FPR | Acc |
|---|---:|---:|---:|---:|
| 2-bit | 2 | 84.17±0.11% | 1.93±0.59% | 97.56±0.04% |
| 4-bit | 2 | 82.32±0.68% | 2.59±0.08% | 97.15±0.16% |
| 8-bit | 1 | 83.13% | 2.06% | 97.33% |

F1 range across widths: **2.41pp**

## 500n × 34b

| Thermometer | Seeds | F1 | FPR | Acc |
|---|---:|---:|---:|---:|
| 8-bit | 1 | 84.73% | 1.91% | 97.68% |
| 16-bit | 2 | 84.43±0.21% | 1.51±0.05% | 97.60±0.05% |
| 32-bit | 2 | 84.46±0.41% | 1.37±0.18% | 97.61±0.09% |
| 64-bit | 2 | 84.87±0.03% | 1.75±0.02% | 97.70±0.01% |

F1 range across widths: **0.72pp**

## Verdict

- **96n × 32b:** best thermometer is **2-bit** with F1 85.63% (+1.26pp vs 8-bit baseline).
- **400n × 8b:** best thermometer is **2-bit** with F1 84.17% (+1.04pp vs 8-bit baseline).
- **500n × 34b:** FLAT — best thermometer 64b at 84.87% F1, range 0.44pp across all widths.

