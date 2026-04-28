# Camera-ready draft update (generated 20260427T234041Z)

## Headline numbers comparison (train_cal threshold)

| Flow | Best F1 | Best FPR | Best Acc |
|---|---|---|---|
| r98 (paper, 38.5M) | 86.58/2.77/98.07 | 84.78/1.87/97.69 | 86.58/2.77/98.07 |
| r125 (canonical, 45M) | 84.75/1.54/98.08 | 85.05/2.46/98.14 | 84.02/0.82/97.93 |
| r124 (canonical, 45M) | 84.74/1.63/98.08 | 84.12/1.50/97.96 | 84.73/1.65/98.07 |

_Format: F1% / FPR% / Acc%. Each cell is the genome optimal for that metric._

## Classical ML baselines on canonical-neto (45M)

| Model | F1 | FPR | Acc |
|---|---|---|---|
| RF | 92.49% | 13.66% | 99.31% |
| XGB | 91.4% | 14.99% | 99.2% |

## Per-class breakdown

| Class | RF | XGB |
|---|---|---|
| Benign | 13.66% | 14.99% |
| BruteForce | 72.47% | 67.64% |
| DDoS | 100.00% | 100.00% |
| DoS | 100.00% | 100.00% |
| Mirai | 100.00% | 100.00% |
| Recon | 81.04% | 77.30% |
| Spoofing | 92.60% | 90.96% |
| Unknown | 100.00% | 100.00% |
| Web-based | 81.11% | 74.72% |

_Benign row is FPR (false alarms); attack rows are recall (detection rate)._

---
## Sources
- DB: `/Users/lacg/wnn/db/wnn.db` (best_genomes table for flows 1156, 1687, 1686)
- Baselines log: `/Users/lacg/wnn/logs/canonical_baselines_perclass_20260426T123105Z.log`