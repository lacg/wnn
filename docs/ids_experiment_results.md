# IDS Experiment Results — UNSW-NB15 Binary Single-Cluster

All flows use: pop=150, neuron_sample_rate=0.25, max_bits=16.
Val metrics: best F1-macro across all experiments in each flow.

## Controlled Experiments (Flows 128-157)

Systematic variation of pool_shuffle, cluster_crossover, assortative_mating.

| Flow | Shuffle | XO  | Mating | Fitness  | Val F1 | Val FPR | Val Acc |
|------|---------|-----|--------|----------|--------|---------|---------|
| 129 |     N/A | N/A |    N/A | recall   | 84.74% | 30.54% | 85.55% |
| 132 |     N/A |   0 |    N/A | security | 83.34% | 33.28% | 84.33% |
| 133 |     N/A |   1 |    N/A | recall   | 81.26% | 37.80% | 82.59% |
| 134 |     N/A | 0.5 |    N/A | security | 87.45% | 15.66% | 87.63% |
| 136 |       1 | 0.5 |    N/A | recall   | 84.36% | 28.84% | 85.06% |
| 137 |       1 |   0 |    N/A | recall   | 83.78% | 32.20% | 84.70% |
| 138 |       1 | 0.5 |    N/A | recall   | 85.28% | 25.78% | 85.82% |
| 139 |       1 |   0 |    N/A | recall   | 82.22% | 35.54% | 83.37% |
| 140 |       0 |   0 |    0.8 | recall   | 87.50% | 21.76% | 87.88% |
| 141 |       1 |   1 |    0.8 | recall   | 84.27% | 29.58% | 85.01% |
| 144 |     0.5 | 0.5 |    0.8 | recall   | 83.60% | 22.48% | 83.94% |
| 146 |       1 |   0 |      0 | recall   | 86.74% | 23.81% | 87.21% |
| 147 |       0 |   0 |      0 | recall   | 80.32% | 39.36% | 81.78% |
| 148 |       0 |   1 |    0.8 | recall   | 86.40% | 22.91% | 86.82% |
| 149 |       1 | 0.8 |    0.8 | recall   | 86.92% | 23.51% | 87.37% |
| 150 |       1 |   1 |    0.8 | recall   | 84.05% | 11.13% | 84.07% |
| 151 |       1 |   0 |    0.8 | recall   | 87.51% | 23.60% | 87.98% |
| 152 |       0 |   1 |    0.8 | recall   | 82.92% | 28.99% | 83.59% |
| 153 |       1 |   1 |    0.8 | recall   | 79.04% | 42.55% | 76.29% |
| 154 |       1 |   1 |    0.8 | security | 84.54% | 24.39% | 84.99% |
| 155 |       1 |   0 |    0.8 | recall   | 89.93% | 14.70% | 90.11% |
| 156 |       1 |   0 |    0.8 | security | 84.41% | 29.33% | 85.13% |
| 157 |       1 |   1 |    0.8 | recall   | 84.83% | 29.78% | 85.60% |

## Grid Search: Harmonic-FPR Fitness (Flows 204-218)

Systematic grid over shuffle × crossover with harmonic_rank fitness.

| Flow | Shuffle | XO  | Val F1 | Val FPR | Val Acc | Status |
|------|---------|-----|--------|---------|---------|--------|
| 204 |       1 |   0 | 84.02% | 19.56% | 72.63% | completed |
| 205 |       0 | 0.5 | 83.94% | 9.14% | 84.82% | completed |
| 206 |       0 | 0.8 | 80.45% | 39.96% | 82.18% | completed |
| 207 |     0.5 |   0 | 85.98% | 21.44% | 83.14% | completed |
| 208 |     0.5 | 0.5 | 86.14% | 19.68% | 79.47% | completed |
| 209 |     0.5 | 0.8 | 84.95% | 26.34% | 83.74% | completed |
| 210 |     0.5 |   1 | 78.58% | 5.60% | 81.19% | completed |
| 211 |     0.8 |   0 | 81.96% | 36.74% | 80.74% | completed |
| 212 |     0.8 | 0.5 | 88.11% | 10.25% | 83.36% | completed |
| 213 |     0.8 | 0.8 | 86.42% | 24.01% | 81.89% | completed |
| 214 |     0.8 |   1 | 87.16% | 11.25% | 84.35% | completed |
| 215 |     0.9 |   0 | 88.84% | 13.30% | 81.00% | completed |
| 216 |     0.9 | 0.5 | 84.78% | 30.15% | 81.92% | completed |
| 217 |     0.9 | 0.8 | 88.28% | 15.37% | 84.40% | completed |
| 218 |     0.9 |   1 | 83.02% | 32.99% | 82.34% | completed |

## Grid Search: Recall Fitness (Flows 219-234)

Systematic grid over shuffle × crossover with ids_recall fitness.

| Flow | Shuffle | XO  | Val F1 | Val FPR | Val Acc |
|------|---------|-----|--------|---------|---------|
| 219 |       1 | 0.5 | 84.08% | 30.75% | 81.37% |
| 220 |       0 | 0.5 | 84.43% | 7.72% | 81.54% |
| 221 |       0 | 0.8 | 82.21% | 33.65% | 82.64% |
| 222 |     0.5 |   0 | 88.95% | 9.41% | 83.47% |
| 223 |     0.5 | 0.5 | 83.32% | 31.83% | 81.60% |
| 224 |     0.5 | 0.8 | 84.38% | 29.54% | 86.32% |
| 225 |     0.5 |   1 | 83.88% | 27.55% | 84.14% |
| 226 |     0.8 |   0 | 80.38% | 35.31% | 80.03% |
| 227 |     0.8 | 0.5 | 90.54% | 11.51% | 82.88% |
| 228 |     0.8 | 0.8 | 82.37% | 34.07% | 87.21% |
| 229 |     0.8 |   1 | 84.41% | 29.69% | 83.25% |
| 230 |     0.9 |   0 | 83.69% | 32.23% | 81.61% |
| 231 |     0.9 | 0.5 | 85.57% | 28.18% | 81.50% |
| 232 |     0.9 | 0.8 | 87.96% | 15.91% | 83.13% |
| 233 |     0.9 |   1 | 81.21% | 38.35% | 83.27% |
| 234 |       1 | 0.5 | 81.84% | 36.66% | 84.82% |

## Leaderboard Top 10

| Rank | Flow | F1-Macro | FPR | Acc | Config |
|------|------|----------|-----|-----|--------|
| 1 | 227 | 90.54% | 11.51% | 82.88% | S0.8-C0.5-M0.8-recall |
| 2 | 227 | 90.50% | 9.72% | 82.64% | S0.8-C0.5-M0.8-recall |
| 3 | 155 | 89.93% | 14.70% | 90.11% | S1-C0-M0.8-recall |
| 4 | 227 | 88.97% | 18.94% | 80.96% | S0.8-C0.5-M0.8-recall |
| 5 | 227 | 88.97% | 18.94% | 80.96% | S0.8-C0.5-M0.8-recall |
| 6 | 222 | 88.95% | 9.41% | 83.47% | S0.5-C0-M0.8-recall |
| 7 | 222 | 88.95% | 9.41% | 83.47% | S0.5-C0-M0.8-recall |
| 8 | 215 | 88.84% | 13.30% | 81.00% | S0.9-C0-M0.8-harmonic |
| 9 | 217 | 88.28% | 15.37% | 84.40% | S0.9-C0.8-M0.8-harmonic |
| 10 | 212 | 88.11% | 10.25% | 83.36% | S0.8-C0.5-M0.8-harmonic |

## Key Findings

- **Best F1**: Flow 227 (90.54%, S0.8-C0.5-M0.85-recall)
- **Best FPR with good F1**: Flow 222 (88.95% F1, 9.41% FPR, S0.5-C0.0)
- **Harmonic fitness**: Competitive with recall (F215 88.84%, F217 88.28%)
- **Phase analysis**: GA Neurons is the only consistently positive phase (+2.4pt avg)
- **TS phases regress** 55-77% of flows — switched to 2-phase explorer template
- **Threshold calibration**: Platt/Beta/Empirical calibration added for distribution-adaptive thresholds
