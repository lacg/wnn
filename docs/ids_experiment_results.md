# IDS Experiment Results — UNSW-NB15 Binary Single-Cluster

All flows use: pop=150, neuron_sample_rate=0.25, max_bits=16.
Val metrics: holdout-calibrated threshold where available (primary metric).
Val F1/FPR/Acc: best genome across all experiments in the flow.

## Early Experiments (Flows 122-139)

Predate configurable shuffle/crossover/mating parameters.

| Flow | Fitness  | Fold F1 | Val F1 | Val FPR | Val Acc | Gap  | Notes |
|------|----------|---------|--------|---------|---------|------|-------|
| 122 | recall   | 93.36% | 90.83% | 7.81% | 90.88% | 3pt  | 20260312-IDS-Bin-FPR1-Single-0 |
| 123 | security | 93.26% | 89.03% | 11.31% | 89.13% | 4pt  | 20260312-IDS-Single-AdaptThresh |
| 124 | security | 93.33% | 84.63% | 29.85% | 85.39% | 9pt  | 20260313-IDS-FPR2-Single-0 |
| 125 | recall   | 93.24% | 85.05% | 30.10% | 85.84% | 8pt  | 20260313-IDS-FPR1-Single-0 |
| 126 | security | 92.88% | 83.91% | 16.72% | 84.05% | 9pt  | 20260313-IDS-FPR2-Double-0 |
| 127 | recall   | 92.99% | 84.78% | 28.77% | 85.49% | 8pt  | 20260313-IDS-FPR1-Double-0 |
| 129 | recall   | 92.87% | 84.74% | 30.54% | 85.55% | 8pt  | 20260313-IDS-FPR1-Single-12b-0 |
| 132 | security | 92.93% | 83.34% | 33.28% | 84.33% | 10pt | 20260313-IDS-Bin-FPR1-Single-16b-2 |
| 133 | recall   | 92.63% | 81.26% | 37.80% | 82.59% | 11pt | 20260313-IDS-Bin-FPR1-Single-16b-3 |
| 134 | security | 92.64% | 87.45% | 15.66% | 87.63% | 5pt  | 20260313-IDS-Bin-FPR1-Single-16b-4 |
| 136 | recall   | 92.70% | 84.36% | 28.84% | 85.06% | 8pt  | 20260313-IDS-Bin-FPR1-Single-16b-6 |
| 137 | recall   | 93.12% | 83.78% | 32.20% | 84.70% | 9pt  | 20260313-IDS-Bin-FPR1-Single-16b-7 |
| 138 | recall   | 93.03% | 85.28% | 25.78% | 85.82% | 8pt  | 20260313-IDS-Bin-FPR1-Single-16b-8 |
| 139 | recall   | 92.96% | 82.22% | 35.54% | 83.37% | 11pt | 20260313-IDS-Bin-FPR1-Single-16b-9 |

## Controlled Experiments (Flows 140-157)

Systematic variation of pool_shuffle, cluster_crossover, assortative_mating.

| Flow | Shuffle | XO  | Mating | Fitness  | Fold F1 | Val F1 | Val FPR | Val Acc | Gap  | Notes |
|------|---------|-----|--------|----------|---------|--------|---------|---------|------|-------|
| 140 |       0 |   0 |    0.8 | recall   | 92.52% | 87.50% | 21.76% | 87.88% | 5pt  | 20260314-IDS-Bin-FPR1-Single-16b-0 |
| 141 |       1 |   1 |    0.8 | recall   | 92.49% | 84.27% | 29.58% | 85.01% | 8pt  | 20260314-IDS-Bin-FPR1-Single-16b-1 |
| 144 |     0.5 | 0.5 |    0.8 | recall   | 93.26% | 83.60% | 22.48% | 83.94% | 10pt | 20260314-IDS-Bin-FPR1-Single-16b-4 |
| 146 |       1 |   0 |      0 | recall   | 93.00% | 86.74% | 23.81% | 87.21% | 6pt  | 20260314-IDS-Bin-FPR1-Single-16b-6 |
| 147 |       0 |   0 |      0 | recall   | 92.80% | 80.32% | 39.36% | 81.78% | 12pt | 20260314-IDS-Bin-FPR1-Single-16b-7 |
| 148 |       0 |   1 |    0.8 | recall   | 92.51% | 86.40% | 22.91% | 86.82% | 6pt  | 20260314-IDS-Bin-FPR1-Single-16b-8 |
| 149 |       1 | 0.8 |    0.8 | recall   | 92.36% | 86.92% | 23.51% | 87.37% | 5pt  | 20260314-IDS-Bin-FPR1-Single-16b-9 |
| 150 |       1 |   1 |    0.8 | recall   | 92.37% | 84.05% | 11.13% | 84.07% | 8pt  | 20260314-IDS-Bin-FPR1-Single-16b-10 |
| 151 |       1 |   0 |    0.8 | recall   | 93.02% | 87.51% | 23.60% | 87.98% | 6pt  | 20260314-IDS-Bin-FPR1-Single-16b-11 |
| 152 |       0 |   1 |    0.8 | recall   | 92.58% | 82.92% | 28.99% | 83.59% | 10pt | 20260314-IDS-Bin-FPR1-Single-16b-12 |
| 154 |       1 |   1 |    0.8 | security | 92.62% | 84.54% | 24.39% | 84.99% | 8pt  | 20260315-IDS-Bin-FPR2-Single-16b-0 |
| 155 |       1 |   0 |    0.8 | recall   | 93.03% | 89.93% | 14.70% | 90.11% | 3pt  | 20260315-IDS-Bin-FPR1-Single-16b-1 |
| 156 |       1 |   0 |    0.8 | security | 93.16% | 84.41% | 29.33% | 85.13% | 9pt  | 20260315-IDS-Bin-FPR2-Single-16b-2 |
| 157 |       1 |   1 |    0.8 | recall   | 92.59% | 84.83% | 29.78% | 85.60% | 8pt  | 20260315-IDS-Bin-FPR1-Single-16b-2 |

## Grid Search: Shuffle x Crossover (Flows 219-234)

Systematic grid over pool_shuffle_ratio and cluster_crossover_ratio.
All use: ids_recall fitness, assortative_mating=0.85, 7-phase.

| Flow | Shuffle | XO  | Fold F1 | Val F1 | Val FPR | Val Acc | Gap  | Status |
|------|---------|-----|---------|--------|---------|---------|------|--------|
| 219 |       1 | 0.5 | 93.39% | — | — | — |      | running |
| 220 |       0 | 0.5 | 92.76% | 84.43% | 7.72% | 81.54% | 8pt  | completed |
| 221 |       0 | 0.8 | 93.06% | 82.21% | 33.65% | 82.64% | 11pt | completed |
| 222 |     0.5 |   0 | 93.11% | 88.95% | 9.41% | 83.47% | 4pt  | completed |
| 223 |     0.5 | 0.5 | 93.22% | 83.32% | 31.83% | 81.60% | 10pt | completed |
| 224 |     0.5 | 0.8 | 93.20% | 84.38% | 29.54% | 86.32% | 9pt  | completed |
| 225 |     0.5 |   1 | 93.05% | 83.88% | 27.55% | 84.14% | 9pt  | completed |
| 226 |     0.8 |   0 | 93.00% | 80.38% | 35.31% | 80.03% | 13pt | completed |
| 227 |     0.8 | 0.5 | 93.14% | 90.54% | 11.51% | 82.88% | 3pt  | completed |
| 228 |     0.8 | 0.8 | 93.21% | 82.37% | 34.07% | 87.21% | 11pt | completed |
| 229 |     0.8 |   1 | 93.01% | 84.41% | 29.69% | 83.25% | 9pt  | completed |
| 230 |     0.9 |   0 | 93.34% | 83.69% | 32.23% | 81.61% | 10pt | completed |
| 231 |     0.9 | 0.5 | 93.17% | 85.57% | 28.18% | 81.50% | 8pt  | completed |
| 232 |     0.9 | 0.8 | 93.19% | 87.96% | 15.91% | 83.13% | 5pt  | completed |
| 233 |     0.9 |   1 | 93.16% | 81.21% | 38.35% | 83.27% | 12pt | completed |
| 234 |       1 | 0.5 | 93.17% | 81.84% | 36.66% | 84.82% | 11pt | completed |

## Key Findings

- **Best Val F1**: Flow 148 (86.40%, shuffle=0, XO=1.0, mating=0.85) — 6pt gap
- **Best balanced (F1+FPR)**: Flow 148 (F1=86.40%, FPR=22.91%)
- **Consistent fold F1**: ~92.2-93.3% across all configs (training is stable)
- **Generalization gap**: 6-22pt between fold F1 and val F1 (threshold transfer is the bottleneck)
- **Grid search**: High variance suggests random seed sensitivity; no clear shuffle/XO optimum

