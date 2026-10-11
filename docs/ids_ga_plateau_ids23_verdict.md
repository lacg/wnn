# IDS-23 — GA plateau spike: verdict (10/10/2026)

Question: why does GA Neurons stop improving held-out F1 (most visibly on UNSW random, ~93.5 F1)?
Three explanations, one diagnostic each, all read-only (no flows, worker or GA config touched):

| Hypothesis | Diagnostic | Result |
|---|---|---|
| Dataset ceiling | IDS-24 grid-best vs GA-best, paired per run on held-out TEST | **Supported on UNSW random**: GA − grid = −0.007±0.023 pp (N=43, 15/19/9). Elsewhere the GA adds: UNSW temporal +0.41 (28/29), CICIDS +0.20 (32/32), CIC-IoT +2.15 (41/41). |
| Population collapse | IDS-26 final-population diversity | **Partial, not causal**: 1 basin per run (131/131), but different seeds/arms end with 0 shared neurons — the plateau is reached from unrelated genomes. |
| Objective mismatch (k-fold fitness ≠ held-out) | IDS-25 Spearman, grid top-50 re-scored on held-out | **Not on UNSW random**: k-fold F1 vs held-out F1 ρ=+0.88, regret 0.011 pp. Real on UNSW temporal (train→test shift, regret up to 6 pp) and CIC-IoT FPR (ρ −0.31, fitness-best 3.0 pp FPR worse). |

**Verdict: the UNSW-random plateau is a dataset/representation ceiling, not a GA failure.**
The GA's measurable value is dataset-dependent (≈0 on UNSW random, +0.2 to +2.1 pp elsewhere) and
that is how the paper should state it. On UNSW random the GA still buys a Pareto move (lower FPR
at F1 ≥ 93 in 31/43 runs), not F1.

Side findings carried forward (none applied; the n=100 tandem keeps the current GA):
- 67-78% of GA-best neurons are exact duplicates (pool-shuffle crossover, no dedupe) → candidate
  for smaller FPGA genomes at equal F1 (IDS-26 proposal 1).
- `immigrant_fraction` is a no-op on the IDS Rust path (IDS-26).
- Harmonic-rank `best_fitness` on a tied population crowns tiny specialists (SP100 unswr −1.5±2.7 pp
  F1). The n=100 readout's criterion A (mean val_cal F1 over 4 datasets) penalises such a cell on
  its own; no rule change needed.
- Remaining real mismatch (temporal shift, CIC-IoT FPR) is an objective-design question, not a
  plateau one.

Sources: docs/ids_ga_plateau_ids24_ids25.md (0f8049c0), docs/ids_ga_plateau_ids26.md (29a0a60a).
