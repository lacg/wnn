# Post-RAID-2026 Research Roadmap

**Status**: paper submitted 17/04/2026. This roadmap covers experiments to run *after* the current UNSW random + 46M GA pipeline finishes (ETA ~20-22/04/2026), before camera-ready or for follow-up publications.

All experiments below run on **UNSW-NB15 temporal** (~8 min/run → fast iteration) unless noted.
**5 flows per config** for mean ± std statistical value.
**Sequential execution** (parallel flows hit memory/CPU contention based on prior investigation).

## Execution order

### Phase 1: Quick wins (config-only, fast convergence)

- [ ] **#1 Population size re-sweep** with aligned fitness + K-fold=5
  - Configs: pop ∈ {50, 100, 150, 200} = **4 configs**
  - 4 × 5 runs × 8 min ≈ **2.7 hours**
  - Hypothesis: prior overfitting at pop=150 was due to misaligned fitness; K-fold + balanced weights should cure it.
  - Expected winner: pop=150 edges pop=50 by ~0.2-0.5 pp F1 with tighter std.

- [ ] **#5 Grid top-K sweep**
  - Configs: top_k ∈ {5, 10, 15, 20, 30} = **5 configs**
  - 5 × 5 runs × 8 min ≈ **3.3 hours**
  - Hypothesis: larger K = more architectural diversity at GA start. Too-large K may dilute strong initial seeds.

### Phase 2: Methodical sweeps

- [ ] **#2 Auto-thermo 2b→64b step-by-2b**
  - Configs: max_bits ∈ {2, 4, 6, ..., 64} = **32 configs**
  - 32 × 5 runs × 8 min ≈ **21 hours**
  - Methodical follow-up to the earlier ad-hoc sweep that found Auto:18b beat uniform 8b by +0.83-1.62 pp F1.
  - Goal: locate the thermometer "elbow" precisely + characterize the FPR-encoding relationship.

- [ ] **#4 Fitness-weight grid (coarse-first, zoom later)**
  - Coarse grid {0, 0.5, 1}⁴: 80 raw → **65 unique after scale-dedup** × 5 runs × 8 min ≈ **43 hours**
  - Fine grid {0, 0.25, 0.5, 1}⁴: 255 raw → **175 unique** × 5 runs × 8 min ≈ **117 hours**
  - Strategy: run coarse first. If a weight region is Pareto-promising, zoom with fine grid in that region only.
  - **Dedup by scale**: canonical form is normalized by sum — (1,1,0,0), (0.5,0.5,0,0), (0.25,0.25,0,0) produce identical rankings.

### Phase 3: Full 10-phase pipeline on current best config (swapped earlier from Phase 4)

**Rationale for moving this forward**: we want to see whether the 10-phase
pipeline's behavior has improved under our recent infrastructure changes
(aligned fitness, K-fold=5, QUAD_WEIGHTED mode, split audit). This runs on
the CURRENT best-known UNSW temporal config — not one yet to be discovered
by Phase 1/2. Phase 1/2 findings can later be combined with this data for
a richer analysis.

- [ ] **#6 10-phase pipeline on the current best UNSW temporal config**
  - Phases (in correct order, connections last):
    1. Grid Search (architecture)
    2. GA Neurons
    3. TS Neurons
    4. GA Bits
    5. TS Bits
    6. Lamarckian Neurons *(if available)*
    7. Lamarckian Bits *(if available)*
    8. GA Connections
    9. TS Connections
    10. Lamarckian Connections *(if available)*
  - 5 flows at the current best-fitness GA config (pop 50, K-fold 5,
    balanced aligned fitness) — the same setup that produced our 101-run
    paper numbers and the overnight 11-flow extension.
  - Expected per-flow time: 20+ min (cascading phases) — so 5 × ~20+ min ≈
    **2+ hours** minimum, more likely 8-20 hours depending on connection-
    phase convergence.
  - **Research question**: does adding bits/connections/Lamarckian phases
    meaningfully move F1 beyond the 2-phase (Grid + GA Neurons) baseline
    of 86.95% train-cal / 87.30% val-cal? Or is GA Neurons already
    saturating the easy gains?

### Phase 4: Methodological improvements (code changes)

- [ ] **Mini-ensemble of 3 seeds** (baseline ensemble approach)
  - Run same config 3× with different seeds, majority vote on predictions.
  - Simplest form of Expert Voting (#3). Establishes ensemble baseline.
  - Expected: +0.5-1.0 pp F1 and tighter FPR variance vs single seed.

- [ ] **#3 Expert voting (easy version)**
  - 2-3 separate GA runs with different fitness weights (e.g., F1-heavy, FPR-heavy), then ensemble predictions via voting.
  - Compared against Mini-ensemble baseline to isolate "specialization" value.
  - Requires post-hoc ensemble script (not a new worker mode).
  - If promising → escalate to **harder version**: single GA that evolves a population of specialists with diverse fitness objectives (NSGA-II style).

- [ ] **3-way split reactivation** (train/test/val) for stronger generalization claim
  - Current: HF `_3way` configs have 80/10/10 but worker merges test+val into single 20% held-out validation. Early stopping uses K-fold CV training fitness → no training-time holdout signal.
  - **Proposed**: stop merging; use the 10% test as **training-time holdout** (early stopping, threshold calibration), keep 10% val **completely untouched** for final validation report.
  - This aligns with standard ML practice: train/val/test where val guides training decisions and test is the final untouched holdout. (Our current "validation" is actually a "test" in the standard nomenclature — which is why the paper uses `val_cal`.)
  - Worker change: add a toggle `ids_merge_test_val=false` so early stopping uses test fitness, final eval uses val fitness.
  - **GATED**: do AFTER current pipeline finishes — changing the protocol mid-run would break camera-ready consistency.
  - **Reviewer appeal**: strengthens the paper's claim from "val is untouched during training" to "val is COMPLETELY untouched AND early stopping uses a separate held-out test set" — closes the only lingering methodology gap.

## Budget

| Phase | Configs | Flows | Time |
|-------|---------|-------|------|
| Phase 1 (pop + top-K) | 9 | 45 | ~6 hours |
| Phase 2 (coarse thermo + fitness-weight) | 97 | 485 | ~64 hours |
| Phase 2 (fine — if needed) | +110 | +550 | +~73 hours |
| Phase 3 (10-phase on current best) | 1 | 5 | ~8-20 hours |
| Phase 4 (methodology: mini-ensemble, expert voting, 3-way split) | ~5-10 | 15-50 | varies |
| **Total first pass** | | | **~80-100 hours (~4 days)** |

## Active research questions

1. **Does the Pareto-favorable-genome pattern generalize?** (see `project_calibrated_threshold_pareto.md`) — run the f1566-Platt-style analysis across all 4 datasets' submitted appendix data. Result feeds into the Expert Voting story.
2. **What genome property predicts Platt/Beta calibration success?** — score distribution structure, bimodality, skew — can we pick the right threshold mode without running all 7?
3. **Per-feature thermometer entropy** — is Auto:18b winning because it matches feature cardinality distributions? Map features to their unique-value counts and correlate with optimal per-feature bit allocation.

## Notes

- Current 46M GA flows (id 1155, 1156) are queued but will run LAST (lowest IDs). Do NOT include in this roadmap — they belong to the original paper's camera-ready completion.
- Keep all experiments on UNSW temporal until a winning config is identified, THEN transfer to the other 3 datasets to confirm generalization (separate follow-on phase).
- All experiments here produce data that supports camera-ready additions or a follow-up paper — they do not alter the submitted paper's contributions.

## Tracking

Update this file as experiments complete. Use the checkboxes above. When results arrive, append findings to each experiment's section.
