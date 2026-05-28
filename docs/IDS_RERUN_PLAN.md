# IDS GA Re-run Plan (post offspring-eval + fitness_scores fix)

**Status**: drafted 2026-05-28 after the dual-bug fix landed in commits `6ab34164` (eval-path) and `d6b658bd` (fitness_scores). Both bugs introduced in commit `18662b5f` on **2026-03-09** when the IDS pipeline was first added.

## Scope of impact

**Every IDS GA run between 2026-03-09 and 2026-05-28** has both bugs in its training data:
1. GA tournament selection used raw CE for parent picking, ignoring the configured `fitness_calculator` weights (ce/acc/f1/fpr).
2. Offspring evaluation read the 20% held-out test set (`cache.full_eval()`) during search — a methodology violation per CLAUDE.md.

**What's still trustworthy in the prior data**:
- Grid search results (the grid phase does NOT use the offspring-eval code path — different function)
- Elite selection within each GA gen (Python-side, uses harmonic_rank correctly)
- The reported "best" genome per metric type (best_ce, best_f1, etc.) is the harmonic-rank survivor of its (biased) candidate pool

**What's affected**:
- The GA EXPLORATION direction — biased toward low-CE genomes regardless of weight set
- Offspring metrics in `genome_evaluations` — computed on test-set, not directly comparable to elite metrics
- Cross-cohort weight-set comparisons (A vs B) — the GA explored identically in both

## Cohort priority + compute estimates

Listed by paper-criticality. Per-flow times are observed/estimated; 30 flows minimum per cohort target (iterative up to 100).

| # | Cohort | Status | Flow count (V1) | Per-flow time | Initial re-run (30) | Notes |
|---|--------|--------|-----------------|---------------|---------------------|-------|
| **1** | **46M Search(46M)** — 2747/2748 family | paper Table 5 | 2 | ~14h | **~17 days serial** | Highest priority; the headline result. Could run 1 at a time, take ~3-5 days for n=10 minimum quorum, then iterate. |
| **2** | **FIXED 500n×34b** on 1.43M subsample | paper baseline | 29 | ~3h | ~4 days serial | Pre-OI cohort. The "FIXED-OLD" runs that beat prior C35. |
| **3** | **OI cohort v2 250n×100b** on 1.43M subsample | paper baseline | ~100 | ~13h | ~16 days serial | The OI-trained 250n×100b cohort. Big compute. |
| **4** | **XDS UNSW-temporal** | now in flight (V2) | 30 (queued at 500n×34b) | ~13h | ~16 days serial | Already started post-fix; verifies the fix end-to-end too. |
| **5** | **XDS UNSW-random** | not yet queued | 0 | ~13h | ~16 days serial | Next cross-dataset target. |
| **6** | **XDS CICIDS-random** | not yet queued | 0 | ~13h | ~16 days serial | Last cross-dataset target. |
| 7 | PUB50 batches (6 × 50 = 300) | older cohorts | 300 | ~30 min | ~6 days serial | Skip if paper doesn't lean on them — may be obsoleted by OI cohort. |

## Iterative schedule (per cohort)

- **Round 1 (now)**: 30 flows. Establishes the new (post-fix) baseline.
- **Round 2 (+20 → 50 total)**: only if Round 1 results warrant tighter quorum.
- **Round 3 (+50 → 100 total)**: paper-grade quorum if needed.

The +20/+50 increments let us decide AFTER seeing Round 1 whether the GA actually behaves differently with the fixes (closer offspring-elite tracking, weight-respecting exploration).

## Paper methodology disclosure (draft)

For the camera-ready, the honest one-paragraph disclosure for the methodology section:

> *We discovered two related bugs in the GA's offspring evaluation pipeline that affected runs prior to 2026-05-28. First, the offspring evaluation used the held-out test partition rather than a K-fold of the training set (a deviation from our stated 80/20 split with K-fold on training). Second, parent selection in the tournament used raw cross-entropy rather than the configured harmonic-rank fitness, neutralizing weight-set variations at the exploration stage (elite selection and final reporting were unaffected). We re-ran all cohorts reported in this paper with corrected code; reported metrics are from the corrected runs. Pre-fix raw data is retained in our research log for reproducibility.*

## Action items (sequenced)

1. ✅ Stop worker, fix both bugs, rebuild Rust accelerator, restart worker (done 2026-05-28).
2. 🟡 **XDS UNSW-temporal (Cohort 4)**: 30 flows queued, currently in flight as verification — does the GA now show offspring metrics that overlap with elite metrics like CIC-IoT historically did?
3. ⏸️ **46M Search(46M) (Cohort 1)**: queue 2-3 flows first (not 30) given 14h/flow. Decide n based on first results.
4. ⏸️ **FIXED 500n×34b (Cohort 2)**: queue 30 once XDS verification passes.
5. ⏸️ **OI cohort v2 (Cohort 3)**: queue 30 — biggest compute commitment.
6. ⏸️ **XDS UNSW-random + CICIDS-random (Cohorts 5, 6)**: queue after Cohort 4 demonstrates the cross-dataset workflow works post-fix.
7. ⏸️ **PUB50 (Cohort 7)**: skip unless paper structure requires it.

## Cost-time math (rough)

Total serial-compute for Cohorts 1-6 at 30 flows each:
- 17 + 4 + 16 + 16 + 16 + 16 = **~85 days serial**

With the worker only running 1 flow at a time on the Mac Studio, that's ~3 months serial. Realistically:
- Parallelize cohorts ONLY if we add a second compute host (Mac Studio can run 1 IDS flow at a time at production parameters).
- Stagger by priority: 46M first (paper Table 5), then FIXED + OI in parallel sequence, cross-dataset last.
- Iterate Round 2/3 only on the cohorts where Round 1 stats are borderline.

## Open question worth discussing

Given the methodology disclosure framing above, **do we re-run the FULL 100-flow cohorts or accept 30-flow re-runs as the camera-ready data?** IDS literature commonly reports best-of-N or mean±std at n=30, which is statistically defensible. The 100-flow cohort sizes were chosen for trim-top/bottom-5% which the cohort-size-n=100 memory notes was "orphaned" anyway. Round 1 at 30 might be sufficient for the paper.
