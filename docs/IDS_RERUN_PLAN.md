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

## Iterative schedule per cohort: Plan C → Plan B → Plan A

Three-stage rollout per cohort. We execute C first across all cohorts, then iterate B and A on the ones whose Round-1 results warrant the deeper quorum.

| Plan | n (cohort size) | Increment from prior | Purpose |
|------|-----------------|----------------------|---------|
| **Plan C** | **30** | — (starting point) | Establish post-fix baseline. Fast first signal that the fix changed GA behavior (offspring metrics now overlap with elites, weights now drive exploration). Statistically defensible per IDS-literature norm. |
| **Plan B** | **50** | +20 more flows | Tightens distribution stats. Used for cohorts where Plan C results are noisy or borderline-promising. |
| **Plan A** | **100** | +50 more flows | The original promise/quorum size. Used for the paper-headline cohorts where the camera-ready needs n=100 to match what was originally pre-registered. |

**Order of operations across cohorts:** queue Plan C for all paper-critical cohorts FIRST, evaluate after each Round 1 finishes, then iterate B/A only on the cohorts that warrant the additional compute. Each step is a separate go/no-go decision based on the prior round's data.

**Why this order matters:**
- Plan C unblocks the methodology disclosure: even at n=30 we have post-fix data to cite in the camera-ready.
- Plan B answers "does the fix improve the stats meaningfully" before committing to Plan A's full compute.
- Plan A is the paper-grade final state, deployed only where needed.

## Paper methodology disclosure (draft)

For the camera-ready, the honest one-paragraph disclosure for the methodology section:

> *We discovered two related bugs in the GA's offspring evaluation pipeline that affected runs prior to 2026-05-28. First, the offspring evaluation used the held-out test partition rather than a K-fold of the training set (a deviation from our stated 80/20 split with K-fold on training). Second, parent selection in the tournament used raw cross-entropy rather than the configured harmonic-rank fitness, neutralizing weight-set variations at the exploration stage (elite selection and final reporting were unaffected). We re-ran all cohorts reported in this paper with corrected code; reported metrics are from the corrected runs. Pre-fix raw data is retained in our research log for reproducibility.*

## Action items — strict sequential order (per user direction)

**All Plan C probes (30 flows each) → all Plan C cohorts (27 new flows each, +3 reused probe winners = 30 total) → 46M (2 flows) → iterate Plan B (+20) → iterate Plan A (+50).**

Sequential queueing only — the worker pulls by id-DESC so queueing multiple cohorts ahead would scramble the order. Each cohort drains fully before the next is queued.

### Plan C sequence (n=30 per cohort)

| # | Cohort | Phase | Flows | Est. wall time | State |
|---|--------|-------|-------|----------------|-------|
| 1 | XDS UNSW-temporal | probe | 30 | ~6-13 days | 🟡 running (2780-2809, 2809 first) |
| 2 | XDS UNSW-temporal | cohort | 27 | ~5-12 days | ⏸️ queue after probe done, at the winning (width, weight) |
| 3 | XDS UNSW-random | probe | 30 | ~6-13 days | ⏸️ queue after UNSW-temp cohort done |
| 4 | XDS UNSW-random | cohort | 27 | ~5-12 days | ⏸️ |
| 5 | XDS CICIDS-random | probe | 30 | ~10-20 days (2.8M dataset) | ⏸️ |
| 6 | XDS CICIDS-random | cohort | 27 | ~9-18 days | ⏸️ |
| 7 | XDS CIC-IoT subsample (1.43M) | probe | 30 | ~6-13 days | ⏸️ |
| 8 | XDS CIC-IoT subsample | cohort | 27 | ~5-12 days | ⏸️ |
| 9 | CIC-IoT 46M | direct | 2 | ~2 days (2 × 14h serial?) | ⏸️ paper-headline finale |
| 10 | **UNSW-temp at 250n×100b (curiosity)** | curiosity | 2 | ~1 day | ⏸️ queue at end of UNSW-temp cohort with the winning (width, weight) — tests whether 250n×100b actually breaks on the smallest dataset post-bug-fix |

**Plan C subtotal: 232 flows** (120 probes + 108 cohorts + 2 × 46M + 2 curiosity).

### Per-dataset architecture (post-correction)

After verifying actual dataset sizes, the architecture map is:

| Dataset | Train rows | Architecture | Rationale |
|---------|-----------|--------------|-----------|
| UNSW-temp | 175K | **500n × 34b** | 8× smaller than 250n×100b derivation set (1.43M ciciot subsample); historical r1681 baseline |
| UNSW-random | 1.27M | 250n × 100b | Close to derivation size |
| CICIDS-random | 2.26M | 250n × 100b | Larger than derivation |
| CIC-IoT subsample | 1.14M | 250n × 100b | THE derivation dataset |
| CIC-IoT 46M | 37.4M | 250n × 100b | Matches paper Table 5 / 2747+2748 |

Per-dataset override is centralized in `scripts/queue_cross_dataset.py:DATASET_ARCH`. The CLI flags `--max-neurons / --max-bits` allow runtime overrides for one-off curiosity experiments (Cohort 10) without touching the default.

### Plan B sequence (+20 per cohort = 50 total)

After Plan C drains end-to-end, queue +20 additional per probe + cohort + 46M (TBD if 46M gets +20):
- 4 × +20 probes = 80
- 4 × +20 cohorts = 80
- (46M increment: probably +0 or +1 due to 14h/flow cost — discuss)
- **Plan B increment: ~160 flows**

### Plan A sequence (+50 per cohort = 100 total)

After Plan B drains, queue +50 additional per probe + cohort (only on the cohorts that survive the Plan B go/no-go review):
- Up to 4 × +50 probes = 200
- Up to 4 × +50 cohorts = 200
- **Plan A increment: up to 400 flows**

### Cumulative totals

| Plan | Per-cohort cumulative | Total flows cumulative |
|------|-----------------------|------------------------|
| C    | 30                    | 230                    |
| C+B  | 50                    | ~390                   |
| C+B+A | 100                  | ~790                   |

**Sequencing implication**: Plan C alone is ~50-90 days serial on the Mac Studio at single-flow-at-a-time throughput. Plan A across all cohorts is multi-month. Iterative go/no-go after each plan lets us trim down if the post-fix stats settle quickly.

## Cost-time math (rough)

Total serial-compute for Cohorts 1-6 at 30 flows each:
- 17 + 4 + 16 + 16 + 16 + 16 = **~85 days serial**

With the worker only running 1 flow at a time on the Mac Studio, that's ~3 months serial. Realistically:
- Parallelize cohorts ONLY if we add a second compute host (Mac Studio can run 1 IDS flow at a time at production parameters).
- Stagger by priority: 46M first (paper Table 5), then FIXED + OI in parallel sequence, cross-dataset last.
- Iterate Round 2/3 only on the cohorts where Round 1 stats are borderline.

## Open question worth discussing

Given the methodology disclosure framing above, **do we re-run the FULL 100-flow cohorts or accept 30-flow re-runs as the camera-ready data?** IDS literature commonly reports best-of-N or mean±std at n=30, which is statistically defensible. The 100-flow cohort sizes were chosen for trim-top/bottom-5% which the cohort-size-n=100 memory notes was "orphaned" anyway. Round 1 at 30 might be sufficient for the paper.
