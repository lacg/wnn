# IDS results — canonical source of truth (S&P-2027 paper track)

Generated 17/08/2026 from `file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro`
(read-only; no flow was created, cancelled, requeued or modified to produce this file).
Previous revision: 13/08/2026 (commit `4277029e`). Since then the SP100 cohort roughly
doubled: **~11 → ~22 of 100 completed per config**.

## Provenance — which tool produced which number

| Section | Producer | Note |
|---|---|---|
| 0. Paper / baseline comparison | `docs/ids_results_preserved/00_*.md` | hand-assembled from the rollup below + measured RF/XGB baselines; see the caveat on best-of-N |
| 1. SP100 live cohort 5-tables | `scripts/build_sp_5tables.py --prefix 'SP100-%'` | |
| 2. SP ablation 5-tables | `scripts/build_sp_5tables.py --prefix 'SP-%' --exclude 'SP100-%'` | |
| 3. Per-config leaderboards + per-dataset roll-up | `scripts/build_ids_leaderboard.py --prefix 'SP100-%' 'SP-%' [--rollup]` | |
| 4. best_fitness delta tables | `scripts/build_ids_leaderboard.py --fitness-delta` | |
| 5. XDS 5-tables (4 cohorts) | `scripts/build_xds_5tables.py --cohort {unsw-temporal,unsw-random,cicids,ciciot}` | LEGACY 2-way split |
| 6. Prior config-lock analysis (09/08) | `docs/ids_results_preserved/60_*.md` | hand-written, auto-appended |
| 7. 46M single-flow manual section | `docs/ids_results_preserved/70_*.md` | hand-written, auto-appended |
| 8. IDSX AC/CE interim (n=2, NOT publishable) | `docs/ids_results_preserved/80_*.md` | hand-written; carries CORRECTIONS to sections 0-7 |
| 9. MCSD-auto multiclass cohort (n=5, complete) | `docs/ids_results_preserved/90_*.md` (`ids-security` agent readout, 27/08/2026) | flows 5980-5994; AUTOMATIC per-task CE anchor; pre-registered macro-F1 + benign-FPR |
| 10. MCST support-tiered arm (n=5, complete) | `docs/ids_results_preserved/91_*.md` (direct DB read, 28/08/2026) | flows 6000-6004 (`-tiered2`); paired vs §9; `-tiered` 5995-5999 VOID |
| 11. MCST bits-band arm (tiered3, n=5, complete) | `docs/ids_results_preserved/92_*.md` (appended 28/08/2026) | flows 6005-6009; paired vs §10 |
| 12. IDSXD AC/CE cohort, cicids2017 cell (24/24) | `docs/ids_results_preserved/93_*.md` (appended 29/08/2026) | flows 5862-5885 |
| Header (this provenance + metric contract) | `docs/ids_results_preserved/00_doc_header.md` | hand-written; sorts before `00_paper_*` |

**No script produces this file end-to-end** (verified 24/08/2026 — nothing in `scripts/`
emits the header above). It is hand-assembled from the generators in the table; the
hand-written blocks are durable in `docs/ids_results_preserved/`, positioned by their
filename prefix (00-09 before the generated sections, 60-99 after). `build_oi_vs_old_report.py` now REFUSES `--out docs/ids_results.md`,
because it emits a different report and would replace every line of this one.

**Tooling gap, stated explicitly (unchanged):** `scripts/build_oi_vs_old_report.py` does NOT
cover the SP cohorts — its auto-detection keys on the `-FIXED-OLD-` rename marker and finds
only `WSWEEP-T20-96b-C35-250n100b` (OLD=63 / NEW=102). Sections 1-4 come from the SP-specific
generators; section 5 is the existing XDS tool verbatim.

## Metric contract (do not violate)

- Every number below is a **HELD-OUT** value read from
  `validation_summaries.threshold_metadata` at `validation_point='final'`.
  **No `iterations.best_f1` anywhere** — that is the during-search 5-fold CV metric and
  reporting it is the train-on-eval leak fixed 28/05.
- All 7 threshold modes are reported in order: `train_cal, fixed_05, platt, beta, empirical,
  empirical_cumulative, val_cal`. Empirical modes use `min_bin_size=200`.
- `best_genomes` is used ONLY for the neurons/bits header lines (it dedups by genome_hash and
  is incomplete); all metrics come from `validation_summaries`.

## ⚠️ Protocol split — these two groups are NOT apples-to-apples

| Group | Split | val_cal meaning |
|---|---|---|
| SP100-* and SP-* (sections 1-4) | `*_3way` (80/10/10) | Protocol v2: thresholds calibrated on the 10% VAL partition, 10% TEST is report-only |
| XDS-* (section 5) | `random` / `temporal` (80/20) | legacy: val_cal = oracle on the report set (known-optimistic, reviewer-attackable) |

Cross-group comparison of `val_cal` is not defensible. Within-group is.
