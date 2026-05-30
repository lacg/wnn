---
name: cohort-report
description: Regenerate docs/ids_results.md with the 5-table OI-v2 vs OLD baseline comparison + delta tables + best-genome mining. Also supports XDS cross-dataset cohorts (xds-temporal, xds-random, xds-cicids) via dispatch to build_xds_5tables.py. Use when the user asks for "update the cohort report", "refresh ids_results", "compare new vs old", "XDS report", or wants the latest paper-style 5-table breakdown.
---

# Cohort report

Regenerate the canonical cohort comparison report (`docs/ids_results.md`) and
display key highlights in chat.

## Steps

1. Run the cohort report generator. By default it auto-detects the active cohort:
   ```bash
   python3 scripts/build_oi_vs_old_report.py --out docs/ids_results.md
   ```
   For XDS cross-dataset cohorts (added 30/05/2026), use the alias:
   ```bash
   python3 scripts/build_oi_vs_old_report.py --cohort xds-temporal --out docs/ids_results.md
   # Aliases: xds-temporal | xds-random | xds-cicids — dispatch to build_xds_5tables.py
   ```
2. If the user names a specific dataset cohort, pass `--cohort PREFIX`. To see what's
   available:
   ```bash
   python3 scripts/build_oi_vs_old_report.py --list
   ```
3. After regenerating, show the user:
   - **Best-genomes mining** for both OLD and NEW cohorts (from the `### Best
     individual genomes — *` sections).
   - **NEW cohort summary line** (completed count, ETA, GA architecture means).
   - **Delta table for `best_fitness`** (GA Neurons phase — the headline genome type).
   - Note any architecture shifts between OLD and NEW (neurons / bits changed
     significantly).
4. Brief insights at the end:
   - F1/FPR direction (positive/negative for NEW vs OLD)
   - Std comparison (NEW should be tighter under OI)
   - Anything noteworthy in the per-genome Pareto extremes (sub-5% FPR with F1≥90%, etc.)

## What to look for

- **Architecture convergence**: if NEW cohort's GA Neurons converges to a
  meaningfully different (neurons, bits) than OLD, flag it — it's a paper-worthy
  finding (the order-independent training unlocks different architecture regimes).
- **fixed_05 threshold**: NEW cohort typically shows lower F1 *and* lower FPR
  here vs OLD. This is the OI distribution shift, NOT a regression. Calibrated
  thresholds (train_cal, platt, val_cal) show NEW > OLD across the board.
- **n threshold for paper update**: see `docs/paper_updates_pending.md` — the
  paper edits are gated on NEW cohort reaching n≥30.

## Output format

The full report goes to `docs/ids_results.md`. In chat, paste the most relevant
sections: NEW cohort summary, best-genomes tables for both cohorts, and the
`best_fitness` delta table. Avoid pasting all 5 cohort tables (too long); refer
the user to the file for those.
