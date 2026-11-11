# Plan: one train / validation / test convention everywhere (the literature's)

Status: DRAFT for Luiz (rewritten 11/11/2026 after the literature check). Nothing implemented.
Supersedes the first draft, which swapped the code to "test = tuning, val = report".

## Decision (Luiz 11/11)

Use the literature convention coherently in code, DB, logs, dashboard, docs, ticks and the paper.
Evidence and quotes: `docs/terminology_train_val_test.md`.

| word | role |
|---|---|
| train | fit |
| validation (`val`) | model selection and tuning (touched repeatedly) |
| test | final report (touched once) |

## What this means for the code (measured 11/11)

Most of the code already follows the literature, so the work is far smaller than the first draft:

- **IDS: no change.** `val_cal` / `margin_val_cal` calibrate on the val partition; the test partition is
  report-only. No worker ABI bump, no wheel swap, no DB migration of `threshold_metadata`.
- **Controller behaviour: no change.** Stage-select ranks on `seeds.val` / `--val-units`; published numbers come
  from `--report-seeds`.
- **The one inverted place: `src/wnn/seeds.py`.** The docstring and `SeedSet` comments say test = "model selection /
  early-stopping" and val = "held-out final report". The controller uses `seeds.val` for selection, so the comments
  contradict the usage as well as the literature.

## Steps

### 1. `src/wnn/seeds.py` — fix the documented roles (comments only, no value moves)
- Docstring and `SeedSet` field comments become: val = model selection / stage-select; test = final report.
- The derived numbers do NOT move: `seeds.val` keeps its value, so every stage-select unit stays the seed flown
  today and every pre-registration and checkpoint re-ranks identically.
- Consumers found: `control/val_units.py:73,89`, `control/phased_ga.py:2088,4458`,
  `ram/experiments/flow_runner.py:240`. All pass the values through; none needs a logic change.

### 2. Controller: make "report" and "test" the same word — [D-A]
Today the controller has three names: `seeds.test` (used only in the disjointness check), `seeds.val`
(stage-select) and `--report-seeds` (published). Under the convention the report units ARE the test set.
- (a) **Recommended: keep the flag names, fix the words.** `--report-seeds` stays (scripts and block 6 pass it).
  Printers, docstrings and docs say "test (report) units". `seeds.test` is documented as "reserved test seed, kept
  disjoint; the published test units are `--report-seeds`". No flag churn, no risk to running chains.
- (b) Rename `--report-seeds` → `--test-seeds` with the old name as a warning alias. One word everywhere, but it
  touches every chain script and the tick prompt.

### 3. DB — [D-B] (was D2/D3)
- `threshold_metadata` keys: already correct, nothing to migrate.
- `seed_runs.test_seed` / `val_seed` columns: values and names stay; only the meaning in the docs is corrected.
- `test_summaries` table holds checkpoint reports, including test-partition metrics. Options:
  (a) **recommended: keep the name**, document it as "checkpoint summaries" (a rename touches the Rust dashboard,
  the worker and about 60 scripts for no behaviour gain); (b) rename to `checkpoint_summaries` with a view under
  the old name.

### 4. Words in docs, ticks, memory, dashboard
- CLAUDE.md: add the vocabulary table and a pointer to `docs/terminology_train_val_test.md`. The Protocol v2 text
  is already right.
- Tick prompt and readouts: "VAL" = tuning, "TEST"/"REPORT" = published. Remove the "Luiz's labels" relabelling
  used on 11/11 (handoff file, memory `feedback_luiz_test_val_naming.md` — rewritten).
- Controller tables: the REPORT column header becomes "TEST (report)".
- Dashboard: check labels only; no schema change.
- `docs/ids_results.md`: already in this convention; no banner needed.

### 5. Verification
- `seeds.py` change is comments only: `git diff` shows no code lines; a tiny `phased_ga` run before/after prints
  identical seeds and gen lines.
- If D-A(b) is chosen: old and new flag produce the same args namespace; a planted val/test collision still trips
  the disjointness assert.

## Timing and effort
- Steps 1 and 4: about 1-2 hours, safe to land any time (no wheel, no worker, no running chain reads them).
- Step 2(b) or 3(b), if chosen: about half a day, lands after block 6.

## Decisions for Luiz
- **D-A:** keep `--report-seeds` and fix the words (recommended), or rename it to `--test-seeds`.
- **D-B:** keep the `test_summaries` table name (recommended), or rename it.
The old D2 (migrate JSON keys) and D3 (swap seed columns) are no longer needed; D4 (flag aliases) only applies if
D-A(b) is chosen.
