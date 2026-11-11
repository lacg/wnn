# Plan: swap "test" and "val" in code to Luiz's convention (IDS + controller)

Status: DRAFT for Luiz (11/11/2026). Nothing implemented. Decisions marked **[D#]**.

## Target vocabulary (one meaning everywhere: code, DB, logs, dashboard, docs, ticks)

| word | meaning | used for |
|---|---|---|
| **train** | fit / evolve | memory writes, GA search folds |
| **test** | tuning partition, touched repeatedly | threshold calibration, stage-select, futility gates, early stop |
| **val** (validation) | REPORT partition, touched once | every published number |

## What the code says today (measured 11/11)

The repo is not even self-consistent. It has two opposite conventions:

- **`src/wnn/seeds.py` already uses Luiz's convention.** Its docstring says `test` = model selection and
  `val` = held-out final report.
- **Controller code does the opposite.**
  - `seeds.val` drives stage-select (`val_units.py`, `phased_ga.py:2119-2164`), which is tuning.
  - The report is a third set (`--report-seeds`).
  - `seeds.test` is only used in the disjointness check.
- **IDS does the opposite too (Protocol v2, worker ABI 3+).**
  - The HF `_3way` split `val` (10%) calibrates thresholds. Mode `val_cal` and JSON key `margin_val_cal` are
    stored in `validation_summaries.threshold_metadata`.
  - The HF split `test` (10%) is report-only.
  - Baselines (`src/wnn/ids/baselines.py`) and `X_val`/`X_test` in `src/wnn/ids/*.py` and
    `IdsEvaluatorFactory.py` follow the same naming.
- **`validation_summaries` already means "report checkpoints"**, which is Luiz-consistent. Keep the table name.

Surface (rough grep counts):
- Mode string `val_cal` appears in 2 Rust files (`multiclass_metrics.rs`, `pyapi/ids_cache_api.rs`) and about 25 scripts.
- `X_val`/`X_test` appear in 7 `src/wnn/ids` / evaluator files.
- Controller: `val_units.py`, `phased_ga.py` (the `--val-units`, `--val-seed` and `--test-seed` flags), and the
  stage-select/report printers.
- Dashboard: a handful of Svelte labels.
- Docs: CLAUDE.md (Protocol v2 text, Rule 7 mode list), docs/ids_results.md, the tick prompt, about 10 memory files.

## [D1] Paper vocabulary: decide before coding

The ML/IDS literature (and S&P reviewers) use the opposite: validation = tuning, test = report. Options:
- (a) Code follows Luiz. The paper keeps the standard words, with one mapping sentence in the protocol section.
  This is my recommendation: the code is for us, the paper is for reviewers.
- (b) Code and paper both follow Luiz. The paper then needs an explicit definition box, or reviewers will read
  "tuned on test" as a leak.
- (c) Neutral names in code (`tune` / `report`). This avoids the ambiguity for everyone, but is not what was asked.

## Rule: pure rename, zero behaviour change

The same examples, the same seeds and the same thresholds land in the same roles. This is verified by
before/after bit-identity (Step 5). In particular, the controller's stage-select units must remain the exact
seeds flown today. Pre-registrations (`docs/ctrl_select_tail_prereg.md`, IDS56U prereg) and every existing
checkpoint must re-rank identically.

## Steps

### 1. IDS loader boundary (Python), one mapping point
- In `src/wnn/ids/dataset.py`, map the HF split names exactly once, at load:
  - HF `val` → code `test` (tuning)
  - HF `test` → code `val` (report)
- HF config names (`random_3way`) and HF split names cannot change. They exist only in this mapping, with a comment.
- Rename `X_val`↔`X_test` (and the `y_` / encoded / streaming variants) in:
  - `dataset.py`, `encoded_array.py`, `bitwise.py`, `train.py`, `baselines.py`
  - `ids_evaluator.py`, `IdsEvaluatorFactory.py`
- 2-way legacy datasets have only a report partition, so they get `val`. The old "val_cal = oracle on the report
  set" becomes `test_cal` = oracle on `val`. Keep the existing "known-optimistic" warning text.

### 2. Threshold mode names (Rust + Python + dashboard), worker ABI bump
- `val_cal` → `test_cal` and `margin_val_cal` → `margin_test_cal`.
  - Rust: `multiclass_metrics.rs`, `pyapi/ids_cache_api.rs`, plus whatever emits `threshold_metadata` keys
    (grep at implementation time).
  - Python: the worker metadata writer, `params.py` if any param names the mode.
  - Svelte: mode labels and the Rule 7 table order.
- Any per-partition metric keys that the worker or probes write as `val`/`test` swap the same way.
- Worker wheel → ABI bump. **Swap at worker idle only** (`scripts/worker_swap.py`), after
  `/pre-deploy-wheel-review`.

### 3. DB history: [D2] migrate or alias
- (a) **Recommended: one-time migration.** Back up `wnn.db` first. With the worker stopped, in the same maintenance
  window as the swap, rewrite the `threshold_metadata` JSON keys (`val_cal`→`test_cal`, `margin_val_cal`→
  `margin_test_cal`, and the per-partition keys) via `json_set`/`json_remove`.
  - This gives one vocabulary in every readout (Rule 5 queries stay simple).
  - Check whether `cache_key` hashes the mode names. If it does, recompute the keys or leave old ones (they only
    gate cache reuse).
- (b) Alias in readers: `test_cal` falls back to `val_cal` for old rows. That is two vocabularies forever, so not
  recommended.
- Also check the seed registry table (`train_seed`/`test_seed`/`val_seed`); see Step 4.

### 4. Controller (no wheel; Python only unless Rust prints the labels)
- **Seeds.** Swap the derivation so the numbers stay put:
  - In `seeds.derive_seeds`, the value that today lands in `val` (used for stage-select) must land in `test`.
  - The old `test` value lands in `val`. It is used only for disjointness today, and the report stays
    `--report-seeds`.
  - Update `SeedSet` docstrings: they are already right, only the call sites move.
  - Registry table: swap the `test_seed`/`val_seed` columns for historic rows in the Step-3 migration, or record a
    `convention_version`. **[D3]** — recommend the swap.
- **Flags.**
  - `--val-units` → `--test-units`
  - `--val-seed` ↔ `--test-seed`
  - Keep the old flag names one release as hidden aliases that warn loudly, so queued scripts don't die mid-block.
    **[D4]** Alternatively, a hard rename after block 6 ends.
- **Files.** `val_units.py` → `test_units.py`. Update the `phased_ga.py` stage-select docstrings and printers
  (STAGE-SELECT/VAL → TEST; REPORT → VALIDATION (report)).
- **Checkpoints.** `--recalc-headline` reads old checkpoint JSON. Read old keys through one alias function
  (files can't be bulk-migrated safely), and write only new keys.
- **Tick/report scripts** that parse log lines (`stable_fail=` etc.) and the rescore scripts: update their label
  regexes.

### 5. Verification (`quality-assurance` agent)
- **IDS.** Tiny flow, same seed, before and after (worker in a scratch DB). Every metric must be identical, with
  `test_cal` == old `val_cal` and `val` partition == old `test` partition. Run the migration on a DB copy, then
  `build_sp_5tables.py` before/after must produce identical numbers.
- **Controller.**
  - Run `--recalc-headline` on 2-3 a49 checkpoints: same stage selected, same report triple.
  - A tiny `phased_ga` run before/after: bit-identical gen lines.
  - The disjointness assert still fires on a planted collision.
- `cargo test -p ram_controller --lib --no-default-features` (parity), plus the worker crate tests.

### 6. Docs and memory (same commit series)
- **CLAUDE.md.**
  - The Protocol v2 paragraph becomes "calibrate on test, report on val".
  - The Rule 7 mode list `val_cal` becomes `test_cal`.
  - Add one "Vocabulary" box with the table above.
- **docs/ids_results.md.** New sections use the new words. Historic sections get a one-line banner:
  "pre-11/2026 naming: VAL = tuning, TEST = report".
- **Tick prompt** (`docs/controller_status_tick_prompt.md`), **memory** (`feedback_luiz_test_val_naming.md`
  becomes "code now matches"), and the readout skills (`cohort-report`, `cohort-status`).

## Order and timing (nothing runs mid-experiment)
1. Luiz answers D1-D4.
2. Implement on a branch: Python rename (IDS + controller) + Rust mode rename + migration script.
   Run all of Step 5 against a DB copy.
3. **Controller part lands after block 6 ends** (s3 finishes in about 2 days). Block 6 passes only
   `--report-seeds`, so it's not at risk either way, but the readout scripts change.
4. **IDS part lands at a worker-idle window.** One maintenance window: stop admitting → back up the DB → migrate →
   `worker_swap.py` with the new ABI → requeue (POST /resume, never PATCH) → verify the first flow writes
   `test_cal`. Natural window: between the IDS56U arm (7543-7545) and the carry-over, or after the carry-over.
   Do not land it mid-IDS56U, so the arm's metadata stays in one vocabulary.
5. Commit and push per step. Write a memory entry with the cut-over timestamp.

## Effort
About 1 day of implementation and about half a day of verification, mostly Python and scripts. The Rust change is
small (mode strings), but it forces a worker swap.
Risk is concentrated in two places: the DB migration (mitigated by backup + copy dry-run) and the controller seed
swap (mitigated by the bit-identity check on stage-select units).
