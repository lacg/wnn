---
name: wsweep-report
description: Report a controller fitness-weight sweep run on the full phased_ga pipeline (grid→GA-neurons→GA-memory, the W1-W4 + C1-C14 combos via scripts/run_weight_sweep_phased.sh). Per combo shows the 4 weights, latest-gen err/stable, the per-stage HELD-OUT (NEURONS + MEMORY, the honest numbers), duration, and a ranking by MEMORY held-out. Use when the user asks "how's the weight sweep", "wsweep status", "combo round 1", "which weight combo is winning", or wants the per-combo controller sweep table.
---

# Controller weight-sweep report (phased_ga pipeline)

Per-combo status for a controller fitness-weight cull-down sweep run via
`scripts/run_weight_sweep_phased.sh` — each of the 18 combos (W1-W4 + C1-C14,
weighting err²/stable/jerk/mono) is a full **phased_ga** run
(grid → GA-neurons → GA-memory) with all four gated options on: splitting
(`WNN_STATE_SPLIT=1`), lamarckian write-back (`--lamarckian`), cell persistence
(`--save-winner`), and per-stage held-out (`--report-seed`).

## Steps

1. Find the active sweep dir (newest `logs/controller/wsweep_phased_*`):
   ```bash
   ls -dt logs/controller/wsweep_phased_* | head -1
   ```
2. Run the report (fast, ~1s — pure log parse, no re-eval needed because the
   held-out is already in each combo's phased_ga log):
   ```bash
   PYTHONPATH="$(pwd)/src/wnn" /Users/lacg/wnn-venv/bin/python \
     scripts/report_weight_sweep.py --dir logs/controller/wsweep_phased_<TS>
   ```

## Reading the table

- **err/stb/jrk/mno** — the combo's 4 fitness weights (sum to 1.0).
- **stage / lastgen / lg_err / lg_stb** — current stage + the latest GA
  generation's during-search err/stable (optimistic; the GA's selection metric).
- **N_err / N_stb** — NEURONS-stage **held-out** (fresh report-seed 99990001,
  matched 5°). Real generalization number after the architecture search.
- **M_err / M_stb** — MEMORY-stage **held-out** = the **final honest result**
  per combo (after the cell-value GA, the high-leverage stage).
- **dur** — wall time (`Total wall time` from the log; `run` while in progress).
- **Ranking** — printed below the table, by MEMORY held-out stable (then err).

## Cull-down workflow (3 rounds)

- **Round 1** = all 18 (`scripts/run_weight_sweep_phased.sh` with no args).
- Rank by **MEMORY held-out stable** → cull ~half.
- **Round 2** = top ~9 (`run_weight_sweep_phased.sh W2 C2 C11 ...` — the script
  takes combo names as a subset filter), ideally heavier config / fresh seed.
- **Round 3** = top 3 → winner.
- Then run gain Run 1/Run 2 (`scripts/run_controller_gain_sweep.sh`) with the
  winning weights (update its `--fit-weight-*` to the winner first).

## Note

This SUPERSEDES the old `run_curriculum_ga.py --mode sweep` weight sweep, which
was NEURONS-only (no memory stage, no lamarckian, cells=None → no real held-out).
The phased_ga sweep is the representative pipeline (same as the gain runs).
