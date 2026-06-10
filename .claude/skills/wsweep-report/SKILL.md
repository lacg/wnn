---
name: wsweep-report
description: Report a controller fitness-weight sweep (run_curriculum_ga.py --mode sweep, the W1-W4 + C1-C14 combos). Per combo shows the 4 weights, last-gen err/stable, during-search best err/stable, duration, and optionally a fresh-train held-out re-eval. Use when the user asks "how's the weight sweep", "wsweep status", "round 1 so far", "which weight combo is winning", or wants the per-combo controller sweep table.
---

# Controller weight-sweep report

Per-combo status for a controller fitness-weight cull-down sweep launched via
`tests/run_curriculum_ga.py --mode sweep` (the 18 combos W1-W4 + C1-C14, each
weighting err²/stable/jerk/mono). Parses the round log + reads the saved combo
winner pkls.

## Steps

1. Find the active sweep dir (newest `logs/controller/wsweep_*`):
   ```bash
   ls -dt logs/controller/wsweep_* | head -1
   ```
2. Fast report (log-based, ~1s — weights, last-gen err/stable, during-search
   best err/stable, duration):
   ```bash
   PYTHONPATH="$(pwd)/src/wnn" /Users/lacg/wnn-venv/bin/python \
     scripts/report_weight_sweep.py --dir logs/controller/wsweep_<TS>
   ```
3. With a fresh-train held-out column (slower; re-trains each completed winner
   on a fresh seed — set the split env to match the sweep's trainer):
   ```bash
   PYTHONPATH="$(pwd)/src/wnn" WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2 \
     /Users/lacg/wnn-venv/bin/python scripts/report_weight_sweep.py \
     --dir logs/controller/wsweep_<TS> --heldout
   ```

## Reading the table

- **lastgen lg_err / lg_stb** — the LAST GA generation's population err/stable.
  This is usually the best signal for "which weight drives stability" (e.g. a
  higher `stable` weight pushes the population's lg_stb up).
- **bst_err / bst_stb** — the during-search **best-FITNESS** genome (harmonic
  rank over err²+stable+jerk+mono), so NOT necessarily the most-stable genome.
  Optimistic (selected on the search folds).
- **dur** — wall time for that combo (`wall=...s` in the log).
- **ho_err / ho_stb** (only with `--heldout`) — a fresh-train re-eval at seed
  987654321, matched 5°. ⚠️ The sweep is Stage-A-only (no memory stage / no
  lamarckian), so the saved winner pkl has `cells=None`; this re-trains the
  architecture from scratch and is an architecture-generalization **lower
  bound**, NOT the GA winner's true held-out — it tends to collapse for the
  tiny sweep archs. For a true held-out, the sweep must persist the trained
  cells (a fix to `run_curriculum_ga.run_sweep` pkl-save).

## Cull-down workflow (3 rounds)

Round 1 = all 18 → rank by lg_stb (and the final SWEEP RESULT block) → cull
~half → Round 2 (`--combos <top~9>`) → Round 3 (`--combos <top3>`) → winner.
Then run the gain Run 1/Run 2 (`scripts/run_controller_gain_sweep.sh`) with the
winning weights.
