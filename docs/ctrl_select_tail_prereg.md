# Pre-registration — tail-aware headline selection (CTRL-48, Luiz 10/11/2026)

Written 10/11/2026 BEFORE any ABI-49 checkpoint exists. Nothing below may be changed after the
first block-4/5 run lands; any change is a new pre-registration.

## Why
The headline genome is picked by stage-select: top-3 of every stage (≈9 candidates), each flown on
5 val units × 100 episodes, ranked by fitness. Report fails are ~1/1000 and cluster on a few hard
units (motor-wear draws 112/113/118/121/123, docs/ctrl_dres_failures.md), so 5 val units rarely
contain one and the pick is blind to the tail. Offline (docs/ctrl_dres_fix_offline_tests.md §B,
chosen AFTER seeing the data, hence this pre-registration): val p99 per-episode error had rank
correlation +0.42 with report fails; fitness ≈ 0.

## The rule under test (fixed)
- `--val-units 20`: stage-select flies every candidate on 20 val units (× 100 episodes), disjoint
  from the 35 test units and from the train/test seeds. Default 5 = today, bit-identical.
- `--select-tail p99`: tail term = the 99th percentile of the candidate's per-episode attitude error
  (the per-episode error the stable bound uses; a diverged episode counts as +inf) over ALL its val
  episodes. Combined exactly like `near_fail`: headline = lowest rank(fit) + rank(p99), average
  ranks, ties to the better fit.
- PRIMARY comparison: **rule P = 20 units + p99** vs **baseline B = 5 units, no tail term**.
  Secondary, descriptive only: 20 units alone; 5 units + p99.

## Test set
Every CTRL-48 block-4 and block-5 run on controller ABI 49 (tags `C-S2-d_wposx2_a49_*`, ≈33 runs),
re-ranked with `--recalc-headline` from their stage checkpoints. No GA run is added; no rule
parameter is tuned on these runs. Each pick is scored on the 35 test units × 100 episodes on the
fixed scorer.

## Pass criterion (P adopted only if ALL hold)
1. Total report fails over the test set: P < B.
2. Mean report err and mean report steady: P not worse than B by more than 0.05° each.
3. P picks a different genome than B on at least 3 runs (otherwise: no evidence either way, NOT adopted).
Report per-run picks and a paired table regardless of outcome.

## Amendment 1 (12/11/2026, Luiz option (a); written BEFORE any P re-pick was run)
State when written: only the reproduction check exists (B on `rf_c200_dres_full_s1`, re-picked MEMORY#1 = the
original headline, bit-identical val rows and test row). No P, S20 or S5p re-pick has been run, so no result can have
shaped this change.

- **Test set:** 30 runs, not 33. The 3 `rf_c200_dres_refit100_full_s{1,2,3}` runs are dropped: phased_ga refuses
  `--recalc-headline` with `--threshold-refit-from-student` (`_refuse_refit_without_grid`: the student-refit pool is
  collected after the grid and is not saved in any checkpoint; refusal log in `experiments/ctrl_select_tail/`).
  Run list: `experiments/ctrl_select_tail/runs_blocks45_a49_amended.txt`.
- Nothing else changes: rules P and B, the secondaries, the scoring (35 test units x 100 episodes, fixed scorer) and
  all three pass criteria stand as written above, evaluated over the 30 runs.

## Amendment 2 (13/11/2026, Luiz; written BEFORE any S5p or S20 re-pick was run)
State when written: the primary test is complete (P ADOPTED 3/3, `docs/ctrl_select_tail_result.md`, 7125992c). No S5p or
S20 re-pick exists (`experiments/ctrl_select_tail/` holds only `.B.log` / `.P.log`; runs.log has no S5p/S20 RECALC line).

**Purpose.** S5p (5 val units + p99 tail term) costs B's selection time (~5.6 min vs P's ~15 min per run). This amendment
turns S5p from a descriptive secondary into a pre-registered **substitute test**: can the cheaper rule replace P?

- **S20 is DROPPED** (Luiz 13/11): it costs as much as P and cannot save time.
- **Test set:** the 30 runs of Amendment 1 (`runs_blocks45_a49_amended.txt`) **plus all 15 block-8 runs** (ABI 50, P inline,
  control + 4 arms x s1-s3) = 45 runs. Runs after block 8, on the block-8 checkpoints as they stand.
- **Comparison:** S5p vs P on the same runs, every pick scored on the 35 test units x 100 episodes (fixed scorer). For the
  30 old runs the P picks already on file are reused; for block 8 the P pick is each run's own published headline.
- **S5p replaces P only if BOTH hold over the 45 runs:**
  1. total test fails S5p <= P;
  2. mean test err and mean test steady, S5p not worse than P by more than 0.05° each.
  Otherwise P stays. No third criterion (a "picks differ" count is not required: identical picks mean S5p is a free
  substitute).
- **Also run, descriptive only:** B re-picks of the 15 block-8 runs = an out-of-sample P-vs-B replication on new configs.
  It cannot un-adopt P; if it reverses (P more fails than B on block 8), report it to Luiz.
- Report per-run picks and a paired S5p-vs-P table regardless of outcome. Nothing above may change after the first S5p
  re-pick lands; any change is a new pre-registration.
