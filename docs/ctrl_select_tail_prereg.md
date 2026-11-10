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
  from the 35 report units and from the train/test seeds. Default 5 = today, bit-identical.
- `--select-tail p99`: tail term = the 99th percentile of the candidate's per-episode attitude error
  (the per-episode error the stable bound uses; a diverged episode counts as +inf) over ALL its val
  episodes. Combined exactly like `near_fail`: headline = lowest rank(fit) + rank(p99), average
  ranks, ties to the better fit.
- PRIMARY comparison: **rule P = 20 units + p99** vs **baseline B = 5 units, no tail term**.
  Secondary, descriptive only: 20 units alone; 5 units + p99.

## Test set
Every CTRL-48 block-4 and block-5 run on controller ABI 49 (tags `C-S2-d_wposx2_a49_*`, ≈33 runs),
re-ranked with `--recalc-headline` from their stage checkpoints. No GA run is added; no rule
parameter is tuned on these runs. Each pick is scored on the 35 report units × 100 episodes on the
fixed scorer.

## Pass criterion (P adopted only if ALL hold)
1. Total report fails over the test set: P < B.
2. Mean report err and mean report steady: P not worse than B by more than 0.05° each.
3. P picks a different genome than B on at least 3 runs (otherwise: no evidence either way, NOT adopted).
Report per-run picks and a paired table regardless of outcome.
