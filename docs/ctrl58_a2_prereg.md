# CTRL-58 A2: staged refinement fitness — prereg DRAFT (Luiz 19/11/2026 ~09:30 EST)

Status: DRAFT until the base is fixed (D+F vs Y+F, decided after the F-on-Y block). Nothing flies before that.

## Decisions taken (Luiz 19/11)
- **A2 goes before A1** (CTRL-57, patience 6/10).
- **Keep the defaults, refine from the start.** Arm A2 = base + `--refine-fitness` and nothing else:
  - trigger: ≥ 80% of the TRAIN pool at stable 100%, err ≤ 1.5°, steady ≤ 1.0°, pos ≤ 0.40 m for 2 consecutive
    generations, or MEMORY start. On this base the pool already meets those bounds, so the switch fires at
    about NEURONS gen 2 ("refine from the start"); the GRID stage is not a GA and is unaffected;
  - refinement gate: stable ≥ 0.99, err ≤ 1.5°, steady ≤ 1.0°, pos ≤ 0.40 m;
  - weights: steady 0.50, reward 0.25, p99 tail 0.15 (ABI 59 tail scorer row), pos 0.10;
  - patience: same counter and max per stage; at the switch the counter resets and steady is watched too, with a
    0.15° floor (5% of the distance above the floor counts as a gain); the trigger streak carries across stages;
  - `--refine-select` is NOT part of A2 (stage-select stays rule P).
- **Arms:** A2 vs the base (B = the base's existing F-block rows, argv-identical, ABI 59). If A2 is adopted, A1 and
  A1+A2 follow as their own block (A1 alone runs only if Luiz asks).
- **Seeds:** s1-s5 seed-major, ABI 59; extend to n=10 if promising.
- **Rule (a starting point, Luiz):** A2 − B, paired by seed, TEST over 35 CF2.1 units: steady better by > 0.03°,
  err not worse by > 0.03°, pos not worse by > 0.010 m, fails within the gate. Luiz judges the whole vector: a large
  gain on two metrics is not vetoed by a hair-thin miss on the third.
- **Expectation:** steady down (the fitness weights steady 0.50 and patience now sees steady gains), err flat or
  down, pos flat; runs somewhat longer (patience kept alive by steady gains).

## To fill at launch
- Base (D+F or Y+F) and its B rows; driver path; tags `..._a59_..._a2_full_sN`.
