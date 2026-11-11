# rf_c200_dres rare failures: offline tests for the five fix ideas (09/11/2026)

These are offline tests only. No GA runs, no repo code changes, nothing queued. They decide how to configure
GA arms for the fix ideas in `docs/ctrl_dres_failures.md`:

- (1) widen the outer ladder;
- (2) student-state refit;
- (3) tail-aware selection;
- (4) high-speed training states;
- (5) the calibration tilt, from `docs/ctrl_outerq_august_audit.md`.

All runs used `nice -n 10`, with at most 2 processes alongside the live oq02 s2 run and the IDS worker.

## Setup (measured)

- **Scorer.** The fixed scorer, from the controller ABI-47 wheel at `/tmp/ctrl_fix_site`.
- **Python source.**
  - The main checkout (`70004ad4`, which includes `8542f4c1` and `e238f36f`) expects ABI 45 and refuses the
    ABI-47 wheel.
  - So I used a scratch tree: main HEAD plus the ABI-46/47 Python diff (`0ed7c4ab..7303ea0a`, `src/wnn` only),
    at `/tmp/dres_fix/src`.
- **Checks on that tree:**
  - Its variant-(i) ladder is byte-identical to the as-flown ladder produced by the ctrl-scorer-fix worktree
    (checked on s2).
  - Refitting from the raw calibration pool reproduces variant (i) on all 3 seeds.
- **Genomes.** The headline genomes of `C-S2-d_wposx2_a45_rf_c200_dres_full_s{1,2,3}` (NEURONS#2, MEMORY#1,
  MEMORY#2), loaded from the ctrl48 stage checkpoints.
- **Replay.** A CPU step mirror of `cpu_score.rs::rollout_one`, the same loop as `/tmp/dres_diag/stats.py`.
  Against the GPU per-episode rows on all 10,500 report episodes:
  - 0 stable-flag mismatches;
  - max |CPU-GPU| per-episode error 0.30° (float32).
- **Scripts.** Scratch scripts are in `/tmp/dres_fix/` and are not committed.

The 14 high-speed episodes (peak lateral speed ≥0.4 m/s) and the 4 fails reproduce exactly:
- s1: 4 high-speed episodes, 0 fails.
- s2: 9 high-speed episodes, 3 fails (112/86, 123/32, 123/70).
- s3: 1 high-speed episode, 1 fail (113/35).

All numbers below are MEASURED unless marked *inferred*.

---

## A) Ladder coverage (ideas 1, 2, 5)

Each ladder was refit at the run's TRAIN seed with the recipe's calibration settings
(`--calib-episodes 200`, quantile ladder, flown 5° tilt). Only the variant flags changed.

Variants:

| variant | flags / method |
|---|---|
| (i) | default |
| (ii) | `--threshold-outer-quantile 0.02` |
| (iii) | `--threshold-outer-quantile 0.005` |
| (iv) | `--threshold-calib-tilt 30` |
| (v) | `--threshold-calib-tilt 30` + q 0.02 |
| (vi) | student-state refit |
| (vii) | student-state refit + q 0.02 |
| (viii)-(xv) | extra combinations, added to find the edge: tilt 15/30 x q 0.002-0.02, and `--calib-ladder centred` |

**How (vi)/(vii) were built.** phased_ga's own refit path cannot run on this recipe (see A.4). So I emulated
what it would do:
- Take the teacher calibration pool: 200 episodes x 2000 steps = 400k samples per feature.
- Add the headline student's own feature samples on the train seed's scorer draw.
- Two sizes:
  - 10 episodes, phased_ga's `--threshold-refit-episodes` default;
  - 100 episodes, the same order as the pool.

### A.1 Clipped fraction: samples below the first or above the last cut

Pooled over the 3 genomes, each on its own train-seed ladders. Samples are taken every 10 ms.
- HS = the 14 high-speed episodes (2,800 samples).
- FAIL = the 4 failing episodes (800 samples).
- ALL = 10,500 report episodes (2.1 M samples).
- "all25" = the mean over all 25 features.

```
set HS (14 eps)          roll  pitch   v_x   v_y   e_x   e_y  cr_x  cr_y | all25
(i)    default           37.4  15.7  20.4  58.4  48.4  21.5  49.6  23.7 | 27.9
(ii)   q0.02             16.2   6.4  12.6  47.9  12.9   3.1  22.6   2.6 |  9.9
(iii)  q0.005            10.4   2.8  10.5  39.7   2.9   0.0  13.1   0.8 |  5.3
(iv)   tilt30            22.8   8.8  13.6  52.5  46.6  19.3  47.2  20.8 | 23.6
(v)    tilt30+q0.02       8.2   2.1   9.3  34.8   5.8   2.0  18.5   2.2 |  6.2
(vi)   refit 10ep        38.3  15.8  20.9  58.9  48.7  20.2  49.5  23.6 | 27.9
(vi)   refit 100ep       41.8  18.6  22.2  62.6  47.5  22.1  48.6  24.9 | 28.7
(vii)  refit100+q0.02    16.5   6.3  13.1  49.9  10.2   3.9  23.3   2.1 |  9.7
(viii) tilt30+q0.005      0.0   0.1   4.5   7.5   0.0   0.0  10.3   0.2 |  1.7
(xi)   centred+q0.02     15.5   6.2  12.7  48.5  12.7   2.5  24.5   3.0 |  9.8
(xiv)  tilt30+q0.002      0.0   0.0   0.5   1.1   0.0   0.0   4.2   0.1 |  0.8

set FAIL (4 eps)
(i)    default           33.1  24.8  44.1  43.2  34.5  29.4  33.2  30.1 | 28.5
(ii)   q0.02             15.1  12.8  32.4  33.2   6.0   3.4  14.1   6.8 | 10.5
(iv)   tilt30            23.0  16.9  34.0  36.6  32.8  24.0  29.5  26.5 | 24.3
(v)    tilt30+q0.02       6.5   5.1  25.9  25.5   0.0   2.8  11.0   6.4 |  6.3
(vi)   refit 100ep       36.8  28.2  44.8  54.6  33.6  30.1  28.9  26.9 | 28.4
(vii)  refit100+q0.02    15.4  12.0  33.2  34.8   5.0   3.9  13.4   4.4 | 10.0
(xi)   centred+q0.02     14.4  12.5  32.5  33.1   6.4   2.4  15.2   8.0 | 10.5
(xiv)  tilt30+q0.002      0.0   0.0   1.6   1.5   0.0   0.0   2.8   0.4 |  1.4

set ALL (10,500 eps)
(i)    default           13.7  12.3   4.8   5.5  29.5  25.9  20.8  17.5 | 16.7
(ii)   q0.02              3.5   3.4   0.3   0.7   6.2   3.9   3.7   2.9 |  3.2
(iv)   tilt30             6.8   6.3   0.6   1.4  27.6  24.5  18.4  15.9 | 13.5
(v)    tilt30+q0.02       0.3   0.2   0.0   0.2   2.6   1.6   2.9   2.4 |  1.6
(vi)   refit 100ep       16.9  14.2   9.3   8.7  27.1  25.9  27.7  20.7 | 17.3
(xi)   centred+q0.02      3.5   3.4   0.3   0.7   6.1   3.4   3.6   2.8 |  3.2
(xiv)  tilt30+q0.002      0.0   0.0   0.0   0.0   0.0   0.0   0.4   0.2 |  0.2
```

Under the DEFAULT ladder, e_x/e_y are clipped in 26-30% of ALL report samples. The ±0.3 m ladder sits inside
the ±0.5 m start offset. This holds on every episode, not just the tail.

### A.2 Clipping is the wrong criterion; aliasing is the right one (measured)

With 8 cut points, a quantile ladder only moves its OUTERMOST pair when q shrinks. The interior cuts barely move.

Example: v_y on s2.

| variant | cut points (m/s) |
|---|---|
| default | -0.118 ... +0.118 |
| q0.02 | -0.193 -0.096 ... +0.103 +0.213 |
| tilt30+q0.002 | -0.438 -0.141 ... +0.138 +0.495 |

So "0 clipped" in (xiv) is bought by putting the stall speed (0.45-0.5 m/s) back in the SAME bin as 0.14 m/s.
That bin is (0.138, 0.495). The genome sees nothing new.

The measure that matters is whether a FAR state gets a different code from a moderate one.

**Metric.** Take the samples that are truly far:
- |v| ≥ 0.3 m/s;
- |attitude error vs level| ≥ 3°;
- |e| ≥ 0.4 m;
- |cresid| ≥ 0.3.

Report the % of those samples whose per-feature thermometer code equals the code of the moderate reference
(same sign):
- 0.15 m/s;
- 1°;
- 0.3 m;
- 0.1.

Lower is better, and 100% = fully aliased.

```
stall window 0.25-1.0 s of the 4 fails      roll  pitch   v_x   v_y   e_x   e_y  cr_x  cr_y
(far-sample n:                                 20    24   136   149    22    12    72    30)
(i)    default                               100   100   100   100    27     0     0     0
(ii)   q0.02                                   0     0     0     0    96    83     6     0
(iii)  q0.005                                 30    25     0     0   100   100    68    97
(iv)   tilt30                                  0     0     0     0    27   100     0     0
(v)    tilt30+q0.02                           50     0     0     1   100   100    38     7
(vi)   refit 100ep                           100   100   100   100    27   100     0     0
(vii)  refit100+q0.02                          0     4     0     0   100    50     0     0
(viii) tilt30+q0.005                           0     0    24    20   100   100    83   100
(xi)   centred+q0.02                           0     0     0     0    86   100     0     0
(xiv)  tilt30+q0.002                           0     0    90    92   100   100     0     0

all report samples (far n: roll 49k, v_y 2.8k, e_x 197k, cr_x 62k)
(i)    default                               100   100   100   100    67    19    54    61
(ii)   q0.02                                   0     0     0     0    34    49     1     1
(iii)  q0.005                                 61    62     0     0    84    65    51    52
(iv)   tilt30                                  0     0     0    28    50    49    21    53
(v)    tilt30+q0.02                           47    28    11    15    73    55    15     3
(vii)  refit100+q0.02                          0     1     0     0    41    58     0     0
(viii) tilt30+q0.005                           0     0    73    56    99    68    71    67
(xi)   centred+q0.02                           0     0     0     0    35    67     0     0
(xiv)  tilt30+q0.002                           0     0    97    69   100    68    57    60
```

**Under the default ladder, a 3°+ attitude error and a 0.3+ m/s lateral speed are indistinguishable from 1° and
0.15 m/s on 100% of samples.** This is the investigation's pinning, confirmed population-wide.

Only three variants remove that aliasing on att-err, v_xy and cresid, both in the stall and population-wide:
- (ii) q0.02;
- (xi) centred+q0.02;
- (vii) refit+q0.02.

Tilt-30 alone (iv) fixes the stall window but leaves 28% v_y and 53% cresid_y aliasing population-wide. Every
WIDER variant re-aliases: q0.005, tilt30+q, and tilt15+q0.005 all score 100% on roll/pitch.

e_x/e_y stay partly aliased in every variant. The far threshold of 0.4 m is close to the start bound of 0.5 m,
so this is a weak signal either way.

### A.3 Resolution cost near zero, where steady error is measured

Mean over s1-s3. "zero bin" = the width of the bin that contains 0.

```
variant              att-err zero bin   cuts within ±1°   v zero bin   cuts within ±0.1 m/s   att-err outer cut
(i)    default        0.117°             8/8               0.025 m/s    6/8                    ±0.89°
(ii)   q0.02          0.144° (+23%)      6/8               0.031 (+25%) 5/8                    ±2.5°
(iv)   tilt30         0.165° (+41%)      6/8               0.036 (+44%) 4.8/8                  ±1.6°
(v)    tilt30+q0.02   0.206° (+76%)      4.8/8             0.045 (+82%) 4/8                    ±5.0°
(vii)  refit+q0.02    0.140°             6/8               0.029        5.7/8                  ±2.4°
(viii) tilt30+q0.005  0.213° (+82%)      4/8               0.047 (+90%) 4/8                    ±14.5°
(xi)   centred+q0.02  0.129° (+10%)      6/8               0.027 (+9%)  6/8                    ±2.5°
(xiv)  tilt30+q0.002  0.214° (+83%)      4/8               0.047 (+90%) 4/8                    ±20°
```

For context, dres steady error is about 1.0°. Variant (xi) keeps near-default resolution: its 4 inner cuts are
placed by the centred ladder at ±0.065/±0.19° (s2). It de-aliases exactly as (ii) does.

### A.4 The student refit (idea 2) is dead on this recipe (measured and code-read)

- **The collector refuses.** phased_ga's collector raises NotImplementedError on the ABI-32 plant: it hits
  `refuse_python_abi32_plant` because `--translation`, `imu_specific_force` and `--rotor-drag 0.78` are set.
  `_refit_thresholds_from_student` catches that exception and prints "keeping the teacher fit". So
  `--threshold-refit-from-student` is a **silent no-op** for rf_c200_dres.
- **The refit would not reach the trained stages anyway.** Even if it ran, the refit ladder only reaches
  `stage0_grid(thresholds_override=...)`. NEURONS (`_run_arch_phase`), MEMORY and `_test_thresholds` all refit
  from the teacher pool without `extra_samples`. This is the same 2-of-5 plumbing defect class that
  `8542f4c1` fixed for the outer quantile. It would also leave GRID's held-out scored on a ladder its cells were
  not written under. Code reading only; nothing changed.
- **Offline, the refit adds nothing.**
  - On its train-seed draw, the student never exceeds 0.4 m/s (0/100 episodes per seed). So its samples
    NARROW the ladder: (vi) is worse than (i), with v_y in HS 62.6% vs 58.4% clipped.
  - (vii) is equivalent to (ii) within noise.

### A.5 Recommendation (A)

1. **(ii) `--threshold-outer-quantile 0.02`** is the minimal change that removes the att-err/v/cresid aliasing.
   It is already flying as `rf_c200_dres_oq02` (s2 live now). Keep that arm; it IS idea 1's test.
2. **(xi) `--calib-ladder centred --threshold-outer-quantile 0.02`** de-aliases the same features as (ii) at
   +10% near-zero bin width instead of +23-25%. It is the one new ladder arm worth flying.
   - Prior evidence: `rf_c200_centred` on the non-dres base was 99.8/1.87/1.07 (22 fails) vs rf_c200
     99.4/1.92/1.14 (61 fails). That arm used the default q, so it was still aliased.
3. **Not worth an arm:**
   - q0.005 and tilt-30 combined with any q: they re-alias, and tilt-30 costs +40-90% near-zero resolution.
   - Student refit: no-op on this recipe, no offline gain.
   - Tilt-30 alone answers the August question and is offline-dominated by (ii)/(xi) here. Fly it only if
     Luiz wants the August tilt question closed for its own sake.

---

## B) Tail-aware selection (idea 3)

**Candidates.** The 9 stage-select candidates per seed are the top-3 of GRID, NEURONS and MEMORY, taken from the
stage checkpoints.
- GRID#2 was arch-only on every seed, so it was trained exactly as `_holdout_report`'s arch-only branch does:
  K-fold accumulate on the train seed, report episodes, write-back.

**Scoring.**
- Each candidate was scored on the run's own 5 VAL seeds (`seeds.val+0..4`, 500 episodes, fixed scorer), with
  per-episode GPU rows plus a CPU replay for peak lateral speed. CPU vs GPU flags: 0 mismatches.
- Each candidate was then scored on the 35 test seeds (3,500 episodes) **for checking only**.

**Ranking.** "fit" is the run's own union-rank calculator: ZRank, the gate, and the run's weights over the 9 val
means, recomputed on the fixed scorer.

**Notation.**
- ">4/>3.5/>3" = the count of val episodes with per-episode err above that threshold.
- v.4 = the count of val episodes whose peak lateral speed reached 0.4 m/s.

```
s1 (flown headline NEURONS#2)       VAL 500 eps: fail >4 >3.5 >3 v.4  p99  err  stdy | REPORT 3500: fail  >4  err  stdy
GRID#0     fit -0.838 <-fixed-scorer pick        0   0   0   0   0  2.69 1.46 0.87 |               21   47 1.61 0.95
GRID#1     fit  1.828                            0   0   0   8   0  3.05 2.09 1.67 |                1    7 2.12 1.62
GRID#2     fit  0.859                            0   0   0   0   0  2.42 1.50 0.97 |                0    5 1.60 1.07
NEURONS#0  fit -0.485  (= MEMORY#0)              0   0   0   0   0  2.39 1.43 0.91 |                3   12 1.55 0.97
NEURONS#1  fit -0.252  <-p99 pick                0   0   0   0   0  2.18 1.39 0.93 |                2    2 1.51 1.02
NEURONS#2  fit -0.357  <-flown (= MEMORY#1)      0   0   0   0   0  2.26 1.39 0.93 |                0    4 1.48 0.97
MEMORY#2   fit  0.938                            0   0   0   0   0  2.40 1.46 1.00 |                0    7 1.55 1.05

s2 (flown MEMORY#1 = fixed-scorer pick)
GRID#0     fit  0.571                            0   0   0   2   0  2.77 1.69 1.12 |                0    5 1.68 1.16
GRID#1     fit  1.854                            0   1   1   6   0  3.08 2.00 1.53 |                2    5 1.84 1.35
GRID#2     fit  2.124                            2   3   3   8   0  3.25 1.74 1.19 |                3   10 1.85 1.33
NEURONS#0  fit -0.907  <-p99 / >3° pick          0   0   0   0   0  2.55 1.54 0.91 |                1    6 1.66 1.12
NEURONS#1  fit -0.123  (= NEURONS#2)             0   0   0   0   0  2.75 1.60 1.08 |                1    4 1.75 1.27
MEMORY#0   fit  0.261                            0   0   3   5   0  3.00 1.64 1.10 |                1    3 1.78 1.25
MEMORY#1   fit -1.178  <-flown <-fixed           0   0   0   2   0  2.60 1.52 0.93 |                3    8 1.65 1.11
MEMORY#2   fit -0.195                            0   0   0   1   0  2.92 1.57 1.06 |                1    2 1.70 1.22

s3 (flown MEMORY#2)
GRID#0     fit  0.313                            0   0   0   1   0  2.52 1.49 1.02 |                1    3 1.56 1.06
GRID#1     fit  1.000                            0   0   0   0   0  2.81 1.63 1.06 |                1    1 1.74 1.21
GRID#2     fit  1.722                            0   0   0   2   0  2.65 1.58 1.12 |                2    6 1.66 1.21
NEURONS#0  fit -0.385  <-fixed pick, every tail pick (= MEMORY#1)  0 0 0 0 0 2.28 1.38 1.03 | 0  1 1.47 1.09
NEURONS#1  fit -0.241  (= MEMORY#0)              0   0   0   0   0  2.37 1.42 0.97 |                0    1 1.54 1.10
NEURONS#2  fit -0.026                            0   0   0   3   0  2.47 1.46 0.96 |                0    1 1.52 1.02
MEMORY#2   fit -0.321  <-flown                   0   0   0   0   0  2.46 1.41 0.94 |                1    3 1.47 0.99
```

### Headline per rule: report fails, err°, steady° (summed / mean over s1-s3)

| rule | s1 | s2 | s3 | total fails |
|---|---|---|---|---|
| as flown (ABI-45 val scores) | 0, 1.48, 0.97 | 3, 1.65, 1.11 | 1, 1.47, 0.99 | **4** |
| union fit on the FIXED scorer | **21**, 1.61, 0.95 (GRID#0) | 3, 1.65, 1.11 | 0, 1.47, 1.09 | **24** |
| val fails / >4° / >3.5° / v≥0.4 first, then fit | 21 (GRID#0) | 3 | 0 | 24 |
| val count >3° first, then fit | 21 (GRID#0) | 1, 1.66, 1.12 | 0 | 22 |
| **val p99 per-episode err first, then fit** | **2**, 1.51, 1.02 | **1**, 1.66, 1.12 | **0**, 1.47, 1.09 | **3** |

### What this shows (measured)

- **The 0.4 m/s regime is invisible on val.** 0 of 27 candidates x 500 val episodes reached 0.4 m/s. A
  count(v≥0.4) selector is blind at this val size. The base rate is about 14/10,500, so a 500-episode draw
  expects 0.7 such episodes.
- **The fixed scorer changes s1's headline to a 21-fail genome.** Re-running stage-select on the fixed scorer
  picks GRID#0, which has 21 report fails (99.4%). The published s1 headline (NEURONS#2, 0 fails) was chosen on
  the pre-fix ABI-45 val scores.
  - So s1's "0 fails" is partly selection luck from the buggy scorer.
  - A future rf_c200_dres re-fly on ABI 47 should expect a GRID-headline risk.
- **GRID#0's fails are invisible on val.** It has 0 val fails, 0 episodes >3°, and p99 2.69°. Its 21 report
  fails sit on 4 units: 99990121 (8), 99990123 (8), 99990118 (4) and 99990112 (1).
- **Fails cluster on hard units** across all candidates: 112, 113, 118, 121 and 123.
  - Each test seed is one CF2.1 unit (motor-wear draw), and val samples only 5 units.
  - *Inferred:* unit-specific weakness is mostly a coverage-of-units problem, which no per-episode tail metric
    on 5 units can fix.
- **Predictiveness of val metrics.** Spearman across 27 candidates, vs report fails / vs report err>4 count:

  | val metric | vs report fails | vs report err>4 count |
  |---|---|---|
  | p99 | +0.42 | +0.30 |
  | mean err | +0.34 | +0.33 |
  | count >4° | +0.34 | +0.24 |
  | val fails | +0.28 | +0.25 |
  | union fit | -0.10 | -0.05 |

  All are weak. The run's own fit has no rank relation to report fails.

**Leak caveat.** Test seeds were used only to check. But I chose "p99" after looking at these correlations,
so it is a POST-HOC rule (n=3 seeds, 3 vs 4 vs 24 fails). It must be pre-registered and validated on runs it did
not come from before it can be claimed.

### Recommendation (B)

Idea 3 does not need a GA arm. It is a selection rule and can be re-scored offline on any finished run's
checkpoints, as done here.

1. Pre-register **"lowest val p99 per-episode err, ties by union fit"** as a shadow selector. Report it next to
   the union-fit headline on block-4 and oq02 runs.
2. Extend val to more UNITS rather than more episodes. For example, 20-35 val seeds x 15-25 episodes, about the
   same cost.
   - This needs a code change: `VAL_SEEDS` is a hard-coded `range(5)` in `_select_headline_stage`.
   - It is the lever that matches the measured unit clustering.
3. Drop the count(v≥0.4) idea at this val size.

---

## C) High-speed training states (idea 4)

### C.1 What the trainer draws (code-read, then measured)

**Training draw (code-read).**
- 8 rounds x 24 episodes per round (RewardGatedConfig defaults; the recipe overrides neither).
- Per-axis tilt uniform up to a curriculum bound that ramps linearly 8° to 30° over the rounds
  (`round_tilt_rad`): 8, 11.1, 14.3, 17.4, 20.6, 23.7, 26.9 and 30°.
- Yaw ±5°, body rate ±0.5 rad/s.
- xy start uniform ±0.5 m **at rest**: `draw_translation_episode` hard-codes vx0 = vy0 = 0.
- `--rg-gate off`, so every episode is written.

**Scored draw.** 5° tilt, the same xy ±0.5 m at rest.

**Measured.** The headline student (a proxy for the in-training student, *inferred* to be weaker) flew 100
episodes per tilt on the train seed's plant draws. The v-clipped fraction uses the default ladder.

```
                         s1: eps≥.4  steps≥.4  v-clipped | s2: eps≥.4  steps≥.4  v-clipped | s3: eps≥.4 steps≥.4 v-clipped
scored regime  5°            0%    0.0%     6%         |      0%    0.0%    17%         |     0%   0.0%   11%
curriculum     8°            0%    0.0%    10%         |      0%    0.0%    21%         |     0%   0.0%   15%
              14.3°          2%    0.3%    25%         |     12%    3.7%    33%         |    12%   2.1%   32%
              20.6°         33%    7.3%    45%         |     34%   11.0%    49%         |    44%  11.6%   51%
              30°           67%   27.5%    64%         |     73%   28.6%    67%         |    70%  30.3%   69%
curriculum mean (8 rounds)  28.4%   9.1%    39%         |     32.0%  11.3%    45%         |   33.5% 11.6%   44%
```

**Training already visits the 0.4-0.5 m/s regime heavily.** About 30% of curriculum episodes and about 10% of
training samples are in it, against 0.13% of report episodes. The rollouts also hit the v-pinned input region
40-45% of the time under the default ladder.

So the stall addresses are not starved of training samples. *Inferred:* they are labelled from very different
physical states (8-30° recoveries) that alias to the same pinned code. That points back to idea 1, not to more
coverage.

### C.2 Written cells at the stall addresses (measured; the planned test cannot discriminate)

`--memory-mode BINARY` has no "unwritten" state. `export_cells` returns only TRUE cells, and absent = FALSE
(`cell_mode.rs`: "BINARY has no untrained state"). So "written cells at the stall addresses" cannot separate
under-trained from trained-to-FALSE.

What I could measure:

- **TRUE-cell read fraction.** The fraction of the 264 output neurons whose read address holds a TRUE cell:

  | genome | stall window, failing | own baseline reads |
  |---|---|---|
  | s2 | 11.1-11.2% | 10.2% |
  | s3 | 10.7% | 9.8% |

  No deficit.
- **Address coverage.** This is the fraction of stall-window (neuron, address) reads that the same genome also
  visits in 400 curriculum-like episodes. Within a genome the address function does not depend on the cells,
  because sn=0. I compared each genome's stall-window coverage against its own slow reads (|v|<0.2) and judged
  per episode:

  | case | stall coverage | own slow reads | outcome |
  |---|---|---|---|
  | s2 on 123/32, 123/70 | 48%, 46% | 64% | fail |
  | s2 on 112/86 | 64% | 64% | fail |
  | s3 on 113/35 | **93%** | 63% | fail |
  | s1 on 112/86, 113/35 | 48% | 64% | pass |
  | s3 on 123/32 | 33% | 63% | pass |

  **Coverage does not predict failure.** The under-training hypothesis is not supported.

### C.3 Training-only start values that would put the training distribution in the stall regime

Measured on the headline students: the scored 5° tilt, xy ±0.5 m, 100 episodes per setting.

```
setting (5° tilt)            eps≥.4 (s1/s2/s3)   peak speed p50-p90     stable (s1/s2/s3)
xy ±0.75 at rest              0/0/0%              0.13-0.20              100/100/100
xy ±1.0 at rest               0/0/0%              0.14-0.22              98/96/99
xy ±0.5 + v0 U(±0.3)/axis     2/2/5%              0.25-0.35              100/100/99
xy ±0.5 + v0 U(±0.4)/axis     25/27/31%           0.34-0.46              100/97/94
xy ±0.5 + v0 U(±0.5)/axis     54/56/61%           0.42-0.57              98/90/88
```

- A larger start offset does NOT produce high speed: the student brakes before 0.3 m/s.
- An initial horizontal velocity of **U(±0.45 m/s) per axis** (with xy ±0.5 m) centres training episodes on the
  0.4-0.5 m/s, 5°-attitude regime. Interpolated: about 40% of episodes ≥0.4, p90 about 0.5 m/s.
- **This needs code.**
  - No trainer knob exists for initial horizontal velocity.
  - `--xy-offset` sets both the scorer and the trainer, so it cannot be made training-only today.

*Inferred:* flown on the DEFAULT ladder, these extra states land in the same aliased corner as the curriculum's
high-speed states, so idea 4 is predicted null there. Fly it only stacked on the idea-1 winner.

---

## Recommended arm configs

All of these are added to the `rf_c200_dres` recipe, paired per seed against `rf_c200_dres` on the ABI-47
scorer, n=3 seeds, and read as triples plus fails/10,500.

1. **Idea 1 (+5): ladder outer quantile.**
   - Flags: `--threshold-outer-quantile 0.02`.
   - This is the existing `rf_c200_dres_oq02` arm (s2 flying now). No new arm is needed; it is the arm the
     offline test ranks first.
2. **Idea 1, low-resolution-cost variant (NEW arm).**
   - Flags: `--calib-ladder centred --threshold-outer-quantile 0.02`.
   - It de-aliases the same features as oq02 at about +10% near-zero bin width instead of about +25%.
3. **Idea 5 (calibration tilt).** No arm recommended.
   - `--threshold-calib-tilt 30` is dominated offline by items 1-2.
   - Tilt 30 combined with any outer q re-aliases.
   - Fly `--threshold-calib-tilt 30` alone only to close the August question.
4. **Idea 2 (student refit).** No arm. `--threshold-refit-from-student` is a silent no-op on this recipe
   (ABI-32 refusal), and its output would reach only GRID. Fixing that is a code task. The offline gain over
   q0.02 is nil.
5. **Idea 3 (tail-aware selection).** No GA arm.
   - Pre-register the shadow selector "min val p99 per-episode err, ties by union fit" (offline, from checkpoints).
   - The structural fix is more val units: 20-35 val seeds x 15-25 episodes. That needs a code change to the
     hard-coded 5 val seeds.
6. **Idea 4 (high-speed training states).** Deferred until items 1-2 read out.
   - Then fly: winner ladder + a NEW training-only flag giving vx0, vy0 ~ U(±0.45 m/s) per axis (scorer
     unchanged, xy ±0.5 m kept). That flag needs code.
   - Do not widen `--xy-offset`: it changes the scored distribution, and the measured speed gain was zero.
