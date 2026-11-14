# CTRL-48 fail-free stress test — plan (Luiz 14/11/2026), written BEFORE any stress run

## Question
Over 35 test units x 100 episodes, a controller with 0-1 fails and the MPCOF cascade with 0 fails are indistinguishable
(95% upper bound on a 0/3500 rate ≈ 1 in 1,200). How do the block-9 winner's controllers and MPCOF compare on fails when
each is flown on ~9x more test units?

## What is run (no training, no search)
- For each of the block-9 WINNING arm's 10 published controllers (s1-s10): `--recalc-headline` under rule P with
  `--test-seeds 99990101..99990400` (300 CF2.1 units x 100 episodes = 30,000 episodes per controller; the first 35 are the
  usual test units; of the other 265, units 99990201-210 were scored once before as report-only fresh units in the CTRL-45 validity audit, and 99990201-05 are the reserved multi-axis 'D5' final test seeds — none was ever used for selection; the remaining 255 are touched here for the first time). Driver logs/controller/ctrl48/stress_test.sh;
  parser logs/controller/ctrl48/stress_result.py REFUSES a run whose re-pick is not the published headline.
- The MPCOF cascade (teacher, est-fed) is scored by the same report blocks on the SAME units and episodes.
- Tooling check: on an existing 35-unit recalc log the parser reproduces MPCOF 0 fails / 1.353° / 0.685° / 0.341 m.

## What is reported (descriptive; nothing is selected on these units)
Per controller: fails/30,000, stable, err, steady, pos. Pooled WNN fails/300,000 with the exact 95% interval; MPCOF
fails/30,000 with its 95% interval; the per-unit breakdown of where fails cluster (hard motor-wear units).
No configuration choice depends on this test; block 9's pre-registered decision stands on its own 35-unit rule.
These 300 units become the "extended test set": like the 35, they are report-only from here on.

## Result (14/11/2026 23:40 UTC, `STRESS DONE`; descriptive, TEST units)

`both` (block 9), rule-P headline of each seed, 300 test units × 100 episodes per controller:

```
controller  pick       stable%   err°   steady°  pos m  fails
s1          NEURONS#1   99.990  1.441   0.83   0.359   3/30000
s2          MEMORY#2   100.000  1.472   0.82   0.360   1/30000
s3          NEURONS#0  100.000  1.312   0.72   0.355   0/30000
s4          NEURONS#1   99.970  1.453   0.87   0.357   8/30000
s5          MEMORY#2   100.000  1.375   0.75   0.358   1/30000
s6          NEURONS#1   99.990  1.310   0.73   0.354   3/30000
s7          NEURONS#2  100.000  1.400   0.77   0.361   1/30000
s8          NEURONS#0  100.000  1.436   0.83   0.359   1/30000
s9          NEURONS#2  100.000  1.437   0.79   0.365   0/30000
s10         MEMORY#2   100.000  1.398   0.81   0.357   1/30000
mean                            1.403   0.79   0.358
```

- **WNN pooled: 19 / 300,000 fails = 0.0063 %** (exact 95 % CI 0.0038–0.0099 %). 8 of the 19 come from s4.
- **MPCOF, no bias compensation (est-fed), same 300 units: 5 / 30,000 = 0.017 %** (95 % CI 0.0054–0.039 %);
  err 1.396°, steady 0.697°, pos 0.340 m.
- WNN vs that MPCOF row, conditional binomial test on exposure 10:1: two-sided p = 0.061 — no significant
  difference. Safe wording: "fail rates of the same order (≈ 1 in 16,000 vs ≈ 1 in 6,000)"; never "fewer fails
  than MPCOF".
- MPCOF is not fail-free at this sample size: its 0/3500 on the 35 units was an under-powered zero.
- **Gap:** the equal-information (bias-compensated) MPCOF COMPARATOR was NOT scored on these 300 units, so
  there is no fail rate yet for the paper's comparator row.
