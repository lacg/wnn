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
