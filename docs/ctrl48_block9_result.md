# CTRL-48 block 9 — dc_ms01 vs both, head-to-head, n=10 — RESULT (pre-registered, docs/ctrl48_block9_prereg.md + Amendments 1-2)

Flown 14/11/2026 01:51-16:36 UTC (s4-s10, 14 runs, controller ABI 50); dc_ms01 s1-s3 re-picked under rule P 16:36-17:24 UTC;
both s1-s3 = block 8. Every controller selected with rule P. All numbers TEST: 35 CF2.1 units x 100 episodes per seed.

```
A = dc_ms01 · B = both
seed    stable %        err °          steady °       pos m           fails   Δ (B − A) = both − dc_ms01: err · steady · pos · fails
oracle-fed (label source, bound)   100.0   0.69   0.16   0.357   0
MPCOF equal-info (COMPARATOR)      100.0   0.69   0.17   0.357   0
MPCOF no bias comp                 100.0   1.35   0.68   0.341   0
s1    100.0 |  99.9    1.38 | 1.40    0.88 | 0.82    0.357 | 0.361   0 | 2   +0.02 · −0.06 · +0.004 · +2
s2    100.0 | 100.0    1.57 | 1.46    1.06 | 0.82    0.363 | 0.363   0 | 0   −0.11 · −0.24 ·  0.000 ·  0
s3     99.9 | 100.0    1.36 | 1.26    0.92 | 0.68    0.354 | 0.358   2 | 0   −0.10 · −0.24 · +0.004 · −2
s4     99.9 | 100.0    1.44 | 1.40    0.94 | 0.83    0.357 | 0.358   3 | 1   −0.04 · −0.11 · +0.001 · −2
s5    100.0 | 100.0    1.38 | 1.34    0.88 | 0.72    0.360 | 0.360   0 | 0   −0.04 · −0.16 ·  0.000 ·  0
s6    100.0 | 100.0    1.38 | 1.28    0.93 | 0.71    0.358 | 0.358   0 | 1   −0.10 · −0.22 ·  0.000 · +1
s7    100.0 | 100.0    1.43 | 1.36    0.87 | 0.75    0.350 | 0.363   1 | 1   −0.07 · −0.12 · +0.013 ·  0
s8    100.0 | 100.0    1.41 | 1.45    0.95 | 0.86    0.359 | 0.361   0 | 1   +0.04 · −0.09 · +0.002 · +1
s9     99.9 | 100.0    1.37 | 1.37    0.93 | 0.74    0.350 | 0.367   5 | 0    0.00 · −0.19 · +0.017 · −5
s10   100.0 | 100.0    1.36 | 1.36    0.90 | 0.80    0.353 | 0.359   1 | 0    0.00 · −0.10 · +0.006 · −1

s1-s10        n    fails/episodes   err °            steady °         pos m            0-fail seeds
A dc_ms01     10   12 / 35000       1.408 ± 0.065    0.926 ± 0.055    0.356 ± 0.004    5 / 10
B both        10    6 / 35000       1.368 ± 0.065    0.773 ± 0.061    0.361 ± 0.003    5 / 10
paired Δ (B − A): steady −0.153 ± 0.066° (better on 10/10) · err −0.040 ± 0.055° (better 6, tied 2, worse 2) · pos +0.005 m
```

## Decision rule (Amendment 2, s1-s10)
1. Safety gate: A 12/35000 (0.034%) and B 6/35000 (0.017%), both <= 17/35000; B not significantly worse than A (exact one-sided
   conditional binomial p = 0.95), A not significantly worse than B (p = 0.12). **Both PASS.**
2. both's mean steady better by > 0.03°: −0.153° -> yes. both's mean err not worse by > 0.03°: −0.040° (better) -> yes.
**DECISION: `both` is the base.** (s4-s10 alone, descriptive: fails A 10 vs B 4, steady −0.141°, err −0.030° — same decision.)

## Notes
- Re-picking dc_ms01 s1-s3 under P changed all three picks; s3 went from 0 to 2 fails (P is not a guarantee per seed).
- Fail-free seeds 5/10 for each arm (95% lower bound on the fail-free-seed fraction ≈ 22%): neither recipe reliably yields a
  0-fail controller on 3500 episodes. dc_ms01's earlier "0/10500" was three lucky seeds under rule B.
- Against the equal-information MPCOF comparator (docs/ctrl_rival_row_ruling.md) `both` is +0.68° err / +0.60° steady; pos is
  task-defined (no claim). Next: the 300-unit stress test on both's ten controllers (docs/ctrl48_stress_test_plan.md), block 10 levers.
