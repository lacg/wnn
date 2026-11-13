# CTRL-48 block 8 — yaw levers on the dc_biasin_ms01 base (controller ABI 50, rule P) — RESULT

Flown 13/11/2026 03:37-19:24 UTC (driver logs/controller/ctrl48/ab_block8.sh; END lines in runs.log). Base = dc_biasin_ms01
(block 7's best) re-flown as the in-block control; every run selected with rule P (--val-units 20 --select-tail p99).
All numbers TEST: 35 CF2.1 units x 100 episodes per run. Paired vs the SAME seed's block-8 control. ileak s3 SKIPPED (Luiz 13/11,
already GATE-FAIL). Gate per the 11/11 FAILURE GATE (pooled rate <= 0.05% AND not worse than control; passers ranked on steady
then err; ±0.03° = tie).

## Per seed (Δ vs same-seed control: stable pp · err ° · steady ° · pos m · fails)
| arm    | s1                              | s2                              | s3                              |
|--------|---------------------------------|---------------------------------|---------------------------------|
| ctl    | 100.0 / 1.34 / 0.85 / 0.359 / 1 | 100.0 / 1.47 / 0.93 / 0.362 / 0 | 100.0 / 1.34 / 0.83 / 0.362 / 1 |
| ileak  | −0.09 · 0.00 · −0.07 · −0.002 · +3 | −0.09 · −0.06 · −0.13 · 0 · +3 | skipped                         |
| ycl    | 0 · +0.09 · +0.03 · +0.002 · 0  | 0 · +0.05 · +0.01 · +0.002 · 0  | +0.03 · 0.00 · −0.03 · −0.004 · −1 |
| yileak | 0 · +0.02 · −0.01 · +0.002 · 0  | 0 · +0.03 · +0.04 · 0 · 0       | 0 · +0.05 · −0.03 · −0.003 · 0  |
| both   | −0.03 · +0.06 · −0.03 · +0.002 · +1 | 0 · −0.01 · −0.11 · +0.001 · 0 | +0.03 · −0.08 · −0.15 · −0.004 · −1 |

## Gate table (n=3; control = block-8 ctl: 2/10500, steady 0.870, err 1.383)
| arm    | runs | fails/episodes | rate   | mean steady | mean err | verdict |
|--------|------|----------------|--------|-------------|----------|---------|
| both   | 3    | 2/10500        | 0.019% | 0.773       | 1.373    | PASS, beats on steady (−0.10°), err tie (−0.01°) |
| ycl    | 3    | 1/10500        | 0.010% | 0.873       | 1.430    | PASS, ties on steady (+0.00°), worse err (+0.05°) |
| yileak | 3    | 2/10500        | 0.019% | 0.870       | 1.417    | PASS, ties on steady (+0.00°), err +0.03° (edge of tie) |
| ileak  | 2    | 7/7000         | 0.100% | —           | —        | GATE-FAIL (s3 skipped) |

## Reading
- Neither yaw lever alone moves steady; together (centred yaw ladder + yaw-only integral leak 0.9995) they cut steady by 0.10°
  at equal err and equal fails — the pre-check's "the yaw leak pays mainly with the ladder", now 3/3 seeds non-positive
  (−0.03 / −0.11 / −0.15; s1 is a tie). both s3 = 0.68° steady = the MPCOF teacher's 0.68°.
- New best configuration: both 0.77° steady / 1.37° err / 2 fails per 10500 (block 7: 0.86 / 1.41 / 2).
- The control re-flown on ABI 50 reproduced block 7 (1.38 / 0.87 / 2); P picked B's genome on s1 and s2, a different one on s3.
- Caveats: n=3, heterogeneous per-seed gains (one tie); a direction strong enough to adopt under the gate, not a paper claim.
