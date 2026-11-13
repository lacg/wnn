# Tail-aware selection — pre-registered readout (CTRL-48)

Generated 13/11/2026 01:10 UTC by `logs/controller/ctrl48/select_tail_result.py` from `experiments/ctrl_select_tail/`. Pre-registration: `docs/ctrl_select_tail_prereg.md` (unchanged). Rule P = `--val-units 20 --select-tail p99`; baseline B = 5 val units, no tail term. Every pick scored on the 35 test units x 100 episodes (fixed scorer, controller ABI 49). Runs in the list: 30; runs with both B and P: 30.

## Per-run picks (genome | test err° | test steady° | test fails/3500)

| run | original headline | B | P | S20 (descr.) | S5p (descr.) | B = original? |
|---|---|---|---|---|---|---|
| rf_c200_dres_full_s1 | MEMORY#1 | MEMORY#1 | 1.467 | 0.900 | 1 | MEMORY#1 | 1.467 | 0.900 | 1 | — | — | yes |
| rf_c200_dres_quad_full_s1 | NEURONS#0 | NEURONS#0 | 1.503 | 0.900 | 6 | MEMORY#1 | 1.520 | 0.910 | 7 | — | — | yes |
| rf_c200_dres_ms01_full_s1 | NEURONS#2 | NEURONS#2 | 1.424 | 0.900 | 1 | NEURONS#0 | 1.448 | 0.970 | 0 | — | — | yes |
| rf_c200_dres_lv32_full_s1 | NEURONS#0 | NEURONS#0 | 1.542 | 1.000 | 6 | NEURONS#2 | 1.512 | 1.000 | 1 | — | — | yes |
| rf_c200_dres_mlab_full_s1 | NEURONS#0 | NEURONS#0 | 1.506 | 0.990 | 26 | NEURONS#0 | 1.506 | 0.990 | 26 | — | — | yes |
| rf_c200_dres_biasin_full_s1 | NEURONS#0 | NEURONS#0 | 1.461 | 0.890 | 3 | NEURONS#0 | 1.461 | 0.890 | 3 | — | — | yes |
| rf_c200_dres_full_s2 | NEURONS#2 | NEURONS#2 | 1.575 | 0.990 | 0 | MEMORY#1 | 1.675 | 1.130 | 0 | — | — | yes |
| rf_c200_dres_quad_full_s2 | MEMORY#2 | MEMORY#2 | 1.521 | 1.000 | 2 | NEURONS#1 | 1.528 | 1.020 | 1 | — | — | yes |
| rf_c200_dres_ms01_full_s2 | NEURONS#2 | NEURONS#2 | 1.654 | 1.160 | 0 | NEURONS#0 | 1.702 | 1.200 | 0 | — | — | yes |
| rf_c200_dres_lv32_full_s2 | NEURONS#0 | NEURONS#0 | 1.639 | 1.050 | 1 | NEURONS#0 | 1.639 | 1.050 | 1 | — | — | yes |
| rf_c200_dres_mlab_full_s2 | NEURONS#1 | NEURONS#1 | 1.622 | 1.160 | 13 | NEURONS#0 | 1.624 | 1.190 | 6 | — | — | yes |
| rf_c200_dres_biasin_full_s2 | MEMORY#2 | MEMORY#2 | 1.557 | 1.000 | 0 | NEURONS#0 | 1.504 | 0.970 | 0 | — | — | yes |
| rf_c200_dres_full_s3 | NEURONS#2 | NEURONS#2 | 1.470 | 0.990 | 1 | NEURONS#1 | 1.506 | 1.040 | 0 | — | — | yes |
| rf_c200_dres_quad_full_s3 | NEURONS#0 | NEURONS#0 | 1.439 | 0.920 | 0 | NEURONS#0 | 1.439 | 0.920 | 0 | — | — | yes |
| rf_c200_dres_ms01_full_s3 | NEURONS#0 | NEURONS#0 | 1.506 | 0.970 | 0 | NEURONS#0 | 1.506 | 0.970 | 0 | — | — | yes |
| rf_c200_dres_lv32_full_s3 | NEURONS#0 | NEURONS#0 | 1.499 | 1.050 | 1 | MEMORY#2 | 1.505 | 0.970 | 0 | — | — | yes |
| rf_c200_dres_mlab_full_s3 | NEURONS#2 | NEURONS#2 | 1.491 | 1.000 | 5 | MEMORY#2 | 1.482 | 1.010 | 1 | — | — | yes |
| rf_c200_dres_biasin_full_s3 | MEMORY#2 | MEMORY#2 | 1.416 | 0.800 | 1 | NEURONS#1 | 1.371 | 0.800 | 0 | — | — | yes |
| rf_c200_dres_oq02_full_s1 | NEURONS#2 | NEURONS#2 | 1.718 | 1.210 | 0 | NEURONS#2 | 1.718 | 1.210 | 0 | — | — | yes |
| rf_c200_dres_cent_oq02_full_s1 | GRID#0 | GRID#0 | 1.570 | 0.970 | 1 | GRID#0 | 1.570 | 0.970 | 1 | — | — | yes |
| rf_c200_dres_ct30_full_s1 | NEURONS#0 | NEURONS#0 | 1.558 | 1.030 | 2 | MEMORY#2 | 1.585 | 1.090 | 3 | — | — | yes |
| rf_c200_dres_tvxy045_full_s1 | NEURONS#0 | NEURONS#0 | 1.655 | 0.910 | 1 | NEURONS#1 | 1.700 | 1.040 | 4 | — | — | yes |
| rf_c200_dres_oq02_full_s2 | NEURONS#1 | NEURONS#1 | 1.730 | 1.250 | 1 | MEMORY#2 | 1.703 | 1.240 | 0 | — | — | yes |
| rf_c200_dres_cent_oq02_full_s2 | NEURONS#0 | NEURONS#0 | 1.753 | 1.220 | 0 | NEURONS#2 | 1.730 | 1.190 | 0 | — | — | yes |
| rf_c200_dres_ct30_full_s2 | NEURONS#0 | NEURONS#0 | 1.687 | 1.210 | 0 | NEURONS#1 | 1.720 | 1.240 | 1 | — | — | yes |
| rf_c200_dres_tvxy045_full_s2 | NEURONS#0 | NEURONS#0 | 1.616 | 0.940 | 0 | NEURONS#0 | 1.616 | 0.940 | 0 | — | — | yes |
| rf_c200_dres_oq02_full_s3 | MEMORY#2 | MEMORY#2 | 1.550 | 1.150 | 1 | MEMORY#2 | 1.550 | 1.150 | 1 | — | — | yes |
| rf_c200_dres_cent_oq02_full_s3 | NEURONS#1 | NEURONS#1 | 1.581 | 1.100 | 0 | NEURONS#0 | 1.551 | 1.080 | 0 | — | — | yes |
| rf_c200_dres_ct30_full_s3 | NEURONS#0 | NEURONS#0 | 1.491 | 1.010 | 0 | NEURONS#0 | 1.491 | 1.010 | 0 | — | — | yes |
| rf_c200_dres_tvxy045_full_s3 | NEURONS#1 | NEURONS#1 | 1.915 | 1.130 | 3 | GRID#0 | 1.829 | 1.170 | 0 | — | — | yes |

## Paired P vs B (test units)

| run | B genome | P genome | differs | Δerr° (P−B) | Δsteady° (P−B) | fails B | fails P |
|---|---|---|---|---|---|---|---|
| rf_c200_dres_full_s1 | MEMORY#1 | MEMORY#1 | no | +0.000 | +0.000 | 1 | 1 |
| rf_c200_dres_quad_full_s1 | NEURONS#0 | MEMORY#1 | yes | +0.017 | +0.010 | 6 | 7 |
| rf_c200_dres_ms01_full_s1 | NEURONS#2 | NEURONS#0 | yes | +0.024 | +0.070 | 1 | 0 |
| rf_c200_dres_lv32_full_s1 | NEURONS#0 | NEURONS#2 | yes | -0.030 | +0.000 | 6 | 1 |
| rf_c200_dres_mlab_full_s1 | NEURONS#0 | NEURONS#0 | no | +0.000 | +0.000 | 26 | 26 |
| rf_c200_dres_biasin_full_s1 | NEURONS#0 | NEURONS#0 | no | +0.000 | +0.000 | 3 | 3 |
| rf_c200_dres_full_s2 | NEURONS#2 | MEMORY#1 | yes | +0.100 | +0.140 | 0 | 0 |
| rf_c200_dres_quad_full_s2 | MEMORY#2 | NEURONS#1 | yes | +0.007 | +0.020 | 2 | 1 |
| rf_c200_dres_ms01_full_s2 | NEURONS#2 | NEURONS#0 | yes | +0.048 | +0.040 | 0 | 0 |
| rf_c200_dres_lv32_full_s2 | NEURONS#0 | NEURONS#0 | no | +0.000 | +0.000 | 1 | 1 |
| rf_c200_dres_mlab_full_s2 | NEURONS#1 | NEURONS#0 | yes | +0.002 | +0.030 | 13 | 6 |
| rf_c200_dres_biasin_full_s2 | MEMORY#2 | NEURONS#0 | yes | -0.053 | -0.030 | 0 | 0 |
| rf_c200_dres_full_s3 | NEURONS#2 | NEURONS#1 | yes | +0.036 | +0.050 | 1 | 0 |
| rf_c200_dres_quad_full_s3 | NEURONS#0 | NEURONS#0 | no | +0.000 | +0.000 | 0 | 0 |
| rf_c200_dres_ms01_full_s3 | NEURONS#0 | NEURONS#0 | no | +0.000 | +0.000 | 0 | 0 |
| rf_c200_dres_lv32_full_s3 | NEURONS#0 | MEMORY#2 | yes | +0.006 | -0.080 | 1 | 0 |
| rf_c200_dres_mlab_full_s3 | NEURONS#2 | MEMORY#2 | yes | -0.009 | +0.010 | 5 | 1 |
| rf_c200_dres_biasin_full_s3 | MEMORY#2 | NEURONS#1 | yes | -0.045 | +0.000 | 1 | 0 |
| rf_c200_dres_oq02_full_s1 | NEURONS#2 | NEURONS#2 | no | +0.000 | +0.000 | 0 | 0 |
| rf_c200_dres_cent_oq02_full_s1 | GRID#0 | GRID#0 | no | +0.000 | +0.000 | 1 | 1 |
| rf_c200_dres_ct30_full_s1 | NEURONS#0 | MEMORY#2 | yes | +0.027 | +0.060 | 2 | 3 |
| rf_c200_dres_tvxy045_full_s1 | NEURONS#0 | NEURONS#1 | yes | +0.045 | +0.130 | 1 | 4 |
| rf_c200_dres_oq02_full_s2 | NEURONS#1 | MEMORY#2 | yes | -0.027 | -0.010 | 1 | 0 |
| rf_c200_dres_cent_oq02_full_s2 | NEURONS#0 | NEURONS#2 | yes | -0.023 | -0.030 | 0 | 0 |
| rf_c200_dres_ct30_full_s2 | NEURONS#0 | NEURONS#1 | yes | +0.033 | +0.030 | 0 | 1 |
| rf_c200_dres_tvxy045_full_s2 | NEURONS#0 | NEURONS#0 | no | +0.000 | +0.000 | 0 | 0 |
| rf_c200_dres_oq02_full_s3 | MEMORY#2 | MEMORY#2 | no | +0.000 | +0.000 | 1 | 1 |
| rf_c200_dres_cent_oq02_full_s3 | NEURONS#1 | NEURONS#0 | yes | -0.030 | -0.020 | 0 | 0 |
| rf_c200_dres_ct30_full_s3 | NEURONS#0 | NEURONS#0 | no | +0.000 | +0.000 | 0 | 0 |
| rf_c200_dres_tvxy045_full_s3 | NEURONS#1 | GRID#0 | yes | -0.086 | +0.040 | 3 | 0 |

## Pass criteria (pre-registered; P adopted only if ALL hold)

1. Total test fails P < B: P 57 vs B 76 (episodes 105000 each) -> **PASS**
2. Mean test err and steady, P not worse than B by > 0.05°: Δerr +0.001°, Δsteady +0.015° -> **PASS**
3. P picks a different genome than B on >= 3 runs: 19/30 -> **PASS**

**Verdict under the pre-registered rule: ADOPTED.**
