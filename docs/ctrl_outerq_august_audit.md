# Audit: the August 2026 `--threshold-outer-quantile` conclusions after the 09/11 plumbing fix

09/11/2026. Read-only audit: no runs, no code changes. Trigger: commit `8542f4c1` (09/11) found that
`--threshold-outer-quantile` reached only 2 of the 5 threshold-ladder fits in `phased_ga.py`.

## 1. What the August code actually did

All runs below were launched from source at `5f3d113c` (10/08 13:56 UTC). No `src/wnn/control` file
changed again until `950f4013` (11/08 14:02 UTC), and the last cell finished at 01:07 UTC on 11/08. I
checked the five fit sites at `5f3d113c` (`git show 5f3d113c:src/wnn/control/phased_ga.py`):

```
  site                                         passes outer_quantile?   ladder actually used
  GRID  ControllerGridSearch.__init__ (cgs:95)   YES                     tilt5 + q
  student refit _refit_thresholds_from_student   YES                     (never flown with q)
  NEURONS _run_arch_phase (pg:642)               NO                      tilt5, default q
  MEMORY  _run_memory_phase (pg:1027)            NO                      tilt5, default q
  _report_thresholds (pg:903): every held-out
    + stage-select val scoring                   NO                      tilt5, default q
  arch-only held-out train fit (pg:1539)         NO                      tilt5, default q
```

So the August behaviour matches today's bug exactly. Four further points settle the rest.

- **`--threshold-calib-tilt` WAS honoured everywhere**, through `calib_episode_config` at every site. Only the
  quantile was dropped.
- **No code path carried the grid ladder forward.** `--holdout-fixed-thresholds` (`8ab1d466`, 01/08)
  became unconditional on 03/08 as `_report_thresholds`. It fixes the fit SEED to the train seed. It never
  carried the grid's ladder, and it rebuilds the default-q ladder.
- **Grid genomes carry no cells.** `controller_grid_search.py:226` sets `genome.cells = None`, and
  `evaluate_batch` does not write back. NEURONS therefore re-trained the grid pool from empty under its own
  default-q ladder, and MEMORY inherited those cells. The GRID held-out goes through `_holdout_report`'s
  arch-only branch, which also re-trains under the default-q ladder.
- **Consequence: there was NO cell-level train/score mismatch (case b) in August.** Every number that was
  reported, including GRID, NEURONS, MEMORY, stage-select and headline, was trained AND scored on the
  tilt-5, default-quantile ladder. The q ladder had exactly one effect: it ranked the GRID stage's
  during-search pool, so it decided WHICH architectures/connectivities seeded NEURONS. That is a
  selection-ladder mismatch: shapes chosen under one address function, then trained and judged under
  another.

**Empirical confirmation, from the markers alone.** The q=0.02 and q=0.005 cells of the SAME seed contain a
bit-identical stage-select candidate on both seeds:

```
  s31337002  q002 NEURONS#2 == q0005 NEURONS#2  : 94.0%/2.20°/1.56° rw -8.56 jrk 0.0252 mono 2.8
  s31337003  q002 NEURONS#1 == q0005 NEURONS#2  : 99.6%/1.24°/0.65° rw -2.31 jrk 0.0279 mono 1.1
```

Identical cells and identical val triples are impossible if the two q values had reached the trained
stages. They did not, so the two "q levels" are one trained-stage encoder flown twice with different grid
pools.

**Translation / ABI-47 scorer:** none of the three scripts passes `--translation` (attitude-only, lqi
teacher, L4C, cf21_brushless, BINARY, levels 16, sn=0). The 25-distinct-translation-starts GPU bug does not
apply to any run here.

**Case (c), refit plus q:** never happened. E1 resolved `E1_Q` empty and flew only `c30`. All 10 E1
markers carry `"outer_q":""`.

## 2. Per-run table

Triples are stable% / err° / steady°, held out over report seeds 99990101-05 (N=5 report seeds, ±
over report seeds). HEADLINE = stage-select pick (val seeds), present since `d3f64e03`.

```
run (…_cf21_brushless_L4C)    seed  headline   affected         headline          GRID ho                NEURONS ho              MEMORY ho
----------------------------+-----+----------+----------------+-----------------+----------------------+-----------------------+----------------------
OQ_lqi_c30                   002   NEURONS#0  unaffected (ctl) 100.0/1.02/0.63   96.8/1.61/1.28         100.0/1.02/0.63         100.0/1.01/0.54
OQ_lqi_c30                   003   MEMORY#1   unaffected (ctl) 100.0/0.88/0.36   97.4/2.33/2.11         100.0/0.89/0.36         100.0/0.92/0.37
OQ_lqi_q002                  002   NEURONS#0  (a) + grid-sel   99.2/1.65/1.16    96.6/1.71/1.09         99.2/1.65/1.16          99.4/1.52/1.06
OQ_lqi_q002                  003   MEMORY#1   (a) + grid-sel   99.6/1.23/0.69    68.6/6.01/6.82         99.2/1.65/1.14          99.8/1.26/0.69
OQ_lqi_q0005                 002   MEMORY#2   (a) + grid-sel   99.6/1.40/0.91    31.0/14.01/20.12       99.8/1.30/0.88          99.0/1.36/0.84
OQ_lqi_q0005                 003   MEMORY#0   (a) + grid-sel   99.8/1.23/0.66    89.4/2.83/2.49         95.8/2.13/2.43          99.8/1.23/0.66
E1_lqi_c30_refit{off,on} x5  002-006  —       unaffected (no flag; E1_Q resolved empty)  — numbers as in l4 doc "Refit (E1 + E1b)"
E2_lqi_g{10,20}_c30 x3       002-004  —       unaffected (no flag; base enc resolved c30) — numbers as in l4 doc "E2"
```

"(a) + grid-sel" means every reported row was trained and scored on the DEFAULT-q ladder at calib tilt 5.
The only thing q changed was which GRID shapes seeded NEURONS. The GRID held-out row is NOT a q-ladder
measurement either: it is the q-selected grid winner re-trained on the default ladder. That is the likely
source of the GRID collapses in the q cells, 31.0% and 68.6% stable.

**What the OQ arm really compared:** `calib-tilt 30, default q` (control) against `calib-tilt 5, default q,
grid pool selected under a q ladder` (both "treatments"). Per-seed headline steady Δ (treatment − control):

```
  seed   q002 cell   q0005 cell   seed mean
  002     +0.53       +0.28        +0.405
  003     +0.33       +0.30        +0.315
  n=2 independent seeds: mean +0.36°, SD 0.064, 95% CI [-0.21, +0.93]  (t, 1 df)
```

The two cells within a seed share the trained-stage encoder and differ only in the grid pool. The
within-seed spread (0.25° on s002) is the size of the grid-selection effect alone, about as large as the
smaller cross-arm deltas.

## 3. Verdict per claimed conclusion

```
claim (where)                                                         verdict              evidence
--------------------------------------------------------------------+--------------------+-------------------------------------------------------------
"Outer-quantile REFUTED 4/4" (l4:1173-1187; reassess:224)             VOID                 q never reached any trained/scored stage; identical NEURONS genomes across q levels
"outside% is dead as a predictor ... flown order exact reverse"        VOID (untested)      the flown trained ladder was tilt5-default, not the q ladders whose outside% was ranked
  (l4:1184-1187; reassess:273)
Cell-count mechanism "q=0.02 populated Σ6.9M vs control Σ12.2M"        VOID                 cross-seed pairing (q002 s002 vs c30 s003); same-seed s003 goes the other way (16.8M vs 12.2M); cells trained on default ladder
  (l4:1189-1191)
"every outer-quantile candidate beats every tilt-only one" (l4:934)    STANDS (as proxy)    offline fit of outside% only, never touches phased_ga; says nothing about flight
E1 degenerated to refit-only "correctly" (l4:1165-1171; reassess:274)  WEAKENED             the gate read a void verdict; the enc x refit 2x2 was never asked
Refit NULL at n=5, CI [-0.36,+0.38] (l4:1193-1247)                    STANDS               c30, no q flag; unaffected
E2 gamma=2 REFUTED 3/3, CI [+0.27,+1.05] (l4:1136-1163)               STANDS               c30 base, no q flag; within-arm paired
Five-seed control band 0.36-0.93°, deterministic pipeline (l4:1105)   STANDS               c30 controls, bit-identical across OQ/E1/E2
"GRID is not the read-out" rule                                       STANDS               derived from the c30 control seeds; E2 confirms it independently
"GRID swung 96.8% -> 31.0%" as encoder evidence (e2 script comment)    VOID                 31.0% = q-selected shape re-trained on a different ladder (selection mismatch)
"Both quantizer routes CLOSED" / "Resolution is not the binding        WEAKENED-needs-rerun action half stands (gamma refuted; L64 unresolved); perception half rests on
  constraint in either channel" (l4:1249-1265; reassess:217-249)                            the refit null ONLY, the quantile lever was never flown
"§6 freeze precondition MET ... settled as a refutation"               WEAKENED             the encoder question was not settled for the quantile
  (reassess:239-241)
"L3 and outer-q are unaffected" by the unanimity-bar withdrawal        VOID (outer-q half)  there is no outer-q result to be affected
  (reassess:257-258)
"Perception was that [second binding] constraint" (l4:992;            WEAKENED             no flown lever on perception has shown a gain; read correctly, the only
  memory project_thermometer_regime_mismatch)                                               same-code tilt-5 vs tilt-30 comparison went AGAINST tilt 5 (below)
```

**What the OQ arm DOES support, read correctly (PROVISIONAL, n=2 seeds, confounded):** the tilt-5
calibrated encoder with default quantiles was worse than the tilt-30 one on all 4 cells. The 2 seeds give
+0.36° mean steady, but the CI straddles 0, and the comparison is confounded with the grid
selection-ladder mismatch. This matters because the tilt-5 calibration is what every recipe has used by
DEFAULT since `4514d5c9` (`calib_episode_config` returns the flown `ec`). The August controls and the sn>0
chain pinned `--threshold-calib-tilt 30`. The current C-S2-d / ctrl48 recipe
(`experiments/ctrl22/ARGS_ma_anchor_lag020.txt`) passes no calib tilt, so it flies tilt 5. I did not find a
flown result that positively justified the switch. Label this an open question, not a finding.

## 4. Docs and memory stating now-invalid conclusions (do not edit here; suggestions only)

```
file:line                                              current text (gist)                                  suggested correction
-----------------------------------------------------+----------------------------------------------------+------------------------------------------------------------
docs/l4_teacher_screen_results.md:1173-1187            "Outer-quantile — REFUTED 4/4 ... outside% is dead"   retitle "VOID — flag never reached trained stages (8542f4c1,
                                                                                                             see ctrl_outerq_august_audit.md)"; keep table as a
                                                                                                             tilt5-default vs c30 comparison, n=2 seeds, provisional
docs/l4_teacher_screen_results.md:1189-1191            cell-count mechanism                                 strike; cross-seed pairing and wrong ladder
docs/l4_teacher_screen_results.md:1165-1171            E1 degenerated "correctly"                           "degenerated on a verdict later voided; the q x refit
                                                                                                             interaction remains unmeasured"
docs/l4_teacher_screen_results.md:1249-1253            perception | outer-quantile | REFUTED 4/4            "NEVER FLOWN (August arm void)"
docs/l4_teacher_screen_results.md:1262-1265            "Resolution is not the binding constraint in either  "...in the ACTION channel; the perception channel's quantile
                                                        channel"                                             lever is untested"
docs/l4_teacher_screen_results.md:934-935              "every outer-quantile candidate beats every          append "(outside% proxy only; the flown test of this proxy is
                                                        tilt-only ... stronger lever"                        void, see audit)"
docs/l4_teacher_screen_results.md:992-993, 999         "Perception was that constraint"; "if the            "perception was the candidate constraint; untested at q"
                                                        outer-quantile arm moves the floor"
docs/controller_reassessment_2026_08.md:8              "pending the outer-quantile arm"                     "...arm, which turned out void (audit 09/11)"
docs/controller_reassessment_2026_08.md:224            outer-quantile REFUTED 4/4                           "VOID (flag dropped by NEURONS/MEMORY/report fits)"
docs/controller_reassessment_2026_08.md:239-241        freeze precondition MET, "settled as a refutation"   "met on the refit null only; quantile question open"
docs/controller_reassessment_2026_08.md:248-249        "Perception coverage failed the same way from two    "from one direction (refit, a null); the quantile direction
                                                        directions"                                          was never flown"
docs/controller_reassessment_2026_08.md:257-258        "L3 and outer-q are unaffected"                      "L3 is unaffected; outer-q has no valid result"
docs/controller_reassessment_2026_08.md:272-274        "Outer-q closed: REFUTED 4/4; outside% dead"         add a dated 09/11 decision-log line voiding it
scripts/e2_delta_gamma_chain.sh (header comment)        "In the outer-q arm GRID ... swung 96.8% to 31.0%"   cite the c30-control GRID spread instead
memory/project_thermometer_regime_mismatch.md           "capability handicap on every arm"; "Perception was  "NOT yet flown" is stale; the only flown same-code comparison
  (body + description)                                  it"; "NOT yet flown"                                 (OQ arm, read correctly) had tilt-5 LOSE to tilt-30 4/4 cells,
                                                                                                             n=2 seeds, CI straddles 0, confounded; provisional
memory/MEMORY.md:49                                    "Thermometer regime mismatch FIXED — encoder ~6x     "...; tilt-5 default never shown better in flight (OQ arm
                                                        mis-scaled"                                          void, 09/11 audit)"
```

`docs/ctrl_dres_vs_dhat.md:64` and `memory/project_stable_gap_to_mpcof_100.md:19` propose
`--threshold-outer-quantile 0.02` from an OFFLINE saturation measurement. That reasoning does not depend on
the August runs and stands. No other memory file cites the outer-q, E1 or E2 verdicts.

## 5. Is a re-test warranted, and does `rf_c200_dres_oq02` answer the August question?

**It answers a DIFFERENT, cleaner question.** `rf_c200_dres_oq02` (block 3, s2/s3 on fixed code; s1
re-flown on ABI 45 before block 4) adds ONLY `--threshold-outer-quantile 0.02` to `rf_c200_dres`. Both arms
calibrate at the flown tilt (no `--threshold-calib-tilt`), with `--calib-episodes 200`, an mpcof teacher,
translation on, RF axis and the dres feature. It is the "isolate the quantile from the tilt" arm that the E1
pre-registration said was needed (l4:1031-1035) and that August never flew. The August question was the
PACKAGE (tilt 5 + q) against legacy tilt 30, on the attitude-only lqi recipe. oq02 does not answer that
question, and that question is not worth re-flying: the recipe lineage is obsolete, and its two
ingredients are better asked separately.

**Reading oq02 correctly:**
- Pair on the same seed against `rf_c200_dres`, headline to headline, n=3 seeds.
- Block 3 runs on ABI 45, i.e. the pre-ABI-47 GPU scorer with translation ON (25 distinct translation
  starts). Re-score BOTH arms with `scripts/ctrl_rescore_abi47.py` before comparing, so pre-fix and
  fixed-scorer rows are not mixed.
- At n=3 a 0.15° null band cannot be resolved (refit SD was ~0.3°). An effect that is not unanimous with a
  CI clear of 0 is provisional.
- q is GLOBAL (every thermometer), so a loss cannot be blamed on the residual feature alone.

**Is a further re-test warranted now?** For the quantile: no. oq02 is the re-test, so wait for it. The
question the August data actually bears on is the CALIBRATION TILT, and it is open. The current default
(tilt 5) was never shown better in flight, and the only same-code comparison leans against it (provisional,
confounded). The smallest experiments that settle it, cheapest first:

1. **Offline, no run:** compute `outside%` (integral channels and all features) for the current recipe's
   ladder at the default tilt vs `--threshold-calib-tilt 30`. The same offline check, done for the August
   tilt-5 default-q disturbance fit, also tells whether the August flown order (c30 < tilt5) is CONSISTENT
   with the outside% proxy instead of contradicting it. Either way the proxy stops being "dead" and becomes
   "untested".
2. **If (1) shows a material coverage difference:** one arm `rf_c200_dres + --threshold-calib-tilt 30`,
   s1-s3, paired against the same-seed `rf_c200_dres` on the ABI-47 scorer, interleaved seed-major like
   block 4. Falsifier: CI of the paired headline-steady Δ lies above 0 (tilt 30 worse) or straddles 0
   (keep the default).
