# CTRL-48 block 10 2×2: why do the arms hurt attitude? PARTIAL diagnosis from s1 (n = 1, 15/11/2026)

**Status: partial, n = 1.** Every claim below is about seed 1 only and carries a confidence label.
- TEST numbers are quoted descriptively and were never used to pick anything.
- The new measurements are VAL replays (20 val units 163226054..073).
- Nothing shipped was changed.

Runs: `logs/controller/ctrl48/C-S2-d_wposx2_a51_rf_c200_dc_biasin_ms01_P_both_L1_{,L2y_,L3_,L2y_L3_}full_s1.out`. A = `both_L1` s1.

## 1. Is the attitude loss real, or a seed-1 selection draw?

Pool[0] of every stage, TEST (35 units; err° / steady°). Source: the `[test-seeds] <STAGE> MULTI-SEED` rows, .out lines 506 / ~1125 / ~1665, headline ~4270.

```
stage pool[0]      | L1 (A)      | L1+L2y      | L1+L3       | L1+L2y+L3
-------------------+-------------+-------------+-------------+------------
GRID               | 1.12 / 0.53 | 1.24 / 0.69 | 1.20 / 0.55 | 1.29 / 0.78
NEURONS            | 1.04 / 0.43 | 1.25 / 0.70 | 1.16 / 0.57 | 1.27 / 0.71
MEMORY             | 1.04 / 0.43 | 1.25 / 0.51 | 1.13 / 0.46 | 1.25 / 0.58
HEADLINE (crowned) | 1.04 / 0.43 | 1.14 / 0.50 | 1.19 / 0.65 | 1.26 / 0.67
                   | NEURONS#0   | MEMORY#1    | GRID#2      | NEURONS#1
```
The same picture on VAL. Stage tables at .out lines 3838 / 3836 / 3831 / 3839 give pool #0 val err / steady per stage:
- L1: 1.16/0.56 → 1.07/0.45 → 1.07/0.45.
- L3: 1.27/0.60 → 1.12/0.53 → 1.11/0.44.
- L2y: 1.38/0.83 → 1.35/0.78 → 1.29/0.57.
- L2y+L3: 1.39/0.87 → 1.38/0.81 → 1.27/0.60.

- **L2y: the loss is in every stage, from the grid onward, on VAL and on TEST. It is not a selection draw (likely real for s1).**
  - The grid stage is where the gap opens (+0.12 err / +0.16 steady on TEST, +0.22 / +0.27 on VAL vs L1's grid #0). At the grid the GA has not yet searched anything, so the cause is the input, not the search.
- **L3, steady: mostly a SELECTION effect (likely).**
  - At MEMORY pool[0] the steady gap vs L1 is only +0.03° on TEST, and 0.44 vs 0.45 on VAL (none).
  - The crowned headline's +0.22° comes from stage-select crowning GRID#2 (val steady 0.63) over MEMORY#2 (val steady 0.44).
- **L3, err: a smaller but consistent cost at every stage (possible-to-likely).** +0.08 / +0.12 / +0.09° on TEST and +0.05° on VAL at NEURONS / MEMORY. Steady is unchanged at MEMORY, so the extra error sits in the transient, the first 80 % of the episode.
- L2y+L3 reads as roughly L2y's input cost plus L3's err cost; with n = 1 no interaction can be read.

## 2. L3: where does the attitude cost come from?

**(c) Fitness and stage-select: LIKELY, for the steady loss of the crowned genome.** Recomputing the stage-select ZRank terms from the VAL stage table (`scripts/ctrl48_stage_select_fit_terms.py`) reproduces the logged fit to ≈ 0.03 for the decisive rows:
- GRID#2: fit −0.08 = reward −0.23 + steady +0.27 + pos −0.12.
- MEMORY#2: fit +0.30 = reward 0.00 + steady −0.24 + **pos +0.545**.

The pos column decides it. Under L3 the nine candidates' VAL pos spans only 0.307-0.313 m, so the robust scale (1.4826·MAD) is ≈ 1-2 mm. MEMORY#2's 0.005 m worse pos then clamps at +3σ (× 0.1818 weight = +0.545). That outweighs a 0.19° better val steady.
- Both genomes rank R = 4 on the union rank (fit rank + p99 tail rank). The tie goes to the lower fit, which is GRID#2 (stage table, L1+L3 .out ~3846).
- The 0.005 m is not noise-free signal. The MEMORY stage raises val alt by 0.007-0.010 m in every run: L3 0.065 → 0.072-0.075 m, L1 0.130 → 0.146-0.147 m. That puts every MEMORY candidate at the pos-column clamp.
- The same clamp hit L1's MEMORY#1/#2 (+0.545). There it did not matter, because NEURONS#0 ≡ MEMORY#0 was best on attitude and pos alike.
- The err² column ranks reward = −Σ err² (the scorer's λ_pos is 0), which is dominated by the initial transient. GRID#2 has the best reward (−1.83) despite a worse mean err.
- Conclusion: under L3 the headline is chosen by a pos difference far below run-to-run noise. The MAD-floor guard (`zrank_mad_floor`, MEASURED_SCALE_FLOORS) has no pos entry and is off in this recipe anyway.

**(b) Stiffer collective labels eat attitude authority: POSSIBLE, for the err cost; not shown.**
- Motor jerk rises at every stage under L3: TEST 0.0024 → 0.0027-0.0030. The collective moves more.
- The teacher pays nothing for the same law (MPCOF comparator under pole3: err +0.009°, steady −0.015°; prereg "L3 reference rows"). So any cost is in IMITATING the stiffer collective, not in the physics.
- Not measured yet: whether the student's collective bank saturates its ±0.25 range or its 16 levels more often, and whether attitude-bank imitation error rises under L3. The VAL replay script refuses L3 (`ctrl_gap_student_trace.py` has no Python integrator), so this needs the Rust label replay with the L3 collective. That was not done in this hour.

**(a) The student's alt_err changed meaning (e_eff = 3e + ωI): POSSIBLE, entangled with (b).**
- With `--receptive-fields axis --rf-cross 0.2`, the attitude banks see alt_err only through cross draws, so the address shift reaches attitude only partly.
- The grid stage already shows the err cost (+0.08° TEST), which fits an input or label effect present from the first DAgger pass rather than a search effect.
- Cannot be separated from (b) without an arm that changes one and not the other.

**(d) Fewer cells or capacity for attitude: NOT CHECKED.** Cell counts per bank were not compared in this pass. Nothing points at it.

## 3. L2y: why no gain?

VAL replay of the headline genomes (`scripts/ctrl_gap_student_trace.py --trace-episodes 0`; the gate reproduced the scorer's steady on every unit). Decomposition by `scripts/ctrl48_2x2_yaw_decomp.py`. Hold window, 20 val units × 100 episodes, degrees:

```
                                          | L1 (A) NEURONS#0 | L1+L2y MEMORY#1
------------------------------------------+------------------+----------------
hold err (= steady)                       | 0.449            | 0.615
share of hold err²: roll / pitch / YAW    | .08 / .13 / .79  | .05 / .07 / .88
per-episode offset |mean|: roll/pitch/yaw | .08 / .11 / .33  | .11 / .13 / .51
within-episode sd: yaw                    | 0.20             | 0.32
ep-mean true yaw vs cresid_z: slope (R²)  | -0.47°/u (.33)   | -0.89°/u (.47)
yaw_err input: within-episode sd          | 0.20             | 0.39
ep-mean yaw_err_i: |mean|; corr with yaw  | 6.4; -0.53       | 20.9; +0.46
```

- **After L1, the remaining hold error is 79 % yaw (VAL).** That makes yaw the right target, but L2y made the yaw offset WORSE: 0.33 → 0.51°. The yaw offset is now more tied to the trim (slope −0.47 → −0.89° per unit of `cresid_z`). The student does not ride the trim better; it parks further from it.
- **Mechanism, LIKELY: L2y also redefined `yaw_err_i`.**
  - With L2y the integral integrates the trim-shifted error, which is not zero at the teacher's rest.
  - Its magnitude tripled (6.4 → 20.9) and its correlation with the true yaw offset flipped (−0.53 → +0.46). The integral now carries the trim, which `cresid_z` already gives, and the student lost the only input that integrated the yaw offset itself.
  - Round 2's spec flagged this exact risk: the integral's share of the label could not be measured offline.
- **The trim-shifted input is noisier, LIKELY a contributor.** Its within-episode sd is 0.39° vs 0.20° for the raw dead-reckoned yaw error. κ = 2.39° per unit carries the d̂-observer noise in `cresid_z` straight into the yaw input.
  - Gyro drift of the dead-reckoned heading is the same in both arms (≈ 0.16° episode-mean |heading − estimate| in A). Drift is not what changed.
- **Dense ladder, near-zero resolution: POSSIBLE, minor.** 23 % of L2y hold samples fall in the ±0.25° dead band, where the sign is lost. Under L1 the yaw ladder had a cut at 0 and ±0.42°.
  - The pre-check found dense ≈ quantile for the label share, so the ladder is not the main cause.
- **Why the pre-check did not catch it:** it scored ONE table on the proportional input, so it could not see that the integral changed meaning.

## 4. Fixes that could be flown later (named, not built)

1. **Stage-select pos column: give it a measured noise floor** (a `zrank_mad_floor` entry for pos, measured like the others), or score pos on `pos_hold`. At minimum, do not let a column whose candidate spread is below its noise clamp at ±3σ. This affects the headline of any arm with a compressed pos spread, which every L3 arm has.
2. **L2y′: shift only the proportional yaw input; keep `yaw_err_i` integrating the raw heading error.**
   - Note: the Metal side cannot do this with the L1 target-shift trick (it shifts every yaw use). It needs a change inside the cascade specialization, not in `derive_features`, to keep the default pipeline's compiled code. A design note, not a build.
   - A smoothed `cresid_z` (or the L1-style f32 combination with a low-pass trim) would also address the 2× input noise.
3. **L3 attribution arm (to separate a from b):** pole3 labels with the student fed the raw `alt_err` plus the integral as its own thrust-family input. Equally fine: a label replay of the L3 student on VAL with collective-bank saturation counted.

## What n = 1 cannot tell, and what the n = 3 interim look settles

- **n = 1 cannot tell:**
  - whether L2y's +0.07° headline steady and its grid-onward offset recur on other seeds (arm 1's paired SD ≈ 0.07°);
  - whether the L3 selection flip (GRID crowned over a better-steady MEMORY) recurs or was a seed-1 tie;
  - whether L3's err cost (+0.05-0.09°) is above seed noise.
- **The n = 3 interim look should settle:**
  - the sign of each arm's paired difference on steady and err, against the prereg's 99 % paired intervals;
  - whether stage-select crowns a worse-steady candidate under L3 on s2/s3 too. Read off the stage tables, not from TEST;
  - whether L2y's VAL yaw offset rises again. Rerun `ctrl_gap_student_trace.py` + `ctrl48_2x2_yaw_decomp.py` on the L2y headlines, VAL only.
- Nothing here changes the pre-registered rule or the 2×2. This is descriptive context for Luiz.

## Correction (round 3, 15/11/2026)

The re-score with the MEASURED pos noise floor (3.1 mm, CTRL-21 estimator; `docs/ctrl48_block10_levers_spec.md` round 3) does NOT change L1+L3 s1's pick. GRID#2 still wins the R = 4 tie on lower fit (−0.031 vs MEMORY#2 +0.061).
- The MEMORY candidates are 4.3-5.8 mm worse on VAL pos, which is 1.4-1.9× the measured noise. So §2(c) above overstated it.
- The clamp DID inflate the pos term (+0.545 → +0.306 with the floor). But the pick is a genuine transient + pos against hold-attitude trade on the run's own weights, not purely a sub-noise artifact.
- Only a floor of about 10 mm, not a measurement, flips it.
