# CTRL-73 — where the model-free base G+T still loses to MPCOF (25/11/2026)

**Partition: TEST** for every table in §1–§4: 35 CF2.1 units 99990101..99990135 × 100 episodes, report-grade.
Nothing here was tuned or selected on TEST. §5 uses **VAL units** (per-tick traces). It is a diagnosis and is
labelled as such; none of its numbers is a result or is compared with a TEST number.

**Arms.** All n = 10 (s1–s10), the published winners in `logs/controller/ctrl48/`:
- **G+T** (`..._a62_..._L1_yld_n0g3gz_full_sN`): the new model-free base, adopted 25/11 (CTRL-71/72 closed).
- **N0** (`..._yld_n0_full_sN`): s1–s5 `_a61_`, s6–s10 `_a62_`; s9 is the plain `_s9` tag.
- **G** (`_n0g3_`) and **T** (`_n0gz_`): context only.

**Rival.** MPCOF>rate(h10):
- **Row B** (est-fed + bias comp, the eq-info comparator) gives the 35-unit means and the hold per-axis split
  (`logs/controller/ctrl62/rival_rescore.log`).
- **Row C** (oracle-fed) stands in for B wherever per-unit, per-episode or whole-episode per-axis numbers are
  needed. It is within 0.002° of B on err and steady (as in CTRL-70).
- **Rival invariance check (new).** The rival does not depend on the WNN flags. Re-scored under the G+T s1 argv
  on 5 TEST units, row C is bit-identical to the CTRL-70 arrays computed under the N0 argv
  (`logs/controller/ctrl73/rival_check.py`). The CTRL-70 rival arrays are reused.

**Constraint (Luiz, 25/11).** Levers come from the WNN side only: its inputs (raw sensors and setpoints), its
encoding, thresholds and ladders, its connectivity and size, its training, its fitness and selection, and its
output decoding. No observer of any kind, and no MPCOF part (its model, predictions, observer, residuals, bias
compensation, cost or gains). Anything that only an observer or MPCOF could close is marked out of scope and is
not ranked.

## 0. Method (production scorers only, the CTRL-70 pipeline unchanged)

**WNN re-score.** `logs/controller/ctrl73/wnn_units.py` is CTRL-70's script, unchanged.
- Each winner's `best_genome` (the HEADLINE) is re-scored per unit on the CPU scorer (`ControllerEvaluator.score_genomes`).
- The captured call is re-issued with `axis_out=` (the per-axis split) and `episode_errs=`.
- Failed episodes are isolated by prefix differencing.
- Installed wheel: ram_controller ABI 62, flags off.
- 40 winners re-scored: 4 arms × 10 seeds.

**Gates.**
- **Re-issued call.** For every unit, the re-issued row equals the scored metrics, and the prefix episodes
  reproduce the full call's per-episode errs. The script asserts both. 40/40 passed.
- **HEADLINE reproduction.** Per-unit logged (in-flight GPU) HEADLINE blocks were compared with the CPU re-score
  (`ctrl70/cmp_logged.py`, output `ctrl73/data/cmp_logged.txt`).
  - 40/40 runs: the same HEADLINE genome.
  - GPU − CPU is +0.0017..+0.0039° on steady and +0.0008..+0.0030° on err, the same sign in 40/40.
  - This is exactly the CTRL-70 side finding (the GPU scorer reads ~0.003° high). 13/40 rows therefore print
    0.01 lower than the logged line at 2-dp precision. No other discrepancy.
- **Numbers in this doc.** All are CPU numbers. The rival rows are CPU-scored too.

**Correction to CTRL-70.** The scorer's "mid" slot (row slot 19) spans `ceil(0.15·steps)` to `0.8·steps`
(`cpu_score.rs` `MID_START_FRAC = 3/20`), which is **0.3–1.6 s** of the 2 s episode. CTRL-70 labelled it
0.15–1.6 s, and its "early" as 0–0.15 s. The correct windows are **early 0–0.3 s** and **mid 0.3–1.6 s**. The
CTRL-70 numbers are unchanged; only the labels were wrong.

**Phases.**
- hold = t ≥ 1.6 s (the steady window).
- pre-hold per axis = (err_axis − 0.2·hold_axis)/0.8. This is derived, and exact because every episode ran
  2000 steps.
- The rival has no mid slot.

## 1. Gap table (TEST, CPU, seed-mean over 35 units × 100 episodes)

```
                        G+T (n=10)            N0 (n=10)             MPCOF B (eq-info)   MPCOF C (oracle; per-axis stand-in)
stable %                99.99                 99.98                 100.00              100.00
fails (eps > 5 deg)     4 / 35000             7 / 35000             0 / 3500            0 / 3500
err deg                 1.023                 1.087                 0.691               0.690
  whole roll/pitch/yaw  0.470 / 0.428 / 0.589 0.479 / 0.440 / 0.653 n/a                 0.392 / 0.351 / 0.287
  pre-hold r/p/y        0.563 / 0.512 / 0.682 0.575 / 0.526 / 0.760 n/a                 0.466 / 0.419 / 0.340
  early 0-0.3 s (norm)  3.492                 3.607                 n/a                 n/a
  mid 0.3-1.6 s (norm)  0.681                 0.750                 n/a                 n/a
steady deg              0.284                 0.290                 0.165               0.163
  hold roll/pitch/yaw   0.101 / 0.090 / 0.219 0.093 / 0.092 / 0.227 0.094 / 0.083 / 0.078  0.095 / 0.083 / 0.076
pos m (3-D)             0.3406                0.3392                0.357               0.3571
alt m                   0.1312                0.1296                n/a                 0.1564
pos_hold m              0.2751                0.2718                n/a                 0.3093
```
Context rows, same layout: G = 1.092 err / 0.299 steady (whole yaw 0.671, hold yaw 0.236), 7 fails. T = 1.020 /
0.311 (whole yaw 0.573, hold roll/pitch/yaw 0.109 / 0.100 / 0.236), 5 fails. Per-axis cells are the mean
|rotation-vector component| in degrees. They decompose err and steady but do not add up to the norm.

**Reading.**
1. **Yaw is still the largest share of the err gap.** G+T minus C on whole |yaw| is +0.302°. Roll is +0.078 and
   pitch +0.077, so yaw is ~66 % of the axis excess (N0: +0.366 / +0.087 / +0.089, ~68 %).
   - **G+T moved the yaw gap:** whole yaw −0.064° vs N0 (lower on 10/10 seeds) and pre-hold yaw −0.078 (10/10).
     That is ~17 % of N0's yaw excess.
   - Roll and pitch barely moved (−0.008 / −0.012).
2. **The err gain is all transient.**
   - Vs N0: early −0.115 (10/10), mid −0.069 (10/10).
   - Hold yaw −0.008 (5/10). Hold barely moved.
3. **The steady gap is still entirely yaw.**
   - Hold yaw 0.219 vs 0.078 (B) = +0.141°.
   - Hold roll and pitch are +0.007 / +0.007 against B, at the rival's level, as in CTRL-70.
4. **It is still a mean shift, not a tail.** On the 600 paired episodes (6 units × 100, same episodes as the
   rival):
   - G+T is worse on 536/600. Median episode gap 0.289°, mean 0.304°.
   - The worst 1 % / 10 % of episodes carry 4 % / 27 % of the summed gap.
   - Pooled G+T episode err (3500 × 10 seeds): p50 0.99, p90 1.40, p99 2.03, max 7.34°.
5. **The gap still scales with the initial yaw offset, a little less steeply.**
   - Paired gap by |yaw0|: 0.20 / 0.22 / 0.28 / 0.38 / 0.44° for [0,1) / [1,2) / [2,3) / [3,4) / [4,5]°.
     N0: 0.24 / 0.28 / 0.36 / 0.46 / 0.52.
   - Per-episode slope of err on |yaw0|: G+T 0.094 °/°, N0 0.108, C 0.036.

## 2. Per unit (TEST): the motor-wear tail

c1 = asym·[+1, −1, +1, −1] is the yaw-torque wear contrast (as in CTRL-70). Over the 35 units its mean is −0.090
and its SD 0.209.

```
seed-mean per unit       G+T     N0      G       T       C
err gap vs C: min/med/max 0.200/0.334/0.443  0.293/0.393/0.614  0.276/0.406/0.615  0.243/0.332/0.433  -
units worse than C (err)  35/35   35/35   35/35   35/35   -
units worse than C (stdy) 33/35   34/35   33/35   34/35   -
err slope on |c1| (deg)  +0.329  +0.533  +0.586  +0.263  +0.016
signed hold yaw on c1    +0.316  +0.183  +0.258  +0.275  ~0      (deg per unit c1; r +0.91 / +0.81 / +0.91 / +0.89)
|seed-mean hold offset|   0.072   0.043   0.056   0.062   -
```
- **The wear tail is real for the WNN, not for the rival.**
  - The WNN's per-unit err rises with |c1|: +0.33 °/unit for G+T, +0.53 for N0, against +0.016 for C. The err
    gap correlates with |c1| at r +0.70 (G+T) and +0.86 (N0).
  - The worst err-gap units are high-wear: u132 (c1 −0.42), u130 (−0.37) and u115 (−0.47), plus u123, the
    fail-prone unit.
  - G+T flattened this slope relative to N0 (T carries it). It did not remove it.
- **G+T almost DOUBLED the wear-proportional hold-yaw offset.**
  - The slope of signed hold yaw on c1 went from 0.18 (N0) to 0.32 °/c1 (r 0.91), and the mean |offset| from
    0.043° to 0.072°.
  - Expected offset share of hold yaw ≈ slope·mean|c1| = 0.316·0.196 ≈ 0.06° of G+T's 0.219 (N0: ≈ 0.036 of
    0.227).
  - Every yaw-authority change so far grew this offset: Y3 (0.55 °/c1), G (0.26), T (0.28), G+T (0.32). §5
    gives the mechanism.

## 3. The fails (TEST; 23 failed episodes over 40 runs)

```
arm  seed unit ep   err   whole r / p / y      mid(0.3-1.6)  hold |v|
G+T  s1   u123 16   7.34  7.07 / 0.18 / 1.00   8.14          0.43
G+T  s3   u123 16   5.93  5.74 / 0.18 / 0.85   5.41          0.49
G+T  s4   u118  1   5.26  5.14 / 0.28 / 0.36   5.17          0.59
G+T  s8   u112 86   6.39  0.35 / 6.28 / 0.38   5.82          0.18
```
(All 23 are in `ctrl73/data/units_eps.txt`: N0 7, G 7, T 5, G+T 4.)

- **Same nature as CTRL-70.** None diverged; every one ran 2000 steps and recovered by the hold. Each is a
  single-axis roll or pitch excursion in the mid window, not a yaw event.
- **Same episodes.**
  - u123 ep16 fails in 8/40 runs and u112 ep86 in 7/40, across every arm.
  - Both are among the rival's own hardest episodes (CTRL-70 §3: extreme collective, mass or xy draws).
  - u118 ep1 (G+T s4) is the only new fail episode.
- **G+T has the fewest fails** (4 vs N0 7) but the same profile. Neither yaw lever touched the roll/pitch
  mid-window excursion.

## 4. Did G+T trade anything? (TEST, paired same seed, n = 10)

```
                 G+T value (mean)  N0 value (mean)   G+T lower on
err              1.023             1.087             10/10
 whole yaw       0.589             0.653             10/10
 pre-hold yaw    0.682             0.760             10/10
 mid (0.3-1.6)   0.681             0.750             10/10
steady           0.284             0.290              6/10
 hold yaw        0.219             0.227              5/10
 hold roll       0.101             0.093              3/10
 hold pitch      0.090             0.092              5/10
 |signed hold yaw| (wear offset)  0.081  0.063    2/10
pos              0.3406            0.3392             4/10
alt              0.1312            0.1296             1/10
pos_hold         0.2751            0.2718             1/10
jerk             0.002439          0.002522           9/10
fails            4                 7
```

**Why steady is only 6/10.** Two yaw effects cancel, plus a small roll cost:
- **(a) The hold-yaw chatter shrank.** VAL diagnosis (§5): the within-episode hold yaw SD fell from 0.233 to
  0.195°, and the hold yaw-rate setpoint sign flips fell from 2.4/s to 1.0/s.
- **(b) The wear-proportional yaw offset grew** (§2, slope 0.18 → 0.32 °/c1; |offset| larger on 8/10 seeds).
  Net hold yaw −0.008 (5/10).
- **(c) Hold roll is +0.008 (3/10).**
  - This comes from T: T alone has hold roll +0.017 (0/10) and hold pitch +0.008 (1/10). G partly undoes it.
  - **Mechanism not identified.** The hypothesis is that, under `--receptive-fields axis --rf-cross 0.2`, some
    roll/pitch neurons address gyro_z bits. The new ladder turns gyro_z from a near-sign bit (62 % of hold ticks
    beyond the old outer cut) into a 4-bin code that is occupied 14 / 42 / 41 / 3 % in the hold. That would
    spread their hold writes over more addresses. This is untested; see the lever-2 pre-check.

**Why pos is +1.4 mm.**
- It is vertical and in the hold: alt +1.6 mm (G+T higher on 9/10 seeds), pos_hold +3.3 mm (higher on 9/10).
- It is diffuse: pos_hold is higher on 30/35 units.
- It is **not** wear-coupled: r(|c1|, Δalt) −0.34, and r(Δalt, Δ|offset|) −0.35.
- T alone: alt +0.7 mm (5/10). G alone: −0.3 mm.
- **Mechanism not identified.** The candidate is the same cross-wiring as (c) on the thrust bank. It is 0.5 % of
  pos and does not change the adoption ruling, but it is a whole-vector "not worse" item to watch on the next
  lever.

## 5. Mechanism (VAL diagnosis only — G+T s1 vs N0 s1 winners on the same 3 val units × 100 episodes)

`ctrl73/trace_val.py` (`ctrl_gap_student_trace.replay`), gated per unit on the scorer's steady (|Δ| ≤ 0.0007°).
Scripts: `trace_cmp.py`, `trace_hold.py`, `trace_leak.py` and `trace_yawsat.py`.

```
                                           N0 s1                G+T s1
mean |yaw| by window 0-.15/.15-.3/.3-.6/.6-1/1-1.6/hold
                                           2.21/1.79/1.25/.54/.37/.30   2.29/1.29/.93/.54/.25/.23
|gyro_z| beyond its outer cut, % ticks     90/91/84/73/66/62    6.5/9.1/0/0/0/0
|gyro_x| beyond +-0.069 rad/s, % ticks     90/73/44/14/5/2      91/75/49/18/4/2
|gyro_y| beyond +-0.093 rad/s, % ticks     86/73/38/7/1/0       84/63/34/7/1/0
max |yaw| in 0.3-1.6 s: median / p90      1.84 / 4.61          1.24 / 2.23
overshoot by |yaw0| [0,1.5)/../[3.5,5]     1.32/1.80/2.58/3.18  1.11/1.30/1.47/1.68
hold: within-episode yaw SD                0.233                0.195
hold: yaw-rate setpoint sign flips /s      2.40                 1.00
corrective yaw-rate cmd (rad/s) at |yaw| 1.5-2 / 2-2.6 / 2.6-3.5 / 3.5-5 deg
                                           .40/.46/.64/.72      .37/.38/.56/.57
```

- **The yaw-rate input is fixed.** T did what CTRL-71 aimed at: gyro_z is resolved, the overshoot halved (p90 4.6
  → 2.2°) and no longer scales steeply with |yaw0|. What remains of the yaw transient gap is not rate blindness.
- **The roll/pitch rate inputs are still saturated in the transient,** unchanged from N0: gyro_x is beyond its
  outer cut on 91 / 75 / 49 % of ticks over 0–0.6 s (gyro_y 84 / 63 / 34 %). This is the same condition gyro_z
  was in before T, on the axes that still carry +0.08° each of err, and that carry every fail.
- **The yaw_err code saturates above 2.6°.** G+T's corrective command is flat from 2.6–3.5° to 3.5–5° (0.56 →
  0.57 rad/s), and nearly flat from 1.5 to 2.6°. 43 % of the 0–0.15 s ticks sit beyond the ±2.62° outer cut.
- **The hold floor is a leaky trim that is rebuilt every action.** This is the main new finding.
  - In the hold, the WNN holds a **unit-specific mean yaw-rate setpoint** of +0.18 / −0.14 / +0.35 rad/s on the
    three VAL units, while the mean gyro_z is ≈ 0 (|·| ≤ 0.004).
  - **Why a trim is needed (for `flight-dynamics` to confirm).** The CF2.1 firmware yaw-rate PID is kp 120,
    ki 16.7 (`airframe.py`): Ti = kp/ki ≈ 7 s, against a 2 s episode. Inside an episode the rate loop is
    effectively P-only on yaw, so a motor-wear yaw torque needs a persistent rate-setpoint offset.
  - **Why it chatters.** The CTBR delta law leaks every bank, yaw included, toward 0 by `--delta-leak 0.7` per
    100 Hz action (`controller.rs` `leaked_baseline`). Holding a trim s therefore needs a fresh delta of 0.3·s
    on **every** action.
    - The traces show exactly that: the implied per-action yaw delta (sp_k − 0.7·sp_{k−1}) has mean +0.054 /
      −0.042 / +0.104 rad/s on the three units, = 0.3 × the held trim to the 3rd decimal (both arms).
    - Its per-action SD is 0.05–0.10 rad/s, as large as the trim itself.
    - The held setpoint therefore fluctuates (within-episode SD 0.16 rad/s), with a ~2.8–2.9 Hz hold
      oscillation. That is the hold-yaw chatter.
    - The wear offset is the part of the trim the WNN does not rebuild, which explains why it grows with c1.
  - **Why yaw-authority changes grow the offset (computed, not measured).** Under gamma 3 at a yaw delta range of
    1.0 rad/s (0.5 × 2.0), a 0.04–0.10 rad/s per-action trim sits at decoded |t| = 0.34–0.46, a third to a half
    of the yaw bank's code range on every hold tick (gamma 2: 0.20–0.32). Y3 (a smaller yaw step) and G (a
    steeper curve) both push the needed trim further up the code range, and both grew the offset.
- **The rival is not traced per tick.** The rival's yaw-wear handling is its bias compensation (row B), which is
  out of scope for the WNN by the 25/11 rule. The WNN-side analogue is to make the trim cheap to HOLD, not to
  estimate the wear.

## 6. Levers, ranked by expected payoff on err/steady (WNN side only; no observer, no MPCOF part)

Costs: a flight is ~55–65 min. n = 5 means 5 flights (~5 h), n = 10 means ~10 h. Pairs go against the G+T
controls already flown on s1–s10. A new arm is a new recipe, so no recipe repeats on a seed.

### L1 — yaw-bank output leak (`--delta-leak-yaw`, e.g. 0.95): hold the wear trim instead of rebuilding it

- **Hypothesis.** The hold yaw floor (chatter plus wear offset) and the wear-dependent err slope come from the yaw
  trim being rebuilt every 10 ms through a 0.7 leak. With a yaw-only leak of 0.95, the sustaining delta falls
  from 0.3·s to 0.05·s (0.04–0.10 → 0.007–0.017 rad/s; decoded |t| 0.19–0.26 under gamma 3). The trim persists,
  the per-action noise stops being re-injected every tick, and the small yaw steps are freed for correction.
- **Expected direction.** Steady lower (target: hold yaw 0.219 vs B 0.078; the chatter part plus the ~0.06°
  offset part); err lower on high-|c1| units (the err slope on |c1| is 0.33 vs C 0.016). Stable, fails and pos
  not expected to move. 0.95 rather than 1.0, because a pure integrator can run away (the flag's own help text);
  0.95 is the former global default.
- **Evidence.** §5 (mean delta = 0.3·trim per unit, per-action SD ≈ the trim, 2.8 Hz hold oscillation) and §2
  (offset slope 0.32 °/c1, r 0.91; err slope on |c1|).
- **Model-free check.** This is output decoding (the WNN's own accumulator). It adds no input, no estimate and no
  MPCOF quantity, and DAgger labels stay the existing teacher's, encoded through the same delta law.
- **Code.** This needs a **wheel change** (controller only, swap-free; ABI 63), mirroring `--delta-max-yaw`
  (ABI 57):
  - the per-bank leak in the CPU `leaked_baseline`;
  - the Metal `P.delta_leak` sites (`controller_rollout.metal` ~2057/2082/2098);
  - the DAgger label baseline;
  - the spec field, the flag and its validation;
  - a flag-off byte-identity pin and a CPU/GPU parity `cargo test`.
  Size it with `rust-code` (~1 day).
- **Cheaper flag-only variant.** Global `--delta-leak 0.95` (exists). It also changes the thrust, roll and pitch
  banks. Roll/pitch have a fast rate integral (ki 400), so they need no trim, and the collective has its own
  anchor. Riskier for alt/pos, and it re-opens a lineage decision. Second choice.
- **Cost.** About 1 day of code, then n = 5 (~5 h), extended to n = 10 (~10 h) on Luiz's word.
- **Pre-check (cheap, before flights).**
  - (1) Flag-off pin plus parity test.
  - (2) A fixed-genome memory-only DAgger re-train of the G+T s1 winner under yaw leak 0.7 vs 0.95, scored on
    the 3 VAL units with the same trace tools. Kill if VAL hold yaw is not lower, or if the hold setpoint SD
    does not fall. Whether a memory-only re-train entry point exists has not been checked; the fallback is a
    1-seed GRID+NEURONS short run on VAL. Both are VAL diagnoses, not results.

### L2 — roll/pitch rate ladder (`--gyro-xy-ladder dense`): T for the axes that still saturate

- **Hypothesis.** The roll/pitch transient excess (pre-hold +0.097 / +0.093 vs C) and the mid-window roll/pitch
  excursions behind every fail have the same rate-blindness T removed on yaw: gyro_x/y are beyond their outer
  cut on 91 / 75 / 49 % of 0–0.6 s ticks. T moved whole yaw −0.080 (10/10) with exactly this change on gyro_z.
- **Expected direction.** Err lower (roll/pitch pre-hold); fails possibly fewer (not registered). Steady flat,
  provided the hold resolution is kept.
- **Design constraint from this analysis.** Hold roll/pitch are at the rival's level, and T showed that changing
  a rate code can cost hold roll (+0.017, 0/10). The ladder must keep the current fine inner cuts (~±0.004–0.005
  and ~±0.013 rad/s from the quantile fit) and extend only the outer pairs, e.g. ±0.005 / ±0.02 / ±0.08 /
  ±0.30 rad/s. Cuts are chosen on VAL traces, never TEST.
- **Code.** **Python only** (no wheel). Extend the CTRL-71 `_dense_inner_for` / `ThresholdCalibration` path to
  `gyro_x` / `gyro_y`, with a flag-off pin as in CTRL-71. This creates a new threshold lineage.
- **Cost.** A few hours of code, then n = 5 / n = 10 flights as above.
- **Pre-check (cheap; the CTRL-71 gate format).** Through the implemented ladder on the G+T s1 VAL traces:
  - (a) Beyond-outer-cut share at most 25 % in every transient window, and at most 5 % in the hold.
  - (b) The hold bin occupancy of the inner pairs is not coarser than today's.
  - (c) A connectivity census of the 10 G+T winners: how many roll/pitch/thrust-bank neurons address gyro_z or
    gyro_x/y bits through `--rf-cross 0.2`. This tests the §4(c) hypothesis.
  - Kill if (a) or (b) fails.

### L3 — yaw-error outer ladder (extend `--yaw-centred-ladder dense` beyond ±2.62°)

- **Hypothesis.** The yaw_err code saturates at ±2.62°. Above it, the corrective command is flat (0.56 → 0.57
  rad/s), and 43 % of the 0–0.15 s ticks are there. The remaining yaw-transient gap still grows with |yaw0|
  (0.20 → 0.44°), and the WNN cannot tell 3° from 5°. One more outer pair at ~±4–5° (the IC draw reaches 5°)
  would let the yaw bank scale its early command.
- **Expected direction.** Err lower (yaw transient, mostly in the |yaw0| > 2.6° third of the episodes); steady
  flat. Smaller than L1 and L2: the early window is short, and the slope is now 0.094 vs C 0.036.
- **Code.** **Python only.** The 8-bit ladder is ±0.10 / ±0.25 fixed plus 2 quantile pairs. Either replace the
  outer quantile pair with a fixed ±4.5°, or give yaw_err its own outer quantile. Flag-off pin. New threshold
  lineage.
- **Cost.** Hours of code, then n = 5 / n = 10 flights.
- **Pre-check.** On VAL traces through the implemented ladder: the share of 0–0.3 s ticks beyond the new outer
  cut is at most 10 %, and the hold occupancy of the ±0.10 / ±0.25 inner bins is unchanged. Kill otherwise.
  Weak lever: fly it only after L1 and L2, or fold it into L2's lineage if Luiz wants a single threshold-lineage
  change.

### Not ranked

- **Hold roll/alt cost of the gyro_z code** (§4 (c) and pos). This is a diagnosis to run first, not a lever: the
  L2 pre-check (c) census. If the census shows the cross-wired gyro_z bits drive it, a receptive-field exclusion
  could follow. That is not proposed until measured.
- **Out of scope (needs an observer or MPCOF).**
  - Estimating the motor-wear yaw torque and feeding or compensating it. This is what the rival's row-B bias
    compensation does, and it is the cleanest way the rival's hold stays offset-free.
  - Any d-hat or residual input.
  - Any MPCOF prediction, cost or gain used as a feature or label beyond the existing teacher labels.
  - The WNN-side analogue that remains in scope is L1: make the trim cheap to hold, without estimating it.
- **Not levers (measured).** Tail-only fixes (the top 1 % of episodes carry 4 % of the gap). Hold roll/pitch
  beyond the T regression (they are at the rival's level). The gyro_z ladder itself (spent; it works).

## 7. Not measured

- Row-B per-unit, per-axis and per-episode numbers (C stands in).
- The rival's mid-window split and its per-tick traces.
- Saturation inside TEST episodes (traces are VAL only, by design).
- The census of rf-cross connectivity to gyro bits: this is L2 pre-check (c).

## Files

- Scripts: `logs/controller/ctrl73/` (`wnn_units.py`, `run_all.sh`, `analyze.py`, `units_eps.py`, `alt_check.py`,
  `rival_check.py`, `trace_val.py`, `trace_cmp.py`, `trace_hold.py`, `trace_leak.py`, `trace_yawsat.py`, `thr.py`).
- Data: `logs/controller/ctrl73/data/` (40 per-run `.npz`, re-score logs, `cmp_logged.txt`, `analyze.txt`,
  `units_eps.txt`).
- The 62 MB VAL trace stays in /tmp (not committed). It is reproducible with `trace_val.py <G+T s1 tag> <out> 3`.
