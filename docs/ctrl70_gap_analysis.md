# CTRL-70 — where the model-free WNN loses to MPCOF eq-info (23/11/2026)

**Partition: TEST** — 35 CF2.1 units 99990101..99990135 × 100 episodes (report-grade; nothing here was tuned
on, and nothing below was selected on it). One section (§5) uses VAL units for a per-tick trace; it is
labelled, it is a diagnosis, and none of its numbers is a result.

Arms: **N0** = `..._yld_n0_full_sN` (s1–s8), **Y3** = `..._yld_n0dmy03_full_sN` (N0 + `--delta-max-yaw 0.3`, s1–s7).
Rival: **MPCOF>rate(h10)**. Row B (est-fed + bias comp, the eq-info comparator) is reachable only through the
cargo probe, which prints 35-unit means and the hold per-axis split only. Per-unit and whole-episode
per-axis numbers use **row C** (oracle-fed) as a stand-in for B. They match within 0.002° on err and steady:
C 0.690/0.163 vs B 0.691/0.165. Hold r/p/y: C 0.095/0.083/0.076 vs B 0.094/0.083/0.078
(`logs/controller/ctrl62/rival_rescore.log`).

## 0. Method (production scorers only, nothing reimplemented)

- **WNN.** Each published winner (`_winner.yaml.gz`, best_genome = the HEADLINE) is re-scored per unit with
  `ControllerEvaluator.score_genomes` on the CPU scorer. The exact `score_controllers_cpu` call is captured
  and re-issued with `axis_out=` (ABI 56 per-axis split) and `episode_errs=` (ABI 59 per-episode err). The
  row's slot 19 gives the mid window (15–80 %).
  - **Gates:** the re-issued row equals the scored metrics, and every run's 35-unit mean reproduces its
    logged HEADLINE line at print precision (15/15).
- **Failed episodes.** Each one is isolated by **prefix differencing**: the same call on the first k+1 and
  first k episodes, so that episode's own per-axis and phase numbers are exact.
  - **Gate:** the prefix episodes are bit-identical to the full call's per-episode errs.
- **Rival.** `classical_baseline.teacher_metrics(axes=True)` is the path the runs print. It reproduces rows
  C and A exactly (0.690/0.163/0.357 and 1.352/0.684/0.341). Per-episode rival errors on 6 units use the same
  prefix differencing; the full prefix equals the full call.
- **Phases.** hold = t ≥ 1.6 s (the steady window). pre-hold = (err_axis − 0.2·hold_axis)/0.8 is DERIVED.
  It is exact here because every episode, the fails included, ran its full 2000 steps. The WNN also has the
  scorer's mid window (0.15–1.6 s). The rival scorer has no mid slot, so its early/mid split is **not measured**.
- **Scripts and data:** `logs/controller/ctrl70/` (`wnn_units.py`, `rival_units.py`, `rival_eps.py`,
  `analyze.py`, `units_corr.py`, `c1.py`, `eps.py`, `ics.py`, `ic_corr.py`, `trace_val.py`, `trace_an.py`,
  `gyro.py`, `data/*.npz`).

**Side finding (affects absolute numbers, not the A/B).** On all 15 winners, the in-flight GPU scorer
(logged blocks) reads **+0.0017 to +0.0036° steady and +0.0013 to +0.0032° err** above the CPU scorer on the
same genome, units and episodes. Same sign 15/15, both arms. Example, Y3 s1 unit 99990101: GPU steady
0.2021 vs CPU 0.1972. The rival rows are CPU-scored, so the logged WNN-vs-MPCOF gap is ~0.003° pessimistic
for the WNN. This is a CPU/GPU parity question for `rust-code`/QA. The tables below are CPU numbers.

## 1. Gap table (TEST, seed-mean over 35 units × 100 episodes; fails = failed episodes / total)

```
arm            n | stable  fails    | err   | err  r / p / y        | pre-hold r / p / y   | steady | hold r / p / y        | pos m
---------------+--+-----------------+-------+-----------------------+----------------------+--------+-----------------------+------
N0 (CPU)       8 | 99.98   6/2800   | 1.087 | 0.476 / 0.438 / 0.657 | 0.571 / 0.523 / 0.762| 0.299  | 0.094 / 0.094 / 0.235 | 0.339
Y3 (CPU)       7 | 99.98   6/2450   | 1.098 | 0.459 / 0.405 / 0.709 | 0.554 / 0.486 / 0.842| 0.233  | 0.081 / 0.081 / 0.176 | 0.339
MPCOF C (orc)  - | 100.00  0/3500   | 0.690 | 0.392 / 0.351 / 0.287 | 0.466 / 0.419 / 0.340| 0.163  | 0.095 / 0.083 / 0.076 | 0.357
MPCOF B (eq)   - | 100.00  0/3500   | 0.691 |   (probe: n/a)        |   (n/a)              | 0.165  | 0.094 / 0.083 / 0.078 | 0.357
MPCOF A (est)  - | 100.00  0/3500   | 1.352 | 0.836 / 0.778 / 0.289 | 0.934 / 0.867 / 0.342| 0.684  | 0.443 / 0.419 / 0.079 | 0.341
```
Per-axis cells are the mean |rotation-vector component| (body frame, vs the scorer's own tilt reference), in
degrees. They decompose err and steady, but they do not add up to the norm.
WNN time split (norm): early 0–0.15 s = 3.60° (N0) / 3.64° (Y3), mid 0.15–1.6 s = 0.748 / 0.778, hold = steady.

**Reading.**
1. **Yaw carries most of the err gap.** Whole-episode |yaw| is 0.657 vs 0.287 (+0.370°). Roll and pitch are
   +0.084 and +0.087.
2. **Hold roll/pitch are at the rival's level** (0.094 vs 0.095/0.083). **The steady gap is entirely yaw:**
   hold yaw 0.235 vs 0.076 for N0, 0.176 for Y3.
3. **The roll/pitch gap is transient-only:** pre-hold +0.10, hold ≈ 0.
4. **It is a mean shift, not a tail.**
   - On 6 units × 100 paired episodes (the same episodes for WNN and rival), the WNN is worse on
     564/600 (N0) and 545/600 (Y3). Median episode gap 0.36 / 0.33° ≈ the mean gap 0.37°.
   - The worst 1 % of episodes carry 4 % of the summed gap, and the worst 10 % carry 23–27 %.
   - Pooled WNN episode err (N0, 2800 episodes): p50 1.04, p90 1.51, p99 2.18, max 7.95°. Dropping the top
     1 % moves the mean only 1.087 → 1.071.
   - Every unit is worse than C on err (35/35). The per-unit err gap is 0.29–0.61° for N0.
5. **The gap scales with the initial yaw offset.** On the 600 paired episodes, binned by |yaw0|:
   - gap 0.24° at |yaw0| < 1°, 0.28 / 0.36 / 0.46, and 0.52° at 4–5°.
   - Per-episode slope of err on |yaw0|: WNN 0.110 °/° (N0) and 0.095 (Y3), vs C 0.036.
   - Of the IC features, |yaw0| is the strongest correlate of the paired gap: r = 0.39 (N0) and 0.32 (Y3).
     The next is |xy0| at 0.14–0.15.
   - Rough split: about half the err gap follows the initial yaw offset, and about half is an IC-independent
     floor (~0.24°).

## 2. Per unit: motor wear explains the yaw OFFSET, not the gap

c1 = asym·[+1, −1, +1, −1] is the 4-motor contrast that tracks yaw torque. It is identified empirically: it is
the contrast the WNN's signed hold yaw follows. Which physical mixer sign this is goes to `flight-dynamics`.
Over the 35 test units, c1 = −0.090 ± 0.209 and mean |c1| = 0.196.

- **Signed hold yaw tracks c1** (seed-averaged, Pearson r over units):
  - N0: r +0.79, slope **+0.18° per unit c1**.
  - Y3: r +0.89, slope **+0.55° per unit c1**.
  - C: r +0.04, slope +0.01 (offset-free).
  - The WNN therefore keeps a constant, wear-proportional yaw offset in hold, and **Y3's is 3× N0's**. The
    per-unit offset is much more repeatable across Y3 seeds (pairwise r 0.79 vs 0.35; mean |offset| 0.104 vs
    0.043°).
  - Expected offset share of hold yaw ≈ slope·mean|c1|: ~0.11° of Y3's 0.176, ~0.035° of N0's 0.235.
- **Hold |yaw| grows with |c1| for everyone,** the rival included (C r +0.95, N0 +0.55, Y3 +0.59).
- **Signed hold roll/pitch follow the roll/pitch contrasts** for the WNN (|r| 0.5–0.7), not for C. The
  magnitudes are small (~0.02–0.05°).
- **Knock-out counterfactual** (N0 s1, Y3 s1, C; same 35 units with one L4C channel zeroed). The WNN gap does
  NOT come from the disturbances:

```
                     N0 s1 err/steady   Y3 s1 err/steady   C err/steady   err gap N0 / Y3   hold yaw N0 / Y3 / C
as flown (L4C)       1.110 / 0.336      1.049 / 0.222      0.690 / 0.163  0.420 / 0.359     0.272 / 0.165 / 0.076
motor wear off       0.839 / 0.248      0.741 / 0.170      0.424 / 0.022  0.415 / 0.317     0.196 / 0.124 / 0.032
torque jitter off    1.101 / 0.333      1.042 / 0.220      0.688 / 0.163  0.413 / 0.354     —
sensor noise off     1.134 / 0.347 (4 fails) 1.052 / 0.230 (1) 0.686 / 0.156  0.448 / 0.366  —
```
- **Motor wear off:** the rival's steady falls to 0.022°. Its hold is wear-driven.
  - The WNN keeps a 0.17–0.25° floor (hold yaw 0.12–0.20, hold roll/pitch 0.06–0.10 vs 0.013), and the err
    gap is unchanged.
  - **The WNN's hold floor is its own control law,** not a robustness deficit.
- **Torque jitter off:** no change for either.
- **Sensor noise off:** this makes the WNN slightly WORSE (out of its training distribution; noise acts as
  dither, as found before).

## 3. The fails (12 episodes over the two arms; all at TEST)

```
arm seed unit       ep  err°  whole r / p / y      mid(0.15-1.6s)  hold |v|
N0  s1  99990123   16  7.95  7.79 / 0.18 / 1.00   7.86            0.56
N0  s1  99990123   33  5.22  5.05 / 0.17 / 0.61   5.24            0.36
N0  s4  99990112   86  5.88  0.33 / 5.72 / 0.67   6.56            0.79
N0  s5  99990134   73  5.92  5.27 / 1.66 / 0.74   5.78            0.23
N0  s7  99990134   86  5.44  5.10 / 0.50 / 1.12   6.17            0.55
N0  s8  99990123   16  5.47  5.31 / 0.22 / 0.80   4.79            0.23
Y3  s1  99990123   16  5.33  5.03 / 0.21 / 1.26   5.04            0.69
Y3  s2  99990123   16  5.67  5.51 / 0.15 / 0.78   4.70            0.28
Y3  s3  99990112   86  5.60  0.25 / 5.43 / 0.54   5.25            0.25
Y3  s4  99990112   86  5.29  0.23 / 5.08 / 0.76   5.09            0.25
Y3  s7  99990121   14  5.78  5.71 / 0.15 / 0.51   7.03            0.48
Y3  s7  99990123   16  5.03  4.77 / 0.19 / 1.29   5.04            0.25
```
- **Nature of the fails.** None diverged; every one ran 2000 steps and recovered by the hold (hold |v| 0.23–0.79°).
  - Each is a **single-axis roll or pitch excursion in the mid window** (0.15–1.6 s, mid 4.7–7.9°), not a
    yaw event and not the initial recovery.
  - Saturation (pwm or rate) is **not measured**. The TEST-unit per-tick trace is refused by design, and
    the scorer exports no saturation counter.
- **The same episodes fail across seeds and arms.**
  - u123 ep16 fails 5 times (N0 s1, s8; Y3 s1, s2, s7). On every winner it is 2.9–7.95°; the rival flies it
    at 1.41°.
  - u112 ep86 fails 3 times. u134 ep73 and ep86, u123 ep33 and u121 ep14 fail once each.
  - u112 ep86 and u134 ep73 are the **rival's own hardest episodes** in their units (rank 1 and 2 of 100;
    C 2.15° and 2.29°). u123 ep16 is the rival's rank 11. **The fails are hard episodes that the WNN
    amplifies,** not WNN-only accidents.
- **What they share (initial conditions, percentile of |x| over the 3500 TEST episodes).**
  - Extreme vertical or horizontal draws: collective fraction p99–p100 (u123 ep16, u112 ep86), mass p97–p98
    (u112 ep86, u134 ep73), xy offset p96–p99 (u134 ep73, u121 ep14), z0 p97 (u112 ep86).
  - The fails sit on the tails of the translation-task draw. A plausible but **unverified** reading: the
    roll/pitch excursion is the WNN's position response departing from the scorer's tilt reference. The
    WNN's pos is better than the rival's (0.339 vs 0.357 m). This goes to `flight-dynamics`.
- **N0 vs Y3:** 6 vs 6 fails. Two episodes fail under both arms (u123 ep16, u112 ep86); N0 adds u123 ep33
  and u134 ep73/ep86, and Y3 adds u121 ep14.
  - The N0-only fails u134 ep73/ep86 run 1.76–3.94° on every Y3 winner.
  - The Y3-only fail u121 ep14 runs 1.24–2.71° on the N0 winners.
  - The axes are the same. Y3 does not change the fail profile.

## 4. N0 vs Y3 (paired s1–s7, CPU, TEST)

```
Y3 - N0          mean     Y3 lower on
steady          -0.066    7/7     <- the Y3 win
 hold yaw       -0.060    6/7     <- carries it (plus hold roll -0.013 7/7, hold pitch -0.012 6/7)
err             +0.010    2/7
 whole yaw      +0.050    0/7     <- what Y3 pays
 pre-hold yaw   +0.077    0/7
 mid window     +0.028    1/7
 whole roll     -0.018    6/7
 whole pitch    -0.030    7/7
pos             +0.001    2/7
fails             6 vs 5 (s1-s7)
```
- **Y3 trades yaw transient for yaw hold.** The smaller yaw increment slows the yaw correction (pre-hold yaw
  worse on 7/7) and cuts hold-yaw chatter. In VAL traces (§5) the yaw-rate setpoint crosses zero 0.4 times/s
  in hold vs 2.4 times/s for N0.
- **Y3 also triples the wear-proportional yaw offset** (§2). Y3's remaining hold yaw is about two-thirds
  wear offset; N0's is mostly chatter.
- **Roll/pitch get slightly better under Y3** (whole err −0.018/−0.030). Net err is a wash (+0.010, 2/7).
- **For the CTRL-67/68 decision:** this is the A/B question restated. Y3 is "steady better on 7/7, err not
  better (2/7), stable/fails equal". No per-axis signal says Y3 is worse elsewhere beyond yaw transient.

## 5. Mechanism (VAL diagnosis only — 3 val units × 100 episodes, N0 s1 and Y3 s1; not a result)

`scripts/ctrl_gap_student_trace.replay`, gated on the scorer's steady per unit (|Δ| ≤ 0.0008°).

- **The yaw loop is underdamped.**
  - The corrective yaw-rate setpoint grows with |yaw|: N0 0.07 → 0.46 → 0.99 rad/s from <0.25° to >5°.
  - Yaw first crosses inside 0.5° after 0.23–0.30 s. But the max |yaw| in 0.3–1.6 s has a median of 1.84°
    (p90 4.6°): the yaw **overshoots and rings**.
  - Mean |yaw| over time windows (N0): 2.21 (0–0.15 s), 1.79, 1.25 (0.3–0.6), 0.54 (0.6–1.0), 0.37
    (1.0–1.6), 0.30 (hold).
- **The yaw-rate input is saturated most of the time.**
  - The gyro_z thermometer (fit from PID calibration rollouts) spans only ±0.030 rad/s (1.7°/s). Its inner
    cuts (±0.002–0.003 rad/s) lie inside the gyro noise σ of 0.0107 rad/s.
  - Share of ticks beyond the outer cut, N0: 90 / 91 / 84 / 73 / 66 / **62 % in the hold**. Y3: 39 % in the
    hold.
  - Median |gyro_z| in the hold is 0.042 rad/s (N0), above the top cut.
  - The WNN therefore sees yaw RATE as little more than its sign, so it cannot brake yaw before zero. The
    ringing and the ~1 Hz hold limit cycle are what that predicts.
- **Roll/pitch.** gyro_x beyond its outer cut (~0.06 rad/s): 91 / 76 / 49 % in the first 0.6 s, then
  17 / 7 / 3 %. This matches the roll/pitch gap being transient-only.
- The yaw_err ladder (±0.10, ±0.25, ±1.23, ±2.62°) saturates above 2.6°. The response flattens only mildly
  there (0.64 → 0.72 rad/s for 2.6–3.5° → 3.5–5°). This is **weak** evidence, below the gyro_z effect.
- **Caveat:** the rival's per-tick yaw is not traced. The underdamping is inferred from the rival's whole yaw
  0.287° vs the WNN's 0.66°, not observed side by side.

## 6. Levers, ranked (no MPCOF internals in the WNN; all model-free)

**1. Yaw-RATE input coverage (gyro_z ladder).**
- **Gap:** yaw transient (whole yaw +0.37°) and hold-yaw chatter (+0.16° N0 / +0.10° Y3). It may also reach
  the roll/pitch transient (+0.09°).
- **Evidence:** §5. gyro_z is saturated 62 % of hold ticks and 84–91 % of transient ticks. The inner cuts are
  below the noise. The yaw rings (median 1.8° overshoot). The gap scales with |yaw0| (§1.5).
- **Expected size:** unknown until flown. Upper bound ≈ the yaw share of the err gap (~0.3°) and of the
  steady gap (~0.10–0.16°). An honest guess is a fraction of that.
- **Flag or code area, cheapest first:**
  - (a) `--threshold-refit-from-student`: exists; refits every ladder on student states; costs one extra GRID
    stage.
  - (b) `--threshold-outer-quantile 0.02`: exists; global; the historic "calib=5deg" note says global
    widening can hurt.
  - (c) a gyro-only rate ladder: fixed outer cuts ~±0.05/0.15/0.3/0.6 rad/s with the inner cut at ~1σ noise,
    modelled on `--yaw-centred-ladder dense` in `src/wnn/control/evaluator.py`
    (`_dense_inner_for`/`_thresholds_from_samples`). This is a Python-side threshold change; whether it
    needs a ram_controller rebuild has not been checked.
  - All three start a new lineage (thresholds are the address function).
- **Cost:** (a) and (b) are one arm × n seeds each. (c) is a small code change plus a pre-check and then the
  arms.
- **Pre-check (cheap, no flight):** the per-sample "true yaw rate carried" metric, measured jointly with the
  inputs the yaw neurons see, per the CTRL-48 rule.

**2. Yaw alphabet shape instead of yaw step size (N0 transient + Y3 hold).**
- **Gap:** the N0/Y3 trade in §4.
- **Evidence:** Y3 is −0.060 hold yaw (6/7) but +0.077 pre-hold yaw (0/7). Hold sign flips: 2.4/s vs 0.4/s.
- **Lever:** `--ctbr-gamma 1 2 2 3` (exists): a steeper yaw bank at `--delta-max-yaw 0.5`, with finer small
  steps and the same largest step. An alternative is more yaw-bank levels.
- **Expected size:** at best Y3's steady with N0's yaw transient (~0.05–0.08° on the better side). It could
  also land between the two. Interacts with lever 1.
- **Cost:** one arm, flags only.

**3. Yaw integral vs motor-wear offset (matters mainly if Y3 is adopted).**
- **Gap:** the wear-proportional hold-yaw offset, ≈ 0.11° of Y3's 0.176 and ≈ 0.035° of N0's.
- **Evidence:** §2, slope 0.55°/c1 (Y3, r 0.89) and 0.18°/c1 (N0) vs 0.01 for the rival.
- **Lever:** `--yaw-integral-leak` (0.9995 now; ~2 s memory equals the whole episode, so the integral may
  never charge in 2 s). Also the `yaw_err_i` ladder (outer ±40 scaled units). Both are flags or ladder
  constants.
- **Expected size:** ≤ the offset share above. Uncertain, because the 2 s horizon limits any integrator.
- **Cost:** one arm.

**4. Fails (tail episodes of the translation draw).**
- **Gap:** 6 fails per arm (~99.98 % stable).
- **Evidence:** §3. The same 5 episodes, single-axis roll/pitch excursions in the mid window, on extreme
  collective/mass/xy draws that the rival also finds hardest.
- **Lever:** probably covered by lever 1 (rate-blind roll/pitch during the translation transient). No
  separate lever is proposed until `flight-dynamics` says whether the excursion is a tilt-reference
  departure (a scorer/position-loop question) rather than a control error.
- **Expected size:** unknown.

**Not levers (measured):**
- Motor wear, torque jitter and sensor noise do not create the gap (§2 knock-outs).
- The heading estimate is not the yaw problem any more: fed-yaw drift is 0.017° for both arms (it was ~0.17°
  before `--obs-yaw-from-att-est`).
- Tail-only fixes cannot move the mean: the top 1 % of episodes carry 4 % of the gap.

## 7. Not measured

- The row-B (eq-info) per-unit, per-axis and per-episode numbers. Row C stands in for B (within 0.002°).
- The rival's mid-window split and its per-tick yaw.
- Saturation or time-of-failure inside TEST episodes (only the phase windows).
- Per-disturbance-type error within flown episodes. The scorer does not split it; the knock-outs are
  counterfactual re-flights on 1 seed per arm.
- Y3 s8 (still flying at writing).
