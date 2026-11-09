# CTRL-48: how MPCOF>rate gets its steady, and what the student lacks (teacher + plant side), 09/11/2026

Investigation only (flight-dynamics). Scratch: /tmp/steady_probe/ (ablate*.py, trace.py, an*.py, units.py). Every number is 35 CF2.1 units (seeds 99990101-135) × 100 eps, from the ARGS_35 regime via `score_classical_baseline`. Baselines reproduce exactly: 1.352° / 0.684° / 0.341 m at xy 0.5, and 0.927° / 0.313° at xy 0. The traces come from a NON-installed copy of the wheel with print statements added; its aggregates are bit-identical.
`steady` = mean geodesic error to the position loop's TILT reference over t in [1.6, 2.0] s (att_err_ref.rs). Error is measured against the TRUE quaternion.

## 1. Teacher mechanism (ctbr_mpcof.rs, ctbr_axis_mpc.rs)
- One axis-decoupled condensed QP per axis on the closed inner loop: rate PID + lpf2p + motor lag + rigid body, 8 states [θ, ω, ua, I, f1, f2, pm, Fh].
  - Horizon 20 outer steps, terminal cost from the DARE. Cost q_att 100, q_rate 0.5, r 1. Box ±4.2 / 4.2 / 2.0 rad/s.
- Offset-free: d̂ ← d̂ + 0.02·(ω' − c·ua − d̂) every 1 kHz tick, and the QP regulates x − x_ss(d̂).
  - x_ss: Fh = ua = −d̂/c and I = −d̂/(c·ki), with sp_ss = 0. The rate integrator carries the trim.
- Hold command (t ≥ 1.6 s, est-flown):
  - roll/pitch sp |·| median 0.007 rad/s, sd 0.010. The QP box is never active.
  - yaw sp is a held offset of 0.18 rad/s (sd 0.20). The yaw integrator (τ ≈ 7 s) has not absorbed the trim, so the trim rides in sp.
- Increments: the δ the student's law needs (sp_k+1 − 0.7·sp_k) has median 0.0058, p90 0.014 rad/s (roll/pitch); the alphabet (BINARY 64 levels, γ = 2) steps 0.0025, RMS quantisation error 0.0024 rad/s ≈ 0.016° of θ. **Resolution is not a limiter.**
- What explains the teacher's hold command (regression R², hold, x̃ = x − x_ss, est-flown; oracle is the same to ±0.05):

| axis | θ, ω | + Ĩ (= the dres coordinate) | + ũa | + F̃h | drop Ĩ |
|---|---|---|---|---|---|
| roll / pitch | .30 / .26 | **.97 / .97** | .996 / .997 | +.000 | .41 / .44 |
| yaw | .06 | **.999** | 1.000 | +.000 | 1.000 |

  - The two "missing" terms together add ≤ 2.5 pp on roll/pitch and 0 on yaw. Ĩ carries the command, and dres already provides it.
  - ũa and F̃h are computable on the student side: ua is `CascadeDhat::measured_force`, and Fh is the student's own rate stage (`ctbr_rate_stage`). They are low value.
  - Fitted gain dsp/dθ ≈ −8.7 (est) / −9.3 (oracle) rad/s per rad.

## 2. Error budget (steady °, teacher MPCOF>rate h10)
| condition | xy 0.5 | xy 0 |
|---|---|---|
| teacher, Mahony-fed (the row) | **0.684** | **0.313** |
| oracle quaternion | 0.163 | 0.160 |
| inclinometer IMU (−g_body), Mahony kept | 0.167 | 0.164 |
| gyro/accel noise + bias walk off | 0.682 | 0.305 |
| motor wear off (est) | 0.515 | 0.120 |
| all disturbances off, est / oracle | 0.511 / 0.002 | 0.096 / 0.000 |
| motor lag 0 / outer hold 1 | 0.585 / 0.674 | 0.338 / 0.265 |

- **Estimator bias dominates: 0.52° of 0.68° (76 %).** It is not noise: noise off changes nothing.
  - The cause is the specific-force accelerometer (ABI 32). With the IMU idealised the row falls to the oracle value.
- Per axis at xy 0.5 (hold): true roll/pitch error 0.44 / 0.42° vs ESTIMATED 0.14 / 0.12°; est − true bias sd 0.44 / 0.41°, corr +0.68 with the tilt reference; yaw 0.08°.
- The teacher regulates its estimate to ~0.13°; the rest is Mahony bias, growing with the tilt/acceleration the translation demands.
- **Coordinator's xy 0 point, corrected.** The xy 0.5 excess is NOT "attitude needed to translate": err is measured against the tilt reference, and the oracle gets 0.163° at xy 0.5 and 0.160° at xy 0.
  - The whole xy-dependent part is estimator bias: 0.52° at xy 0.5, 0.15° at xy 0.
  - The oracle (plant + wear) floor is 0.16° in both regimes; a perfect controller on Mahony with no wear gets ≈ 0.51 / 0.10. (Not investigated: plain MPCOF at xy 0 reads 0.114°, below the cascade oracle.)

## 3. Per unit (dres s1-s3 mean, from the HEADLINE per-seed rows in the .out logs)
- Student 1.029°, teacher 0.684°. The gap is +0.346 ± 0.158 (−0.13 … +0.66); per seed +0.30 / +0.42 / +0.31.
- **The gap is not concentrated on worn units:**
  - corr(gap, roll/pitch asymmetry) = −0.27; corr(gap, the teacher's own steady) = −0.50;
  - mild third +0.41°, worst-asymmetry third +0.31°;
  - the student's absolute steady does track asymmetry (+0.65).
- So it is roughly an additive ~0.35° on top of the teacher, unit-independent, NOT a trim deficit (dres absorbs the unit's share).

## 4. Ranked hypotheses for the student gap (plant/teacher side)
1. **LIKELY: oracle labels vs Mahony observation.**
   - DAgger labels are computed from `sim.quaternion()` (dagger_train.rs:1765 → `pilot.command(.., q, ..)`). The student addresses on the Mahony "up" vector (att_estimator.rs).
   - In the hold the label therefore carries K_θ·(θ_true − θ_est) with sd **0.065-0.068 rad/s**. That is 6.5× the spread of the est-teacher's own hold label (0.010).
   - That term is 96-97 % linearly predictable from pos_err_xy + vel_xy, which the student does see. Learnable in principle (the ceiling is the oracle's 0.16°).
   - Learning it needs a fine pos/vel × attitude interaction in the hold, where pos/vel sit in a few middle codes. Unresolved, it is per-address label conflict.
   - Fits the unit-independent, xy-linked shape of §3.
2. **OPEN: hold-precision floor in the estimate frame.** The teacher holds its estimate to 0.13°. The student's estimate-frame error is unmeasured; test 1 separates this.
3. **REFUTED as main:** the missing ũa / F̃h terms (≤ 2.5 % of hold variance); alphabet resolution (0.016°); sensor noise (0.00°); wear/trim (the gap anti-correlates with asymmetry).
4. **Minor:** outer rate. hold 1 vs 10 is −0.05° at xy 0 and ~0 at xy 0.5.

## 5. Cheap tests
1. **Score-only (controller agent):** re-score the 3 dres HEADLINE genomes at `--xy-offset 0 --att-err-ref level` on the same 35 seeds.
   - H1 predicts the gap shrinks with the bias, from +0.35 toward ~+0.1 (bias 0.15/0.52).
   - A gap that stays at ~+0.35 points to H2 (hold regulation).
2. **Offline, no training:** on recorded student hold addresses (t > 1.6 s), compare per-address label variance between oracle labels and est-fed labels (`ctbr_label` given the Mahony q).
   - The H1 signature is oracle variance ≫ est variance at the same address.
3. **If 1 and 2 support H1, an A/B (needs a flag, Luiz's call):** est-fed DAgger labels, which are consistent with the observation and target 0.68. The alternative is a feature carrying the bias predictor, which targets toward 0.16.
