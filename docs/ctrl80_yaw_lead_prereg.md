# CTRL-80 — yaw LEAD input on the G+T base — pre-registration (26/11/2026)

Luiz 26/11: "test first; if it improves we decide whether it counts as model-free, otherwise moot". Written before any
flight. Nothing flies until Luiz says so. Branch `ctrl80-yaw-lead`, NOT merged, wheel NOT installed.

## Why (VAL diagnosis only; G+T s1 winner, never TEST)

`docs/ctrl77_gap_analysis_n3.md` + `logs/controller/ctrl80/data/yaw80.txt`, `phase80.txt` (3 VAL units x 100 episodes):

- 57 % of the pre-hold yaw error of G+T is overshoot + tail (0.419 of 0.738 deg), not the rise.
- Yaw rate follows the held yaw-rate setpoint ~100 ms late (cross-correlation peak 100 ms, r 0.86).
- In the cell |yaw_err| 1.23-2.62 deg x closing rate 0.15-0.45 rad/s the student commands the same setpoint whether
  the time-to-go is 87 / 129 / 180 ms (+0.189 / +0.196 / +0.211 rad/s): it cannot see when to brake. Output and plant are
  not saturated (delta saturated 0.00 %, setpoint at the clamp 0.00 %).

## Pre-check (VAL; `logs/controller/ctrl80/ylead80.py` -> `data/ylead80.txt`; decision ticks 0-0.6 s, n = 18 000)

`yaw_lead = yaw_err - tau * gyro_z`, tau = 0.10 s. Both operands are the student's own input features as recorded.

- **Sign.** d(yaw_err over 100 ms) = -0.077 x gyro_z (ideal -0.100: the rate is already falling). RMSE of predicting
  yaw_err 100 ms ahead: yaw_err itself 1.034 deg | yaw_err - tau*gyro_z **0.627** | the wrong sign 2.059.
- **Ladder** = the 20 / 40 / 60 / 80 % quantiles of |yaw_lead| in 0-0.6 s (0.39 / 0.81 / 1.35 / 2.23 deg), rounded:

      -2.25  -1.35  -0.80  -0.40  +0.40  +0.80  +1.35  +2.25   deg

  Code occupancy (9 codes): 0-0.6 s 6.8 / 6.3 / 8.7 / 7.7 / 20.8 / 10.8 / 12.0 / 13.9 / 13.0 %;
  hold 0.0 / 0.1 / 3.3 / 13.2 / 62.0 / 14.4 / 6.8 / 0.2 / 0.0 %.
- **What the codes can tell apart** (table predictor from the code, leave-one-unit-out):

      predict                            sign of yaw_err 100 ms ahead    yaw_err 100 ms ahead, RMSE deg
      cell / codes                        today      + lead code          today      + lead code
      all ticks 0-0.6 s                   87.1 %     88.3 %               0.663      0.580
      |ye| 1.23-2.62 x closing .15-.45    62.9 %     78.1 %               0.692      0.525
      |ye| 0.25-1.23 x closing .15-.45    52.5 %     71.9 %               0.833      0.585
      |ye| 0.25-1.23 x closing .05-.15    85.4 %     86.0 %               0.530      0.484

  "today" = yaw_err code x gyro_z code (what a RAM neuron can address now). In the two braking cells the student has
  ONE code per sign; the lead splits them: time-to-go terciles 87 / 129 / 180 ms sit on lead codes
  (10, 23, 67 %) / (56, 43 %) / (35, 55, 10 %) of three adjacent code groups.
- **What it does NOT show.** The lead code does not predict the closed-loop event "overshoot > 0.5 deg within 300 ms"
  better than today's codes (all ticks 75.0 -> 73.5 %; first cell 73.8 -> 67.7 %). The information gain is the error
  100 ms ahead; whether the student turns it into less overshoot is the flight's question.
- Limits: one seed (s1), 3 VAL units; the cuts were chosen on the traces they are evaluated on.

## Arm (a fixed linear combination of two inputs the student already reads; no observer, no plant model, no MPCOF part)

`ylead` = G+T + `--obs-yaw-lead --obs-yaw-lead-tau 0.10`: ONE new input, directly after `yaw_err_i`, with its own
fixed ladder above. bits_per_feature stays 8; no cut of any other input moves (pinned).

- Control = the flown G+T sN run (`C-S2-d_wposx2_a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0g3gz_full_sN`), never re-flown.
  TEST values: s1 100.0 % / 1.01 deg / 0.27 deg / pos 0.343 m / 1 fail; s2 100.0 / 1.00 / 0.27 / 0.341 / 0;
  s3 100.0 / 1.03 / 0.29 / 0.340 / 1.
- Tags `C-S2-d_wposx2_a64_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_ylead_full_sN`, seeds s1-s3 (base 31337001-3),
  35 CF2.1 TEST units, **ABI 64** (new wheel: one more input feature; a new lineage, not pooled with another).
- Driver `logs/controller/ctrl80/ab_ylead80.sh` ('Y80' lines): argv = the control's + the two flags + save paths,
  checked byte for byte; refuses to run before the merge and the ABI-64 wheel install.
- Code: Rust `controller/yaw_lead.rs` (the one formula) + compute_features, Metal FC_YAW_LEAD(6) (score / train /
  record), Python layout + ladder + flags. Flag off pinned == ABI 63 (`tests/controller_ctrl80_yaw_lead.py`,
  `ctrl80_yaw_lead_tests.rs`).

## Hypothesis

- **H-lead.** With the error-100-ms-ahead in its address, the yaw bank brakes before the crossing instead of after:
  overshoot and tail shrink.
  - **Expected vs G+T: err lower**, through **pre-hold yaw lower** (TEST n=3 re-score, ctrl77_gap_analysis_n3:
    0.680 deg vs MPCOF 0.340) and whole yaw lower (0.585 vs 0.287).
    steady / hold yaw: no change expected. stable / fails / pos: no change expected.
- **Risks (stated).**
  1. gyro_z noise enters an error input: in the hold sd(lead - yaw_err) = 0.38 deg, the size of the inner cut
     (+-0.40), so the lead code toggles in the hold (centre code 62 % of hold ticks). Watch **steady and hold yaw**.
  2. The distribution was measured in closed loop under today's policy and will move once the student acts on it.
  3. tau is fixed at the measured lag of the G+T base; it is not tuned and not swept here.
  4. Interaction with `yil` (CTRL-79) and `l2b` is NOT tested: this arm stacks on plain G+T only.

## Open question (explicit)

**Does a fixed lead term count as model-free?** It uses no plant model, observer or teacher internal, but tau was read
off the measured loop lag. **Decided by Luiz only if the arm improves**; if it does not, the question is moot. Until
then the arm is reported as "G+T + yaw lead", never as part of the model-free base.

## Adoption rule

Luiz's usual rule: better only if **consistently better in the expected direction across seeds (err, through pre-hold
yaw) and not worse elsewhere** (stable / steady / pos / fails). No magnitude bar. n=3 (s1-s3) is a screen:

- **Drop at n=3** if (A) err not lower than G+T on a majority of seeds, or (B) steady higher than G+T on a majority
  of seeds, or (C) fails sum > G+T's + 2, or (D) more fails than G+T on a majority of seeds.
- A survivor goes to n=5, n=7, n=10 on Luiz's word; adoption (and the model-free question) is decided at n=10.
- **Primary readout, each END:** the whole vector paired vs G+T same seed — stable / err / steady / pos / fails,
  35 CF2.1 TEST units, as VALUES.
- **Per-END CPU re-score** (`logs/controller/ctrl80/rescore_end.sh <arm_tag> <control_tag>`, descriptive): pre-hold
  yaw, whole yaw, hold yaw (and roll / pitch), plus the fixed hard-episode panel.
- After s1: a VAL trace of the winner (`trace80.py` + `phase80.py` + `ylead80.py`: overshoot by |yaw0|, the braking
  cells, lead-code occupancy in the hold) as diagnosis, never a result.
- cells / minutes are tie-breakers only.

## Amendment 1 (Luiz 27/11/2026 ~07:45 EST): fly the lead on the adopted l2b base

CTRL-77 closed with **l2b adopted** as the new model-free base (G+T + `--gyro-xy-ladder dense_hold`). The CTRL-80 arm
is therefore **l2bylead = l2b + `--obs-yaw-lead --obs-yaw-lead-tau 0.10`** (tag `…_a64_…_gt_l2bylead_full_sN`),
not G+T + lead. The plain `ylead` arm stays in the driver but is not flown by default.

- **Primary pair:** l2bylead sN vs the flown **l2b sN** (`…_a63_…_gt_l2b_full_sN`, s1–s3), same readout and the same
  adoption / drop rule as above (G+T replaced by l2b everywhere in the rule). G+T sN is a descriptive reference.
- **Argv validation** is unchanged in kind: the arm argv minus its flags and tag must equal the flown G+T sN argv byte
  for byte; the resolver must give the G+T digest with token `…,gyroxy:dense_hold` and `lead True 0.1`. The driver also
  refuses to fly sN unless l2b sN has finished. Dry run (unpacked ABI-64 wheel) validated 3/3 on 27/11.
- Note: l2b sN were flown on ABI 63 and l2bylead flies on ABI 64; the flag-off pin (ABI 64 == ABI 63) is what makes
  that pair valid.

## Amendment 2 (Luiz 27/11/2026 ~18:30 EST): arm l2byleadngz — the lead replaces gyro_z in the yaw bank

l2bylead s1–s3 (vs l2b): helps the transient and the tails on 3/3 (whole-err p99.9 2.55/2.34/2.51 vs 2.78/2.63/3.16,
pre-hold yaw lower 3/3), costs the hold (steady higher 2/3, wear offset larger 3/3); by the letter it dropped on (B).
VAL diagnosis (logs/controller/ctrl80/hold81.py, genome81.py; mechanism only): gyro-noise chatter is NOT the main cause;
the yaw bank spreads its 24 bits over its yaw-family inputs, so the 4th input cut yaw_err_i from ~6.6 to ~5.0 bits per
neuron (27-28 % → 20-21 % share, 3/3), and the trim is learned less well (offset per trim larger 3/3; inferred causal).

- **Arm l2byleadngz** = l2bylead + `--drop-features gyro_z` (tag `…_a64_…_gt_l2byleadngz_full_sN`): yaw inputs become
  [yaw_err, yaw_err_i, yaw_lead]; the lead still carries the rate (yaw_err − 0.10·gyro_z). No code change, ABI 64.
- **Hypothesis H-ngz:** with the integral's bits back and the hold address chatter halved (VAL re-code: 24/22/20 vs
  51/50/49 toggles/s), the lead's transient gain stays and the trim cost goes: err lower via pre-hold yaw; steady not
  higher; wear offset per c1 at or below l2b's; stable / fails / pos unchanged.
- **Pre-checks (VAL, re-coded flown traces):** braking-cell accuracy identical to l2bylead (83.9/76.4, 89.0/70.9,
  84.4/78.7 %) vs 67.5/60.7/62.5 % with today's yaw_err × gyro_z; hold-yaw representability .117/.194/.084 vs
  .104/.189/.079 deg.
- **Stated risk:** removing the dense gyro_z ladder (lever G of G+T) from the yaw bank may coarsen hold rate damping.
- **Rule:** same as above, l2b sN as control: drop at n=3 if (A) err not lower on a majority, (B) steady higher on a
  majority, (C) fails sum > l2b's + 2, (D) more fails on a majority. l2bylead sN is a descriptive reference.
- **Readout per END:** paired vector vs l2b sN, `rescore_end.sh <arm> <l2b sN>`, `yawrun_end.sh <arm> <l2b sN>`; after
  s1 the VAL hold check (hold81.py) on the new winner.
- Driver `logs/controller/ctrl80/ab_ylead80.sh` (default arm now l2byleadngz; digest skip adds `drop_features`); dry run
  3/3 validated 27/11.
