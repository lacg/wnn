# CTRL-48 yaw decode gain 1 (`--delta-max-yaw 0.3`, no F): prereg (Luiz go 19/11/2026 ~18:40 EST)

## Question
Is F (`--yaw-trim-ff`, a float feed-forward on the yaw output) needed at all, or does a pure-table change deliver the
same thing?

## Hypothesis
The yaw bank is trained on the teacher's absolute yaw-rate setpoint and the delta law integrates what the table says.
At rest the held setpoint is G times the label, G = delta_max_yaw / (1 − leak). On base Y, G = 0.5 / 0.3 = 1.67, so the
drone holds 1.67× the push the teacher asked for; the excess is proportional to the airframe's yaw trim. With
`--delta-max-yaw 0.3` (leak unchanged at 0.7) G is exactly 1 and the excess is gone.

**Evidence** (VAL tuning numbers, no trained student; `docs/ctrl60_pure_table_routes_spec.md`, "C2 pre-check result"):
- Label law through the student's output law, perfect table, 20 units: hold yaw 0.532° at G 1.67 (R² 1.00 against the
  trim) vs 0.077° at G 1 (R² 0.00); the label law itself holds 0.075°. Transient cost at G 1: err +0.034°.
- Calibrated hold-offset model on real genomes: G 1 lands within 0.003–0.007° of F (M genomes 0.209° vs F 0.206°;
  Y genomes 0.153° vs F 0.146°).
- The model is validated on one step only (G 2 → 1.67: predicted −0.049°, measured −0.054°). G 1 is an extrapolation,
  and a retrained student adapts, so flight numbers will differ.

## Arms
- **Y3 (flown):** adopted base (both_L1 + `--zrank-pos-floor auto` + `--obs-yaw-from-att-est`, `--delta-max 0.6`) +
  `--delta-max-yaw 0.3`, WITHOUT `--yaw-trim-ff`, quantile yaw ladder. Rebuilt from each seed's a51 both_L1 `.out`
  exactly as the Y rows were; the dry check shows the argv differs from the flown Y row only by `0.5 → 0.3` and the
  output paths. ABI 59. Tags `C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_dmy03_full_sN`.
- **Y (not re-flown):** `..._a57_..._dmyY_full_s1..s5`. Pairing across ABI 57 → 59 rests on the flag-off determinism
  gates passed at ABI 58 and ABI 59.
- **Y+F (not re-flown):** `..._a59_..._ffF_full_s1..s5`.

**Seeds:** s1-s5 (base 31337001-005), about 7 h, driver `logs/controller/ctrl48/ab_dmaxyaw03.sh`, queued behind the
dense yaw ladder block. TEST = 35 CF2.1 units × 100 episodes.

## Expectation
Against Y: steady −0.15 to −0.22°, err −0.10 to −0.20°, pos flat, 0 fails. Against Y+F: steady within ±0.02°, err
0 to +0.04° (the transient cost seen in the probe), pos flat.

## Rule (a starting point; Luiz judges the whole vector: stable / err / steady / pos / fails)
- **Y3 delivers its expected effect** if, paired by seed against Y: mean steady better by ≥ 0.10° and better on at
  least 4 of 5 seeds, err not worse, pos not worse by more than 0.010 m, at most 1 fail in 17,500.
- **"F is not needed"** if, in addition, paired against Y+F: steady not worse by more than 0.02°, err not worse by more
  than 0.03°, pos not worse by more than 0.010 m, fails no worse. Then Y3 replaces Y+F as the base and the float term
  leaves the paper.
- If Y3 beats Y but falls short of Y+F, F stays (one sentence + the ablation row, already ruled final) and Y3 + F is
  the follow-up question, not flown here.
- Flat or worse than Y on the first three seeds makes it a not-adopt candidate.

## Checks at readout
- Per-axis split on each Y3 winner (`dmax_yaw/dmy_axis.py --genome WINNER`): the gain must show on hold yaw, and the
  per-unit standing yaw error must lose its dependence on trim size (R² toward 0).
- Report the headline stage per seed and the yaw setpoint step noise (a smaller yaw step could slow the yaw transient:
  watch whole-episode yaw, not only hold yaw).
