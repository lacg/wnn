# CTRL-71 / CTRL-72 — levers 1 and 2 of the CTRL-70 gap analysis — pre-registration

Written 23/11/2026 before any flight of either arm. Luiz ruled 23/11 ~17:20 EST: Y3 (`--delta-max-yaw 0.3`,
CTRL-68) is NOT adopted, the n=10 extension is stopped, N0 (CTRL-67) stays the model-free control, and levers 1
and 2 of `docs/ctrl70_gap_analysis.md` §6 are pre-registered and flown.

**Flight order (Luiz, 23/11 evening, before any flight): the two arms fly in TANDEM from ONE driver,**
`logs/controller/ctrl62/ab_gt7172.sh`, seed-interleaved G s1, T s1, G s2, T s2, ..., G s5, T s5. Nothing flies
until the lever-1 code has landed and passed its flag-off pin and its VAL pre-check (both below). G is never
flown alone as a fallback.

## Control: N0 (CTRL-67), not re-flown

`C-S2-d_wposx2_a6[12]_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_sN` in `logs/controller/ctrl48` — s1-s5 on
ABI 61, s6-s8 on ABI 62 (flags off; ABI 62 flags-off is byte-identical to ABI 61,
`tests/controller_ctrl62_abi61_pin.py`). Model-free: no MPCOF observer input, no `--yaw-trim-ff`.

## Arms

Both arms run on the installed ram_controller ABI 62 with its rate-loop flags OFF. Each sN argv is the flown N0 s1
argv with only `--base-seed 31337001 -> 31337000+N`, the two save paths (tag) and the arm's ONE added flag; the
driver also checks it against the flown N0 sN argv (arm flag + tag removed -> must equal it byte for byte).

- **G (CTRL-72, lever 2) = N0 + `--ctbr-gamma 1 2 2 3`**, tags `..._a62_..._L1_yld_n0g3_full_sN`. N0 carries no `--ctbr-gamma` (default 1 2 2 2), so G changes the YAW bank's
  alphabet exponent 2 -> 3 and nothing else: thrust, roll and pitch banks unchanged, `--delta-max-yaw 0.5` kept
  (same largest yaw step, finer small steps). The driver verifies through phased_ga's own resolver that the
  resolved gamma is (1, 2, 2, 3) against N0's (1, 2, 2, 2) and that every other resolved ctbr field is equal.
  Thresholds (the address function) are untouched: same input lineage as N0.
- **T (CTRL-71, lever 1, option c) = N0 + a fixed gyro-only rate ladder on gyro_z**, tags
  `..._a62_..._L1_yld_n0gz_full_sN`, flag `--gyro-z-ladder dense`.
  - The gyro_z thermometer takes the dense ladder with FIXED symmetric cuts **±0.011, ±0.05, ±0.15, ±0.45 rad/s**
    (8 bits per feature = 4 pairs, no cut at 0), replacing the PID-calibration quantile fit
    (N0 s1: ±0.002/0.003, ±0.008/0.009, ±0.015/0.016, ±0.029/0.031 rad/s). Inner cut ≈ 1σ of the gyro noise
    (0.0107 rad/s). Every other feature keeps its N0 ladder.
  - **gyro_z only, not all three axes.** VAL traces (N0 s1, 3 val units × 100 episodes, CTRL-70 §5; a diagnosis,
    not a result): gyro_z is beyond its outer cut on 62 % of hold ticks and 66-91 % of every earlier window;
    gyro_x / gyro_y are beyond theirs on 2 % / 0 % of hold ticks and 5 % / 1 % at 1.0-1.6 s — they saturate only
    in the first 0.6 s (90/73/44 % and 86/73/38 %). Hold roll/pitch are already at the rival's level (0.094 vs
    0.095/0.083), so a roll/pitch ladder change would spend their hold resolution on a transient-only gap.
    Roll/pitch rate coverage is a separate lever, not part of T.
  - Cuts chosen on those same VAL traces (never TEST): share of |gyro_z| ticks beyond ±0.45 rad/s is
    11 / 20 / 5 / 1.3 / 0.4 / **0.1 %** over 0-0.15 / 0.15-0.3 / 0.3-0.6 / 0.6-1.0 / 1.0-1.6 s / hold, against
    90 / 91 / 84 / 73 / 66 / **62 %** for the flown ladder; hold occupancy of the five |gyro_z| bins is
    14 / 44 / 37 / 5 / 0.1 %.
  - **No-flight gate before T flies:** the same saturation fractions re-measured through the IMPLEMENTED ladder
    (the thresholds phased_ga actually fits with the flag on) on the VAL traces: hold ≤ 5 % and every transient
    window ≤ 25 % beyond the outer cut, else T is not flown and the result is reported instead.
  - **New threshold lineage.** Thresholds are the address function, so T is a new input lineage and is never
    pooled with another ladder. Its control is still valid as flown: the flag is a Python-side threshold change;
    with the flag OFF the fitted thresholds and the resolved run configuration are byte-identical to today's N0
    (pinned by a test that must pass before the merge), and T flies on the same ABI-62 wheel as N0 s6-s8. If the
    implementation turns out to need a ram_controller rebuild / ABI change, T is NOT flown under this prereg —
    stop and report (the control would then need its own flag-off pin on the new wheel).

## Hypotheses (from CTRL-70)

- **H-G (lever 2, gap = the N0/Y3 trade, §4):** Y3 cut hold yaw (−0.060°, 6/7) by shrinking every yaw step and
  paid in yaw transient (+0.077° pre-hold yaw, 0/7). A steeper yaw alphabet keeps the largest step and refines
  the small ones, so G should get the hold-yaw gain without the transient cost.
  Expected direction vs N0: **steady lower; err lower** (at best Y3's steady with N0's yaw transient,
  ~0.05-0.08° on steady; it may land between N0 and Y3). stable / fails / pos: no change expected.
- **H-T (lever 1, gap = yaw transient + hold-yaw chatter, §5-§6):** the WNN sees yaw rate as little more than
  its sign (gyro_z saturated 62-91 % of ticks), so it cannot brake yaw before zero: overshoot, ringing, and a
  ~1 Hz hold limit cycle. A ladder that resolves the flown yaw rates should damp both.
  Expected direction vs N0: **err lower** (whole yaw is +0.37° of the err gap) **and steady lower** (hold yaw
  +0.16°). Upper bound ≈ 0.3° err / 0.10-0.16° steady; an honest guess is a fraction of that. stable / fails /
  pos: no change expected (a drop in the roll/pitch mid-window fails is possible, not registered).

## Seeds, pairing, readout

- Seeds s1-s5 (base seeds 31337001-31337005), n = 5 pairs per arm. Extension to s6-s8 only on Luiz's word.
- Pairing: arm sN vs N0 sN, same seed. Never a recipe repeated on a seed already flown.
- TEST: the same 35 CF2.1 units 99990101..99990135 × 100 episodes (HEADLINE block of each run), the whole
  vector: stable / err / steady / pos / fails (stable_fail). Gen-line and VAL numbers are not results.
- `--num-eval-folds 5`, rule P (`--val-units 20 --select-tail p99`), everything else as N0.

## Adoption rule (Luiz's rule, per arm, vs N0)

ADOPT if consistently better than N0 in the expected direction on err and steady (per-seed pairs: the majority
of the 5 in that direction and the mean delta in that direction) AND not worse on stable, fails and pos (3-D,
includes altitude). No magnitude bar. Flat or worse -> not adopted (N0 stays the control). Cells (TRUE keys) and
wall minutes are tie-breakers only. Luiz rules on each readout.

**G + T combination:** decided only after both arms have read out (they interact: both act on the yaw loop).
No combined arm is flown or registered here.

## Addendum (23/11/2026, before any flight): lever 1 landed — pin and VAL pre-check

- **Code:** `--gyro-z-ladder dense` (branch `ctrl71-gyro-z-ladder`, merge 0fe3e8be). Python-side threshold fit only
  (`evaluator.DENSE_GYRO_Z_INNER_RAD_S`, `ThresholdCalibration.gyro_z_dense`, provenance `calib=...,gyroz:dense`);
  **no ram_controller rebuild, ABI stays 62.** Every fit site (grid, held-out/test refits, stage-select, the
  C export's train-seed refit) reads it through `calibration_from_args`.
- **Flag-off pin (passed):** `tests/controller_ctrl71_gyro_z_ladder.py` — on the flown N0 s1 argv, the 192 fitted
  thresholds, the calibration token, the resolved ctbr dict and all 234 parsed arguments are identical between
  the pre-CTRL-71 tree (4959e143) and the merged tree on the same wheel. Flag on: the gyro_z block is exactly
  ±0.011/0.05/0.15/0.45 rad/s and the other 23 ladders are byte-identical.
- **VAL pre-check (passed the gate; a diagnosis, not a result; never TEST):**
  `logs/controller/ctrl71/precheck_val.py`, the ladder phased_ga fits, on the CTRL-70 VAL traces (N0 s1 winner,
  3 val units × 100 episodes). Windows 0-0.15 / 0.15-0.3 / 0.3-0.6 / 0.6-1.0 / 1.0-1.6 s / hold:

```
                         beyond the outer cut, % of ticks       | yaw-rate variance carried by the gyro_z code
flag off (flown N0)      90.4 / 91.2 / 84.5 / 73.7 / 66.3 / 63.0 | 0.59 / 0.66 / 0.60 / 0.58 / 0.61 / 0.65 (all 0.50)
--gyro-z-ladder dense    11.4 / 19.8 /  5.4 /  1.3 /  0.4 /  0.1 | 0.92 / 0.94 / 0.93 / 0.92 / 0.91 / 0.90 (all 0.93)
```
  Gate: hold 0.1 % ≤ 5 %, worst transient window 19.8 % ≤ 25 %. (Y3 s1 traces: 6.4 / 19.3 / 1.1 / 0.1 / 0 / 0 %.)
  The carried share is single-input (the 9-bin gyro_z code alone against the ±5-tick finite-difference yaw rate),
  not the joint measure with the other yaw inputs.

## Addendum (23/11/2026, before any G+T flight): third arm G+T, flown in parallel

Written while G s1 is in the air (started 22:25:12Z) and before any T or G+T flight. This supersedes the line
above that no combined arm is flown: at Luiz's request (23/11 ~18:00 EST) the combination is flown **in
parallel** with G and T rather than after their readouts.

- **G+T = N0 + `--ctbr-gamma 1 2 2 3` + `--gyro-z-ladder dense`**, tags `..._a62_..._L1_yld_n0g3gz_full_sN`,
  ABI 62 flags-off, story CTRL-71/72. Vs the flown N0 sN argv it differs only by the two arm flags and the tag:
  through the resolver, only the yaw gamma (1,2,2,2 -> 1,2,2,3) and the calibration token (`+,gyroz:dense`) move,
  and the digest of every other resolved setting equals the control's. It is on T's threshold lineage.
- **Order:** per seed G sN, T sN, G+T sN, seeds s1-s5 (15 runs), one driver
  `logs/controller/ctrl62/ab_gt7172b.sh`, which takes over from `ab_gt7172.sh` between two flights (the old
  driver logs the END of the run it is flying; no flight is interrupted, none is launched twice).
- **Pairing and rule:** G+T sN vs N0 sN, same seed, n = 5, the same 35 TEST units, the same adoption rule as G
  and T. **Registered direction: err lower and steady lower; stable / fails / pos not worse.**
- **Descriptive only (no adoption rule):** G+T sN vs G sN and vs T sN on the same seed — does the combination
  add up (G+T at least as good as the better of G and T on err and steady), or do the two yaw levers interfere?
- Seeds s6-s8 only on Luiz's word, as for G and T.

## Addendum (24/11/2026, before any s6-s10 flight): n=10 extension of G, T and G+T

Written after `GT DONE` (24/11 12:50:58Z; G, T, G+T s1-s5 all finished) and before any flight below, at Luiz's
request (24/11 ~11:15 EST): extend all three arms to n=10.

- **Arms, recipes UNCHANGED:** G, T and G+T on seeds s6-s10 (base seeds 31337006-31337010), tags
  `..._a62_..._L1_yld_n0g3_full_sN`, `..._yld_n0gz_full_sN`, `..._yld_n0g3gz_full_sN`. Each argv = the flown N0
  s1 argv with only `--base-seed`, the save paths (tag) and the arm's flag(s); G/T/G+T differ from N0 sN only by
  their flags (checked byte for byte and through the resolver, as for s1-s5). ABI 62, rate-loop flags off; the
  CTRL-71 code is unchanged since its pin (`src/wnn/control` has no commits after merge 0fe3e8be).
- **N0 controls s9 and s10 (CTRL-67 recipe, model-free control):** N0 has finished rows s1-s8 only.
  - s10: tag `C-S2-d_wposx2_a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s10`.
  - s9: the tag `..._a62_..._L1_yld_n0_full_s9` is VOID (killed 23/11 at the MF10 stop: .out, no END, no .done)
    and is never reused. The s9 control flies under `..._a62_..._L1_yld_n0_full_s9r`. This is the **first
    completed flight of the N0 recipe on s9**, not a repeat of a finished result: no N0 s9 row exists.
  - Both argv = the flown N0 s1 argv with only `--base-seed` and the save paths changed (the n=10 rule of the
    CTRL-67/68 extension). Tick-table regex for N0: `_a6[12]_.*_L1_yld_n0_full_s{s}r?$` (a finished row wins over
    the void START-only s9).
- **Order:** per seed N0 sN (s9, s10 only), G sN, T sN, G+T sN; seeds s6..s10 = 17 runs, one driver
  `logs/controller/ctrl62/ab_gt7172_n10.sh`. If N0 s9r or s10 does not finish (rc != 0), the driver stops
  (`GT10 ABORT`) rather than fly arms without their control.
- **Pairing and rule, read at n=10:** arm sN vs N0 sN, same seed, n = 10 pairs per arm (s1-s5 and s6-s8 controls
  on ABI 61/62 flags-off as already registered, s9r/s10 on ABI 62). Same 35 TEST units, same adoption rule
  (majority of the 10 pairs and the mean delta in the expected direction on err and steady; not worse on stable,
  fails, pos; no magnitude bar; cells + wall minutes tie-breakers only). G+T vs G and vs T stays descriptive.
