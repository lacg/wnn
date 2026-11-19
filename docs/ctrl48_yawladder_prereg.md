# CTRL-48 dense yaw ladder block: prereg (Luiz go 19/11/2026 ~14:15 EST)

Status: PREREG. Nothing below may change after the first `YLD` END line. No driver exists yet; this document
says what the driver must do.

## Hypothesis
On the Y+F base the remaining gap to the MPCOF eq-info comparator is mostly the yaw axis, and it is a
transient, not a standing offset (gap investigation 19/11, `logs/controller/ctrl48/yf_gap/`):

- **TEST, 35 units, Y+F s1-s3 winners vs the oracle-fed label law** (per-axis split, `yf_gap/YF_s{1,2,3}_test.npz`):
  whole-episode yaw error 0.395° vs 0.287° (+0.108 ± 0.016 across units, 35/35 units, all three seeds);
  roll / pitch +0.044 / +0.044 (seed-variable: +0.017 / +0.016 on s1, +0.076 / +0.062 on s3).
- **The `yaw_err` input cannot resolve magnitude inside ±0.39°.** Cuts of the s1 winner (`yf_gap/ladders.py`):
  −2.52, −1.17, −0.39, 0, 0, +0.39, +1.17, +2.52°. One cut is spent on a duplicate 0.
- **The yaw bank is a relay there** (VAL, s1 winner, 6 units × 100 episodes, `yf_gap/gain_YF_s1_val.npz`): the
  yaw-rate setpoint (episode hold-mean removed) is +0.016 / +0.021 / +0.020 rad/s for yaw in 0–0.1 / 0.1–0.2 /
  0.2–0.39° on the negative side and −0.010 / −0.008 / −0.002 on the positive side, then jumps at the ±0.39°
  cut (+0.068 / −0.044). The label law's proportional term is about 9.6 rad/s per rad, i.e. ±0.05 rad/s at 0.3°.
- **The yaw error creeps** (`yf_gap/prof_YF_s1_val.log`): mean |yaw| 0.53° at 0.3–0.4 s, 0.39, 0.29, 0.23, 0.19 …
  and reaches its 0.11° floor only at about 1.4 s. 85% of ticks after 150 ms are inside ±0.39°.

`--yaw-centred-ladder dense` puts fixed cuts at ±0.10° and ±0.25° on `yaw_err`. With the trim now carried in
floating point by `--yaw-trim-ff`, the yaw bank's label is the proportional remainder, so resolving the
magnitude of a small yaw error is what the bank lacks. The same construction (dense tilt ladder, L1) brought
roll / pitch to the teacher's level.

**Why earlier NO verdicts do not close this.** The 14/11 pre-check ("not supported as specified"), the 16/11
ladder pre-check ("at most 0.02°") and L2y′ (block 10b, not adopted) all targeted the HOLD offset, before the
heading fix and before F, when the trim (0.13–0.17 rad/s) swamped the proportional term. None measured the
0.3–1.6 s window. They remain a stated risk: this lever has a history of predicted-then-absent gains.

## The flag (verified in `phased_ga.build_arg_parser`, `src/wnn/control/phased_ga.py:3494`)
- Exact spelling: **`--yaw-centred-ladder dense`** (`nargs="?"`, `const="quantile"`, choices `quantile | dense`).
  The base argv already carries the BARE flag (= `quantile`), immediately followed by `--yaw-integral-leak 0.9995`.
- **`rebuild_argv.py … fly` refuses an extra flag the source argv already carries**, so the driver must
  substitute, not append: replace the single token pair `--yaw-centred-ladder --yaw-integral-leak` by
  `--yaw-centred-ladder dense --yaw-integral-leak`, then abort unless the argv contains exactly one
  `--yaw-centred-ladder dense` and the run's `[provenance]` line reads `yaw:dense` (base: `yaw:centred`).
- Python only (threshold fitting, `evaluator._dense_thresholds`); thresholds reach Rust as data. No wheel
  change, **ABI 59 as installed**.
- What it changes, at 8 cuts per feature (`evaluator.py:805-847`):
  - `yaw_err`: ±0.10°, ±0.25° fixed, then two |x|-quantile pairs of the samples beyond 0.25°. **No cut at 0.**
  - `yaw_err_i`: the same angles times `integral_scale / (1 − yaw_integral_leak)` = 1 / 0.0005 = 2000, i.e.
    ±3.49 and ±8.73 in the integral's units (today ±3.03, ±11.15, ±32.9 with a duplicate 0), then two quantile
    pairs. **No cut at 0.**
  - Every other feature, including `cresid_z` and `gyro_z`, keeps its ladder.
- Interactions found in code (`resolve_block10_args`, `phased_ga.py:4447-4505`):
  - `--yaw-trim-ff`: none. The FF reads the float `cresid_z`, not a ladder.
  - `--obs-yaw-from-att-est`: none. It changes which heading `yaw_err` is computed from, before the ladder.
  - `--delta-max-yaw`: none. Output side only.
  - `dense` is refused only together with `--obs-yaw-trim-err[-p]` (not in this recipe).
- **Known cost of the ladder as built.** Neither yaw input keeps a 0 edge, so the sign of a yaw error below
  0.10° (and of an integral below 3.49) is not in the address. The hold yaw is about 0.10° today, so the hold
  window may not improve and could get slightly worse. This is why the steady expectation below is "flat".

## Arms
- **B = Y+F**, NOT re-flown: the existing rows `C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_ffF_full_s1..s5`
  (`logs/controller/ctrl48/runs.log`). Valid as the control because (i) arm L's argv is rebuilt from the same
  source `.out` (`…_a51_…_both_L1_full_sN`) by the same `rebuild_argv.py`, with the same base flags, and differs
  from B's by the one token `dense` plus the checkpoint / winner paths; (ii) both fly ABI 59, whose flag-off
  determinism gate passed against ABI 58 (`logs/controller/detgate/`); (iii) the driver asserts ABI 59 before
  every run. s4 and s5 of B are still flying as this is written; the pair for a seed exists when both END.
- **L = Y+F + `--yaw-centred-ladder dense`**, seeds s1-s5 (base 31337001-005), seed-major, ABI 59.
- Tags: `C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_full_sN` (`ffF` = the `ab_yawff.sh` arm
  name for base Y, `yld` = yaw ladder dense). START / END lines in `runs.log`, block ends `YLD DONE`
  (abort `YLD ABORT`). About 70–110 min per run, 5 runs ≈ 7 h. It must queue behind the running F-on-Y block
  (one phased_ga at a time; idle gate `pgrep -f "wnn.control.phased_g[a]"`).

## Launch addendum (to be written by the driver before the first `YLD` START)
- The argv diff B vs L for s1 (must be exactly `dense` + paths).
- After L s1 ENDs: the fitted `yaw_err` / `yaw_err_i` cuts of its winner (`yf_gap/ladders.py <tag>`). They are
  descriptive; the rule does not depend on them.

## Expected (stated in advance)
- **err −0.03 to −0.06°** (mean over seeds, L − B), carried by the yaw axis: whole-episode yaw −0.04 to −0.07°.
- **steady flat: 0 to −0.02°**, with +0.01° possible (the missing 0 edge, above).
- **pos flat** (a yaw-only input), stable 100%, 0 fails.
- Roll / pitch per-axis errors unchanged within the seed spread.

## Rule (TEST, 35 CF2.1 units per seed, paired by seed, L − B)
Luiz judges the WHOLE vector (stable / err / steady / pos / fails); pos is the 3-D error including altitude and
there is no separate altitude guard. A win on a seed = the metric is better on that seed.

L is ADOPTED if all of these hold at n=5:
- **It delivers its expected effect:** mean err better by ≥ 0.03°, and err better on at least 4 of 5 seeds.
- **The gain is on yaw:** in the per-axis split of each headline (`dmax_yaw/dmy_axis.py --genome WINNER
  --units test`), mean whole-episode yaw error is lower by ≥ 0.03°, and roll / pitch are not worse by more
  than 0.02° each.
- **steady** not worse by more than 0.01° in the mean.
- **pos** not worse by more than 0.010 m in the mean.
- **fails / stable:** B has 0 fails so far. Any fail in L is flagged and weighs heavily; L is not adopted with
  more than 1 fail in 17,500 episodes or with pooled fails above B's unless Luiz rules otherwise.

**Expectation-consistency (Luiz 16/11).** If err is flat or worse on the first three seeds (mean Δerr > −0.01°),
L is a not-adopt candidate at that point; Luiz decides on stopping. A result that is consistently flat is a
not-adopt even if its interval is undecided.

n=5 is a first read. If L passes, extend to s6-s10 against the F-on-Y rows before it becomes the base.

## Mechanism check (on L's s1 winner; measurement only, VAL units, nice 15, no flight)
`logs/controller/ctrl48/yf_gap/yaw_gain.py --tag <L s1 tag> --max-units 6`, then the banded table with the
bands re-cut at the new edges (0–0.10, 0.10–0.25, 0.25–0.39°), and `yaw_profile.py` for the time profile.
- **Pass:** on both signs the yaw-rate setpoint (episode hold-mean removed) grows in magnitude across the three
  bands, with a local slope between 0.10 and 0.39° of at least 4 rad/s per rad (today about 0 on the negative
  side and wrong-signed on the positive side: +0.021 → +0.020 and −0.008 → −0.002 rad/s).
- **And** mean |yaw| in 0.5–1.0 s is lower than B's 0.285 / 0.229 / 0.188 / 0.164 / 0.149° (s1, 100 ms bins).
- If err improves but the slope does not appear, the gain is not the hypothesised mechanism: report it so,
  and do not describe the result as "the relay zone is fixed".

## Reported
- The `ctrl-seed-table` pair table (B = Y+F · L = Y+F + dense yaw ladder) with the MPCOF eq-info reference row.
- Per-axis split of each headline, VAL and TEST.
- Each headline's stage and the fit-best candidate per stage on VAL (an n=1 headline gap can be the
  stage-select draw).
- The mechanism check above.

## Not in this block
- A variant that keeps a 0 edge (e.g. ±0.13°, ±0.39°, ±1.17°, ±2.5° on `yaw_err`) needs a change to
  `evaluator.DENSE_YAW_INNER_RAD` / `_dense_thresholds`. It is the follow-up if L improves err but costs steady.
- The float yaw P-term (`docs/ctrl59_yaw_pterm_spec.md`) is built in parallel and tested in its own block.
