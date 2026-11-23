# Model-free controller block (N0, NR) — pre-registration

Luiz 22/11/2026: "What can we do to not have MPCOF running as part of the WNN?" → stop R/H (CTRL-38 re-test;
R s1 completed and valid, H s1 killed 7 min in, not a result) and fly model-free arms only.

## What MPCOF contributes on board today (Y+F+L)
- `--obs-cascade-dhat-residual`: 3 inputs computed by MPCOF's offset-free disturbance observer (airframe model:
  kt, 0.2 s motor lag, torque gains, observer gain L).
- `--yaw-trim-ff` (F): yaw-rate feed-forward cT·cresid_z, cT taken from MPCOF's gain row; requires the residual.
- Training-time only (unchanged here): the DAgger teacher is MPCOF (`--teacher mpcof`) — learning from it offline,
  nothing of it runs on the drone.

## Arms (installed ram_controller ABI 60, no code change; driver logs/controller/ctrl48/ab_n0_nr.sh)
- **N0** = Y+F+L − `--obs-cascade-dhat-residual` − `--yaw-trim-ff`: the model-free baseline (Y = `--delta-max-yaw 0.5`
  and L = dense yaw ladder stay; neither uses a model).
- **NR** = N0 + `--obs-rate-i`: the firmware rate loop's own integrator (I / i_limit) as model-free disturbance memory.
- Seeds s1-s5, seed-major interleaved (N0 sN, NR sN); paired vs Y+F+L `..._a59_..._ffF_yld_full_sN` (not re-flown).

## Questions
1. How much does removing MPCOF's observer cost (N0 vs Y+F+L)? Expected: worse on err/steady (a45 observer-free
   arm on an older base: err 1.92° vs 1.54°).
2. Does a model-free disturbance memory recover it (NR vs N0, and NR vs Y+F+L)?

## Readout
TEST, 35 CF2.1 units, paired per seed: stable / err / steady / pos / fails. Luiz decides adoption; a model-free arm
that is close to Y+F+L may be preferred for the paper even if slightly worse (the model-free claim).

## Addendum (22/11/2026): ABI-61 lineage — N0 control re-fly (CTRL-67) + N0 + `--delta-max-yaw 0.3` (CTRL-68)

Written before any ABI-61 flight. Luiz picked N0 (the model-free arm) as the new-lineage control: N0 beat NR on
steady and fails on 5/5 seeds. ABI 61 = f32 Mahony attitude estimator for the WNN AND for the rivals (merge of
`ctrl-f32-estimator` dbd123b0, landed by `logs/controller/ctrl61/land_abi61_after_mf.sh`). A new lineage: ABI-60 rows
are never paired with ABI-61 rows for adoption.

### Arms (driver `logs/controller/ctrl61/ab_mf61.sh`; seed-interleaved A sN, B sN, s1-s5)
- **A (CTRL-67) = N0 on ABI 61**, tags `C-S2-d_wposx2_a61_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_sN`: the exact
  argv of the flown ABI-60 N0 row `..._a60_..._yld_n0_full_sN` (rebuilt with `rebuild_argv.py fly`); only the save
  paths (tag) differ.
- **B (CTRL-68) = A with the yaw bank's delta_max 0.5 -> 0.3**, tags `..._a61_..._L1_yld_n0dmy03_full_sN`. N0 already
  carries Y (`--delta-max-yaw 0.5`), so B replaces that one value; nothing else differs from A. 0.3 gives a yaw DC
  gain G = 0.3/(1-0.7) = 1 under `--delta-leak 0.7`. Compatible with N0: `--output-mode ctbr` + `--ctbr-delta-law`,
  no `--yaw-trim-ff` (F) — checked by the driver through phased_ga's own validator before arming.

### Hypotheses
- H-A (lineage check, NOT an adoption test): A vs the ABI-60 N0 rows on the same seeds measures what the f32
  estimator moves. Expected: small (the estimator is the same algorithm at f32), no direction pre-registered.
  A is the ABI-61 control from here on, whatever it shows.
- H-B: with no FF and no observer, a smaller yaw step (decode gain 1 instead of 5/3) lowers the yaw-dominated
  steady-state error and err without costing stability. Expected direction: B < A on err and steady.

### Pairing and readout
B vs A, same seed, ABI 61 only (n = 5 pairs). TEST, 35 CF2.1 units per run (HEADLINE block), the whole vector:
stable / err / steady / pos / fails (stable_fail). Gen-line and VAL numbers are not results.

### Adoption rule (Luiz's rule)
ADOPT B if it is consistently better than A in the expected direction on err and steady (per-seed pairs, the
majority of the 5 in that direction and the mean delta in that direction), AND not worse elsewhere — stable,
fails and pos (3-D, includes altitude) not worse. No magnitude bar. Flat or worse -> not adopted (A stays the
control). Luiz rules on the readout.

## Addendum (23/11/2026): n=10 extension of CTRL-67 (A) / CTRL-68 (B), seeds s6-s10

Written before any s6-s10 flight, at Luiz's request (23/11 ~07:30 EST): n=5 is too close to call. TEST n=5
(35 CF2.1 units per run): B steady 0.240° vs A 0.308° (B lower on 5/5 seeds, 2-dp END rows), err 1.100° vs 1.088°, stable
99.97% vs 99.97%, fails 4 vs 4, pos 0.340 vs 0.339 m — steady in the expected direction, err not, the rest equal.

### Arms (driver `logs/controller/ctrl62/ab_mf61_n10.sh`; seed-interleaved A s6, B s6, ..., A s10, B s10)
- Same two arms as the ABI-61 addendum, recipes UNCHANGED: each sN argv is the flown s1 argv of the same arm
  (`..._a61_..._L1_yld_n0_full_s1` for A, `..._a61_..._L1_yld_n0dmy03_full_s1` for B) with only the seed-dependent
  fields changed — `--base-seed 31337001 -> 31337000+N` and the two save paths. The flown s1-s5 argv differ in
  exactly those fields. The 35 TEST units 99990101..99990135 are the same for every seed.
- Tags `C-S2-d_wposx2_a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_sN` (A) and `..._a62_..._yld_n0dmy03_full_sN`
  (B); `a62` records the wheel actually installed.
- Wheel: ABI 62 (landed by `ab_r62.sh`, merge 0b2c6338) with its new flags (`--obs-rate-err`, `--obs-rate-i-axes`)
  OFF. ABI 62 flags-off is byte-identical to ABI 61 — `tests/controller_ctrl62_abi61_pin.py` (grid ladder, features,
  Lamarckian write-back cell digest, scorer rows all equal across the two builds). So s6-s10 (ABI 62) pair validly
  with s1-s5 (ABI 61): same lineage. The driver refuses any rate-loop flag in the argv.

### Pairing, readout, adoption rule
B sN vs A sN, same seed, n = 10 pairs (s1-s5 ABI 61 + s6-s10 ABI 62). TEST HEADLINE block, the whole vector:
stable / err / steady / pos / fails. Hypothesis H-B and the adoption rule are UNCHANGED from the ABI-61 addendum
(ADOPT B if consistently better than A on err and steady — majority of the 10 pairs and the mean delta in that
direction — and not worse on stable, fails, pos; no magnitude bar; Luiz rules on the readout).
