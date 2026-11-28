# CTRL-87 — nominal-mass anchor (D2 a) + wider training spread, on the l2b base (prereg, 28/11/2026)

Approved by Luiz 28/11/2026. Driver: `logs/controller/ctrl87/ab_nm87.sh` (N87 lines in
`logs/controller/ctrl48/runs.log`). ABI 66 (merge cc3539d5). Dry run: `logs/controller/ctrl87/dry_run_2811.txt`.

## Why

1. **D2 (CTRL-83 fairness, Luiz chose option a).** Up to ABI 65 the WNN's collective anchor and its `coll_cmd` input
   are built from each episode's TRUE mass draw (trainer, calibration, recorder, both scorers); the classical rivals
   use the NOMINAL mass. `coll_cmd` feeds 99-112 of the 256 output neurons in the l2b winners and ~36 % of its
   variance is mass, so the WNN has had privileged plant information. `--anchor-mass nominal` (ABI 66) removes it.
   This is a **correctness/fairness change, not a lever**: it becomes the new base whatever it costs (subject to Luiz).
2. **Wider training spread.** The CTRL-83 stress panel (`docs/ctrl83_stress_panel_abi66.txt`) shows the WNN brittle
   beyond its training spread: wear ±30 % → 770 fails / 17 500 (MPCOF 16 / 3 500); lag T 0.30 s → 1.38° / 51 fails,
   0.40 s → 2.48° / 448. Training today = wear ±20 % (`train_asym=per_episode:0.2`) and a FIXED lag T = 0.20 s.

## Arms (each argv = the flown l2b sN DEDUPED argv + exactly the arm flags; tag `C-S2-d_wposx2_a66_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_<arm>_full_sN`)

| arm | flags added to l2b sN | primary pair | role |
|---|---|---|---|
| `l2bnm` | `--anchor-mass nominal` | vs l2b sN | fairness base — cost readout, no adopt/drop |
| `l2bnmw` | `--anchor-mass nominal --train-motor-asym-mag 0.30` (+ lag range, see Code gap) | vs **l2bnm sN** | lever, rule A-D |

l2b sN = `C-S2-d_wposx2_a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b_full_sN` (ABI 63, flown, never re-flown).
The l2b → a66 pairing is valid through the ABI-66 flag-off pins (`tests/controller_ctrl83_flag_off.py`,
`ctrl83_flag_off_pins.rs`: default `--anchor-mass true` == ABI 65 == ABI 63 bit for bit). TEST units unchanged:
35 CF2.1 units 99990101..135 at wear ±20 %, lag 0.20 s (the scoring plant does NOT widen with the training spread).

Resolver check (driver): l2b resolves to `anchor true | train per_episode:0.2`, lag 0.2, calibration token
`eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense,gyroz:dense,gyroxy:dense_hold`; the arm moves ONLY
`anchor_mass` (and, for l2bnmw, the training-wear token → `per_episode:0.3`); token + digest of everything else equal.

## Code gap (l2bnmw is PENDING; the driver refuses it)

Neither half of the wide spread can be set from the CLI today:

- **Wear.** The training wear magnitude is NOT a flag: it comes only from `--disturbance L4C`
  (`DisturbanceConfig._L4_LEVELS`, 0.20), and the same `ec.disturbance.motor_asym_mag` drives both the trainer
  (`evaluator._train_asym_kwargs` → Rust `dist_motor_asym_mag`) and the Python per-episode draws of the scorers
  (search folds, val, the 35 TEST units). Needed: a Python-only flag `--train-motor-asym-mag M` that overrides ONLY
  `_train_asym_kwargs` (DAgger, per-round eval, calibration sampler, MEMORY recorder) and `train_asym_token`
  provenance; scorers untouched; add to the resolver/provenance; flag-off pin. Rust already accepts any mag in [0, 1)
  (dagger_train.rs) — **no ABI bump**. Open choice for Luiz: training hosts only (proposed), or also the GA search folds.
- **Lag range.** Does not exist. The trainer takes ONE scalar `motor_lag_s` per config and builds the actuator once
  (`dagger_train.rs` `with_actuator(self.motor_lag_s, …)`). Needed: Rust — a per-episode lag draw (e.g.
  `motor_lag_range_s: (lo, hi)`, uniform, from the episode seed after the asym draw, like `train_motor_asym`) in the
  DAgger trainer, per-round eval, `calib_sampler.rs` and `record_ops.rs`; the WNN's actuator twin (`DesignModel`,
  today scorer-only) must stay at the nominal 0.20 in training when the plant lag varies; PyO3 kwargs; Python flag
  `--train-motor-lag-range LO HI` + provenance; flag-off pins in `cargo test`. Metal (`controller_rollout.metal`)
  only if the range is also flown by the GPU scorer (search folds) — then a per-episode lag buffer + a CPU/GPU parity
  sweep. **ABI 67**, controller wheel only (swap-free).

Until the wear flag lands, only `l2bnm` flies. When it lands: wear-only `l2bnmw` (this prereg) or wait for the lag
range and fly wear + lag as one arm — Luiz's call; the argv/rule here are written for wear-only.

## Hypotheses

- **H-nm (cost readout):** removing the true-mass anchor costs a little on the nominal TEST vector (err / steady /
  alt up slightly, most on `mass25`), stable / fails ~unchanged. Reported as values vs l2b sN; no adopt/drop.
- **H-w:** a ±30 % training wear spread cuts the OOD wear fails (`wear30`, `wear25`/`wear40` secondary) and does not
  worsen the nominal TEST vector vs l2bnm sN.

## Seeds and rule

n = 3 first (s1-s3), seed-major (l2bnm sN before l2bnmw sN; the driver refuses l2bnmw sN before l2bnm sN is done).

## Rule

**l2bnm vs l2b:** reported as a cost/fairness readout (whole vector, values) — no adopt/drop; l2bnm becomes the base
subject to Luiz.

**l2bnmw vs l2bnm (Luiz's usual rule, nominal TEST vector, 35 units):** better only if consistently better in the
expected direction and not worse elsewhere; no magnitude bar. Drop at n=3 if
(A) err not lower than l2bnm on a majority of seeds, or (B) steady higher on a majority, or
(C) fails sum > l2bnm's + 2, or (D) more fails than l2bnm on a majority. A survivor goes to n=5 on Luiz's word.
Stated tension (for Luiz before l2bnmw flies): H-w predicts the gain in the STRESS cells, not at nominal, so (A)
may drop a lever that is neutral at nominal and better OOD. Alternative (only if Luiz amends before l2bnmw s1):
(A') `wear30` fails lower on a majority, with B-D unchanged at nominal as non-inferiority.

## Readout per END

1. Paired whole vector (stable / err / steady / pos / alt / fails, TEST) vs the primary pair, values.
2. **Stress readout** (report-only, CPU, nice 15, automatic in the driver): the CTRL-83 grid's 6 PRIMARY cells
   (`nom, wear30, mass25, lag30, lag10, noise2`) on the new winner —
   `logs/controller/ctrl83/run_grid.py --controllers <arm>_sN --cells nom,wear30,mass25,lag30,lag10,noise2 --no-row-b`
   (extra WNN ids in `ctrl83/cells.py` `EXTRA_WNN`; the `nom` cell's gate G4 must reproduce the logged END). The
   l2b sN and rival rows already exist in `ctrl83/data`. A readout failure is logged `N87 STRESS FAIL`, never stops
   the block. Cells/minutes are tie-breakers only.

## Risks

- l2bnm might cost more than "a little" (coll_cmd is large in the winners): it is still the base; the size of the
  cost is the finding.
- The stress panel's `mass25` cell is the most exposed for l2bnm (the anchor no longer tracks the mass draw).
- Cross-ABI pairing (a63 → a66) rests on the flag-off pins; any pin failure voids the pairing.
- The wider wear spread trains on plants the TEST units never fly: a nominal cost is possible (H-w says none).
- The calibration token does not record `anchor_mass` (the provenance does via [argv]); not a confound here
  (argv-validated), but worth noting for later comparisons.

## Cost

2 arms × 3 seeds × ~1 h (l2b / CTRL-80 runs: 50-71 min each) ≈ 6 h of controller time, plus ~2-3 min CPU stress
readout per END. Phase 1 (l2bnm s1-s3, flies now) ≈ 3 h; l2bnmw (≈ 3 h) waits on the code gap.

## Amendment 1 (Luiz 28/11/2026 ~06:00 EST)
- **l2bnm becomes the new base** after its cost readout (fairness baseline from here on).
- **l2bnmw = wear AND lag together** (one combined arm): training wear ±30 % and a per-episode training motor-lag
  range around 0.20 s, applied COHERENTLY to the memory writes (DAgger/training episodes) AND the GA search's
  evaluation episodes (otherwise selection would still optimise for ±20 % / fixed lag). The 35 TEST units stay at
  the reporting conditions (±20 %, lag 0.20) so in-range numbers stay comparable; the stress cells test beyond.
  The WNN's internal actuator twin stays at the nominal 0.20 s design lag. Needs code (wear flag: Python; lag range:
  Rust + Metal, controller ABI 68 after the per-bank-width branch takes 67).
- **Rule for l2bnmw = A'**: fewer wear30 fails than l2bnm on a majority of seeds; B–D kept as non-inferiority in range.
