# CTRL-90 prereg: `l2bnmwz`, l2bnmw trained ON the Zhang golden cell's ranges (Luiz 28/11/2026)

Written 28/11/2026, before any l2bnmwz flight. Driver `logs/controller/ctrl90/ab_z90.sh` (controller ABI 69).
Yardstick: the CTRL-83 golden joint-OOD cell `all`, re-levelled by Luiz on 28/11 to Zhang 2022's rule (test = train
lower ×0.8 / upper ×1.2) on the FIXED l2b reference ranges (`docs/ctrl83_stress_panel_prereg.md` §11, Amendment 3;
`logs/controller/ctrl83/cells.py` `ALL_LEVELS`). Base: l2bnmw (adopted BASE, Luiz 28/11).

## Why

Luiz (28/11): "train an arm for it". The arm is trained and searched ON the golden cell's own ranges, wherever a
training-only flag exists. **This means the golden cell is (partly) IN-DISTRIBUTION for this arm.** A gain on `all`
is therefore **covered-range robustness, not OOD generalization**, and the paper must report it that way: "trained
on the stress ranges, robust across them". It must never be presented as "generalizes to unseen plants". The honest
OOD read for this arm is the SECONDARY cell `allz90` (below), which sits 20 % beyond its OWN training ranges.

## Arm (argv = the flown l2bnmw sN DEDUPED argv with exactly 3 value tokens changed; tag `C-S2-d_wposx2_a69_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2bnmwz_full_sN`)

```
axis       | `all` (golden cell)        | l2bnmw trains           | l2bnmwz trains                       | in-dist for l2bnmwz?
-----------+----------------------------+-------------------------+--------------------------------------+---------------------
wear       | factor U(0.64, 1.44)       | ±0.30 (0.70-1.30)       | --train-motor-asym-mag 0.44          | YES (superset: 0.56-1.44,
           |                            |                         |   (0.56-1.44)                        |   symmetric fallback)
lag        | T_e ~ U(0.16, 0.24) s      | U(0.15, 0.30)           | --train-motor-lag-range 0.16 0.24    | YES (exact)
mass       | ±0.38 (0.62-1.38)          | ±0.15                   | ±0.15 (unchanged, no training flag)  | NO, OOD
noise      | ×1.2                       | ×1                      | ×1 (unchanged, no training flag)     | NO, OOD
inertia    | ±20 % (pending ABI 70)     | nominal                 | nominal (ABI 70 not installed)       | (pending)
latency    | 8-12 ms (pending ABI 70)   | none                    | none (ABI 70 not installed)          | (pending)
```

Everything else is l2bnmw sN byte for byte: anchor nominal, TEST 35 units at L4C ±0.20 / lag 0.20 / mass ±0.15,
VAL stage-select, teachers and actuator twin at 0.20, `--num-eval-folds 5`, rule P.
Note: the lag range is NARROWER than l2bnmw's (0.16-0.24 vs 0.15-0.30), because the brief puts training exactly on the
cell's range. Any difference vs l2bnmw therefore mixes "wider wear" with "narrower lag".

## Code gaps (stated; no src/ edits in this prereg)

1. **Asymmetric training wear.** `--train-motor-asym-mag` replaces only the magnitude. The base multiplier that makes
   the TEST cell exact (1.04 × (1 ± 0.3846)) lives on `disturbance.motor_asym`, which TEST shares. Exact 0.64-1.44
   training needs a training-only base or range flag (e.g. `--train-motor-wear-range LO HI` → `EpisodeConfig.
   train_motor_asym_base` + mag, threaded like `train_asym_mag()` into `_train_asym_kwargs` / the GA pools).
   Fallback used here: symmetric ±0.44 (0.56-1.44, a superset, 0.08 harsher at the low end).
2. **Training-only mass spread.** `--mass-jitter` feeds the TEST units and VAL too. A flag like `--train-mass-jitter`
   (on the pattern of `--train-motor-asym-mag`: training hosts + GA pools only) is needed. The cell's mass is itself a
   symmetric fallback (±0.38) for the exact 0.68-1.38.
3. **Training-only sensor noise.** No flag exists (`--disturbance` sets noise for train and TEST alike). This would need
   e.g. `--train-noise-scale K`.
4. **Inertia / latency.** ABI 70 (`--train-inertia-jitter`, `--train-latency-range`, branch ctrl89-latency-inertia)
   exists but is not installed. The arm flies on ABI 69 without them.

With gaps 2-4 closed, "trained on the golden cell" would be complete. As flown, the arm covers wear + lag only.

## Rule (n = 3, pairs l2bnmwz sN vs l2bnmw sN, N = 1, 2, 3)

**The rule reads ON THE GOLDEN CELL `all`** (Luiz 28/11: "arms are judged there"): 35 TEST units × 100 episodes under
the Zhang joint draw, `ctrl83/data/<k>__all.npz`. Luiz's usual rule applies: better only if consistently better in
the expected direction and not worse elsewhere, with no magnitude bar. Drop at n=3 if, on `all`:
(A) err is not lower than l2bnmw's on a majority of seeds, or (B) steady is higher on a majority, or
(C) the fails sum exceeds l2bnmw's + 2, or (D) there are more fails than l2bnmw on a majority.
Stable is floored from fails. The whole vector (stable / err / steady / pos / fails) is shown as values.
A survivor goes to n=5 on Luiz's word.

**In-range TEST (`nom`) is a REFERENCE row**: shown as values next to `all`, with no veto in A-D. Its job is gate G4,
reproducing the logged END.

**SECONDARY (report-only, no rule):** `allz90` = Zhang ×0.8/×1.2 on l2bnmwz's OWN training ranges: wear 0.448-1.728
(exact), lag U(0.128, 0.288), mass ±0.38, noise ×1.2. Scored on l2bnmwz sN AND l2bnmw sN. This is the arm's honest OOD
generalization read. Its bounds exceed the golden cell's on wear and lag only, since mass and noise were not trained.

## Readout per END (automatic in the driver, CPU, nice 15)

1. `run_grid.py --controllers l2bnmwz_sN --cells nom,all --no-row-b` (nom = G4 vs the logged END).
2. `run_grid.py --controllers l2bnmwz_sN,l2bnmw_sN --cells allz90 --no-row-b`.
3. `stress_table.py` (the arm row joins automatically; `--diagnostic` shows `allz90` / `all_v1`).
A readout failure is logged as `Z90 STRESS FAIL` and does not stop the block.

## Risks

- Training at wear factor 0.56 (one motor at 56 %) may be beyond what the plant can recover from at some ICs. DAgger
  labels from a saturated teacher could then teach the student noise. That is a nominal cost, read in the reference row.
- In-distribution on `all` by construction (wear, lag): a win is the expected direction and supports a covered-range
  claim only.
- If ABI 70 is installed before launch, the ABI-69 assert aborts. Either relaunch on 70 (new `_a70_` lineage, plus
  inertia/latency training) or pin. This is Luiz's call.

## Cost

3 flights × ~l2bnmw's wall time (one at a time, yields to the IDS worker), plus ~1 min of stress scoring per END.
