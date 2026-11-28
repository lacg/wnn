# CTRL-90 prereg: `l2bnmwz`, l2bnmw trained ON the Zhang golden cell's ranges (Luiz 28/11/2026)

Written 28/11/2026, before any l2bnmwz flight. Driver `logs/controller/ctrl90/ab_z90.sh` (controller ABI 70;
amended 28/11 before any flight: the ABI-69 draft trained wear + lag only, see §Amendment 1).
Yardstick: the CTRL-83 golden joint-OOD cell `all`, re-levelled by Luiz on 28/11 to Zhang 2022's rule (test = train
lower ×0.8 / upper ×1.2) on the FIXED l2b reference ranges (`docs/ctrl83_stress_panel_prereg.md` §11, Amendments 3+4;
`logs/controller/ctrl83/cells.py` `ALL_LEVELS`). Base: l2bnmw (adopted BASE, Luiz 28/11).

## Why

Luiz (28/11): "train an arm for it". The arm is trained and searched ON the golden cell's own ranges, wherever a
training-only flag exists. **This means the golden cell is (partly) IN-DISTRIBUTION for this arm.** A gain on `all`
is therefore **covered-range robustness, not OOD generalization**, and the paper must report it that way: "trained
on the stress ranges, robust across them". It must never be presented as "generalizes to unseen plants". The honest
OOD read for this arm is the SECONDARY cell `allz90` (below), which sits 20 % beyond its OWN training ranges.

## Arm (argv = the flown l2bnmw sN DEDUPED argv with 2 flags replaced in place + 4 training-only flags appended; tag `C-S2-d_wposx2_a70_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2bnmwz_full_sN`)

```
axis       | `all` (golden cell)        | l2bnmw trains           | l2bnmwz trains                         | in-dist for l2bnmwz?
-----------+----------------------------+-------------------------+----------------------------------------+---------------------
wear       | factor U(0.64, 1.44)       | ±0.30 (0.70-1.30)       | --train-motor-asym-range 0.64 1.44     | YES (exact)
           |                            |                         |   (replaces --train-motor-asym-mag 0.30 in place)
lag        | T_e ~ U(0.16, 0.24) s      | U(0.15, 0.30)           | --train-motor-lag-range 0.16 0.24      | YES (exact)
mass       | ±0.38 (0.62-1.38)          | ±0.15                   | --train-mass-jitter 0.38               | YES (exact)
noise      | ×1.2                       | ×1                      | --train-noise-scale 1.2                | YES (exact)
inertia    | ±20 % (ABI 70)             | nominal                 | --train-inertia-jitter 0.20            | YES (exact)
latency    | 8-12 ms (ABI 70)           | none                    | --train-latency-range 8 12             | YES (exact)
```

All six are TRAINING-only (training hosts + GA-search pools). Everything else is l2bnmw sN byte for byte: anchor
nominal, TEST 35 units at L4C ±0.20 / lag 0.20 / mass ±0.15 / noise ×1 / nominal inertia / no latency, VAL
stage-select, teachers and actuator twin at 0.20, `--num-eval-folds 5`, rule P.
Note: the lag range is NARROWER than l2bnmw's (0.16-0.24 vs 0.15-0.30), because the brief puts training exactly on the
cell's range. Any difference vs l2bnmw therefore mixes "the golden cell's ranges" with "narrower lag".

## Amendment 1 (28/11, before any flight): the full golden cell

The ABI-69 draft trained wear (symmetric fallback ±0.44) + lag only: no training-only flag existed for asymmetric wear,
mass, noise, inertia or latency (the draft's "Code gaps" 1-4, in git history). ABI 70 closes all four (`--train-motor-asym-range`,
`--train-mass-jitter`, `--train-noise-scale`, `--train-inertia-jitter`, `--train-latency-range`), and the golden cell
`all` itself gained inertia ±20 % + latency 8-12 ms (ctrl83 prereg Amendment 4). The arm now trains on the FULL cell,
so `all` is fully IN-DISTRIBUTION for it. The lineage moves to `_a70_`; no `_a69_` l2bnmwz run was flown.

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

**SECONDARY (report-only, no rule):** `allz90` = Zhang ×0.8/×1.2 on l2bnmwz's OWN training ranges (Amendment 1):
wear factor U(0.512, 1.728) (exact), lag U(0.128, 0.288) (exact), mass ±0.656 (symmetric fallback for 0.496-1.656),
noise ×1.44 (the interval's worst case), inertia ±0.44 (symmetric fallback for 0.64-1.44: inertia_jitter has no base
factor), latency 6.4-14.4 ms (exact; rounded to physics steps by the scorer). Scored on l2bnmwz sN AND l2bnmw sN. This
is the arm's honest OOD generalization read; it exceeds the golden cell on every axis.

## Readout per END (automatic in the driver, CPU, nice 15)

1. `run_grid.py --controllers l2bnmwz_sN --cells nom,all --no-row-b` (nom = G4 vs the logged END).
2. `run_grid.py --controllers l2bnmwz_sN,l2bnmw_sN --cells allz90 --no-row-b`.
3. `stress_table.py` (the arm row joins automatically; `--diagnostic` shows `allz90` / `all_v1`).
A readout failure is logged as `Z90 STRESS FAIL` and does not stop the block.

## Risks

- Training at wear factor 0.56 (one motor at 56 %) may be beyond what the plant can recover from at some ICs. DAgger
  labels from a saturated teacher could then teach the student noise. That is a nominal cost, read in the reference row.
- In-distribution on `all` by construction (every axis): a win is the expected direction and supports a covered-range
  claim only.
- Mass ±0.38 in training (a 62 % airframe) and the `allz90` mass ±0.656 (a 34 % airframe) are severe; the latter is the
  symmetric fallback and harsher than Zhang's exact low bound (0.496).
- The driver asserts ABI 70: it aborts on an ABI-69 wheel.

## Cost

3 flights × ~l2bnmw's wall time (one at a time, yields to the IDS worker), plus ~1 min of stress scoring per END.
