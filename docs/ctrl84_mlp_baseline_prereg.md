# CTRL-84 — MLP baseline under the WNN's own DAgger (prereg, 28/11/2026)

Approved design: Luiz 28/11/2026, option (b) — DAgger with the MLP as the student, same teacher,
same inputs, same scorer as the WNN. Branch `ctrl84-mlp-baseline`, controller ABI 71 (69 = the
joint-OOD cell, merged; 70 = latency + inertia, not merged). Driver `logs/controller/ctrl84/run_mlp.sh`.
**This document is written before any cohort training.** The only MLP flown so far is the smoke run
(§Smoke), which is shrunk and is not a result.

## Question

Is the WNN's in-range and OOD vector within noise of a conventional dense network trained on
exactly the same data, inputs and outputs, and what does each cost per decision?

## What is held equal (the "same information" claim)

| | WNN l2bnm sN | MLP arm (a) / (b), seed N |
|---|---|---|
| inputs | the 24 pre-thermometer features of `compute_features` (Rust, one source of truth), thermometer-encoded | the SAME 24 features, standardised (mean/std of the first fit's records; the 3 constant `tgt_*` inputs get scale 0 — the WNN drops them too) |
| what is learned | per bank (T, p, q, r) the raw decode in [0, 1] — BINARY antagonist vote, 64 levels | per bank the raw decode in [0, 1]: linear output, clamped to [0, 1] at deploy |
| downstream | CTBR alphabet (range 0.25/4.2/4.2/2.0, γ 1/2/2/3), delta law (leak 0.7, δmax 0.6, yaw 0.5), firmware rate PID at 500 Hz, 100 Hz outer loop | identical code (the head replaces only `raw_bank_decode`) |
| training target | the teacher label encoded to that raw decode (`output_target_at`), written as thermometer bits | the same value (`output_target_at`), MSE regression |
| teacher / DAgger | MPCOF, 8 rounds × 24 episodes × 2000 steps, tilt 8°→30°, gate off, best-checkpoint, student flies, 5 fold seeds accumulated (train_seed·100 + k) | identical config object (`ControllerEvaluator.packed_train_config`) and seeds; data AGGREGATED over all rounds and folds; PyTorch fit between rounds; rollouts fly the Rust forward |
| training plant | cf21, L4C, wear ±20 % per episode, lag 0.20 s, **nominal-mass anchor**, specific-force IMU, rotor drag 0.78 | the same (the l2bnm sN argv is parsed from its `.out`) |
| scorer | `ControllerEvaluator` → Rust `score_controllers_cpu` | the same call (`ControllerEvaluator.score_mlp`), captured and gated by `ctrl83/score_cell.py` (G1 re-issue, G2 plant identity with the rivals, G3 stress reached) |

Input choice: real-valued features, not thermometer bits. The bits are a lossy 16-level quantisation
of these features, so the MLP gets at least the WNN's information. That is the fairer baseline for
the MLP. The paper says: "the same sensor features, before the WNN's input encoding".
Output choice: regression to the WNN's own decode target, not a softmax over the 65-value alphabet.
The WNN's head is an ordinal vote count (0.5 + (ΣE − ΣI)/64), not a categorical distribution, so
regression keeps the ordinality, and the shared downstream decode keeps the alphabet and scaling identical.

Not equal by construction:
- The WNN's GA (grid → neurons → memory, Lamarckian) adds more DAgger passes and an architecture
  search on top of one training call. The MLP gets one training call (5 folds × 8 rounds × 24 episodes
  = 960 episodes, ≈192k decision records) plus an HP search on VAL (below).
- The WNN's vote write rule (share 0.1) has no MLP counterpart. MSE is its dense analogue.

## Arms

| arm | hidden | params | fp32 bytes | int8 bytes | MAC / decision |
|---|---|---|---|---|---|
| (a) | 2 × 64, ReLU | 6,020 | 24.1 kB | 6.0 kB | 5,912 |
| (b) | 2 × 576, ReLU | 349,060 | 1.396 MB | 349 kB | 347,928 |

The story's "~21 kB" for (a) assumed 21 inputs. With 24 inputs it is 24.1 kB.

**"Matched" for (b) = fp32 weight bytes ≈ the WNN's table bytes.** The l2bnm winners hold 296,260 /
386,398 / 356,077 TRUE keys (s1/s2/s3, mean 346,245). The BINARY tables store only TRUE keys, and
the C exporter packs each in ~4 B: CTRL-51 measured 308,514 keys → 1,247 KB. So the WNN's footprint
is ≈ 4 B × keys = 1.39 MB (range 1.19–1.55 MB), and (b) at fp32 is 1.40 MB (+0.8 % of the mean).
One (b) size for all seeds, not one per seed. Equivalently, the parameter count ≈ the TRUE-key count.
At int8, (b) would be 4× smaller than the WNN. That is reported as a cost row, not flown.

## Seeds and pairing

n = 5, seed N paired with l2bnm sN (same argv, train seed, fold seeds, VAL units, TEST episodes).
**l2bnm s4 and s5 do not exist yet.** The MLP s4/s5 cannot be trained until they fly: their argv,
train seed and VAL units come from the paired run. s1–s3 can fly now.

## HP grid (VAL only; pre-declared)

lr ∈ {1e-3, 3e-4} × epochs per DAgger round ∈ {4, 16} × weight decay ∈ {0, 1e-4} = 8 configs
(`mlp_baseline.py HP_GRID`, indices 0–7). Fixed: Adam, batch 1024, ReLU, MSE. The output layer
starts at zero weights with bias = the decode neutral, so round 0 flies the empty-WNN hold policy.
Torch seed = 1000·N + config.

Each config is trained and then scored on the paired run's **20 VAL units** (its stage-select units,
100 episodes each, with the TEST evaluator construction). **Selection, lexicographic, on VAL only:**
fewest fails, then lowest mean err, then lowest steady. The selected head is scored ONCE on the 35
TEST units and on the joint-OOD cell `all`. No TEST number feeds any choice.

## Metrics (report)

The whole vector, per seed, paired with l2bnm sN: stable % / err ° / steady ° / pos m / alt m / fails
(/3500).
- In-range TEST row: 35 CF2.1 units 99990101..135 × 100 episodes (`score_cell.py mlp<arm>_sN nom`).
- Joint-OOD cell `all`: wear ±30 % + mass ±25 % + lag U(0.15, 0.30) s + noise ×2 (`score_cell.py mlp<arm>_sN all`).
- Readout: `stress_table.py "WNN l2bnm=l2bnm" "MLP a=mlpa" "MLP b=mlpb"`, the standard ctrl83 orientation.

## Hypotheses (decision rule, Luiz's usual whole-vector reading)

- **H1 (parity at no multiplies):** the WNN is within noise of the MLP. For each of arm (a) and arm (b):
  on the in-range TEST row, err and steady differ by less than the seed-to-seed SD of l2bnm (s1–s3:
  err SD ≈ 0.04°, steady ≈ 0.02°) in mean over the paired seeds, and fails are not worse by more than
  2 in total. The same check runs on the `all` cell.
- **H2 (MLP clearly ahead → feasibility framing):** the MLP is better on err OR steady on ≥ 4/5 paired
  seeds by more than that SD, with fails not worse. Then the paper frames the WNN as a
  multiply-free, feasible alternative, not a parity result.
- Anything else (mixed, the WNN ahead) is reported as values, without a verdict word.

## Cost table (per decision = one 100 Hz outer-loop step)

| | multiplies | adds | parameter/table bytes read | µs CPU (Rust, M4 Max, release) |
|---|---|---|---|---|
| WNN l2bnm | 0 | ~0 (384 threshold compares, 256 × 24-bit address gathers, vote count) | ≈ 256 neurons × ~11 binary-search probes × 4 B ≈ 11 kB of a 1.4 MB table | 5.99 per 10-tick window (whole controller, 346k random keys) |
| MLP (a) | 5,912 | 5,984 | 24.3 kB | 2.37 forward; 3.16 per window |
| MLP (b) | 347,928 | 349,084 | 1.40 MB | 260.96 forward; 270.08 per window |

Measured 28/11 (release, nice 15, box loaded by the IDS worker). The MLP forward is the scalar fma chain
that gives CPU/GPU bit parity (no SIMD reordering), so (b)'s µs is an upper bound for an optimised
GEMV. The multiply count and bytes read are layout facts.

µs comes from `cargo test --release ... ctrl84_bench -- --ignored --nocapture`: the forward alone,
and one whole decision window (features + head or lookup + inner loop) with a 256 × 24-bit WNN holding
346k random keys.

## Cost of the campaign

Estimate per (arm, seed): 8 configs × (training + 20 VAL units × 100 episodes), plus TEST nom + all,
scaled from the smoke and the bench. A single controller scores serially, so (b) is forward-bound.
- (a): ≈ 2 min per config → ≈ 20 min per seed.
- (b): ≈ 7 min per config (rollouts ~1 min, fits ~3 min on MPS, VAL ~2 min) → ≈ 1 h per seed,
  plus ~6 min of TEST.
- n = 5 × 2 arms ≈ 6.5 h. s1–s3 (flyable now) ≈ 4 h. Everything runs at nice 15
on the CPU scorer. The PyTorch fit runs on MPS (or CPU).

## Risks

- The MLP's DAgger budget is ONE training call. The WNN's lineage got many through the GA. If the MLP
  under-fits, the HP grid's epochs axis is the only lever. A WNN win would then be partly a budget
  effect: state it.
- (b) at 2 × 576 is wide for ~192k records and may over-fit. Weight decay is on the grid; VAL selects.
- Non-determinism: MPS kernels are not bitwise deterministic, so the same config may not reproduce
  exactly. The Rust rollout and scorer are deterministic (cargo test).
- s4/s5 depend on l2bnm s4/s5 flying first.
- CPU vs Metal scoring of an MLP agrees to the CTRL-54 tolerance, not bit for bit (the feature
  pipeline differs). The forward itself is bit-identical (ReLU). Every report row is CPU.

## Smoke (28/11, NOT a result: shrunk, against an UNPACKED ABI-71 wheel, never installed)

`mlp_baseline.py --arm a --seed 1 --hp 0 --rounds 3 --episodes 6 --folds 2 --val-units 2 --val-episodes 20`:
- trains in 3 s (5,000 records) and selects on VAL.
- `score_cell.py mlpa_s1 nom`: G1/G2 gates pass, ABI 71, 15 s. Stable 7.1 %, err 9.48°, 3251/3500 fails.
  With ~1/30 of the DAgger budget that is expected; it is a plumbing check only.
- `all`: 18 s.

Flag-off: `tests/controller_ctrl84_flag_off.py` passes vs ABI 69 on the CPU and the default device.
