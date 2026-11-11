# CTRL-52: bit-flip (single-event-upset) robustness, WNN cascade vs MPCOF cascade

Plane CTRL-52, approved by Luiz 10/11/2026. Simulation only. Measured 10/11/2026 on the installed `ram_controller` wheel, ABI 49. Script: `scripts/ctrl52_bitflip.py`. Figure: `docs/figures/ctrl52_bitflip.png`.

## TL;DR

- **Measured: the WNN's lookup memory degrades gracefully.** Flips in the TRUE-key table and connection table cost nothing measurable up to about 1e-4 upsets per stored bit (1,024 flips on s1, still 99.94% stable). Stability drops below 99% somewhere between 1e-4 and 1e-3 (the 4,096-to-16,384-flip grid points, depending on the genome), and collapses (≤25% stable) at about 4e-3 to 7e-3 (65,536 flips). That is thousands of independent upsets before anything visible happens.
- **Measured: the WNN's weak spot is its few floats.** The 200 float32 thermometer thresholds are 0.04 to 0.06% of the image's bits but cause most of the damage at moderate rates. At the s1 rate where the whole image gives 76% stable (1.65e-3), flipping only the key table at that rate gives 98.8%. Flipping only the thresholds at that rate (about 11 flips) gives 97.4 ± 3.9%, and at 4x that rate, 83.8 ± 22.2%. Every isolated bad draw (61%, 77.7%, 54%) at 1e-4 to 1e-3 coincides with threshold flips.
- **Measured, on a shared target: the firmware rate-PID gain table.** This is 12 float32 = 384 bits, and both cascades fly on it. Flipping each bit alone, about 8 to 10% of single flips drop either controller below 90% stable: WNN 38/384, MPCOF 29/384, on the same 7 units. These are exponent and sign bits of kp, ki and kd. No controller survives a corrupted inner loop. MPCOF additionally *refuses to run* on 14/384 flips: its planner build panics on a non-positive-definite QP or an `assert kp > 0`.
- **Not measured: MPCOF's own parameters** (condensed QP matrices, DARE terminal cost, observer gain/plant constants). There is no Python hook (see "Hook needed" below), so per the brief I stopped there; no Rust was built. **Inferred, not measured:** MPCOF stores about 2,226 doubles (about 142k bits). Suppose its matrix entries were as fragile per flip as the rate gains (about 7.6% of single flips catastrophic). Then at the 1e-4 per-bit rate where every WNN image is still ≥99.9% stable, MPCOF would take about 14 flips, and P(no catastrophic flip) would be about 0.924^14, roughly 33%. Treat this as a hypothesis for the hook to test, not a result.
- **Verdict.** "WNN degrades gracefully" is **supported for the table memory, which is 99%+ of its stored bits**. It is **not a blanket property**. The thresholds and the shared rate-PID floats are as fragile as any float controller, and they set the failure floor. The paper-safe claim: per upset, a WNN key/connection flip is about 1,000x less likely to matter than a float-parameter flip. The cheap hardening follows directly: ECC/TMR on the 6,400 threshold bits plus the 384 rate-gain bits (under 0.1% of the image). The head-to-head per-bit-rate comparison with MPCOF still needs the hook.

## Method

**Subjects (WNN).** These are the a49 dres control headline genomes, scored exactly as flown (each run's `[argv]` rebuilds the episode config; checkpoint per `[stage-select] HEADLINE`):
`C-S2-d_wposx2_a49_rf_c200_dres_full_s1` (MEMORY#1), `_s2` (NEURONS#2) and `_s3` (NEURONS#2). Also included is the block-3 dropconst winner `C-S2-d_wposx2_a45_rf_c200_dres_dropconst_full_s1` (NEURONS#0, an a45 genome rescored on the ABI-49 wheel). BINARY, sn=0, 24-bit output neurons, CTBR cascade (100 Hz outer over the 500 Hz firmware rate PID). Thresholds are refit on the TRAIN seed, as `_report_thresholds` does.

**Subject (MPCOF).** The `MPCOF>rate(h10)` cascade, est-fed, via `classical_baseline.cascade_rival_metrics`, i.e. `score_classical_baseline(teacher_id=4, ctbr_outer_hold=10)`. Same episodes, same plant.

**Scoring.** 35 CF2.1 report units (seeds 99990101..99990135) x 100 episodes. A fresh evaluator per unit gives fold 0 of the report pool, the held-out report convention. Each fault draw is one corrupted controller image flown across all 35 units. The scorer is the installed Rust scorer; Python only edits the stored data. Reproduction at k=0: s1 99.97 / 1.47 / 0.90 (one fail; the run's HEADLINE prints 100.0 / 1.47 / 0.90). MPCOF 100.00 / 1.35 / 0.68 (0 fails).

**Storage model (what a flip hits).** The bit universe is the MCU export (`mcu/README.md`, `scripts/export_controller_c.py`):
`wnn_conn` (uint8 per address bit), `wnn_keys` (uint32 per TRUE cell, each neuron's sorted on-set), `wnn_off` (uint32 x N+1) and `wnn_thresholds` (float32). BINARY stores only the written-TRUE set, so:
- A key flip deletes a TRUE entry and creates one at the XORed address. Flips of bits 24-31 create unreachable keys.
- A key flip can break the sort order and hide *other* keys from the firmware's binary search. The decode emulates `mcu/wnn_cascade.c key_present()` exactly (vectorised), so the scored on-set is the one the firmware would find. The "visible TRUE cells" column shows the effect.

Two documented approximations, both in a negligible share of bits:
- A conn index pushed past the 200 input bits would read past `frame[]` in C. It is modelled as the constant-0 bit (as for a negative index).
- An offset pushed past the key array is clamped to its end.

The firmware rate-PID gains and the CTBR/d-hat constants are **not** in the WNN universe; they are the shared table, swept separately.

**Sweep.** k ∈ {0, 1, 4, 16, 64, 256, 1,024, 4,096, 16,384, 65,536}, stopping once mean stable < 50%. Every subject collapsed at 65,536. 10 seeded draws per k (`default_rng([52, subject, k, draw])`, reproducible), uniform without replacement over the image's bits. Values are mean ± SD over draws.

**Normalisation.** k is reported as a count and as k / stored bits, i.e. upsets per stored bit, because radiation hits bits uniformly. The WNN images are 9.9M to 18.8M bits; MPCOF's own parameters are about 142k bits (f64, inferred from `ctbr_axis_mpc.rs`/`box_qp.rs`/`ctbr_mpcof.rs`: 3 axes x (ustar_map 20x8 + g_map 20x8 + H 20x20) + about 70 plant/observer/shadow scalars). So at equal per-bit rate the WNN takes about 70 to 130x more flips.

**Attribution (s1).** The same per-bit rates were applied to one segment at a time (`--only`).

## Results: persistent flips, WNN image (measured)

#### C-S2-d_wposx2_a49_rf_c200_dres_full_s1 (MEMORY#1) — 9,938,016 stored bits (conn 50,688 / keys 9,872,448 / off 8,480 / thr 6,400)

| k flips | fraction of stored bits | draws | stable % | err ° | steady ° | fails /3500 | visible TRUE cells |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00e+00 | 1 | 99.97 ± 0.00 | 1.47 ± 0.00 | 0.90 ± 0.00 | 1.0 ± 0.0 | 308,514 |
| 1 | 1.01e-07 | 10 | 99.97 ± 0.00 | 1.47 ± 0.00 | 0.90 ± 0.00 | 1.0 ± 0.0 | 308,511 |
| 4 | 4.02e-07 | 10 | 99.97 ± 0.00 | 1.47 ± 0.00 | 0.90 ± 0.00 | 1.0 ± 0.0 | 308,501 |
| 16 | 1.61e-06 | 10 | 99.96 ± 0.03 | 1.47 ± 0.00 | 0.90 ± 0.00 | 1.3 ± 0.9 | 308,451 |
| 64 | 6.44e-06 | 10 | 99.97 ± 0.01 | 1.47 ± 0.01 | 0.89 ± 0.01 | 1.1 ± 0.3 | 308,866 |
| 256 | 2.58e-05 | 10 | 99.97 ± 0.02 | 1.47 ± 0.00 | 0.89 ± 0.00 | 1.1 ± 0.7 | 307,394 |
| 1,024 | 1.03e-04 | 10 | 99.94 ± 0.05 | 1.50 ± 0.03 | 0.91 ± 0.02 | 2.1 ± 1.6 | 306,353 |
| 4,096 | 4.12e-04 | 10 | 95.76 ± 12.17 | 2.06 ± 1.21 | 1.58 ± 1.85 | 148.5 ± 425.8 | 300,715 |
| 16,384 | 1.65e-03 | 10 | 76.29 ± 24.26 | 4.40 ± 2.31 | 3.76 ± 3.51 | 830.0 ± 849.3 | 265,263 |
| 65,536 | 6.59e-03 | 10 | 1.17 ± 0.54 | 23.03 ± 2.26 | 33.11 ± 4.28 | 3459.2 ± 18.8 | 156,836 |

- first k with mean stable < 99%: **4,096** (= 4.12e-04 of stored bits; previous grid point 1,024 still ≥ 99%)

- first k with mean stable < 90%: **16,384** (= 1.65e-03 of stored bits; previous grid point 4,096 still ≥ 90%)

#### C-S2-d_wposx2_a49_rf_c200_dres_full_s2 (NEURONS#2) — 18,816,448 stored bits (conn 90,624 / keys 18,704,288 / off 15,136 / thr 6,400)

| k flips | fraction of stored bits | draws | stable % | err ° | steady ° | fails /3500 | visible TRUE cells |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00e+00 | 1 | 100.00 ± 0.00 | 1.58 ± 0.00 | 0.99 ± 0.00 | 0.0 ± 0.0 | 584,509 |
| 1 | 5.31e-08 | 10 | 100.00 ± 0.00 | 1.58 ± 0.00 | 0.99 ± 0.00 | 0.0 ± 0.0 | 584,503 |
| 4 | 2.13e-07 | 10 | 100.00 ± 0.00 | 1.58 ± 0.00 | 0.99 ± 0.00 | 0.0 ± 0.0 | 584,501 |
| 16 | 8.50e-07 | 10 | 100.00 ± 0.00 | 1.58 ± 0.00 | 0.99 ± 0.00 | 0.0 ± 0.0 | 584,453 |
| 64 | 3.40e-06 | 10 | 99.64 ± 1.14 | 1.64 ± 0.19 | 1.00 ± 0.05 | 12.7 ± 39.8 | 584,307 |
| 256 | 1.36e-05 | 10 | 100.00 ± 0.00 | 1.58 ± 0.00 | 0.99 ± 0.00 | 0.0 ± 0.0 | 583,316 |
| 1,024 | 5.44e-05 | 10 | 99.74 ± 0.82 | 1.66 ± 0.18 | 1.02 ± 0.05 | 9.1 ± 28.8 | 580,502 |
| 4,096 | 2.18e-04 | 10 | 97.73 ± 7.05 | 1.88 ± 0.80 | 1.36 ± 1.11 | 79.4 ± 246.9 | 575,106 |
| 16,384 | 8.71e-04 | 10 | 93.34 ± 14.42 | 2.55 ± 1.26 | 2.00 ± 1.98 | 233.1 ± 504.6 | 534,149 |
| 65,536 | 3.48e-03 | 10 | 25.22 ± 17.60 | 10.26 ± 3.49 | 11.04 ± 5.75 | 2617.3 ± 616.1 | 406,618 |

- first k with mean stable < 99%: **4,096** (= 2.18e-04 of stored bits; previous grid point 1,024 still ≥ 99%)

- first k with mean stable < 90%: **65,536** (= 3.48e-03 of stored bits; previous grid point 16,384 still ≥ 90%)

#### C-S2-d_wposx2_a49_rf_c200_dres_full_s3 (NEURONS#2) — 14,702,752 stored bits (conn 87,552 / keys 14,594,176 / off 14,624 / thr 6,400)

| k flips | fraction of stored bits | draws | stable % | err ° | steady ° | fails /3500 | visible TRUE cells |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00e+00 | 1 | 99.97 ± 0.00 | 1.47 ± 0.00 | 0.99 ± 0.00 | 1.0 ± 0.0 | 456,068 |
| 1 | 6.80e-08 | 10 | 99.97 ± 0.00 | 1.47 ± 0.00 | 0.99 ± 0.00 | 1.0 ± 0.0 | 456,067 |
| 4 | 2.72e-07 | 10 | 99.97 ± 0.00 | 1.47 ± 0.00 | 0.99 ± 0.00 | 1.0 ± 0.0 | 456,054 |
| 16 | 1.09e-06 | 10 | 99.98 ± 0.01 | 1.47 ± 0.00 | 0.99 ± 0.00 | 0.8 ± 0.4 | 456,000 |
| 64 | 4.35e-06 | 10 | 99.97 ± 0.01 | 1.47 ± 0.00 | 0.99 ± 0.00 | 0.9 ± 0.3 | 456,025 |
| 256 | 1.74e-05 | 10 | 99.99 ± 0.02 | 1.47 ± 0.00 | 0.99 ± 0.00 | 0.5 ± 0.5 | 455,136 |
| 1,024 | 6.96e-05 | 10 | 99.99 ± 0.01 | 1.48 ± 0.01 | 1.00 ± 0.01 | 0.4 ± 0.5 | 452,825 |
| 4,096 | 2.79e-04 | 10 | 99.86 ± 0.40 | 1.57 ± 0.07 | 1.04 ± 0.06 | 5.0 ± 14.1 | 440,397 |
| 16,384 | 1.11e-03 | 10 | 97.09 ± 3.94 | 2.31 ± 0.47 | 1.48 ± 0.28 | 101.9 ± 137.8 | 398,100 |
| 65,536 | 4.46e-03 | 10 | 13.58 ± 7.95 | 11.85 ± 3.73 | 13.92 ± 6.58 | 3024.8 ± 278.4 | 278,566 |

- first k with mean stable < 99%: **16,384** (= 1.11e-03 of stored bits; previous grid point 4,096 still ≥ 99%)

- first k with mean stable < 90%: **65,536** (= 4.46e-03 of stored bits; previous grid point 16,384 still ≥ 90%)

#### C-S2-d_wposx2_a45_rf_c200_dres_dropconst_full_s1 (NEURONS#0) — 17,413,664 stored bits (conn 49,152 / keys 17,349,888 / off 8,224 / thr 6,400)

| k flips | fraction of stored bits | draws | stable % | err ° | steady ° | fails /3500 | visible TRUE cells |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00e+00 | 1 | 99.94 ± 0.00 | 1.37 ± 0.00 | 0.78 ± 0.00 | 2.0 ± 0.0 | 542,184 |
| 1 | 5.74e-08 | 10 | 99.94 ± 0.00 | 1.37 ± 0.00 | 0.78 ± 0.00 | 2.0 ± 0.0 | 542,181 |
| 4 | 2.30e-07 | 10 | 99.94 ± 0.00 | 1.37 ± 0.00 | 0.78 ± 0.00 | 2.0 ± 0.0 | 542,175 |
| 16 | 9.19e-07 | 10 | 99.94 ± 0.00 | 1.37 ± 0.00 | 0.78 ± 0.00 | 2.0 ± 0.0 | 542,835 |
| 64 | 3.68e-06 | 10 | 99.94 ± 0.00 | 1.37 ± 0.00 | 0.78 ± 0.00 | 2.0 ± 0.0 | 541,920 |
| 256 | 1.47e-05 | 10 | 99.95 ± 0.01 | 1.37 ± 0.00 | 0.78 ± 0.00 | 1.9 ± 0.3 | 541,760 |
| 1,024 | 5.88e-05 | 10 | 99.89 ± 0.16 | 1.39 ± 0.05 | 0.78 ± 0.02 | 3.8 ± 5.7 | 540,767 |
| 4,096 | 2.35e-04 | 10 | 99.90 ± 0.04 | 1.44 ± 0.06 | 0.81 ± 0.05 | 3.5 ± 1.3 | 531,973 |
| 16,384 | 9.41e-04 | 10 | 98.49 ± 1.11 | 1.91 ± 0.23 | 1.04 ± 0.20 | 52.7 ± 38.8 | 490,673 |
| 65,536 | 3.76e-03 | 10 | 24.32 ± 11.12 | 9.47 ± 1.96 | 9.46 ± 3.23 | 2648.8 ± 389.1 | 339,826 |

- first k with mean stable < 99%: **16,384** (= 9.41e-04 of stored bits; previous grid point 4,096 still ≥ 99%)

- first k with mean stable < 90%: **65,536** (= 3.76e-03 of stored bits; previous grid point 16,384 still ≥ 90%)


### Which stored bits do the damage (s1, measured; 10 draws each)

| segment flipped alone | k (≈ whole-image rate 1.65e-3) | stable % | err ° | steady ° | k (≈ 6.6e-3) | stable % | err ° | steady ° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| keys (99.3% of bits) | 16,275 | 98.81 ± 0.38 | 2.00 ± 0.08 | 1.15 ± 0.07 | 65,100 | 37.46 ± 5.08 | 7.32 ± 0.35 | 6.01 ± 0.62 |
| thresholds (0.06%) | 11 | 97.35 ± 3.86 (min 90.6) | 1.95 ± 0.60 | 1.39 ± 0.70 | 42 | 83.81 ± 22.20 (min 24.3) | 3.40 ± 2.01 | 2.97 ± 3.68 |
| conn (0.5%) | 84 | 99.74 ± 0.16 | 1.59 ± 0.07 | 0.94 ± 0.05 | 334 | 97.21 ± 1.58 | 2.16 ± 0.21 | 1.25 ± 0.17 |
| off (0.09%) | 14 | 99.75 ± 0.34 | 1.60 ± 0.10 | 0.96 ± 0.07 | 56 | 94.97 ± 5.02 | 2.51 ± 0.41 | 1.60 ± 0.49 |
| whole image | 16,384 | 76.29 ± 24.26 | 4.40 ± 2.31 | 3.76 ± 3.51 | 65,536 | 1.17 ± 0.54 | 23.03 ± 2.26 | 33.11 ± 4.28 |

How to read this:
- **The key table degrades smoothly.** Low variance and monotone: err and steady creep up as TRUE entries vanish, because the antagonist decode pulls toward neutral.
- **The variance comes from a handful of float flips.** At 4,096 flips the s1 61.1% draw had 8 threshold flips, and the s2 77.7% draw had 3.
- **Per flip, the floats matter far more than the table.** One threshold flip is worth roughly 1,000 key flips: 11 threshold flips cost about as much as 16k key flips.
- **The segments interact.** The whole image at 1.65e-3 (76%) is worse than any single segment alone. Thresholds plus key loss compound.

## Results: shared firmware rate-PID gains (measured)

Each of the 384 bits of the 12 float32 gains ([kp, ki, kd, i_limit] x roll/pitch/yaw) was flipped alone and flown. WNN: `spec.ctbr["rate_gains"]`, which the WNN cascade's inner loop flies. MPCOF: `af_pid_rate`, which feeds its flying rate loop, its shadow and its planner model, i.e. a boot-time corruption the planner is then derived from. The WNN sweep used the first 7 units (65 s per configuration at 35 units would have taken 7 h); MPCOF is shown at 7 and at 35.

| controller | units | baseline stable % | single flips < 99% | < 90% | < 50% | crashes (controller refused to run) |
|---|---:|---:|---:|---:|---:|---:|
| WNN cascade (s1) | 7 | 100.00 | 43 / 384 | 38 | 30 | 0 |
| MPCOF>rate(h10) | 7 | 100.00 | 31 / 384 | 29 | 21 | 14 |
| MPCOF>rate(h10) | 35 | 100.00 | 34 / 384 | 29 | 21 | 14 |

- **The harmful bits are exponent and sign bits of kp, ki and kd** for both controllers. Mantissa flips are harmless.
- **The WNN is somewhat more sensitive here**, 38 vs 29 flips below 90%. MPCOF re-derives its plan from the corrupted gains, so its model stays consistent with the loop it flies. The WNN was trained on the nominal loop.
- **A "crash" is a Rust panic while building MPCOF's planner.** Examples are a kp sign flip (`assert kp > 0`) and an exponent flip making `H_FF` non-positive-definite. These are scored as all-episodes-failed. On hardware this is a dead control task unless the firmware has a fallback.
- **This table is common-mode.** It belongs in both controllers' fault budget, and it is about 1/30,000 of the WNN image.

## Not done (needs code; nothing built)

**(A) MPCOF's own parameters: hook needed.** `score_classical_baseline` (`dagger_train.rs:3683`) builds the teacher internally (`af.teacher(teacher_id)`, `dagger_train.rs:~3905` → `AirframeRs::teacher`, `dagger_train.rs:1179` → `CtbrMpcOf::new`, `ctbr_mpcof.rs:71`). The condensed planners are built lazily on the first decision (`CtbrMpcOf::planner_index`, `ctbr_mpcof.rs:~200` → `CtbrAxisMpc::build`, `ctbr_axis_mpc.rs:194`). No API takes perturbed matrices. Minimal hook (new controller ABI, worktree build, not installed):
1. `CtbrMpcOf::corrupt(&mut self, n_o, flips: &[(seg u8, idx u32, bit u8)])`. It forces `planner_index(n_o, phase)` for every phase, then XORs bits in `CtbrAxisMpc.{ustar_map, g_map}`, `BoxQpSolver.h` (`box_qp.rs:27`), `AxisPlant` fields (`ctbr_axis_mpc.rs:65`), `CascadeDhat.{c, alpha, clamp}` (`cascade_dhat.rs:40`) plus the L gain, and the shadow `RateStageFw` gains. It needs a `Mat::data_mut()` accessor.
2. A `param_flips: Vec<(u8,u32,u8)> = vec![]` kwarg on `score_classical_baseline` that calls it once after `af.teacher(...)`. The planners persist across episodes, so the corruption is persistent. Empty means bit-identical.

Then this script's `mpcof` mode can sweep k the same way, for the true per-bit-rate head-to-head.

**(B) Transient upsets in runtime state.** Neither controller's state is reachable from Python mid-episode. MPCOF needs a kwarg `transient_flip=(step, target, bit)` in the same scorer's episode loop, acting on `CtbrMpcOf.obs.dhat`, `CtbrAxisMpc.warm` and the shadow integrator. The WNN needs the same in its rollout (`cpu_score.rs` and the Metal twin `controller_rollout.metal`), acting on the integrals, the yaw heading, the held setpoint and the frame. Skipped per brief.

## Measured vs inferred

- **Measured:**
  - every table above;
  - the k=0 reproductions;
  - the segment attribution;
  - the rate-gain sweep.
- **Inferred:**
  - MPCOF's stored-bit count (from code, not exported);
  - the about-33% MPCOF survival estimate at 1e-4 per bit;
  - "WNN about 1,000x less sensitive per key flip than per float flip" is a ratio of measured operating points, not a fitted AVF;
  - the real-hardware effect of an out-of-range conn index or offset (approximated, under 0.6% of bits).
- **Caveats:**
  - n = 3 genomes plus 1 for the main sweep, 10 draws per k;
  - the rate-gain sweep covers one WNN genome on 7 units;
  - the grid is 4x-spaced, so the 99%/90% crossings are brackets, not points.
