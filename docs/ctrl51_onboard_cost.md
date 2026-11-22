# CTRL-51 — on-board cost: WNN cascade vs the MPCOF cascade teacher (10/11/2026)

**Question.** The paper says the WNN is cheaper on board than the controller it
imitates. Here both run on the same Cortex-M4 bench, on real trajectories, and
that claim becomes a number.

**Short answer.** *The claim does not hold as stated.* For the current dres
winner, the WNN's **average** cost is not lower than MPCOF's. In double
precision MPCOF averages 161k instructions per 10 ms window and the WNN 257k.
Its 1.2 MB of flash is about 30× MPCOF's 42 KB. Two things do hold:

- **The WNN's cost is bounded and flat:** its worst window is 1.006× its mean.
- **Exact MPCOF is not bounded:** when the rate box binds, its exact box QP
  costs 1.1–2.2M instructions in software double. That is 7–14× its mean, and
  ~20 ms at 168 MHz for a decision that must land within a 1 ms tick.

If MPCOF is run in float32 its average cost falls far below the WNN's (see the
f32 rows). That build's setpoints deviate ≤ 2.8e-6 rad/s from the f64 teacher
on these trajectories (open loop).

## What was built (all under `mcu/`, repo Rust untouched)

| piece | file |
|---|---|
| MPCOF outer loop, online part only: observer, shadow rate stage, state build, per-axis plan + exact box QP (PDAS warm start, primal fallback) | `mcu/mpcof_cascade.[ch]` |
| Offline tables: the condensed horizon-20 QP (H, g_map, ustar_map) and its DARE terminal cost, per axis and rate-loop phase, plus the observer and rate-stage constants | `mcu/mpcof_model.h` (35 KB of f64), made by `mcu/ctrl51/export_mpcof.py` |
| The offset-free observer, now ONE C definition shared by the WNN and MPCOF (as in Rust `cascade_dhat.rs`) | `mcu/cascade_dhat.[ch]` |
| Rust reference: a `#[cfg(test)]` module added to a **/tmp copy** of the controller workspace. It builds the tables with the teacher's own `outer_map → dare_doubling → condense_horizon` and flies the teacher to record episodes | `mcu/ctrl51/ctrl51_export.rs` |
| Host parity (C vs Rust episodes, bit equality) | `mcu/ctrl51/mpcof_parity_host.c` |
| M4 bench (semihosting replay, per-call markers, on-target parity) | `mcu/bench_ctrl51.c`, `mcu/run_bench_ctrl51.sh` |
| Per-region instruction / soft-double / libm / VFP counter (QEMU TCG plugin) | `mcu/insnmark.c` |
| WNN input-stream recorder (installed wheel, via the parity test's own recorder) | `mcu/ctrl51/record_wnn_flights.py` |
| Table generator | `mcu/ctrl51/summarize_bench.py` |

No Python reimplements Rust logic. The tables come from Rust functions, and
every reference label comes from the Rust teacher (`Teacher::ctbr_label`).

### Exporter bug found and fixed on the way (affects the WNN side)

`scripts/export_controller_c.py` wrote the a49 **dres** winner as `CASCADE
(runnable)`. But `wnn_cascade.c` had no CTRL-48 residual block, so the three
`cresid` features stayed at **0**. The parity test fails on it at step 1:
`feature[22] rust=0.0038 c=0.0`.

- **Fix in the exporter:** it now emits `WNN_OBS_DRES` and the observer plant.
- **Fix in the C:** it computes `r = d̂·inv_a + w·(I/L)` (`cascade_obs.rs
  CasResidual`).
- **Guard for old headers:** a header without `WNN_OBS_DRES` defaults to 0, and
  `wnn_cascade.h` says to re-export.
- **Re-verified:**
  - the a49 dres winner is now bit-identical over 4,000 host ticks;
  - the default C-S2-d parity run still passes (6,000 ticks plus the synthetic
    rate-I, d-hat and both cases);
  - the mutant is still caught.
  - Command: `test_cascade_parity.py --winner <a49 dres winner>`.

## Method

**Platform.** QEMU `mps2-an386` (Cortex-M4F), `-O2 -ffp-contract=off`, newlib-nano
`libm` + `libgcc`, from the Arm GNU Toolchain 15.3.rel2. The homebrew gcc has no
newlib, which the MPCOF build needs (atan2/asin/sqrt in double).

**Trajectories.**

- **MPCOF:** the Rust teacher flying its own cascade (MPCOF over the firmware
  rate PID). Setup: cf21_brushless, motor lag 0.20 s, L4C weather, stage-2
  position loop, specific-force IMU, rotor drag 0.78, n_o = 10, 2000 ticks per
  episode. Three regimes:

  | regime | episodes | initial tilt | initial rate |
  |---|---|---|---|
  | nominal | 20 | 5° | production |
  | stress | 10 | 30° | 2 rad/s |
  | upset | 20 | 60° | 8 rad/s |

  The upset regime is a worst-case probe: it is where the rate box binds.
- **WNN:** the a49 dres s1 winner (264 neurons × 24 bits, 308,514 TRUE keys)
  flying the run's own plant. 5 × 2000 ticks, recorded from the installed
  wheel (ABI 49).

**Counting.**

- `insnmark.c` counts the retired instructions of every controller call
  between two marker instructions. The marker overhead is measured (1 instr)
  and subtracted. This gives a mean and a worst case per call, not just a mean.
- **Cost split:** the WNN's d-hat observer and its Mahony estimator are each
  measured on a twin instance fed the same inputs. Same code and state
  trajectory, so the twin costs what the copy inside `wnn_tick` costs.

**Measured vs estimated.**

- **Measured:** instruction counts, soft-double / libm / VFP counts, flash and
  RAM (from the ld map), stack high-water mark, and parity.
- **Estimated:** cycles = 1.5 × instructions, and energy.
- **Energy model:** STM32F405 DS8626, "Run mode, data processing from Flash,
  ART on", 168 MHz, typical at 25 °C: ~40 mA with peripherals disabled
  (0.786 mW/MHz at 3.3 V) and ~87 mA enabled (1.709 mW/MHz). Energy is taken
  as cycles × mW/MHz (linear in f, so the clock cancels). At low clocks static
  current makes the real per-MHz figure higher.

## Cost table (instructions; 1 window = 10 ticks = 10 ms = one decision)

| | window mean | window worst | %10 ms @16 / 48 / 168 MHz (mean) | same, worst window | worst decision tick, % of its 1 ms tick @168 | energy/window (periph off / on) | flash | RAM (state + stack peak) |
|---|---|---|---|---|---|---|---|---|
| **WNN a49 dres, f64 observer** (bit-exact) | 257,213 | 258,866 | 241 / 80 / 23.0 % | 243 / 81 / 23.1 % | 77 % | 303 / 659 µJ | 1,247 KB tables + code, 0.8 KB observer | 2,008 B + 396 B |
| WNN a49, f32 observer (motor commands still bit-identical on 10k ticks) | 205,128 | 206,846 | 192 / 64 / 18.3 % | 194 / 65 / 18.5 % | 72 % | 242 / 526 µJ | same | 1,928 B + 352 B |
| WNN a49, f64 obs, **without its own Mahony** (q given, like MPCOF) | 144,878 | 145,473 | 136 / 45 / 12.9 % | 136 / 45 / 13.0 % | 67 % | 171 / 371 µJ | — | — |
| **MPCOF f64** (bit-exact on host), nominal | 161,369 | 1,173,622 | 151 / 50 / 14.4 % | 1100 / 367 / 105 % | 984 % | 190 / 414 µJ | 42.3 KB (35 KB tables) | 13,816 B + 688 B |
| MPCOF f64, stress | 163,750 | 1,791,735 | 154 / 51 / 14.6 % | 1680 / 560 / 160 % | 1536 % | 193 / 420 µJ | | |
| MPCOF f64, upset | 176,349 | 2,285,108 | 165 / 55 / 15.8 % | 2142 / 714 / 204 % | 1976 % | 208 / 452 µJ | | |
| MPCOF f64 **core** (shadow = the firmware's own rate loop), nominal / upset | 129,203 / 144,126 | 1,145,876 / 2,257,358 | 121 / 40 / 11.5 % (nominal) | 1074 / 358 / 102 % (nominal) | 981 % | 152 / 331 µJ | | |
| **MPCOF f32**, nominal / upset | 41,882 / 42,730 | 92,314 / 149,373 | 39 / 13 / 3.7 % | 87 / 29 / 8.2 % (nominal) | 58 % / 109 % | 49 / 107 µJ | 23.7 KB (17 KB tables) | 7,240 B + 392 B |
| MPCOF f32 core, nominal / upset | 9,346 / 10,138 | 64,198 / 121,253 | 8.8 / 2.9 / 0.83 % | 60 / 20 / 5.7 % (nominal) | 55 % / 106 % | 11 / 24 µJ | | |

Library flash, pulled in by both (shared with any firmware that already uses
libm): libm 9–11 KB, libgcc soft-float 3.2 KB.

### Per-call detail (mean / p99 / max instructions)

**MPCOF f64.**

| component | instructions |
|---|---|
| observer, per tick | 5,417 / 5,499 / 5,539 (81 soft-double calls) |
| shadow rate stage, per tick | 3,217 mean (6.5k on rate-update ticks) |
| decision, no QP | 74,390 / 75,537 / 76,137 (≈1,300 soft-double calls, 3 libm) |
| decision **with QP** (nominal / stress / upset) | mean 0.96M / 1.06M / 0.76M; max 1.10M / 1.72M / **2.21M** |
| QP worst on these trajectories | ≤ 6 Cholesky solves per decision, 33,639 soft-double calls |

How often the QP ran: 2 / 4,000 decisions (nominal), 6 / 2,000 (stress),
89 / 4,000 (upset). The primal fallback never ran.

**MPCOF f32.**

| component | instructions |
|---|---|
| decision, no QP | 6,535 |
| decision, QP | up to 118,715 |
| observer, per tick | 259 |
| shadow, per tick | 3.2k (it stays f64: `wnn_ratefw.c`) |

**WNN f64.**

| component | instructions |
|---|---|
| hold tick | 19,050 mean |
| decision tick | 85,762 mean, 86,216 max |
| encode + lookup + decode | 66.7k (decision minus hold) |
| Mahony f64, every tick | 11,234 |
| d-hat observer, every tick | **5,421** |
| other features | ~2.4k |

## How much of MPCOF's model the WNN dres input still carries (task 4)

The CTRL-48 residual is built from **MPCOF's own observer**: the measured lagged
actuator with kt and the motor lag, and the d̂ law. It is the same C code
(`cascade_dhat.c`) and the same cost.

| observer cost inside the WNN | instructions per tick | share of a hold tick | per 10 ms window |
|---|---|---|---|
| f64 (bit-exact) | 5,421 | 28 % | 54k = 21 % of the WNN's window |
| f32 | 258 | — | 2.6k |

The f32 observer changed no motor command on 10k ticks; that is evidence, not
a guarantee. The residual formula itself costs a few float ops. What the WNN
does NOT carry from MPCOF: the condensed QP, the DARE terminal cost, the shadow
rate model and the box solve.

## Parity evidence

- **C MPCOF (f64) vs the Rust teacher, host:** bit-identical, every tick, on
  100,000 ticks / 10,000 decisions. Compared: the held setpoint, and the hidden
  plan inputs (`diag_hidden`: measured ua, shadow I, shadow Fh, d̂). This
  includes 97 decisions on the box-QP path, up to 6 Cholesky solves.
- **Mutant check:** observer gain × (1 + 1e-12), changed in the C only. It is
  caught on 39,800 / 40,000 ticks.
- **Table check:** the tables are rebuilt in the export test, but parity is
  checked against the teacher's OWN planner output. A table that differed from
  `CtbrAxisMpc::build` would fail.
- **On the M4 (newlib libm):** max |Δsetpoint| ≤ 1e-15 rad/s. The 11,860 /
  9,940 / 22,650 differing axis-ticks are last-ulp `atan2`/`asin` differences
  (README: bit parity is a host claim).
- **f32 build:** max |Δsetpoint| 2.6e-6 / 2.3e-6 / 2.8e-6 rad/s (nominal /
  stress / upset). This is open-loop replay; f32 closed loop was not flown.
- **WNN a49 dres:** host bit-identical over 4,000 ticks; on the M4, every motor
  command bit-identical over 10,000 ticks (f64 and f32 observer).

## Caveats

1. **Instructions, not cycles.** 1.5 cycles/instr is an estimate. This likely
   **understates the WNN more than MPCOF**:
   - the WNN binary-searches a 1.2 MB key table in flash, about 18 probes per
     neuron × 264 neurons, so the ART cache will miss and 5 wait states apply
     at 168 MHz;
   - MPCOF streams 35 KB of tables sequentially.
2. **The M4F FPU is single precision.** Every f64 op above is a libgcc
   soft-float call (≈50–100 instructions). That is why exact MPCOF and the f64
   Mahony and observer are expensive, and why the f32 rows differ by 4–8×.
   The Rust teacher is f64; an f32 MPCOF is a different (if close) controller.
3. **The worst case is empirical, not a bound.**
   - PDAS is capped at 30 iterations and the primal fallback at 50·N steps, so
     the theoretical worst case is far above 2.2M instructions.
   - The fallback never ran on these trajectories.
   - The trajectories are teacher-flown. Student-driven (DAgger) states bind
     the box more often.
4. **Equal footing.** MPCOF here is handed `q`. The WNN computes its own Mahony
   estimate in f64 (11.2k instructions/tick). Giving MPCOF the same estimator
   adds 112k per window: f64 MPCOF becomes ~274k vs the WNN's 257k, which is
   about parity on average. MPCOF's shadow rate stage equals the firmware's
   rate loop state for state when the teacher flies, so a vehicle could read
   that state instead (the "core" rows).
5. **Coverage.** One WNN winner (a49 dres s1) was measured. One action repeat
   (n_o = 10, so only rate-loop phase 0 ever occurs at a decision).
   Energy is per window from a typical, linear datasheet figure.
6. **Toolchain.** The Arm GNU Toolchain was unpacked under /tmp and is not
   installed. Set `ARM_TC` to reproduce.

## Reproduce

    python mcu/ctrl51/export_mpcof.py                      # /tmp workspace copy -> tables + episodes
    cc -O2 -std=c11 -ffp-contract=off -I mcu mcu/ctrl51/mpcof_parity_host.c \
       mcu/mpcof_cascade.c mcu/cascade_dhat.c mcu/wnn_ratefw.c -lm -o /tmp/ctrl51/parity64
    /tmp/ctrl51/parity64 /tmp/ctrl51/ref/traj_upset.bin -d /tmp/ctrl51/bench/dec_upset.txt   # x3 regimes
    PYTHONPATH=src/wnn python scripts/export_controller_c.py --winner <a49 dres winner> --out /tmp/ctrl51/wnn_model_a49.h
    PYTHONPATH=src/wnn python mcu/ctrl51/record_wnn_flights.py --winner <same> --out /tmp/ctrl51/ref/wnn_a49_flights.bin
    cc -shared -fPIC -O2 -o /tmp/ctrl51/insnmark.dylib mcu/insnmark.c <glib flags as mcu/README.md>
    ARM_TC=<arm-gnu-toolchain>/bin bash mcu/run_bench_ctrl51.sh
    python mcu/ctrl51/summarize_bench.py

Datasheet: STMicroelectronics DS8626, *STM32F405xx/STM32F407xx*, table
"Typical and maximum current consumption in Run mode, code with data processing
running from Flash memory (ART accelerator enabled)" (st.com).


---

# Re-run 22/11/2026 — current base (Y+F+L) and the closest exportable WNN

**Why.** The 10/11 run measured one a49 dres winner. The current base is Y+F+L
(`..._a59_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_full_sN`, 256–504 neurons).
This re-run adds that base where it can be measured, three seeds of the closest
exportable WNN, an observer-free arm, and per-output parity.

## Y+F+L cannot be exported to the on-board C (yet)

`wnn_cascade.c` computes none of six blocks the base uses. Before this re-run the
exporter refused only `obs_att_bias_pred`. A winner carrying any of the other
five without it would have been exported as "runnable" with the wrong features
or decode.

**Fixed:** `scripts/export_controller_c.py` now refuses all six. The a49 dc
header it writes is byte-identical to before.

| block | what the C must add | runtime cost (op count, NOT measured) |
|---|---|---|
| `obs_att_bias_pred` (CTRL-46, +2 features) | f32 Mahony-bias predictor (`att_bias_pred.rs`) | ~40 VFP flops/tick |
| `obs_tilt_ref_err` (block 10 L1) | roll/pitch err vs the position loop's tilt reference, bias-corrected | ~15 flops + 2 clamps/tick |
| `obs_yaw_from_att_est` (ABI 55) | heading = Euler yaw of the Mahony quaternion | 1 `atan2f`/tick |
| `yaw_trim_ff` (ABI 58) | yaw setpoint += cT·cresid_z (cT exported) | 1 FMA/decision |
| `delta_max_yaw` (ABI 57) | the yaw bank's own delta_max | a constant |
| `yaw_integral_leak` (ARM 3) | the yaw integrals' own leak | a constant |

- **Upper bound for all six together:** ≤ 4,000 instructions per 10 ms window,
  under 2 % of a WNN window.
- **Porting them** with Rust↔C bit parity is a separate job.
- **No new parity inputs needed:** velocity, quaternion and cresid are already
  in the recorded stream.

## Method changes vs 10/11

- **Toolchain:** Arm GNU Toolchain **15.2.rel1**, under `/tmp/ctrl51/tc`.
  15.3.rel2 is no longer on the Arm CDN.
- **MPCOF tables:** rebuilt from the ABI-60 Rust source. They are byte-identical
  to the committed `mcu/mpcof_model.h`, so the teacher is unchanged.
- **Bench runner:** `run_bench_ctrl51.sh` takes `WNN_SET="tag:header ..."`.
- **Summary:** `summarize_bench.py --wnn-tags ...` also prints float ops per
  window.
- **Host parity:** `mpcof_parity_host.c` prints max |C − Rust| per output and per
  hidden element.

## Cost table — QEMU mps2-an386 Cortex-M4F

**Window** = 10 ticks = 10 ms = one decision; counts are instructions.
**Cycles** = 1.5 × instructions (an estimate). **Energy per decision** uses
DS8626 typical figures, peripherals off / on.

| controller | window mean | window worst | % of 10 ms @16 / 48 / 168 MHz (mean) | same, worst window | worst decision tick, % of its 1 ms tick @168 MHz | float ops / window | energy / decision | flash (controller) | RAM state + stack |
|---|---|---|---|---|---|---|---|---|---|
| **MPCOF f64** (bit-exact), nominal | 161,369 | 1,173,622 | 151 / 50.4 / 14.4 % | 1100 / 367 / 105 % | 984 % | 2,614 soft-double calls (max 18,364); 188 VFP | 190 / 414 µJ | 42.3 KB (+ libm 10 KB, libgcc 3.2 KB) | 13,816 + 688 B |
| MPCOF f64, upset (60°, 8 rad/s) | 176,349 | 2,285,108 | 165 / 55.1 / 15.8 % | 2142 / 714 / 204 % | 1976 % | 2,839 (max 34,836); 192 | 208 / 452 µJ | | |
| **MPCOF f32**, nominal | 41,872 | 92,304 | 39.3 / 13.1 / 3.74 % | 86.5 / 28.8 / 8.24 % | 58 % | 540 soft (all in the f64 shadow); 2,406 VFP (max 14,758) | 49.3 / 107 µJ | 23.7 KB | 7,240 + 392 B |
| MPCOF f32, upset | 42,720 | 149,363 | 40.1 / 13.4 / 3.81 % | 140 / 46.7 / 13.3 % | 109 % | 540; 2,587 (max 27,618) | 50.3 / 110 µJ | | |
| MPCOF f32 core (shadow = the firmware's loop), nominal / upset | 9,336 / 10,128 | 64,188 / 121,243 | 8.75 / 2.92 / 0.83 % | 60.2 / 20.1 / 5.73 % (nominal) | 55 / 106 % | — | 11.0 / 23.9 µJ | | |
| **WNN a49 dc s1** (256 n, dres), f64 observer | 256,945 | 258,552 | 241 / 80.3 / 22.9 % | 242 / 80.8 / 23.1 % | 77 % | 2,180 soft; 2,840 VFP | 303 / 659 µJ | 2,027 KB tables + code (uint32 keys) | 1,968 + 396 B |
| WNN a49 dc s1, f32 observer | 204,860 | 206,254 | 192 / 64.0 / 18.3 % | 193 / 64.5 / 18.4 % | 72 % | 1,340 soft (the f64 Mahony); 3,660 VFP | 241 / 525 µJ | same | 1,888 + 352 B |
| WNN a49 dc s1, f32 obs, **q given** (as MPCOF is) | 92,302 | 92,594 | 86.5 / 28.8 / 8.24 % | 86.8 / 28.9 / 8.27 % | 62 % | — | 109 / 237 µJ | | |
| WNN a49 dc s3 (264 n): f64 obs / f32 obs / f32 q-given | 258,498 / 206,399 / 93,837 | 260,314 / 208,278 / 94,170 | 242 / 80.8 / 23.1 % (f64) | | 78 % (f64) | | 305 / 243 / 111 µJ | 2,201 KB | 2,008 + 396 B |
| **WNN a49 dc s2** (480 n): f64 obs / f32 obs / f32 q-given | 312,721 / 260,631 / 148,087 | 314,370 / 262,218 / 148,494 | 293 / 97.7 / 27.9 % (f64); 139 / 46.3 / 13.2 % (q-given) | | **127 %** (f64) | 2,180 soft; 3,510 VFP (f64) | 369 / 307 / 175 µJ | 3,901 KB | 3,088 + 396 B |
| WNN **observer-free** a45 s1 (256 n, no dres) | 181,110 | 182,832 | 170 / 56.6 / 16.2 % | 171 / 57.1 / 16.3 % | 68 % | 1,280 soft; 2,583 VFP | 214 / 464 µJ | 922 KB | 1,936 + 344 B |
| WNN observer-free a45 s1, q given | 68,611 | 68,897 | 64.3 / 21.4 / 6.13 % | | 58 % | | 80.9 / 176 µJ | | |
| *Y+F+L 264 n (s1/s3/s4), **ESTIMATE*** | ≈ 265k (f64 obs) / ≈ 213k (f32 obs) / ≈ 101k (f32, q given) | | ≈ 248 / 83 / 23.7 % (f64) | | ≈ 80 % | | ≈ 312 / 251 / 119 µJ | 2,595 KB uint32, **1,005 KB pef64** | ≈ 2 KB |
| *Y+F+L 504 n (s2/s5), **ESTIMATE*** | ≈ 323k / ≈ 271k / ≈ 158k | | ≈ 303 / 101 / 28.8 % (f64) | | ≈ 133 % | | ≈ 381 / 320 / 186 µJ | 2,764–2,967 KB uint32, 1,121–1,199 KB pef64 | ≈ 3 KB |

**How the Y+F+L estimate is built.** Three parts, the first two measured:

- **264 n rows:** the measured a49 dc s3 row.
- **504 n rows:** a49 dc s2's decision part scaled by 504/480.
- **Lookup delta:** the measured Y+F+L lookup below minus an assumed
  ~25k-instruction a49 lookup (≈ 96 instructions × neurons).
- **Unported blocks:** the ≤ 4k bound above.

**Y+F+L lookup on the M4 — the part of the base that runs on the C today.**
Real VAL addresses, 200 decisions, `run_bench_pef.sh CPU=m4`.
Instructions per decision, every neuron looked up:

| winner | uint32 mean / p99 / max | pef64 mean / p99 / max |
|---|---|---|
| s1, 264 n | 27,778 / 27,949 / 27,971 | 74,287 / 76,204 / 76,267 |
| s5, 504 n | 47,442 / 47,771 / 47,903 | 127,388 / 136,753 / 136,811 |

**Flash fit.**

- **STM32H743 (2 MB internal flash):** Y+F+L fits only as pef64, which costs
  +46.5k (264 n) to +80k (504 n) instructions per window on top of the
  estimate rows.
- **STM32F405 (1 MB flash):** only the observer-free a45 s1 (922 KB) fits; no
  dres or Y+F+L winner does. MPCOF f32 needs 24 KB.

## Parity (step for step, same inputs → same outputs)

**MPCOF, C vs the Rust teacher, host.** 100k ticks / 10k decisions, 97 of them
on the box-QP path (up to 6 Cholesky solves). Max |C − Rust| per output:

| build | p (rad/s) | q (rad/s) | r (rad/s) | hidden: ua_meas / shadow I / shadow Fh / d̂ (max over x, y, z) |
|---|---|---|---|---|
| f64, nominal / stress / upset | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | all 0 (bit-identical) |
| **f32**, nominal | 2.52e-6 | 2.60e-6 | 1.45e-6 | 1.5e-7 / 3.9e-7 / 1.1e-7 / 2.0e-4 |
| f32, stress | 2.21e-6 | 2.34e-6 | 1.48e-6 | 1.4e-7 / 5.9e-7 / 8.4e-8 / 1.7e-4 |
| f32, upset | 2.47e-6 | 2.75e-6 | 1.54e-6 | 1.5e-7 / 5.0e-7 / 1.1e-7 / 1.9e-4 |

- **Mutant** (observer gain × (1 + 1e-12), C only): caught on 39,800 / 40,000
  ticks.
- **On the M4 (newlib libm), f64:** max |Δsetpoint| ≤ 1e-15 rad/s. The
  differing axis-ticks are last-ulp `atan2`/`asin` differences.
- **On the M4, f32:** max |Δsetpoint| ≤ 2.75e-6 rad/s.
- **f32 caveat:** open-loop replay only; f32 MPCOF was not flown closed-loop.

**WNN, C vs the installed Rust wheel (ABI 60).**

| check | result |
|---|---|
| host, `test_cascade_parity.py`, 3 × 2000 ticks per winner | a49 dc s1 / s2 / s3 and a45 s1 **bit-identical**: features, addresses, fire bits, setpoint, pwm. Max abs diff 0 on every output |
| on the M4, 10,000 ticks per winner | **0 motor-command mismatches**, f64 and f32 observer |
| mutant (1.5 × d̂ gain) | caught every time |

## How much of MPCOF's model the WNN still carries

**Carried.** The dres input runs MPCOF's offset-free observer, the same C code
(`cascade_dhat.c`). Its constants:

- torque gains (arm, k_drag, inertia);
- kt;
- motor-lag alpha (0.2 s);
- observer gain L;
- clamp c·out_limit.

Y+F+L adds one more MPCOF number. The cT in `yaw_trim_ff` is MPCOF's own yaw-law
integral gain (`CtbrAxisMpc::gain_row`, horizon-20 unconstrained) × i_limit. It
is computed offline and costs one FMA per decision on board.

**Not carried:** the condensed QP tables (17–35 KB), the DARE, the box solve,
and the shadow rate stage.

**Measured cost of carrying the observer** (dres winner minus the observer-free
arm, same C, same harness, both 256 n):

| | dres a49 dc s1 | observer-free a45 s1 | delta | share of the dres window |
|---|---|---|---|---|
| hold tick, f64 observer | 19,072 | 11,744 | +7,328/tick (observer 5,420 + residual / extra features ~1,900) | — |
| window, f64 observer | 256,945 | 181,110 | **+75.8k** | 29.5 % |
| window, f32 observer | 204,860 | 181,110 | **+23.8k** | 11.6 % |
| flash | | | +764 B of observer code, plus its constants | — |

**What it buys in flight.** ABI-45 pair, same recipe with and without dres,
TEST over 35 units, n = 3, means:

| arm | stable | err | steady |
|---|---|---|---|
| observer-free | 99.5 % | 1.92° | 1.14° |
| dres | 100 % | 1.54° | 1.03° |

That is a direction only. Significance belongs to experiment-design.

## Reading

1. **Average cost: exact MPCOF in f32 costs far less than any WNN.**
   - Full loop: MPCOF 42k instructions per window vs WNN 205–261k (f32
     observer).
   - With q handed to both: MPCOF 9.3k (core) or 42k (full) vs WNN 92–148k.
   - The "WNN is cheaper on board" claim does not survive single-precision
     MPCOF.
2. **What the WNN does have: a flat cost with no QP.**
   - WNN worst/mean ≤ 1.007.
   - f32 MPCOF's worst window is 2.2–3.6× its mean.
   - At 168 MHz, MPCOF's worst decision tick under upset overruns its 1 ms tick
     (109 %).
3. **The bigger WNNs overrun too.** At 480–504 neurons the WNN decision tick
   also overruns 1 ms at 168 MHz (127 %, f64 observer). It needs the decision
   spread over the hold ticks, or a faster core (H743).
4. **Flash is the WNN's binding constraint.** It needs 1–4 MB of tables vs 24 KB
   for MPCOF f32.
   - No dres or Y+F+L winner fits an F405.
   - Y+F+L fits the H743 only as pef64, at about +50–80k instructions per
     window.
