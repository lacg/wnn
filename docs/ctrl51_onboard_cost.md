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

**Update (same day, later): Y+F+L now runs on the M4 and is MEASURED.** The six
blocks below were ported to `wnn_cascade.c`, and the estimate rows were replaced by
measured ones (section "Y+F+L measured on the M4").

## Y+F+L on board: the six blocks, ported (22/11/2026)

At first `wnn_cascade.c` computed none of the six blocks the base uses. Before this
re-run the exporter refused only `obs_att_bias_pred`, so a winner carrying any of the
other five would have been exported as "runnable" with the wrong features or decode.
That refusal gap was closed first. Then all six were ported, each mirroring its Rust
source in op order and precision:

| block | C (`mcu/wnn_cascade.c`) | Rust source | constant(s) in the header, read off the wheel |
|---|---|---|---|
| `obs_att_bias_pred` (CTRL-46, +2 features at the tail) | `abp_step`, `rot_t_xy` (f32) | `att_bias_pred.rs` | none new: kp, ki, dt are the estimator's; g = 9.81 is a Rust source constant |
| `obs_tilt_ref_err` (block 10 L1) | `tilt_ref_target` (f32) | `tilt_ref_err.rs` | `WNN_TILT_REF_SHAPE_INIT` = the controller's `obs_tilt_ref_err` getter (b_xy, ω_n, ζ, max tilt) |
| `obs_yaw_from_att_est` (ABI 55) | `mahony_heading`, `wrap_angle` (f64, like Rust) | `controller.rs mahony_heading`, `quat_to_euler_f64` | none |
| `yaw_trim_ff` (ABI 58) | held yaw setpoint += cT·cresid_z, clamped; the bank value is untouched | `controller.rs decode_ctbr`, `yaw_ff_value` | `WNN_YAW_TRIM_FF_CT` = the `yaw_trim_ff` getter (cT = −0.3993 rad/s per unit) |
| `delta_max_yaw` (ABI 57) | `delta_max_bank` | `controller.rs delta_max_bank` | `WNN_DELTA_MAX_YAW` |
| `yaw_integral_leak` (ARM 3) | `push_integral(..., leak)` | `compute_features` | `WNN_YAW_INTEGRAL_LEAK` |

- **Exporter** (`scripts/export_controller_c.py`): the six refusals are lifted; L2y /
  L2y' (`obs_yaw_trim_err[_p]`) are still refused. `wheel_block_constants()` builds
  a cell-less `WnnController` from the winner's spec and reads every constant from
  its getters, never from a hand copy. Each macro is emitted only when its block is
  on, so headers of winners without these blocks are unchanged.
- **The dense yaw ladder** (`--yaw-centred-ladder dense`) and the dense tilt ladder
  are thermometer thresholds only. The exporter refits them with the scorer's own
  `_test_thresholds` (Y+F+L s1: yaw and yaw-I rows symmetric about 0). No C change
  and no seventh item.

## Method changes vs 10/11
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
| **Y+F+L**, all five seeds | MEASURED, see the next section | | | | | | | | |

## Y+F+L measured on the M4 (replaces the 22/11 estimate rows)

**Setup.** Same harness as above: QEMU mps2-an386 Cortex-M4F, `-O2 -ffp-contract=off`,
newlib-nano. Each winner flies 5 × 2000 = 10,000 recorded ticks (1,000 decisions)
from `record_wnn_flights.py`: the installed wheel (ABI 60) on the run's own plant.
Observer is f64 (the bit-exact build). All counts are retired instructions.
**Cycles = 1.5 × instructions: an estimate, not measured.**

**Columns.**

- **hold / decision tick:** mean / max.
- **window:** 10 ticks = 10 ms = one decision; mean / worst.
- **% of 10 ms:** mean window.
- **worst decision tick:** its 1 ms tick.
- **flash:** `.text` of the whole M4 image. It includes ~12 KB libm, 3.2 KB libgcc,
  2.2 KB of firmware rate loop and the bench driver.

| winner (n, TRUE keys) | keys | hold tick | decision tick | window | % of 10 ms @16 / 48 / 168 / 480 MHz | worst decision tick, % of 1 ms @16 / 48 / 168 / 480 | flash B | window, q given |
|---|---|---|---|---|---|---|---|---|
| s1 (264 n, 662,535) | uint32 | 22,123 / 22,439 | 92,295 / 92,839 | 291,400 / 293,290 | 273 / 91.1 / 26.0 / 9.1 | 870 / 290 / 82.9 / 29.0 | 2,681,848 | 178,931 |
| | pef64 | same | 146,578 / 150,832 | 345,683 / 351,283 | 324 / 108 / 30.9 / 10.8 | 1414 / 471 / 135 / 47.1 | 1,055,000 | 233,214 |
| s2 (504 n, 703,938) | uint32 | 22,131 / 22,431 | 146,550 / 147,191 | 345,731 / 347,694 | 324 / 108 / 30.9 / 10.8 | 1380 / 460 / 131 / 46.0 | 2,854,184 | 233,199 |
| | pef64 | same | 238,964 / 244,171 | 438,145 / 443,463 | 411 / 137 / 39.1 / 13.7 | 2289 / 763 / 218 / 76.3 | 1,173,616 | 325,613 |
| s3 (256 n, 420,562) | uint32 | 22,134 / 22,442 | 86,172 / 86,642 | 285,380 / 287,142 | 268 / 89.2 / 25.5 / 8.9 | 812 / 271 / 77.4 / 27.1 | 1,713,736 | 172,848 |
| | pef64 | same | 131,393 / 135,345 | 330,602 / 335,519 | 310 / 103 / 29.5 / 10.3 | 1269 / 423 / 121 / 42.3 | 685,056 | 218,069 |
| s4 (264 n, 385,470) | uint32 | 22,120 / 22,489 | 87,845 / 88,325 | 286,928 / 289,056 | 269 / 89.7 / 25.6 / 9.0 | 828 / 276 / 78.9 / 27.6 | 1,573,592 | 174,476 |
| | pef64 | same | 134,320 / 137,893 | 333,403 / 337,940 | 313 / 104 / 29.8 / 10.4 | 1293 / 431 / 123 / 43.1 | 645,816 | 220,951 |
| s5 (504 n, 756,032) | uint32 | 22,126 / 22,439 | 148,389 / 149,054 | 347,525 / 349,642 | 326 / 109 / 31.0 / 10.9 | 1397 / 466 / 133 / 46.6 | 3,062,560 | 235,041 |
| | pef64 | same | 243,015 / 248,557 | 442,151 / 448,056 | 415 / 138 / 39.5 / 13.8 | 2330 / 777 / 222 / 77.7 | 1,253,600 | 329,667 |

**Float ops per window** (f64 observer): 2,700 soft-double calls (270 per tick,
flat). VFP instructions: 3,624–3,650 (256–264 n) and 4,368–4,372 (504 n). These are
the same for uint32 and pef64.

**RAM.**

- Controller state (`sizeof(wnn_ctrl_t)`, incl. the per-tick telemetry arrays):
  2,048 B (256 n), 2,088 B (264 n), 3,288 B (504 n).
- Stack peak: 436 B (uint32), 484 B (pef64).
- Image `.data + .bss`: 84 + 2,840–4,080 B.

**f32 observer** (`-DWNN_DHAT_F32`), cost only. Windows: s1 239,359 / 293,643,
s2 293,677 / 386,091, s3 233,300 / 278,522, s4 234,887 / 281,362, s5 295,460 /
390,085 (uint32 / pef64). This build is **no longer command-identical**: 8,172–8,888
of 10,000 ticks differ. On a49 dc it was 0 of 10,000. The cause is the yaw-trim FF.
It writes cT·cresid_z, a continuous function of d̂, straight into the held yaw
setpoint, so any f32 rounding of d̂ reaches the motors. Before, d̂ only moved a
thermometer bit. The f32 rows are a cost figure for a DIFFERENT controller.

**What the six blocks cost** (difference of two measurements on the same harness:
Y+F+L s1 vs a49 dc s1, f64 observer):

- **hold tick:** 22,123 vs 19,072 instructions, +3.05k per tick ≈ +30k per window;
- **soft-double calls:** 270 vs 218 per tick.

Most of this is the ABI-55 heading. It is f64 like Rust (`quat_to_euler_f64` →
`atan2`), and every f64 op is a libgcc call on the M4F. The 22/11 bound of ≤ 4k
per window assumed an `atan2f`, and it was wrong. The other five blocks are a few
dozen f32 ops per tick or per decision.

**Measured vs the 22/11 estimate** (window mean, f64 observer, uint32):

| | estimate | measured |
|---|---|---|
| 264 n | ≈ 265k | 286.9–291.4k |
| 504 n | ≈ 323k | 345.7–347.5k |

**Flash fit** (image `.text`):

- **STM32H743, 2 MB internal (2,097,152 B):**
  - every pef64 image fits;
  - uint32 fits for s3 (1.71 MB) and s4 (1.57 MB) only;
  - s1, s2 and s5 need pef64. That costs +54k (264 n) to +95k (504 n)
    instructions per window.
- **STM32F405, 1 MB (1,048,576 B):**
  - only s3 and s4 fit, as pef64 (685 KB, 646 KB);
  - s1 pef64 misses by 6.4 KB.

**Timing at the 1 ms decision tick** (worst decision tick, mean-cost windows):

- **168 MHz:**
  - 256–264 n uint32 fits (77–83 %);
  - 504 n uint32 overruns (131–133 %);
  - every pef64 build overruns (121–222 %).
- **480 MHz (H743):** every build fits, at most 78 % (s5 pef64).
- **16 MHz:** nothing fits, even on average (≥ 268 % of the 10 ms window).
- **48 MHz:** only 256–264 n uint32 fits on average (89–91 % of the window), and
  its decision tick still overruns its own 1 ms tick (271–290 %).

## Parity (step for step, same inputs → same outputs)
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
| **Y+F+L s1–s5**, host, 3 × 2000 ticks each | **bit-identical**, uint32 AND pef64: features (27 incl. both abp columns), frame, addresses, fire bits, held setpoint, pwm. The synthetic rate_i / dhat / both cases built on the Y+F+L spec also pass |
| Y+F+L block mutants, C header only (s1–s5) | all caught: cT × 1.5, tilt ω_n × 1.01, delta_max_yaw × 1.01, yaw leak = integral_leak, heading by dead reckoning, abp g = 9.80 |
| Y+F+L s1–s5 on the M4, 10,000 ticks each | **PWM_MISMATCH_TICKS 0**, uint32 and pef64 (f64 observer). The f32-observer build is NOT command-identical (see above) |
| regression after the port | ct40 C-S2-d default, a49 dc s1 and a45 s1 still pass host parity |

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
   also overruns 1 ms at 168 MHz (127 % a49 dc s2; Y+F+L 504 n 131–133 %, f64
   observer). It needs the decision spread over the hold ticks, or a faster core
   (H743: every Y+F+L build ≤ 78 % at 480 MHz, measured instructions × 1.5).
4. **Flash is the WNN's binding constraint.** It needs 0.6–3 MB of tables vs
   24 KB for MPCOF f32.
   - Y+F+L s3 and s4 fit an F405 as pef64; no other dres or Y+F+L winner does.
   - On the H743, s3 and s4 fit uint32; s1, s2 and s5 fit only as pef64,
     at +54–95k instructions per window (measured).
5. **Y+F+L measured.** At 285–348k instructions per window (f64 observer,
   uint32), it costs 7–11 % more than the a49 dc winners. The ABI-55 f64 heading
   is most of that.
