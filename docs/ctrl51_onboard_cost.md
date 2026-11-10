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
