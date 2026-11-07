# MCU: on-board cascade controller + control-step cost harness (Cortex-M4)

Two things live here:

1. **The on-board CASCADE controller** (`wnn_cascade.[ch]`, 07/11/2026): the C twin
   of the Rust `WnnController::step` for an exported `--output-mode ctbr` winner,
   bit-identical to the simulator on the host (see *Parity* below).
2. **The instruction-count harness** (`bench.c`, `bench_inc.c`): per-step cost of
   the WNN lookup vs PID vs MLP under QEMU (`mps2-an386`), on synthetic inputs.

## Cascade controller

### What it ports (and what it refuses)

| Part | Rust reference (ABI 41) | C |
|---|---|---|
| features: 9 raw, tilt p/i, per-axis p/i (yaw dropped or not), integral leak/scale, yaw err/err_i (dt-integrated heading), collective / alt err / vz, pos err xy / vel xy | `controller.rs compute_features` | `compute_features` |
| Mahony attitude estimator (f64, kp/ki from the wheel's defaults) + body "up" | `estimator.rs`, `att_estimator.rs` | `mahony_*`, `up_from_quat` |
| CTRL-38 rate-I: I / i_limit per axis (raw I when i_limit = 0) | `ctbr.rs rate_integral_norm`, `cascade_obs.rs` | `cascade_block` |
| CTRL-38 d-hat: MPCOF offset-free observer on the APPLIED command, modelled motor lag, prime/seed protocol, clamp at c*out_limit, per-flight reset | `cascade_dhat.rs` | `dhat_*` |
| thermometer, output-layer addresses (MSB first), TRUE on-set binary search | `step`, `neuron_memory.rs address_over` | `encode_frame`, `forward_output` |
| BINARY antagonist decode, CTBR absolute or leaky delta law, gamma alphabet, action-repeat hold | `decode_ctbr`, `decoded_to_delta` | `decode_ctbr`, `wnn_tick` |
| firmware rate PID (lpf2p D filter, decimation, '+' mixer, collective transport) | `pid_firmware.rs RateStageFw`, `ctbr.rs CtbrInner::fly` | `wnn_ratefw.[ch]` (reference only, see below) |

The exporter **refuses** (lists every reason, writes nothing) a winner the C cannot
run: motor-level (`pwm`) output, any memory mode but BINARY, a non-antagonist
decode, state neurons (sn > 0), `output_full_window`, `obs_pwm`, `dhat_b`/`dhat_ff`,
`decouple_outputs`, > 32 address bits, and runs whose thermometer cannot be
reproduced (`--threshold-refit-from-student`, `--threshold-outer-quantile`).

### Export a cascade winner

    PYTHONPATH=src/wnn python scripts/export_controller_c.py \
        --winner logs/controller/ct40/full/C-S2-d_full_s2_winner.yaml.gz --out mcu/wnn_model.h

A stage checkpoint (`ckpt_<tag>/stage4_memory.yaml.gz`) works the same way. The
thresholds are NOT in the checkpoint — every scorer refits them on the run's TRAIN
seed (`phased_ga._report_thresholds`), and so does the exporter, from the run's
argv in the checkpoint meta (`--run-argv <run .out>` if the meta lacks it). The
header carries every block flag, the controller constants, the CTBR alphabet, the
rate-loop gains, and for CTRL-38 the d-hat plant `[arm, k_drag, Ixx, Iyy, Izz]`,
its gain (read from the wheel's `CtbrSpec`, never hand-copied) and the motor lag.
`--tables-only` keeps the old bench-only header (no `WNN_EXPORT_CASCADE`;
`wnn_cascade.h` refuses to compile against it).

`wnn_conn` is `uint8_t` only while every index fits 0..255, else `int16_t`
(`wnn_conn_t`). The old exporter always wrote `uint8_t`, silently truncating input
indices >= 256 (any cascade with 32+ features x 8 bits) and -1 slots.

### The firmware interface

    wnn_ctrl_t ctl;
    wnn_init(&ctl);                         /* boot */
    wnn_reset(&ctl, init_yaw, q0);          /* every flight */
    /* every physical tick (1 kHz): */
    wnn_tick_in_t in = { gyro, accel, target, vert, horiz, rate_integ, applied_prev };
    int decided = wnn_tick(&ctl, &in, sp);  /* sp = held [coll dev, p, q, r] */

* `rate_integ[3]` (CTRL-38 rate-I): the firmware rate PID's integrator `pid.integ`
  per axis, **as it stands before this tick's rate update**, in SI (rad). The C
  normalises by the exported i_limit. On a Crazyflie that is the rate
  controller's own state — the WNN never runs a second copy.
* `applied_prev[4]` (CTRL-38 d-hat): the motor command the plant actually received
  **last** tick (after the rate loop and any trim). The observer models the motor
  lag itself and ignores the first call after `wnn_reset` (priming), exactly as
  the teacher's observer does.
* `sp` goes to the firmware rate loop as rate setpoints `[p, q, r]` (rad/s) and a
  collective of `(anchor - hover_pwm) + sp[0]` pwm, added after the mixer — see
  `wnn_ratefw_fly`, the bit-exact port of the simulator's inner stage, which the
  parity test uses as "the firmware" and which a vehicle without its own rate loop
  can run.

### Build flags that matter

* **`-ffp-contract=off` is mandatory.** Rust never fuses `a*b+c`; GCC targeting
  the M4F does by default (VFMA). Measured on the parity run: with
  `-ffp-contract=fast` 5,210 of 6,000 held setpoints differ from the simulator.
* Estimator, rate-I and d-hat are f64 like the CPU reference. On an M4F (single
  precision FPU) that means libgcc soft-double (`__aeabi_dadd`, ...): link libgcc
  and a libm (`atan2f`, `powf`, `sqrtf`, `sqrt`). `-DWNN_DHAT_F32` selects the
  Metal kernel's f32 observer instead: cheaper, and NOT bit-identical (every tick's
  d-hat differs in the last bits; over the 6,000-tick parity run no thermometer
  bit flipped, which is evidence, not a guarantee).
* Bit parity is a host claim: an MCU libm may round `atan2f`/`powf` differently
  in the last ulp.
* The homebrew `arm-none-eabi-gcc` here has no newlib, so the controller was only
  compile-checked for the M4F (with stub libc headers), not linked or flown.
  The C-S2-d winner's tables are 7.06 MB of `.rodata` (1,761,565 keys): off-chip.

### Parity test (C vs the installed Rust wheel, step for step)

    PYTHONPATH=src/wnn python mcu/parity/test_cascade_parity.py   # 3 episodes x 2000 steps
    PYTHONPATH=src/wnn python -m pytest mcu/parity/test_cascade_parity.py   # 2 x 1000

The Rust C-S2-d winner flies the run's own plant (cf21_brushless, specific-force
IMU, rotor drag, 0.2 s motor lag, translation, L4C weather incl. motor asymmetry);
the per-tick inputs are recorded and replayed into a fresh Rust controller (which
must reproduce the recording) and the C (`wnn_parity_shim.c` + `wnn_ratefw.c`,
built with `cc -ffp-contract=off`). Asserted bit-for-bit every tick: features,
decision cadence, held setpoint, motor command; every decision tick: thermometer
frame, every neuron's address, every fire bit. Cases: the trained flags-off winner
and three SYNTHETIC CTRL-38 models (rate-I, d-hat, both — no trained winner with
the blocks exists yet) built through the Python API with production thresholds;
plus a mutant (1.5x d-hat gain in the C header only) that MUST be caught, and the
f32-observer / fp-contract builds reported for information.

## Instruction-count harness

## Method and its limits

QEMU's mps2-an386 **does not implement `DWT_CYCCNT`** (it reads 0), so the
register real STM32F405 silicon would use is unavailable. We count **retired
instructions** with a small TCG plugin (`insncount.c`) instead.

Every variant is built at `ITERS=0` and `ITERS=N` and the per-step cost is
`(I_N - I_0) / N`; the `NONE` row (input generation only) is then subtracted.
That double subtraction cancels startup, semihosting and input synthesis
exactly, leaving the control step alone.

**These are instructions, not cycles.** Real M4 loads cost ~2 cycles, taken
branches ~2-3, plus flash wait states — hardware cycles run above these counts.
Use them as a like-for-like ratio and a lower bound, never as a timing claim.
Hardware DWT numbers are task #6 (flight test), where the same `bench.h` works
unchanged.

## Run

    python scripts/export_controller_c.py --winner <stage4_memory.yaml.gz> --out mcu/wnn_model.h --tables-only
    cc -shared -fPIC -O2 -o mcu/insncount.dylib mcu/insncount.c \
       -I/opt/homebrew/include -I/opt/homebrew/opt/glib/include/glib-2.0 \
       -I/opt/homebrew/opt/glib/lib/glib-2.0/include -L/opt/homebrew/opt/glib/lib \
       -lglib-2.0 -Wl,-undefined,dynamic_lookup
    N=200 bash mcu/run_bench.sh

Needs `brew install arm-none-eabi-gcc qemu`.

## Caveats that matter for reading the table

- The MLP row is a **naive C** implementation of a *representative* 15-64-64-4
  net — NOT the deleted `run_mlp_ga.py` baseline, whose shape is unrecoverable.
  A CMSIS-NN/SIMD MLP would be substantially faster, so the WNN's advantage over
  it is **not** settled on an M4F. What is settled: the WNN needs no FPU at all.
- MLP weights live in `.bss` here (runtime-initialised), so its `.text` excludes
  ~21 KB of weights that a real build would place in flash.
- Synthetic inputs make every WNN lookup miss. That does not bias the result:
  the search runs to convergence with no early exit, so hit and miss cost
  identically — which is the bounded-WCET property being claimed.

## Measured (QEMU mps2-an386, Cortex-M4F, N=200, input generation subtracted)

    variant             instr/step   .text B
    WNN addr-only           11,423     2,332   bit gather, search skipped
    WNN (naive gather)      18,645   462,732   <- best WNN
    WNN (level table)       22,879   523,xxx   SLOWER — see below
    PID (firmware shape)       394       870
    MLP 15-64-64-4          34,423       794   naive C; ~21 KB weights in .bss

### The level-indexed gather did NOT pay off — prediction refuted

I predicted a level-indexed partial-address table would be 4-6x cheaper than the
per-bit gather, on the reasoning that each feature is an 8-bit thermometer so its
whole contribution is fixed by its level (0..8): 15 table lookups per neuron instead
of 30 load/shift/or chains. Measured twice, it is SLOWER both times:

    naive per-bit gather                     11,423
    level table, [n][f][L], stride 9         19,500   (2 runtime muls per access)
    level table, stride 16 + hoisted index   15,657   still worse

The second version fixed the obvious faults — power-of-two stride so the index math
is shifts not multiplies, and hoisting `f*16+level` out of the neuron loop since
inlevel[] is identical for all 64 neurons. It closed most of the gap and still lost.

Why the naive version is hard to beat here: `addr = (addr<<1) | inbits[c[b]]` is a
byte load, a shift and an OR in a tight 30-iteration loop the compiler unrolls well —
about 6 instructions per bit with no index arithmetic at all. The table version trades
that for a 15-iteration loop of indexed 32-bit loads that unrolls worse, and touches
61 KB of table instead of 120 bytes of input.

So the gather is not the soft target I claimed. The levers that remain are real ones:
fewer bits per neuron (less to gather AND a denser memory), or hardware, where the
address is wiring and all 64 neurons resolve in parallel.

## ⚠️ THE 820 INSTR/STEP FIGURE BELOW IS VOID (found 02/09/2026)

`bench.h`'s `Reset_Handler` was `bl main` and nothing else, while `link.ld` places
`.data` in RAM with its load image in FLASH (`> RAM AT > FLASH`). Nothing copied
it and nothing zeroed `.bss`, so **every initialised static read as zero**.
`bench_inc.c` seeds its walk with `static uint32_t rng_s = 0x12345678`; read back
as 0, a xorshift is a fixed point at 0, so every feature level stayed at its
initial 0 and **the bounded random walk never moved**. The INC path therefore
found ZERO dirty neurons on every step and its "cost" was an empty step.

Fixed: `Reset_Handler` now copies `.data` and zeroes `.bss` (`link.ld` gained the
`_sidata/_sdata/_edata/_sbss/_ebss` symbols). Re-measured on the same model,
same jitter, input generation subtracted:

    INC incremental+hash, jitter 2      820  ->  1,683 instr/step   (2.05x)

`BASE` is unaffected — it re-gathers and searches every neuron regardless of
whether the input moves. The equivalence check still reports 0 address and 0
lookup mismatches after the fix, so incremental addressing was always CORRECT;
it was simply never exercised. The 25x-vs-BASE claim survives in direction
(20,231 -> 1,683 is 12x) but not in magnitude. Everything below this banner
predates the fix.

## 820 instructions/step — incremental addressing + O(1) lookup (bench_inc.c)

The 18,645 figure above was measured with a benchmark that draws fresh RANDOM feature
levels every step. That is not what a 1 kHz attitude loop looks like: levels move by a
step or two per millisecond, and that temporal coherence is the single biggest lever.
Random inputs destroyed it before it could be measured.

    variant                   per-step   minus input synthesis
    BASE full-gather+binsearch  20,245              20,231
    INC  incremental+hash          834                 820      <- 25x
    INC  (jitter 4)                902                 822

Two optimisations, both of which need the realistic input model to show up at all:

  OPT 1  Incremental addressing. Keep a running 30-bit address per neuron. A feature
         level moving L->L+1 flips EXACTLY one input bit. An inverted index (input bit
         -> the ~16 neurons that read it, with their address-bit masks) turns a flip
         into ~16 XORs instead of re-gathering 1,920 bits.
  OPT 2  O(1) lookup, only where dirty. An open-addressed hash over the ON-set
         replaces the ~13-iteration binary search, and a neuron whose address did not
         change cannot change output, so it is never looked up.
         Incremental decode likewise: motor thermometer counts move only when a
         neuron flips, so 64 adds become a +-1.

EQUIVALENCE IS VERIFIED, not assumed. Synthetic inputs make every lookup miss, so the
output checksum cannot distinguish the two paths. bench_inc.c therefore checks
directly at end of run: every running address against a full re-gather, and hash
membership against binary search, for all 64 neurons. Both report 0 mismatches over
500 steps at jitter 1 and 4.

CAVEATS. (1) Instructions under emulation, not silicon cycles. (2) All lookups MISS on
synthetic inputs; an open-addressed miss stops at the first empty slot (~1 probe at
load factor 0.44) while a hit averages more, so the hash row is somewhat optimistic —
real trajectories hit far more often. (3) The hash costs 2 MB of RAM at HBITS=18;
a minimal perfect hash would cut that hard and is the obvious next step.


## Probe accounting for external memory (02/09/2026)

`-DPROBE_STATS` counts, per control step, how many neurons go DIRTY and how many
key-array reads the lookups perform; `-DINC_LOOKUP_BINSEARCH` selects the sorted
array over the hash. Both are OFF by default so the timing build is unchanged —
counting costs instructions, so a probe build and an instr/step build are never
the same run. Runner: `N=300 MODEL=<hdr> bash mcu/run_bench_probes.sh`.

**Why probes and not instructions.** At 480 MHz a 1 ms control period is 480,000
cycles, so instructions are nearly free. What is not free is a RANDOM READ into
external memory, which pays a command/address/dummy phase before data moves. The
price of an off-chip model is (reads/step) x (random-read latency).

**Measured, b=32 n=256 (the hd 0.1129 record run), 894,552 TRUE keys, 300 steps:**

    jitter  dirty/step  dirty_max  reads/step  reads_max  reads/dirty
      1         48.95         70       599.43       3881        12.24
      2         88.08        119      1070.72       4410        12.16
      4        141.75        184      1728.50       5119        12.19
      8        195.74        236      2385.91       5517        12.19

`reads/dirty` ~= 12.2 is log2(894,552/256) = 11.8 plus the confirm read — the
binary search behaving exactly as theory says, which is the cross-check that the
counter is measuring what it claims.

Instruction cost on the same model, binsearch lookup, jitter 2: **14,713
instr/step** (~4.6% of a 1 ms budget at ~1.5 cycles/instr). The old b30 n64 model
is 4,759 with the same lookup.

**Budget arithmetic.** `.text` is 3,590,168 B = 3.42 MiB, so this model does not
fit the H743's 2 MB flash and must go off-chip. At jitter 2 (1,071 reads/step):

    external SDRAM  @ ~100 ns/random read    ~107 us    ~11% of 1 ms
    QSPI NOR (XIP)  @ ~250 ns/random read    ~268 us    ~27% of 1 ms

Both fit, and caching helps further: 32-byte lines hold 8 keys, so the last ~3
probes of each search land in an already-fetched line. At jitter 8 the QSPI case
reaches ~600 us and the margin is thin. **These latencies are assumed, not
measured** — the probe counts are the measurement; pricing them needs a part
number and, ultimately, hardware.


## Classical baselines re-measured on the fixed harness (02/09/2026)

Re-run after the `.data`/`.bss` startup fix, same model, `N=200`, input
generation subtracted. The question was whether the dead-RNG bug moved anything
besides the INC row:

    variant      before   after   delta
    WNN_ADDR     11,423  11,424      +1
    WNN          18,645  18,433    -212
    WNN_FAST     22,879  22,897     +18
    PID             394     395      +1
    MLP          34,423  34,468     +45

**The bench.c table survives.** Every variant there re-gathers all neurons and
runs the binary search to convergence with no early exit, so its cost does not
depend on whether the input moved — the residual deltas are just different
search paths and real (rather than all-zero) float inputs. PID was never at risk
either: its gains are assigned at runtime in `main()`, not initialised statics.

Only `bench_inc.c`'s INC row was destroyed, and precisely because it is the one
variant with input-dependent control flow — which is the entire point of it.
That is the lesson worth keeping: **the bug could only bite the measurement that
depended on the input actually changing**, so a table of stable numbers around it
was not evidence that it was fine.
