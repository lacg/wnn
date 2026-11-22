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
* Rate-I and d-hat are f64 like the CPU reference. On an M4F (single
  precision FPU) that means libgcc soft-double (`__aeabi_dadd`, ...): link libgcc
  and a libm (`atan2f`, `powf`, `sqrtf`, `sqrt`).
* The Mahony estimator + ABI-55 heading follow the header's `WNN_EST_F32`
  (ram_controller ABI 61, 22/11/2026): the exporter emits `#define WNN_EST_F32 1`
  when the wheel's `ATT_EST_PRECISION` is `"f32"`; then the filter is f32 and the
  heading uses `atan2_det` (IEEE +, -, *, /, sqrt only, no libm), bit-identical to
  Rust and to the Metal kernel. A header without the macro (any export from an
  ABI <= 60 wheel) keeps the f64 estimator + f64 heading of that lineage. `-DWNN_DHAT_F32` selects the
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


## CTRL-51: WNN vs the MPCOF teacher on the same M4 (10/11/2026)

`mpcof_cascade.[ch]` is the C port of the cascade MPCOF outer loop: the online
part only, with tables from `ctrl51/export_mpcof.py` written to
`mpcof_model.h`. `cascade_dhat.[ch]` is the observer, now shared with
`wnn_cascade.c`. On the host the port is bit-identical to the Rust teacher
over 100k ticks.

`bench_ctrl51.c` with `run_bench_ctrl51.sh` measures every call. It uses the
`insnmark.c` plugin, which gives per-region instructions, soft-double calls,
libm calls and VFP instructions. That yields a worst case as well as a mean.
It needs a toolchain WITH newlib (`ARM_TC`).

The exporter now supports the CTRL-48 residual (`WNN_OBS_DRES`). Before this
it exported dres winners as runnable with those three features left at 0, so
re-export any such header.

Results and caveats are in `docs/ctrl51_onboard_cost.md`.

Re-run 22/11/2026: three a49 dc winners (256/264/480 n), an observer-free arm, and
per-output parity. Several winners in one run with
`WNN_SET="tag:header ..." bash mcu/run_bench_ctrl51.sh`, then
`summarize_bench.py --wnn-tags tag ...`.

**Y+F+L on board (22/11/2026, later the same day).** `wnn_cascade.c` now ports the
six blocks the base needs:

- the Mahony-bias predictor;
- the L1 tilt-reference error;
- the ABI-55 Mahony heading;
- the ABI-58 yaw-trim FF;
- the yaw delta_max;
- the yaw-integral leak.

Each new `WNN_*` macro is emitted only when its block is on, and the exporter reads
its constants off the wheel (`wheel_block_constants`). L2y / L2y' are still refused.

**Parity.**

- Host, s1–s5: bit-identical, uint32 and pef64.
- One C-only mutant per block: every one caught.
- M4, 10k ticks per winner: 0 motor mismatches with the f64 observer.
- The f32 observer is no longer command-identical under the yaw-trim FF.

Measured cost (cycles = 1.5 × instructions, an estimate) is in
`docs/ctrl51_onboard_cost.md`, "Y+F+L measured on the M4". Raw summary:
`mcu/ctrl51/yfl_2211/`. The summary now also prints 480 MHz.


## pef64: compressed TRUE-key storage (20/11/2026)

**What.** An alternative backend for the SAME lookup ("is address A in neuron n's
TRUE set?"): partitioned Elias-Fano, blocks of B=64 keys. Per neuron, a block
index (`wnn_pef_first[]` = each block's smallest key, `wnn_pef_meta[]` = bit
offset << 6 | low width l, `wnn_pef_blk[]` = the neuron's block range); per block,
the other <= 63 keys as Elias-Fano (l-bit lows, then the high parts in unary) in
one LSB-first bitstream `wnn_pef_data[]`. Lossless; chosen by the compression
probe (`logs/compression_probe/`: Rice 3x and FOR 2x slower at the same size).

Lookup (`mcu/wnn_pef.h`, header-only C99, no malloc, const tables only): binary
search on the neuron's block first-keys, then inside ONE block find the
(x >> l)-th zero of the high part from whichever end is nearer, 32 bits at a time
(popcount by a 256-byte LUT — the Cortex-M7 has no popcount instruction), and
binary-search that bucket's lows.

**Build.**

    PYTHONPATH=src/wnn python scripts/export_controller_c.py --winner <winner> \
        --out mcu/wnn_model.h [--tables-only] --key-format pef64

The header then has no `wnn_keys[]` and defines `WNN_KEY_FORMAT_PEF64`;
`wnn_cascade.c`, `bench.c`, `bench_inc.c` switch their lookup on it (no other
flag). The default `--key-format uint32` output is byte-identical to before
(checked on a winner), and `wnn_cascade.c` against a uint32 header compiles to a
byte-identical object, as do the `bench.c` and `bench_inc.c` firmware images (every
variant). `bench_inc.c` under pef64 always looks up in place (there
is no `wnn_keys` to hash), i.e. it implies `-DINC_LOOKUP_BINSEARCH`.
`scripts/count_true_keys.py` records `bytes_pef64` next to `bytes_uint32` (the
flash rule still reads `bytes_uint32`).

**Exactness** — `mcu/parity/test_pef64_exactness.py --c` (pytest: synthetic sets
+ a corrupted-bitstream mutant that must be caught). Y+F+L s1-s5 and Y3 s1-s2:
every stored key, every key+-1, every distinct (neuron, address) of the recorded
VAL traces (s1, s5, Y3 s2 have traces) and 1M random addresses, Python reference
decoder AND the host-compiled C (`pef_host_check`, uint32 C binary search vs
`wnn_pef.h`): **0 mismatches on all 7 winners** (2.2M-5.9M queries each).

**Flash bytes** (uint32 = keys + `wnn_off` + conn; pef64 = every PEF array incl.
`wnn_off` + conn; H743 internal flash = 2048 KB):

    winner        keys   uint32 KB  pef64 KB   ratio
    Y+F+L s1    662,535     2,595     1,005    0.387
    Y+F+L s2    703,938     2,764     1,121    0.406
    Y+F+L s3    420,562     1,650       644    0.390
    Y+F+L s4    385,470     1,513       605    0.400
    Y+F+L s5    756,032     2,967     1,199    0.404
    Y3    s1    687,060     2,695     1,073    0.398
    Y3    s2    789,923     3,099     1,232    0.398

**Cost — MEASURED retired instructions** (`mcu/run_bench_pef.sh`, QEMU
mps2-an500 Cortex-M7, `-O2`; the M4 board gives the same within 3%).
Real VAL addresses, 200 decision ticks, marker overhead subtracted; the address
gather is NOT included (it is the same for both):

    Y+F+L s1 (264 n)        uint32 mean / p99 / max     pef64 mean / p99 / max
    one lookup                  96 /    149 /    149       273 /    467 /    543
    decision, every neuron  27,778 / 27,949 / 27,971    75,781 / 77,798 / 77,868
    decision, incremental   21,949 / 28,729 / 28,785    58,345 / 77,506 / 78,007
    Y3 s2 (480 n)
    one lookup                  88 /    140 /    149       249 /    455 /    536
    decision, every neuron  46,535 / 46,704 / 46,714   125,824 /130,230 /133,435
    decision, incremental   36,431 / 46,848 / 47,951    95,350 /125,132 /127,437

Both formats report identical hit counts on target. Synthetic walk (`bench_inc.c`,
jitter 2, input generation subtracted): BASE 67,702 vs 118,923 (s1),
119,362 vs 198,671 (Y3 s2); INC 6,666 vs 15,011 and 11,164 vs 24,177.
On real traces ~70% of neurons change address every decision, so incremental
addressing saves only ~20% of the lookups there.

**pef64 costs ~2.8x the INSTRUCTIONS of the binary search** (mean; max 3.6x).
The probe's "worst case ~ today's binary search" was a CYCLE ESTIMATE dominated
by flash line misses, and that part holds: distinct 32-byte lines touched per
lookup on the same traces (Python count, ESTIMATE of flash traffic) are
uint32 7.2 / pef64 7.7 (s1), 6.5 / 7.3 (Y3 s2). With 1 cycle/instruction and
every line a 15-cycle miss (the probe's model, an ESTIMATE, not silicon) one
lookup is ~190-205 vs ~360-390 cycles, i.e. ~1.9x; a full 480-neuron decision ~0.36 ms
of a 1 ms tick at 480 MHz. Silicon DWT numbers remain task #6.


### CTRL-60 (22/11/2026): pef64 in the FULL cascade, exportable winners

The 20/11 numbers above are the lookup alone on replayed addresses. This is the whole
`wnn_tick` (features, Mahony, observer, gather, lookup, decode) with a pef64 header,
on the four exportable winners CTRL-51 flew (`logs/controller/ctrl48/`
`C-S2-d_wposx2_a49_rf_c200_dc_full_s{1,2,3}`, `..._a45_rf_c200_full_s1`).

**Parity (0 mismatches everywhere).**
* Host, `test_cascade_parity.py --winner <w>` (3 x 2000 ticks): new strict cases
  `flags_off/pef64` (the trained winner) and `both/pef64` (rate-I + d-hat synthetic)
  are bit-identical to Rust on features, addresses, fire bits, setpoint and motor
  command, as are their uint32 twins, so uint32 == pef64 on every tick.
* Host, `test_pef64_exactness.py --c --tags <4 winners>`: every key, key+-1 and 1M
  random addresses, Python and C: 0 mismatches (1.7M-4.0M queries per winner).
* M4 (QEMU mps2-an386, `ONLY=wnn run_bench_ctrl51.sh`), 10,000 recorded ticks per
  winner, f64 and f32 observer: `PWM_MISMATCH_TICKS 0` for pef64 (as for uint32).
  The re-exported uint32 headers are byte-identical to the CTRL-51 ones and give the
  same instruction totals.

**Cost** (instructions, f64 observer; `summarize_bench.py --wnn-only`;
raw: `logs/controller/ctrl60/`). Firmware = `.text` of the whole M4 image.

    winner        n    firmware B uint32 -> pef64      decision tick mean / max            10-tick window mean / worst
    a49 dc s1   256    2,091,368 ->   827,840 (0.40)    85,298 /  85,793 -> 136,327 / 139,464   256,945 / 258,552 -> 307,973 / 311,087
    a49 dc s3   264    2,270,368 ->   889,016 (0.39)    86,835 /  87,323 -> 139,076 / 142,509   258,498 / 260,314 -> 310,740 / 314,333
    a49 dc s2   480    4,010,424 -> 1,561,696 (0.39)   141,082 / 141,641 -> 238,178 / 243,448   312,721 / 314,370 -> 409,817 / 415,116
    a45 s1      256      960,448 ->   415,832 (0.43)    75,410 /  75,814 -> 126,500 / 130,310   181,110 / 182,832 -> 232,199 / 235,993

pef64 adds ~51k instructions per decision at 256-264 n and ~97k at 480 n (1.6-1.7x
the decision tick, 1.2-1.3x the window); hold ticks are unchanged. Hit and miss are not
equal-cost under pef64 (worst/mean 1.02-1.03 vs 1.006 for uint32).

### ABI 61 (22/11/2026): f32 Mahony estimator + heading, and the attitude split

* **Estimator.** `wnn_cascade.c` carries both estimators, chosen by `WNN_EST_F32`
  (see "Build flags"). The f32 one mirrors `mahony_f32.rs` op for op; the Metal kernel
  runs the same ops under a strict-FP scope, so the estimator is CPU == GPU == C bit for
  bit (cargo pin `ctrl61_f32_estimator_tests.rs`).
* **Bench regions.** `bench_ctrl51.c` now splits the attitude cost: MPCOF per tick
  [observe][decide][shadow][quat->Euler twin on decision ticks], WNN per tick
  [wnn_tick][d-hat twin][Mahony twin][heading twin][roll/pitch twin].
  `summarize_bench.py` reads the region count from the run log (old 3-region benches
  still summarize) and prints the "ATTITUDE EXCLUDED" rows: both controllers handed the
  firmware's attitude (q and roll / pitch / yaw).
* Results: `docs/ctrl51_onboard_cost.md`, "Attitude excluded (fair)".

