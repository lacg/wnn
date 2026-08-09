# MCU control-step cost harness (Cortex-M4)

Measures per-control-step cost for the WNN controller against PID and an MLP on
a Cortex-M4, under QEMU (`mps2-an386`).

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

    python scripts/export_controller_c.py --winner <stage4_memory.yaml.gz> --out mcu/wnn_model.h
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
