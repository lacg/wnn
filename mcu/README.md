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
