# GPU port of the bptt window walk — the controller's last CPU island

Design notes, 04/08/2026. Supersedes the "port phase 1, keep the beam on CPU" sketch,
which was wrong about where the constraint lives.

## What is already on GPU

```
controller_rollout        1 thread = 1 (genome, episode) closed-loop rollout    ✅
controller_train          split_retrain_output                                   ✅
controller_record         split_record                                           ✅
controller_scan / sep_walk / sep_counts                                          ✅
controller_plant_table / plant_bidir / mht_populate / mht_probe                   ✅
split_train_loop          full multi-round loop, resident memory, bit-exact      ✅
──────────────────────────────────────────────────────────────────────────────────
bptt_train_window  →  solve_partial_connectivity_qsr_reachable                   ❌
```

One island, and it is the one that costs ~150×. `metal_controller.rs:4` still declares
*"Trains stay on CPU (branchy QSR solver + DashMap writes)"* — that sentence scoped a
previous port, it is not a property of the work. Note that `split_train_loop` was ported
*around* the solver rather than through it, and that detour is the ~5800×-cell
regression measured on 21/07 (`aa20679c`). The solver has never actually been attacked.

## The window walk, section by section

`bptt_train_window` walks records **backwards** (`for d in (0..n_rec).rev()`):

| | work | touches memory |
|---|---|---|
| (a) | per-motor beam solve → `vote` over desired state bits | **reads** `output_memory.neuron_entries` |
| (b) | transition constraint — state-layer solve into `d_next` | **reads** `state_memory` |
| (c) | commit STATE layer at the recorded address | `state_memory.write_cell` |
| (d) | commit OUTPUT layer toward the teacher's PWM | `output_memory.write_cell` |

At `sn=0`, `solve_motors = 0` collapses this to (d) alone — the classic supervised
RAMLayer direct write. That is the whole 150×.

**The write path is not the problem.** (a) and (b) are pure reads producing a vote; (c)
and (d) are cheap `write_cell` calls — the same ones `sn=0` already executes at full
speed. I previously flagged "a GPU solve that round-trips to a CPU DashMap gives the win
back"; that concern was misplaced, because the expensive part writes nothing.

## The actual constraint: read-after-write inside the walk

Record `d` commits (c)/(d), and record `d-1` then **solves against the memory those
commits just changed**. So records are strictly sequential. Worse, the sequential scope
is larger than one window: `bptt_train_window` is called per trajectory chunk on ONE
`controller`, so every window and every episode of a genome shares — and mutates — the
same `state_memory`/`output_memory`.

Sequential: records → windows → episodes, all within a genome.
Parallel: **genomes** (pop 50, already exploited by `dagger_train_batch_inplace`'s
rayon `par_iter`) × **motors** (4 independent bank solves per record, combined into one
vote).

This kills the naive shape. Dispatching a kernel per record would be thousands of tiny
launches against a serial dependency — the Phase-5 kernel-launch lesson, exactly.

## Consequence: port the WALK, not the solve

The only design that respects the dependency is the one `split_train_loop` already
uses — **resident GPU memory, whole loop in-kernel**:

```
per genome:
  upload memory ONCE          SparseLayerMemory::export_for_gpu() already emits the
                              needed SparseGpuExport { keys(sorted), values, offsets,
                              counts } — the same format MetalSparseEvaluator binary-
                              searches for IDS. No new export format needed.
  ONE dispatch, threadgroup per genome:
      for d in (n_rec-1 ..= 0):          # sequential IN-KERNEL, no host round-trip
          (a) 4 motor beam solves in parallel  → vote
          (b) state-layer solve                → d_out
          (c)(d) write_cell into the RESIDENT GPU memory twin
  read memory back ONCE (or keep resident across windows — better)
```

Threadgroup mapping: **4 motors × 64 beam entries = 256 threads**, a standard Metal
threadgroup. Divergence is only across beam entries doing identical expand/score/select
work — SIMD-shaped, not branchy. Threadgroup memory for the beam is `W×K = 256`
candidates × 8 B ≈ **2 KB**, trivial.

Occupancy at production settings: pop 50 → 50 threadgroups × 256 threads = 12,800
threads on a 40-core GPU. Adequate, not lavish — and it is the *only* parallelism the
algorithm admits, so this is the ceiling, not a choice.

## Precedents that de-risk it

- `ControllerTrainer` already maintains a resident GPU twin of each genome's
  `output_memory` and applies trained cells back (`controller.rs:1194`).
- `controller_rollout.metal` runs thread-private state arrays across whole episodes
  (1800 lines), so complex per-thread state in-kernel is established.
- `export_for_gpu()` exists on `SparseLayerMemory`; the sorted-array + binary-search
  read pattern is shipping in the worker crate.

## Verification

`cargo test -p ram_controller --lib --no-default-features` now runs all 14 CPU/GPU
parity sweeps as first-class tests (04/08 — they were previously reachable only from
Python, and one had been failing undetected since the 20/07 bit-packing change). A
`parity_bptt_window` sweep comparing CPU and GPU walks bit-for-bit is the acceptance
gate for this port, and it must be written **before** the kernel.

## Open questions before code

1. Can the memory twin stay resident across windows/episodes within a genome, so a
   genome uploads once per DAgger round rather than per window?
2. Does the state-layer solve in (b) share enough structure with (a) to use one kernel
   with a mode flag, or does it want its own?
3. GPU contention: the IDS worker owns the GPU during cohorts. Same gate as
   `WNN_CONTROLLER_GPU_TRAIN` — opt-in, run when free.
4. Crate placement: `controller/` keeps it swap-free; only touch `ram_core` if the
   kernel needs shared substrate, which forces a both-wheel rebuild and a worker-idle
   swap.
