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

## Q1 ANSWERED — residency is not an optimisation, it is the whole port

Measured, not estimated. Export is 9 B/cell (u64 key + u8 value); the loop dimensions
are `reward_gated.py` defaults — 8 rounds × 24 episodes × ceil(2000/32) = 63 windows =
**12,096 windows per genome per training**.

| | export/genome | per-window traffic | resident (pop 50) |
|---|---|---|---|
| sn=8 at `--max-cells 180000` | 1.62 MB | 19.6 GB/genome → **980 GB per GA generation** | 81 MB |
| P4a measured (986,313 cells) | 8.88 MB | 107 GB/genome → **5,369 GB per generation** | 444 MB |

Four to five orders of magnitude. A GPU walk that ships memory per window is not slower
than the CPU — it is impossible. Residency is therefore a hard precondition, not a
tuning choice.

**And it does not exist today.** `ControllerTrainer::train_seeded`'s own doc is explicit:
*"rounds 2+ must seed the marker table with the controller's present cells before
nudging. **The seed is host-side**: replay the existing cells into the marker buffers."*
The existing resident chain (`record_dispatch` → scan → resolve) keeps *records* resident
within one dispatch; cells still round-trip to the host on every train call.

### Consequence: the scope is the ROUND LOOP, not the walk

For memory to stay on the GPU for a genome's whole training, nothing in the loop may
require host-side cells. That means porting the reward-gated round loop itself:

```
per genome, memory uploaded ONCE:
  for round in 0..8:
      rollout  N episodes        ← ALREADY GPU (controller_rollout)
      reward + gate              ← small, trivially GPU-able
      for each gated trajectory:
          for each window:       ← the walk, sequential, in-kernel
              (a)(b) solves      → vote
              (c)(d) write_cell  → into the RESIDENT twin
  download cells ONCE
```

This is the same architectural move `split_train_loop` already made for its algorithm —
which is why that one is "100% GPU" and this one is not. The honest scope of the port is
therefore **"GPU-resident `dagger_train`"**, not "GPU solve".

The pieces mostly exist (rollout kernel, sparse export format, memory-twin precedent);
what is missing is the orchestration that never lets cells touch the host mid-training.

### Inventory correction — the substrate is ~80% built

Reading `controller_train` (shaders/controller_rollout.metal:1148) changes the scope
again, favourably. That kernel ALREADY is a per-genome in-kernel training walk with a
resident, population-wide, writable cell table:

```
uint g = gid;                     // one thread per GENOME — the batch axis, already chosen
device atomic_uint* out_values;   // cells are ATOMICALLY WRITABLE in-kernel
device ulong*       out_keys;
device atomic_uint* out_markers;
device const uint*  slot_off;     // [num_genomes * num_out]  <- per-(genome,neuron) layout
device const uint*  slot_cap;     // [num_genomes * num_out]     ALREADY population-wide
device const ulong* state_keys;   // state layer: sorted READ path
ep_base/ep_count/step_base/step_count + gyros/accels/targets/pid_pwms
                                  // it already walks episodes and steps in-kernel
```

Supporting helpers that already exist and are parity-tested (`parity_mht_lookup` is one
of the 14 green sweeps): `find_or_claim_slot` (atomic open-addressing insert),
`mht_lookup`, `bsearch_cell`, and `plant_cell`-equivalent mode-native cell values.

So the resident writable twin is NOT new work — it exists, it is laid out for a whole
population, and its read path is bit-exact against the CPU. What `controller_train`
implements is `split_retrain_output`, i.e. the direct output write — the analogue of the
walk's section (d). What it does NOT implement:

| missing | what it needs |
|---|---|
| (a) per-motor beam solve → vote | the new kernel work; threadgroup per genome, 4 motors × 64 beam entries |
| (b) transition constraint solve | same machinery, state layer |
| (c) commit STATE layer | the state layer is currently a READ-ONLY sorted export; it needs the same `markers/keys/values` + `slot_off/slot_cap` treatment the output layer already has |

**The port is an extension of the `controller_train` kernel family, not a new pipeline.**
That is a materially smaller and better-understood piece of work than "port
dagger_train", and it inherits an already-parity-proven memory substrate.

### Next concrete step

Give the STATE layer the writable-table treatment the OUTPUT layer already has
(`state_markers`/`state_keys`/`state_values` + `slot_off`/`slot_cap`), so section (c)
can commit in-kernel. It is bounded, it mirrors an existing verified structure, and
`parity_bptt_window` already exists to judge it.

## Open questions before code
2. Does the state-layer solve in (b) share enough structure with (a) to use one kernel
   with a mode flag, or does it want its own?
3. GPU contention: the IDS worker owns the GPU during cohorts. Same gate as
   `WNN_CONTROLLER_GPU_TRAIN` — opt-in, run when free.
4. Crate placement: `controller/` keeps it swap-free; only touch `ram_core` if the
   kernel needs shared substrate, which forces a both-wheel rebuild and a worker-idle
   swap.
