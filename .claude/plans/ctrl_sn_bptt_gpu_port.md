# Plan: GPU-resident sn>0 BPTT walk (controller ABI 35)

Status: DONE — shipped as controller ABI 35 (commit 3e0ba2d2, 05/11/2026), bit-exact vs CPU, selectable via `--bptt-walk gpu`. Original design notes below (05/11/2026). Builds on
`docs/gpu_solve_port_design.md` (04/08) and commits 553d45fe / 876793f3 / 33ca6546.

## 0. What already exists (do not redo)
- Per-record GPU solve: `controller_phase1_topk` + `controller_beam_search`, `solve_layers`,
  genome coalescer (`metal_controller.rs:11893-12127`). Gated `WNN_CONTROLLER_GPU_SOLVE=1`,
  default OFF. Bit-exact (`parity_bptt_window ... gpu_walk_matches_cpu`).
- Measured 04/08: CPU 0.304 ms/record, GPU batched 0.323 ms/record (0.94x), i.e. no gain.
  Cause: every record re-exports and sorts the whole memory on the host, then waits about 0.9 ms
  for the sync. The beam is one thread per instance, serial over neurons.
- `controller_state_commit`, `controller_train` (output commit), atomic open-addressing tables
  (`find_or_claim_slot`, `mht_lookup`). All parity-tested.

## 1. Where the time goes (sn=4 smoke, `/private/tmp/ma_smoke_axC_sn4.out`)
- 2134 s real, 28077 s user (13 cores saturated). sn=0 anchor: 129 s. Grid: 377 s vs 28 s.
  Neurons gen: 791 s vs 33 s. Memory stage: 3 s (no training). So the 16-40x is entirely DAgger
  training, in `bptt_train_window` sections (a) and (b). The smoke ran the CPU path (flag unset).
- Per record (CPU), cost is O(total cells), with 370-490k cells per genome:
  `neuron_entries()` clones the DashMap of every one of 256 output and 4 state neurons, builds a
  `HashSet`, and does a Hamming scan over all of them. The phase-2 beam then runs SERIALLY over
  4 motors (only phase 1 uses rayon). The 4 state neurons hold the largest entry lists, because
  every record writes them, so the state solve is a long serial tail.
- Step 0 confirms this split with `WNN_BPTT_PROFILE=1` section timers before any kernel work.

## 2. Parallel vs sequential
- Sequential (read-after-write): records → windows → episodes → rounds within a genome.
  Record d-1 solves against d's commits (`*_order_dependent` test proves it).
- Parallel: genomes (pop), the 4 motor solves plus the (b) state solve within a record, neurons
  (phase 1), entries within a neuron, and beam candidates (64 × k_top).
- Decomposition: the CPU must NOT drive the backward loop per record. At about 0.9 ms per sync
  and about 387k records per genome in production, that is roughly 350 s per genome of sync alone.
  The CPU orchestrates per WINDOW-BATCH (or per round). The backward record loop runs inside the
  kernel, and memory stays resident across the round loop.

## 3. Kernels and layout (controller/shaders only; prepend common.metal + marker_slots.metal)
- Resident population tables for BOTH layers: keys/values/markers plus `slot_off/slot_cap` per
  (genome, neuron). The state layer gets the same writable treatment the output layer has.
  Upload once per training, download once.
- Host precomputes per step, once per round: frame bits, `rec_step`, and teacher target bits.
  These are state-independent because they come from replayed observations (verify).
- `controller_bptt_walk`: one threadgroup per genome, 256 threads.
  - Forward roll: state reads use `wnn_addr_*` and `mht_lookup`.
  - Backward d-loop:
    - phase 1: one simdgroup per neuron, with a SIMD top-k merge over the slots. The untrained
      low-Hamming enumeration is split across lanes.
    - beam: 64 × k candidates scored in parallel, then a bitonic sort on the key
      (int rank, parent, addr_idx). This reproduces the CPU stable sort.
    - vote, (b), (c) `state_commit`, (d) output commit.
  - The beam lives in threadgroup memory.
- Cell semantics only via `wnn_cell_weight` / the existing `ctrl_nudge_distance`. No copies.

## 4. Determinism and parity
- BIT-EXACT, no tolerance. The walk is order-dependent and chaotic, so a single flipped vote
  compounds. Integer ranks already reproduce the f64 order (per-neuron saturation is a common
  shift), and atomics only claim slots. Values are written by the single owning threadgroup.
- The CPU reference stays the default and the fallback.
- `cargo test -p ram_controller --lib --no-default-features`:
  - extend `parity_bptt_window` to multi-window and multi-episode resident walks, at production
    shape (sn 4/8, sb 24, ob 24, levels 64, obs-pwm, translation), for BINARY/TERNARY/QUAD;
  - add a whole-`dagger_train` cell-digest parity test.

## 5. Expected gain and memory
- About 400k cells × 16 B ≈ 6-12 MB per genome; pop 50 ≈ 0.6 GB.
- Phase 1 is bandwidth-bound, at about 5 MB scanned per record per genome.
- Estimate: 10-30x on sn>0 training, so the sn=4 smoke goes from about 35 min to about 3-5 min.
  This must be measured on a free GPU with the existing `#[ignore]` bench before any default flips.

## 6. ABI, effort, risks
- `ram_controller` ABI 35 only. No `ram_core` change, so no worker wheel and no swap.
  Bump the `wnn/control/_accel.py` assert.
- Effort (agent wall-clock): profile + CPU quick wins 1-2 h; state writable table 2 h;
  walk kernel 4-6 h; parity 2 h; wiring/flag/bench 1-2 h. Total about 10-14 h.
- Risks:
  - slot_cap overflow mid-round (needs an overflow flag and a resize between batches);
  - GPU watchdog on long kernels (dispatch per window-batch);
  - cancel responsiveness;
  - Option A integral-target and `protect_learned` must be ported or refused loudly;
  - GPU contention with the IDS worker.
- Pre-existing defects to fix in the same change:
  1. The beam kernel has no `total_input_bits <= 256` guard (`CTRL_BITS_WORDS=16`), so it
     silently overflows into the next beam slot whenever the state input exceeds 256 bits.
     That is reachable with `input_window_k >= 2` at current feature counts (not yet checked
     at a specific recipe).
  2. GPU `k_top.min(8)` vs CPU unbounded: silent divergence if topk > 8.

## 7. Decisions for Luiz
1. GPU policy while IDS cohorts own the GPU: opt-in flag (default off) vs default on.
2. Do the Stage-0 CPU quick wins first (exact; speeds the CPU fallback too)? Recommended.
3. Confirm bit-exact as the acceptance bar (recommended) vs a tolerance.
4. Retire the per-record GPU solve path once the walk lands?
