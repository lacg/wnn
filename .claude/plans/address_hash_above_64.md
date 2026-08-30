# Plan: unbounded bits_per_neuron via address hashing (drafted 29/08/2026, NOT approved)

## Problem (confirmed at source, memory: project_bits_above_64_or_fold)
Every address fn does `address |= 1 << (bits-1-i)` into a u64. Above 64 bits the shift
wraps: slots i and i+64 OR onto the same bit. b=96 = a 64-bit neuron with 32 input pairs
merged, and the OR is biased (P(1): 0.5 -> 0.75), so addresses cluster. Shared substrate
(ram_core) -> BOTH wheels. 214 of 2,565 completed flows carry a winner with a neuron > 64
bits (WSWEEP 121, SP 38, XDS 29, SP100 12, IDSX 10, IDSXD 2, PHASE6 1, zINVALIDQUAD 1).
Luiz: NO cap. Must allow b=1024 and beyond.

## Design decision: hash ONLY above 64 bits; identity at or below
    bits <= 64 : address = raw tuple integer (EXACTLY today's value)  -> every existing
                 <=64 run stays bit-identical; no re-run, parity suites untouched.
    bits >  64 : gather the tuple MSB-first into ceil(bits/64) u64 words, then
                 address = mix(word_0, word_1, ..., word_k)  (64-bit output, unbiased)
Why not always-hash: the controller's cell_remap (recurrent_genome.py:354, cell_remap.rs)
depends on the address STRUCTURE A = P*2^w + S, and the dense path (bits <= 12) indexes
arrays by the raw value. Identity below 64 keeps both untouched. Why a hash and not u128:
SparseMemory keys, export_for_gpu, the sorted-key binary search and every Metal buffer
stay u64; ~1e6 keys over 2^64 buckets => collision ~1e-8 (effectively lossless), and no
new ceiling. Hash choice: a fixed-seed 64-bit mixer with a Metal twin (splitmix64 /
murmur3 finalizer chain over the words) — NOT rustc-hash's FxHasher (no Metal twin,
weak mixing). Same constants on both sides, verified by parity test.

## Touch list (one helper, every caller switches to it)
ram_core
  core/neuron_memory.rs   add `compute_address_wide(bit_reader, connections, bits) -> u64`
                          used by compute_address_sparse / _packed_bytes_sparse /
                          _packed for bits > 64 (<=64 path unchanged, byte-for-byte).
  core/shaders/common.metal  wnn_compute_address_u64: same split (<=64 raw, >64 mix),
                          expressed as ONE accumulator (WnnAddr begin/push/finish) that
                          every address loop uses — the "template". marker_train.metal
                          was NOT a copy (its compute_address already wraps the common
                          helper; the line-109 comment describes the OLD LSB-first bug).
  controller/shaders/controller_rollout.metal  3 lookup loops (state forward,
                          out_neuron_addr, split-train state address) switched to the
                          accumulator. The REMAP ops (ctrl_projected_address & co,
                          cell_remap.rs) decode bits FROM addresses and need the
                          injective raw name — they only run at <=64 today (ladder tops
                          at 36, probe at 64); wide controller neurons will fall back to
                          drop-and-relearn (the connectivity-changed path). Not wired yet.
ram_accelerator (worker)   callers already route through the helpers: ramlm.rs,
                          bitwise_ramlm.rs, adaptive/{eval_export,memory,gating_eval,
                          eval_parallel}.rs, gating.rs, multistage.rs, marker_train.rs
                          -> no logic change, recompile. Bump ABI_VERSION 11 -> 12.
ram_controller             controller.rs 17 call sites via helpers; b<=64 today so
                          bit-identical; bump ABI 25 -> 26 anyway (shared core changed).
Python
  src/wnn/hf/modeling_wnn.py:254  recomputes addresses in torch (powers-of-2 sum) — must
                          apply the same >64 hash or exported HF models silently diverge.
  wnn/ram/experiments/params.py  no new params (behaviour keyed on bits only).
Tests
  cargo test: new `wide_address_no_alias` (b=96/128/1024: slots i, i+64 give distinct
  addresses; all-ones popcount ~32 not 64-capped), `wide_address_cpu_gpu_parity`
  (extend run_cpu_gpu_parity to 96 and 200), `address_identity_below_64` (b=8,32,64
  equal old formula bit-for-bit).

## STATUS 29/08/2026 — IMPLEMENTED, TESTED, WHEELS BUILT (NOT installed)
- ram_core 82/82 (incl. 4 new wide-address tests: identity <=64 bit-exact vs the old
  formula; slots i/i+64 never alias at 65..1024; all 5 readers agree; 20k random
  96-bit tuples collision-free, mean popcount 32 (OR-fold gave ~48)).
- ram_controller 187/187 incl. all 14 CPU/GPU parity sweeps (identity path unchanged).
- ram_accelerator: CPU/GPU parity at 8, 16, 96, 200 bits — the 96/200 cases are the
  Metal twin check. Harness fix: eval re-reads TRAINED rows (fresh rows never hit a
  trained address above 64 bits, so the vacuity guard fired — correctly).
- ABI: worker 11 -> 12, controller 25 -> 26 (Rust side ONLY — see deploy).

Wheels (BUILT 29/08, ~23:40Z, NOT installed):
  /Volumes/20260401-WDBlack-SN850X-2TB/cargo-target/wheels/ram_accelerator-2026.212.37-cp313-cp313-macosx_11_0_arm64.whl  (ABI 12)
  /Volumes/20260401-WDBlack-SN850X-2TB/cargo-target/wheels/ram_controller-2026.212.37-cp313-cp313-macosx_11_0_arm64.whl   (ABI 26)
  ⚠️ same version string as the installed ABI-11/25 wheels — verify ABI_VERSION after install, not the filename.

## Deploy (rules: never deploy while a chain is armed; stage Python WITH the wheel)
**The Python facades still expect ABI 11 / 25 on purpose.** They assert EQUALITY, and
the ladder spawns a fresh python per run that imports src/wnn/control/_accel.py from the
tree — bumping EXPECTED_ABI before the wheel is installed kills the next run at import
(the 3-runs-in-<1s lesson). At deploy, in ONE step per wheel:
  worker:     scripts/worker_swap.py <ABI-12 wheel>  AND  wnn/accel.py EXPECTED -> 12
  controller: pip install <ABI-26 wheel>              AND  wnn/control/_accel.py EXPECTED_ABI -> 26
1. Branch; implement; `cargo test -p ram_core` + `-p ram_controller --no-default-features`.
2. `maturin build --release` (worker) — BUILD ONLY; swap via scripts/worker_swap.py at
   worker-idle. 75 IDSXD flows queued -> earliest natural idle is the end of the cohort
   unless Luiz pauses the queue.
3. Controller wheel installs any time (worker never imports it) BUT the sweep ladder +
   handoff supervisor are armed -> wait for the probe/ladder boundary.
4. Smoke ONE b=96 IDS flow before any cohort.

## What it changes for results
b <= 64: nothing (identity). b > 64: addresses differ -> any >64 run is a DIFFERENT
neuron; results will NOT match and are not expected to. Existing >64 results remain true
measurements of "64-bit OR-folded neurons", invalid only as evidence about wide neurons.
