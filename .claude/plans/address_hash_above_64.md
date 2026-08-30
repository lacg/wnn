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


## 29/08 late — CONTROLLER drop-and-relearn + IDS deploy in flight
Controller (Rust, committed): genome_cells.rs `remap_bits_state/output(d, old_bits=None,
new_bits=None)`, `state_neuro(..., sb0, sb1, ob0, ob1)`, `remove_state_neuron(..., state_bits,
output_bits)` — a layer wider than 64 bits before OR after the mutation is CLEARED (relearns
from the next training pass, the same fate as a connectivity change). `None` = legacy caller =
historical remap (only correct <=64). Loud asserts where a raw address is DECODED:
controller_training.rs projected_address (beam trainer) and controller.rs
split_visited_bases (state splitting). 190/190 controller tests incl. 3 new.
Staged Python (NOT applied; tree untouched): .claude/plans/staged/
  recurrent_genome_wide.patch  (pass the widths at the 4 call sites)
  controller_abi26.patch       (_accel.py EXPECTED_ABI 25 -> 26)
  worker_abi12.patch           (accel.py EXPECTED_ABI 11 -> 12)   <- the handoff applies this one
Controller wheel rebuilt (ABI 26) — install + apply the two controller patches together, at the
ladder/probe boundary (the probe's b=40/48/64 are identity-path: old wheel gives identical results).

IDS deploy (armed 29/08 ~03:00Z, both PPID=1):
  scripts/worker_swap.py --auto-detect-running --no-restart --marker /private/tmp/worker_swap_abi12.json
  scripts/worker_abi12_handoff.sh  (log /private/tmp/worker_abi12_handoff.log)
    waits for the marker -> pip install ABI-12 wheel + sed accel.py EXPECTED 11->12 -> verify ->
    relaunch worker (rayon 13) -> requeue anything the stop interrupted -> SMOKE flow 5895 ->
    on completed: release 5896-5909 (paused) + restart reruns 6050/6051 (pending). Fails closed.
Blast radius decided by WINNER width, not config: of 32 completed "96b" IDSXD flows only 2 carry
>64-bit neurons (5888 B05-AC-r20405, 5890 B05-CE-r20404) -> rerun as 6050/6051 "-w64fix".
The 15 queued ciciot-96b (max_bits 100) were PAUSED so they fly on the fixed wheel.
Paper cohorts with >64 winners: SP 38 (abl2big/ablpln/ablqsr/bin arms), XDS 29 — Luiz's call;
every Vivado-synthesized design is <=64 bits, so the FPGA claims are untouched.

## 30/08/2026 02:5xZ — BOTH WHEELS NOW INSTALLED; all three staged patches APPLIED
Worker landed 29/08 02:27Z by scripts/worker_abi12_handoff.sh (wheel + accel.py 11->12).
Controller landed 30/08 02:5xZ, all three parts in one step:
  pip install --force-reinstall --no-deps <ABI-26 wheel>
  patch -p0 src/wnn/control/_accel.py          < staged/controller_abi26.patch
  patch -p0 src/wnn/control/recurrent_genome.py < staged/recurrent_genome_wide.patch

Installed: ram_accelerator 12, ram_controller 26. Facades: accel.py 12, _accel.py 26 —
all four agree, so the next spawned run imports cleanly.

WHY IT WAS SAFE MID-RUN (the ladder was 1h45m into b34-desir, ~2h of runway):
`lsof` on the live controller showed ram_controller.cpython-313-darwin.so ALREADY mapped,
so `import ram_controller` was already cached in that process's sys.modules — it can never
see the replacement file, and recurrent_genome.py was likewise already executed. A running
run is therefore inert to both halves of the deploy; only the NEXT spawned python picks
them up, and it picks up wheel AND patches together. The chain starts a run only when the
previous one exits, and prior runs take 3.7-4.0 h, so the collision window was ~0.

VERIFIED after the deploy: wheel ABI 26 from a neutral cwd (cwd shadows site-packages —
an unpacked wheel in `.` reports itself, which briefly gave a false 26 for the installed
build); `from wnn.control import _accel` imports (that assert is the gate the next run
hits); and every new-arity call accepted on a live GenomeCells —
remap_bits_state(d, old, new), remap_bits_output(d, old, new),
state_neuro(..., sb0, sb1, ob0, ob1), plus remap_bits_output at 96 bits taking the
drop-and-relearn path rather than remapping.

The three staged/*.patch files are now ALL APPLIED — kept for the record, do NOT re-apply.

ALL rerun sets are now queued. SP abl2big 6052-6071; SP ciciot 6072-6089 (ablpln 9,
ablqsr 4, abl3s 1, bin 4); XDS 6090-6116 (27), interleaved across the four DATASETS
(ciciot-subsample 15, unsw-temporal 7, unsw-random 4, cicids-random 1).

XDS IS 27, NOT THE 29 THIS DOC SAID. The 29 counted two CANCELLED flows —
XDS-ciciot-46M-96b-Wc-...-r63432 and -r15385. A cancelled run banked no result, so there
is nothing to invalidate and nothing to re-run; only completed flows carry a claim. Same
correction applies to any "214 flows carry >64 winners" headline: that figure is over all
statuses, and the re-runnable subset is smaller.
