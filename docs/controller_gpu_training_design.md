# GPU-only controller — training-on-GPU design (task #11)

**Goal (user, 20/06):** GPU is THE controller path, never CPU. Move training onto
GPU so train + score share ONE forward/decode (the Metal shader), retiring the CPU
path and the encode/decode duplication that caused the absolute+decouple torque bug.

> **Correction (20/06):** an initial draft centered on the EDRA beam-search solver
> in `bptt_train_window`. That path is **NOT live.** Production sets
> `WNN_STATE_SPLIT=1`, so DAGGER trains via `split_train_loop` (`controller.rs`),
> which is **already solver-free.** This doc reflects the real live path.

## The live training path (WNN_STATE_SPLIT=1)

Per DAGGER round: rollout episodes → gate → **`split_train_loop`** → eval/checkpoint.
`split_train_loop` iterates:

1. **`split_record`** — forward-roll the gated episodes on current memory; record per
   step the output-layer input, PID target PWM, state inputs. *(read-only forward roll)*
2. **`scan_conflicts`** — bucket records by (frame,state) address; flag buckets whose
   PID targets disagree (PWM spread > τ). *(polynomial analysis)*
3. **resolve conflicts** — per conflict: `discriminative_walk` / `detect_accumulator`
   (polynomial separator search) → plant a Type-1 latch or Type-2 counter (a handful
   of **direct** state-cell writes at visited addresses). *(branchy, but cheap: O(conflicts), ~1–3k writes/round)*
4. **`split_retrain_output`** — re-roll on the now-modified state; nudge output cells
   toward the PID target (`output_decode_target` → `nudge_toward_pub`). *(forward roll + nudges)*
5. re-scan; repeat until no conflicts.

**Cost (≈5s/genome, rayon across genomes today):** ~90% is the **forward rolls**
(`split_record` + `split_retrain_output`) — sequential per step within an episode,
parallel across episodes/genomes. The conflict scan + planting is ~10%.

**Key facts that shape the port:**
- Already **solver-free** (no EDRA, no O(2^bits) enumeration).
- The bottleneck (forward rolls) is the **same forward** the scoring shader already
  runs on GPU (1 thread = (genome, episode); per-step recurrence handled per-thread).
- State memory is **read-only** during the output retrain; only output cells are written.
- Output writes from different episodes hit the **same** genome cells → needs
  order-independent concurrent accumulation = the IDS **`MarkerHashTable` + OI counters**.

## Design: 100% GPU compute, all data resident, CPU only sequences rounds

End-state is **full GPU** — every phase runs on-device; the CPU only launches the
kernels in the sequential round loop (normal GPU orchestration, not compute). This
is better than a hybrid, not just purer: a hybrid would have to ship ~96k records
back to the CPU EVERY round for conflict detection — that readback likely costs more
than the branchy planting itself. Keeping everything on-device kills that transfer.

| Phase | Today (CPU) | GPU form |
|-------|-------------|----------|
| forward roll (`split_record`, `split_retrain_output`) — 90% | sequential per step | reuse the scoring shader's forward+decode; 1 thread=(genome,episode) |
| output-cell nudges | sparse RMW | IDS `MarkerHashTable` + OI (order-independent across episodes) |
| `scan_conflicts` (group records by (frame,state) addr, flag PWM-spread>τ) | bucket+reduce | sort-by-address-key + segment-reduce (a GPU group-by) |
| `discriminative_walk` / `detect_accumulator` (separator search) | per-conflict scan | parallel scoring over conflicts × candidate-bits × lags → reduce to best |
| plant latch / install counter (state-cell writes) | branchy direct writes | per-conflict write kernel via MarkerHashTable (warp-divergent but parallel over conflicts×genomes) |

CPU keeps only the round-loop control flow (launch kernels, test convergence) — and
becomes the `cpu_fallback_matches_gpu` parity oracle. The branchy planting is the
LAST thing to port (P5), not a permanent CPU resident.

Why this also fixes the duplication: train + score share the ONE Metal forward/decode
(spec'd by `output_decode_target`), so encode/decode can't drift again.

### Reused primitives (from IDS GPU training)
- `MarkerHashTable` (`atomic_hashtable.rs`) — GPU cell writes via 32-bit marker-FSM + atomic CAS.
- **OI packed counters** — order-independent (algebraic-sum) nudging; perfect for
  multi-episode/multi-step nudges to the same cell.
- batched multi-genome dispatch + `common.metal` address computation.

## Phased plan — incremental toward 100% GPU (each phase parity-gated)

- **P1 — GPU `split_retrain_output`** (foundation): one kernel, 1 thread=(genome,
  episode), reuses the scoring forward (read-only state) + nudges output cells via
  MarkerHashTable+OI. Establishes the GPU-write table + forward-reuse + parity harness.
- **P2 — GPU `split_record`**: forward roll that emits per-step records to on-device
  buffers (no readback).
- **P3 — GPU `scan_conflicts`**: on-device group-by (sort by (frame,state) key +
  segment-reduce for PWM spread) → conflict list stays on GPU.
- **P4 — GPU separator search** (`discriminative_walk`/`detect_accumulator`): parallel
  scoring over conflicts×bits×lags → best separator per conflict, on-device.
- **P5 — GPU planting** (latch/counter state-cell writes via MarkerHashTable): the
  branchy last mile; per-conflict write kernel. After this the whole round is on-device.
- **P6 — retire CPU**: CPU becomes the parity oracle (`cpu_fallback_matches_gpu`);
  train+score share the one forward/decode shader.

Each phase is parity-gated against the CPU reference and must not regress the GA result.

## Decision log
- **2026-06-20: bit-exact GPU port of ALL phases (user).** No CPU residency, no
  GPU-native redesign of the branchy 10% — port scan_conflicts / discriminative_walk /
  detect_accumulator / plant_latch / install_counter to Metal kernels matching the CPU
  bit-exact, each parity-gated. Consequence: P2's records stay fully on-device (the GPU
  scan + separator consume them) — no readback. Accepted cost: the branchy ports are
  hard + risky for subtle parity bugs; mitigated by per-phase bit-exact parity tests.
- **2026-06-20: P1 done + it caught a latent scoring bug** (2-bit vs 1-bit recurrence;
  see project_resume_19jun). The forward is now correct AND single-source.

## P1 record / status
- forward_state + out_neuron_addr: shared, 1-bit MSB recurrence (correct), parity-verified.
- marker_slots.metal: shared GPU cell-write primitives (find_or_claim_slot / slot_nudge / OI).
- controller_train kernel + ControllerTrainer host + run_controller_train_parity_test: 0 mismatches.

## Validation gate (replaces the moot "solver-free" P0)
Since the live path is already solver-free, the gate is **parity + speedup**, not an
algorithm change:
1. **Parity**: GPU `split_retrain_output` forward+nudge must match the CPU trainer
   (statistical tolerance, like the existing scoring parity test) on a small fixture.
2. **Speedup**: confirm the GPU forward rolls beat rayon-CPU at production pop sizes.

If parity holds and it's faster, proceed P2→P4. The encode/decode single-source
(`output_decode_target`, landed 20/06) is the spec the shader mirrors.

## Risks / open
1. Readback volume for conflict detection (P2): mitigate by computing conflict buckets
   on-GPU or compacting records before readback.
2. State recurrence is per-step sequential within an episode — unavoidable, but the
   scoring shader already does exactly this per thread, so it's a solved pattern.
3. MarkerHashTable sizing for controller cell counts (small: state/output ≤ a few k cells/genome).
