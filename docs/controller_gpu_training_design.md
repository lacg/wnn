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

## Design: GPU the hot path, reuse what exists

| Piece | Today | Plan |
|-------|-------|------|
| Forward roll (record + retrain) — the 90% | CPU | **GPU**, reuse the scoring shader's forward+decode (one source of truth) |
| Output-cell nudges | CPU sparse RMW | **GPU** via IDS `MarkerHashTable` + OI (order-independent across episodes) |
| Conflict scan / `discriminative_walk` / planting — the 10% | CPU | **stays CPU** (branchy, cheap, O(conflicts)); revisit later if it ever dominates |

This is a **hybrid** (hot path GPU, cheap branchy planting CPU), not "100% GPU" — but
it puts GPU where ~90% of the time is, and crucially makes train+score share **one**
Metal forward/decode (the duplication you flagged is gone). Pure-GPU planting is a
later stretch only if the CPU 10% becomes the bottleneck after the forward rolls move.

### Reused primitives (from IDS GPU training)
- `MarkerHashTable` (`atomic_hashtable.rs`) — GPU cell writes via 32-bit marker-FSM + atomic CAS.
- **OI packed counters** — order-independent (algebraic-sum) nudging; perfect for
  multi-episode/multi-step nudges to the same cell.
- batched multi-genome dispatch + `common.metal` address computation.

## Revised phased plan

- **P1 — GPU `split_retrain_output`** (biggest single win): one kernel, 1 thread =
  (genome, episode), reuses the scoring forward (read-only state) + nudges output cells
  via MarkerHashTable+OI. Parity vs CPU (`cpu_fallback_matches_gpu`-style).
- **P2 — GPU `split_record`**: emit the per-step records (or the conflict buckets
  directly) from a GPU forward roll, minimizing CPU readback.
- **P3 — keep conflict-detect + planting on CPU**, fed by GPU records; commit planted
  state cells, then loop P1.
- **P4 — unify**: train + score share the one shader forward/decode; CPU forward
  becomes the `cpu_fallback_matches_gpu` parity oracle.

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
