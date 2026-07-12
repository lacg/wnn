# MODE_BINARY — classical RAM memory mode (Luiz order 12/07/2026)

Goal: a WiSARD/N-tuple-classical arm for the granularity ablation (IJCNN-2004
lineage): binary cells, one-shot training, no graduated confidence.

## Semantics (locked)
- Cell states: FALSE(0) / TRUE(3) only, stored in the existing 2-bit fabric
  (SEMANTIC 1 bit/cell; physical packing unchanged — note for FPGA projection:
  count 1 bit/cell).
- TRAIN rule: target_true visit → cell = TRUE (saturating set, one observation
  suffices). target_false / negatives → NO-OP (classical discriminators train
  on own-class examples only). Order-independent BY CONSTRUCTION (set is
  commutative) — no OI counter machinery needed; WNN_ORDER_INDEPENDENT_TRAIN
  is a no-op for this mode.
- EVAL weight: cell >= 2 → 1.0 else 0.0. Sparse empty default = FALSE(0) →
  weight 0 ("never seen → no vote"). empty_value unused.

## Wiring
- ram_core/neuron_memory.rs: MODE_BINARY=3; cell_to_weight arm;
  empty_word_for_mode → build_empty_word(FALSE); ClusterStorage::new
  empty_cell=0 for BINARY; add `set_cell_true()` (or branch nudge callers).
- core/shaders/common.metal: WNN_MODE_BINARY=3 + wnn_cell_weight arm
  (single source; NO per-shader copies — CLAUDE.md rule).
- TRAIN sites (branch on mode where TERNARY/QUAD already branch):
  * bitwise_ramlm.rs ~1059 (mode match: nudging/commit_ternary) — add BINARY
    arm: set-TRUE on target, skip negatives, skip OI machinery.
  * marker_train.metal + core/shaders/marker_slots.metal `slot_nudge` FSM —
    BINARY arm: claim slot → write TRUE; target_false → no write.
  * marker_train.rs host + CPU reference (parity fns).
  * adaptive/: check evaluate_genomes_parallel_hybrid CPU train inner loop
    (grep `nudge`/train writes in adaptive/) — same rule.
- QUAD_BINARY readout paths (bitwise_ramlm ~1254 count_true) can be reused
  for BINARY eval on LM paths, but v1 scope = IDS ONLY: make LM paths return
  a loud unsupported error for MODE_BINARY.
- worker.py memory_mode_map: add "BINARY": 3. params registry already has
  memory_mode.
- Dashboard: memory_mode passthrough is generic (string param) — verify only.

## Tests (cargo, worker crate)
1. cell_to_weight table: BINARY [0,0,1,1]; empty-sparse read = 0.
2. Train rule: one positive visit → TRUE; negatives never clear; repeat
   visits idempotent; order-independence trivially (shuffle = same cells).
3. GPU↔CPU train parity on the marker path for BINARY (mirror the existing
   cpu_fallback_matches_gpu pattern).
4. e2e: tiny IDS eval QUAD vs BINARY on synthetic separable data — BINARY
   must be functional (nonzero F1), not necessarily better.

## Cohort (after implementation + worker swap)
- SP-unswt-abl1b-16bWb-n10-r{seed} — same 10 seeds as abl3s (extend
  scripts/create_ternary_ablations_wave1b.py pattern; memory_mode=BINARY).
- Deploy: build ram_accelerator wheel; swap at worker idle
  (scripts/worker_swap.py — VERIFY its between-flows semantics; the BINARY
  flows must not surface before the swap or they fail on the unknown mode →
  simply requeue after swap if so).

## Analysis
Extend the paired QUAD-vs-TERNARY analysis to 3 arms (QUAD/TERNARY/BINARY,
seed-matched n=10, all 5 genome types × 7 modes × 3 metrics + best-of-best).
Framing: granularity ablation 1-bit classical → 3-state → 4-state.
