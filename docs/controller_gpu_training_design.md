# GPU-only controller — training-on-GPU design (task #11)

**Goal (user, 20/06):** GPU is THE controller path, never CPU. Move DAGGER training
onto GPU so train + score share ONE forward/decode (the Metal shader), retiring
the CPU path and the encode/decode-must-agree duplication that caused the
absolute+decouple torque bug.

## What we have today (mapped 20/06)

| Path | Where | Status |
|------|-------|--------|
| **Scoring** (closed-loop rollout, forward + RK4 physics + reward) | `controller_rollout.metal` / `metal_controller.rs`; 1 GPU thread = (genome, episode); cells READ-ONLY (sorted sparse + binary search) | **already 100% GPU** |
| **Training** (DAGGER: rollout → gate → BPTT windows → `bptt_train_window`) | `dagger_train.rs` + `controller.rs`; rayon across genomes | **CPU** |
| **GPU cell WRITES** (the usual GPU blocker) | IDS `MarkerHashTable` (`atomic_hashtable.rs`) + OI packed counters + `marker_train.metal` | **solved for IDS, reusable** |

## The pivotal finding

Controller training = **forward rollout** + **backward (credit assignment + nudge)**. Of those:

- **Forward rollout** — already on GPU (the scoring shader does forward + physics).
- **Cell nudging** (output + state writes) — GPU-able today via IDS's `MarkerHashTable` + **OI counters** (order-independent algebraic-sum nudging, *exactly* what a recurrent multi-nudge-per-cell trainer needs). Reusable nearly as-is.
- **The EDRA constraint solver** (`solve_partial_connectivity_qsr_reachable`, beam search, ~543 calls/window, ~17s/window) — the **bottleneck** AND the one piece that does **not** GPU-ify (branchy beam search + conflict checking + solve→nudge→infer→check). This is the whole blocker.

So the design question reduces to: **what do we do about the EDRA solver?**

## Two tracks

### Track A — solver-FREE training (recommended)

The EDRA solver only exists to do **credit assignment for the recurrent STATE layer** (infer "what state bits *should* have been" to make the output match the teacher). But the codebase **already has a solver-free alternative**: the **integral-target** mode (`WNN_STATE_INTEGRAL_TARGET`, `state_integral_targets` in `bptt_train_window`) trains the state layer toward a **direct thermometer encoding of the PID integral** — explicitly described in-code as replacing "the fragile indirect output∧transition solve" so the state "actually learns to be the integrator."

If both layers use **direct** targets:
- **output cells** → teacher motor-target (already direct; the bug we just fixed is exactly this encoding),
- **state cells** → direct integral-target (or another direct state target),

then **the solver disappears**, and controller training becomes **structurally identical to IDS GPU training**:

```
1. GPU rollout (reuse the scoring shader's forward+physics, READ-ONLY cells)
   → record per step: (state_addr[t,n], out_addr[t,m], state_target, motor_target)
2. GPU batch-nudge (IDS marker_train.metal pattern: MarkerHashTable + OI counters)
   → nudge each recorded address toward its target, order-independent
3. commit OI counters → cells; export sparse for the next round/scoring
```

This is the clean win on every axis:
- **GPU-only** (no CPU solver in the loop).
- **One forward/decode** — the rollout reuses the *scoring* shader, so train+score share a single Metal implementation; the Rust `decode_outputs` becomes a parity *oracle*, not a live twin. **Kills the duplication you flagged.**
- **Reuses proven IDS machinery** (MarkerHashTable, OI, batched dispatch, `common.metal` address computation) — minimal new GPU code.

**The catch — it changes the learning algorithm.** Direct state-targets ≠ EDRA credit assignment. We must verify solver-free training matches or beats EDRA on the controller. *But* the project already leaned this way (integral-target was added precisely because the indirect solve is "fragile"), so it's plausibly **better**, not just faster. De-risk by validating solver-free training **on CPU first** (cheap, no GPU work) before porting.

### Track B — port the EDRA solver to GPU

Keep EDRA credit assignment, move the beam-search solver into a Metal kernel (one warp per solver call, atomic writes). Much harder: branchy beam expansion + shared-bit conflict checking + the sequential solve→nudge→infer loop. High risk, uncertain payoff, and it *keeps* two forward implementations. **Not recommended.**

## Proposed phased plan (Track A)

- **P0 — algorithm validation (CPU, cheap):** make solver-free direct-target training a first-class CPU mode (it half-exists behind `WNN_STATE_INTEGRAL_TARGET`); run a controller GA with it and confirm it matches/beats EDRA. **Gate: if solver-free can't learn, stop — GPU port is moot.**
- **P1 — GPU memory writes:** instantiate `MarkerHashTable` + OI for the controller's state/output layers (per-genome slot regions). Reuse `atomic_hashtable.rs` unchanged.
- **P2 — GPU rollout-with-record:** extend the scoring shader to also emit per-step (addresses, targets) into buffers (teacher PID can be precomputed or run in-shader).
- **P3 — GPU batch-nudge kernel:** adapt `marker_train.metal` (strip its address-compute, feed recorded addresses) → nudge via OI.
- **P4 — unify + parity:** train+score share the one shader forward/decode; retire CPU training to a `cpu_fallback_matches_gpu`-style parity oracle.

## Risks / open questions
1. **Does solver-free training work?** (P0 answers this — cheap, do first.)
2. Read-during-write: avoided — DAGGER collects the trajectory first (read-only rollout), then nudges (separate phase), so no mid-rollout cell mutation. Cross-episode nudges to the same genome's cells are fine (OI = order-independent sum).
3. Teacher (PID) on GPU vs precomputed per episode — minor.
4. Parity tolerance: training parity is statistical (chaotic feedback), like the existing scoring parity test.

## Recommendation
**Track A, P0 first.** Validate solver-free direct-target training on CPU before any GPU work — it's the cheap gate that determines whether the whole GPU-only vision is reachable, and it's the same change that unifies the forward path.
