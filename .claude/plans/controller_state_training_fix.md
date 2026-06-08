# Proposal: give the controller's recurrent STATE layer a real learning signal

Status: PROPOSED (08/06/2026), NOT implemented. Rule 1 — investigated first.

## Problem (data-backed)
The WNN attitude controller settles to a flat **steady-state offset** (5° stability 83% vs PID 98%; 0% divergence; soft-fails hold ~5.6° flat, tail-std≈0). Root cause: the recurrent **state layer is effectively untrained** → the controller is a memoryless proportional controller (no integral action) → can't drive residual error to 0.

### Why (corrected from the first guess)
- NOT the `train_state_step` guard (controller.rs:787, `state_neurons != num_motors*levels → return 0`). That's a dead/legacy per-step path; the GA/lamarckian training does NOT use it.
- The real path is **`bptt_train_window`** (controller.rs:965), proper BPTT-through-time, wired via `reward_gated.py:311` + `dagger_train.rs:534`, `bptt_window=32` default. It DOES write state cells (line 1135) — but empirically writes **~1 state cell vs 24,780 output cells** (state_cell_counts `[1,0,...]` after 50 eps). The state layer is **STARVED, not gated**.
- Mechanism (chicken-and-egg): empty state → constant baseline output → constant `prev_state` fed back → state layer sees ~1 address pattern → can write ~1 cell. Plus BPTT's "commit a state bit only where the output-constraint solve and the transition solve AGREE" gate (controller.rs:1112) almost never fires when both layers start empty. Output trains (direct PID-PWM target); state never bootstraps.

## Fix options (ranked)

### A. Direct integral target for the state layer (RECOMMENDED)
The PID teacher already maintains an explicit **integral accumulator** (∫ attitude error). Expose it; thermometer-encode it (like the sensor inputs) into a per-state-neuron target; train the state cells toward it with the same QSR nudge — exactly how the output layer gets its direct PID-PWM target. The state layer then learns to *be* the integrator; the output layer (which reads `[frame|state]`, Mealy) can use it like PID uses its I term.
- Pros: bypasses the fragile indirect BPTT solve + the chicken-egg; gives a strong, direct, dense signal; targets the EXACT deficit (missing integral); helps at every tilt.
- Cons: Rust change (new target path in bptt/dagger train); need to plumb the PID integral out of `AttitudePID`; design the integral→state-bits encoding.

### B. Bootstrap two-phase (output-first, then state)
Train output to confidence, THEN enable state BPTT once the output can vote a meaningful "desired state." Cheaper, but still relies on the fragile solve + the constant-prev_state starvation may persist.

### C. Break the constant-prev_state starvation (cheap DIAGNOSTIC first)
Inject `prev_state` diversity during training (random state init per episode, or small state perturbations) so the state layer visits varied addresses and BPTT can write more than 1 cell. Run THIS first as a probe: if state_universe jumps from ~1 to many and stability improves, it confirms the chicken-egg mechanism and may itself be a partial fix.

### D. Relax the agreement gate (controller.rs:1112)
Commit state toward `d_out` even when the transition solve disagrees. More writes, noisier; a knob to A/B.

## Recommended sequence
1. **Probe (C):** cheapest test of the mechanism — does breaking constant-prev_state let BPTT populate the state layer? Re-measure `state_universe` + stability. ~1h, no architecture change (training-time perturbation).
2. **If confirmed → implement A** (direct PID-integral target) behind a flag (`WNN_STATE_INTEGRAL_TARGET`), test on the 5° winner spec first.
3. Validation: `state_universe` ≫ 1; 5° stability climbs 83%→toward 98%; trajectory steady-state offset shrinks (flat at lower value); err holds or improves. Multi-seed (Step-0 harness) to confirm signal.

## Risk / scope
- Rust change (controller.rs + dagger_train.rs), rebuild `maturin develop --release`. Gate behind env flag, default OFF until measured. Re-eval tooling exists: build_controller(controller_genome_from_arch(...)) + run_episode (Steps 0/1).
- The tilt sweep ([[project_lamarckian_redundancy_finding]]) stays SHELVED until this closes/characterizes the 5° gap.

## Key code anchors
- controller.rs:965 `bptt_train_window` (state commit 1119-1138, agreement gate 1112).
- controller.rs:787 `train_state_step` (dead path, the guard red herring).
- reward_gated.py:293 chunked bptt call; dagger_train.rs:534 Rust path; bptt_window default 32.
- AttitudePID (pid.py) — source of the integral term for option A.
