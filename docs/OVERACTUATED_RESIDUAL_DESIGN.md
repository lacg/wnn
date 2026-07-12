# WNN Residual Control for Overactuated Multirotors — Design

Status: **Phase 0 (design + allocation substrate) — 11/07/2026.**
Owner memory: `project_overactuated_multirotor_control`. Builds on the quad
attitude paper #1 stack (AttitudeSim + phased-GA + teachers LQR/MPC/PID).

## Thesis

Overactuated multirotors (octocopters, canted hexes, omnidirectional /
tilting-rotor vehicles — ETH Omnicopter, Voliro) have an N>4 rotor set whose
thrust map has a null space: many rotor commands realize the same body wrench.
Classical control allocation picks one solution (weighted pseudo-inverse of
the allocation matrix) computed from the NOMINAL geometry. Real vehicles
deviate — motor asymmetry, frame flex, prop wear, tilt-servo backlash — and
the allocator silently misallocates inside its null space.

**Contribution:** keep the classical allocator as the baseline and let a WNN
learn a RESIDUAL correction on top of it (same substrate as paper #1's
attitude controller: thermometer-encoded observations, QSR-weighted decode,
GA-optimized connectivity, DAGGER against the allocator-aware teacher). The
WNN never has to learn the (well-understood) geometry — only the mismatch,
which is exactly the small-signal regime WNN lookups generalize well in.

## Architecture (planned)

```
attitude error ──► attitude controller (existing WNN or PID/LQR/MPC teacher)
                        │ desired wrench w = (τx, τy, τz, Fz)  [6-DoF for tilted]
                        ▼
             classical allocator  u_base = B⁺ w      (nominal geometry)
                        +
             WNN residual         Δu = f(obs, w)      (learned, small, clamped)
                        ▼
             pwm = clamp(u_base + Δu) ──► N-rotor dynamics (sim)
```

- Residual is CLAMPED (|Δu| ≤ δ, δ ≈ 0.1) so the baseline allocator bounds
  worst-case behavior — the safety argument for the paper.
- Fitness/metrics: reuse the controller fitness (err², stable%, jerk, mono —
  C10 weights) + a new allocation-efficiency term (Σu² vs the pseudo-inverse
  optimum) later.

## Phased plan

| Phase | What | Where | Gate |
|-------|------|-------|------|
| 0 | `overactuated.rs`: RotorGeometry (N rotors: position/axis/spin/k), 6×N allocation matrix, damped-pinv allocator, presets (quad+, octo-X, hex-X, canted hex), unit tests incl. quad-mixer parity with AttitudeSim | controller crate (additive module, NOT wired) | cargo tests green ✅ |
| 1 | Generalize AttitudeSim step to RotorGeometry (feature-flagged; quad preset must stay bit-identical — parity test) + Metal shader N-rotor port | controller crate + shaders | DISCUSS FIRST (live hot path) |
| 2 | Residual injection point + teacher (allocator-aware LQR) + DAGGER plumbing | controller crate + wnn/control | after Phase 1 parity |
| 3 | Phased-GA runs: nominal-geometry sanity (residual→0 expected), then D3-style asymmetry/tilt-error curricula (residual must recover) | experiments | GPU headroom (after PID-full/seed-pairs + wave-1) |
| 4 | Paper #2 experiments: octo-X + canted hex, transient metrics, FPGA projection | — | Phase 3 results |

## Phase-0 conventions (locked in code)

- Body frame: z-up, x forward, y left (matches AttitudeSim).
- Rotor thrust `T_i = k_i · pwm_i²` along the rotor's `axis` (unit, body
  frame); fixed-pitch props: T ≥ 0 only (allocator clamps negative demands).
- Drag torque: `spin_i · k_drag_i · T_i` about `axis` (CCW spin ⇒ +axis drag
  torque on the airframe, the AttitudeSim sign convention).
- Wrench = (τ; F) ∈ R⁶, columns of B are per-UNIT-THRUST contributions:
  `B[:,i] = [r_i × a_i + spin_i·k_drag_i·a_i ; a_i]`.
- Allocation solves for THRUSTS (linear), then `pwm_i = √(T_i/k_i)` — the
  quadratic motor map stays out of the linear algebra.
- Damped pseudo-inverse `Bᵀ(BBᵀ + λI)⁻¹` with small λ: planar vehicles make
  BBᵀ rank-deficient (Fx/Fy rows are zero) and damping handles it without
  case analysis.

## Why this beats "just re-tune the allocator"

Adaptive/robust allocation exists (sequential LS, cascaded QP), but runs
per-cycle optimization on the flight controller. The WNN residual is a
LOOKUP — the FPGA story from paper #1 carries over: allocation correction at
line rate with zero DSP blocks. Design-time evolutionary sparsity vs
run-time optimization is the same positioning as the IDS paper vs pruning
(memory `project_positioning_vs_pruning`).
