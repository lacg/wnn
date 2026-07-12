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
| 1 | Generalize AttitudeSim step to RotorGeometry (feature-flagged; quad preset must stay bit-identical — parity test) + Metal shader N-rotor port | controller crate + shaders | DISCUSS FIRST (live hot path) — CPU ✅ (step_n, cd6deec5) · Metal step 1 ✅ (function-constant kernel, 2af70830) · Metal step 2 ✅ (specialized pipeline cache + RotorGpu upload + `score_controllers_metal(geometry=, rotor_asym=)` + CPU/GPU parity suite, 477f02d6) · step 3 ✅ (ABI 4 wheel INSTALLED 12/07 — Luiz reordered ahead of seed-pairs; `score_controllers_cpu` geometry twin, `GeometryConfig`/`EpisodeConfig.geometry` → evaluator passthrough, octo GPU/CPU smoke parity; known pre-existing scorer difference: mono is last-step on GPU vs per-step mean on CPU — untouched for seed-pairs fitness comparability) |
| 2 | Residual injection point + teacher (allocator-aware LQR) + DAGGER plumbing | controller crate + wnn/control | after Phase 1 parity — step 1 ✅ (`AllocLqrRs` teacher, ABI 5) · step 2 ✅ (residual composition: shared `AllocBaseline` precomputed-pinv f32 twin — in-kernel `alloc_step` buffer(28) + CPU `rollout_one` composition; `alloc_*` kwargs on BOTH scorers + `AllocResidualConfig`/`EpisodeConfig.alloc_residual` → evaluator; residual→0 ≡ teacher proven GPU↔CPU, flies 0.94° on octo; ABI 7 installed). step 3 ✅ + step 4 ✅ (12/07, Luiz-approved design: geometry-aware reference driver — AllocLqrRs + step_n — in fit_thresholds/record_address_universe; ALL stages score-only via score_genomes when ec.geometry set (grid/NEURONS/MEMORY/holdout/stage-result — no DAGGER anywhere); MEMORY builds random cell genomes over the geometry-recorded universe; `--geometry {octo-x,canted-hex,quad-plus}` + perturb/asym/alloc CLI with loud gates (--lamarckian/--teacher-schedule/--decouple N≠4/--action-repeat≠1 refused); final summary reports the honest "vs alloc-LQR" baseline via an empty-controller composed score; AttitudeSim.geometry_rows exporter ABI 8). E2E octo smoke green + quad --lamarckian regression green. |
| 3 | Phased-GA runs: nominal-geometry sanity (residual→0 expected), then D3-style asymmetry/tilt-error curricula (residual must recover) | experiments | CPU screenings 12/07/2026 (single-seed, pop 20, 1000 steps, C10 attitude fitness): octo nominal sanity PASS (winner 1.42° ≈ baseline 1.35°, residual→0) · octo L1 ±2°/3mm/5% FLAT (1.38≈1.36°) · octo L2 ±5°/8mm/10% FLAT (1.39≡1.39°) · canted-hex(20°) L2 CONFIRMATION n=4 (seeds 31337002-5): winner-vs-baseline Δ = +0.06/0.00/−0.05/+0.04° → mean ≈ +0.01±0.05° = **statistically ZERO** (the single-seed 'win' was noise) · hex-L3 ±10°/15mm/20% @31337002: Δ +0.06° (same noise scale; baseline degrades only 1.41→1.50°). VERDICT: the classical allocator + LQR is remarkably ATTITUDE-robust to geometry mismatch at all tested levels — the residual's contribution must be shown in ALLOCATION EFFICIENCY (the Σu² term, shipped ABI 9), not attitude error. That robustness is itself a paper-worthy framing. (Baseline-row caveat: runs before the 12/07 pure-baseline fix used the offset-composed baseline; deltas comparable to ~0.01°.) First effort-weighted screening (err .35/stable .25/jerk .15/mono .05/effort .20, octo-L2) launched 12/07 — logs/controller/octo_effort_20260712. Production multi-seed GPU runs after chain drain. |
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

## Phase-2 step 3 — proposed design (12/07/2026, for Luiz review)

**Status: IMPLEMENTED 12/07/2026 (Luiz-approved) — see Phase-2 row above.**

**Key collapse: the residual student needs NO teacher-label plumbing.** With
teacher ≡ baseline (AllocLqrRs delegates to the same AllocBaseline the
scorers compose on — proven by `gpu_alloc_residual_zero_equals_teacher`),
the inverse-composed DAGGER label is identically neutral:
`y = 0.5 + (pwm_teacher − pwm_base)/scale = 0.5`, on nominal AND perturbed
vehicles (no oracle knows the perturbation — the teacher only has the nominal
model). So DAGGER-against-the-allocator-teacher degenerates to "output
neutral", and an EMPTY memory already decodes 0.5 = neutral. The mismatch is
learned by GA FITNESS (connectivity + cell evolution), not by imitation —
which is the paper's story anyway.

Step 3 therefore reduces to:
1. **Record path for GA-Memory (paradigm B) on N motors** — the address
   universe along residual-composed rollouts. The record/train kernels'
   `pwm_acc` arrays are already MAX_ROTORS-wide (12/07); remaining: the
   record host path takes composed rollouts (alloc baseline) + N-motor
   traces instead of the quad teacher rollout.
2. **phased_ga residual mode** — skip DAGGER/Lamarckian training stages
   (empty memory = neutral residual is the correct init); GA-Neurons evolves
   connectivity, GA-Memory evolves cells over recorded addresses; fitness
   from the residual-composed scorers (already plumbed, ABI 7).
3. NOT needed: generalizing TrajectoryRs/[f32;4] label plumbing, the split
   trainer, or Teacher enum to N motors — deferred until a mismatch-aware
   teacher exists (none planned).

Step 4 (CLI) then wires: `--geometry {octo-x,canted-hex,quad-plus}`
[+ cant/tilt-err/pos-err/rotor-asym] + `--alloc-residual scale,clamp` →
GeometryConfig/AllocResidualConfig, and gates unsupported stage combos loudly.

## Why this beats "just re-tune the allocator"

Adaptive/robust allocation exists (sequential LS, cascaded QP), but runs
per-cycle optimization on the flight controller. The WNN residual is a
LOOKUP — the FPGA story from paper #1 carries over: allocation correction at
line rate with zero DSP blocks. Design-time evolutionary sparsity vs
run-time optimization is the same positioning as the IDS paper vs pruning
(memory `project_positioning_vs_pruning`).

## RESOLVED (ABI 11, Luiz rule): residual anchor = NEUTRAL_DECODE

The offset below is FIXED: the residual now anchors at the untrained-cell
decode value DERIVED from the cell semantics (`QSR_WEIGHTS[EMPTY_U8]` — QUAD
0.75, a ternary substrate would give 0.5 automatically), single-sourced as
`controller::NEUTRAL_DECODE` (exported to Python; compose_residual /
residual_train_target and the delta-control neutral all share it). An
untrained residual is now EXACTLY the baseline (bit-identical rows —
property-tested). Pre-ABI-11 runs (incl. E5) composed at a hardcoded 0.5
and carried the hidden offset described below.

## Historical correction 12/07/2026 — "EMPTY memory = neutral residual" was imprecise (pre-ABI-11)

The controller's untrained sparse cells read EMPTY = 2 = WEAK_TRUE, which the
QSR decode maps to **0.75, not 0.5**. An empty-memory residual therefore
composes as `clamp((0.75−0.5)·scale, ±clamp)` = a **+clamp collective offset**
on every rotor — measured on the symmetric octo: attitude ≈ unchanged
(Δ≈0.01°; only approximately neutral because thrust is quadratic in PWM) but
**+69% allocation effort** (mean Σu² 3.38 vs the pure allocator's 2.00).
Consequences:
- All screening-ladder ATTITUDE conclusions stand (the 0.01° offset effect is
  ~6× below the hex signal), and comparisons were internally consistent (the
  baseline row used the same composition).
- The Σu² fitness term (ABI 9) immediately gives the GA gradient to shed the
  offset — even on nominal geometry the memory stage now has real work.
- The `vs alloc-LQR` baseline row now forces `residual_scale=0` (the pure
  classical allocator — the paper's actual comparison target) and reports
  `mean_effort` alongside attitude metrics.

## Phase-3 effort-excess screenings (12/07/2026, single-seed pop-20 CPU scale)

Fitness err .35 / stable .25 / jerk .15 / mono .05 / **effort .20** on the
excess metric (ABI 10), octo-X ±5°/8mm/10%. Winner vs baseline excess
(held-out seed 99990101):

| base seed | winner | baseline | verdict |
|---|---|---|---|
| 31337002 | 0.014 | 0.024 | WIN −42% (attitude 1.35 ≤ 1.38°) |
| 31337003 | 0.018 | 0.034 | WIN −47% (attitude 1.33 ≤ 1.38°) |
| 31337004 | 0.145 | 0.014 | LOSS 10× (attitude 1.51 > 1.36° — cell overfit to search episodes) |
| 31337005 | 0.032 | 0.015 | LOSS 2× (attitude ≈ tie) |

Verdict: the metric measures the right thing and the GA demonstrably CAN
learn allocation corrections (2/4 seeds beat the nominal allocator's own
model-mismatch floor), but screening budget is high-variance — the residual
can overfit its evolved cells and blow up held-out excess. Phase-3 PRODUCTION
runs (GPU, larger pop/gens/episodes/folds, post-drain) are where the claim
gets made or broken. History of the metric: raw Σu² v1 was gamed by
collective shedding (see ABI-10 commit) — always report winner AND baseline
effort/excess.
