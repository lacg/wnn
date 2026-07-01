# Plan: break the ~90% controller stability ceiling

**Date:** 30/06/2026 (bit-sweep in flight; frame-fix concluded)
**Owner:** Andrew Martin / Luiz

## Problem statement
Best controller cells hit ~87–90% held-out stable but not *reliably* (seed-bimodal
70–90%), and the mean sits ~80–83%. PID reaches ~98%. The ceiling is **NOT a capacity
limit**:
- `wish_bits=0`, `saturation=0` even at the lowest bit-width (bit-sweep) → no bit starvation.
- No split-pressure for more neurons; R2's 100b/200+ output neurons didn't beat R1's tiny config.
- `steady` is already the highest-weighted fitness term (0.35) → not a reward-spec problem.

Frame-fix also proved that **feeding hand-computed PID-error/integral features
(pidmix `obs_*_i`) does NOT help** — the GA leaves them ≈uniformly wired (bpf analysis),
mean ≤ obs-OFF s16. You can't hand the network an integral as a static input.

## Diagnosis (two independent walls)
1. **Search reliability** — the 90% basin is reachable (pidmix seed10 hit 90.0) but the GA
   finds it only some seeds. Root cause: **premature convergence** (one lineage fixates the
   population by ~gen 50 — [[project_ga_diversity_preservation]]).
2. **Missing integrator** — the ~20% failures are SOFT: 0% divergence, settle *flat* at ~5.6°
   = steady-state offset = no integral action ([[project_controller_stability_diagnosis]]).
   The integrator must live in the **recurrent STATE**, not the inputs.

## The three levers (+ the built-in one)
1. **Grow / activate the recurrent STATE integrator.** ⭐ KEY, and mostly ALREADY BUILT:
   `--state-integral` (phased_ga) → `WNN_STATE_INTEGRAL_TARGET=1` trains the recurrent state
   against a **direct PID-integral target** in the Rust trainer (help: "use *small*
   grid-state-neurons"). This is the un-pulled "v2 state universe" lever. Complementary knob:
   `--grid-state-neurons` (raise the state-layer size).
2. **Fix GA premature convergence** — niching / crowding / speciation ([[project_ga_diversity_preservation]]).
   NOT yet implemented → separate code task. Attacks the bimodality → raises the *pooled* mean.
3. **Curriculum / longer episodes** — lengthen `--steps` (and/or an episode curriculum) so an
   integrator is *necessary* to score (steady-state error only bites over long holds), forcing
   the GA to grow one. Reweighting won't do it (steady already top weight).

## NEXT RUN — state-integrator A/B/C on s16 (folds=5, 2 seeds)
Substrate = **s16** (obs-OFF, 9f) — the best + tightest frame-fix variant, cleanest base to
test "does integrator capacity break 90%" without the pidmix feature-wiring confound.

| arm | flags | tests |
|-----|-------|-------|
| **A ctrl** | s16, small state (`--grid-state-neurons 8 12 16`), no integral | baseline anchor (≈ frame-fix s16) |
| **B integral** | A + `--state-integral` | ⭐ the targeted integrator fix (small state, PID-integral target) |
| **C grow** | s16, `--grid-state-neurons 24 32 40`, no integral | capacity control — does growth ALONE help? (expect ~no, per no-pressure) |

Shared: `--grid-bits 24` (bits are wasted — keep it cheap), `--num-eval-folds 5`,
`--neurons-gens 15 --memory-gens 15`, `--pop 24`, `--eval-episodes 100 --steps 500`,
`--report-seeds 99990001 99990101 12345 67890`, `--tilt 5.0`, fit-weights = C10
(err²0.25 steady0.35 stable0.20 jerk0.15 mono0.05), seeds {20260609, 20260610}, seed-outer.
3 arms × 2 seeds = **6 cells**, ~1–2 h each at 24b/folds5 → ~8–12 h.
Report/pool exactly like the bit-sweep (mean±SD over 8 held-outs per arm).

**Success criterion:** arm B pooled ho-mem stable **> 90%** (or clearly > A/C beyond the
±SD), AND its soft-fail steady° drops toward PID's ~2.3°. If B beats A but C doesn't →
confirms "integrator, not capacity." If B ≈ A → the integral target isn't being learned →
escalate to lever 3 (longer episodes) + lever 2 (diversity).

**Watch:** large-arch winner-save OOM ([[project_controller_save_oom]]) — but 24b keeps archs
small, so lower risk than the 100b frame-fix cells.

## Follow-ups (after the A/B/C read)
- If integral helps but variance still high → implement lever 2 (niching) and re-run best arm.
- Lever 3 (episode curriculum / longer steps) as a second sweep if B alone doesn't clear 90%.
- If B clears 90% cleanly → promote `--state-integral` to the production phased_ga recipe and
  re-baseline the paper's controller numbers.

## Status
- [x] Build driver `scripts/state_integral_ab_driver.sh` — DONE 01/07, launched PID 13564
      (PPID=1), parked on the bit-sweep done-marker. Own marker `/tmp/wnn_state_integral_done.json`.
- [x] Status script `scripts/state_integral_status.py` + cron ba38636e (:22/:52) — DONE 01/07.
- [ ] Run + report pooled A/B/C (auto-starts when bit-sweep done-marker fires).

## Bug check (Luiz recalled state-integral collapsed badly before)
Verified NOT a blocker: (1) the Rust integral-target uses `npa = self.state_neurons/3` — reads
the LIVE state size, no hardcoded assumption; (2) the past collapse was the frame-misalignment
bug (state-prefix at wrong offset) which landed AFTER state-integral (git: 6451a35f/d8d9e8ed →
tested/collapsed → ae3e0214 fix 27/06) and is now fixed + validated by the whole frame-fix sweep;
(3) arm B runs on s16 (exactly 9 features) where the `9≠actual` mismatch is a no-op. The A/B/C is
self-diagnosing: if a latent bug remains, arm B's first held-out collapses to ~14-20% visibly at
24b (~1-2h), and control arms A/C localize it to B. Low-risk falsifiable probe.

## Lever 2 (niching) — code scoping
GA loop is `GenericGAStrategy` (tournament + elitism), subclassed by `ControllerGAStrategy`
(ga_strategy.py) and the variable-shape `arch_strategy.py`. Diversity signal already logged =
`shapes=N` per gen (collapses to 1-2 by ~gen 50 = the fixation). Options, cheapest→heaviest:
1. **Random immigrants** (~15 lines): each gen, replace bottom X% with fresh random genomes.
   Cheapest anti-convergence; no distance metric needed.
2. **Crowding / restricted-tournament replacement** (~30-40 lines): offspring replaces its
   MOST-SIMILAR population member (if better), not the global worst. Needs a genome-distance
   helper (shape descriptor sn/sb/ob/on + optional connectivity overlap).
3. **Fitness sharing / niching** (~40-60 lines): divide each genome's fitness by its niche
   density (# genomes within ε in shape-space) before selection.
4. **NEAT speciation** (~150+ lines): partition into species by distance, protect young species.
Recommend 1 first (validate it lifts `shapes=N` past gen 50 + tightens the seed-bimodality),
escalate to 2 if needed.
