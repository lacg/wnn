# Plan v2: break 90% — re-anchored after the ki=0 ablation

**Date:** 01/07/2026 (supersedes the "missing integrator" framing of controller_break_90pct.md)
**Owner:** Andrew Martin / Luiz

## The re-anchor (scripts/pid_ki_ablation.py, 01/07)
On the exact A/B/C held-out protocol (tilt 5.0, body 0.5, yaw-rate 0.3, steps 500,
4 seeds × 100 eps):

| arm | stable | err | steady |
|-----|--------|-----|--------|
| PID default (ki .05/.05/.02) | **100.0%±0.0** | 2.28±0.09° | 0.89±0.03° |
| PID PD-only (ki=0 all axes)  | **100.0%±0.0** | 2.30±0.10° | 0.90±0.04° |

**The I-term is worth 0.02°.** The sim has no constant disturbances (no wind/CG
offset/motor asymmetry/noise), so there is nothing to integrate away. This explains
why FOUR integrator fixes failed (delta accumulator 28%, obs integral features ≤ s16,
pidmix ≤ s16, --state-integral B 83.6 ≈ A 84.3): they solved a non-existent problem.

**New diagnosis:** the WNN gap (84-85% vs PID 100%) =
1. **PD-approximation quality** — the memoryless lookup settles at slightly-wrong fixed
   points (flat ~5.6° soft-fails). A PD law is expressible memoryless: D input = gyro
   (observed), P input = attitude error from accel+target (observable EXCEPT yaw).
2. **Yaw observability** — failures cluster in the 4-6° init-yaw band (73% vs 96%);
   yaw-blind ceiling ≈ 91%. PID reads the true quaternion; the WNN cannot see yaw.
3. **Search reliability** — seed-bimodal 70-90%; the 90% basin exists (cells hit 86-90)
   but the GA reaches it unreliably (premature convergence).

## Sequenced experiments (each cheap, mostly flags; one code change)

### E1 — GA diversity: random immigrants  [code ~15 lines, HIGHEST priority]
`GenericGAStrategy`: each gen replace bottom X% (start 15%) with fresh random genomes.
Validate: `shapes=N` stays >1 past gen 50; seed-SD tightens; pooled mean +3-6pp.
Then it rides along in EVERY later experiment for free.

### E2 — combined A/B sweep after C_grow + low-edge finish (s16, folds=5, 2 seeds)
One driver, 4 arms (seed-outer, same C10 recipe as A/B/C):
- **arm L (long episodes):** `--steps 2000` (rationale WEAKENED by ki=0 — settling
  precision weighting, not integrator pressure — keep 1 arm, it's a flag)
- **arm R (action-repeat):** hold each WNN decision N=5 steps (Sajus frame-skip;
  needs a small rollout-loop change + Metal mirror — temporal abstraction, jerk↓)
- **arm C (hard-IC curriculum):** `--difficulty-adaptive` (BUILT, never run) —
  oversample the failing 4-6° init-yaw shell; boosting on the known failure band
- **arm A (anchor retry):** `--obs-yaw-err` + immigrants + more gens (the anchor's
  brittleness is a search failure, not a signal failure — 0.06° drift)
Success per arm: pooled ho-mem > A_ctrl 84.3±4.4 beyond SD; >90% = jackpot.

### E3 — encoding precision near hover  [NEW lever from the ki=0 re-anchor]
The soft-fail fixed points are ~5.6° — precision of the thermometer mapping near zero
error decides where the lookup's equilibrium lands. Probe: densify threshold quantiles
near zero (fit_thresholds_from_pid_rollouts currently uses rollout quantiles) and/or
raise bits-per-feature for the accel/gyro channels only. Cheap A/B on s16.

### E4 — deployment-honest reporting (no new training)
Best-of-K seed selection on held-out (best singles already 86-90) + optional 2-3-WNN
QSR-sum ensemble (FPGA-cheap). Publishable as "selection + ensembling of evolved
controllers". Do this with EXISTING winners first — it may already clear 90.

### E5 — PID+WNN residual hybrid  [reframe, after E1-E3 read]
PID stays in the actuation path, WNN learns clamped Δu (±10% PWM); floor = PID 100%.
Post-ki=0 nuance: the PID's value is the PD floor, not the I-term. Becomes compelling
when the sim gains disturbances (paper #2 bridge) — where BOTH a plain PD and a plain
WNN degrade and the learned residual can shine.

## Kill/keep criteria
- E1 fails to tighten seed-SD → escalate to crowding (restricted-tournament, ~40 lines).
- E2 arms that don't beat A_ctrl+SD → drop; combine winners into one recipe.
- E4 already >90% → the milestone is met deployment-style; E2 winners then push the
  single-seed mean.

## Status
- [x] ki=0 ablation (scripts/pid_ki_ablation.py) — DONE 01/07, I-term worth 0.02°.
- [ ] E1 immigrants (code) — next code task.
- [ ] E2 driver (after C_grow + low-edge finish; box is busy until then).
- [ ] E3 threshold-density probe.
- [ ] E4 best-of-K report over existing winners (can run ANYTIME — read-only).
- [ ] E5 residual hybrid design (after E1-E3 read).
