# E5 Residual Hybrid — plan (drafted 08/07/2026)

**Trigger:** E5.2 L1→L2 curriculum verdict (pooled ho 56.2% @L2: s09 46.8 / s10 65.5) — beats
from-scratch L2 (19.5) ~3× and MLP (26.7), but ≈ L1-transfer 57.2 and short of PD 84. Fine-tuning
under L2 does NOT reliably surpass inheriting L1 → the memoryless-PD ceiling (84) is the wall.
Residual hybrid is the registered architectural answer (findings Finding 8 + w2_disturbances.md).

## Core idea
`action = PD(err) + WNN_residual(obs)` — split the labor:
- **Analytic PD** (memoryless Kp/Kd, Ki=0) does the fast stabilization → 84 @L2 for free, robust.
- **Learned WNN residual** supplies ONLY the correction PD lacks — primarily the INTEGRAL action
  that rejects the constant L2 `tau_bias`.

### Why it should clear the wall pure-WNN hit (56/84)
1. Residual is a smaller, smoother target (PD carries the fast stabilization; WNN supplies a
   slowly-varying bias-rejection term) — far lower-dimensional than the full policy.
2. The WNN already has the machinery: integral obs (`obs_tilt_i`/`obs_peraxis_i`/`obs_yaw_err_i`)
   + `delta_control` learned accumulator. Pure-WNN failed on the WHOLE controller; the residual
   plays straight to these integral features.
3. There is now a FLOOR: analytic PD never collapses @L2 (=84), so even a weak/early residual can't
   score below 84. Pure-WNN had no floor (0% on failure). De-risks training; "beat 84" = clean bar.

### Key code-map facts (from Explore agent 08/07)
- DAGGER expert is ALREADY the PID (`dagger.py:210-216`: student=WNN.step, expert=pid.step,
  edra_train toward expert). So residual training just changes the LABEL, not the machinery.
- WNN controller: `controller.rs:2259-2381` (step), obs = 9 base (gyro/accel/target) + H2 extras;
  output = 4 motor PWM [0,1]; has delta_control (signed accumulator) + decouple (H3) modes.
- `AttitudePID` (`pid.py:98-217`): stock Kp=1.2/Ki=0.05/Kd=0.30 (r,p), yaw 0.6/0.02/0.20, i_clamp=0.5.
  Stock=97% @L2, PID+ (ki×4,i_clamp×4)=99.8%, PD (Ki=0)=84%. Input q+gyro+target_rpy → 4 PWM.
- Episode loop hook: `training.py:378-381` (`pwm = action_fn(...)` then `sim.step(pwm)`).
- ControllerSpec ALREADY has `delta_max`, `delta_leak` (evaluator.py:90-170) — clamp param exists.
- Disturbances: counter-based RNG, `--disturbance OFF|L1|L2|L3`, clean-path Option guard.

## Design decisions (Luiz, 08/07)
1. **Baseline = BOTH as ablation**: PD-base arm (84 ceiling, learns full integral 84→99.8) AND
   stock-PID-base arm (97, learns small residual 97→99.8). Shows residual value scales with how
   much integral the baseline lacks.
2. **Authority = LEARN THE CLAMP**: `delta_max` per-axis becomes a searched genome param (phased_ga
   picks per-axis residual authority), not a fixed band.

## Training = residual-DAGGER
- Expert = PID+ (99.8). Baseline = {PD | stock-PID}. Per-step label = `expert.step() − baseline.step()`
  (the integral contribution). On-policy: plant steps with `baseline + clamp(student_residual)` so
  DAGGER sees the composed action's consequences.
- WNN output in signed/delta mode (neutral at untrained), clamped to the (learned) per-axis band.

## Build sequence
**Phase 0 — proof (Python, FIXED clamp ±0.2, no rebuild):** validate `PD + clamped_residual` beats
PD-alone @L2 with a hand-set delta_max, both baselines. Hook at training.py:378-381; reuse AttitudePID
for baseline (Ki=0) + expert (PID+). If a fixed-clamp residual can't beat 84, the idea is wrong —
cheap to find out (an afternoon). Gate to Phase 1.

**Phase 1 — real experiment:** delta_max per-axis → searched genome param; 2 baseline arms × 2 seeds;
residual-DAGGER (expert=PID+); phased_ga NEURONS+MEMORY under --disturbance L2. Rulers: beat 84
(PD-base) / 97 (stock-PID-base), aspire 99.8 (PID+). Report 4-seed held-out per arm, pooled.

**Phase 2 — generalization:** does the L2-trained residual transfer to L3 (where pure controllers
collapse: PID+ 27 / PD 2.2)? The real robustness test.

## Code changes (well-scoped)
1. Residual composition in episode loop (`training.py:378-381`): residual mode where
   `applied = baseline.step(...) + clamp(wnn.step(...), delta_max_per_axis)`; baseline passed in.
2. Residual-DAGGER label (`dagger.py:210-216`): label = expert − baseline; applied = baseline + student.
3. `delta_max` per-axis searchable (ControllerSpec + a phased_ga mutation stage).
4. CLI/flow wiring (`phased_ga.py`): `--residual-baseline PD|STOCK_PID`, `--residual-expert PID_PLUS`,
   reuse `--disturbance L2`. flow_adapter passthrough if run via dashboard.
5. Evaluator/rulers (`evaluator.py`): score baseline+residual vs baseline-alone vs expert.

## ⚠️ Architecture-integrity note
This adds a COMPOSITION wrapper around the existing controller, NOT a new RAM-neuron type — reuses
Memory/RAMLayer/ControllerSpec unchanged. Per CLAUDE.md, discuss any deviation before implementing.
One-controller-at-a-time rule still applies to the eventual training runs.
