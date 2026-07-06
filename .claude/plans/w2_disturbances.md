# W2 — Disturbances (weather) for the attitude sim

Status: APPROVED 06/07/2026 (Luiz: torque-only; W2.3 gated on W2.1/W2.2; L-levels PID-relative)
Feeds: docs/controller_research_roadmap.md W2 → W3 (PID+WNN residual hybrid) → paper.

## Why (one paragraph)

The clean sim cannot separate PID from PD from WNN: no constant disturbance →
nothing to integrate (ki=0 ablation, 01/07), PID = 100% at every horizon. Every
comparison so far measures transient recovery on an ideal plant. Disturbances
give the integrator something to do, give the GA a reason to discover
integrator-like state, and make the WNN-vs-PID claim honest. This is the
paper's load-bearing section.

## Ground truth from the stack map (06/07)

- `AttitudeSim` physics lives ONCE in Rust (`controller.rs:229-269 step`,
  `:480-496 body_torque`, `:276-284 read_imu`) — Python `sim.py` re-exports it;
  the CPU rollouts (`dagger_train.rs eval_closed_loop_rs` /
  `eval_ensemble_closed_loop`) use the same struct. ONE edit covers Python+CPU.
- The Metal kernel (`controller_rollout.metal:496-511`) is an independent f32
  port → SECOND edit site. Parity guards: `test_controller_gpu_parity.py`,
  `test_pid_parity.py`.
- The sim is ROTATION-ONLY: no linear velocity/position state. "Wind force"
  has no home — wind must enter as TORQUE. (Translation = W4, explicitly out
  of scope here.)
- No physics params flow through `EpisodeConfig` today; a `disturbance` config
  is the first, threading: EpisodeConfig → `AttitudeSim::new` (PyO3) →
  `score_controllers_metal`/`RolloutParams`/Metal `Params` →
  `eval_ensemble_closed_loop`/`sim_default` → `RewardGatedConfigPacked`
  (in-search training rollouts). ABI bump `_accel.py EXPECTED_ABI 2→3`.

## The four primitives (all body-frame, attitude-domain)

| # | Primitive | Model | Physical story |
|---|-----------|-------|----------------|
| D1 | Constant torque bias | τ_bias[3] added at `step` | CG offset / asymmetric payload / steady wind on frame — THE integrator test |
| D2 | Gusts | Ornstein-Uhlenbeck torque per axis: dτ = −τ/τ_c·dt + σ·√dt·ξ, τ_c≈100ms | correlated turbulence, not white noise |
| D3 | Airframe asymmetry | per-motor k_thrust multiplier (1+δ_i), δ drawn per episode | motor wear / prop damage — enters `body_torque` |
| D4 | Sensor noise | gyro: white σ_g + slow bias walk; accel: white σ_a — in `read_imu` | real IMU |

Determinism/parity: every stochastic draw uses a counter-based xorshift hash
(episode_seed, step, axis, channel) — same integer path in Rust and Metal (the
`should_skip_sample` precedent) → bit-parity preserved, no RNG state to sync.
D3's δ_i drawn once per episode from the episode seed.

## Intensity ladder

`DisturbanceConfig { level: OFF|L1|L2|L3, plus explicit per-primitive fields }`.
Levels are presets over the explicit fields (explicit fields win — no hidden
constants). Calibration experiment W2.0 sets the numbers such that:
- L1: PID keeps 100% stable, visible err/steady increase
- L2: PID keeps ≥95% stable only WITH integrator (PD degrades) — target zone
- L3: PID degrades too (stress case, tail of the plots)
Initial guesses (to calibrate): τ_bias as {2,5,10}% of max control torque
(L·k_thrust), OU σ matching, δ_i ±{3,6,10}%, σ_g {0.005,0.02,0.05} rad/s.

## Experiments (after E4 chain drains — one controller job at a time)

- **W2.0 calibrate**: PID + PD sweep over the ladder, pick L1/L2/L3 constants.
  Cheap (PID evals, minutes).
- **W2.1 re-anchor matrix**: {PID, PD} × {OFF,L1,L2,L3} × steps {500,2000,5000}
  — the first honest PID-vs-PD separation; these anchors replace 100%/2.28°
  in every future table.
- **W2.2 brittleness audit** (zero training): PWM2K pool + best committee
  re-scored under L1/L2 — how much of clean-trained performance survives
  weather? (Committee drift-cancellation may also cancel gust response —
  interesting either way.)
- **W2.3 train-under-weather**: LONG/PWM2K recipe @2000 with L1 (and L2 if L1
  is free) during training rollouts, 2 seeds — does the GA discover
  integrator-like state? Success = weather-trained beats clean-trained under
  weather WITHOUT losing the clean score. This is E5's go/no-go gate.
- **W2.4 physical anchoring (LATER — the W5a bridge; Luiz 06/07)**: once W5a
  swaps in Crazyflie-scale plant params (27 g, ~46 mm arms, inertia ~1.4e-5 —
  the current sim's arm=0.075/k_thrust=2.4/inertia≈2.3e-3 is a bigger quad),
  re-run W2.0 on the Crazyflie plant and re-express L1-L3 in physical units:
  τ_bias/gusts mapped to equivalent wind speed via a drag-moment estimate
  (m/s, citable), D3 δ_i vs typical prop-wear numbers, D4 σ_g vs the BMI088
  gyro datasheet noise density. Paper then reports BOTH definitions: the
  PID-relative ladder (methodology — the separation is guaranteed by
  construction) and the physical magnitudes (defensibility — "L2 ≈ x m/s
  gusting"). The `DisturbanceConfig` explicit-fields design already supports
  this — presets are just numbers; no code change expected beyond a second
  preset table (`level` presets keyed by plant).

## Implementation order (single PR-sized commit each)

1. `DisturbanceConfig` (Rust struct + PyO3 + EpisodeConfig field) + D1-D4 in
   `controller.rs` + hash-RNG helper. OFF = bit-identical legacy (guard test).
2. Metal port + `RolloutParams` extension + parity tests extended to L1/L2
   (CPU vs GPU under identical disturbance draws).
3. Plumbing: evaluator paths + `eval_ensemble_closed_loop` + reward-gated
   training config + ABI bump + `flow_adapter` passthrough.
4. W2.0 calibration script + anchors doc update.

Estimate: steps 1-3 ≈ 4-8h focused work (2 edit sites + plumbing + tests);
W2.0 ≈ 1h. Compute for W2.1 ≈ minutes, W2.2 ≈ 1-2h, W2.3 ≈ 2 GA cells ≈ 8-16h.

## Open questions for Luiz

1. Torque-only wind OK (rotation-only sim; translation stays W4)? Or pull W4
   forward instead?
2. W2.3 (train-under-weather) in scope now, or gate on W2.1/W2.2 readout?
3. Any physical realism constraints you want honored for the L-levels (e.g.
   Crazyflie-scale gust magnitudes from W5a), or calibrate purely against PID?
