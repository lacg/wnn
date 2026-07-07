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

- **W2.0 calibrate**: ✅ DONE 06/07 (two rounds; logs/controller/W2Calibrate_20260706).
  v1 ({2,5,10}% bias) destabilized nothing but exposed that the STOCK PID
  integrator (ki=0.05, i_clamp=0.5) trims only ~26% of a bias offset → added
  PID+ arm (ki×4, i_clamp×4) as the honest with-integrator ceiling. v2 ladder
  (×3: 5/15/30% bias) = ALL LEVELS MET: L2 @2000 = PID+ 99.8 / stock 97.0 /
  PD 84.0 (+15.8pp separation — THE zone); L3 = PID+ 27.0 / PD 2.2.
- **W2.1 re-anchor matrix**: ✅ SATISFIED BY W2.0 v2 (same cells, fresh seeds
  × 100 eps, plus PID+): the v2 table IS the anchor table. Anchors @2000:
  OFF 100/100/100 (PID+/stock/PD), L1 100/100/100 (err separates only),
  L2 99.8/97.0/84.0, L3 27.0/5.8/2.2.
- **W2.2 brittleness audit**: ✅ DONE 06/07 (W22Brittleness_20260706). Clean-trained
  WNNs collapse off-distribution: −9 to −63pp at L1 (PID/PD hold 100), ZERO at L2
  (PD holds 84). Committees soften L1 (84-86.5) but not L2 (common-mode no-integrator
  blind spot). No implicit integral action. Full table in findings doc Finding 8.
- **W2.3 train-under-weather — L1 arm ✅ GATE MET 07/07 02:27Z**
  (W23Weather_20260706): s09 ho 93.5±2.7 under L1 (gate 80.2 +13.3pp); fresh
  matrix @2000: clean 86.2 / L1 90.2 / **L2 57.2 (clean-trained = 0.0)** —
  partial integrator-like transfer, still under PD's 84 @L2. NEURONS flat,
  MEMORY did the lift (distribution problem, not capacity). Finding 8 updated.
  **L2 arm ❌ DECISIVE NEGATIVE (07/07 15:25Z, W23WeatherL2_20260707)**: from-scratch
  L2 training FAILS — ho s09 2.8 / s10 16.5 (pooled ~9.6); fresh @L2 s09 1.0 / s10 19.5,
  all far below PD's 84 and even the L1-trained transfer 57.2. KILLER: L1-trained @ L2
  (57.2) beats L2-trained @ L2 (19.5) by 3× — training one level down is the better L2
  controller. Mechanism: under L2 the population never flies during search (0-9%), so no
  gradient toward integral action. Answer to the headline question = NO (integral action
  is not emergent under harsh-only training); need a curriculum. → **E5 = GO** (residual
  hybrid or L1→L2 curriculum fine-tune; the negative pinpoints the wall). Finding 8 +
  matrix updated.
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
