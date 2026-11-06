# Paper library index

PDFs live in `papers/` (gitignored, local); extractions live here (committed).
Convention: `<firstauthor><year>_<slug>` for both. Maintained by the paper-scout
agent (.claude/agents/paper-scout.md) and by hand.

- [Molchanov et al. 2019 — Sim-to-(Multi)-Real](https://arxiv.org/abs/1903.04628) —
  PDF: `papers/molchanov2019_sim_to_multi_real.pdf`. Plant-DR Table I; motor lag
  T=0.15s = 2% SETTLING time (eq 7-8; τ=T/4=37.5 ms), DR U(0.1,0.2), nominals from Förster [8]; OU motor noise; 20%-helps/30%-hurts DR finding; NO obs delay/dropout/
  torque bias. Extraction: docs/disturbance_param_sources.md (S1).
- [Panerati et al. 2021 — gym-pybullet-drones](https://arxiv.org/abs/2103.02142) —
  PDF: `papers/panerati2021_gym_pybullet_drones.pdf`. System-identified aero
  (drag/ground effect/downwash); ground-truth obs, NO sensor pathology; motors
  near-instantaneous. Extraction: docs/disturbance_param_sources.md (S3).
- [RotorS ADIS16448 IMU defaults](https://github.com/ethz-asl/rotors_simulator)
  (`component_snippets.xacro`) — gyro density 0.0003394 rad/s/√Hz, turn-on bias
  0.0087 rad/s, accel density 0.004 m/s²/√Hz. Extraction:
  docs/disturbance_param_sources.md (S2). No local PDF (source file).
- [Dryden MIL-F-8785C (MathWorks doc)](https://www.mathworks.com/help/aeroblks/drydenwindturbulencemodelcontinuous.html)
  — σ_w=0.1·W20; light/moderate/severe = 15/30/45 kt at 20 ft. Extraction:
  docs/disturbance_param_sources.md (S4). No local PDF (web doc).
- [arXiv:2603.02114 — thermal-inertial odometry](https://arxiv.org/html/2603.02114v1)
  — NUC outage "250 ms to 1 s"; NUC deferral scheduler >5 m/s. Extraction:
  docs/disturbance_param_sources.md (S5). PDF not stored (HTML read).
- [Förster 2015 — Crazyflie 2.0 system identification (ETH BSc)](forster2015_crazyflie2_sysid.md) —
  load-cell + wind-tube rotor drag: Kaero xy −9.1785e-7 kg/rad × ΣΘ̇ → k_d ≈ 0.25 1/s at 28 g hover (derived).
- [Folk 2026 — wind inference PhD (arXiv 2604.00343)](folk2026_wind_inference_thesis.md) —
  free-flight CF2.1-Brushless 57 g: kd 5.09e-6 N·s²/(m·rad) → k_d ≈ 0.65 1/s (derived).
- [Luis & Le Ny 2016 — nanoquad trajectory controller](luis2016_nanoquad_trajectory_controller.md) —
  CF2.0 hover 16073 rpm at 33 g; NO translational drag term.
- [Faessler et al. 2018 — rotor-drag flatness (RA-L)](faessler2018_rotor_drag_flatness.md) —
  610 g / 6-inch quad: dx 0.43–0.54, dy 0.24–0.39 s⁻¹ (model form reference, not micro).
- [Gräfe et al. 2026 — How to Model Your Crazyflie Brushless (ICRA'26)](grafe2026_model_crazyflie_brushless.md) —
  MEASURED CF Brushless (44 g) motor τ = 50 ms (in-flight IR rpm, first-order on Ω) → T_ours = 0.20 s; DR ±20%.
- [Eschmann et al. 2024 — sysid subject to motor delays (arXiv 2404.07837)](eschmann2024_sysid_motor_delays.md) —
  27 g CF2.1 flight-fit τ = 0.072 s (rpm inferred), 3.35 kg quad τ = 0.065 s → T_ours ≈ 0.29 / 0.26 s.
- [Eschmann et al. 2024 — Learning to Fly in Seconds (arXiv 2311.13081)](eschmann2024_learning_to_fly_in_seconds.md) —
  ASSUMED sim τ = 0.15 s ("common" 0.05–0.25 s, uncited) → T_ours = 0.60 s.
- [Bitcraze 2015 — CF2.0 M1 step response plot](bitcraze2015_m1_step_response.md) —
  brushed 0→100→0% PWM: digitised τ_rise ≈ 0.064 s, τ_fall ≈ 0.23–0.26 s (coast); Greiff 2017 J−/J+ ≈ 4.2.

### Action space / cascade (CTBR vs motor commands) — added 06/11/2026
- [Kaufmann 2022 — Benchmark of learned control policies (ICRA'22)](kaufmann2022_benchmark_learned_control_policies.md) — SRT vs CTBR vs LV; SRT crashes under BEM+20 ms delay, undeployable at 60 ms real latency; CTBR wins.
- [Kaufmann 2023 — Swift champion drone racing (Nature)](kaufmann2023_swift_champion_drone_racing.md) — CTBR → Betaflight PID, 100 Hz policy, previous action in obs.
- [Hwangbo 2017 — Quadrotor control with RL (RA-L)](hwangbo2017_quadrotor_control_rl.md) — state→rotor thrust, NO motor model, 7 µs/step; motor dynamics named as top sim gap.
- [Molchanov 2019 — Sim-to-(Multi)-Real (IROS)](molchanov2019_sim_to_multi_real.md) — direct thrusts, memoryless MLP, T=0.15 s U(0.1,0.2); w/o-delay 0.21 m vs 0.19 m.
- [Eschmann 2024 — Learning to Fly in Seconds (RA-L)](eschmann2024_learning_to_fly_in_seconds.md) — addendum: RPM output + action history; no-history 0/10 traj, no-rotor-delay 0/10.
- [Ferede 2023 — E2E RL vs CTBR→INDI (arXiv 2311.16948)](ferede2023_e2e_time_optimal_quadcopter.md) — motor-level beats INDI by 0.17 s real but with measured RPM in obs; bigger sim-real gap.
- [Zhang 2022 — single near-hover controller, RMA (arXiv 2209.09232)](zhang2022_single_nearhover_controller.md) — classical CTBR outer → learned motor-speed inner + 400-step history adaptation.
- [O'Connell 2022 — Neural-Fly (Sci. Robotics)](oconnell2022_neural_fly.md) — learned outer loop over PX4 attitude PID; −66% vs nonlinear, −35% vs INDI in wind.
- [Shi 2019 — Neural Lander (ICRA'19)](shi2019_neural_lander.md) — SN-DNN residual force in position loop; z err 0.13 m → 0.
- [Koch 2019 — Neuroflight (arXiv 1901.06553)](koch2019_neuroflight.md) — 1344-weight NN rate loop on STM32F722 at 2 kHz; argues RPM input hurts transfer (Δe input: −45.07% error in sim).
- [Kresse 2026 — Differentiable Weightless Controllers (ICML'26)](kresse2026_differentiable_weightless_controllers.md) — LUT/thermometer policies, MuJoCo only, 2–3 cycles, ~2 nJ/action on Artix-7.
