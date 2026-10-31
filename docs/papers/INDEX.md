# Paper library index

PDFs live in `papers/` (gitignored, local); extractions live here (committed).
Convention: `<firstauthor><year>_<slug>` for both. Maintained by the paper-scout
agent (.claude/agents/paper-scout.md) and by hand.

- [Molchanov et al. 2019 — Sim-to-(Multi)-Real](https://arxiv.org/abs/1903.04628) —
  PDF: `papers/molchanov2019_sim_to_multi_real.pdf`. Plant-DR Table I; motor lag
  T=0.15s; OU motor noise; 20%-helps/30%-hurts DR finding; NO obs delay/dropout/
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
