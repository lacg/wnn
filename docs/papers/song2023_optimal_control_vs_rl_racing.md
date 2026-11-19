# Song, Romero, Müller, Koltun, Scaramuzza 2023 — Reaching the limit in autonomous racing: Optimal control versus reinforcement learning

Science Robotics 8, eadg1462 (2023). DOI 10.1126/scirobotics.adg1462 (DOI per UZH repository search hit, not printed
in the arXiv PDF). arXiv:2310.10943. Local PDF: `papers/song2023_optimal_control_vs_rl_racing.pdf`. Read 19/11/2026.

**Relevance.** Flagship "neural network controller beats MPC" paper — and the net sits ABOVE a classical rate loop
and BELOW a mocap+EKF estimator. **Regime:** real racing quad, Vicon (36 cameras, 400 Hz) + EKF, PPO in Flightmare.

## Verbatim extraction
- Abstract: "a neural network controller trained with reinforcement learning (RL) outperforms optimal control (OC) methods".
- Action: 4-dim, mass-normalized collective thrust + body rates, "tracked by a low-level controller that operates at a
  higher frequency" (Methods, Sim-to-real Transfer). Low-level controller not named in the paper.
- State: "Vicon motion capture system together with an Extended Kalman Filter"; feedback at "100 Hz".
- Policy: "two-layer multilayer perceptron (MLP) as the policy network, with 256" units per layer.
- Results (Fig. 2C; lap time s, success %): nominal sim — Trajectory Tracking 4.92±0.10 / 44%, Contouring Control
  5.03±0.18 / 76%, RL 5.14±0.09 / 100%; realistic sim — TT 0%, CC 5.34±0.27 / 20%, RL 5.26±0.32 / 100%;
  real — TT 0%, CC 5.54±0.21 / 50%, RL 5.35±0.15 / 85% (text: RL "flies 15 laps and consistently achieves a 100% success rate").
- Peak "108 km/h and a maximum acceleration of 12.58 g"; RL best single lap 5.12 s.
- Training: PPO, DR of thrust mapping and drag; "in only ten minutes on a standard workstation"; zero-shot.

## Absent
- Policy inference time / onboard-vs-offboard compute; inner-loop controller identity, gains, rate; the phrase
  "end-to-end"; parameter count; any energy number.
