# Peng, Andrychowicz, Zaremba, Abbeel 2018 — Sim-to-Real Transfer of Robotic Control with Dynamics Randomization

Peng, X. B., Andrychowicz, M., Zaremba, W., Abbeel, P. ICRA 2018. arXiv:1710.06537 (v3, 3 Mar 2018).
https://arxiv.org/abs/1710.06537 — Local PDF: `papers/peng2018_sim_to_real_dynamics_randomization.pdf`. Read 28/11/2026.

**Relevance.** The canonical dynamics-randomization paper; the robustness protocol reference. **Regime:** 7-DOF Fetch arm,
puck pushing, MuJoCo, LSTM policy (RDPG+HER). NOT a quadrotor. Real robot = the OOD test.

## Verbatim extraction
- Sampling: "At the start of each episode, a random set of dynamics parameters µ are sampled ... held fixed for the
  duration of the episode." "Parameters such as mass, damping, friction, and controller gains are logarithmically
  sampled, while other parameters are uniformly sampled." 95 randomized parameters in total.
- Latency: "Δt ∼ Δt0 + Exp(λ), where Δt0 = 0.04s is the default control timestep"; "While Δt varies every timestep,
  λ is fixed within each episode." Obs noise: Gaussian, std "5% of the running standard deviation of each feature".
  Action exploration noise std "0.01rad".
- Table I (ranges): Link Mass [0.25, 4]× default; Joint Damping [0.2, 20]× default; Puck Mass [0.1, 0.4] kg;
  Puck Friction [0.1, 5]; Puck Damping [0.01, 0.2] N s/m; Table Height [0.73, 0.77] m; Controller Gains [0.5, 2]×
  default; Action Timestep λ [125, 1000] s^-1.
- Sim eval: "performance of each policy is evaluated over 100 episodes, with randomized dynamics parameters for each
  episode" (i.e. the TRAINING distribution, jointly drawn). "Four policies initialized with different random seeds".
- Table II (success sim / real / real trials): LSTM 0.91±0.03 / 0.89±0.06 / 28; FF no Rand 0.51±0.05 / 0.0±0.0 / 10;
  FF 0.83±0.04 / 0.67±0.14 / 12; FF+Hist 0.87±0.03 / 0.70±0.10 / 20.
- Table III (real, LSTM trained with subsets FIXED — a train-time ablation, main text): all 0.89±0.06 (28);
  fixed action timestep 0.29±0.11 (17); no observation noise 0.25±0.12 (12); fixed link mass 0.64±0.10 (22);
  fixed puck friction 0.48±0.10 (27).
- Robustness (real): chips packet under puck → 0.91±0.04 vs 0.89±0.06 without.

## Absent
- No sim test beyond the training ranges; no one-parameter-at-a-time TEST sweep (the ablation removes parameters from
  TRAINING); OOD = the real robot only. No quadrotor.
