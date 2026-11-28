# Molchanov et al. 2019 — Sim-to-(Multi)-Real (action-space / memory notes)

Molchanov, A., Chen, T., Hönig, W., Preiss, J. A., Ayanian, N., Sukhatme, G. S. *Sim-to-(Multi)-Real:
Transfer of Low-Level Robust Control Policies to Multiple Quadrotors.* IEEE/RSJ IROS 2019.
arXiv:1903.04628. Local PDF: `papers/molchanov2019_sim_to_multi_real.pdf`. Plant-parameter extraction
lives in `docs/disturbance_param_sources.md` (S1); this note covers action space / motor-lag handling.
Read 06/11/2026.

**Regime:** PPO, Crazyflie 2.0 + two larger quads, Crazyswarm state estimation, feed-forward MLP
(2×64 tanh), outputs normalized motor thrusts directly ("we do not use any additional stabilizing PID
controllers and directly control the motor thrusts").

## Verbatim extraction
- Input: "an 18-dimensional vector" of errors to goal + rotation matrix. **No action history, no recurrence.**
- Motor lag simulated (settling time T; T = 0.15 → τ = T/4 = 37.5 ms, derivation already in S1).
- Table IV (sim-to-multi-real; ēθ [deg], fo [Hz], et [m]) on CF:
  Mellinger 1.93/1.4/0.11; Mellinger (no memory) 3.68/6.2/0.20; NN CF 1.53/0.9/0.19;
  NN CF (w/o delay) 1.90/1.0/0.21; NN CF 20% random 1.53/1.1/0.20; NN CF 30% random 2.65/1.4/0.23.
- "Our neural network with the motor settling time T = 0.15 has a mean position error of 0.19 m, which is
  similar to the hand-tuned baseline controller without memory. A network trained without motor delays
  (T nearly zero) overshoots frequently and has a larger mean position error of 0.21 m."
- "Surprisingly, modeling the motor delay during training has a very small impact on tracking performance (row 5)."
- DR: "it works best if the randomization is fairly small (20 % in our case)".
- Recovery: CF recovers "80% of all throws and up to 100% throws with moderate attitude changes (≤ 35◦)".

## Absent
- No action history / recurrence in the observation; no CTBR comparison arm.
- Note: Table I DOES list motor settling time T = 0.15 s with randomization "∼ U (0.1, 0.2)" (verified
  06/11), and on-board "evaluation, EKF, control) is done at 500 Hz. Evaluating the neural network takes
  about 0.8 ms." — but the motor-lag-specific A/B in Table IV is only "with" vs "w/o delay".

## Robustness protocol addendum (read 28/11/2026)
- "During training, we sample dynamics parameters for each individual trajectory" (joint, per trajectory); two modes:
  around nominal (CF2.0 estimates [8]) or "within a set of limits" (Table I: m ≤5 kg; l_body,w U(0.05,0.2) m;
  T U(0.1,0.2) s; rt2w U(1.8,2.5); rt2t U(0.005,0.02)).
- "In each scenario, we train five policies by varying the seed"; real test = best seed by training cost + visual pick.
- Test = REAL vehicles (CF, Small, Medium; Table IV), hover 10 s at 100 Hz + figure-eight; no sim OOD sweep, no
  widened-range sim test. Rows 6–8 = training DR level 10/20/30% (train-time sweep, main text).
