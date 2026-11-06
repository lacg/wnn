# Ferede, De Wagter, Izzo, de Croon 2023 — End-to-end Reinforcement Learning for Time-Optimal Quadcopter Flight

Ferede, R., De Wagter, C., Izzo, D., de Croon, G. C. H. E. arXiv:2311.16948 (v1, 28 Nov 2023).
Venue: not stated in the v1 PDF (search snippets suggest ICRA 2024 — NOT verified).
Local PDF: `papers/ferede2023_e2e_time_optimal_quadcopter.pdf`. Read 06/11/2026.

**Relevance.** The one head-to-head REAL-flight comparison found of motor-command (E2E) vs CTBR → INDI
inner loop. **Regime:** Parrot Bebop 1 with Paparazzi, onboard RPM measurement, PPO, drone racing.

## Verbatim extraction
- Abstract: "the sim-to-real gap, often addressed by employing a robust inner loop controller —an
  abstraction that, in theory, constrains the optimality"; "E2E showcases a significant 1.39-second
  advantage in simulation and a 0.17-second edge in real-world testing".
- Motor lag handling in E2E: motor model "ω̇ = (u − ω)/τ" and the policy input includes measured motor
  speeds ω plus onboard-estimated external force/moment: "xin = [p, v, λ, Ω, ω, Mext, Fext, gates]";
  Bebop provides "RPM measurements for each propeller, essential for our control method." RPM range
  3000–11000. Residual thrust/moment nets: single hidden layer of 32.
- INDI Net: "INDI controller's command response can be expressed as a first-order delay model … τΩ and τT ,
  as 0.03 based on flight data." (units not printed; s implied).
- Margin argument: "An inherent limitation of relying on an INDI inner loop controller is that it requires
  some margins to reject disturbances and track the desired command."
- Results: INDI 17.14 s sim → 17.40 s real (avg); E2E 15.75 s sim → 17.35 s real (avg); real average gain
  "0.05 seconds over the entire track"; fastest-run gain 0.17 s; advantage mostly in lap 1 from hover.
- Conclusion: "our E2E approach might exhibit greater sensitivity to modeling errors, as opposed to the
  robustness of the INDI controller."

## Absent
- E2E motor τ value not printed in this paper (model refers to prior Bebop work). No action history.
- No disturbance-rejection test; n = 5 real flights per method.
